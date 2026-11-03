"""Micro de smartphone simulé (2020–2026), en temps réel.

Chaîne, dans l'ordre où le son voyage réellement :
  voix → chambre (réflexions + réverbération) → bruits de fond (pièce, ventilateur, rue,
  maison, manipulation) → capsule MEMS (réponse, souffle, saturation) → traitement « appel »
  du téléphone (réduction de bruit, AGC, égalisation) → codec (bande passante, artefacts
  de compression, pertes de paquets) → limiteur → sortie.

Blocs de 10 ms à 48 kHz ; le traitement du téléphone se fait dans le domaine fréquentiel
(trames de 20 ms, recouvrement 50 %), comme les vrais codecs. Latence propre : ~10 ms.
"""
import math
import queue
import threading
import time

import numpy as np
from scipy import signal

SR = 48000
BLOCK = 480                 # 10 ms
NFFT = 2 * BLOCK            # trames de 20 ms
FREQS = np.fft.rfftfreq(NFFT, 1 / SR)

DEFAULTS = {
    "enabled": True,        # False : son d'entrée tel quel (comparaison A/B)
    "input_gain": 0,        # dB
    "output_gain": 0,       # dB
    # chambre
    "room": 70,             # réverbération, % du niveau physique pour cette distance
    "room_size": 12,        # surface au sol, m² (hauteur 2,5 m)
    "damping": 55,          # amortissement : lit, rideaux, tapis… %
    "distance": 30,         # bouche → téléphone, cm
    # bruits de fond (0 = coupé)
    "room_tone": 45,        # ambiance de la pièce
    "fan": 25,              # ventilateur / PC
    "street": 15,           # rue lointaine, derrière la fenêtre
    "house": 15,            # bruits rares de la maison (porte, pas, objet posé)
    "handling": 10,         # manipulation du téléphone
    # capsule
    "mic_noise": 30,        # souffle de la capsule
    "saturation": 20,       # saturation quand on parle fort
    # traitement du téléphone
    "noise_suppression": 45,
    "agc": 60,
    "tone": 0,              # −100 chaud … +100 brillant
    "air": 0,               # recrée des aigus (entrée Bluetooth limitée à 8 kHz), %
    # codec
    "codec": 1,             # index dans CODECS
    "bitrate": 65,          # qualité du codec, %
    "packet_loss": 0,       # trames perdues, ‰
    # synchro
    "delay": 0,             # retard ajouté, ms
    "sync_video": False,    # + latence mesurée de la vidéo
}

CODECS = [  # nom, coupure basse Hz, coupure haute Hz, bande adaptée au débit, « voix d'appel »
    ("Appel classique / 3G (AMR bande étroite)", 300, 3400, False, True),
    ("Appel 4G HD Voice (AMR-WB)", 100, 7000, False, True),
    ("Appel VoLTE / VoWiFi (EVS)", 80, 14000, False, True),
    ("WhatsApp / Messenger (Opus 16 kHz)", 80, 7600, False, True),
    ("Discord / Teams / Snap (Opus 48 kHz)", 60, 16000, True, True),
    ("Mémo vocal / vidéo (AAC 48 kHz)", 40, 19500, True, False),
]


def db(x: float) -> float:
    return 10 ** (x / 20)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) + 1e-12


class Filt:
    """Filtre IIR avec état, appliqué bloc par bloc."""

    def __init__(self, sos):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2))

    def __call__(self, x):
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


def butter(order, fc, kind):
    return signal.butter(order, fc, kind, fs=SR, output="sos")


def peaking(f0, gain_db, q):
    """Biquad en cloche (RBJ)."""
    a = db(gain_db / 2)
    w = 2 * math.pi * f0 / SR
    al = math.sin(w) / (2 * q)
    b = [1 + al * a, -2 * math.cos(w), 1 - al * a]
    d = [1 + al / a, -2 * math.cos(w), 1 - al / a]
    return signal.tf2sos(b, d)


class Drift:
    """Valeur aléatoire lente (Ornstein-Uhlenbeck), rendue en rampe sur le bloc (pas de zip)."""

    def __init__(self, rng, tau_s, sigma, mean=0.0):
        self.rng, self.tau, self.sigma, self.mean = rng, tau_s, sigma, mean
        self.v = mean

    def ramp(self) -> np.ndarray:
        dt = BLOCK / SR
        old = self.v
        self.v += (self.mean - self.v) * dt / self.tau + \
            self.sigma * math.sqrt(2 * dt / self.tau) * self.rng.standard_normal()
        return np.linspace(old, self.v, BLOCK, endpoint=False)


class Events:
    """Sons ponctuels en cours de lecture. Synthétisés à côté (thread) : jamais d'à-coup audio."""

    def __init__(self):
        self.items: list = []

    def spawn(self, fn) -> None:
        threading.Thread(target=lambda: self.items.append([fn(), 0]), daemon=True).start()

    def mix(self, n: int) -> np.ndarray:
        out = np.zeros(n)
        for it in list(self.items):
            sig, pos = it
            chunk = sig[pos:pos + n]
            out[:len(chunk)] += chunk
            it[1] = pos + n
            if it[1] >= len(sig):
                self.items.remove(it)
        return out


# ---------------------------------------------------------------- chambre

BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
ALPHA = np.array([0.12, 0.16, 0.22, 0.28, 0.30, 0.32, 0.36])  # chambre meublée (plâtre, lit, rideaux)


def room_ir(area: float, damping: float, dist: float, rng) -> np.ndarray:
    """Réponse de la pièce SANS le son direct, normalisée par rapport à lui (direct = 1).

    Réflexions précoces par sources images (pièce rectangulaire, ordre 3) puis queue
    diffuse avec un temps de réverbération par bande (Sabine) : les aigus s'éteignent
    plus vite que les graves. L'énergie suit la théorie (distance critique) : plus le
    téléphone est loin de la bouche, plus la pièce s'entend.
    """
    h = 2.5
    ln = math.sqrt(area * 1.25)
    wd = area / ln
    vol = area * h
    surf = 2 * (ln * wd + ln * h + wd * h)
    alpha = np.clip(ALPHA * (0.4 + 1.2 * damping / 100), 0.04, 0.9)
    rt = 0.161 * vol / (surf * alpha)
    rc = 0.057 * math.sqrt(vol / rt[3])               # distance critique à 1 kHz
    n = int(SR * min(1.2, max(0.25, rt.max() * 1.1)))
    t = np.arange(n) / SR

    noise = rng.standard_normal(n)
    tail = np.zeros(n)
    for i, (fc, T) in enumerate(zip(BANDS, rt)):
        lo, hi = fc / math.sqrt(2), min(fc * math.sqrt(2), 0.45 * SR)
        sos = butter(4, hi, "low") if i == 0 else butter(4, lo, "high") if i == len(BANDS) - 1 \
            else butter(2, [lo, hi], "band")
        tail += signal.sosfilt(sos, noise) * np.exp(-6.91 * t / T)
    fade = np.clip((t - 0.006) / 0.04, 0, 1)
    tail *= fade * fade * (3 - 2 * fade)

    # Sources images : micro au bureau, bouche à « dist » mètres.
    mic = np.array([0.35 * ln, 0.45 * wd, 1.0])
    d = np.array([0.6, 0.7, 0.4])
    src = np.clip(mic + d / np.linalg.norm(d) * dist, 0.1, [ln - 0.1, wd - 0.1, h - 0.1])
    direct = float(np.linalg.norm(src - mic))
    m = np.arange(-2, 3)
    q = np.array([0, 1])
    pos, refl = [], []
    for size, s in zip((ln, wd, h), src):
        mm, qq = np.meshgrid(m, q, indexing="ij")
        pos.append(((1 - 2 * qq) * s + 2 * mm * size).ravel())
        refl.append((np.abs(mm - qq) + np.abs(mm)).ravel())
    px, py, pz = np.meshgrid(*pos, indexing="ij")
    rx, ry, rz = np.meshgrid(*refl, indexing="ij")
    order = (rx + ry + rz).ravel()
    r = np.sqrt((px.ravel() - mic[0]) ** 2 + (py.ravel() - mic[1]) ** 2 + (pz.ravel() - mic[2]) ** 2)
    keep = (order >= 1) & (order <= 3)
    beta = math.sqrt(1 - alpha[3])
    idx = ((r[keep] - direct) / 343 * SR).astype(int)
    amp = direct / r[keep] * beta ** order[keep]
    ok = (idx > 0) & (idx < n)
    er = np.zeros(n)
    np.add.at(er, idx[ok], amp[ok])
    er = signal.sosfilt(butter(1, 5000, "low"), er)    # chaque réflexion perd un peu d'aigus

    target = (direct / rc) ** 2                         # énergie réverbérée / énergie directe
    e_tail = max(target - float(np.sum(er * er)), target * 0.3)
    tail *= math.sqrt(e_tail / float(np.sum(tail * tail)))
    return er + tail


class Convolver:
    """Convolution par partitions uniformes (overlap-save), coût constant par bloc."""

    def __init__(self):
        self.H = None
        self._pending = None

    def set_ir(self, ir: np.ndarray) -> None:
        P = -(-len(ir) // BLOCK)
        ir = np.pad(ir, (0, P * BLOCK - len(ir)))
        self._pending = np.fft.rfft(ir.reshape(P, BLOCK), n=NFFT, axis=1)

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._pending is not None:
            H, self._pending = self._pending, None
            if self.H is None or H.shape != self.H.shape:
                self.fdl = np.zeros_like(H)
                self.idx = 0
                self.prev = np.zeros(BLOCK)
            self.H = H
        if self.H is None:
            return np.zeros_like(x)
        P = self.H.shape[0]
        self.idx = (self.idx + 1) % P
        self.fdl[self.idx] = np.fft.rfft(np.concatenate([self.prev, x]))
        self.prev = x
        order = (self.idx - np.arange(P)) % P
        Y = np.einsum("pk,pk->k", self.fdl[order], self.H)
        return np.fft.irfft(Y, NFFT)[BLOCK:]


# ---------------------------------------------------------------- bruits de fond

PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
PINK_A = [1, -2.494956002, 2.017265875, -0.522189400]

# Niveaux de référence à 100 % (dBFS avant le traitement du téléphone ; voix ≈ −20 dBFS à 30 cm).
REF = {"room_tone": -50, "fan": -47, "street": -52, "house": -24, "handling": -22}


class Ambience:
    def __init__(self, rng):
        self.rng = rng
        self.pink_z = np.zeros(3)
        self.tone_lp = Filt(butter(2, 3500, "low"))
        self.tone_am = Drift(rng, 5.0, 0.07, 1.0)
        self.fan_bp = Filt(butter(2, [160, 2400], "band"))
        self.fan_am = Drift(rng, 2.5, 0.10, 1.0)
        self.fan_fj = Drift(rng, 1.5, 0.004)
        self.fan_f0 = rng.uniform(95, 140)
        self.fan_ph = np.zeros(4)
        self.brown_z = np.zeros(1)
        self.street_lp = Filt(butter(2, 420, "low"))
        self.street_swell = Drift(rng, 8.0, 0.3, 1.0)
        self.cars, self.house_ev, self.handling_ev = Events(), Events(), Events()
        self.t_car = rng.exponential(12) * SR
        self.t_house = rng.exponential(25) * SR
        self.t_handling = rng.exponential(18) * SR
        # Normalisations (RMS unitaire) mesurées une fois sur une seconde de bruit.
        w = rng.standard_normal(SR)
        self.n_pink = 1 / _rms(signal.sosfilt(self.tone_lp.sos, signal.lfilter(PINK_B, PINK_A, w)))
        self.n_fan = 1 / _rms(signal.sosfilt(self.fan_bp.sos, w))
        self.n_street = 1 / _rms(signal.sosfilt(self.street_lp.sos, signal.lfilter([1], [1, -0.997], w)))

    @staticmethod
    def level(p, key) -> float:
        return db(REF[key]) * (p[key] / 100) ** 1.5

    def generate(self, p):
        """Renvoie (ambiance diffuse, sons à placer dans la pièce, sons collés au micro)."""
        rng, n = self.rng, BLOCK
        amb = np.zeros(n)
        if p["room_tone"]:
            pink, self.pink_z = signal.lfilter(PINK_B, PINK_A, rng.standard_normal(n), zi=self.pink_z)
            amb += self.tone_lp(pink) * self.n_pink * self.tone_am.ramp() * self.level(p, "room_tone")
        if p["fan"]:
            f = self.fan_f0 * (1 + self.fan_fj.ramp())
            tonal = np.zeros(n)
            for k, a in enumerate((1, 0.45, 0.25, 0.12)):
                ph = self.fan_ph[k] + 2 * math.pi * (k + 1) * np.cumsum(f) / SR
                self.fan_ph[k] = ph[-1] % (2 * math.pi)
                tonal += a * np.sin(ph)
            broad = self.fan_bp(rng.standard_normal(n)) * self.n_fan * self.fan_am.ramp()
            amb += (broad + 0.3 * tonal) * self.level(p, "fan")
        if p["street"]:
            brown, self.brown_z = signal.lfilter([1], [1, -0.997], rng.standard_normal(n), zi=self.brown_z)
            base = self.street_lp(brown) * self.n_street * np.clip(self.street_swell.ramp(), 0.3, None)
            self.t_car -= n
            if self.t_car <= 0:
                self.t_car = rng.exponential(22) * SR
                self.cars.spawn(self._car)
            amb += (base + self.cars.mix(n)) * self.level(p, "street")
        in_room = np.zeros(n)
        if p["house"]:
            self.t_house -= n
            if self.t_house <= 0:
                self.t_house = rng.exponential(35) * SR
                self.house_ev.spawn(self._house)
            in_room = self.house_ev.mix(n) * self.level(p, "house")
        close = np.zeros(n)
        if p["handling"]:
            self.t_handling -= n
            if self.t_handling <= 0:
                self.t_handling = rng.exponential(25) * SR
                self.handling_ev.spawn(self._handling)
            close = self.handling_ev.mix(n) * self.level(p, "handling")
        return amb, in_room, close

    # --- synthèse des événements (thread à part)
    def _car(self) -> np.ndarray:
        """Voiture qui passe dans la rue : grondement qui monte, s'éclaircit, puis s'éloigne."""
        rng = np.random.default_rng()
        m = int(rng.uniform(4, 9) * SR)
        u = np.linspace(0, 1, m)
        peak = rng.uniform(0.35, 0.65)
        e = np.where(u < peak, u / peak, (1 - u) / (1 - peak))
        e = e * e * (3 - 2 * e)
        w = rng.standard_normal(m)
        brown = signal.lfilter([1], [1, -0.995], w)
        lo = signal.sosfilt(butter(2, 250, "low"), brown)
        hi = signal.sosfilt(butter(2, 1000, "low"), brown)
        hiss = signal.sosfilt(butter(2, [1200, 3000], "band"), w)
        sig = (lo / _rms(lo) * (1 - e) + hi / _rms(hi) * e) * e ** 1.5 + 0.12 * hiss / _rms(hiss) * e ** 4
        sig = signal.sosfilt(butter(2, 1500, "low"), sig)   # à travers la fenêtre
        return sig * db(8) * rng.uniform(0.4, 1.0)

    def _house(self) -> np.ndarray:
        """Porte au loin, pas à l'étage ou petit objet posé dans la pièce."""
        rng = np.random.default_rng()
        kind = rng.choice(["porte", "pas", "objet"], p=[0.35, 0.35, 0.3])

        def thud(dur, fc, f0):
            k = int(dur * SR)
            t = np.arange(k) / SR
            env = np.exp(-t / (dur / 4))
            x = signal.sosfilt(butter(2, fc, "low"), rng.standard_normal(k)) * env
            return x / _rms(x) * 0.5 + np.sin(2 * math.pi * f0 * t) * env * 0.6

        if kind == "porte":
            x = thud(0.5, 250, rng.uniform(60, 90))
            click = signal.sosfilt(butter(2, [1800, 4000], "band"), rng.standard_normal(int(0.015 * SR)))
            x[:len(click)] += click / _rms(click) * 0.15
            x = signal.sosfilt(butter(2, 1200, "low"), x)    # derrière un mur
            return x * rng.uniform(0.3, 1.0)
        if kind == "pas":
            steps = int(rng.integers(4, 10))
            gap = rng.uniform(0.45, 0.62)
            out = np.zeros(int((steps * gap + 0.5) * SR))
            for i in range(steps):
                s = thud(0.18, 180, rng.uniform(50, 80)) * rng.uniform(0.5, 1.0)
                at = int((i * gap + rng.normal(0, 0.02)) * SR)
                at = max(0, at)
                out[at:at + len(s)] += s[:len(out) - at]
            ramp = np.sin(np.linspace(0, math.pi, len(out))) ** 0.7  # qui approchent puis s'éloignent
            return signal.sosfilt(butter(2, 400, "low"), out * ramp) * rng.uniform(0.25, 0.6)
        k = int(0.35 * SR)
        t = np.arange(k) / SR
        x = np.zeros(k)
        for _ in range(4):
            x += np.sin(2 * math.pi * rng.uniform(1500, 5200) * t) * np.exp(-t / rng.uniform(0.03, 0.12)) \
                * rng.uniform(0.3, 1.0)
        x[:int(0.004 * SR)] += rng.standard_normal(int(0.004 * SR)) * 0.8
        return x / np.abs(x).max() * rng.uniform(0.1, 0.35)

    def _handling(self) -> np.ndarray:
        """Le téléphone bouge dans la main ou sur le bureau : frottement et/ou petit choc."""
        rng = np.random.default_rng()
        dur = rng.uniform(0.25, 0.7)
        k = int(dur * SR)
        t = np.arange(k) / SR
        out = np.zeros(k)
        kind = rng.choice(["frottement", "choc", "les deux"], p=[0.6, 0.25, 0.15])
        if kind != "choc":
            env = np.zeros(k)
            for _ in range(int(rng.integers(2, 5))):
                c, w = rng.uniform(0.1, 0.9) * dur, rng.uniform(0.02, 0.08)
                env += np.exp(-0.5 * ((t - c) / w) ** 2) * rng.uniform(0.3, 1.0)
            r = signal.sosfilt(butter(2, [600, 6000], "band"), rng.standard_normal(k))
            out += r / _rms(r) * env * 0.25
        if kind != "frottement":
            at = int(rng.uniform(0, 0.5) * k)
            tt = t[:k - at]
            f0 = rng.uniform(80, 160)
            out[at:] += np.sin(2 * math.pi * f0 * tt) * np.exp(-tt / 0.025) * 0.9
        return out


# ---------------------------------------------------------------- le téléphone

class PhoneMic:
    def __init__(self):
        self.p = dict(DEFAULTS)
        self.extra_delay_ms = lambda: 0.0   # latence vidéo (synchro), fournie par l'interface
        self.rng = np.random.default_rng()
        self.amb = Ambience(self.rng)
        self.conv = Convolver()
        self.capsule = Filt(np.vstack([butter(2, 70, "high"), peaking(12500, 3.0, 1.4)]))
        self.win = np.sqrt(np.hanning(NFFT + 1)[:NFFT])   # sqrt-Hann périodique : reconstruction parfaite
        self.prev = np.zeros(BLOCK)
        self.ola = np.zeros(BLOCK)
        self.S = self.N = None
        self.G = np.ones(len(FREQS))
        self.P_prev = np.zeros(len(FREQS))
        self.agc_db = 0.0
        self.lim = 1.0
        self.pkt = 0
        self.in_loss = False
        self.lost = 0
        self.last_X = np.zeros(len(FREQS), complex)
        self.dbuf = np.zeros(SR * 3)
        self.dpos = 0
        self.cur_delay = 0
        self.ms = 0.0
        self.in_db = self.out_db = -120.0
        self._room_key = self._w_key = None
        self.W = np.ones(len(FREQS))
        self.bands = np.minimum(np.round(np.log2(np.maximum(FREQS, 200) / 200) * 3), 40).astype(int)

    # --- réglages dépendants (recalculés seulement quand ils changent)
    def _update(self):
        p = self.p
        key = (p["room_size"], p["damping"], p["distance"])
        if key != self._room_key:
            self._room_key = key
            # Calcul à côté : quelques dizaines de ms, à ne pas faire dans le fil audio.
            threading.Thread(target=lambda: self.conv.set_ir(
                room_ir(key[0], key[1], key[2] / 100, np.random.default_rng(7))), daemon=True).start()
        key = (p["codec"], p["tone"], p["bitrate"])
        if key != self._w_key:
            self._w_key = key
            _, hp, lp, adaptive, call = CODECS[int(p["codec"])]
            q = p["bitrate"] / 100
            if adaptive:
                lp = min(lp, lp * (0.45 + 0.55 * q))
            f = np.maximum(FREQS, 1)
            w = 1 / np.sqrt(1 + (hp / f) ** 8)
            x = np.clip((f - lp * 0.93) / (lp * 0.09), 0, 1)
            w *= 0.5 + 0.5 * np.cos(math.pi * x)
            tone = p["tone"] / 100
            w *= db(-3 * tone) ** (1 / (1 + (f / 250) ** 2))       # plateau grave
            w *= db(5 * tone) ** (1 / (1 + (3000 / f) ** 2))       # plateau aigu
            if call:  # réglage « voix » des téléphones en appel : présence vers 2,5 kHz
                w *= db(2.0 * np.exp(-0.5 * np.log2(f / 2500) ** 2))
            self.W = w

    def process(self, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        p = self.p
        x = x.astype(np.float64) * db(p["input_gain"])
        self.in_db = 20 * math.log10(float(np.abs(x).max()) + 1e-9)
        if not p["enabled"]:
            y = x * db(p["output_gain"])
        else:
            self._update()
            dist = p["distance"] / 100
            dry = x * float(np.clip(0.3 / dist, 0.25, 4))   # le son direct baisse avec la distance
            amb, in_room, close = self.amb.generate(p)
            s = dry + amb + close + in_room * 0.25
            if p["room"] > 0:
                s += self.conv.process(dry + in_room) * (p["room"] / 100)
            # Capsule MEMS : réponse, souffle propre, saturation puis convertisseur.
            s = self.capsule(s) + self.rng.standard_normal(BLOCK) * db(-80) * p["mic_noise"] / 100
            thr = 1 - 0.75 * p["saturation"] / 100
            s = np.clip(thr * np.tanh(s / thr), -1, 1)
            y = self._phone(s)
            y = self._limiter(y) * db(p["output_gain"])
        y = self._delay(y)
        self.out_db = 20 * math.log10(float(np.abs(y).max()) + 1e-9)
        self.ms = (time.perf_counter() - t0) * 1000
        return np.clip(y, -1, 1).astype(np.float32)

    def _phone(self, s: np.ndarray) -> np.ndarray:
        """Traitement du téléphone + codec, dans le domaine fréquentiel."""
        p = self.p
        X = np.fft.rfft(np.concatenate([self.prev, s]) * self.win)
        self.prev = s
        P = (X * np.conj(X)).real
        # Estimation du bruit (statistiques minimales) : minimum glissant sur ~1,6 s de la
        # puissance lissée en temps et en fréquence, corrigé de son biais.
        Pf = np.convolve(P, [0.25, 0.5, 0.25], "same")
        if self.S is None:
            self.S = Pf.copy()
            self.cur_min = Pf.copy()
            self.mins = []
            self.frames = 0
        self.S = 0.85 * self.S + 0.15 * Pf
        self.cur_min = np.minimum(self.cur_min, self.S)
        self.frames += 1
        if self.frames % 20 == 0:
            self.mins = (self.mins + [self.cur_min])[-8:]
            self.cur_min = self.S.copy()
        self.N = np.maximum(np.minimum.reduce(self.mins + [self.cur_min]) * 2.0, 1e-13)
        pre_lvl = 10 * math.log10(P.sum() / (NFFT * NFFT / 4) + 1e-12)
        noise_lvl = 10 * math.log10(self.N.sum() / (NFFT * NFFT / 4) + 1e-12)

        ns = p["noise_suppression"] / 100
        if ns > 0:  # Wiener « decision-directed » (Ephraim-Malah), comme dans les téléphones
            n_eff = self.N * (1 + 1.5 * ns)
            xi = 0.98 * self.G ** 2 * self.P_prev / n_eff + 0.02 * np.maximum(P / n_eff - 1, 0)
            G = xi / (1 + xi)
            G = np.convolve(G, [0.25, 0.5, 0.25], "same")
            self.G = np.maximum(G, 10 ** (-1.5 * ns))          # jusqu'à −30 dB
            X = X * self.G
        else:
            self.G = np.ones_like(P)
        self.P_prev = P

        X = X * db(self._agc(X, pre_lvl > noise_lvl + 6 and pre_lvl > -60))
        if p["air"] > 0:  # comme le SBR des codecs : recopie 3,5–7,5 kHz une octave plus haut
            sh = int(4000 / (SR / NFFT))
            src = slice(int(3500 / (SR / NFFT)), int(7500 / (SR / NFFT)))
            g = np.linspace(1, 0.3, src.stop - src.start) * 0.3 * p["air"] / 100
            X[src.start + sh:src.stop + sh] += X[src] * g
        X = X * self.W
        X = self._codec(X)
        y = np.fft.irfft(X, NFFT) * self.win
        out = y[:BLOCK] + self.ola
        self.ola = y[BLOCK:]
        return out

    def _agc(self, X, voice: bool) -> float:
        """Contrôle automatique de gain (vise −18 dBFS), figé dans les silences."""
        amt = self.p["agc"] / 100
        if amt <= 0:
            self.agc_db = 0.0
            return 0.0
        lvl = 10 * math.log10(float((X * np.conj(X)).real.sum()) / (NFFT * NFFT / 4) + 1e-12)
        if voice:
            want = float(np.clip(-18 - lvl, -12, 24 * amt))
            rate = 0.25 if want < self.agc_db else 0.012
            self.agc_db += (want - self.agc_db) * rate
        else:
            self.agc_db *= 0.9995
        return self.agc_db

    def _codec(self, X):
        p = self.p
        q = p["bitrate"] / 100
        if q < 0.98:
            # Peu de débit : les composantes faibles de chaque bande sont remplacées par du bruit
            # (« noise filling », son métallique / qui tourbillonne), et l'amplitude est quantifiée.
            mag = np.abs(X) + 1e-12
            e = mag * mag
            mean = np.bincount(self.bands, e) / np.maximum(np.bincount(self.bands), 1)
            hole = e < 1.4 * (1 - q) ** 1.5 * mean[self.bands]
            phase = np.exp(2j * math.pi * self.rng.random(len(X)))
            X = np.where(hole, mag * 0.55 * phase, X)
            step = 4.5 * (1 - q)
            if step > 0.6:
                ldb = 20 * np.log10(np.abs(X) + 1e-12)
                qdb = np.round(ldb / step + self.rng.uniform(-0.5, 0.5, len(X)) * 0.5) * step
                X = X * db(qdb - ldb)
        # Pertes de paquets (trames de 20 ms, par rafales) masquées comme le font les codecs.
        loss = p["packet_loss"] / 1000
        self.pkt ^= 1
        if self.pkt == 0:
            if self.in_loss:
                self.in_loss = self.rng.random() < 0.45
            else:
                self.in_loss = loss > 0 and self.rng.random() < loss
        if self.in_loss:
            self.lost += 1
            mag = np.abs(self.last_X) * 0.85 ** self.lost
            return mag * np.exp(2j * math.pi * self.rng.random(len(X)))
        self.lost = 0
        self.last_X = X
        return X

    def _limiter(self, y):
        pk = float(np.abs(y).max())
        target = min(1.0, 0.89 / pk) if pk > 0 else 1.0
        new = target if target < self.lim else self.lim + (target - self.lim) * 0.05
        g = np.linspace(self.lim, new, BLOCK)
        self.lim = new
        return 0.98 * np.tanh(y * g / 0.98)

    def _delay(self, y):
        L = len(self.dbuf)
        idx = (self.dpos + np.arange(BLOCK)) % L
        self.dbuf[idx] = y
        self.dpos = (self.dpos + BLOCK) % L
        want = self.p["delay"] + (self.extra_delay_ms() if self.p["sync_video"] else 0)
        want = int(np.clip(want, 0, 2500) * SR / 1000)
        if self.cur_delay == 0 and want == 0:
            return y
        read = lambda d: self.dbuf[(idx - d) % L]
        if abs(want - self.cur_delay) > SR // 100:  # > 10 ms : on change, en fondu sur le bloc
            fade = np.linspace(0, 1, BLOCK)
            out = read(self.cur_delay) * (1 - fade) + read(want) * fade
            self.cur_delay = want
            return out
        return read(self.cur_delay)


# ---------------------------------------------------------------- entrées / sorties

# Latence demandée aux cartes son. En « low », l'entrée WASAPI avec conversion 44,1 → 48 kHz
# (câbles VB-Audio) livre par à-coups et trop lentement : coupures en sortie.
LATENCY = 0.03
PHONE_INPUT = "Micro du téléphone (PhoneCam)"   # entrée réseau, à la place d'une carte son


def devices(kind: str) -> list[str]:
    """Périphériques WASAPI (noms complets, une seule entrée par appareil). kind : input | output."""
    import sounddevice as sd
    apis = [h["name"] for h in sd.query_hostapis()]
    wasapi = apis.index("Windows WASAPI") if "Windows WASAPI" in apis else None
    return [d["name"] for d in sd.query_devices()
            if (wasapi is None or d["hostapi"] == wasapi) and d[f"max_{kind}_channels"] > 0]


def rescan() -> None:
    """Relit la liste des périphériques (ex. écouteurs Bluetooth connectés après le lancement)."""
    import sounddevice as sd
    sd._terminate()
    sd._initialize()


class Fifo:
    """Tampon entre le traitement et une carte son ; absorbe les petits écarts d'horloge.

    Après un manque, il attend d'avoir de nouveau « keep » échantillons avant de reprendre :
    une arrivée par à-coups (Wi-Fi) ne provoque qu'une seule coupure, pas une rafale.
    """

    def __init__(self, keep=BLOCK * 5, limit=BLOCK * 16, adaptive=False):
        self._buf = np.zeros(keep, np.float32)
        self._lock = threading.Lock()
        self.keep, self.limit = keep, limit
        self.adaptive = adaptive        # réseau : le tampon grandit après chaque coupure (≤ 200 ms)
        self.underruns = 0
        self._gap = False
        self._refill = False
        self.started = None             # le démarrage des flux (premières 0,5 s) ne compte pas

    def write(self, x):
        now = time.perf_counter()
        if self.started is None:
            self.started = now
        if self._gap:  # le son reprend après un manque : c'était bien une coupure
            self._gap = False
            self.underruns += 1
        with self._lock:
            self._buf = np.concatenate([self._buf, x])
            if len(self._buf) > self.limit:
                self._buf = self._buf[-self.keep:]

    def read(self, n):
        with self._lock:
            if self._refill and len(self._buf) < self.keep:
                return np.zeros(n, np.float32)
            self._refill = False
            out, self._buf = self._buf[:n], self._buf[n:]
        if len(out) < n:
            self._refill = True
            if self.adaptive:
                self.keep = min(self.keep + BLOCK * 2, BLOCK * 20)
                self.limit = max(self.limit, self.keep * 3)
            # Comptée à la reprise (write) : une source qui s'arrête n'est pas une coupure.
            # Le démarrage des flux (0,5 s) ne compte pas non plus.
            if self.started is not None and time.perf_counter() - self.started > 0.5:
                self._gap = True
            out = np.concatenate([out, np.zeros(n - len(out), np.float32)])
        return out


class AudioLink:
    """Entrée → PhoneMic → sortie (+ écoute de contrôle facultative)."""

    def __init__(self, mic: PhoneMic, server=None):
        self.mic = mic
        self.server = server            # PhoneServer : source du micro du téléphone
        self.running = False
        self.status = "Arrêté."
        self.monitor_gain = 1.0
        self.muted = False              # micro coupé (bouton de l'appli en mode Appel)
        self._worker = None
        self._streams = []

    def start(self, in_name: str, out_name: str, mon_name: str | None = None) -> None:
        import sounddevice as sd
        self.stop()
        try:
            apis = [h["name"] for h in sd.query_hostapis()]
            wasapi = apis.index("Windows WASAPI")
            devs = sd.query_devices()

            def find(name, kind):
                for i, d in enumerate(devs):
                    if d["name"] == name and d["hostapi"] == wasapi and d[f"max_{kind}_channels"] > 0:
                        return i, d
                raise RuntimeError(f"périphérique introuvable : {name}")
            ext = sd.WasapiSettings(auto_convert=True)   # 44,1 / 48 kHz : Windows convertit
            net = in_name == PHONE_INPUT
            self._q = queue.Queue(maxsize=100)
            self._fifos = []
            self._streams = []
            if net:
                if self.server is None:
                    raise RuntimeError("serveur PhoneCam absent")
            else:
                ii, di = find(in_name, "input")
                self._streams.append(sd.InputStream(
                    device=ii, channels=min(2, di["max_input_channels"]), samplerate=SR, blocksize=BLOCK,
                    dtype="float32", latency=LATENCY, extra_settings=ext, callback=self._in_cb))
            for name, gain in ((out_name, None), (mon_name, "monitor")):
                if not name:
                    continue
                oi, do = find(name, "output")
                # Par le réseau, le son arrive par paquets irréguliers : tampon plus large (~100 ms).
                fifo = Fifo(BLOCK * 10, BLOCK * 40, adaptive=True) if net else Fifo()
                self._fifos.append((fifo, gain))
                self._streams.append(sd.OutputStream(
                    device=oi, channels=min(2, do["max_output_channels"]), samplerate=SR,
                    blocksize=BLOCK, dtype="float32", latency=LATENCY, extra_settings=ext,
                    callback=self._out_cb(fifo)))
            self.running = True
            self._worker = threading.Thread(target=self._work, daemon=True, name="phonemic")
            self._worker.start()
            if net:
                self.server.audio_sink = self._net_in
            for s in self._streams:
                s.start()
            self.status = f"En marche : {in_name} → {out_name}" + (f" (écoute : {mon_name})" if mon_name else "")
        except Exception as e:
            self.stop()
            self.status = f"Impossible de démarrer : {e}"

    def stop(self) -> None:
        self.running = False
        if self._worker is not None:      # un seul fil de traitement à la fois
            self._worker.join(1)
            self._worker = None
        if getattr(self.server, "audio_sink", None) == self._net_in:
            self.server.audio_sink = None
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams = []
        if not self.status.startswith("Impossible"):
            self.status = "Arrêté."

    def underruns(self) -> int:
        return sum(f.underruns for f, _ in getattr(self, "_fifos", []))

    def _in_cb(self, indata, frames, t, status):
        try:
            self._q.put_nowait(indata.mean(axis=1).copy())
        except queue.Full:
            pass

    def _net_in(self, pcm):
        try:
            self._q.put_nowait(pcm)
        except queue.Full:
            pass

    def _out_cb(self, fifo):
        def cb(outdata, frames, t, status):
            outdata[:] = fifo.read(frames)[:, None]
        return cb

    def _work(self):
        acc = np.zeros(0, np.float32)
        while self.running:
            try:
                acc = np.concatenate([acc, self._q.get(timeout=0.2)])
            except queue.Empty:
                continue
            while len(acc) >= BLOCK:
                blk, acc = acc[:BLOCK], acc[BLOCK:]
                try:
                    y = self.mic.process(blk)
                except Exception as e:  # ne jamais couper le son pour une erreur de traitement
                    self.status = f"Erreur de traitement : {e}"
                    y = blk
                if self.muted:
                    y = np.zeros_like(y)
                for fifo, gain in self._fifos:
                    fifo.write(y * self.monitor_gain if gain else y)
