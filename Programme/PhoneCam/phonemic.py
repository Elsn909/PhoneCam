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
import os
import queue
import threading
import time

import numpy as np
from scipy import signal

from effets_micro import DEFAULTS as FX_DEFAULTS, VoiceFx
from entree_micro import DEFAULTS as IN_DEFAULTS, MicInput

SR = 48000
BLOCK = 480                 # 10 ms
NFFT = 2 * BLOCK            # trames de 20 ms
FREQS = np.fft.rfftfreq(NFFT, 1 / SR)

DEFAULTS = {
    "enabled": True,        # False : son d'entrée tel quel (comparaison A/B)
    "output_gain": 0,       # dB
    # chambre
    "color": 60,            # couleur d'un micro de smartphone récent, %
    "room": 30,             # réverbération, % du niveau physique pour cette distance
    "room_size": 12,        # surface au sol, m² (hauteur 2,5 m)
    "damping": 60,          # amortissement : lit, rideaux, tapis… %
    "distance": 25,         # bouche → téléphone, cm
    # bruits de fond (0 = coupé)
    "room_tone": 30,        # ambiance de la pièce (très léger : ≈ −68 dBFS)
    "fan": 0,               # ventilateur / PC
    "street": 0,            # (ancienne simulation, plus utilisée)
    "house": 0,
    "handling": 0,
    # capsule
    "mic_noise": 15,        # souffle du micro, très léger
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

DEFAULTS.update(FX_DEFAULTS)    # effets de voix (égaliseur, compresseur…), avant la simulation
DEFAULTS.update(IN_DEFAULTS)    # entrée : sensibilité, niveau automatique, voix à distance (avant tout)

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
        self.fx = VoiceFx(self.p)               # chaîne de studio, avant le téléphone
        self.inp = MicInput(self.p)             # entrée du micro, avant le changeur de voix
        self.amb = Ambience(self.rng)
        self.conv = Convolver()
        self.capsule = Filt(np.vstack([butter(2, 70, "high"), peaking(12500, 3.0, 1.4)]))
        self._color_key = None
        self.color = None
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
        x = self.fx.process(x.astype(np.float64))
        if not p["enabled"]:
            y = x * db(p["output_gain"])
        else:
            # Smartphone d'aujourd'hui, filmé ou en vocal : son propre, une petite pièce, un souffle
            # à peine audible. Pas de codec, de pertes ni de saturation (plus de « micro pourri »).
            self._update()
            dist = p["distance"] / 100
            dry = x * float(np.clip(0.25 / dist, 0.3, 3))   # le son direct baisse avec la distance
            amb, _in_room, _close = self.amb.generate(dict(p, street=0, house=0, handling=0))
            s = dry + amb
            if p["room"] > 0:
                s += self.conv.process(dry) * (p["room"] / 100)
            s = self._color(s) + self.rng.standard_normal(BLOCK) * db(-84) * p["mic_noise"] / 100
            y = self._limiter(s) * db(p["output_gain"])
        y = self._delay(y)
        self.out_db = 20 * math.log10(float(np.abs(y).max()) + 1e-9)
        self.ms = (time.perf_counter() - t0) * 1000
        return np.clip(y, -1, 1).astype(np.float32)

    def _color(self, s):
        """Réponse d'un micro de smartphone récent (MEMS + traitement « vidéo / mémo ») : graves
        coupés vers 100 Hz, bas-médium allégé, présence vers 4 kHz, extrême aigu adouci."""
        amt = self.p["color"] / 100
        key = (round(amt, 2), self.p["tone"])
        if key != self._color_key:
            self._color_key = key
            t = self.p["tone"] / 100
            sos = np.vstack([butter(2, 40 + 70 * amt, "high"), peaking(330, -2.5 * amt, 0.8),
                             peaking(4200, 3.0 * amt + 2 * t, 0.9), peaking(13500, -3.5 * amt + 2 * t, 0.7),
                             peaking(180, -2 * t, 0.7)])
            zi = self.color.zi if self.color is not None and self.color.zi.shape[0] == sos.shape[0] else None
            self.color = Filt(sos)
            if zi is not None:
                self.color.zi = zi
        return self.color(s)

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


class Ring:
    """Les n derniers échantillons (tampon circulaire)."""

    def __init__(self, n):
        self.buf = np.zeros(n, np.float32)
        self.pos = 0
        self.full = False

    def write(self, x):
        x = np.asarray(x, np.float32)[-len(self.buf):]
        k = len(x)
        end = self.pos + k
        if end <= len(self.buf):
            self.buf[self.pos:end] = x
        else:
            first = len(self.buf) - self.pos
            self.buf[self.pos:] = x[:first]
            self.buf[:k - first] = x[first:]
        if end >= len(self.buf):
            self.full = True
        self.pos = end % len(self.buf)

    def get(self):
        return np.concatenate([self.buf[self.pos:], self.buf[:self.pos]]) if self.full else self.buf[:self.pos].copy()


class Fifo:
    """Tampon entre le traitement et une carte son ; absorbe les petits écarts d'horloge.

    Après un manque, il attend d'avoir de nouveau « keep » échantillons avant de reprendre :
    une arrivée par à-coups (Wi-Fi) ne provoque qu'une seule coupure, pas une rafale.
    """

    def __init__(self, keep=BLOCK * 5, limit=BLOCK * 16, adaptive=False, pad=0):
        self._buf = np.zeros(keep, np.float32)
        # Changeur de voix : le son arrive par blocs entiers (200 ms). Après un manque, on remet
        # « pad » échantillons de silence devant le bloc suivant, pour retrouver une réserve.
        self.pad = pad
        self._pad_next = False
        self._lo = [1 << 30, 1 << 30]   # niveau le plus bas sur la seconde en cours et la précédente
        self._lo_t = time.perf_counter()
        self._last_gap = 0.0            # instant du dernier manque
        # Changeur de voix : réserve gardée en plus de « pad » quand on raccourcit un silence. Le temps
        # de calcul d'un bloc varie (40 à 170 ms mesurés avec la vidéo) : avec seulement « pad », un
        # bloc un peu lent vidait la réserve (coupure toutes les 12–14 s). Elle grandit de 30 ms à
        # chaque manque (jusqu'à 200 ms) et redescend de 10 ms par minute sans manque.
        self.safety = int(0.06 * SR) if pad else 0
        self._safety_t = time.perf_counter()
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
        if self._pad_next:
            self._pad_next = False
            x = np.concatenate([np.zeros(self.pad, np.float32), x])
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
            if self.pad:
                self._lo[0] = min(self._lo[0], len(self._buf))
                if time.perf_counter() - self._lo_t > 1.0:
                    self._lo = [1 << 30, self._lo[0]]
                    self._lo_t = time.perf_counter()
        if len(out) < n:
            self._refill = True
            self._pad_next = self.pad > 0
            self._last_gap = time.perf_counter()
            if self.pad and self.started is not None and time.perf_counter() - self.started > 0.5:
                self.safety = min(self.safety + int(0.03 * SR), int(0.2 * SR))
            if self.adaptive:
                self.keep = min(self.keep + BLOCK * 2, BLOCK * 20)
                self.limit = max(self.limit, self.keep * 3)
            # Comptée à la reprise (write) : une source qui s'arrête n'est pas une coupure.
            # Le démarrage des flux (0,5 s) ne compte pas non plus.
            if self.started is not None and time.perf_counter() - self.started > 0.5:
                self._gap = True
            out = np.concatenate([out, np.zeros(n - len(out), np.float32)])
        return out


    def excess(self) -> int:
        """Retard en trop (échantillons) : la réserve n'est jamais descendue sous « pad » depuis 1 à 2 s."""
        now = time.perf_counter()
        if now - self._last_gap < 20.0:    # juste après un manque : on garde la réserve agrandie
            return 0
        if self.safety > int(0.06 * SR) and now - max(self._last_gap, self._safety_t) > 60:
            self.safety -= int(0.01 * SR)
            self._safety_t = now
        lo = min(self._lo)
        return max(0, lo - self.pad - self.safety) if self.pad and lo < 1 << 29 else 0

    def dropped(self, n: int) -> None:
        self._lo = [v - n for v in self._lo]


def load_audio(path: str, limit_s: float = 600) -> np.ndarray:
    """N'importe quel fichier audio (ou vidéo) → 48 kHz mono float32, par ffmpeg."""
    import subprocess
    out = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-t", str(limit_s), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(out.stderr.decode("utf-8", "replace").strip().splitlines()[-1] if out.stderr else "fichier illisible")
    return np.frombuffer(out.stdout, np.float32).copy()


class TestClip:
    """Extrait de test : remplace le micro, au rythme du micro, et passe par toute la chaîne."""

    def __init__(self, data, name):
        self.data, self.name = data.astype(np.float32), name
        self.pos = 0
        self.loop = True
        self.original = False       # A/B : le fichier tel quel, sans voix ni effets
        self.to_cable = False       # par défaut : seulement dans l'écoute (personne d'autre n'entend)
        self.playing = False

    def take(self, n):
        if not self.playing or not len(self.data):
            return np.zeros(n, np.float32)
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        if len(out) < n:
            if self.loop:
                self.pos = n - len(out)
                out = np.concatenate([out, self.data[:self.pos]])
            else:
                self.playing = False
                self.pos = 0
                out = np.concatenate([out, np.zeros(n - len(out), np.float32)])
        return out


class AudioLink:
    """Entrée → [changeur de voix] → PhoneMic → sortie (+ écoute de contrôle facultative)."""

    def __init__(self, mic: PhoneMic, server=None, voice=None):
        self.mic = mic
        self.voice = voice              # voix.Voix : changeur de voix (actif si voice.p["enabled"])
        self.vox = False                # changeur de voix en service sur ce démarrage
        self.test = None                # TestClip : un fichier à la place du micro (banc d'essai)
        self.rec = None                 # [échantillons, nombre voulu, rappel] : enregistrer le micro brut
        self.server = server            # PhoneServer : source du micro du téléphone
        self.running = False
        self.status = "Arrêté."
        self.monitor_gain = 1.0
        self.muted = False              # micro coupé (bouton de l'appli en mode Appel)
        self._worker = None
        self._streams = []
        self.lost_in = 0                # blocs du micro perdus (traitement en retard)
        # Boîte noire : les 30 dernières secondes (micro brut, sortie), gardées sur demande
        # (« Garder les 30 dernières secondes ») ou toutes seules si ta voix est nette mais ne sort pas.
        self._bb_in = Ring(SR * 30)
        self._bb_out = Ring(SR * 30)
        self._bb_last = 0.0
        self.bb_status = ""
        self._diag = (0, 0, 0)

    def _journal(self, msg):
        """data/voix-journal.log : ce qui a pu couper le son (pour comprendre une coupure après coup)."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "voix-journal.log")
        try:
            if os.path.exists(path) and os.path.getsize(path) > 300_000:
                os.replace(path, path + ".1")
            with open(path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%d/%m %H:%M:%S ") + msg + "\n")
        except OSError:
            pass

    def _check(self, n):
        """Toutes les secondes : coupures, son perdu, blocs de voix trop lents."""
        u, lost = self.underruns(), self.lost_in
        u0, l0, _ = self._diag
        v = self.voice if self.vox else None
        slow = v.stat("ms_max", 0.0) if v is not None else 0.0
        block = (v.p["block"] * 10) if v is not None else 0
        if u > u0 or lost > l0 or (v is not None and slow > 0.75 * block):
            parts = []
            if u > u0:
                parts.append(f"{u - u0} manque(s) de son en sortie")
            if lost > l0:
                parts.append(f"{lost - l0} bloc(s) du micro perdus")
            if v is not None:
                parts.append(f"voix : bloc {v.stat('ms', 0.0):.0f} ms (max {slow:.0f} ms pour {block} ms)"
                             + (f", {v.status}" if "rreur" in str(v.status) else ""))
            parts.append(f"file d'entrée {self._q.qsize()}")
            self._journal(" · ".join(parts))
        self._diag = (u, lost, n)
        # Ta voix nette (une hauteur, assez forte) pendant ~1 s, mais Smart fermé : on garde tout.
        eng = getattr(self.voice, "engine", None) if self.vox else None
        if eng is not None and len(getattr(eng, "diag", ())) >= 5 and time.perf_counter() - self._bb_last > 120:
            d = np.concatenate(list(eng.diag)[-5:])
            clear = (d[:, 1] > 0.6) & (d[:, 0] > -45)
            if clear.sum() >= 50 and (d[clear, 3] == 0).mean() > 0.8:
                self.keep_last("automatique : ta voix est nette mais Smart reste fermé")

    def keep_last(self, reason="à ta demande"):
        """Enregistre les 30 dernières secondes dans data/coupures-voix (micro brut à gauche, sortie à
        droite) avec l'état de Smart trame par trame, pour comprendre une coupure après coup."""
        self._bb_last = time.perf_counter()
        a, b = self._bb_in.get(), self._bb_out.get()
        eng = getattr(self.voice, "engine", None)
        d = np.concatenate(list(eng.diag)) if eng is not None and len(getattr(eng, "diag", ())) else np.zeros((0, 6))
        info = {"raison": reason, "voix": bool(self.vox), "muet": bool(self.muted), "statut": self.status,
                "moteur": str(getattr(self.voice, "status", "")),
                "reglages_voix": {k: v for k, v in (self.voice.p.items() if self.voice is not None else [])
                                  if isinstance(v, (int, float, bool))},
                "entree": {k: self.mic.p[k] for k in ("input_gain", "auto_level")},
                "gain_auto": float(self.mic.inp.gain), "manques": self.underruns()}

        def save():
            import json
            from scipy.io import wavfile
            folder = getattr(self, "bb_folder", None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "coupures-voix")
            os.makedirs(folder, exist_ok=True)
            name = time.strftime("%Y-%m-%d_%H-%M-%S")
            n = min(len(a), len(b))
            st = np.stack([a[-n:], b[-n:]], 1) if n else np.zeros((0, 2))
            wavfile.write(os.path.join(folder, name + ".wav"), SR, (np.clip(st, -1, 1) * 32767).astype(np.int16))
            np.save(os.path.join(folder, name + "_smart.npy"), d)
            with open(os.path.join(folder, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=1)
            old = sorted(x for x in os.listdir(folder) if x.endswith(".wav"))
            for x in old[:-10]:                     # on garde les 10 dernières
                for ext in (".wav", "_smart.npy", ".json"):
                    try:
                        os.remove(os.path.join(folder, x[:-4] + ext))
                    except OSError:
                        pass
            self.bb_status = f"30 dernières secondes gardées ({reason}) : data/coupures-voix/{name}.wav"
            self._journal(f"boîte noire : {name}.wav ({reason})")
        threading.Thread(target=save, daemon=True).start()

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
            self.vox = self.voice is not None and bool(self.voice.p["enabled"])
            for name, gain in ((out_name, None), (mon_name, "monitor")):
                if not name:
                    continue
                oi, do = find(name, "output")
                if self.vox:            # blocs entiers + réserve contre les à-coups de la carte graphique
                    blk = int(self.voice.p["block"]) * BLOCK
                    pad = int(self.voice.p["margin_ms"]) * SR // 1000 + (BLOCK * 5 if net else 0)
                    fifo = Fifo(pad, 2 * blk + 4 * pad, adaptive=net, pad=pad)
                # Par le réseau, le son arrive par paquets irréguliers : tampon plus large (~100 ms).
                elif net:
                    fifo = Fifo(BLOCK * 10, BLOCK * 40, adaptive=True)
                else:
                    fifo = Fifo()
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
            self.status = (f"En marche : {in_name} → {'changeur de voix → ' if self.vox else ''}{out_name}"
                           + (f" (écoute : {mon_name})" if mon_name else ""))
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
            self.lost_in += 1

    def _net_in(self, pcm):
        try:
            self._q.put_nowait(pcm)
        except queue.Full:
            self.lost_in += 1

    def _out_cb(self, fifo):
        def cb(outdata, frames, t, status):
            outdata[:] = fifo.read(frames)[:, None]
        return cb

    def _work(self):
        acc = np.zeros(0, np.float32)
        n = 0
        while self.running:
            try:
                acc = np.concatenate([acc, self._q.get(timeout=0.2)])
            except queue.Empty:
                continue
            while len(acc) >= BLOCK:
                blk, acc = acc[:BLOCK], acc[BLOCK:]
                n += 1
                if n % 100 == 0:
                    self._check(n)
                rec = self.rec
                if rec is not None:     # extrait de ta voix pour le banc d'essai (son brut du micro)
                    rec[0].append(blk.copy())
                    if len(rec[0]) * BLOCK >= rec[1]:
                        self.rec = None
                        rec[2](np.concatenate(rec[0]))
                self._bb_in.write(blk)
                test = self.test if self.test is not None and self.test.playing else None
                if test is not None:
                    blk = test.take(BLOCK)
                bypass = test is not None and test.original
                if not bypass:          # sensibilité, niveau automatique, voix à distance : avant tout
                    try:
                        blk = self.mic.inp.process(blk)
                    except Exception as e:  # noqa: BLE001
                        self.status = f"Erreur d'entrée : {e}"
                if bypass:
                    pieces = [blk]
                elif self.vox:          # voix transformée : rendue par blocs entiers (ou rien)
                    v = self.voice.feed(blk)
                    # Retard en trop (après une coupure) : rattrapé en raccourcissant un silence.
                    if len(v) and self.voice.stat("last_silent", False):
                        extra = min(f.excess() for f, _g in self._fifos) // BLOCK * BLOCK if self._fifos else 0
                        if extra >= BLOCK * 2:
                            cut = min(extra, len(v))
                            v = v[cut:]
                            for f, _g in self._fifos:
                                f.dropped(cut)
                    pieces = [v[i:i + BLOCK] for i in range(0, len(v) - BLOCK + 1, BLOCK)]
                else:
                    pieces = [blk]
                private = test is not None and not test.to_cable and len(self._fifos) > 1
                for piece in pieces:
                    try:
                        y = piece if bypass else self.mic.process(piece)
                    except Exception as e:  # ne jamais couper le son pour une erreur de traitement
                        self.status = f"Erreur de traitement : {e}"
                        y = piece
                    if self.muted:
                        y = np.zeros_like(y)
                    self._bb_out.write(y)
                    for fifo, gain in self._fifos:
                        if private and not gain:        # test : rien dans le câble, seulement l'écoute
                            fifo.write(np.zeros_like(y))
                        else:
                            fifo.write(y * self.monitor_gain if gain else y)
