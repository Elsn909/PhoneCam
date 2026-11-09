"""Effets de voix du micro (studio) : égaliseur, de-esser, compresseur, chaleur.

S'appliquent à la voix (transformée ou non) AVANT la simulation du téléphone, comme une chaîne
de studio : coupe-bas → égaliseur 7 bandes → de-esser → compresseur → chaleur.
Blocs de 10 ms à 48 kHz, filtres biquad (formules de R. Bristow-Johnson), état conservé entre
les blocs : aucun clic. Coût : bien moins d'une milliseconde par bloc.
"""
import math

import numpy as np
from scipy import signal

SR = 48000

# Bandes de l'égaliseur : (type, fréquence, gain dB, Q). Les types ne changent pas.
BANDS = [("hp", 80, 0, 0.71), ("low", 120, 0, 0.71), ("peak", 300, 0, 1.0), ("peak", 1200, 0, 1.0),
         ("peak", 3500, 0, 1.0), ("peak", 7000, 0, 1.0), ("high", 10000, 0, 0.71), ("lp", 18000, 0, 0.71)]
BAND_NAMES = ["Coupe-bas", "Grave", "Bas-médium", "Médium", "Présence", "Brillance", "Air", "Coupe-haut"]

DEFAULTS = {"fx_on": True, "eq_on": True, "hp_on": True, "lp_on": False,
            "comp_on": False, "comp_thr": -24, "comp_ratio": 30, "comp_att": 8, "comp_rel": 120, "comp_makeup": 0,
            "deess_on": False, "deess_freq": 6500, "deess_thr": -30, "deess_amt": 60,
            "warm": 0}
for _i, (_t, _f, _g, _q) in enumerate(BANDS):
    DEFAULTS[f"eq{_i}_f"] = _f
    DEFAULTS[f"eq{_i}_g"] = _g
    DEFAULTS[f"eq{_i}_q"] = int(round(_q * 100))

PRESETS = {
    "Neutre": {},
    "Voix claire": {"eq0_f": 90, "eq2_g": -2, "eq2_f": 280, "eq4_g": 3, "eq4_f": 3200, "eq6_g": 2, "comp_on": True,
                    "comp_thr": -26, "comp_ratio": 30, "deess_on": True},
    "Chaude et proche": {"eq0_f": 60, "eq1_g": 3, "eq1_f": 140, "eq3_g": -2, "eq3_f": 900, "eq6_g": -1, "comp_on": True,
                         "comp_thr": -22, "comp_ratio": 25, "warm": 25},
    "Podcast / radio": {"eq0_f": 85, "eq1_g": 2, "eq2_g": -3, "eq2_f": 350, "eq4_g": 4, "eq4_f": 3000, "eq6_g": 3,
                        "comp_on": True, "comp_thr": -30, "comp_ratio": 45, "comp_makeup": 4, "deess_on": True, "warm": 15},
    "Petit haut-parleur": {"eq0_f": 280, "lp_on": True, "eq7_f": 7500, "eq3_g": 4, "eq3_f": 1500, "comp_on": True,
                           "comp_thr": -28, "comp_ratio": 40, "warm": 35},
}


def biquad(kind, f, g, q):
    """Coefficients (sos, 1 section) d'un biquad RBJ."""
    f = min(max(f, 10.0), SR * 0.49)
    w = 2 * math.pi * f / SR
    cw, sw = math.cos(w), math.sin(w)
    q = max(q, 0.1)
    a = sw / (2 * q)
    A = 10 ** (g / 40)
    if kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        d = [1 + a, -2 * cw, 1 - a]
    elif kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        d = [1 + a, -2 * cw, 1 - a]
    elif kind == "peak":
        b = [1 + a * A, -2 * cw, 1 - a * A]
        d = [1 + a / A, -2 * cw, 1 - a / A]
    else:  # étagères (pente S = 1)
        sa = 2 * math.sqrt(A) * a
        if kind == "low":
            b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
            d = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
            d = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return np.array([b[0] / d[0], b[1] / d[0], b[2] / d[0], 1.0, d[1] / d[0], d[2] / d[0]])


def eq_sections(p):
    """Les biquads actifs de l'égaliseur, dans l'ordre (liste de (indice de bande, sos))."""
    out = []
    if not p["eq_on"]:
        return out
    for i, (kind, _f, _g, _q) in enumerate(BANDS):
        if kind == "hp" and not p["hp_on"] or kind == "lp" and not p["lp_on"]:
            continue
        g = 0 if kind in ("hp", "lp") else p[f"eq{i}_g"]
        if kind in ("peak", "low", "high") and g == 0:
            continue
        out.append((i, biquad(kind, p[f"eq{i}_f"], g, p[f"eq{i}_q"] / 100)))
    return out


def response(p, freqs):
    """Réponse de l'égaliseur (dB) aux fréquences données, pour la courbe de l'interface."""
    h = np.ones(len(freqs), complex)
    for _i, sos in eq_sections(p):
        _w, hh = signal.sosfreqz(sos[None], worN=freqs, fs=SR)
        h *= hh
    return 20 * np.log10(np.abs(h) + 1e-9)


class VoiceFx:
    def __init__(self, p):
        self.p = p
        self._key = None
        self.sos = np.zeros((0, 6))
        self.zi = np.zeros((0, 2))
        self.env = -90.0            # niveau suivi par le compresseur (dB)
        self.gr = 0.0               # réduction de gain actuelle (dB), pour l'interface
        self.de_env = -90.0
        self.de_gr = 0.0
        self._de_key = None
        self.ms = 0.0
        self.spec = np.zeros(129)   # spectre de sortie lissé (pour la courbe), 0–24 kHz
        self._win = np.hanning(480)

    def _update(self):
        p = self.p
        key = tuple(p[k] for k in sorted(DEFAULTS) if k.startswith(("eq", "hp", "lp")))
        if key != self._key:
            self._key = key
            secs = eq_sections(p)
            sos = np.array([s for _i, s in secs]) if secs else np.zeros((0, 6))
            if sos.shape[0] != self.sos.shape[0]:
                self.zi = np.zeros((sos.shape[0], 2))     # nombre de bandes changé : on repart à zéro
            self.sos = sos
        key = p["deess_freq"]
        if key != self._de_key:
            self._de_key = key
            self.de_sos = signal.butter(2, key, "high", fs=SR, output="sos")
            self.de_zi = np.zeros((self.de_sos.shape[0], 2))

    def process(self, x):
        import time
        t0 = time.perf_counter()
        p = self.p
        if not p["fx_on"]:
            self.gr = self.de_gr = 0.0
            return x
        self._update()
        y = x.astype(np.float64)
        if len(self.sos):
            y, self.zi = signal.sosfilt(self.sos, y, zi=self.zi)
        if p["deess_on"]:
            y = self._deess(y)
        if p["comp_on"]:
            y = self._comp(y)
        else:
            self.gr = 0.0
        w = p["warm"] / 100
        if w > 0:  # saturation douce, surtout des harmoniques paires (asymétrique), volume compensé
            drive = 1 + 5 * w
            s = np.tanh(drive * y + 0.15 * w) - math.tanh(0.15 * w)
            y = (1 - w) * y + w * s / math.tanh(drive)
        X = np.abs(np.fft.rfft(y[-256:] * np.hanning(256)))
        self.spec = 0.8 * self.spec + 0.2 * 20 * np.log10(X + 1e-9)
        self.ms = (time.perf_counter() - t0) * 1000
        return y

    def _deess(self, y):
        """Bande aiguë (sifflantes) suivie à part ; au-dessus du seuil, elle est baissée."""
        p = self.p
        hi, self.de_zi = signal.sosfilt(self.de_sos, y, zi=self.de_zi)
        lvl = 10 * math.log10(float(np.mean(hi ** 2)) + 1e-12)
        a = 0.5 if lvl > self.de_env else 0.08
        self.de_env += (lvl - self.de_env) * a
        over = max(0.0, self.de_env - p["deess_thr"])
        want = -min(over * p["deess_amt"] / 100, 18.0)
        g0, g1 = 10 ** (self.de_gr / 20), 10 ** (want / 20)
        self.de_gr = want
        g = np.linspace(g0, g1, len(y))
        return y - hi * (1 - g)

    def _comp(self, y):
        """Compresseur RMS, genou doux de 6 dB, attaque / relâchement en ms, par tranches de 1 ms."""
        p = self.p
        thr, ratio = p["comp_thr"], max(1.0, p["comp_ratio"] / 10)
        at = math.exp(-1 / max(0.1, p["comp_att"]))           # par ms
        rt = math.exp(-1 / max(1.0, p["comp_rel"]))
        n = 48
        seg = y.reshape(-1, n)
        lv = 10 * np.log10(np.mean(seg ** 2, axis=1) + 1e-12) + 3       # ≈ crête d'une sinusoïde
        gains = np.empty(len(lv))
        env, knee = self.env, 6.0
        for k, l in enumerate(lv):
            c = at if l > env else rt
            env = c * env + (1 - c) * l
            over = env - thr
            if over <= -knee / 2:
                red = 0.0
            elif over >= knee / 2:
                red = over * (1 - 1 / ratio)
            else:
                red = (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)
            gains[k] = -red
        self.env = env
        # Gain interpolé d'une tranche à l'autre (pas de marches audibles).
        prev = getattr(self, "_last_g", gains[0])
        self._last_g = gains[-1]
        curve = np.interp(np.arange(len(y)), np.concatenate([[-1], np.arange(len(lv)) * n + n - 1]),
                          np.concatenate([[prev], gains]))
        self.gr = float(gains[-1])
        return y * 10 ** ((curve + p["comp_makeup"]) / 20)
