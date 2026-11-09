"""Voix vivante : quand tu parles bas (ton plus grave et plus plat que d'habitude), la voix est
remontée et rendue plus expressive, pour sonner claire et enjouée plutôt qu'éteinte.

  · Ta hauteur habituelle est apprise en continu (et gardée d'une session à l'autre) : elle monte en
    une dizaine de secondes quand tu parles normalement, et ne descend que très lentement, pour qu'une
    longue phrase à voix basse ne devienne pas « l'habitude ».
  · Si les 400 dernières ms sont plus basses que ce repère moins le seuil, la hauteur est relevée
    d'une partie de l'écart, en bloc (ton intonation est gardée), avec des transitions douces.
  · Expressivité : la mélodie de la voix est un peu élargie autour de sa moyenne (moins monotone).
  · Clarté : présence et air ajoutés à la sortie, seulement pendant que la voix est relevée.
"""
import math

import numpy as np
from scipy import signal


class Lively:
    def __init__(self):
        self.reset()
        self._key = None
        self.sos = None
        self.zi = None

    def reset(self):
        self.ref = None              # hauteur habituelle (demi-tons, référence 1 Hz)
        self.shift = 0.0             # relèvement actuel (demi-tons)
        self.prev_shift = 0.0
        self.low = False
        self.cur_hz = 0.0            # ta hauteur actuelle (affichée pour régler à l'oreille)
        self.measure = None          # « Mesurer ma voix normale » : hauteurs recueillies

    def pitch(self, f0, p, emit_lo, emit_hi, n_new):
        """f0 (Hz, 0 = sans hauteur) sur toute la fenêtre → f0 transformé. emit_lo/hi : trames jouées
        par ce bloc ; n_new : trames nouvelles (fin de fenêtre) pour apprendre la hauteur habituelle."""
        v = f0 > 0
        if not v.any():
            self.prev_shift = self.shift
            return f0
        st = np.where(v, 12 * np.log2(np.maximum(f0, 1.0)), 0.0)
        new = st[-n_new:][v[-n_new:]]
        if len(new) >= 5:
            self.cur_hz = float(2 ** (np.median(new) / 12))
        if self.measure is not None and len(new):   # « Mesurer » : ~5 s de voix, puis la médiane
            self.measure.extend(new.tolist())
            if len(self.measure) >= 300:            # ~3 s de voix réelle
                p["lift_ref"] = int(round(2 ** (float(np.median(self.measure)) / 12)))
                self.measure = None
        if p.get("lift_auto") and len(new) >= 10:   # option : apprendre toute seule, en continu
            m = float(np.median(new))
            ref = 12 * math.log2(max(60, p["lift_ref"]))
            m = min(max(m, ref - 4), ref + 4)       # monte en ~10 s, descend en ~1 min
            ref += (m - ref) * (0.04 if m > ref else 0.0035)
            p["lift_ref"] = round(2 ** (ref / 12), 1)
        self.ref = 12 * math.log2(max(60, p["lift_ref"]))
        # ton des 400 dernières ms jouées (ou juste avant)
        recent = st[max(0, emit_hi - 40):emit_hi][v[max(0, emit_hi - 40):emit_hi]]
        target = self.shift
        if len(recent) >= 6:
            low = (self.ref - p["lift_floor"] / 10) - float(np.median(recent))
            self.low = low > 0
            target = max(0.0, low) * p["lift_amount"] / 100
            target = min(target, 4.0)
        self.prev_shift, self.shift = self.shift, self.shift + (target - self.shift) * 0.5
        # relèvement en rampe sur les trames jouées (pas de saut entre deux blocs)
        ramp = np.full(len(f0), self.shift)
        ramp[:emit_lo] = self.prev_shift
        k = max(1, emit_hi - emit_lo)
        ramp[emit_lo:emit_hi] = np.linspace(self.prev_shift, self.shift, k)
        # expressivité : écarts à la moyenne de la fenêtre élargis
        c = float(np.mean(st[v]))
        out = st + ramp
        if p["lively"] > 0:
            w = 1.0 if p["lively_always"] else 0.4 + 0.6 * min(1.0, self.shift / 1.0)   # pleine quand le ton est relevé
            out = c + ramp + (st - c) * (1 + p["lively"] / 100 * w)
        return np.where(v, 2.0 ** (out / 12), 0.0)

    def brighten(self, y, p, sr=48000):
        """Présence (≈ 3,5 kHz) et air (≈ 10 kHz) en proportion du relèvement."""
        amt = p["lift_bright"] / 100 * min(1.0, self.shift / 1.5)
        key = round(amt, 2)
        if key != self._key:
            self._key = key
            g1, g2 = 4.0 * amt, 3.0 * amt
            sos = np.vstack([_peak(3500, g1, 0.8, sr), _shelf(10000, g2, sr)])
            if self.zi is None:
                self.zi = np.zeros((2, 2))
            self.sos = sos
        if key == 0 and abs(self.zi).max() < 1e-9:
            return y
        out, self.zi = signal.sosfilt(self.sos, y, zi=self.zi)
        return out.astype(np.float32)


def _peak(f, g, q, sr):
    A, w = 10 ** (g / 40), 2 * math.pi * f / sr
    al = math.sin(w) / (2 * q)
    b = [1 + al * A, -2 * math.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * math.cos(w), 1 - al / A]
    return np.array(b + a) / a[0]


def _shelf(f, g, sr):
    A, w = 10 ** (g / 40), 2 * math.pi * f / sr
    cw, al = math.cos(w), math.sin(w) / 2 * math.sqrt(2)
    sa = 2 * math.sqrt(A) * al
    b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
    a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return np.array(b + a) / a[0]
