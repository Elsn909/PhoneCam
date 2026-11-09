"""Entrée du micro : sensibilité, niveau automatique et voix à distance.

Tout premier traitement, sur le son brut du micro, AVANT le changeur de voix et les effets : la
détection de voix (Smart), la hauteur et le modèle reçoivent une voix au bon niveau et plus nette,
même quand tu parles à quelques dizaines de centimètres du micro.
  · Sensibilité : gain fixe (le micro « entend » plus fort).
  · Niveau automatique : ta voix est ramenée à son niveau « près du micro ». Le gain ne bouge que
    sur une vraie voix (son périodique : une hauteur entre 80 et 600 Hz), jamais sur le clavier, les
    souffles, les frottements ou le bruit de fond ; sans à-coup (les nuances et l'intonation restent).
    Après 8 s sans voix, il revient doucement à zéro (le bruit de fond ne reste pas remonté).
  · Moins d'écho (coupé par défaut) : de loin, la pièce se mêle à la voix et les consonnes s'empâtent. La réverbération
    tardive est prédite à partir du son d'il y a 50 ms (elle décroît de façon exponentielle) et
    retirée bande par bande (100 Hz), avec un plancher : pas de son « métallique ».
  · Voix proche (coupé par défaut) : de loin, une voix perd ses graves (effet de proximité du micro) ; ils sont rendus
    selon la remontée en cours, donc rien quand tu es près.
  · Protection : jamais de saturation.
Mesuré sur des prises éloignées (simulées à 25 et 45 cm) passées dans un modèle de voix : le niveau automatique
fait presque tout (parole coupée par Smart : 31 % → 0–2 %, sortie bien plus proche de la prise faite
près du micro). Moins d'écho et Voix proche n'aident pas le modèle (un peu moins ressemblant) : ils
servent surtout sans changeur de voix, à l'oreille.
Blocs de 10 ms à 48 kHz. Moins d'écho : trames de 10 ms toutes les 5 ms, 5 ms de latence en plus
(seulement quand il est en marche). Coût : moins de 0,1 ms par bloc.
"""
import math

import numpy as np
from scipy import signal

from effets_micro import biquad

SR = 48000
BLOCK = 480
N, H = 480, 240                         # trame et pas de l'analyse (10 ms, 5 ms)
T60 = 0.5                               # réverbération d'une chambre / d'un salon (s)
TD = 10                                 # réverbération tardive : au-delà de 10 pas (50 ms)

DEFAULTS = {
    "input_gain": 0,        # sensibilité du micro, dB
    "auto_level": True,     # niveau automatique : je peux m'éloigner du micro
    "auto_target": -26,     # niveau de voix visé (dB, comme près du micro)
    "auto_max": 18,         # remontée maximale, dB
    "dereverb": 0,          # % : moins d'écho de la pièce
    "proximity": 0,         # % : graves rendus quand tu t'éloignes
}


class MicInput:
    def __init__(self, p):
        self.p = p
        self.win = np.sqrt(np.hanning(N + 1)[:N])        # sqrt-Hann périodique : reconstruction parfaite
        self.buf = np.zeros(N)
        self.ola = np.zeros(N - H)
        nb = N // 2 + 1
        self.hist = np.zeros((TD, nb))                     # puissance lissée des 10 derniers pas
        self.hpos = 0
        self.Ps = np.zeros(nb)
        self.G = np.ones(nb)
        self.decay = math.exp(-2 * (3 * math.log(10) / T60) * TD * H / SR)
        self.floor = None           # bruit de fond (dB)
        self.level = None           # niveau de ta voix (dB, trames de parole)
        self.run = 0
        self.gain = 0.0             # remontée automatique en cours (dB)
        self._g_prev = 0.0
        self.lim = 1.0
        self._prox = None
        self.prox_sos = np.zeros((0, 6))
        self.prox_zi = np.zeros((0, 2))
        self.raw_db = self.in_db = -120.0
        self.speech = False
        self.hist_x = np.zeros(1440)                       # 30 ms : de quoi voir une période de 80 Hz
        self.quiet = 0                                     # blocs sans voix

    def reset(self):
        self.__init__(self.p)

    # --------------------------------------------------------------------- niveau automatique
    def _voiced(self, x):
        """Son périodique (une voix qui a une hauteur) ? Autocorrélation normalisée, 80–600 Hz."""
        self.hist_x[:-len(x)] = self.hist_x[len(x):]
        self.hist_x[-len(x):] = x
        h = self.hist_x - self.hist_x.mean()
        n = len(h)
        F = np.fft.rfft(h, 4096)
        r = np.fft.irfft(F * np.conj(F), 4096)[:601]
        if r[0] <= 1e-12:
            return False
        lags = np.arange(80, 601)
        rn = r[lags] / (r[0] * (n - lags) / n)
        return float(rn.max()) > 0.55

    def _auto(self, x):
        p = self.p
        lv = 10 * math.log10(float(np.mean(x * x)) + 1e-14)
        if self.floor is None:
            self.floor = lv
        # bruit de fond : suit tout de suite ce qui est plus bas, remonte de 2 dB/s au plus
        self.floor = lv if lv < self.floor else self.floor + min(0.02, (lv - self.floor) * 0.01)
        crest = 20 * math.log10(float(np.abs(x).max()) + 1e-9) - lv
        speech = lv > self.floor + 12 and lv > -80 and crest < 15 and self._voiced(x)
        self.run = self.run + 1 if speech else 0
        self.speech = self.run >= 3                             # 30 ms de voix d'affilée
        if not p["auto_level"]:
            self.gain += (0.0 - self.gain) * 0.05
            return
        if not self.speech:
            self.quiet += 1
            if self.quiet > 800 and self.gain > 0:              # 8 s sans voix : retour doux à zéro
                self.gain = max(0.0, self.gain - 0.005)
            return                                              # sinon le gain ne bouge pas
        self.quiet = 0
        cur = lv + self.gain                                    # niveau après remontée
        self.level = cur if self.level is None else self.level + (cur - self.level) * 0.02
        err = p["auto_target"] - self.level
        step = min(err * 0.03, 0.1) if err > 0 else max(err * 0.08, -0.4)      # ≤ 10 dB/s ↑, ≤ 40 dB/s ↓
        self.gain = float(np.clip(self.gain + step, -6.0, p["auto_max"]))
        self.level += step                                      # le niveau suivi tient compte du nouveau gain

    # --------------------------------------------------------------------- moins d'écho
    def _dereverb(self, X, amt):
        P = X.real ** 2 + X.imag ** 2
        self.Ps = 0.6 * self.Ps + 0.4 * P
        late = self.decay * self.hist[self.hpos]                # prédiction de la queue de réverbération
        self.hist[self.hpos] = self.Ps
        self.hpos = (self.hpos + 1) % TD
        if amt <= 0:
            self.G[:] = 1.0
            return X
        gmin = 10 ** (-(4 + 10 * amt) / 20)
        g = np.sqrt(np.clip(1 - 1.5 * amt * late / (self.Ps + 1e-12), gmin * gmin, 1.0))
        g = np.convolve(g, (0.25, 0.5, 0.25), "same")           # voisins lissés : pas de « bulles »
        self.G = np.maximum(g, self.G * 0.75)                   # retombée douce
        return X * self.G

    def _stft(self, x, amt):
        out = np.empty(len(x))
        for j in range(0, len(x), H):
            self.buf[:-H] = self.buf[H:]
            self.buf[-H:] = x[j:j + H]
            X = self._dereverb(np.fft.rfft(self.buf * self.win), amt)
            y = np.fft.irfft(X, N) * self.win
            out[j:j + H] = self.ola + y[:H]
            self.ola = y[H:]                                    # recouvrement de 50 % (N = 2 H)
        return out

    # --------------------------------------------------------------------- voix proche
    def _proximity(self, y):
        g = round(self.p["proximity"] / 100 * 6.0 * min(1.0, max(0.0, self.gain) / 18), 1)
        if g != self._prox:
            self._prox = g
            n_old = self.prox_sos.shape[0]
            self.prox_sos = biquad("low", 200, g, 0.71)[None] if g > 0 else np.zeros((0, 6))
            if self.prox_sos.shape[0] != n_old:
                self.prox_zi = np.zeros((self.prox_sos.shape[0], 2))
        if len(self.prox_sos):
            y, self.prox_zi = signal.sosfilt(self.prox_sos, y, zi=self.prox_zi)
        return y

    def process(self, x):
        p = self.p
        x = x.astype(np.float64)
        self.raw_db = 20 * math.log10(float(np.abs(x).max()) + 1e-9)
        x = x * 10 ** (p["input_gain"] / 20)
        self._auto(x)
        g = np.linspace(10 ** (self._g_prev / 20), 10 ** (self.gain / 20), len(x))
        self._g_prev = self.gain
        y = x * g
        if p["dereverb"] > 0:
            y = self._stft(y, p["dereverb"] / 100)
        elif self.buf.any():                # coupé : l'analyse repart de zéro la prochaine fois
            self.buf[:] = 0
            self.ola[:] = 0
            self.hist[:] = 0
            self.Ps[:] = 0
            self.G[:] = 1
        y = self._proximity(y)
        pk = float(np.abs(y).max())
        target = min(1.0, 0.89 / pk) if pk > 0 else 1.0
        new = target if target < self.lim else self.lim + (target - self.lim) * 0.05
        y = 0.98 * np.tanh(y * np.linspace(self.lim, new, len(y)) / 0.98)
        self.lim = new
        self.in_db = 20 * math.log10(float(np.abs(y).max()) + 1e-9)
        return y.astype(np.float32)
