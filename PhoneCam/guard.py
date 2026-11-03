"""Garde anti-fuite : ne laisse jamais sortir une image où le visage d'origine apparaîtrait.

Quand la détection rate (visage tourné, caché, flou de mouvement…), Deep-Live-Cam renvoie
l'image telle quelle : le vrai visage serait visible. La garde retient quelques images
(anticipation) : une image ne sort que si elle ET les suivantes sont bien remplacées.
Les images juste avant une perte, souvent déjà ratées (swap partiel, visage de profil),
sont donc remplacées elles aussi par la dernière image sûre : la caméra se fige avant.
Au retour du visage, il faut le même nombre d'images correctes avant de repartir.

De profil, la confiance du détecteur oscille autour du seuil d'une image à l'autre : jugée image
par image, l'image se figeait par à-coups de 3 ou 4 images (des « micro-sauts »). La confiance
est donc lissée sur quelques images, avec un petit écart entre le seuil de gel et celui de reprise ;
un visage perdu fige toujours aussitôt. À la reprise, l'image figée se fond dans l'image vivante
en 3 images au lieu de sauter.
"""
import time
from collections import deque

import numpy as np

DEFAULTS = {
    "enabled": True,
    "lookahead": 2,      # images retenues pour anticiper (≈ 33 ms chacune à 30 img/s)
    "min_score": 55,     # confiance de détection minimale, % (en dessous : image jugée risquée)
    "max_freeze": 3000,  # durée du gel avant d'appliquer « after », ms
    "after": 0,          # ensuite : 0 rester figé, 1 image noire, 2 laisser passer
    "steady": 1,         # confiance lissée + reprise en fondu (1) ou jugement image par image (0)
}
FADE = 3                 # images de fondu à la reprise
SMOOTH = 0.4             # lissage de la confiance (part de la nouvelle image)
MARGIN = 0.04            # écart entre seuil de gel et seuil de reprise
AFTER_CHOICES = ["Rester figé", "Image noire", "Laisser passer"]


class FaceGuard:
    def __init__(self):
        self.p = dict(DEFAULTS)
        self.frozen = False
        self.frozen_total = 0          # images remplacées depuis le lancement
        self._buf: deque = deque()     # (image, visages, t_recv, sûre)
        self._last = None              # dernière image sûre sortie : (image, visages)
        self._frozen_since = None
        self._expected = 0             # nombre de visages remplacés dans les dernières images sûres
        self._score = None             # confiance lissée
        self._bad = False              # dernier jugement (pour l'écart gel / reprise)
        self._fade = 0                 # images de fondu restantes après un gel
        self._fade_from = None

    def _judge(self, info) -> bool:
        expected, scores = info
        if not expected:               # face swap coupé ou sans source : rien à protéger
            self._expected = 0
            return True
        thr = self.p["min_score"] / 100
        if self.p.get("steady", 1) and scores:
            s = min(scores)
            self._score = s if self._score is None else self._score + SMOOTH * (s - self._score)
            ok = self._score >= thr + (MARGIN if self._bad else 0) and len(scores) >= self._expected
        else:
            self._score = None
            ok = (bool(scores) and min(scores) >= thr
                  and len(scores) >= self._expected)  # un visage a disparu → l'autre serait visible
        self._bad = not ok
        if ok:
            self._expected = len(scores)
        elif self._frozen_since is not None and \
                (time.perf_counter() - self._frozen_since) * 1000 > self.p["max_freeze"]:
            # Gel trop long : on accepte le nouveau nombre de visages (quelqu'un est parti).
            self._expected = min(self._expected, len(scores))
        return ok

    def push(self, frame: np.ndarray, faces, t_recv: float, info) -> list:
        """info = (swap attendu, scores de détection des visages remplacés).
        Renvoie les images à envoyer à l'étage suivant : [(image, visages, t_recv)]."""
        if not self.p["enabled"]:
            self._buf.clear()
            self.frozen, self._frozen_since = False, None
            return [(frame, faces, t_recv)]
        self._buf.append((frame, faces, t_recv, self._judge(info)))
        out = []
        while len(self._buf) > max(0, int(self.p["lookahead"])):
            img, fcs, t, ok = self._buf.popleft()
            if ok and all(b[3] for b in self._buf):  # sûre, et les suivantes aussi
                if self.frozen and self.p.get("steady", 1) and self._last is not None:
                    self._fade, self._fade_from = FADE, self._last[0]
                self._last = (img, fcs)
                self.frozen, self._frozen_since = False, None
                if self._fade and self._fade_from is not None and self._fade_from.shape == img.shape:
                    # Reprise après un gel : l'image figée s'efface en quelques images.
                    w = self._fade / (FADE + 1)
                    img = (img.astype(np.float32) * (1 - w) + self._fade_from.astype(np.float32) * w).astype(np.uint8)
                    self._fade -= 1
                out.append((img, fcs, t))
                continue
            self._fade = 0
            now = time.perf_counter()
            if self._frozen_since is None:
                self._frozen_since = now
            self.frozen = True
            self.frozen_total += 1
            after = self.p["after"] if (now - self._frozen_since) * 1000 > self.p["max_freeze"] else 0
            if after == 2:
                out.append((img, fcs, t))
            elif after == 1 or self._last is None:  # pas encore d'image sûre : noir plutôt que fuite
                out.append((np.zeros_like(img), [], t))
            else:
                out.append((self._last[0], self._last[1], t))
        return out
