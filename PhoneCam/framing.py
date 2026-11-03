"""Cadrage de la sortie : zoom numérique, position et suivi du visage (comme le « cadrage
automatique » des webcams récentes).

Appliqué avant les effets webcam : on recadre l'image nette, puis les effets compressent
le résultat, exactement comme une vraie caméra qui zoome avant d'encoder.
"""
import numpy as np

DEFAULTS = {
    "zoom": 100,          # %, 100 = image entière
    "pan_x": 0,           # position horizontale, -100 (gauche) … 100 (droite)
    "pan_y": 0,           # position verticale, -100 (haut) … 100 (bas)
    "follow": False,      # garder le visage dans le cadre
    "follow_smooth": 70,  # douceur des mouvements du cadre, %
}


class Framing:
    def __init__(self):
        self.p = dict(DEFAULTS)
        self._c = None    # centre courant du cadre (pixels), lissé

    def active(self) -> bool:
        return self.p["zoom"] > 100

    def wants_faces(self) -> bool:
        return self.active() and self.p["follow"]

    def apply(self, frame: np.ndarray, faces) -> np.ndarray:
        p = self.p
        if not self.active():
            self._c = None
            return frame
        h, w = frame.shape[:2]
        z = p["zoom"] / 100
        cw, ch = w / z, h / z
        target = np.array([w / 2, h / 2], np.float64)
        following = p["follow"] and faces
        if following:
            boxes = np.array([f.bbox[:4] for f in faces], np.float64)
            x0, y0 = boxes[:, :2].min(0)
            x1, y1 = boxes[:, 2:].max(0)
            # Visage un peu au-dessus du centre, comme un cadrage de visio.
            target = np.array([(x0 + x1) / 2, (y0 + y1) / 2 + ch * 0.08])
        target += np.array([p["pan_x"] / 100 * (w - cw) / 2, p["pan_y"] / 100 * (h - ch) / 2])
        if self._c is None:
            self._c = target
        else:
            d = target - self._c
            # Petite zone morte en suivi : le cadre ne tremble pas avec la détection.
            if not following or np.hypot(*d) > 0.03 * cw:
                a = 1 - p["follow_smooth"] / 100 * 0.95
                self._c = self._c + a * d
        cx = float(np.clip(self._c[0], cw / 2, w - cw / 2))
        cy = float(np.clip(self._c[1], ch / 2, h - ch / 2))
        x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
        return frame[y0:y0 + int(round(ch)), x0:x0 + int(round(cw))]
