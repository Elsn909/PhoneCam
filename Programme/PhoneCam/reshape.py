"""Filtres du visage (façon Instagram / TikTok) : forme (déformation locale guidée par les
repères) et maquillage (couleur des lèvres, blush, peau, dents).

Le champ de déplacement n'est calculé que sur une grille grossière autour de chaque visage,
puis ré-agrandi : quelques millisecondes, même en 1080p. Il s'annule au bord de la zone,
donc la déformation se raccorde au reste de l'image sans couture.
Le maquillage passe avant la déformation : il suit ensuite le visage déformé.
Avec le détourage (segment.py), seule la personne est déformée : le décor reste droit.
Avec les repères 3D (head3d.py), l'échelle ne dépend plus de l'angle de la tête, et chaque côté
du visage reçoit un effet à sa mesure (la joue tournée vers la caméra paraît plus large).
Le contour est déplacé par un maillage souple (« thin-plate spline ») qui le pose exactement où
il faut, tandis que les yeux, sourcils, nez et bouche y sont tenus immobiles.
"""
import time

import cv2
import numpy as np

DEFAULTS = {  # forme : -100…100, maquillage : 0…100 ; 0 = désactivé
    # visage
    "face_size": 0,     # taille du visage entier
    "slim": 0,          # affiner les joues (négatif : visage plus rond)
    "cheekbones": 0,    # pommettes plus marquées (négatif : plus discrètes)
    "jaw": 0,           # mâchoire en V
    "chin": 0,          # longueur du menton
    "forehead": 0,      # hauteur du front
    "nose": 0,          # taille du nez
    "nose_width": 0,    # nez plus fin (négatif : plus large)
    "nose_lift": 0,     # bout du nez relevé (négatif : tombant)
    # yeux
    "eyes": 0,          # taille des yeux
    "eye_distance": 0,  # écart entre les yeux
    "eye_tilt": 0,      # coins extérieurs relevés (« yeux de chat »)
    "brow_height": 0,   # sourcils plus hauts (négatif : plus bas)
    "brow_tilt": 0,     # queue des sourcils relevée (négatif : air triste)
    # bouche
    "mouth": 0,         # largeur de la bouche
    "lips": 0,          # volume des lèvres
    "smile": 0,         # coins de la bouche relevés
    "philtrum": 0,      # bouche rapprochée du nez (négatif : éloignée)
    # corps
    "body": 0,          # affiner le cou et le buste (négatif : plus large)
    # maquillage et peau
    "skin_smooth": 0,   # lissage de la peau
    "glow": 0,          # éclat du teint
    "blush": 0,         # blush sur les joues
    "lip_color": 0,     # intensité du rouge à lèvres
    "lip_shade": 0,     # teinte (index dans LIP_SHADES)
    "teeth": 0,         # blanchir les dents
    "under_eye": 0,     # atténuer les cernes
    "tan": 0,           # teint hâlé
    "freckles": 0,      # taches de rousseur
    "contour": 0,       # ombre sous les pommettes
    "highlight": 0,     # lumière sur le nez et le haut des joues
    "eyeshadow": 0,     # fard à paupières
    "eyeshadow_shade": 0,  # teinte (index dans EYE_SHADES)
    "eyeliner": 0,      # trait de liner
    "brow_color": 0,    # sourcils plus marqués
    "iris": 0,          # couleur des yeux
    "iris_shade": 0,    # teinte (index dans IRIS_SHADES)
    "eye_bright": 0,    # regard plus lumineux
    "smooth": 60,       # stabilisation des repères, % (évite que les filtres tremblent)
    "protect_bg": 1,    # ne déformer que la personne (le décor reste droit)
    "hrffa": 0,         # repères HRFFA (FaceFusion, tiennent mieux de profil) au lieu d'InsightFace
}
SHAPE_KEYS = ["face_size", "slim", "cheekbones", "jaw", "chin", "forehead", "nose", "nose_width", "nose_lift",
              "eyes", "eye_distance", "eye_tilt", "brow_height", "brow_tilt", "mouth", "lips", "smile", "philtrum", "body"]
MAKEUP_KEYS = ["skin_smooth", "glow", "blush", "lip_color", "teeth", "under_eye", "tan", "freckles", "contour",
               "highlight", "eyeshadow", "eyeliner", "brow_color", "iris", "eye_bright"]
SHADE_KEYS = ["lip_shade", "eyeshadow_shade", "iris_shade"]
SHAPE_GROUPS = [  # titre, [(clé, libellé, minimum)]  ; forme : −100…100, maquillage : 0…100
    ("Visage", [("face_size", "Taille du visage", -100), ("slim", "Affiner les joues", -100),
                ("cheekbones", "Pommettes", -100), ("jaw", "Mâchoire en V", -100),
                ("chin", "Menton (longueur)", -100), ("forehead", "Front (hauteur)", -100),
                ("nose", "Taille du nez", -100), ("nose_width", "Affiner le nez", -100),
                ("nose_lift", "Bout du nez relevé", -100)]),
    ("Yeux", [("eyes", "Taille des yeux", -100), ("eye_distance", "Écart des yeux", -100),
              ("eye_tilt", "Yeux de chat (coins relevés)", -100), ("brow_height", "Hauteur des sourcils", -100),
              ("brow_tilt", "Sourcils relevés", -100)]),
    ("Bouche", [("mouth", "Largeur de la bouche", -100), ("lips", "Volume des lèvres", -100),
                ("smile", "Sourire", -100), ("philtrum", "Bouche plus proche du nez", -100)]),
    ("Corps", [("body", "Affiner le cou et le buste", -100)]),
    ("Maquillage et peau", [("skin_smooth", "Lissage de la peau", 0), ("glow", "Éclat du teint", 0),
                            ("under_eye", "Atténuer les cernes", 0), ("tan", "Teint hâlé", 0),
                            ("freckles", "Taches de rousseur", 0), ("contour", "Contouring", 0),
                            ("highlight", "Highlighter", 0), ("blush", "Blush", 0),
                            ("lip_color", "Rouge à lèvres", 0), ("teeth", "Dents plus blanches", 0)]),
    ("Regard", [("eye_bright", "Regard lumineux", 0), ("iris", "Couleur des yeux", 0),
                ("eyeshadow", "Fard à paupières", 0), ("eyeliner", "Eyeliner", 0),
                ("brow_color", "Sourcils plus marqués", 0)]),
]
SHAPE_PRESETS = {
    "Aucun": {},
    "Naturel": {"slim": 25, "jaw": 15, "eyes": 15, "nose": -15, "skin_smooth": 30, "glow": 15},
    "Glamour": {"slim": 45, "jaw": 35, "chin": 15, "eyes": 30, "nose": -30, "nose_width": 30,
                "cheekbones": 25, "lips": 25, "smile": 15, "skin_smooth": 55, "glow": 30,
                "blush": 35, "lip_color": 55, "lip_shade": 0, "teeth": 30},
    "Poupée": {"face_size": -10, "slim": 55, "jaw": 50, "chin": -20, "eyes": 70, "eye_distance": 15,
               "nose": -45, "nose_width": 40, "mouth": -25, "lips": 30, "skin_smooth": 75, "glow": 40,
               "blush": 60, "lip_color": 60, "lip_shade": 1},
    "Yeux de chat": {"slim": 30, "cheekbones": 35, "eyes": 20, "eye_tilt": 70, "nose_width": 25,
                     "skin_smooth": 40, "lip_color": 45, "lip_shade": 6},
    "Silhouette": {"slim": 30, "jaw": 25, "body": 45, "skin_smooth": 25},
    "Caricature": {"face_size": 25, "slim": -60, "chin": 60, "forehead": 50, "eyes": 90, "nose": 80,
                   "mouth": 60, "lips": 50},
    "Bonne mine": {"skin_smooth": 25, "glow": 20, "under_eye": 50, "blush": 15, "tan": 15, "eye_bright": 30},
    "Soirée": {"slim": 20, "skin_smooth": 40, "glow": 15, "contour": 45, "highlight": 40, "blush": 25,
               "eyeshadow": 55, "eyeshadow_shade": 3, "eyeliner": 60, "brow_color": 30, "lip_color": 60, "lip_shade": 6},
    "Été": {"skin_smooth": 20, "tan": 45, "freckles": 40, "glow": 25, "eye_bright": 25, "blush": 15},
    "Regard clair": {"under_eye": 40, "eye_bright": 45, "iris": 55, "iris_shade": 1, "brow_color": 15},
}

LIP_SHADES = [  # nom, couleur BGR
    ("Rouge", (40, 30, 200)),
    ("Rose", (150, 110, 230)),
    ("Corail", (90, 120, 250)),
    ("Nude", (130, 150, 200)),
    ("Fuchsia", (160, 40, 220)),
    ("Prune", (90, 40, 120)),
    ("Bordeaux", (50, 20, 120)),
]
EYE_SHADES = [  # nom, couleur BGR
    ("Brun doré", (60, 110, 170)),
    ("Bronze", (50, 80, 120)),
    ("Rose", (150, 120, 220)),
    ("Prune", (110, 50, 110)),
    ("Bleu nuit", (120, 60, 40)),
    ("Émeraude", (90, 130, 30)),
]
IRIS_SHADES = [  # nom, couleur BGR
    ("Noisette", (50, 110, 150)),
    ("Bleu", (200, 130, 60)),
    ("Vert", (110, 150, 70)),
    ("Gris", (150, 140, 130)),
    ("Ambre", (40, 140, 210)),
]
GRID = 6             # pas de la grille de calcul, px
_FRECKLES = np.random.default_rng(7).uniform(0, 1, (110, 4)).astype(np.float32)   # toujours les mêmes taches


def _resample(pts, n, closed):
    """n points répartis le long d'une suite de points (fermée ou non), tous distincts."""
    m = len(pts)
    t = np.arange(n) * m / n if closed else np.linspace(0, m - 1, n)
    i = np.floor(t).astype(int)
    f = (t - i)[:, None]
    return pts[i % m] * (1 - f) + pts[(i + 1) % m if closed else np.minimum(i + 1, m - 1)] * f


def layout_106(p68):
    """68 repères (iBUG) rangés comme les 106 d'InsightFace, là où les filtres s'en servent :
    contour 0–32, œil 33–42, sourcil 43–51, bouche 52–71 (les mêmes 20 points), nez 72–86,
    œil 87–96, sourcil 97–105."""
    jaw = p68[0:17]
    contour = np.empty((33, 2), np.float32)
    contour[0::2], contour[1::2] = jaw, (jaw[:-1] + jaw[1:]) / 2
    return np.vstack([contour, _resample(p68[36:42], 10, True), _resample(p68[17:22], 9, False),
                      p68[48:68], _resample(p68[27:36], 15, False), _resample(p68[42:48], 10, True),
                      _resample(p68[22:27], 9, False)]).astype(np.float32)


def _falloff(d2: np.ndarray, r: float) -> np.ndarray:
    """Poids lisse 1 → 0 entre le centre et le rayon r (dérivée nulle au bord)."""
    w = np.clip(1 - d2 / (r * r), 0, 1)
    return w * w


def _remap_roi(img, D, x0, y0, x1, y1, grid, person=None):
    """Applique un champ de déplacement grossier D (src = p + D) à la zone [x0:x1, y0:y1].
    person : silhouette (0…1, à n'importe quelle échelle) ; seule la personne est alors déplacée."""
    gh, gw = D.shape[:2]
    rw, rh = x1 - x0, y1 - y0
    if person is not None:
        # Un point de la grille ne bouge que s'il est sur la personne, ou s'il va la montrer
        # (visage agrandi) : le décor reste droit, sans collage ni reconstitution.
        k = person.shape[1] / img.shape[1]
        px = (x0 + np.arange(gw, dtype=np.float32) * grid)[None, :].repeat(gh, 0)
        py = (y0 + np.arange(gh, dtype=np.float32) * grid)[:, None].repeat(gw, 1)
        here = cv2.remap(person, px * k, py * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        there = cv2.remap(person, (px + D[..., 0]) * k, (py + D[..., 1]) * k, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        D = D * np.maximum(here, there)[..., None]
    dx = cv2.resize(D[..., 0], (gw * grid, gh * grid), interpolation=cv2.INTER_LINEAR)[:rh, :rw]
    dy = cv2.resize(D[..., 1], (gw * grid, gh * grid), interpolation=cv2.INTER_LINEAR)[:rh, :rw]
    map_x = dx + np.arange(x0, x1, dtype=np.float32)[None, :]
    map_y = dy + np.arange(y0, y1, dtype=np.float32)[:, None]
    img[y0:y1, x0:x1] = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _tps(ctrl, disp, P, scale, reg=1e-3):
    """Champ lisse (thin-plate spline) qui vaut disp aux points ctrl, évalué aux points P."""
    def kern(a, b):
        r2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        r2 = np.maximum(r2, 1e-9)
        return r2 * np.log(r2)
    c = (ctrl / scale).astype(np.float64)
    n = len(c)
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = kern(c, c) + reg * np.eye(n)
    L[:n, n] = 1
    L[:n, n + 1:] = c
    L[n, :n] = 1
    L[n + 1:, :n] = c.T
    rhs = np.zeros((n + 3, 2))
    rhs[:n] = disp
    coef = np.linalg.solve(L, rhs).astype(np.float32)
    q = P.reshape(-1, 2) / np.float32(scale)
    out = kern(q, c.astype(np.float32)) @ coef[:n] + coef[n] + q @ coef[n + 1:]
    return out.reshape(P.shape).astype(np.float32)


def _grid(x0, y0, x1, y1, grid):
    gw, gh = (x1 - x0) // grid + 2, (y1 - y0) // grid + 2
    gx, gy = np.meshgrid(x0 + np.arange(gw, dtype=np.float32) * grid,
                         y0 + np.arange(gh, dtype=np.float32) * grid)
    return np.stack([gx, gy], -1)


class FaceReshape:
    def __init__(self):
        self.p = dict(DEFAULTS)
        self.ms = 0.0
        self._tracks: list[np.ndarray] = []   # repères lissés de l'image précédente
        self.segmenter = None                 # segment.PersonSegmenter, fourni par l'appelant
        self.head3d = None                    # head3d.Head3D, fourni par l'appelant
        self.hrffa = None                     # landmarks.Hrffa, fourni par l'appelant
        self._person = None

    def active(self) -> bool:
        return any(self.p[k] != 0 for k in SHAPE_KEYS + MAKEUP_KEYS)

    def uses_hrffa(self) -> bool:
        """Repères HRFFA prêts et choisis : les 106 repères d'InsightFace sont alors inutiles."""
        return bool(self.p["hrffa"]) and self.hrffa is not None and self.hrffa.ready()

    def apply(self, frame: np.ndarray, faces) -> np.ndarray:
        t0 = time.perf_counter()
        if not self.active():
            self._tracks = []
            self.ms = 0.0
            return frame
        out = frame
        tracks = []
        self._person = None
        if (faces and self.p["protect_bg"] and self.segmenter is not None
                and any(self.p[k] for k in SHAPE_KEYS)):
            self._person = self.segmenter.mask(frame)
        if faces and self.head3d is not None and any(self.p[k] for k in SHAPE_KEYS):
            self.head3d.add(frame, faces)
        use_hrffa = bool(self.p["hrffa"]) and self.hrffa is not None
        if faces and use_hrffa:
            self.hrffa.add(frame, faces)
        for face in faces or []:
            pts = self._points(face, use_hrffa)
            if pts is None:
                continue
            pts = self._smooth(pts)
            tracks.append(pts)
            if out is frame:
                out = frame.copy()  # l'image d'entrée peut appartenir à un autre étage
            if any(self.p[k] for k in MAKEUP_KEYS):
                self._makeup(out, pts)
            self._warp(out, pts)
            if self.p["body"]:
                self._body(out, pts)
        self._tracks = tracks
        self.ms = (time.perf_counter() - t0) * 1000
        return out

    @staticmethod
    def _points(face, use_hrffa=False):
        """5 points clés (yeux, nez, coins de la bouche), les 106 repères, puis si disponibles les
        68 repères 3D (x, y) et leur profondeur (z, 0) : tout est lissé ensemble.
        Avec HRFFA : ses 68 repères donnent les positions (5 points, 106 rangés pareil, 68), le
        modèle 3D d'InsightFace ne donne plus que la profondeur."""
        kps = getattr(face, "kps", None)
        lmk = getattr(face, "landmark_2d_106", None)
        l3 = face.get("landmark_3d_68") if hasattr(face, "get") else None
        h68 = face.get("landmark_hrffa_68") if use_hrffa and hasattr(face, "get") else None
        if h68 is not None:
            from landmarks import to_5
            z = l3[:, 2] if l3 is not None else np.zeros(68)
            return np.vstack([to_5(h68), layout_106(h68), h68, np.c_[z, np.zeros(68)]]).astype(np.float32)
        if kps is None or lmk is None:
            return None
        rows = [kps[:5], lmk[:106]]
        if l3 is not None:
            rows += [l3[:, :2], np.c_[l3[:, 2], np.zeros(68)]]
        return np.vstack(rows).astype(np.float32)

    def _smooth(self, pts: np.ndarray) -> np.ndarray:
        """Lissage exponentiel, relâché quand le visage bouge vite (pas de traînée)."""
        if not self._tracks:
            return pts
        ed = float(np.linalg.norm(pts[1] - pts[0])) or 1.0
        prev = min(self._tracks, key=lambda t: float(np.linalg.norm(t.mean(0) - pts.mean(0))))
        if prev.shape != pts.shape or np.linalg.norm(prev.mean(0) - pts.mean(0)) > ed:
            return pts  # autre visage
        motion = float(np.abs(pts - prev).mean())
        a = 1 - self.p["smooth"] / 100 * 0.9
        a = min(1.0, a + motion / (0.15 * ed))
        return prev + a * (pts - prev)

    @staticmethod
    def _frame(pts):
        """Repère du visage : yeux, axes, distance entre les yeux (l'échelle de tout le reste)."""
        le, re = pts[0], pts[1]
        ed = float(np.linalg.norm(re - le))
        ux = (re - le) / max(ed, 1e-3)                     # axe horizontal du visage
        uy = np.array([-ux[1], ux[0]], np.float32)         # axe vertical, vers le menton
        return le, re, ed, ux, uy, (le + re) / 2

    @staticmethod
    def _head(pts, ed, ux):
        """Avec les repères 3D : échelle vraie (écart des yeux en 3D, le même quel que soit
        l'angle), ligne médiane du visage (haut du nez → menton) et poids de chaque côté (largeur
        visible de chaque moitié du visage). Sans eux : valeurs d'un visage de face."""
        if len(pts) < 247:
            return ed, None, 1.0, 1.0
        l3, z = pts[111:179], pts[179:247, 0]
        e2 = l3[42:48].mean(0) - l3[36:42].mean(0)
        e3 = np.r_[e2, z[42:48].mean() - z[36:42].mean()]
        ed_true = ed * float(np.clip(np.linalg.norm(e3) / max(np.linalg.norm(e2), 1e-3), 1.0, 2.0))
        a, b = l3[27], l3[8]
        v = (b - a) / max(float(np.linalg.norm(b - a)), 1e-3)
        n = np.array([v[1], -v[0]], np.float32)
        if n @ ux < 0:
            n = -n
        off = (l3[0:17] - a) @ n
        hl, hr = max(float(-off.min()), 1e-3), max(float(off.max()), 1e-3)
        g_l, g_r = (float(np.clip(x / ((hl + hr) / 2), 0.35, 1.65)) for x in (hl, hr))
        return ed_true, (a, n), g_l, g_r

    # ------------------------------------------------------------ forme
    def _warp(self, img: np.ndarray, pts: np.ndarray) -> None:
        p = self.p
        if not any(p[k] for k in SHAPE_KEYS if k != "body"):
            return
        le, re, ed, ux, uy, eye_c = self._frame(pts)
        nose, ml, mr = pts[2], pts[3], pts[4]
        contour = pts[5:38]
        if ed < 8:
            return

        ed3, mid, g_l, g_r = self._head(pts, ed, ux)

        # --- déplacement voulu de chaque point du contour (horizontal : largeur visible de son
        # côté ; vertical : échelle vraie)
        rel = contour - eye_c
        yr = rel @ uy
        xr = (contour - mid[0]) @ mid[1] if mid is not None else rel @ ux
        height = max(float(yr.max()), 0.6 * ed3)
        t = yr / height                        # 0 = niveau des yeux, 1 = menton
        side = np.where(xr >= 0, 1.0, -1.0)[:, None]
        gh = np.where(xr >= 0, g_r, g_l)[:, None] * ed      # échelle horizontale de chaque point
        vec = np.zeros_like(contour)
        if p["slim"]:
            w = np.exp(-((t - 0.55) / 0.3) ** 2)
            vec += -side * ux * gh * (w * 0.13 * p["slim"] / 100)[:, None]
        if p["cheekbones"]:
            w = np.exp(-((t - 0.22) / 0.16) ** 2)
            vec += side * ux * gh * (w * 0.09 * p["cheekbones"] / 100)[:, None]
        if p["jaw"]:
            w = np.clip((t - 0.5) / 0.35, 0, 1) ** 2 * np.clip(np.abs(xr) / (0.25 * ed), 0, 1)
            vec += (-side * ux * gh * 0.14 - uy * ed3 * 0.04) * (w * p["jaw"] / 100)[:, None]
        if p["chin"]:
            w = np.exp(-(xr / (0.45 * ed)) ** 2) * (t > 0.6)
            vec += uy * (w * 0.2 * ed3 * p["chin"] / 100)[:, None]

        # --- zone de calcul : le visage + le front + une marge où tout retombe à zéro
        forehead = eye_c - uy * 1.3 * ed3
        allp = np.vstack([pts[:38], forehead[None]])
        m = (0.6 + 0.5 * (p["face_size"] != 0)) * ed3
        h, w_img = img.shape[:2]
        x0, y0 = (max(0, int(v)) for v in allp.min(0) - m)
        x1, y1 = int(min(w_img, allp[:, 0].max() + m)), int(min(h, allp[:, 1].max() + m))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return
        P = _grid(x0, y0, x1, y1, GRID)
        D = np.zeros_like(P)                   # src = p + D (correspondance inverse)

        if np.any(vec):
            # Maillage souple : chaque point du contour arrive exactement à sa place (contour + vec),
            # yeux, sourcils, nez et bouche ne bougent pas, le bord de la zone non plus.
            gh_, gw_ = P.shape[:2]
            bx = np.linspace(x0, x0 + (gw_ - 1) * GRID, 12)
            by = np.linspace(y0, y0 + (gh_ - 1) * GRID, 12)
            border = np.vstack([np.c_[bx, np.full(12, by[0])], np.c_[bx, np.full(12, by[-1])],
                                np.c_[np.full(10, bx[0]), by[1:-1]], np.c_[np.full(10, bx[-1]), by[1:-1]]])
            features = pts[5 + 33:5 + 106]
            ctrl = np.vstack([contour + vec, features, border]).astype(np.float32)
            disp = np.vstack([-vec, np.zeros((len(features) + len(border), 2), np.float32)])
            # Calculé sur une grille deux fois plus grossière (le champ est très lisse).
            coarse = _tps(ctrl, disp, P[::2, ::2], ed3)
            D += cv2.resize(coarse, (gw_, gh_), interpolation=cv2.INTER_LINEAR)

        def bulge(c, r, s):   # agrandit (s > 0) ou rétrécit (s < 0) autour de c
            d = P - c
            D[:] -= d * (s * _falloff((d * d).sum(-1), r))[..., None]

        def push(c, v, r):    # déplace le voisinage de c de v
            d = P - c
            D[:] -= v * _falloff((d * d).sum(-1), r)[..., None]

        # Côtés gauche / droit de l'image : poids de largeur visible (1 de face) ; les rayons
        # suivent moins fort.
        sl, sr = 0.5 + 0.5 * g_l, 0.5 + 0.5 * g_r
        if p["face_size"]:
            bulge(eye_c + uy * 0.55 * ed3, 1.7 * ed3, 0.22 * p["face_size"] / 100)
        if p["eyes"]:
            for e, k in ((le, sl), (re, sr)):
                bulge(e, 0.42 * ed3 * k, 0.4 * p["eyes"] / 100)
        if p["eye_distance"]:
            s = 0.07 * ed * p["eye_distance"] / 100
            push(le, -ux * s * g_l, 0.4 * ed3 * sl)
            push(re, ux * s * g_r, 0.4 * ed3 * sr)
        if p["eye_tilt"]:
            s = 0.06 * ed3 * p["eye_tilt"] / 100
            push(le - ux * 0.22 * ed * g_l, -uy * s, 0.28 * ed3 * sl)
            push(re + ux * 0.22 * ed * g_r, -uy * s, 0.28 * ed3 * sr)
        if p["nose"]:
            bulge(nose - uy * 0.1 * ed3, 0.5 * ed3, 0.36 * p["nose"] / 100)
        if p["nose_width"]:
            s = 0.06 * ed * p["nose_width"] / 100
            push(nose - ux * 0.2 * ed * g_l, ux * s * g_l, 0.3 * ed3 * sl)
            push(nose + ux * 0.2 * ed * g_r, -ux * s * g_r, 0.3 * ed3 * sr)
        if p["nose_lift"]:
            push(nose + uy * 0.08 * ed3, -uy * 0.06 * ed3 * p["nose_lift"] / 100, 0.26 * ed3)
        if p["brow_height"] or p["brow_tilt"]:
            for brow, sgn, g, k in ((pts[5 + 43:5 + 52], -1, g_l, sl), (pts[5 + 97:5 + 106], 1, g_r, sr)):
                bc = brow.mean(0)
                if p["brow_height"]:
                    push(bc - uy * 0.06 * ed3, -uy * 0.075 * ed3 * p["brow_height"] / 100, 0.36 * ed3 * k)
                if p["brow_tilt"]:
                    push(bc + ux * sgn * 0.26 * ed * g - uy * 0.04 * ed3, -uy * 0.06 * ed3 * p["brow_tilt"] / 100, 0.24 * ed3 * k)
        if p["mouth"]:
            s = 0.15 * ed * p["mouth"] / 100
            push(ml, -ux * s * g_l, 0.45 * ed3 * sl)
            push(mr, ux * s * g_r, 0.45 * ed3 * sr)
        if p["lips"]:
            mc = pts[5 + 52:5 + 72].mean(0)
            bulge(mc, 0.42 * ed3, 0.3 * p["lips"] / 100)
        if p["smile"]:
            s = 0.1 * ed3 * p["smile"] / 100
            push(ml, (-uy - ux * 0.3 * g_l) * s, 0.35 * ed3 * sl)
            push(mr, (-uy + ux * 0.3 * g_r) * s, 0.35 * ed3 * sr)
        if p["philtrum"]:
            push(pts[5 + 52:5 + 72].mean(0), -uy * 0.055 * ed3 * p["philtrum"] / 100, 0.5 * ed3)
        if p["forehead"]:
            push(eye_c - uy * 0.95 * ed3, -uy * 0.2 * ed3 * p["forehead"] / 100, 0.9 * ed3)
        _remap_roi(img, D, x0, y0, x1, y1, GRID, self._person)

    def _body(self, img: np.ndarray, pts: np.ndarray) -> None:
        """Resserre (ou élargit) horizontalement tout ce qui est sous le menton : cou, épaules, buste."""
        le, re, ed, ux, uy, eye_c = self._frame(pts)
        if ed < 8:
            return
        contour = pts[5:38]
        chin = contour[int(np.argmax((contour - eye_c) @ uy))]
        if self._person is not None and self._body_silhouette(img, chin, ed):
            return
        h, w_img = img.shape[:2]
        width = 1.9 * ed                         # largeur du visage, à peu près
        R = 4.5 * width
        cx = float(chin[0])
        y_start = float(chin[1]) + 0.2 * ed
        x0, x1 = max(0, int(cx - R)), min(w_img, int(cx + R))
        y0, y1 = max(0, int(y_start)), h
        if x1 - x0 < 16 or y1 - y0 < 16:
            return
        g = GRID * 2
        P = _grid(x0, y0, x1, y1, g)
        dx = P[..., 0] - cx
        ramp = np.clip((P[..., 1] - y_start) / (1.2 * ed), 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)
        s = 0.18 * self.p["body"] / 100
        D = np.zeros_like(P)
        D[..., 0] = s * dx * _falloff(dx * dx, R) * ramp
        _remap_roi(img, D, x0, y0, x1, y1, g, self._person)

    def _body_silhouette(self, img, chin, ed) -> bool:
        """Avec la silhouette : chaque ligne sous le menton est resserrée vers le milieu du corps à
        cette hauteur, en proportion de sa largeur réelle (buste penché ou décalé suivi)."""
        person = self._person
        h, w_img = img.shape[:2]
        k = person.shape[1] / w_img
        y_start = float(chin[1]) + 0.2 * ed
        g = GRID * 2
        rows = np.arange(int(y_start), h + g, g, dtype=np.float32)
        if len(rows) < 3:
            return True
        centre, half = np.full(len(rows), np.nan, np.float32), np.full(len(rows), np.nan, np.float32)
        cx = float(chin[0]) * k
        for i, y in enumerate(rows):
            line = person[min(person.shape[0] - 1, int(y * k))] > 0.5
            if not line.any():
                continue
            # Morceau de silhouette sous le visage (celui qui contient, ou approche le plus, le menton).
            edges = np.flatnonzero(np.diff(np.r_[0, line.astype(np.int8), 0]))
            runs = edges.reshape(-1, 2)
            dist = np.where((runs[:, 0] <= cx) & (runs[:, 1] > cx), 0, np.minimum(abs(runs[:, 0] - cx), abs(runs[:, 1] - cx)))
            a, b = runs[int(np.argmin(dist))]
            centre[i], half[i] = (a + b) / 2 / k, (b - a) / 2 / k
            cx = centre[i] * k                  # la ligne suivante suit le corps, pas le menton
        ok = ~np.isnan(centre)
        if ok.sum() < 2:
            return False
        idx = np.arange(len(rows))
        centre = np.interp(idx, idx[ok], centre[ok]).astype(np.float32)
        half = np.interp(idx, idx[ok], half[ok]).astype(np.float32)
        centre = cv2.GaussianBlur(centre[:, None], (1, 0), 2).ravel()
        half = cv2.GaussianBlur(half[:, None], (1, 0), 2).ravel()
        margin = 0.6 * ed
        x0 = max(0, int((centre - half).min() - margin))
        x1 = min(w_img, int((centre + half).max() + margin))
        y0 = int(rows[0])
        if x1 - x0 < 16 or h - y0 < 16:
            return True
        P = _grid(x0, y0, x1, h, g)
        yi = np.clip((P[:, 0, 1] - rows[0]) / g, 0, len(rows) - 1)
        c = np.interp(yi, idx, centre)[:, None]
        hw = np.interp(yi, idx, half)[:, None]
        dx = P[..., 0] - c
        out = np.clip((np.abs(dx) - hw) / margin, 0, 1)      # 0 dans le corps → 1 à la marge
        w = (1 - out * out) ** 2
        ramp = np.clip((P[..., 1] - y_start) / (1.2 * ed), 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)
        D = np.zeros_like(P)
        D[..., 0] = 0.18 * self.p["body"] / 100 * dx * w * ramp
        _remap_roi(img, D, x0, y0, x1, h, g, person)
        return True

    # ------------------------------------------------------------ maquillage
    def _makeup(self, img: np.ndarray, pts: np.ndarray) -> None:
        p = self.p
        le, re, ed, ux, uy, eye_c = self._frame(pts)
        if ed < 12:
            return
        lmk = pts[5:]
        contour = lmk[:33]
        top = [eye_c - uy * 1.15 * ed + ux * k * ed for k in (-0.9, -0.45, 0, 0.45, 0.9)]
        face_pts = np.vstack([contour, np.array(top)])
        h, w_img = img.shape[:2]
        m = int(0.15 * ed)
        x0, y0 = (max(0, int(v) - m) for v in face_pts.min(0))
        x1, y1 = (int(v) + m for v in face_pts.max(0))
        x1, y1 = min(w_img, x1), min(h, y1)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return
        off = np.array([x0, y0], np.float32)
        roi = img[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]

        def poly_mask(points, dilate=0.0):
            mk = np.zeros((rh, rw), np.uint8)
            cv2.fillConvexPoly(mk, cv2.convexHull(((points - off) * 4).astype(np.int32)), 255, cv2.LINE_AA, 2)
            if dilate > 0:
                k = max(1, int(dilate))
                mk = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
            return mk

        def feather(mk, sigma):
            return cv2.GaussianBlur(mk.astype(np.float32) / 255, (0, 0), max(0.5, sigma))

        # Bouche : enveloppe des 20 points ; l'intérieur (ouverture) = les points les plus centraux.
        mouth = lmk[52:72]
        mc = mouth.mean(0)
        d = mouth - mc
        u = d @ ux / (np.abs(d @ ux).max() + 1e-3)
        v = d @ uy / (np.abs(d @ uy).max() + 1e-3)
        inner = mouth[np.argsort(u * u + v * v)[:8]]
        mouth_full = poly_mask(mouth)
        mouth_open = poly_mask(inner)

        out = roi.astype(np.float32)
        lab = None

        def stroke(points, width, sigma):
            """Trait flou le long de points : ombres, lumières, liner."""
            mk = np.zeros((rh, rw), np.uint8)
            cv2.polylines(mk, [((np.asarray(points) - off) * 4).astype(np.int32)], False, 255, max(1, int(width)), cv2.LINE_AA, 2)
            return feather(mk, sigma)

        def spot(c, rx, ry, sigma):
            """Ovale flou dans l'axe du visage."""
            mk = np.zeros((rh, rw), np.uint8)
            angle = float(np.degrees(np.arctan2(ux[1], ux[0])))
            cv2.ellipse(mk, ((c - off) * 4).astype(np.int32), (int(rx * 4), int(ry * 4)), angle, 0, 360, 255, -1, cv2.LINE_AA, 2)
            return feather(mk, sigma)

        def near_l(lab_):
            return cv2.GaussianBlur(lab_[..., 0], (0, 0), max(1.0, 0.06 * ed))

        eye_rings = (lmk[33:43], lmk[87:97])
        brows = (lmk[43:52], lmk[97:106])

        if p["skin_smooth"] or p["glow"]:
            skin = poly_mask(face_pts)
            for idx in (range(33, 43), range(87, 97), range(43, 52), range(97, 106)):
                skin = cv2.subtract(skin, poly_mask(lmk[list(idx)], 0.12 * ed))
            skin = cv2.subtract(skin, poly_mask(mouth, 0.08 * ed))
            wskin = feather(skin, 0.08 * ed)[..., None]
            if p["skin_smooth"]:
                # Filtre bilatéral à basse résolution : lisse le grain et les petits défauts,
                # garde les contours (yeux, nez, mâchoire).
                f = min(1.0, 70 / ed)
                small = cv2.resize(roi, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else roi
                sm = cv2.bilateralFilter(small, 0, 18 + 0.3 * p["skin_smooth"], 3)
                if f < 1:
                    sm = cv2.resize(sm, (rw, rh), interpolation=cv2.INTER_LINEAR)
                out += (sm.astype(np.float32) - out) * wskin * (p["skin_smooth"] / 100 * 0.85)
            if p["glow"]:
                g = p["glow"] / 100
                lifted = 255 * (out / 255) ** (1 - 0.35 * g)       # éclaircit les tons moyens
                out += (lifted - out) * wskin

        if any(p[k] for k in ("blush", "lip_color", "teeth", "under_eye", "tan", "freckles", "contour", "highlight",
                              "eyeshadow", "eyeliner", "brow_color", "iris", "eye_bright")):
            lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            face_soft = None
            if p["tan"] or p["freckles"] or p["contour"]:
                face_soft = poly_mask(face_pts)
                for ring in eye_rings:
                    face_soft = cv2.subtract(face_soft, poly_mask(ring, 0.1 * ed))
                face_soft = cv2.subtract(face_soft, poly_mask(mouth, 0.05 * ed))
            if p["tan"]:
                k = p["tan"] / 100
                wt = feather(face_soft, 0.13 * ed)
                lab[..., 0] -= 7 * k * wt
                lab[..., 1] += 2 * k * wt
                lab[..., 2] += 8 * k * wt
            if p["under_eye"]:
                k = p["under_eye"] / 100
                # Luminosité de la peau voisine, calculée petit : le flou est large.
                small = cv2.resize(lab[..., 0], (max(8, rw // 6), max(8, rh // 6)), interpolation=cv2.INTER_AREA)
                near = cv2.resize(cv2.GaussianBlur(small, (0, 0), 0.22 * ed / 6), (rw, rh), interpolation=cv2.INTER_LINEAR)
                for ring in eye_rings:
                    wu = spot(ring.mean(0) + uy * 0.2 * ed, 0.24 * ed, 0.075 * ed, 0.055 * ed)
                    lab[..., 0] += np.maximum(near + 5 - lab[..., 0], 0) * 0.85 * k * wu
                    lab[..., 1] -= (lab[..., 1] - 132) * 0.3 * k * wu
            if p["contour"]:
                k = p["contour"] / 100
                wc = np.zeros((rh, rw), np.float32)
                for edge, corner in ((contour[6], pts[3]), (contour[26], pts[4])):
                    wc = np.maximum(wc, stroke([edge, edge + (corner - edge) * 0.6], 0.16 * ed, 0.075 * ed))
                wc *= feather(face_soft, 0.05 * ed)
                lab[..., 0] -= 30 * k * wc
                lab[..., 1] += 3 * k * wc
                lab[..., 2] += 3 * k * wc
            if p["highlight"]:
                k = p["highlight"] / 100
                wh = stroke([eye_c + uy * 0.18 * ed, pts[2] - uy * 0.08 * ed], 0.05 * ed, 0.035 * ed)
                for eye, sgn in ((le, -1), (re, 1)):
                    wh = np.maximum(wh, spot(eye + uy * 0.42 * ed + ux * sgn * 0.26 * ed, 0.22 * ed, 0.07 * ed, 0.06 * ed))
                lab[..., 0] += 17 * k * wh
            if p["freckles"]:
                k = p["freckles"] / 100
                mk = np.zeros((rh, rw), np.uint8)
                n = int(30 + 80 * k)
                for u, v, r, a in _FRECKLES[:n]:
                    # Arc nez-joues, plus dense au centre.
                    x = (u * 2 - 1) * abs(u * 2 - 1) ** 0.3 * 0.85
                    y = 0.3 + v * 0.45 + 0.22 * abs(x)
                    c = eye_c + ux * x * ed + uy * y * ed
                    cv2.circle(mk, ((c - off) * 4).astype(np.int32), max(2, int((0.004 + 0.01 * r) * ed * 4)),
                               int(70 + 185 * a), -1, cv2.LINE_AA, 2)
                wf = feather(mk, 0.009 * ed) * feather(face_soft, 0.03 * ed) * min(1.0, 0.5 + k)
                lab[..., 0] -= 26 * wf
                lab[..., 2] += 12 * wf
                lab[..., 1] += 6 * wf
            if p["eyeshadow"]:
                k = p["eyeshadow"] / 100 * 0.7
                wes = np.zeros((rh, rw), np.float32)
                for ring, brow in zip(eye_rings, brows):
                    lid = poly_mask(np.vstack([ring, ring + (brow.mean(0) - ring.mean(0)) * 0.72]), 0.02 * ed)
                    lid = cv2.subtract(lid, poly_mask(ring, 0.012 * ed))
                    lid = cv2.subtract(lid, poly_mask(brow, 0.03 * ed))
                    wes = np.maximum(wes, feather(lid, 0.06 * ed))
                tgt = cv2.cvtColor(np.uint8([[EYE_SHADES[int(p["eyeshadow_shade"]) % len(EYE_SHADES)][1]]]),
                                   cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
                for ch, m in ((0, 0.55), (1, 1.0), (2, 1.0)):
                    lab[..., ch] += (tgt[ch] - lab[..., ch]) * m * k * wes
            if p["brow_color"]:
                k = p["brow_color"] / 100
                wbr = np.zeros((rh, rw), np.float32)
                for brow in brows:
                    wbr = np.maximum(wbr, feather(poly_mask(brow), 0.012 * ed))
                # Surtout les poils, plus sombres que la peau : pas de bloc peint.
                hair = np.clip((near_l(lab) - lab[..., 0] + 2) / 16, 0.25, 1)
                lab[..., 0] *= 1 - 0.42 * k * wbr * hair
            if p["eyeliner"]:
                k = p["eyeliner"] / 100
                wel = np.zeros((rh, rw), np.float32)
                for ring, eye, sgn in ((eye_rings[0], le, -1), (eye_rings[1], re, 1)):
                    c = ring.mean(0)
                    top = ring[(ring - c) @ uy <= 0.15 * ed * 0.08]
                    top = top[np.argsort((top - c) @ ux * sgn)]
                    wing = top[-1] + (ux * sgn * 0.075 - uy * 0.045) * ed
                    wel = np.maximum(wel, stroke(np.vstack([top, wing]), 1 + 0.022 * ed * (0.5 + k), 0.005 * ed))
                lab[..., 0] -= (lab[..., 0] - 22) * 0.9 * min(1.0, k * 1.4) * wel
            if p["iris"] or p["eye_bright"]:
                for ring in eye_rings:
                    eye_mask = feather(poly_mask(ring), 0.012 * ed)
                    if p["eye_bright"]:
                        k = p["eye_bright"] / 100
                        lab[..., 0] += 15 * k * eye_mask
                        white = eye_mask * np.clip((lab[..., 0] - 105) / 50, 0, 1)
                        lab[..., 1] -= (lab[..., 1] - 128) * 0.6 * k * white
                        lab[..., 2] -= (lab[..., 2] - 128) * 0.6 * k * white
                    if p["iris"]:
                        # Le regard bouge : l'iris est retrouvé dans l'œil, c'est sa partie sombre.
                        y0e, x0e = np.maximum(0, (ring - off).min(0).astype(int))[::-1]
                        y1e, x1e = ((ring - off).max(0).astype(int) + 2)[::-1]
                        patch, wpatch = lab[y0e:y1e, x0e:x1e, 0], eye_mask[y0e:y1e, x0e:x1e]
                        if patch.size == 0 or wpatch.max() < 0.3:
                            continue
                        dark = np.where(wpatch > 0.5, patch, 255.0)
                        limit = np.percentile(dark[wpatch > 0.5], 30)
                        ys, xs = np.nonzero((dark <= limit))
                        if len(xs) == 0:
                            continue
                        c = np.array([xs.mean() + x0e, ys.mean() + y0e], np.float32) + off
                        r = 0.095 * ed
                        pupil = spot(c, r * 0.36, r * 0.36, 0.008 * ed)
                        wi = spot(c, r, r, 0.012 * ed) * eye_mask * (1 - pupil)
                        tgt = cv2.cvtColor(np.uint8([[IRIS_SHADES[int(p["iris_shade"]) % len(IRIS_SHADES)][1]]]),
                                           cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
                        k = p["iris"] / 100
                        lab[..., 1] += (tgt[1] - lab[..., 1]) * 0.6 * k * wi
                        lab[..., 2] += (tgt[2] - lab[..., 2]) * 0.6 * k * wi
                        lab[..., 0] += np.maximum(tgt[0] * 0.7 - lab[..., 0], 0) * 0.6 * k * wi
            if p["blush"]:
                k = p["blush"] / 100
                wb = np.zeros((rh, rw), np.float32)
                yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
                for eye, mouth_corner, sgn in ((le, pts[3], -1), (re, pts[4], 1)):
                    c = eye + (mouth_corner - eye) * 0.55 + ux * sgn * 0.18 * ed - off
                    wb += np.exp(-(((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * (0.3 * ed) ** 2)))
                wb = np.minimum(wb, 1) * feather(cv2.subtract(poly_mask(face_pts), mouth_full), 0.05 * ed)
                lab[..., 1] += 16 * k * wb
                lab[..., 2] += 3 * k * wb
                lab[..., 0] -= 4 * k * wb
            if p["lip_color"]:
                k = p["lip_color"] / 100 * 0.85
                wl = feather(cv2.subtract(mouth_full, mouth_open), 0.025 * ed)
                tgt = cv2.cvtColor(np.uint8([[LIP_SHADES[int(p["lip_shade"]) % len(LIP_SHADES)][1]]]),
                                   cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
                # Couleur remplacée, texture et reflets gardés (la luminosité ne bouge qu'un peu).
                lab[..., 1] += (tgt[1] - lab[..., 1]) * k * wl
                lab[..., 2] += (tgt[2] - lab[..., 2]) * k * wl
                lab[..., 0] += (tgt[0] - lab[..., 0]) * 0.3 * k * wl
            if p["teeth"]:
                k = p["teeth"] / 100
                wt = feather(cv2.erode(mouth_open, np.ones((3, 3), np.uint8)), 0.02 * ed)
                wt *= np.clip((lab[..., 0] - 90) / 60, 0, 1)      # seulement ce qui est clair : les dents
                lab[..., 0] += 22 * k * wt
                lab[..., 2] -= (lab[..., 2] - 128) * 0.7 * k * wt
                lab[..., 1] -= (lab[..., 1] - 128) * 0.4 * k * wt
            out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        img[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
