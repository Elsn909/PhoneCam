"""Effets « webcam / visio », optimisés pour le temps réel.

Comme dans une vraie visio, l'image est traitée à la résolution du flux (réduite),
compressée à cette taille, puis ré-agrandie : plus réaliste, et bien plus rapide.
"""
import random
import threading
import time
from collections import deque

import cv2
import numpy as np

DEFAULTS = {
    "scale": 100,          # % de la résolution du flux
    "blur": 0,             # flou optique, dixièmes de pixel
    "sharpen": 0,          # sur-accentuation typique des webcams bon marché, %
    "noise": 0,            # bruit capteur
    "chroma_noise": 0,     # bruit de couleur (faible lumière)
    "exposure": 0,         # dérive d'exposition / balance des blancs, %
    "contrast": 0,         # écrasement des noirs / blancs (petit capteur), %
    "jpeg": 100,           # qualité de compression (100 = désactivé)
    "glitch": 0,           # chutes de débit (gros blocs), ‰ par image
    "latency": 0,          # retard fixe, ms
    "jitter": 0,           # variation aléatoire du retard, ms
    "fps": 60,             # cadence de sortie maximale
    "freeze": 0,           # micro-gels, ‰ par image
    "motion_blur": 0,      # flou de bougé (pose longue en intérieur), %
    "adaptive": 0,         # compression qui suit le mouvement (débit visio), %
    "vignette": 0,         # vignettage + aberration chromatique d'un petit objectif, %
    # étalonnage et ambiance (tables de couleur précalculées : ~1 à 3 ms)
    "grade_temp": 6500,    # couleur de la lumière de la pièce, K (2700 ampoule chaude, 9000 froide)
    "grade_rgb": 0,        # ampoule RGB : intensité de la couleur, %
    "grade_hue": 280,      # ampoule RGB : teinte, degrés
    "grade_dark": 0,       # pièce sombre, % (image assombrie, gain de la caméra : plus de bruit)
    "grade_exposure": 0,   # exposition, −100 … 100 (± 2 diaphragmes)
    "grade_contrast": 0,   # contraste, −100 … 100
    "grade_saturation": 0, # saturation, −100 … 100
}
GRADE_KEYS = ["grade_temp", "grade_rgb", "grade_hue", "grade_dark", "grade_exposure", "grade_contrast",
              "grade_saturation"]


WEBCAM_PRESETS = {
    "Aucun": {},
    "Webcam PC portable": {"scale": 70, "jpeg": 70, "blur": 6, "sharpen": 40, "noise": 8,
                           "chroma_noise": 3, "exposure": 30, "contrast": 20, "latency": 120,
                           "jitter": 30, "fps": 30, "motion_blur": 40, "vignette": 40},
    "Visio (Zoom/Teams)": {"scale": 60, "jpeg": 45, "glitch": 4, "blur": 4, "sharpen": 20,
                           "noise": 5, "chroma_noise": 2, "exposure": 25, "contrast": 15,
                           "latency": 250, "jitter": 80, "fps": 24, "freeze": 3,
                           "motion_blur": 35, "adaptive": 50, "vignette": 25},
    "Mauvaise connexion": {"scale": 35, "jpeg": 18, "glitch": 20, "blur": 8, "noise": 10,
                           "chroma_noise": 6, "exposure": 30, "contrast": 25, "latency": 600,
                           "jitter": 250, "fps": 12, "freeze": 15, "motion_blur": 50,
                           "adaptive": 80, "vignette": 30},
}

AMBIANCES = {
    "Aucune": {},
    "Ampoule chaude": {"grade_temp": 2700},
    "Lumière froide (LED, néon blanc)": {"grade_temp": 9000, "grade_saturation": -10},
    "Pièce sombre, ampoule chaude": {"grade_temp": 2800, "grade_dark": 60, "grade_contrast": 15},
    "Soirée devant l'écran": {"grade_temp": 9500, "grade_dark": 70, "grade_contrast": 10,
                              "grade_saturation": -15},
    "Ampoule RGB violette": {"grade_rgb": 45, "grade_hue": 285, "grade_dark": 35},
    "Ampoule RGB rouge": {"grade_rgb": 45, "grade_hue": 0, "grade_dark": 35},
    "Ampoule RGB bleue": {"grade_rgb": 45, "grade_hue": 225, "grade_dark": 35},
}


def kelvin_bgr(k):
    """Couleur linéaire (B, G, R) d'une source à k kelvins, rapportée au blanc 6500 K."""
    def raw(k):
        t = k / 100.0
        r = 255.0 if t <= 66 else 329.7 * (t - 60) ** -0.1332
        g = 99.47 * np.log(t) - 161.12 if t <= 66 else 288.12 * (t - 60) ** -0.0755
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.52 * np.log(t - 10) - 305.04)
        return (np.clip([b, g, r], 1, 255) / 255.0) ** 2.2
    return (raw(k) / raw(6500)).astype(np.float32)


def hue_bgr(h):
    hsv = np.uint8([[[int(h / 2) % 180, 255, 255]]])
    return (cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32) / 255) ** 2.2


def _luma(c):
    return float(c[0] * 0.114 + c[1] * 0.587 + c[2] * 0.299)


def grade_lut(p):
    """Table 256 × 3 : lumière de la pièce (couleur, ampoule RGB, pièce sombre) vue par une
    caméra (balance des blancs qui ne corrige qu'une partie de la teinte, gain plafonné),
    puis exposition et contraste. Calculée en lumière linéaire, comme un capteur."""
    x = (np.arange(256, dtype=np.float32) / 255) ** 2.2
    # Lumière chaude / froide : la balance des blancs automatique n'en corrige qu'une partie (~30 %).
    col = kelvin_bgr(p["grade_temp"]) ** 0.7
    # Ampoule RGB : une couleur franche, que la caméra corrige peu.
    s = p["grade_rgb"] / 100
    if s > 0:
        h = hue_bgr(p["grade_hue"]) + 0.08
        col = col * (1 - s) + (h / _luma(h)) * s
    col = col / _luma(col)
    # Pièce sombre : moins de lumière, que la caméra compense en partie (gain plafonné à ×4).
    d = p["grade_dark"] / 100
    room = 0.03 ** d
    net = room * (float(np.clip(0.75 / room, 1, 4)) if d > 0 else 1.0)
    net *= 2 ** (p["grade_exposure"] / 50)
    v = x[:, None] * col[None, :] * net
    v = np.clip(v - 0.004 * d, 0, 1) ** (1 / 2.2)           # ombres bouchées d'un petit capteur
    c = p["grade_contrast"] / 100
    if c > 0:
        v = v + (v * v * (3 - 2 * v) - v) * c                # courbe en S
    elif c < 0:
        v = 0.5 + (v - 0.5) * (1 + 0.5 * c)                  # image plus plate
    return np.clip(v * 255 + 0.5, 0, 255).astype(np.uint8).reshape(1, 256, 3)


def grade_active(p) -> bool:
    return any(p[k] != DEFAULTS[k] for k in GRADE_KEYS)


class Effects:
    def __init__(self):
        self.p = dict(DEFAULTS)
        self._gain = np.ones(3, np.float32)
        self._lut = None
        self._lut_key = None
        self._glitch_left = 0
        self._noise = {}
        self._jit = 0.0
        self.ms = 0.0
        self.ref_width = None                 # largeur de la sortie (référence de « scale »)
        self._prev = None                     # image précédente (flou de bougé, mouvement)
        self._motion = 0.0                    # quantité de mouvement lissée (0…1)
        self._vig = None
        self._grade_key = None
        self._grade_lut = None

    def jitter_ms(self) -> float:
        """Retard variable qui évolue doucement (comme une file réseau), pas au hasard à chaque image :
        des tirages indépendants regrouperaient les images et en feraient sauter beaucoup."""
        amp = self.p["jitter"]
        if amp <= 0:
            self._jit = 0.0
            return 0.0
        self._jit += random.gauss(0, amp * 0.12) + (amp / 2 - self._jit) * 0.05
        self._jit = min(max(self._jit, 0.0), float(amp))
        return self._jit

    def _noise_buf(self, shape) -> np.ndarray:
        buf = self._noise.get(shape)
        if buf is None:
            buf = self._noise[shape] = np.empty(shape, np.float32)
        return buf

    def _tone_lut(self) -> np.ndarray:
        """Table 256×3 : gain d'exposition/WB par canal + courbe de contraste."""
        contrast = self.p["contrast"] / 100
        key = (tuple(np.round(self._gain, 3)), contrast)
        if key != self._lut_key:
            x = np.arange(256, dtype=np.float32) / 255
            if contrast > 0:  # courbe en S (smoothstep) : noirs bouchés, hautes lumières brûlées
                x = (1 - contrast) * x + contrast * x * x * (3 - 2 * x)
            lut = np.stack([np.clip(x * g * 255, 0, 255) for g in self._gain], axis=-1)
            self._lut = lut.astype(np.uint8).reshape(1, 256, 3)
            self._lut_key = key
        return self._lut

    def active(self) -> bool:
        p = self.p
        return any(p[k] != DEFAULTS[k] for k in ("scale", "blur", "sharpen", "noise", "chroma_noise",
                                                 "exposure", "contrast", "jpeg", "glitch", "motion_blur",
                                                 "adaptive", "vignette")) or grade_active(p)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if not self.active():
            self.ms = 0.0
            return frame
        p = self.p
        h, w = frame.shape[:2]
        img = frame
        # « Résolution du flux » est relative à la sortie (ref_width), pas à l'image reçue : un
        # téléphone en 540p donne le même rendu qu'en 720p, avec moins de travail.
        ref = self.ref_width or w
        s = min(1.0, p["scale"] / 100 * ref / w)
        k = s * w / ref                      # taille de travail par rapport à la sortie
        if s < 1:
            img = cv2.resize(img, (max(32, int(w * s)) & ~1, max(32, int(h * s)) & ~1),
                             interpolation=cv2.INTER_AREA)
        sh, sw = img.shape[:2]

        dark = p["grade_dark"] / 100
        if grade_active(p):
            img = self._grade(img)
        # Pièce sombre : le débruitage de la caméra lisse un peu les détails.
        blur = p["blur"] + 8 * dark
        if blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur / 10 * max(k, 0.3))

        if p["motion_blur"] > 0 or p["adaptive"] > 0:
            img = self._motion_stage(img, k)

        if p["vignette"] > 0:
            img = self._vignette(img)

        if p["sharpen"] > 0:
            soft = cv2.GaussianBlur(img, (0, 0), 1.2)
            a = p["sharpen"] / 100
            img = cv2.addWeighted(img, 1 + a, soft, -a, 0)

        if p["exposure"] > 0:
            # Marche aléatoire lente rappelée vers 1 : auto-exposition et auto-WB.
            amt = p["exposure"] / 100
            self._gain += np.random.normal(0, 0.004 * amt, 3).astype(np.float32)
            self._gain += (1 - self._gain) * 0.02
            self._gain[:] = self._gain.mean() + (self._gain - self._gain.mean()) * 0.6
        elif not np.all(self._gain == 1):
            self._gain[:] = 1
        if p["exposure"] > 0 or p["contrast"] > 0:
            img = cv2.LUT(img, self._tone_lut())

        # Pièce sombre : gain du capteur poussé, donc plus de bruit (luminance et couleur).
        noise_l, noise_c = p["noise"] + 14 * dark, p["chroma_noise"] + 9 * dark
        if noise_l > 0 or noise_c > 0:
            # Bruit à demi-résolution (grain d'un petit capteur), additions saturées uint8.
            # cv2.randn est bien plus rapide que numpy.random.
            nh, nw = max(8, sh // 2), max(8, sw // 2)
            luma = self._noise_buf((nh, nw))
            cv2.randn(luma, 0, noise_l / 4)
            noise = cv2.merge([luma, luma, luma])
            if noise_c > 0:
                chroma = self._noise_buf((nh, nw, 3))
                cv2.randn(chroma, (0, 0, 0), (noise_c / 4,) * 3)
                noise = cv2.add(noise, chroma)
            noise = cv2.resize(noise, (sw, sh), interpolation=cv2.INTER_LINEAR)
            img = cv2.add(img, noise, dtype=cv2.CV_8U)

        quality = p["jpeg"]
        if p["adaptive"] > 0:
            # Débit limité : quand ça bouge, l'encodeur n'a plus assez de bits → image plus
            # compressée ; à l'arrêt, elle redevient nette en une demi-seconde environ.
            quality = max(4, int(quality - (quality - 8) * self._motion * p["adaptive"] / 100))
        if p["glitch"] > 0 and self._glitch_left == 0 and random.random() < p["glitch"] / 1000:
            self._glitch_left = random.randint(4, 15)
        if self._glitch_left > 0:
            self._glitch_left -= 1
            quality = min(quality, random.randint(3, 10))
        if quality < 100:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if (sw, sh) != (w, h):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        self.ms = (time.perf_counter() - t0) * 1000
        return img


    def _grade(self, img):
        """Étalonnage : une table de couleur par canal (recalculée seulement quand un réglage
        change), puis saturation (ternie aussi par la pièce sombre)."""
        p = self.p
        key = tuple(p[k] for k in GRADE_KEYS)
        if self._grade_key != key:
            self._grade_lut = grade_lut(p)
            self._grade_key = key
        img = cv2.LUT(img, self._grade_lut)
        sat = 1 + p["grade_saturation"] / 100 - 0.3 * p["grade_dark"] / 100
        if abs(sat - 1) > 0.01:
            gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            img = cv2.addWeighted(img, sat, gray, 1 - sat, 0)
        return img

    def _motion_stage(self, img, k):
        """Mouvement entre deux images (à petite taille) → flou de bougé là où ça bouge, et
        quantité de mouvement pour la compression adaptative."""
        p = self.p
        small = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        prev = self._prev
        self._prev = (small, img)
        if prev is None or prev[1].shape != img.shape:
            return img
        diff = np.abs(small - prev[0]).mean(2)
        level = float(np.clip((np.percentile(diff, 90) - 4) / 25, 0, 1))
        # Monte vite quand ça bouge, redescend doucement (comme un débit qui se rétablit).
        self._motion += (level - self._motion) * (0.6 if level > self._motion else 0.12)
        if p["motion_blur"] <= 0:
            return img
        # Pose longue : l'image est la moyenne de l'instant présent et d'un peu du précédent,
        # seulement là où ça bouge ; plus un léger flou directionnel du mouvement.
        m = np.clip((diff - 4) / 20, 0, 1)
        if m.max() <= 0:
            return img
        m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 1.5), (img.shape[1], img.shape[0]))
        a = 0.45 * p["motion_blur"] / 100
        blurred = cv2.blur(img, (max(1, int(5 * k * p["motion_blur"] / 100)) | 1,) * 2)
        mix = cv2.addWeighted(blurred, 1 - a, prev[1], a, 0)
        return cv2.blendLinear(mix, img, m, 1 - m)

    def _vignette(self, img):
        """Coins assombris et légers liserés colorés vers les bords (petit objectif de webcam)."""
        h, w = img.shape[:2]
        a = self.p["vignette"] / 100
        key = (h, w, a)
        if self._vig is None or self._vig[0] != key:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
            gain = 1 - 0.35 * a * np.clip(r2 / 2, 0, 1) ** 1.5
            gain = cv2.merge([np.round(gain * 255).astype(np.uint8)] * 3)
            # Aberration : le rouge un peu agrandi, le bleu un peu réduit, depuis le centre.
            s = 1 + 0.004 * a
            mr = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 / s)
            mb = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
            self._vig = (key, gain, mr, mb)
        _, gain, mr, mb = self._vig
        b, g, r = cv2.split(img)
        r = cv2.warpAffine(r, mr, (w, h), borderMode=cv2.BORDER_REPLICATE)
        b = cv2.warpAffine(b, mb, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return cv2.multiply(cv2.merge([b, g, r]), gain, scale=1 / 255)


class DelayLine:
    """File de sortie : latence + gigue, cadence max et micro-gels.

    La sortie est pilotée par les images (pas par une horloge fixe) : chaque image part
    dès que son heure est venue, donc pas de doublons ni de sauts dus à un décalage d'horloges.
    """

    def __init__(self, effects: Effects):
        self.fx = effects
        self._q: deque = deque()
        self._cv = threading.Condition()
        self._next_allowed = 0.0
        self._freeze_until = 0.0

    def push(self, frame: np.ndarray, t_capture: float) -> None:
        p = self.fx.p
        delay = (p["latency"] + self.fx.jitter_ms()) / 1000
        ready = time.perf_counter() + delay
        with self._cv:
            if self._q and ready < self._q[-1][0]:
                ready = self._q[-1][0]  # l'ordre des images est conservé
            self._q.append((ready, frame, t_capture))
            while len(self._q) > 600:
                self._q.popleft()
            self._cv.notify()

    def wait_next(self, timeout: float):
        """Renvoie (image, t_capture) quand une image doit sortir, sinon None."""
        deadline = time.perf_counter() + timeout
        with self._cv:
            while True:
                now = time.perf_counter()
                if self._q and self._q[0][0] <= now:
                    # On prend la plus récente des images prêtes (rattrape un retard éventuel).
                    item = None
                    while self._q and self._q[0][0] <= now:
                        item = self._q.popleft()
                    if self._accept(now):
                        return item[1], item[2]
                    continue
                wait = deadline - now
                if self._q:
                    wait = min(wait, self._q[0][0] - now)
                if wait <= 0:
                    return None
                self._cv.wait(wait)

    def _accept(self, now: float) -> bool:
        p = self.fx.p
        if now < self._freeze_until:
            return False
        if p["freeze"] > 0 and random.random() < p["freeze"] / 1000:
            self._freeze_until = now + random.uniform(0.15, 0.6)
            return False
        interval = 1 / max(1, p["fps"])
        # Tolérance de 25 % : une image un peu en avance n'est pas jetée (sinon la cadence
        # tombe à la moitié quand l'entrée et la limite sont au même rythme).
        if now < self._next_allowed - interval * 0.25:
            return False
        # Grille régulière ; si on a pris du retard (plus d'un intervalle), on repart de maintenant.
        base = self._next_allowed if now - self._next_allowed < interval else now
        self._next_allowed = base + interval
        return True
