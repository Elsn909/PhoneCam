"""Moteur de remplacement du visage « Plus » (celui de la 1re livraison), plus réaliste que le collage d'origine.

Pour chaque visage, dans le repère du visage aligné (256 × 256) :
  1. repères lissés (filtre « One Euro ») : le visage ne flotte plus d'une image à l'autre ;
  2. remplacement : HyperSwap 256 (FaceFusion, 2025) ou inswapper 128 ;
  3. restaurateur d'expression (LivePortrait) : l'expression réelle de la personne filmée
     (bouche, lèvres, sourcils, regard) est réappliquée au visage généré ;
  4. couleur et lumière accordées à la vraie peau, puis grain et netteté accordés à l'image ;
  5. masques : analyseur de visage (BiSeNet : peau, yeux, nez, bouche — pas les cheveux ni le
     décor) et détecteur d'occlusion (XSeg : une main, un micro devant le visage restent devant) ;
  6. collage progressif.
Modèles dans data\\models-plus (supprimer le dossier pour tout retirer).
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import onnxruntime as ort

from tour_gpu import video_turn

# Repère « arcface_128 » (yeux, nez, coins de la bouche), normalisé : celui de FaceFusion pour
# HyperSwap, et aussi celui d'InsightFace (norm_crop en 128 px) avec lequel inswapper a été entraîné.
ARCFACE_128 = np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                        [0.38710391, 0.72160547], [0.61507734, 0.72034453]], np.float32)
SIZE = 256
LOT = "_lot"               # modèle dérivé qui traite toutes les passes du pixel boost d'un coup (models_plus.prepare_batch)
LOT_MAX = 16               # 4 × 4 passes (1024 px, Studio)
FIT_MARGIN = 1.1           # détail adapté : le visage généré peut être agrandi de 10 % au plus
EMBED_EVERY = 0.2          # s entre deux calculs de l'empreinte de la personne filmée (poids de l'identité)
# Classes de l'analyseur BiSeNet gardées : peau, sourcils, yeux, lunettes, nez, intérieur de la bouche, lèvres.
FACE_CLASSES = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]

# Bornes des expressions LivePortrait (FaceFusion, processors/live_portrait.py).
EXPRESSION_MIN = np.array([
    [-2.88067125e-02, -8.12731311e-02, -1.70541159e-03], [-4.88598682e-02, -3.32196616e-02, -1.67431499e-04],
    [-6.75425082e-02, -4.28681746e-02, -1.98950816e-04], [-7.23103955e-02, -3.28503326e-02, -7.31324719e-04],
    [-3.87073644e-02, -6.01546466e-02, -5.50269964e-04], [-6.38048723e-02, -2.23840728e-01, -7.13261834e-04],
    [-3.02710701e-02, -3.93195450e-02, -8.24086510e-06], [-2.95799859e-02, -5.39318882e-02, -1.74219604e-04],
    [-2.92359516e-02, -1.53050944e-02, -6.30460854e-05], [-5.56493877e-03, -2.34344602e-02, -1.26858242e-04],
    [-4.37593013e-02, -2.77768299e-02, -2.70503685e-02], [-1.76926646e-02, -1.91676542e-02, -1.15090821e-04],
    [-8.34268332e-03, -3.99775570e-03, -3.27481248e-05], [-3.40162888e-02, -2.81868968e-02, -1.96679524e-04],
    [-2.91855410e-02, -3.97511162e-02, -2.81230678e-05], [-1.50395725e-02, -2.49494594e-02, -9.42573533e-05],
    [-1.67938769e-02, -2.00953931e-02, -4.00750607e-04], [-1.86435618e-02, -2.48535164e-02, -2.74416432e-02],
    [-4.61211195e-03, -1.21660791e-02, -2.93173041e-04], [-4.10017073e-02, -7.43824020e-02, -4.42762971e-02],
    [-1.90370996e-02, -3.74363363e-02, -1.34740388e-02]], np.float32)[None]
EXPRESSION_MAX = np.array([
    [4.46682945e-02, 7.08772913e-02, 4.08344204e-04], [2.14308221e-02, 6.15894832e-02, 4.85319615e-05],
    [3.02363783e-02, 4.45043296e-02, 1.28298725e-05], [3.05869691e-02, 3.79812494e-02, 6.57040102e-04],
    [4.45670523e-02, 3.97259220e-02, 7.10966764e-04], [9.43699256e-02, 9.85926315e-02, 2.02551950e-04],
    [1.61131397e-02, 2.92906128e-02, 3.44733417e-06], [5.23825921e-02, 1.07065082e-01, 6.61510974e-04],
    [2.85718683e-03, 8.32320191e-03, 2.39314613e-04], [2.57947259e-02, 1.60935968e-02, 2.41853559e-05],
    [4.90833223e-02, 3.43903080e-02, 3.22353356e-02], [1.44766076e-02, 3.39248963e-02, 1.42291479e-04],
    [8.75749043e-04, 6.82212645e-03, 2.76097053e-05], [1.86958015e-02, 3.84016186e-02, 7.33085908e-05],
    [2.01714113e-02, 4.90544215e-02, 2.34028921e-05], [2.46518422e-02, 3.29151377e-02, 3.48571630e-05],
    [2.22457591e-02, 1.21796541e-02, 1.56396593e-04], [1.72109623e-02, 3.01626958e-02, 1.36556877e-02],
    [1.83460284e-02, 1.61141958e-02, 2.87440169e-04], [3.57594155e-02, 1.80554688e-01, 2.75554154e-02],
    [2.17450950e-02, 8.66811201e-02, 3.34241726e-02]], np.float32)[None]
EXPRESSION_KEEP = [0, 4, 5, 8, 9]   # composantes de pose/forme : toujours celles du visage généré
# Zones du restaurateur d'expression (FaceFusion, expression_restorer_areas).
EXPRESSION_UPPER = [1, 2, 6, 10, 11, 12, 13, 15, 16]   # sourcils, yeux
EXPRESSION_LOWER = [3, 7, 14, 17, 18, 19, 20]          # bouche, mâchoire
# Zones du visage remplacées (FaceFusion, face_mask_regions) : classes de BiSeNet.
REGIONS = {"skin": [1], "brows": [2, 3], "eyes": [4, 5], "glasses": [6], "nose": [10], "mouth": [11],
           "lips": [12, 13]}

# Entrées de chaque modèle (tailles fixes : TensorRT prépare un moteur exactement pour elles).
INPUT_SHAPES = {
    "hyperswap": {"source": (1, 512), "target": (1, 3, 256, 256)},
    "xseg_1": {"input": (1, 256, 256, 3)},
    "xseg_2": {"input": (1, 256, 256, 3)},
    "xseg_3": {"input": (1, 256, 256, 3)},
    "hrffa": {"input": (1, 3, 256, 256)},
    "bisenet_face_256": {"input": (1, 3, 512, 512)},
    "bisenet_labels_256": {"input": (1, 3, 512, 512)},
    "live_portrait_feature_extractor": {"input": (1, 3, 256, 256)},
    "live_portrait_motion_extractor": {"input": (1, 3, 256, 256)},
    "live_portrait_generator": {"feature_volume": (1, 32, 16, 64, 64), "source": (1, 21, 3), "target": (1, 21, 3)},
}
# En demi-précision ces deux-là se trompent (masque faux, valeurs invalides) : pleine précision.
FP32_MODELS = {"xseg_1", "xseg_2", "xseg_3", "live_portrait_motion_extractor"}


_BOX_MASKS = {}


def _box_mask(size, padding=6, blur=4):
    """Bord progressif du collage : marge et fondu en % du repère (FaceFusion : face_mask_padding,
    face_mask_blur)."""
    key = (size, padding, blur)
    m = _BOX_MASKS.get(key)
    if m is None:
        m = np.zeros((size, size), np.float32)
        pad = int(size * (padding / 100))
        if pad > 0:
            m[pad:-pad, pad:-pad] = 1
        else:
            m[:] = 1
        if blur > 0:
            m = cv2.GaussianBlur(m, (0, 0), size * (blur / 100))
        _BOX_MASKS[key] = m
    return m


_LOW_MASKS = {}


def _box_mask_low(size, padding=6, blur=4):
    """Même bord que _box_mask, mais ouvert vers le bas : le collage peut descendre sous le menton
    jusqu'au bord du cadre (fondu sur les dernières lignes)."""
    key = (size, padding, blur)
    m = _LOW_MASKS.get(key)
    if m is None:
        m = np.zeros((size, size), np.float32)
        pad = max(1, int(size * (padding / 100)))
        m[pad:, pad:-pad] = 1
        if blur > 0:
            m = cv2.GaussianBlur(m, (0, 0), size * (blur / 100), borderType=cv2.BORDER_REPLICATE)
        fade = max(3, size // 50)
        m[-fade:] *= np.linspace(1, 0, fade + 1, dtype=np.float32)[:-1, None]
        _LOW_MASKS[key] = m
    return m


def _implode(img, total, size):
    """Pixel boost (FaceFusion) : l'image total·size découpée en total² images entrelacées de size."""
    c = img.shape[2:]
    axes = (1, 3, 0, 2) + tuple(range(4, 4 + len(c)))
    return img.reshape(size, total, size, total, *c).transpose(axes).reshape(total * total, size, size, *c)


def _explode(tiles, total, size):
    c = tiles.shape[3:]
    axes = (2, 0, 3, 1) + tuple(range(4, 4 + len(c)))
    return tiles.reshape(total, total, size, size, *c).transpose(axes).reshape(total * size, total * size, *c)
_PARSER_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
_PARSER_SCALE = 1 / (np.array([0.229, 0.224, 0.225], np.float32) * 255)
_PARSER_MEAN_RGB = tuple(float(v) for v in _PARSER_MEAN)
_PARSER_SCALE4 = _PARSER_SCALE.reshape(1, 3, 1, 1)


def _kind(name):
    return "hyperswap" if name.startswith("hyperswap") else name


DEFAULTS = {
    "model": "hyperswap_1a_256",   # ou « inswapper » (modèle d'origine, 128 px)
    "expression": 80,              # restaurateur d'expression, % (0 = coupé)
    "occlusion": True,             # mains, micro… devant le visage
    "parser": True,                # masque du visage seul (sans cheveux ni décor)
    "color": 70,                   # accord des couleurs, %
    "grain": 100,                  # accord du grain et de la netteté, %
    "stabilize": 50,               # stabilisation des repères, %
    "profile_smooth": 70,          # stabilisation du cadrage de profil / tête levée seulement, % (0 = coupée)
    "hold": True,                  # visage manqué une ou deux images : garder le précédent
    "detail": 80,                  # détail du pixel boost gardé après l'expression, % (0 : le boost est perdu)
    "boost_fit": True,             # direct : pas plus de passes que la taille du visage dans l'image n'en demande
    # Réglages avancés repris de FaceFusion (valeurs par défaut = rendu d'origine)
    "occluder": "xseg_1",          # détecteur d'occlusion : xseg_1 (d'origine), xseg_2 ou xseg_3
    "landmarker": "detector",      # repères du cadrage : ceux du détecteur (d'origine) ou « hrffa »
    "boost": 1,                    # pixel boost : 1 = 256 px, 2 = 512 px (4 passes), 3 = 768 px (9 passes)
    "weight": 50,                  # poids de l'identité : 50 = photo source seule (face_swapper_weight)
    "regions": list(REGIONS),      # zones remplacées (face_mask_regions)
    "expr_upper": True,            # expression du haut du visage (sourcils, yeux)
    "expr_lower": True,            # expression du bas du visage (bouche, mâchoire)
    "mask_padding": 6,             # marge du bord, %
    "mask_blur": 4,                # fondu du bord, %
    "mask_soften": 5,              # adoucissement des contours (analyseur, occlusion), px
    "occ_threshold": 50,           # XSeg : note sous laquelle un pixel compte comme caché, %
    "occ_islands": 0,              # XSeg : taches ignorées dans le visage, en % de sa surface (moustache…)
    "occ_steady": 0,               # XSeg : masque stabilisé d'une image à l'autre, %
    "guard_check": True,           # anti-fuite : comparer deux systèmes de repères (voir _agreement)
    "motion_mask": True,           # tête qui bouge vite : masques assouplis (voir _relax)
    "beard": False,                # expérimental : barbe ou bouc restés au bas du menton recouverts (voir _beard)
}


STABLE_BETA = 8.0                # stabilisation : réactivité aux mouvements rapides
RELAX_SPEED = [1.5, 5.0]         # masques assouplis : vitesse de la tête, en yeux par seconde
RELAX_TURN = [0.05, 0.4]         # et tête tournée (_profileness)


def _profileness(kps):
    """0 de face → 1 de profil ou tête très levée / baissée, d'après les 5 points : écart des yeux
    rapporté à la distance yeux ↔ bouche, comparé au visage de face du modèle. Mesuré sur les
    clips de test : rapport ~1,1 de face (0,96 à 1,2), ~0,94 à 25–40°, ~0,4 au-delà de 40°."""
    eye = np.linalg.norm(kps[1] - kps[0])
    v = np.linalg.norm((kps[3] + kps[4]) / 2 - (kps[0] + kps[1]) / 2)
    t = ARCFACE_128
    t_eye = np.linalg.norm(t[1] - t[0])
    t_v = np.linalg.norm((t[3] + t[4]) / 2 - (t[0] + t[1]) / 2)
    r = (eye / max(float(v), 1e-3)) / (t_eye / t_v)
    side = float(np.clip((0.95 - r) / 0.45, 0, 1))      # tête tournée : les yeux se rapprochent
    tilt = float(np.clip((r - 1.3) / 0.3, 0, 1))        # tête levée ou baissée : yeux et bouche se rapprochent
    return max(side, tilt)


def _euler_matrix(pitch, yaw, roll):
    """Rotation « xyz » en degrés (même convention que scipy Rotation.from_euler('xyz'))."""
    p, y, r = np.radians([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


class OneEuro:
    """Filtre « One Euro » : lisse beaucoup quand c'est immobile, presque pas quand ça bouge vite."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1 / (2 * np.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


class SwapPlus:
    def __init__(self, models_dir: str, providers=None):
        self.dir = models_dir
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.p = dict(DEFAULTS)
        self._sessions = {}
        self._lock = threading.Lock()
        self._tracks = []          # [centre, filtre, numéro] des visages des images précédentes
        self._next_id = 0
        self._profile = []         # [centre, filtre, instant] : stabilisation de profil, par visage
        self._color = {}           # statistiques de couleur lissées, par visage
        self._occ = {}             # masque d'occlusion stabilisé, par visage
        self._motion = {}          # dernier passage de chaque visage : [repères, instant, assouplissement]
        self._fixed = set()        # visages suivis par le Studio (numéros gardés pour l'accord des couleurs)
        self._temb = []            # empreinte de la personne filmée, par visage : [centre, moyenne, instant]
        self._boost = {}           # détail adapté : nombre de passes en cours, par visage
        self._grain = {}           # grain à ajouter (ou netteté à retirer) lissé, par visage
        self._chin = {}            # masque de la barbe sous le menton lissé, par visage
        self._failed = set()       # modèles facultatifs dont la préparation a échoué : pas de nouvel essai
        self._no_trt = "TensorrtExecutionProvider" not in ort.get_available_providers()
        self.boost_now = 1         # passes par côté de la dernière image (affiché dans le diagnostic)
        self.ms = {}
        self.inswapper = None      # INSwapper d'insightface (mode « inswapper »), fourni par l'appelant
        self.use_trt = True
        # Masques et expression réelle ne dépendent que de l'image filmée : calculés pendant le swap.
        self._pool = ThreadPoolExecutor(2, thread_name_prefix="swap-plus")
        self._building = set()     # sessions préparées en arrière-plan (le flux continue)
        self.note = ""             # état d'un modèle préparé en arrière-plan (affiché sous le moteur Plus)
        self._tiles = ThreadPoolExecutor(4, thread_name_prefix="swap-boost")   # passes du pixel boost
        self._seq = 0              # numéro de la prochaine image préparée
        self._turn = 0             # numéro de l'image dont c'est le tour d'accorder ses couleurs
        self._turn_cv = threading.Condition()

    def close(self):
        self._pool.shutdown(wait=False)
        self._tiles.shutdown(wait=False)
        self._sessions.clear()

    confidence = []
    recognizer = None

    # Deux étapes, pour pouvoir chevaucher deux images (engine.Engine.parallel) : prepare() dans
    # l'ordre des images (repères, cadrage, remplacement), render() éventuellement en même temps
    # que l'image suivante (expression, masques, couleur, collage). Chaque image donne exactement
    # le même résultat que process() ; seul l'accord des couleurs (lissé d'une image à l'autre)
    # attend son tour.
    def prepare(self, frame, pairs):
        return self._stage_a(frame, pairs)

    def render(self, frame, prep):
        return self._stage_b(frame, prep), prep["conf"]

    @staticmethod
    def available(models_dir: str) -> bool:
        return os.path.isfile(os.path.join(models_dir, "hyperswap_1a_256.onnx"))

    def session(self, name):
        """Session TensorRT (moteur optimisé pour la carte, mis en cache dans models-plus\\trt-cache),
        sinon CUDA. La 1re préparation d'un modèle prend de quelques secondes à une minute."""
        s = self._sessions.get(name)
        if s is None:
            with self._lock:
                s = self._sessions.get(name)
                if s is None:
                    so = ort.SessionOptions()
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    providers = list(self.providers)
                    if name.endswith(LOT):
                        import models_plus
                        models_plus.prepare_batch(name[:-len(LOT)], log=lambda m: setattr(self, "note", m))
                    if self.use_trt and not self._no_trt:
                        def shapes(n=None):   # n : taille du lot (modèle dérivé seulement)
                            return ",".join(f"{k}:{'x'.join(map(str, v if n is None else (n,) + tuple(v[1:])))}"
                                            for k, v in INPUT_SHAPES[_kind(name)].items())
                        lo, opt, hi = (shapes(1), shapes(9), shapes(LOT_MAX)) if name.endswith(LOT) else (shapes(),) * 3
                        cache = os.path.join(self.dir, "trt-cache")
                        os.makedirs(cache, exist_ok=True)
                        providers.insert(0, ("TensorrtExecutionProvider", {
                            "trt_fp16_enable": _kind(name) not in FP32_MODELS,
                            "trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
                            "trt_timing_cache_enable": True, "trt_timing_cache_path": cache,
                            "trt_profile_min_shapes": lo, "trt_profile_max_shapes": hi,
                            "trt_profile_opt_shapes": opt}))
                    s = ort.InferenceSession(os.path.join(self.dir, name + ".onnx"), so, providers=providers)
                    self._sessions[name] = s
        return s

    def ready_session(self, name):
        """Session si elle est prête ; sinon la prépare en arrière-plan (téléchargement d'un modèle
        facultatif, TensorRT : jusqu'à une minute) et renvoie None : le rendu continue sans elle."""
        s = self._sessions.get(name)
        if s is None and name not in self._building and name not in self._failed:
            self._building.add(name)

            def work():
                try:
                    import models_plus
                    if name in models_plus.OPTIONAL:
                        models_plus.download_optional(name, log=lambda m: setattr(self, "note", m))
                    self.note = f"Préparation de {name} (la 1re fois : jusqu'à une minute)…"
                    self.session(name)
                    self.note = ""
                except Exception as e:
                    self.note = f"{name} indisponible : {e}"
                    self._failed.add(name)
                finally:
                    self._building.discard(name)
            threading.Thread(target=work, daemon=True, name="plus-" + name).start()
        return s

    def warmup(self):
        crop = np.zeros((SIZE, SIZE, 3), np.uint8)
        names = [self.p["model"]] if self.p["model"] != "inswapper" else []
        names += ["xseg_1", "bisenet_face_256", "live_portrait_feature_extractor",
                  "live_portrait_motion_extractor", "live_portrait_generator"]
        for n in names:
            self.session(n)
        if self.p["model"] != "inswapper":
            self._swap(crop, np.zeros(512, np.float32))
        self._occlusion(crop)
        self._side(crop)
        self._restore(self._expression_of(crop), crop, 0.8)

    # ------------------------------------------------------------ étapes
    def _stable_kps(self, kps, now):
        """Repères lissés, associés au visage le plus proche de l'image précédente."""
        c = kps.mean(0)
        # Taille du visage : l'écart des yeux seul s'effondre de profil, d'une image à l'autre.
        size = max(float(np.linalg.norm(kps[1] - kps[0])),
                   float(np.linalg.norm((kps[3] + kps[4]) / 2 - (kps[0] + kps[1]) / 2)), 1.0)
        best = None
        for i, (pc, _f, _id) in enumerate(self._tracks):
            if np.linalg.norm(pc - c) < size * 1.5 and (best is None or np.linalg.norm(pc - c) < best[0]):
                best = (np.linalg.norm(pc - c), i)
        s = self.p["stabilize"] / 100
        if best is None:
            f = OneEuro(min_cutoff=3.0 - 2.5 * s, beta=0.05)
            self._tracks.append([c, f, self._next_id])
            self._next_id += 1
            track = self._tracks[-1]
        else:
            track = self._tracks[best[1]]
            track[0] = c
            f = track[1]
            f.min_cutoff = 3.0 - 2.5 * s
        # Filtrer en pixels : des positions divisées par la taille puis remultipliées par celle de
        # l'image suivante projetaient tout le visage hors de la tête de profil (capture réelle).
        f.beta = STABLE_BETA / size                  # vitesses rapportées à la taille du visage
        return f(kps, now).astype(np.float32), track[2]

    def _hrffa_kps(self, frame, bbox, session, face=None):
        """HRFFA (FaceFusion 3.9, « face_landmarker_model ») : 68 repères de toute la tête, prévus pour
        rester justes jusqu'au profil complet ; ramenés aux 5 repères du cadrage comme FaceFusion
        (« 5/68 » : centre de chaque œil, bout du nez, coins de la bouche).
        Les 68 repères sont laissés sur le visage (face["landmark_hrffa_68"]) : les filtres de forme
        les reprennent au lieu de relancer le modèle sur la même image."""
        bbox = np.asarray(bbox, np.float32)
        scale = 256 / max(1.0, float((bbox[2:] - bbox[:2]).max()) * 1.7)
        tx, ty = (256 - (bbox[2:] + bbox[:2]) * scale) * 0.5
        M = np.array([[scale, 0, tx], [0, scale, ty]], np.float32)
        crop = cv2.warpAffine(frame, M, (256, 256))
        x = np.ascontiguousarray(((crop[:, :, ::-1].astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None])
        pts = session.run(None, {"input": x})[0].reshape(-1, 2) * 256
        pts = (pts - (tx, ty)) / scale
        # Un visage gardé d'une image manquée est le même objet : on ne remplace pas ses repères.
        if face is not None and face.get("landmark_hrffa_68") is None:
            face["landmark_hrffa_68"] = pts.astype(np.float32)
        return np.array([pts[36:42].mean(0), pts[42:48].mean(0), pts[30], pts[48], pts[54]], np.float32)

    def _relax(self, track_id, kps, M, work, now, conf):
        """0 → 1 quand les masques ne sont plus fiables : tête qui bouge vite, tournée, ou repères qui se
        contredisent. XSeg et le masque du modèle y ouvrent des trous qui montrent le vrai visage.
        Ces masques s'assouplissent le temps du passage (retour en ~0,25 s) ; visage de face et
        immobile, le rendu ne change pas."""
        if not self.p["motion_mask"]:
            return 0.0
        w = max(float(np.interp(_profileness(kps), RELAX_TURN, [0.0, 1.0])), float(np.interp(conf, [0.55, 0.85], [1.0, 0.0])))
        last = self._motion.get(track_id)
        if last is not None and now - last[1] < 0.3:
            dt = max(1e-3, now - last[1])
            eye = float(np.linalg.norm(ARCFACE_128[1] - ARCFACE_128[0])) * work / max(float(np.hypot(M[0, 0], M[1, 0])), 1e-6)
            speed = float(np.linalg.norm(kps - last[0], axis=1).mean()) / (max(eye, 1.0) * dt)   # yeux par seconde
            w = max(w, float(np.interp(speed, RELAX_SPEED, [0.0, 1.0])), last[2] * float(np.exp(-dt / 0.25)))
        self._motion[track_id] = [kps.copy(), now, w]
        return w

    @staticmethod
    def _agreement(detector, hrffa, M, work):
        """Confiance pour la garde anti-fuite. Sur une tête qui tourne vite, le détecteur reste sûr
        de lui alors que le remplacement lâche ; mais ses repères et ceux de HRFFA divergent : mesuré
        en aller-retour rapide, cet écart repère 92 % des images en fuite (la confiance du détecteur
        87 %) et, image figée gardée, les fuites passent de 63 images à 0 sur le même test."""
        eye = np.linalg.norm(ARCFACE_128[1] - ARCFACE_128[0]) * work / max(float(np.hypot(M[0, 0], M[1, 0])), 1e-6)
        gap = float(np.linalg.norm(hrffa - detector, axis=1).mean()) / max(float(eye), 1.0)
        return float(np.interp(gap, [0.10, 0.20, 0.30], [1.0, 0.55, 0.1]))

    def _profile_smooth(self, kps, now):
        """De profil ou tête levée, les repères (donc le cadre du visage) tremblent et sautent
        d'une image à l'autre : ils y sont lissés (filtre « One Euro », qui laisse passer les
        mouvements rapides). De face, rien ne change : le lissage s'efface progressivement
        en dessous de ~40° (mesuré : gros sauts du cadre de profil 29 % → ~10 % des images)."""
        s = self.p["profile_smooth"] / 100
        if s <= 0:
            self._profile = []
            return kps
        c = kps.mean(0)
        size = max(float(np.linalg.norm(kps[1] - kps[0])),
                   float(np.linalg.norm((kps[3] + kps[4]) / 2 - (kps[0] + kps[1]) / 2)), 1.0)
        self._profile = [t for t in self._profile if now - t[2] < 0.3]     # visages disparus
        near = [t for t in self._profile if np.linalg.norm(t[0] - c) < size * 1.5]
        track = min(near, key=lambda t: np.linalg.norm(t[0] - c)) if near else None
        mc, beta = float(np.interp(s, [0, 1], [3.0, 0.7])), float(np.interp(s, [0, 1], [2.0, 0.5]))
        if track is None:
            track = [c, OneEuro(min_cutoff=mc, beta=beta), now]
            self._profile.append(track)
        f = track[1]
        f.min_cutoff, f.beta = mc, beta / size           # vitesses rapportées à la taille du visage
        smooth = f(kps, now)
        track[0], track[2] = c, now
        w = _profileness(kps)
        return kps if w <= 0 else (kps + (smooth - kps) * w).astype(np.float32)

    def _swap(self, crop, embedding):
        video_turn()            # la voix transformée passe d'abord (tour_gpu.py)
        x = (crop[:, :, ::-1].astype(np.float32) / 255 - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None]
        out, mask = self.session(self.p["model"]).run(None, {"source": embedding.reshape(1, -1).astype(np.float32),
                                                             "target": x})
        img = (out[0].transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)[:, :, ::-1]
        return (img * 255).astype(np.uint8), mask[0, 0].clip(0, 1)

    def batch_session(self, wait=False):
        """Session qui fait toutes les passes du pixel boost en un calcul, si elle est prête (préparée
        en arrière-plan la 1re fois : environ une minute ; les passes se font une à une en attendant).
        TensorRT seulement : sur CUDA seul, le modèle par lot ne rend pas exactement la même image."""
        if self._no_trt or not self.use_trt or not str(self.p["model"]).startswith("hyperswap"):
            return None
        name = self.p["model"] + LOT
        s = self.session(name) if wait else self.ready_session(name)
        if s is not None and s.get_providers()[0] != "TensorrtExecutionProvider":
            return None
        return s

    def _swap_batch(self, session, tiles, embedding):
        """Toutes les passes du pixel boost d'un coup : tiles (n, 256, 256, 3) → images et masques."""
        video_turn()
        x = np.ascontiguousarray(tiles[..., ::-1].transpose(0, 3, 1, 2), np.float32)
        x *= 1 / 127.5
        x -= 1
        src = np.repeat(embedding.reshape(1, -1).astype(np.float32), len(tiles), 0)
        out, mask = session.run(None, {"source": src, "target": x})
        out = out.transpose(0, 2, 3, 1)[..., ::-1]
        return np.clip(out * 127.5 + 127.5, 0, 255).astype(np.uint8), mask[:, 0].clip(0, 1)

    def _fit_boost(self, track_id, kps, limit):
        """Détail adapté à la taille du visage : inutile de générer 768 px pour un visage qui en occupe
        350 dans l'image. Le nombre de passes ne change qu'une fois la taille franchie de 6 % (sinon
        il oscillerait d'une image à l'autre)."""
        M = cv2.estimateAffinePartial2D(kps, ARCFACE_128 * SIZE, method=cv2.LMEDS)[0]
        if M is None:
            return limit
        side = SIZE / max(float(np.hypot(M[0, 0], M[1, 0])), 1e-6)     # côté du cadre du visage, en pixels de l'image
        need = max(1, int(np.ceil(side / (SIZE * FIT_MARGIN))))
        prev = self._boost.get(track_id)
        if prev is not None and abs(need - prev) == 1:
            edge = SIZE * FIT_MARGIN * min(need, prev)
            if abs(side - edge) < edge * 0.06:
                need = prev
        self._boost[track_id] = need
        return min(limit, need)

    def _swap_inswapper(self, small, embedding):
        # Contigu : l'adaptateur « CUDA graph » de Deep-Live-Cam copie la mémoire telle quelle.
        blob = np.ascontiguousarray((small[:, :, ::-1].astype(np.float32) / 255).transpose(2, 0, 1)[None])
        latent = (embedding.reshape(1, -1) @ self.inswapper.emap)
        latent /= np.linalg.norm(latent)
        pred = self.inswapper.session.run(self.inswapper.output_names,
                                          {self.inswapper.input_names[0]: blob,
                                           self.inswapper.input_names[1]: latent.astype(np.float32)})[0]
        return (pred[0].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)[:, :, ::-1]

    def _embedding(self, src, tgt, frame, kps=None, now=None):
        """Poids de l'identité (FaceFusion, face_swapper_weight) : à 50 la photo source seule ; en
        dessous on garde un peu de la personne filmée, au-dessus on s'en éloigne davantage.
        L'identité de la personne filmée ne change pas d'une image à l'autre : son empreinte est
        recalculée 5 fois par seconde (moyenne glissante) au lieu de chaque image."""
        emb = src.normed_embedding
        w = float(np.interp(self.p["weight"] / 100, [0, 1], [0.35, -0.35]))
        if abs(w) < 1e-6:
            return emb
        cached = None
        if kps is not None:
            # Le visage le plus proche (même critère que le suivi des repères) ; oubliés après 1 s.
            c, size = kps.mean(0), float(np.linalg.norm(kps[1] - kps[0])) or 1.0
            self._temb = [e for e in self._temb if now - e[2] < 1.0]
            near = [e for e in self._temb if np.linalg.norm(e[0] - c) < size * 1.5]
            cached = min(near, key=lambda e: np.linalg.norm(e[0] - c)) if near else None
            if cached is not None:
                cached[0] = c
        t = getattr(tgt, "normed_embedding", None)
        if t is None and cached is not None and now - cached[2] < EMBED_EVERY:
            t = cached[1]
        else:
            if t is None and self.recognizer is not None:
                self.recognizer.get(frame, tgt)        # empreinte de la personne filmée
                t = getattr(tgt, "normed_embedding", None)
            if t is not None and kps is not None:
                t = t / np.linalg.norm(t)
                if cached is not None:
                    t = cached[1] * 0.5 + t * 0.5
                    cached[1], cached[2] = t, now
                else:
                    self._temb.append([c, t, now])
        if t is None:
            return emb
        return (emb * (1 - w) + t / np.linalg.norm(t) * w).astype(np.float32)

    def _keep_detail(self, sharp, restored):
        """LivePortrait rend en 512 px à partir d'un visage en 256 : au-delà, le détail du pixel boost
        serait perdu. Il est reporté là où l'expression n'a pas déplacé le visage.
        Les grandes formes (ce que l'expression a changé) sont comparées en 256 px, où le flou large
        ne coûte presque rien ; seul le détail fin est repris à pleine taille."""
        k = self.p["detail"] / 100
        if k <= 0:
            return restored
        n = sharp.shape[0]
        small = lambda im: cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        la, lb = (cv2.GaussianBlur(small(im), (0, 0), 2.0) for im in (sharp, restored))
        same = 1 - np.clip(cv2.GaussianBlur(np.abs(la - lb).mean(-1), (0, 0), 2.0) / 16, 0, 1)
        up = lambda im: cv2.resize(im, (n, n), interpolation=cv2.INTER_LINEAR)
        # détail fin du visage net − détail fin du visage animé = (net − animé) − (formes du net − formes de l'animé)
        fine = cv2.subtract(sharp, restored, dtype=cv2.CV_32F)
        fine -= up(la - lb)
        w = up(same * k)
        fine *= cv2.merge([w, w, w])
        return cv2.add(restored, fine, dtype=cv2.CV_8U)

    def _soften(self, m):
        return cv2.GaussianBlur(m, (0, 0), self.p["mask_soften"]) if self.p["mask_soften"] > 0 else m

    def _occlusion(self, crop):
        video_turn()            # la voix transformée passe d'abord (tour_gpu.py)
        x = crop.astype(np.float32)[None] / 255
        # Autre version de XSeg choisie : utilisée dès qu'elle est prête (XSeg 1 en attendant).
        name = self.p["occluder"]
        s = self.ready_session(name) if name in ("xseg_2", "xseg_3") else None
        m = (s or self.session("xseg_1")).run(None, {"input": x})[0][0, :, :, 0].clip(0, 1)
        t = self.p["occ_threshold"] / 100
        if self.p["occ_islands"] > 0:
            m = self._ignore_islands(m, t)
        m = self._soften(m)
        return ((m.clip(t, 1) - t) / (1 - t)).astype(np.float32)

    def _ignore_islands(self, m, t):
        """Moustache, grain de beauté… : petites taches dans le contour du visage, là où une main ou
        un micro font une grande tache. Le contour est l'enveloppe convexe du visage : une moustache
        qui rejoint le bord de la joue reste dedans (le décor, lui, compte aussi comme caché)."""
        hidden = (m < t).astype(np.uint8)
        contours = cv2.findContours(1 - hidden, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            return m
        face = max(contours, key=cv2.contourArea)
        hull = np.zeros_like(hidden)
        cv2.fillConvexPoly(hull, cv2.convexHull(face), 1)
        labels, blobs = cv2.connectedComponentsWithStats(hidden & hull, connectivity=8)[1:3]
        small = blobs[:, cv2.CC_STAT_AREA] < hull.sum() * self.p["occ_islands"] / 100
        small[0] = False
        if not small.any():
            return m
        lost = cv2.dilate(small[labels].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) & hull
        return np.where(lost > 0, np.float32(1), m)

    def _steady(self, key, occlusion):
        """Masque moyenné avec les images précédentes pour qu'il ne clignote plus. Une grande zone qui
        change d'un coup (main qui arrive ou repart) passe sans retard."""
        prev = self._occ.get(key)
        if prev is None:
            self._occ[key] = occlusion
            return occlusion
        k = self.p["occ_steady"] / 100 * 0.85
        out = cv2.addWeighted(prev, k, occlusion, 1 - k, 0)
        moved = (np.abs(occlusion - prev) > 0.5).astype(np.uint8)
        labels, blobs = cv2.connectedComponentsWithStats(moved, connectivity=8)[1:3]
        big = blobs[:, cv2.CC_STAT_AREA] >= 0.03 * SIZE * SIZE
        big[0] = False
        if big.any():
            out = np.where(big[labels], occlusion, out)
        self._occ[key] = out
        return out

    def _region(self, crop):
        video_turn()            # la voix transformée passe d'abord (tour_gpu.py)
        # Agrandissement, ordre des couleurs et moyenne en une passe (2,5 ms de moins que par numpy).
        x = cv2.dnn.blobFromImage(crop, 1.0, (512, 512), _PARSER_MEAN_RGB, swapRB=True)
        x *= _PARSER_SCALE4
        regions = self.p["regions"]
        if set(regions) != set(REGIONS):
            # Zones choisies : classe de chaque pixel (modèle dérivé, voir models_plus.prepare_labels).
            s = self.ready_session("bisenet_labels_256")
            if s is not None:
                label = np.rint(s.run(None, {"input": x})[0][0, 0]).astype(np.uint8)
                m = np.isin(label, [c for r in regions for c in REGIONS.get(r, [])]).astype(np.float32)
                m = self._soften(m)
                return ((m.clip(0.5, 1) - 0.5) * 2).astype(np.float32), None
        # « Visage oui/non » calculé dans le modèle (voir models_plus.prepare_parser) : canal 0 ;
        # canal 1 : cheveux et poils.
        out = self.session("bisenet_face_256").run(None, {"input": x})[0][0]
        m = self._soften(out[0])
        return ((m.clip(0.5, 1) - 0.5) * 2).astype(np.float32), (out[0], out[1])

    @staticmethod
    def _lp_input(c):
        return cv2.dnn.blobFromImage(c, 1 / 255, swapRB=True)

    def _expression_of(self, crop):
        """Expression LivePortrait du vrai visage filmé."""
        return self.session("live_portrait_motion_extractor").run(None, {"input": self._lp_input(crop)})[5]

    def _restore(self, t_expr, temp_crop, factor):
        """LivePortrait : anime le visage généré (temp) avec l'expression de la vraie personne (t_expr)."""
        video_turn()
        s_in = self._lp_input(temp_crop if temp_crop.shape[0] == SIZE else
                              cv2.resize(temp_crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        motion = self.session("live_portrait_motion_extractor")
        # Volume de traits (8 Mo) laissé sur le GPU entre l'extracteur et le générateur.
        fx = self.session("live_portrait_feature_extractor")
        io = fx.io_binding()
        io.bind_cpu_input("input", s_in)
        io.bind_output("feature_volume", "cuda")
        fx.run_with_iobinding(io)
        feat = io.get_outputs()[0]
        pitch, yaw, roll, scale, trans, s_expr, points = motion.run(None, {"input": s_in})
        rot = _euler_matrix(float(pitch), float(yaw), float(roll))
        t_expr = t_expr.copy()
        t_expr[:, EXPRESSION_KEEP] = s_expr[:, EXPRESSION_KEEP]
        if not self.p["expr_upper"]:
            t_expr[:, EXPRESSION_UPPER] = s_expr[:, EXPRESSION_UPPER]
        if not self.p["expr_lower"]:
            t_expr[:, EXPRESSION_LOWER] = s_expr[:, EXPRESSION_LOWER]
        t_expr = t_expr * factor + s_expr * (1 - factor)
        t_expr = np.clip(t_expr, EXPRESSION_MIN, EXPRESSION_MAX)
        t_pts = scale * (points @ rot.T + t_expr) + trans
        s_pts = scale * (points @ rot.T + s_expr) + trans
        gen = self.session("live_portrait_generator")
        io = gen.io_binding()
        io.bind_ortvalue_input("feature_volume", feat)
        io.bind_cpu_input("source", np.ascontiguousarray(t_pts, np.float32))
        io.bind_cpu_input("target", np.ascontiguousarray(s_pts, np.float32))
        io.bind_output("output")
        gen.run_with_iobinding(io)
        out = io.copy_outputs_to_cpu()[0][0]
        out = cv2.multiply(cv2.merge([out[2], out[1], out[0]]), 255.0, dtype=cv2.CV_8U)    # borné à 0…255
        size = temp_crop.shape[0]          # LivePortrait rend en 512 px : ramené au repère de travail
        if out.shape[0] == size:
            return out
        return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA if size < out.shape[0] else cv2.INTER_CUBIC)

    def _match_color(self, key, swapped, original, mask):
        """Couleur et lumière du visage généré accordées à la vraie peau (statistiques lissées)."""
        k = self.p["color"] / 100
        step = 2 * swapped.shape[0] // SIZE    # statistiques sur un pixel sur quatre (en 256) : bien assez
        sub = mask[::step, ::step] > 0.6
        if k <= 0 or sub.sum() * step * step < 200:
            return swapped
        lab = lambda im: cv2.cvtColor(np.ascontiguousarray(im[::step, ::step]), cv2.COLOR_BGR2LAB)[sub].astype(np.float32)
        ps, po = lab(swapped), lab(original)
        st = np.concatenate([ps.mean(0), ps.std(0), po.mean(0), po.std(0)])
        prev = self._color.get(key)
        st = st if prev is None else prev + 0.35 * (st - prev)
        self._color[key] = st
        ms, ss, mo, so = st[:3], st[3:6], st[6:9], st[9:]
        ratio = np.clip(so / np.maximum(ss, 1e-3), 0.8, 1.25)
        ratio[1:] = 1.0                        # contraste des couleurs : celui du visage généré
        # Même calcul pour chacune des 256 valeurs, appliqué en table : bien plus rapide.
        x = np.arange(256, dtype=np.float32)[:, None]
        lut = np.clip((x - ms) * (1 + (ratio - 1) * k) + ms + (mo - ms) * k, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.LUT(cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB), lut.reshape(1, 256, 3)), cv2.COLOR_LAB2BGR)

    def _match_grain(self, key, warped, roi, m):
        """Même grain et même netteté que l'image filmée (un visage trop lisse se remarque).
        Mesuré et corrigé dans l'image elle-même, après la mise en place du visage : le grain ajouté
        a la finesse de celui du capteur quelle que soit la taille du visage ou du pixel boost (ajouté
        avant, il était moyenné par le collage : sur une image bruitée, le visage n'en gardait que 40 %).
        Le bruit est estimé par la médiane des écarts (les contours des yeux, de la bouche ou des
        cils ne comptent pas comme du grain), puis lissé d'une image à l'autre."""
        k = self.p["grain"] / 100
        if k <= 0:
            return warped
        sel = m > 0.6
        if sel.sum() < 400:
            self._grain.pop(key, None)
            return warped

        def noise_of(im):
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            hp = cv2.subtract(g, cv2.GaussianBlur(g, (0, 0), 1.2), dtype=cv2.CV_32F)[sel]
            return float(np.median(np.abs(hp[::max(1, hp.size // 20000)]))) * 1.4826
        e_o, e_s = noise_of(roi), noise_of(warped)
        # > 0 : grain manquant (écart-type à ajouter) ; < 0 : visage plus net que l'image (part de flou).
        if e_o > e_s * 1.05:
            want = np.sqrt(e_o ** 2 - e_s ** 2) / 0.91      # 0,91 : part d'un bruit blanc gardée par la mesure
        elif e_s > e_o * 1.05:
            want = -(1 - e_o / e_s)
        else:
            want = 0.0
        prev = self._grain.get(key)
        want = want if prev is None else prev + 0.3 * (want - prev)
        self._grain[key] = want
        if want > 0.15:
            noise = np.empty(warped.shape[:2], np.float32)
            cv2.randn(noise, 0, want * k)
            return cv2.add(warped, cv2.merge([noise, noise, noise]), dtype=cv2.CV_8U)
        if want < -0.03:
            r = 1 + want * k
            return cv2.addWeighted(warped, r, cv2.GaussianBlur(warped, (0, 0), 1.2), 1 - r, 0)
        return warped

    @staticmethod
    def _beard(crop, face, hair, reach=30):
        """Expérimental. Barbe ou bouc de la personne filmée restés au bas du menton : le visage généré
        s'arrête au menton, et les poils juste en dessous restent visibles (surtout sous un visage sans
        barbe). Ne traite que ce qui tient dans le cadre du visage : ce qui descend sur le cou reste
        (le reconstruire de façon réaliste a été essayé, puis abandonné).
        Renvoie le masque (0…1, repère 256) de ce qui, sous le visage, est nettement plus sombre que
        la peau des joues et n'est pas de la chevelure : le visage généré y est étendu — il porte à
        cet endroit l'ombre normale du menton. Sans barbe, seule cette ombre est concernée."""
        g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
        inside = face > 0.5
        cheeks = inside[90:185]
        if cheeks.sum() < 400:
            return None
        ref = float(np.median(g[90:185][cheeks]))                   # peau des joues et du nez
        dark = np.clip((ref * 0.80 - g) / max(ref * 0.18, 1.0), 0, 1)   # 0 à 80 % de la peau, 1 à 62 %
        k = np.zeros((2 * reach + 1, 13), np.uint8)
        k[reach:] = 1                                             # le visage glissé vers le bas (et 6 px de côté)
        zone = cv2.dilate(inside.astype(np.uint8), k, anchor=(6, reach)).astype(np.float32)
        zone[:150] = 0                                            # seulement sous le nez
        zone *= 1 - np.clip(hair * 2, 0, 1)                       # pas dans les cheveux qui tombent près du cou
        return np.clip(cv2.GaussianBlur(dark * zone, (0, 0), 3) * 1.6, 0, 1)

    def _side(self, crop):
        """Dans un fil parallèle : analyseur de visage, barbe sous le menton et occlusions."""
        region = beard = None
        if self.p["parser"]:
            region, raw = self._region(crop)
            if raw is not None and self.p["beard"]:
                beard = self._beard(crop, raw[0], raw[1])
        occlusion = self._occlusion(crop) if self.p["occlusion"] else None
        return region, occlusion, beard

    # ------------------------------------------------------------ principal
    def process(self, frame: np.ndarray, pairs) -> np.ndarray:
        """pairs : [(visage source avec normed_embedding, visage cible avec kps)]."""
        return self._stage_b(frame, self._stage_a(frame, pairs))

    def _stage_a(self, frame, pairs):
        """Repères, cadrage, masques et expression lancés en parallèle, remplacement du visage."""
        now = time.perf_counter()
        jobs = []
        live = set()
        conf = []                  # confiance par visage pour la garde ; 0 : visage non remplacé
        kept = set(self._fixed)
        for src, tgt in pairs:
            conf.append(0.0)
            if src is None or getattr(tgt, "kps", None) is None:
                continue
            t0 = time.perf_counter()
            second = None
            fixed = tgt.get("fixed_kps")
            if fixed is not None:                  # Studio : repères déjà suivis et lissés sur toute la vidéo
                kps, track_id = np.asarray(fixed, np.float32), tgt["track_id"]
                kept.add(track_id)
            else:
                raw = np.asarray(tgt.kps, np.float32)
                if (self.p["landmarker"] == "hrffa" or self.p["guard_check"]) and getattr(tgt, "bbox", None) is not None:
                    s = self.ready_session("hrffa")    # repères du détecteur en attendant qu'il soit prêt
                    if s is not None:
                        detected, second = raw, self._hrffa_kps(frame, tgt.bbox, s, tgt)
                        if self.p["landmarker"] == "hrffa":
                            raw = second
                raw = self._profile_smooth(raw, now)
                kps, track_id = self._stable_kps(raw, now)
                live.add(track_id)
            # Pixel boost : le modèle passe total² fois sur des images entrelacées (plus de détails).
            inswapper = self.p["model"] == "inswapper" and self.inswapper is not None
            model_size = 128 if inswapper else SIZE
            total = max(1, int(self.p["boost"]))
            if total > 1 and fixed is None and self.p["boost_fit"]:
                total = self._fit_boost(track_id, kps, total)
            self.boost_now = total
            work = max(SIZE, model_size * total)
            M = cv2.estimateAffinePartial2D(kps, ARCFACE_128 * work, method=cv2.LMEDS)[0]
            if M is None:
                continue
            conf[-1] = self._agreement(detected, second, M, work) if second is not None and self.p["guard_check"] else 1.0
            relax = 0.0 if fixed is not None else self._relax(track_id, kps, M, work, now, conf[-1])
            big = cv2.warpAffine(frame, M, (work, work), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
            # Masques et expression : toujours dans le repère 256 des modèles.
            crop = big if work == SIZE else cv2.resize(big, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            side = self._pool.submit(self._side, crop)
            expr = self._pool.submit(self._expression_of, crop) if self.p["expression"] > 0 else None
            tm = {}
            emb = self._embedding(src, tgt, frame, kps, now)
            small = big if model_size * total == work else \
                cv2.resize(big, (model_size * total,) * 2, interpolation=cv2.INTER_AREA)
            if inswapper:
                tiles = [self._swap_inswapper(np.ascontiguousarray(t), emb) for t in
                         (_implode(small, total, model_size) if total > 1 else [small])]
                swapped = tiles[0] if total == 1 else _explode(np.stack(tiles), total, model_size)
                if swapped.shape[0] != work:
                    swapped = cv2.resize(swapped, (work, work), interpolation=cv2.INTER_CUBIC)
                smask = np.ones((work, work), np.float32)
            elif total == 1:
                swapped, smask = self._swap(small, emb)
            else:
                tiles = _implode(small, total, model_size)
                lot = self.batch_session()
                if lot is not None:
                    imgs, masks = self._swap_batch(lot, tiles, emb)
                else:
                    res = list(self._tiles.map(lambda t: self._swap(np.ascontiguousarray(t), emb), tiles))
                    imgs, masks = np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
                swapped, smask = _explode(imgs, total, model_size), _explode(masks, total, model_size)
            tm["swap"] = time.perf_counter()
            jobs.append(dict(t0=t0, tm=tm, track_id=track_id, work=work, M=M, big=big, crop=crop, side=side,
                             expr=expr, swapped=swapped, smask=smask, relax=relax,
                             second=second, kps=kps))
        # On oublie les visages qui ont disparu.
        self._tracks = [t for t in self._tracks if t[2] in live]
        self._fixed = kept
        alive = {t[2] for t in self._tracks} | kept
        self._motion = {k: v for k, v in self._motion.items() if k in alive}
        self._boost = {k: v for k, v in self._boost.items() if k in alive}
        with self._turn_cv:
            seq, self._seq = self._seq, self._seq + 1
        return dict(seq=seq, jobs=jobs, alive=alive, conf=conf)

    def _stage_b(self, frame, prep):
        """Expression, masques, couleur et grain, collage."""
        out = frame
        jobs = prep["jobs"]
        try:
            for j in jobs:
                tm, work, crop, swapped = j["tm"], j["work"], j["crop"], j["swapped"]
                if j["expr"] is not None:
                    before = swapped
                    swapped = self._restore(j["expr"].result(), swapped,
                                            np.interp(self.p["expression"], [0, 100], [0, 1.2]))
                    if work > SIZE:
                        swapped = self._keep_detail(before, swapped)
                tm["expression"] = time.perf_counter()
                region, occlusion, beard = j["side"].result()
                if occlusion is not None and self.p["occ_steady"] > 0:
                    self._wait_turn(prep["seq"])
                    occlusion = self._steady(j["track_id"], occlusion)
                if beard is not None:
                    # Lissé d'une image à l'autre : le bord du masque sous le menton ne clignote pas.
                    self._wait_turn(prep["seq"])
                    prev = self._chin.get(j["track_id"])
                    if prev is not None:
                        beard = cv2.addWeighted(prev, 0.5, beard, 0.5, 0)
                    self._chin[j["track_id"]] = beard
                if work != SIZE:
                    up = lambda m: None if m is None else cv2.resize(m, (work, work), interpolation=cv2.INTER_LINEAR)
                    region, occlusion, beard, crop = up(region), up(occlusion), up(beard), j["big"]
                tm["masks"] = time.perf_counter()
                # Masques : bord progressif × masque du modèle × visage seul × occlusions.
                box = _box_mask(work, self.p["mask_padding"], self.p["mask_blur"])
                shape = region if region is not None else j["smask"]
                if beard is not None and region is not None:
                    shape = np.maximum(shape, beard)
                    low = _box_mask_low(work, self.p["mask_padding"], self.p["mask_blur"])
                    box = box + (low - box) * np.minimum(beard * 4, 1)
                keep = occlusion
                if j["relax"] > 0:
                    shape = shape + (1 - shape) * j["relax"]
                    keep = None if occlusion is None else occlusion + (1 - occlusion) * j["relax"]
                mask = np.minimum(box, shape)
                if keep is not None:
                    mask = np.minimum(mask, keep)
                stat_mask = region if region is not None else mask
                self._wait_turn(prep["seq"])
                swapped = self._match_color(j["track_id"], swapped, crop, stat_mask)
                tm["match"] = time.perf_counter()
                out = self._paste(out if out is not frame else frame.copy(), swapped, mask, j["M"], j["track_id"])
                tm["paste"] = time.perf_counter()
                prev = j["t0"]
                for k, v in tm.items():
                    self.ms[k] = 0.8 * self.ms.get(k, 0) + 0.2 * (v - prev) * 1000
                    prev = v
        finally:
            self._wait_turn(prep["seq"])
            self._color = {k: v for k, v in self._color.items() if k in prep["alive"]}
            self._occ = {k: v for k, v in self._occ.items() if k in prep["alive"]}
            self._grain = {k: v for k, v in self._grain.items() if k in prep["alive"]}
            self._chin = {k: v for k, v in self._chin.items() if k in prep["alive"]}
            with self._turn_cv:
                self._turn = max(self._turn, prep["seq"] + 1)
                self._turn_cv.notify_all()
        return out

    def _wait_turn(self, seq):
        """Les couleurs sont lissées d'une image à l'autre : dans l'ordre des images (au plus 1 s
        d'attente, au cas où une image n'aurait jamais été rendue)."""
        with self._turn_cv:
            self._turn_cv.wait_for(lambda: self._turn >= seq, timeout=1.0)

    def _paste(self, frame, crop, mask, M, key=None):
        """Visage remis à sa place dans l'image, grain accordé (voir _match_grain), collage progressif."""
        h, w = frame.shape[:2]
        n = crop.shape[0]
        IM = cv2.invertAffineTransform(M)
        corners = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
        pts = corners @ IM[:, :2].T + IM[:, 2]
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return frame
        IMs = IM.copy()
        IMs[:, 2] -= (x0, y0)
        size = (x1 - x0, y1 - y0)
        warped = cv2.warpAffine(crop, IMs, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(mask, IMs, size, flags=cv2.INTER_LINEAR)
        roi = frame[y0:y1, x0:x1]
        if key is not None:
            warped = self._match_grain(key, warped, roi, m)
        frame[y0:y1, x0:x1] = cv2.blendLinear(warped, roi, m, cv2.subtract(1.0, m))
        return frame
