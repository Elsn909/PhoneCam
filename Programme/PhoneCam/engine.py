"""Appelle directement le moteur de Deep-Live-Cam (même logique que son aperçu live, toutes options)."""
import os
import sys
import threading
import time

import cv2
import numpy as np

from tour_gpu import video_turn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT, "app")


def _register_cuda_dlls() -> None:
    # Même chose que app/run.py : onnxruntime-gpu trouve cuDNN/cuBLAS dans torch/lib.
    sp = os.path.join(sys.prefix, "Lib", "site-packages")
    # tensorrt_libs : moteur « Plus » (modèles optimisés pour la carte graphique), s'il est installé.
    dirs = [os.path.join(sp, "torch", "lib"), os.path.join(sp, "tensorrt_libs")]
    nvidia = os.path.join(sp, "nvidia")
    if os.path.isdir(nvidia):
        dirs += [os.path.join(nvidia, p, "bin") for p in os.listdir(nvidia)]
    for d in dirs:
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            os.add_dll_directory(d)


_register_cuda_dlls()
sys.path.insert(0, APP_DIR)

import modules.globals as G  # noqa: E402

G.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
G.frame_processors = ["face_swapper"]
G.headless = True
G.enable_interpolation = False

from modules.cluster_analysis import find_closest_centroid  # noqa: E402
from modules.face_analyser import (  # noqa: E402
    detect_many_faces_fast, detect_one_face_fast, ensure_landmarks,
    get_face_analyser, get_many_faces, get_one_face, reset_face_analyser,
)
from modules.processors.frame import face_swapper  # noqa: E402

import models_plus  # noqa: E402
import trt_insight  # noqa: E402
import modules.face_analyser as _face_analyser  # noqa: E402
from swap_plus import DEFAULTS as PLUS_DEFAULTS, SwapPlus  # noqa: E402

_optimize_det_model = _face_analyser._optimize_det_model


def _optimize_with_trt(fa, providers, det_size):
    """À chaque création de l'analyseur InsightFace (y compris quand la taille de détection
    change) : détecteur et repères 106 passés sur TensorRT, en arrière-plan (trt_insight)."""
    _optimize_det_model(fa, providers, det_size)
    trt_insight.upgrade_analyser(fa, det_size)


_face_analyser._optimize_det_model = _optimize_with_trt

# Réglages qui correspondent directement à modules.globals (nom : valeur par défaut).
GLOBAL_OPTIONS = {
    "opacity": 1.0,
    "sharpness": 0.0,
    "mouth_mask": False,
    "mouth_mask_size": 0.0,
    "show_mouth_mask_box": False,
    "mask_feather_ratio": 12,
    "mask_down_size": 0.1,
    "mask_size": 1.0,
    "poisson_blend": False,
    "many_faces": False,
    "enable_interpolation": False,
    "interpolation_weight": 0.0,
}

ENHANCERS = {  # clé interface : module Deep-Live-Cam
    "face_enhancer": "face_enhancer",                # GFPGAN 1024
    "face_enhancer_gpen256": "face_enhancer_gpen256",
    "face_enhancer_gpen512": "face_enhancer_gpen512",
}


def read_image(path: str):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


# Repères calculés sur une image donnée : à refaire sur la suivante.
PER_FRAME_KEYS = ("landmark_hrffa_68", "landmark_3d_68", "landmark_2d_106")


def _renew(found):
    """Visages repris de l'image précédente (détection une image sur deux, visage manqué) : copies
    sans les repères propres à cette image-là, pour qu'ils soient recalculés sur la nouvelle."""
    from insightface.app.common import Face
    faces, pairs = found
    new = {id(f): Face({k: v for k, v in f.items() if k not in PER_FRAME_KEYS}) for f in faces}
    return ([new[id(f)] for f in faces],
            [(s, new.get(id(t)) or Face({k: v for k, v in t.items() if k not in PER_FRAME_KEYS}))
             for s, t in pairs])


class Engine:
    def __init__(self):
        self.enabled = True
        self.mirror = False
        self.color_correction = False   # « Fix Blueish Cam » dans Deep-Live-Cam
        self.det_interval = 1           # détection toutes les N images (1 = le plus précis)
        self.enhancers = {k: False for k in ENHANCERS}
        self.map_faces = False
        self.map_threshold = 0.3        # similarité minimale pour reconnaître une personne
        self.status = "Chargement du modèle…"
        self.ms = 0.0
        self._source_face = None
        self._pairs = []                # [(Face source, empreinte de la cible)]
        self._jobs: list = []
        self._lock = threading.Lock()
        self._classic_lock = threading.Lock()
        self._modules = {}
        self._count = 0
        self._faces = None
        self._missed = 0                   # images de suite sans visage trouvé
        self.faces_out = []             # visages de la dernière image (pour les filtres de forme)
        self.swap_info = (False, [])    # (swap attendu, scores de détection des visages remplacés)
        # Moteur « Plus » (swap_plus.py) : chargé en arrière-plan, le flux continue en attendant.
        self.plus_enabled = False
        self.parallel_render = True     # moteur Plus : deux images en chevauchement (voir parallel())
        self.ai_fps = 0                 # images traitées par seconde au plus (0 : toutes ; voir Pipeline)
        self.plus_p = dict(PLUS_DEFAULTS)
        self.plus_status = ""
        self._plus = None
        self._plus_loading = False
        for k, v in GLOBAL_OPTIONS.items():
            setattr(G, k, v)

    # --- API pour l'interface : tout est appliqué dans le thread de traitement
    def submit(self, fn) -> None:
        with self._lock:
            self._jobs.append(fn)

    def set_option(self, name: str, value) -> None:
        if name in GLOBAL_OPTIONS:
            setattr(G, name, value)
        elif name == "det_size":
            if int(value) != int(G.det_size) or _face_analyser.FACE_ANALYSER is None:
                self.submit(lambda: self._set_det_size(int(value)))
        elif name.startswith("plus_"):
            self._set_plus(name[5:], value)
        elif name in self.enhancers:
            if value:
                self.submit(lambda: self._load_enhancer(name))
            else:
                self.enhancers[name] = False
        else:
            setattr(self, name, value)

    def set_source(self, path: str, more=()) -> None:
        """more : d'autres photos de la même personne ; leurs empreintes sont moyennées."""
        more = list(more)
        self.submit(lambda: self._load_source(path, more))

    def set_pairs(self, pairs: list) -> None:
        """pairs : [(image source, image de référence de la personne à remplacer)]."""
        self.submit(lambda: self._load_pairs(pairs))

    # --- moteur « Plus »
    def _set_plus(self, key, value) -> None:
        if key == "enabled":
            self.plus_enabled = bool(value)
        elif key in self.plus_p:
            self.plus_p[key] = value
            if self._plus is not None:
                if key == "model" and value != self._plus.p["model"]:
                    self._plus_load()      # nouveau modèle préparé à part, puis échangé
                    return
                self._plus.p[key] = value
        if self.plus_enabled and self._plus is None:
            self._plus_load()

    def _plus_load(self) -> None:
        if self._plus_loading:
            return
        self._plus_loading = True
        threading.Thread(target=self._plus_worker, daemon=True, name="plus-load").start()

    def _plus_worker(self) -> None:
        try:
            while True:
                want = dict(self.plus_p)
                if models_plus.missing():
                    models_plus.download(log=lambda m: setattr(self, "plus_status", m))
                models_plus.prepare_parser(log=lambda m: setattr(self, "plus_status", m))
                models_plus.prepare_labels(log=lambda m: setattr(self, "plus_status", m))
                self.plus_status = "Préparation du moteur Plus (la 1re fois : jusqu'à quelques minutes)…"
                sp = SwapPlus(models_plus.MODELS_DIR)
                if self._plus is not None:     # masques et LivePortrait déjà chargés : repris tels quels
                    sp._sessions.update({k: v for k, v in self._plus._sessions.items() if not k.startswith("hyperswap")})
                sp.p.update(want)
                if want["model"] == "inswapper":
                    sp.inswapper = face_swapper.get_face_swapper()
                sp.recognizer = get_face_analyser().models.get("recognition")
                sp.warmup()
                if dict(self.plus_p)["model"] == want["model"]:
                    sp.p.update(self.plus_p)
                    self._plus = sp            # l'ancien peut finir son image en cours, puis est libéré
                    self.plus_status = "Moteur Plus prêt."
                    return
                sp.close()                 # le modèle a changé pendant la préparation : on recommence
        except Exception as e:
            self.plus_status = f"Moteur Plus indisponible : {e}"
            self.plus_enabled = False
        finally:
            self._plus_loading = False

    # --- exécuté dans le thread de traitement
    def load(self) -> bool:
        ok = face_swapper.pre_start()
        if ok:
            get_face_analyser()
            # Préchauffage : la première image ne provoque pas de pause de compilation.
            detect_one_face_fast(np.zeros((720, 1280, 3), np.uint8))
        self.status = "Modèle chargé. Choisis un visage source." if ok else "Échec du chargement du modèle."
        return ok

    @staticmethod
    def trt_pending() -> bool:
        """Un modèle InsightFace est-il encore en préparation pour TensorRT ?"""
        return trt_insight.pending()

    def _set_det_size(self, size: int) -> None:
        G.det_size = size
        reset_face_analyser()
        get_face_analyser()
        self._faces = None

    def _load_enhancer(self, name: str) -> None:
        if name not in self._modules:
            self.status = f"Chargement de {name}…"
            import importlib
            mod = importlib.import_module(f"modules.processors.frame.{ENHANCERS[name]}")
            if not mod.pre_check():  # télécharge le modèle s'il manque (taille vérifiée)
                self.status = f"{name} : modèle indisponible."
                return
            self._modules[name] = mod
            self.status = f"{name} activé."
        self.enhancers[name] = True

    def _load_source(self, path: str, more=()) -> None:
        img = read_image(path)
        face = get_one_face(img) if img is not None else None
        if face is None:
            self.status = f"Aucun visage trouvé dans {os.path.basename(path)}."
            return
        # Plusieurs photos de la même personne : empreinte moyenne, plus fidèle et plus stable qu'une
        # seule photo (angle, lumière, expression particuliers). Une photo d'une autre personne
        # (empreinte trop différente) est écartée.
        embs, skipped = [face.normed_embedding], 0
        for p in more:
            im = read_image(p) if os.path.isfile(p) else None
            f = get_one_face(im) if im is not None else None
            if f is None or float(np.dot(f.normed_embedding, face.normed_embedding)) < 0.35:
                skipped += 1
                continue
            embs.append(f.normed_embedding)
        if len(embs) > 1:
            mean = np.mean(embs, 0)
            norm = float(np.linalg.norm(face["embedding"]))
            face = type(face)(face)
            # normed_embedding est calculé à partir d'embedding (InsightFace) : on remplace celui-ci.
            face.embedding = (mean / np.linalg.norm(mean) * norm).astype(np.float32)
        self._source_face = face
        self.status = (f"Visage source : {os.path.basename(path)}"
                       + (f" + {len(embs) - 1} photo(s)" if len(embs) > 1 else "")
                       + (f" ({skipped} écartée(s) : autre personne ou pas de visage)" if skipped else ""))

    def _load_pairs(self, pairs: list) -> None:
        loaded = []
        for src_path, ref_path in pairs:
            src, ref = read_image(src_path), read_image(ref_path)
            sf = get_one_face(src) if src is not None else None
            rf = get_one_face(ref) if ref is not None else None
            if sf is None or rf is None:
                self.status = f"Visage introuvable : {os.path.basename(src_path if sf is None else ref_path)}"
                continue
            loaded.append((sf, rf.normed_embedding))
        self._pairs = loaded
        self.status = f"{len(loaded)} paire(s) de visages chargée(s)."

    def _run_jobs(self) -> None:
        with self._lock:
            jobs, self._jobs = self._jobs, []
        for job in jobs:
            try:
                job()
            except Exception as e:
                self.status = f"Erreur : {e}"

    def _detect(self, frame):
        if self.map_faces:
            # Analyse complète (empreintes) pour reconnaître chaque personne.
            faces = get_many_faces(frame) or []
            pairs = []
            if self._pairs:
                refs = [emb for _, emb in self._pairs]
                for f in faces:
                    if f.normed_embedding is not None:
                        idx, ref = find_closest_centroid(refs, f.normed_embedding)
                        # Une personne qui n'est pas dans la liste n'est pas remplacée.
                        if float(np.dot(ref, f.normed_embedding)) >= self.map_threshold:
                            pairs.append((self._pairs[idx][0], f))
            return faces, pairs
        if G.many_faces:
            faces = detect_many_faces_fast(frame) or []
        else:
            face = detect_one_face_fast(frame)
            faces = [face] if face is not None else []
        return faces, [(self._source_face, f) for f in faces]

    def process(self, frame: np.ndarray, want_faces: bool = False) -> np.ndarray:
        """want_faces : détecter les visages (avec repères 106 points) même sans face swap,
        pour les filtres de forme ; ils sont alors laissés dans self.faces_out."""
        ctx = self.begin(frame, want_faces)
        return self.end(ctx, self.render(ctx))

    # Le traitement d'une image en trois temps, pour en traiter deux à la fois avec le moteur
    # Plus (voir Pipeline) : begin() et end() dans l'ordre des images, render() en parallèle.
    def parallel(self) -> bool:
        """render() traite-t-il deux images en même temps ? Moteur Plus seulement (le collage de
        Deep-Live-Cam garde des tampons partagés) : son travail est découpé en deux étapes
        (swap_plus.SwapPlus.prepare / render), le résultat de chaque image est le même. Mesuré
        avec un moteur à ~40 ms par image : 26 → 29 img/s, images sautées 24 → 2 sur 250,
        latence 148 → 112 ms (les images ne s'accumulent plus en attente)."""
        return self.parallel_render and self.plus_enabled and self._plus is not None

    def begin(self, frame: np.ndarray, want_faces: bool = False, landmarks_106: bool = True) -> dict:
        """landmarks_106 : False quand les filtres de forme prennent les repères HRFFA (les 106
        repères d'InsightFace ne leur servent alors plus)."""
        self._run_jobs()
        video_turn()            # la voix transformée passe d'abord (voir tour_gpu.py)
        ctx = {"t0": time.perf_counter(), "frame": frame, "faces": [], "pairs": [], "info": (False, []),
               "plus": None, "jobs": None}
        if self.color_correction:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        ctx["frame"] = frame
        has_source = self._pairs if self.map_faces else self._source_face is not None
        swap = self.enabled and has_source
        if not swap and not want_faces:
            return ctx

        self._count += 1
        fresh = False
        if self._faces is None or self._count % max(1, self.det_interval) == 0:
            found = self._detect(frame)
            # Visage manqué une image ou deux (souvent de profil) : on garde le précédent plutôt
            # que de laisser passer le vrai visage le temps d'une image.
            if (not found[1] and self._faces is not None and self._faces[1] and self.plus_enabled
                    and self.plus_p.get("hold", True) and self._missed < 2):
                self._missed += 1
            else:
                self._missed = 0
                self._faces = found
                fresh = True
        faces, pairs = self._faces if fresh else _renew(self._faces)
        if want_faces:
            if landmarks_106:
                ensure_landmarks(frame, faces)
            ctx["faces"] = faces
        pairs = [(s, t) for s, t in pairs if s is not None] if swap else []
        # Pour la garde anti-fuite : un swap était attendu, et sur quels visages il a eu lieu.
        ctx["info"] = (bool(swap), [float(getattr(t, "det_score", None) or 1.0) for _, t in pairs])
        ctx["pairs"] = pairs
        ctx["all_faces"] = faces
        if not pairs:
            return ctx
        plus = self._plus if self.plus_enabled else None
        if plus is not None:
            ctx["plus"] = plus
            ctx["jobs"] = plus.prepare(frame, pairs)
        elif G.mouth_mask:
            ensure_landmarks(frame, faces)
        return ctx

    def render(self, ctx: dict) -> np.ndarray:
        frame, pairs, plus = ctx["frame"], ctx["pairs"], ctx["plus"]
        if not pairs:
            return frame
        video_turn()
        if plus is not None:
            out, conf = plus.render(frame, ctx["jobs"])
            # Confiance pour la garde : celle du détecteur, abaissée si le visage n'est pas
            # vraiment visible (tête de dos, points placés dans les cheveux).
            ctx["info"] = (True, [min(d, c) for d, c in zip(ctx["info"][1], conf)])
            return out
        with self._classic_lock:
            out = frame
            for src, target in pairs:
                out = face_swapper.swap_face(src, target, out)
        return out

    def end(self, ctx: dict, out: np.ndarray) -> np.ndarray:
        pairs = ctx["pairs"]
        if not pairs:
            # Lissage temporel de Deep-Live-Cam : sans visage cette image, l'image gardée en
            # mémoire devient périmée ; au retour du visage elle serait mélangée (image fantôme).
            face_swapper.PREVIOUS_FRAME_RESULT = None
        if pairs:
            out = face_swapper.apply_post_processing(out, [t.bbox.astype(int) for _, t in pairs])
            faces = ctx.get("all_faces", [])
            for name, on in self.enhancers.items():
                if on and name in self._modules:
                    video_turn()
                    mod = self._modules[name]
                    if name == "face_enhancer":
                        out = mod.process_frame(None, out, detected_faces=faces)
                    else:  # les modules GPEN ne traitent que le premier visage fourni
                        for f in faces:
                            out = mod.process_frame(None, out, detected_faces=[f])
        self.faces_out, self.swap_info = ctx["faces"], ctx["info"]
        self.ms = (time.perf_counter() - ctx["t0"]) * 1000
        return out
