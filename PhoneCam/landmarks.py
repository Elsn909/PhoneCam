"""Repères HRFFA (FaceFusion 3.9) pour les filtres de forme : 68 points de toute la tête, prévus pour
rester justes jusqu'au profil complet. Modèle data\\models-plus\\hrffa.onnx (téléchargé au 1er choix).

Chargé en arrière-plan à la première demande ; tant qu'il n'est pas prêt, get() renvoie None et
les filtres gardent les repères InsightFace.
"""
import os
import threading
import time

import cv2
import numpy as np

from models_plus import MODELS_DIR

SIZE = 256


def hrffa_68(session, frame, bbox):
    """68 repères (iBUG) du visage dans bbox, en pixels de l'image (préparation de FaceFusion)."""
    bbox = np.asarray(bbox, np.float32)
    scale = SIZE / max(1.0, float((bbox[2:] - bbox[:2]).max()) * 1.7)
    tx, ty = (SIZE - (bbox[2:] + bbox[:2]) * scale) * 0.5
    M = np.array([[scale, 0, tx], [0, scale, ty]], np.float32)
    crop = cv2.warpAffine(frame, M, (SIZE, SIZE))
    x = np.ascontiguousarray(((crop[:, :, ::-1].astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None])
    pts = session.run(None, {"input": x})[0].reshape(-1, 2) * SIZE
    return ((pts - (tx, ty)) / scale).astype(np.float32)


def to_5(pts68):
    """Centre de chaque œil, bout du nez, coins de la bouche (« 5/68 » de FaceFusion)."""
    return np.array([pts68[36:42].mean(0), pts68[42:48].mean(0), pts68[30], pts68[48], pts68[54]], np.float32)


class Hrffa:
    def __init__(self):
        self.path = os.path.join(MODELS_DIR, "hrffa.onnx")
        self.status = ""
        self.ms = 0.0
        self._session = None
        self._loading = False

    def _load(self):
        try:
            import onnxruntime as ort
            import models_plus
            if not os.path.isfile(self.path):
                self.status = "Téléchargement de HRFFA (~36 Mo)…"
                models_plus.download_optional("hrffa", log=lambda m: setattr(self, "status", m))
            self.status = "Préparation de HRFFA (la 1re fois : jusqu'à une minute)…"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "TensorrtExecutionProvider" in ort.get_available_providers():
                cache = os.path.join(MODELS_DIR, "trt-cache")
                os.makedirs(cache, exist_ok=True)
                shape = f"input:1x3x{SIZE}x{SIZE}"
                providers.insert(0, ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
                    "trt_timing_cache_enable": True, "trt_timing_cache_path": cache,
                    "trt_profile_min_shapes": shape, "trt_profile_max_shapes": shape,
                    "trt_profile_opt_shapes": shape}))
            self._session = ort.InferenceSession(self.path, providers=providers)
            self.status = "Repères HRFFA prêts."
        except Exception as e:
            self.status = f"HRFFA indisponible : {e}"
        finally:
            self._loading = False

    def ready(self) -> bool:
        return self._session is not None

    def add(self, frame, faces) -> None:
        """Ajoute face["landmark_hrffa_68"] (68 × 2, pixels) à chaque visage."""
        if self._session is None:
            if not self._loading and not self.status.startswith("HRFFA indispo"):
                self._loading = True
                threading.Thread(target=self._load, daemon=True, name="hrffa-load").start()
            return
        t0 = time.perf_counter()
        for f in faces:
            if getattr(f, "bbox", None) is not None and f.get("landmark_hrffa_68") is None:
                f["landmark_hrffa_68"] = hrffa_68(self._session, frame, f.bbox)
        self.ms = (time.perf_counter() - t0) * 1000
