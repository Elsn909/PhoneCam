"""Silhouette de la personne filmée (MODNet, détourage de portrait) : les filtres de forme ne
déplacent qu'elle, le décor derrière (mur, porte, étagère) reste droit.

Chargé en arrière-plan à la première demande ; tant qu'il n'est pas prêt, mask() rend None
et les filtres déforment comme avant.
"""
import os
import threading
import time

import cv2
import numpy as np

SIZE = 512


class PersonSegmenter:
    def __init__(self, models_dir: str):
        self.path = os.path.join(models_dir, "modnet.onnx")
        self.status = ""
        self.ms = 0.0
        self._session = None
        self._loading = False
        self._last = None

    def available(self) -> bool:
        return os.path.isfile(self.path)

    def _load(self):
        try:
            import onnxruntime as ort  # après engine.py, qui rend visibles les DLL CUDA / TensorRT
            self.status = "Préparation du détourage…"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "TensorrtExecutionProvider" in ort.get_available_providers():
                cache = os.path.join(os.path.dirname(self.path), "trt-cache")
                shape = f"input:1x3x{SIZE}x{SIZE}"
                providers.insert(0, ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
                    "trt_timing_cache_enable": True, "trt_timing_cache_path": cache,
                    "trt_profile_min_shapes": shape, "trt_profile_max_shapes": shape,
                    "trt_profile_opt_shapes": shape}))
            s = ort.InferenceSession(self.path, providers=providers)
            s.run(None, {"input": np.zeros((1, 3, SIZE, SIZE), np.float32)})
            self._session = s
            self.status = "Détourage prêt."
        except Exception as e:
            self.status = f"Détourage indisponible : {e}"
        finally:
            self._loading = False

    def mask(self, frame: np.ndarray):
        """Probabilité « personne » (0…1, float32), au quart de la taille de l'image ; None si pas prêt."""
        if self._session is None:
            if not self._loading and self.available() and not self.status.startswith("Détourage indisponible"):
                self._loading = True
                threading.Thread(target=self._load, daemon=True, name="modnet-load").start()
            return None
        t0 = time.perf_counter()
        # Filtres de forme et éclairage le demandent pour la même image : résultat réutilisé.
        if self._last is not None and self._last[0] == frame.shape and t0 - self._last[1] < 0.025:
            return self._last[2]
        x = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32)
        x = np.ascontiguousarray(((x - 127.5) / 127.5).transpose(2, 0, 1)[None])
        m = self._session.run(None, {"input": x})[0][0, 0]
        # Au quart de la taille : bien assez pour guider une déformation lisse.
        m = cv2.resize(m, (frame.shape[1] // 4, frame.shape[0] // 4), interpolation=cv2.INTER_AREA)
        self.ms = (time.perf_counter() - t0) * 1000
        m = np.clip(m, 0, 1)
        self._last = (frame.shape, time.perf_counter(), m)
        return m
