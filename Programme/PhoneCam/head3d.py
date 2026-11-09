"""Repères 3D du visage (68 points avec la profondeur, modèle « 1k3d68 » d'InsightFace, déjà
fourni avec Deep-Live-Cam) : orientation de la tête pour des filtres de forme justes de profil.

Chargé en arrière-plan à la première demande ; tant qu'il n'est pas prêt, add() ne fait rien
et les filtres utilisent les repères 2D seuls.
"""
import os
import threading
import time

import numpy as np


class Head3D:
    def __init__(self):
        self.path = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l", "1k3d68.onnx")
        self.status = ""
        self.ms = 0.0
        self._model = None
        self._loading = False

    def available(self) -> bool:
        return os.path.isfile(self.path)

    def _load(self):
        try:
            from insightface.model_zoo import get_model  # déjà importé par le moteur
            import trt_insight
            # TensorRT (FP32) : même précision, un peu moins de temps de carte graphique.
            m = get_model(self.path, providers=trt_insight.providers("data", (1, 3, 192, 192)))
            m.prepare(0)
            m.session.run(None, {"data": np.zeros((1, 3, 192, 192), np.float32)})  # 1re préparation ici
            self._model = m
            self.status = "Repères 3D prêts."
        except Exception as e:
            self.status = f"Repères 3D indisponibles : {e}"
        finally:
            self._loading = False

    def add(self, frame, faces) -> None:
        """Ajoute face.landmark_3d_68 (x, y en pixels, z) à chaque visage."""
        if self._model is None:
            if not self._loading and self.available() and not self.status.startswith("Repères 3D indispo"):
                self._loading = True
                threading.Thread(target=self._load, daemon=True, name="head3d-load").start()
            return
        t0 = time.perf_counter()
        for f in faces:
            if getattr(f, "bbox", None) is not None and f.get("landmark_3d_68") is None:
                self._model.get(frame, f)
        self.ms = (time.perf_counter() - t0) * 1000
