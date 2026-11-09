"""Modèles InsightFace (détecteur, repères 106 et 3D) passés sur TensorRT, comme ceux du moteur
Plus (mesuré seul : détecteur 4,5 → 2,8 ms, repères 106 0,8 → 0,5 ms, repères 3D 2,3 → 1,9 ms).

Ils restent en FP32 (même précision que CUDA, seule la façon de calculer change) : écart de
l'ordre du millième sur les sorties, invisible à l'image (vérifié image par image). La
reconnaissance (empreinte du visage source) n'est pas touchée.

Le moteur TensorRT se prépare en arrière-plan (la 1re fois : jusqu'à une minute, puis mis en cache
dans data\\models-plus\\trt-cache) ; en attendant, les modèles tournent comme avant sur CUDA.
"""
import os
import threading

import numpy as np

from models_plus import MODELS_DIR

CACHE = os.path.join(MODELS_DIR, "trt-cache")
_pending = set()
_lock = threading.Lock()


def available() -> bool:
    import onnxruntime as ort
    return "TensorrtExecutionProvider" in ort.get_available_providers()


def providers(input_name: str, shape, fp16: bool = False, sub: str = ""):
    """Fournisseurs pour une entrée de taille fixe : TensorRT, puis CUDA en secours.

    sub : sous-dossier du cache, pour un modèle préparé à plusieurs tailles. Le fichier du cache
    porte le nom du modèle, pas sa taille : sans ça, chaque taille écrase la précédente et doit
    être refaite au démarrage suivant."""
    base = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if not available():
        return base
    engines = os.path.join(CACHE, sub) if sub else CACHE
    os.makedirs(engines, exist_ok=True)
    sh = f"{input_name}:{'x'.join(map(str, shape))}"
    return [("TensorrtExecutionProvider", {
        "trt_fp16_enable": fp16, "trt_engine_cache_enable": True, "trt_engine_cache_path": engines,
        "trt_timing_cache_enable": True, "trt_timing_cache_path": CACHE,
        "trt_profile_min_shapes": sh, "trt_profile_max_shapes": sh, "trt_profile_opt_shapes": sh})] + base


def upgrade(model, shape, name: str) -> None:
    """Remplace model.session (modèle InsightFace) par une session TensorRT, en arrière-plan.
    La session en place sert jusqu'à ce que la nouvelle soit prête et préchauffée."""
    if model is None or not available() or getattr(model, "_trt", False):
        return
    path = getattr(model, "model_file", None)
    if path is None or not os.path.isfile(path):
        return
    inp = model.session.get_inputs()[0].name

    def work():
        try:
            import onnxruntime as ort
            sub = f"{name}-{shape[3]}x{shape[2]}" if name == "detection" else ""
            s = ort.InferenceSession(path, providers=providers(inp, shape, sub=sub))
            s.run(None, {inp: np.zeros(shape, np.float32)})
            model.session, model._trt = s, True
        except Exception:  # noqa: BLE001 — on garde la session CUDA
            pass
        finally:
            with _lock:
                _pending.discard(name)

    with _lock:
        _pending.add(name)
    threading.Thread(target=work, daemon=True, name="trt-" + name).start()


def upgrade_analyser(fa, det_size) -> None:
    """Détecteur (à la taille de détection choisie) et repères 106 de l'analyseur InsightFace."""
    upgrade(fa.det_model, (1, 3, det_size[1], det_size[0]), "detection")
    upgrade(fa.models.get("landmark_2d_106"), (1, 3, 192, 192), "landmark_2d_106")


def pending() -> bool:
    """Une session TensorRT est-elle encore en préparation ?"""
    with _lock:
        return bool(_pending)
