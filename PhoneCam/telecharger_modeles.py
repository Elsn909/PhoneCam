"""Télécharge tous les modèles qui manquent, depuis leurs sources officielles (Installer.bat le lance).
Taille ou CRC32 vérifiés ; relancer reprend ce qui manque. À lancer avec ~ = data\\home (voir Installer.bat)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))

import models_plus  # noqa: E402
from modules import model_downloader as dlc  # noqa: E402

# Deep-Live-Cam (huggingface.co/hacksider/deep-live-cam) : remplacement, améliorateurs.
APP = ["inswapper_128_fp16.onnx", "gfpgan-1024.onnx", "GPEN-BFR-256.onnx", "GPEN-BFR-512.onnx"]


def main():
    bad = []
    for name in APP:
        if dlc.ensure_model(name) is None:
            bad.append(name)
    # Détection des visages (insightface buffalo_l) dans ~\.insightface.
    if not dlc.ensure_insightface_pack("buffalo_l"):
        bad.append("buffalo_l")
    # Moteur Plus (FaceFusion), y compris les modèles facultatifs : tout marche ensuite hors ligne.
    try:
        models_plus.download()
        for name in models_plus.OPTIONAL:
            models_plus.download_optional(name)
        models_plus.prepare_labels()
    except Exception as e:  # noqa: BLE001
        print(f"Moteur Plus : {e}")
        bad.append("models-plus")
    if bad:
        print("ÉCHEC pour : " + ", ".join(bad) + ". Relance Installer.bat pour réessayer.")
        sys.exit(1)
    print("Tous les modèles sont là.")


if __name__ == "__main__":
    main()
