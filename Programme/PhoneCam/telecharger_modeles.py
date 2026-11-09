"""Télécharge tous les modèles qui manquent, depuis leurs sources officielles (Installer.bat le lance).
Taille ou CRC32 vérifiés ; relancer reprend ce qui manque. À lancer avec ~ = data\\home (voir Installer.bat).
Prépare aussi les voix fournies (dossier « Voix » à côté du lanceur) : GitHub refuse les fichiers de plus
de 100 Mo, les gros .index y sont donc coupés en morceaux (.001, .002…), recollés ici."""
import glob
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))

# Deep-Live-Cam (huggingface.co/hacksider/deep-live-cam) : remplacement, améliorateurs.
APP = ["inswapper_128_fp16.onnx", "gfpgan-1024.onnx", "GPEN-BFR-256.onnx", "GPEN-BFR-512.onnx"]

# Changeur de voix (RVC) : contenu (HuBERT / ContentVec), hauteur (RMVPE), détection de voix (Silero),
# débruitage (GTCRN). (fichier dans data\models-voix, adresse, taille en octets)
RVC = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
VOICE = [
    ("hubert_base/config.json", RVC + "hubert_base/config.json", 1492),
    ("hubert_base/preprocessor_config.json", RVC + "hubert_base/preprocessor_config.json", 225),
    ("hubert_base/pytorch_model.bin", RVC + "hubert_base/pytorch_model.bin", 189206711),
    ("rmvpe.pt", RVC + "rmvpe.pt", 181184272),
    ("silero_vad.onnx", "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx", 2327524),
    ("gtcrn_simple.onnx", "https://github.com/Xiaobin-Rong/gtcrn/raw/main/stream/onnx_models/gtcrn_simple.onnx", 535190),
    ("LICENSE-silero.txt", "https://raw.githubusercontent.com/snakers4/silero-vad/master/LICENSE", None),
    ("LICENSE-gtcrn.txt", "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main/LICENSE", None),
]


def fetch(url, path, size):
    if os.path.isfile(path) and (size is None or os.path.getsize(path) == size):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    print(f"  {os.path.basename(path)}…", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhoneCam-installer"})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if size is not None and os.path.getsize(part) != size:
            print(f"  taille inattendue pour {os.path.basename(path)}")
            os.remove(part)
            return False
        os.replace(part, path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  {os.path.basename(path)} : {e}")
        return False


def voice_models():
    dest = os.path.join(ROOT, "data", "models-voix")
    return [name for name, url, size in VOICE if not fetch(url, os.path.join(dest, *name.split("/")), size)]


def join_voices():
    """Recolle les morceaux des voix fournies (Voix\\…\\x.index.001, .002…) en un seul fichier."""
    top = os.path.join(os.path.dirname(ROOT), "Voix")
    for first in glob.glob(os.path.join(top, "**", "*.001"), recursive=True):
        base = first[:-4]
        parts = sorted(glob.glob(glob.escape(base) + ".[0-9][0-9][0-9]"))
        tmp = base + ".part"
        with open(tmp, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 20)
        os.replace(tmp, base)
        for p in parts:
            os.remove(p)
        print(f"  voix prête : {os.path.relpath(base, top)}")


def main():
    import models_plus
    from modules import model_downloader as dlc
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
    print("Changeur de voix :")
    bad += voice_models()
    join_voices()
    if bad:
        print("ÉCHEC pour : " + ", ".join(bad) + ". Relance Installer.bat pour réessayer.")
        sys.exit(1)
    print("Tous les modèles sont là.")


if __name__ == "__main__":
    main()
