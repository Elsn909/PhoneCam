"""Voix sur un fichier (Studio, post-production) : le même moteur que le Direct, sans contrainte de temps.

Différences avec le temps réel :
  · blocs plus longs (0,5 s) et plus de contexte (4 s) : voix plus stable, raccords plus rares ;
  · traité aussi vite que la carte graphique le permet (≈ 10 à 30 × plus vite que la durée du son) ;
  · le retard du moteur est retiré : le son transformé tombe pile sur l'original (même durée) ;
  · un moteur à part de celui du Direct : une conversion ne coupe jamais la voix en direct.
"""
import threading

import numpy as np

from .reglages import DEFAULTS, find_index  # noqa: F401

SR = 48000
ZC = SR // 100
POST = {"block": 50, "extra": 40, "crossfade": 4, "margin_ms": 0}
# Réglages du Studio : ceux de la voix, sans ce qui ne sert qu'au direct (bloc, réserve…).
KEYS = ("pitch", "formant", "index_rate", "protect", "follow", "gain", "smart", "denoise", "denoise_db",
        "sensitivity", "release_ms", "preroll_ms", "floor_db", "hold_voiced", "breath", "whisper", "voicing",
        "lift_on", "lift_floor", "lift_amount", "lift_ref", "lively", "lively_always", "lift_bright")


class FileVoice:
    def __init__(self):
        self.eng = None
        self.lock = threading.Lock()

    def loaded(self):
        e = self.eng
        return (e.p["pth"], e.p["index"]) if e is not None and e.ready else None

    def load(self, pth, index):
        """Charge le modèle (dans le fil appelant, qui attend) ; renvoie sa description."""
        if self.eng is None:
            from .moteur import VoiceChanger
            self.eng = VoiceChanger()
        eng = self.eng
        if eng.ready and eng.p["pth"] == pth and (eng.p["index"] or "") == (index or ""):
            return eng.model.describe()
        eng.p.update(POST)                        # graphes CUDA préparés pour ces blocs-là
        eng.p["lift_on"] = False
        done, res = threading.Event(), []
        eng.load(pth, index, lambda ok, msg: (res.append((ok, msg)), done.set()))
        done.wait()
        ok, msg = res[0]
        if not ok:
            raise RuntimeError(msg)
        return msg

    def delay(self):
        """Retard du moteur (échantillons à 48 kHz) : fondu + recherche du raccord, et nettoyage."""
        e = self.eng
        d = e.cross + e.search // 2
        if e.p["smart"] and e.p["denoise"]:
            d += e.denoiser.delay * 3
        return d

    def convert(self, x, params, progress=None, cancel=None):
        """x : son à 48 kHz mono → voix transformée, même longueur. None si annulé."""
        eng = self.eng
        if eng is None or not eng.ready:
            raise RuntimeError("aucun modèle de voix chargé")
        with self.lock, eng.lock, eng._ctx():
            eng.p.update({k: params[k] for k in KEYS if k in params})
            eng.p.update(POST)
            eng.p["lift_auto"] = False
            eng._configure()
            d = self.delay()
            blk = eng.block
            src = np.concatenate([np.asarray(x, np.float32), np.zeros(d + 2 * blk, np.float32)])
            src = np.concatenate([src, np.zeros(-len(src) % blk, np.float32)])
            out = np.zeros(len(src), np.float32)
            n = len(src) // blk
            for i in range(n):
                if cancel is not None and cancel():
                    return None
                out[i * blk:(i + 1) * blk] = eng._step(src[i * blk:(i + 1) * blk])
                if progress is not None and (i % 4 == 0 or i == n - 1):
                    progress((i + 1) / n)
            if eng.stream is not None:
                eng.stream.synchronize()
        return out[d:d + len(x)].copy()

    def release(self):
        """Rend la mémoire de la carte graphique (le moteur sera rechargé à la prochaine conversion)."""
        with self.lock:
            eng, self.eng = self.eng, None
        if eng is not None:
            eng.graphs.clear()
            eng.model = None
            del eng
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
