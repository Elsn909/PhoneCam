"""Voix : changeur de voix en temps réel (RVC 2.3 optimisé + Smart) pour PhoneCam.

Ce module ne charge rien de lourd : PyTorch et les modèles ne sont chargés qu'au premier modèle
de voix choisi, dans un fil à part (le démarrage de PhoneCam n'est pas ralenti).
Fichiers de base : data/models-voix (HuBERT, RMVPE, Silero VAD, GTCRN).
"""
import threading

import numpy as np

from .reglages import DEFAULTS, RESTART_KEYS, find_index  # noqa: F401


class Voix:
    def __init__(self):
        self.p = dict(DEFAULTS)
        self.engine = None
        self._status = "Aucun modèle chargé."
        self.loading = False

    # --- état
    def ready(self):
        return self.engine is not None and self.engine.ready

    @property
    def status(self):
        return self.engine.status if self.engine is not None else self._status

    def stat(self, name, default=0.0):
        return getattr(self.engine, name, default) if self.engine is not None else default

    def latency_ms(self):
        return self.engine.latency_ms() if self.ready() else 0.0

    # --- commandes
    def set(self, key, value):
        self.p[key] = value
        if self.engine is not None:
            self.engine.set(key, value)

    def load(self, pth, index, done=None):
        """Charge un modèle (.pth) et son index en arrière-plan ; done(ok, message) à la fin."""
        self.loading = True                  # tout de suite : l'écran de chargement l'attend
        self._status = "Préparation du moteur de voix…"

        def work():
            try:
                if self.engine is None:
                    from .moteur import VoiceChanger
                    eng = VoiceChanger()
                    eng.p = self.p           # mêmes réglages, partagés
                    self.engine = eng
            except Exception as e:  # noqa: BLE001
                self.loading = False
                self._status = f"Moteur de voix indisponible : {e}"
                if done:
                    done(False, self._status)
                return

            def finished(ok, msg):
                self.loading = False
                if done:
                    done(ok, msg)
            self.engine.load(pth, index, finished)
        threading.Thread(target=work, daemon=True, name="voix").start()

    def feed(self, x):
        """Son du micro (48 kHz) → voix transformée. Silence tant qu'aucun modèle n'est prêt :
        la voix d'origine ne sort jamais quand le changeur est actif."""
        if self.engine is None:
            return np.zeros(len(x), np.float32)
        return self.engine.feed(x)
