"""Dossiers de PhoneCam, journal d'erreurs, et la boîte où la caméra dépose ses images (FrameSource)."""
import os
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
# Version distribuée : le dossier de l'utilisateur (lanceur, Voix, Presets, Captures) est au-dessus de
# « Programme ». Sinon (installation de travail), tout reste dans le dossier du logiciel.
_TOP = os.path.dirname(os.path.dirname(HERE))
TOP_DIR = _TOP if os.path.isfile(os.path.join(_TOP, "Lancer PhoneCam.bat")) else ""
CAPTURES_DIR = os.path.join(TOP_DIR, "Captures") if TOP_DIR else os.path.join(DATA_DIR, "captures")
VOICES_DIR = os.path.join(TOP_DIR, "Voix") if TOP_DIR else ""
PRESETS_DIR = os.path.join(TOP_DIR, "Presets") if TOP_DIR else ""


def log_error(where: str) -> None:
    """Écrit l'erreur dans data\\phonecam-errors.log : l'appli tourne sans console."""
    try:
        with open(os.path.join(DATA_DIR, "phonecam-errors.log"), "a", encoding="utf-8") as f:
            f.write(f"--- {time.strftime('%d/%m %H:%M:%S')} {where}\n{traceback.format_exc()}\n")
    except OSError:
        pass


class FrameSource:
    """Boîte à une place entre la caméra (webcam.py) et le flux : seule l'image la plus récente compte
    (latence minimale ; une image que le moteur n'a pas eu le temps de prendre est remplacée)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._frame = None
        self._frame_t = 0.0
        self._frame_id = 0
        self._recv_times: list[float] = []

    def start(self):
        """Rien à démarrer (gardé pour les scripts qui l'appellent)."""

    def publish(self, img):
        with self._cv:
            self._frame = img
            self._frame_t = time.perf_counter()
            self._frame_id += 1
            self._recv_times.append(self._frame_t)
            self._cv.notify_all()

    def wait_frame(self, last_id: int, timeout: float = 0.1):
        """Attend une image plus récente que last_id → (id, image, instant de réception)."""
        with self._cv:
            if self._frame_id == last_id:
                self._cv.wait(timeout)
            return self._frame_id, self._frame, self._frame_t

    def latest(self):
        with self._cv:
            return self._frame_id, self._frame

    def fps(self) -> float:
        with self._cv:
            now = time.perf_counter()
            self._recv_times = [t for t in self._recv_times if now - t < 1.0]
            return float(len(self._recv_times))
