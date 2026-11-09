"""Webcam branchée sur le PC, à la place du téléphone : ses images entrent dans le flux comme celles
reçues du téléphone (PhoneServer.publish_local), le reste ne change pas."""
import threading
import time

import cv2

from server import log_error

SIZES = {"1280×720": (1280, 720), "1920×1080": (1920, 1080), "640×480": (640, 480)}
# Nos propres sorties : les prendre en entrée ferait tourner l'image en boucle.
VIRTUAL = ("OBS Virtual Camera", "Unity Video Capture", "PhoneCam")


def cameras() -> list[str]:
    """Noms des webcams (DirectShow, dans l'ordre des numéros d'OpenCV), sans les caméras virtuelles."""
    from cv2_enumerate_cameras import enumerate_cameras
    return [c.name for c in enumerate_cameras(cv2.CAP_DSHOW) if not c.name.startswith(VIRTUAL)]


def _index(name: str) -> int | None:
    from cv2_enumerate_cameras import enumerate_cameras
    return next((c.index for c in enumerate_cameras(cv2.CAP_DSHOW) if c.name == name), None)


class PcWebcam:
    def __init__(self, server):
        self.server = server
        self.status = "Webcam arrêtée."
        self.running = False
        self.size = (0, 0)              # taille réellement donnée par la webcam
        self._stop = threading.Event()
        self._thread = None

    def start(self, name: str, size: tuple[int, int], fps: int = 30):
        self.stop()
        self._stop = threading.Event()
        self.running = True
        self.status = f"Ouverture de « {name} »…"
        self._thread = threading.Thread(target=self._run, args=(name, size, fps, self._stop), daemon=True, name="webcam")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)
            self._thread = None
        self.running = False
        self.status = "Webcam arrêtée."

    def _run(self, name, size, fps, stop):
        # Une webcam débranchée ou prise par un autre logiciel : on réessaie, sans arrêter PhoneCam.
        while not stop.is_set():
            try:
                self._capture(name, size, fps, stop)
            except Exception:  # noqa: BLE001
                log_error("webcam")
                self.status = "Erreur de la webcam : nouvel essai…"
            stop.wait(2)

    def _capture(self, name, size, fps, stop):
        index = _index(name)
        if index is None:
            self.status = f"« {name} » introuvable : branchée ? Nouvel essai…"
            return
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                self.status = f"« {name} » ne s'ouvre pas (prise par un autre logiciel ?). Nouvel essai…"
                return
            # MJPG : sans lui, la plupart des webcams USB plafonnent à 5–10 img/s en 720p.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # la plus récente : pas de retard accumulé
            misses = 0
            while not stop.is_set():
                ok, img = cap.read()
                if not ok or img is None:
                    misses += 1
                    if misses > 30:
                        self.status = f"« {name} » n'envoie plus d'image. Nouvel essai…"
                        return
                    time.sleep(0.02)
                    continue
                misses = 0
                h, w = img.shape[:2]
                if (w, h) != self.size or not self.status.startswith("Webcam"):
                    self.size = (w, h)
                    self.status = f"Webcam active : {name} ({w}×{h})."
                self.server.publish_local(img)
        finally:
            cap.release()
