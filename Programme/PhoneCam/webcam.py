"""Caméra du PC : webcam USB, appareil photo ou caméra en mode webcam (Osmo Pocket, GoPro…), carte de
capture HDMI, ou téléphone utilisé comme caméra (Lien avec Windows, Camo, DroidCam…).

Chaque caméra a ses propres formats : certaines n'envoient que du MJPG, d'autres que du YUY2 ou du NV12,
certaines (caméras « virtuelles » de Windows 11, téléphones par Lien avec Windows) ne sont visibles que
par Media Foundation. Plutôt que d'imposer un format, plusieurs façons de l'ouvrir sont essayées dans
l'ordre, et la première qui donne de vraies images est gardée (et retenue pour la prochaine fois).
Les images entrent dans le flux par FrameSource.publish (server.py)."""
import threading
import time

import cv2
import numpy as np

from server import log_error

# Résolutions demandées : la caméra donne la plus proche qu'elle sait faire.
SIZES = {"1280×720 (conseillé)": (1280, 720), "1920×1080": (1920, 1080), "960×540": (960, 540),
         "640×480": (640, 480)}
ROTATIONS = {"Aucune": None, "90° (sens horaire)": cv2.ROTATE_90_CLOCKWISE, "180°": cv2.ROTATE_180,
             "90° (sens inverse)": cv2.ROTATE_90_COUNTERCLOCKWISE}
# Nos propres sorties : les prendre en entrée ferait tourner l'image en boucle.
VIRTUAL = ("OBS Virtual Camera", "OBS-Camera", "Unity Video Capture")
WAIT = 4.0              # secondes pour qu'une façon d'ouvrir la caméra donne une vraie image
_ATTEMPTS = {           # façon d'ouvrir : (moteur Windows, format demandé)
    "dshow-mjpg": (cv2.CAP_DSHOW, "MJPG"),
    "dshow": (cv2.CAP_DSHOW, None),
    "msmf": (cv2.CAP_MSMF, None),
}
_LABELS = {"dshow-mjpg": "DirectShow, MJPG", "dshow": "DirectShow", "msmf": "Media Foundation"}


def _list(api):
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        return [(c.name, c.index) for c in enumerate_cameras(api)]
    except Exception:  # noqa: BLE001 — un moteur absent ou en erreur : simplement aucune caméra par lui
        return []


def cameras() -> list[str]:
    """Noms des caméras (DirectShow puis Media Foundation, sans doublon), sans nos propres sorties."""
    names = []
    for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        for name, _i in _list(api):
            if name not in names and not name.startswith(VIRTUAL):
                names.append(name)
    return names


def _plan(name, known=None):
    """Façons d'ouvrir la caméra, à essayer dans l'ordre : (clé, moteur, numéro, format)."""
    ds = next((i for n, i in _list(cv2.CAP_DSHOW) if n == name), None)
    mf = next((i for n, i in _list(cv2.CAP_MSMF) if n == name), None)
    out = []
    if ds is not None:
        # MJPG d'abord : sans lui, la plupart des webcams USB plafonnent à 5–10 img/s en 720p.
        out += [("dshow-mjpg", ds), ("dshow", ds)]
    if mf is not None:
        out.append(("msmf", mf))
    if known in [k for k, _i in out]:           # la façon qui a marché la dernière fois, en premier
        out.sort(key=lambda a: a[0] != known)
    return [(k, _ATTEMPTS[k][0], i, _ATTEMPTS[k][1]) for k, i in out]


def _bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


class PcWebcam:
    def __init__(self, frames):
        self.frames = frames
        self.status = "Caméra arrêtée."
        self.running = False
        self.size = (0, 0)              # taille réellement donnée par la caméra
        self.fps = 0.0                  # images par seconde réellement reçues
        self.mode = ""                  # façon d'ouvrir qui a marché (voir _ATTEMPTS)
        self.known = {}                 # nom de la caméra → façon d'ouvrir retenue (gardé dans les réglages)
        self.rotate = None              # cv2.ROTATE_… ou None
        self._stop = threading.Event()
        self._thread = None

    def start(self, name: str, size: tuple[int, int], fps: int = 30):
        self.stop()
        self._stop = threading.Event()
        self.running = True
        self.status = f"Ouverture de « {name} »…"
        self._thread = threading.Thread(target=self._run, args=(name, size, fps, self._stop), daemon=True, name="camera")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(3)
            self._thread = None
        self.running = False
        self.fps = 0.0
        self.status = "Caméra arrêtée."

    def _run(self, name, size, fps, stop):
        # Une caméra débranchée ou prise par un autre logiciel : on réessaie, sans arrêter PhoneCam.
        while not stop.is_set():
            try:
                self._capture(name, size, fps, stop)
            except Exception:  # noqa: BLE001
                log_error("caméra")
                self.status = "Erreur de la caméra : nouvel essai…"
            stop.wait(2)

    def _open(self, key, api, index, fourcc, size, fps, stop):
        """Ouvre la caméra d'une façon ; renvoie (capture, première image) si de vraies images arrivent."""
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            return None, None, "ne s'ouvre pas"
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # la plus récente : pas de retard accumulé
        end = time.perf_counter() + WAIT
        got, black = 0, None
        while time.perf_counter() < end and not stop.is_set():
            ok, img = cap.read()
            img = _bgr(img) if ok else None
            if img is None or img.size == 0:
                time.sleep(0.03)
                continue
            got += 1
            # Une image toute noire au démarrage : réglage pas encore pris, ou mauvais format. On attend
            # une image qui ne l'est pas ; à défaut, la noire est gardée en dernier recours.
            if int(img[::8, ::8].max()) > 12:
                return cap, img, ""
            black = img
        if black is not None:
            return cap, black, "noire"
        cap.release()
        return None, None, "n'envoie pas d'image" if not got else "image noire"

    def _capture(self, name, size, fps, stop):
        plan = _plan(name, self.known.get(name))
        if not plan:
            self.status = f"« {name} » introuvable : branchée ? Nouvel essai…"
            return
        cap = img = None
        why = []
        kept = None                 # (capture, image, clé) noire : en dernier recours
        for key, api, index, fourcc in plan:
            if stop.is_set():
                return
            self.status = f"Ouverture de « {name} » ({_LABELS[key]})…"
            c, im, err = self._open(key, api, index, fourcc, size, fps, stop)
            if c is not None and not err:
                cap, img, self.mode = c, im, key
                break
            if c is not None and kept is None:
                kept = (c, im, key)
            elif c is not None:
                c.release()
            why.append(f"{_LABELS[key]} : {err}")
        if cap is None and kept is not None:
            cap, img, self.mode = kept
            kept = None
        elif kept is not None:
            kept[0].release()
        if cap is None:
            busy = any("ne s'ouvre pas" in w for w in why)
            self.status = (f"« {name} » ne donne aucune image" + (" : elle est peut-être utilisée par un autre logiciel "
                           "(OBS, Discord, Teams, le logiciel de la caméra…). Ferme-le ou retire-lui la caméra." if busy else ".")
                           + "  (" + " · ".join(why) + ")  Nouvel essai…")
            return
        self.known[name] = self.mode
        try:
            misses, times = 0, []
            while not stop.is_set():
                if img is not None:
                    misses = 0
                    if self.rotate is not None:
                        img = cv2.rotate(img, self.rotate)
                    now = time.perf_counter()
                    times = [t for t in times if now - t < 1.0] + [now]
                    self.fps = float(len(times))
                    h, w = img.shape[:2]
                    if (w, h) != self.size or not self.status.startswith("Caméra active") or len(times) == 1:
                        self.size = (w, h)
                    self.status = f"Caméra active : {name} — {w}×{h}, {self.fps:.0f} img/s ({_LABELS[self.mode]})."
                    self.frames.publish(np.ascontiguousarray(img))
                ok, img = cap.read()
                img = _bgr(img) if ok else None
                if img is None:
                    misses += 1
                    if misses > 60:
                        self.status = f"« {name} » n'envoie plus d'image. Nouvel essai…"
                        self.fps = 0.0
                        return
                    time.sleep(0.02)
        finally:
            cap.release()
