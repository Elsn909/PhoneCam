"""PhoneCam Studio. Direct : caméra du téléphone → remplacement du visage → filtres → effets → caméra
virtuelle. Studio (ui_studio.py, studio.py) : la même chose sur des vidéos importées, en qualité maximale."""
import json
import os
import queue
import subprocess
import sys
import threading
import time

import home_fix  # noqa: F401  (en premier : voir ce fichier)

# Avant tout import de torch : par défaut ses threads OpenMP tournent à vide ~200 ms après chaque
# opération (le collage du visage en fait à chaque image) → ~15 cœurs occupés pour rien.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
# Idem pour le calcul matriciel de numpy (OpenBLAS) : ses petits calculs (maillage des filtres)
# prennent 8 ms sur 32 threads contre 0,1 ms sur un seul.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np

# OpenCV répartit même un petit redimensionnement sur les 32 cœurs, qui tournent ensuite à vide.
# Sur des images 720p, 2 threads vont presque aussi vite pour une fraction du processeur.
cv2.setNumThreads(2)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import usb  # noqa: E402
from effects import DelayLine, Effects  # noqa: E402
from phonemic import AudioLink, PhoneMic  # noqa: E402
from guard import FaceGuard  # noqa: E402
from framing import Framing  # noqa: E402
from reshape import FaceReshape  # noqa: E402
from server import DATA_DIR, PhoneServer, log_error  # noqa: E402
from segment import PersonSegmenter  # noqa: E402
from head3d import Head3D  # noqa: E402
from landmarks import Hrffa  # noqa: E402
from models_plus import MODELS_DIR  # noqa: E402
from snapweb import SnapBridge  # noqa: E402
from webcam import PcWebcam  # noqa: E402

OUTPUT_SIZES = {"1280×720": (1280, 720), "1920×1080": (1920, 1080), "960×540": (960, 540),
                "640×480": (640, 480)}
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")


def fit(frame: np.ndarray, size: tuple[int, int], fill: bool) -> np.ndarray:
    """Adapte l'image au format de sortie : recadrage centré (fill) ou bandes noires."""
    W, H = size
    h, w = frame.shape[:2]
    if (w, h) == (W, H):
        return frame
    s = max(W / w, H / h) if fill else min(W / w, H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if fill:
        x, y = (nw - W) // 2, (nh - H) // 2
        return img[y:y + H, x:x + W]
    out = np.zeros((H, W, 3), np.uint8)
    x, y = (W - nw) // 2, (H - nh) // 2
    out[y:y + nh, x:x + nw] = img
    return out


RENDER_TIMEOUT = 3.0            # secondes : au-delà, l'image est abandonnée
STALL_RESTART = 8               # secondes sans image rendue avant de relancer PhoneCam


class Rate:
    """Compteur d'images/s sur la dernière seconde."""

    def __init__(self):
        self._t: list[float] = []

    def tick(self):
        now = time.perf_counter()
        self._t = [t for t in self._t if now - t < 1.0] + [now]

    def value(self) -> float:
        now = time.perf_counter()
        return float(sum(1 for t in self._t if now - t < 1.0))


class Mailbox:
    """Boîte à une place entre deux threads : garde seulement le dernier élément."""

    def __init__(self):
        self._cv = threading.Condition()
        self._item = None

    def put(self, item):
        with self._cv:
            self._item = item
            self._cv.notify()

    def get(self, timeout: float):
        with self._cv:
            if self._item is None:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item


class Pipeline:
    """Quatre étages en parallèle : décodage (server), Deep-Live-Cam (GPU) + garde anti-fuite,
    filtres de forme puis effets (CPU), sortie.

    Chaque étage travaille sur une image différente en même temps : le débit est celui de
    l'étage le plus lent, et non la somme de tous.
    """

    def __init__(self, server: PhoneServer):
        self.server = server
        self.guard = FaceGuard()
        self.mic = PhoneMic()           # micro de smartphone simulé (indépendant de la vidéo)
        self.audio = AudioLink(self.mic, server)
        # Snapchat Web piloté depuis l'appli ; ses boutons « pause / caméra / micro » agissent ici.
        self.snap = SnapBridge(server, DATA_DIR)
        self.snap.on_output = self._snap_output
        self.snap.mic_running = lambda: self.audio.running
        self.snap.mic_gain = lambda: int(self.mic.p["output_gain"])
        self.webcam = PcWebcam(server)  # webcam du PC, à la place du téléphone
        self.shape = FaceReshape()
        self.shape.segmenter = PersonSegmenter(MODELS_DIR)
        self.shape.head3d = Head3D()
        self.shape.hrffa = Hrffa()
        self.framing = Framing()        # zoom / position / suivi du visage de la sortie
        self.fx = Effects()
        self.delay = DelayLine(self.fx)
        self.engine = None
        self.engine_ready = False
        self.sizes = OUTPUT_SIZES
        self.size = OUTPUT_SIZES["1280×720"]
        self.fill = True
        self.vcam_fps = 30
        self.vcam_on = False
        self.out_freeze = False         # pause : la caméra virtuelle répète la dernière image
        self.out_black = False          # caméra coupée : image noire
        self._held = None
        self.vcam_status = "Caméra virtuelle arrêtée."
        self.on_preview = None          # appelé avec chaque image de sortie (aperçu)
        self.process_rate, self.output_rate, self.render_rate = Rate(), Rate(), Rate()
        self.restart_wanted = False
        self.latency_ms = 0.0
        self._vcam_restart = threading.Event()
        self._cam = None
        self._stop = threading.Event()
        self._to_fx = Mailbox()
        self._rendering = queue.Queue()       # images en cours de rendu, dans l'ordre
        self._rec = None                      # enregistrement de la sortie : [writer, fin, chemin]
        self._last_out = None                 # dernière image sortie, pour une photo
        self.rec_status = ""
        self._diag = None                     # capture de diagnostic : [file d'attente, fin]
        self.diag_status = ""
        self._slots = threading.Semaphore(2)
        self._next_ai = 0.0                   # limite d'images/s de l'IA : prochaine image acceptée

    def _skip(self) -> bool:
        """Limite d'images par seconde de l'IA (Engine.ai_fps) : les images en trop ne sont pas
        traitées du tout, ce qui allège d'autant la carte graphique. Grille régulière, comme la
        cadence des effets (une image un peu en avance n'est pas jetée)."""
        fps = getattr(self.engine, "ai_fps", 0)
        if not fps:
            return False
        now = time.perf_counter()
        interval = 1 / fps
        if now < self._next_ai - interval * 0.25:
            return True
        base = self._next_ai if now - self._next_ai < interval else now
        self._next_ai = base + interval
        return False

    def _snap_output(self, key, value):
        if key == "paused":           # pause : image figée et micro coupé, comme une coupure réseau
            self.out_freeze = value
            self.audio.muted = value or self.snap.out_state["muted"]
        elif key == "black":          # caméra coupée : image noire
            self.out_black = value
        elif key == "gain":           # volume de sortie, réglé depuis l'appli
            self.mic.p["output_gain"] = value
        elif key == "muted":
            self.audio.muted = value or self.snap.out_state["paused"]

    def start(self):
        for name, loop in (("process", self._process_loop), ("effects", self._fx_loop),
                           ("output", self._output_loop)):
            threading.Thread(target=self._keep_alive, args=(name, loop), daemon=True, name=name).start()
        threading.Thread(target=self._watchdog, daemon=True, name="watchdog").start()

    def stop(self):
        self._stop.set()

    def restart_vcam(self):
        self._vcam_restart.set()

    def _keep_alive(self, name, loop):
        """Une erreur dans un fil du flux figeait la sortie jusqu'au redémarrage de PhoneCam."""
        while not self._stop.is_set():
            try:
                loop()
                return
            except Exception as e:  # noqa: BLE001
                log_error(f"fil {name}")
                self.vcam_status = f"Flux relancé après une erreur ({name}) : {e}"
                time.sleep(0.1)

    def _watchdog(self):
        """Des images arrivent mais le moteur n'en rend plus aucune : en général le pilote graphique
        s'est réinitialisé (erreur 141 de Windows) et le moteur a perdu la carte. Seul un nouveau
        processus la retrouve : PhoneCam repart seul et le téléphone se reconnecte."""
        import faulthandler
        quiet = 0
        while not self._stop.wait(1):
            stuck = self.engine_ready and self.server.fps() >= 5 and self.render_rate.value() == 0
            quiet = quiet + 1 if stuck else 0
            if quiet < STALL_RESTART:
                continue
            quiet = 0
            try:
                with open(os.path.join(DATA_DIR, "phonecam-errors.log"), "a", encoding="utf-8") as f:
                    f.write(f"--- {time.strftime('%d/%m %H:%M:%S')} moteur bloqué, redémarrage. État des fils :\n")
                    faulthandler.dump_traceback(file=f, all_threads=True)
            except OSError:
                pass
            self.server.send_json({"t": "web", "status": "Carte graphique réinitialisée : PhoneCam redémarre…"})
            self.restart_wanted = True      # l'interface enregistre les réglages puis relance
            if not self._stop.wait(3):
                self.restart_now()

    def restart_now(self):
        subprocess.Popen([sys.executable, os.path.abspath(__file__)], cwd=HERE)
        os._exit(0)

    def _process_loop(self):
        """Détection et préparation dans l'ordre des images ; avec le moteur Plus, la fin du rendu
        d'une image chevauche le début de la suivante (voir Engine.parallel). Le fil _collect_loop
        les reprend dans l'ordre."""
        from engine import Engine  # import lourd (torch, onnxruntime) : hors du thread de l'interface
        from concurrent.futures import Future, ThreadPoolExecutor
        if self.engine is None:      # déjà chargé si ce fil repart après une erreur
            self.engine = Engine()
            self.engine_ready = self.engine.load()
            threading.Thread(target=self._keep_alive, args=("collect", self._collect_loop),
                             daemon=True, name="collect").start()
        workers = ThreadPoolExecutor(2, thread_name_prefix="render")
        last_id = 0
        while not self._stop.is_set():
            fid, frame, t_recv = self.server.wait_frame(last_id, 0.1)
            if frame is None or fid == last_id:
                if self.engine_ready:
                    self.engine._run_jobs()  # applique les réglages même sans image
                continue
            last_id = fid
            if self.engine_ready and self._skip():
                continue
            # Au plus deux images en cours : sinon on prend du retard au lieu de sauter des images.
            while not self._slots.acquire(timeout=0.1):
                if self._stop.is_set():
                    return
            if not self.engine_ready:
                done = Future()
                done.set_result(frame)
                self._rendering.put((None, done, t_recv))
                continue
            try:
                ctx = self.engine.begin(frame, want_faces=self.shape.active() or self.framing.wants_faces(),
                                        landmarks_106=not self.shape.uses_hrffa())
            except Exception as e:  # noqa: BLE001 — une image ratée ne doit jamais arrêter le flux
                # Jamais l'image d'origine ici : elle montrerait le vrai visage.
                log_error("préparation d'une image")
                self.engine.status = f"Erreur de traitement : {e}"
                self._slots.release()
                continue
            if self.engine.parallel():
                try:
                    fut = workers.submit(self.engine.render, ctx)
                except RuntimeError:  # fermeture de l'application
                    return
            else:
                fut = Future()
                try:
                    fut.set_result(self.engine.render(ctx))
                except Exception as e:  # noqa: BLE001 — l'image d'origine plutôt qu'un arrêt du flux
                    fut.set_exception(e)
            self._rendering.put((ctx, fut, t_recv))

    def _collect_loop(self):
        while not self._stop.is_set():
            try:
                ctx, fut, t_recv = self._rendering.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                out = fut.result(timeout=RENDER_TIMEOUT)
                faces, info = [], (False, [])
                if ctx is not None:
                    out = self.engine.end(ctx, out)
                    faces, info = ctx["faces"], ctx["info"]
                self.render_rate.tick()
            except Exception as e:  # noqa: BLE001
                # L'image ratée ne sort pas : la sortie garde la dernière, sans montrer le vrai visage.
                log_error("rendu d'une image")
                self.engine.status = f"Erreur de traitement : {e}"
                out, faces, info = None, [], (False, [])
            finally:
                self._slots.release()
            if out is None:
                continue
            if self._diag is not None and ctx is not None:
                self._diag_push(ctx, out)
            # La garde retient quelques images et fige la sortie avant une perte du visage.
            try:
                for item in self.guard.push(out, faces, t_recv, info):
                    self._to_fx.put(item)
            except Exception:  # noqa: BLE001
                log_error("garde du visage")
                self._to_fx.put((out, faces, t_recv))
            self.process_rate.tick()

    def _fx_loop(self):
        while not self._stop.is_set():
            item = self._to_fx.get(0.1)
            if item is not None:
                frame, faces, t_recv = item
                # Le filtre de forme passe avant les effets : la « webcam » filme le visage filtré.
                frame = self.shape.apply(frame, faces)
                # Zoom et position avant les effets : comme une webcam qui zoome avant d'encoder.
                frame = self.framing.apply(frame, faces)
                self.fx.ref_width = self.size[0]
                self.delay.push(self.fx.apply(frame), t_recv)

    def start_recording(self, seconds: float = 10.0) -> None:
        """Enregistre la sortie (ce que reçoit la caméra virtuelle) dans data\\captures."""
        folder = os.path.join(DATA_DIR, "captures")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("sortie_%Y-%m-%d_%H-%M-%S.mp4"))
        self._rec = [None, time.perf_counter() + seconds, path]
        self.rec_status = "Enregistrement…"

    def fitted(self, frame, size):
        """Même cadrage que la sortie, pour une image à comparer dans l'aperçu."""
        return fit(frame, size, self.fill)

    def recording(self) -> bool:
        return self._rec is not None

    def stop_recording(self) -> None:
        rec = self._rec
        if rec is not None:
            rec[1] = 0.0                      # refermé proprement par le fil de sortie

    def snapshot(self):
        """Photo de la sortie dans data\\captures. Renvoie le chemin, ou None sans image."""
        out = self._last_out
        if out is None:
            return None
        folder = os.path.join(DATA_DIR, "captures")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("photo_%Y-%m-%d_%H-%M-%S.png"))
        ok, data = cv2.imencode(".png", out)
        if ok:
            data.tofile(path)
        return path if ok else None

    def start_diag(self, seconds: float = 15.0) -> None:
        """Capture de diagnostic : image rendue (jamais l'image reçue), avec les valeurs du moteur
        pour chaque image (.jsonl) et tous les réglages. Écrite à part : le flux n'attend pas le disque."""
        if self._diag is not None:
            return
        folder = os.path.join(DATA_DIR, "captures")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("diagnostic_%Y-%m-%d_%H-%M-%S"))
        q = queue.Queue()
        self._diag = [q, time.perf_counter() + seconds]
        self.diag_status = "Capture en cours…"
        threading.Thread(target=self._diag_write, args=(q, path), daemon=True, name="diagnostic").start()

    def _diag_push(self, ctx, out):
        diag = self._diag
        now = time.perf_counter()
        if diag is None:
            return
        if now >= diag[1]:
            self._diag = None
            diag[0].put(None)
            return
        e, plus = self.engine, ctx.get("plus")
        jobs = ctx["jobs"]["jobs"] if ctx.get("jobs") else []
        row = {"t": round(now, 4), "moteur": plus.p["model"] if plus is not None else "origine",
               "info": ctx["info"][1], "fige": self.guard.frozen, "ms": round(e.ms, 1),
               "assoupli": [round(j["relax"], 3) for j in jobs],
               "hrffa": [None if j["second"] is None else [[round(float(x), 1) for x in k] for k in j["second"]] for j in jobs],
               "cadre": [[[round(float(x), 1) for x in k] for k in j["kps"]] for j in jobs],
               "boites": [[round(float(x), 1) for x in t.bbox] for _, t in ctx["pairs"]],
               "reperes": [[[round(float(x), 1) for x in k] for k in t.kps] for _, t in ctx["pairs"]]}
        diag[0].put((out, row))

    def _diag_write(self, q, path):
        writer, log = None, None
        try:
            with open(path + ".jsonl", "w", encoding="utf-8") as log:
                import modules.globals as G
                log.write(json.dumps({"reglages": {"dlc": self.engine.plus_p, "ai_fps": self.engine.ai_fps, "garde": self.guard.p,
                                                   "lissage": [G.enable_interpolation, G.interpolation_weight],
                                                   "det": [G.det_size, self.engine.det_interval]}},
                                     ensure_ascii=False) + "\n")
                while True:
                    item = q.get()
                    if item is None:
                        break
                    out, row = item
                    if writer is None:
                        fps = self.engine.ai_fps or 30
                        writer = cv2.VideoWriter(path + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
                    writer.write(np.ascontiguousarray(out))
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.diag_status = f"Diagnostic enregistré : {path}.mp4"
        except Exception as e:  # noqa: BLE001 — jamais une capture ne doit gêner le flux
            log_error("capture de diagnostic")
            self.diag_status = f"Capture impossible : {e}"
        finally:
            if writer is not None:
                writer.release()

    def _record(self, out):
        rec = self._rec
        if rec is None:
            return
        if rec[0] is None:
            rec[0] = cv2.VideoWriter(rec[2], cv2.VideoWriter_fourcc(*"mp4v"), 30, (out.shape[1], out.shape[0]))
        rec[0].write(np.ascontiguousarray(out))
        if time.perf_counter() >= rec[1]:
            rec[0].release()
            self._rec = None
            self.rec_status = f"Enregistré : {rec[2]}"

    def _output_loop(self):
        import pyvirtualcam
        cam = self._cam              # conservée si ce fil repart après une erreur
        while not self._stop.is_set():
            self._cam = cam
            if self._vcam_restart.is_set():
                self._vcam_restart.clear()
                if cam is not None:
                    cam.close()
                    cam = None
            if self.vcam_on and cam is None:
                try:
                    cam = pyvirtualcam.Camera(*self.size, self.vcam_fps, fmt=pyvirtualcam.PixelFormat.BGR)
                    cam.send(np.zeros((self.size[1], self.size[0], 3), np.uint8))
                    self.vcam_status = (f"Envoi vers « {cam.device} » — {self.size[0]}×{self.size[1]} "
                                        f"à {self.vcam_fps} img/s.")
                except Exception as e:
                    self.vcam_on = False
                    self.vcam_status = f"Caméra virtuelle indisponible : {e}"
            if not self.vcam_on and cam is not None:
                cam.close()
                cam = None
                self.vcam_status = "Caméra virtuelle arrêtée."

            item = self.delay.wait_next(0.05)
            if item is None:
                continue
            frame, t_recv = item
            if self.out_freeze and self._held is not None:
                frame = self._held
            else:
                self._held = frame
            if self.out_black:
                frame = np.zeros_like(frame)
            out = fit(frame, self.size, self.fill)
            self._last_out = out
            self._record(out)
            if cam is not None:
                # Envoi immédiat, sans attendre une horloge fixe : pas de saccade de rythme.
                cam.send(np.ascontiguousarray(out))
            self.latency_ms = (time.perf_counter() - t_recv) * 1000
            self.output_rate.tick()
            if self.on_preview is not None:
                self.on_preview(out)
        if cam is not None:
            cam.close()


# ---------------------------------------------------------------- interface

def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_qr(text):
    import qrcode
    qr = qrcode.QRCode(border=2, box_size=4)
    qr.add_data(text)
    return np.array(qr.make_image(fill_color="black", back_color="white").convert("RGB"))[:, :, ::-1]


def build_ui(pipeline: Pipeline, server: PhoneServer, keep_alive=None):
    from PySide6.QtCore import QByteArray, QEvent, QObject, QSize, Qt, QTimer
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout, QInputDialog, QLabel, QMenu, QMessageBox,
                                   QPushButton, QStackedWidget, QToolButton, QVBoxLayout, QWidget)
    from ui_kit import C, Palette, Registry, Toasts, ValueBar, WheelGuard, apply_theme, dark_titlebar, fix_file_dialogs, icon
    from ui_live import LiveSpace
    from ui_presets import PresetHub
    from ui_studio import StudioSpace

    settings = load_settings()
    app = QApplication(sys.argv)
    app.setApplicationName("PhoneCam Studio")
    apply_theme(app)
    fix_file_dialogs()
    wheel_guard = WheelGuard(app)
    app.installEventFilter(wheel_guard)
    win = QWidget()
    win.setObjectName("app")
    win.setWindowTitle("PhoneCam Studio")
    win.setMinimumSize(1100, 680)
    outer = QVBoxLayout(win)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    toasts = Toasts(win)
    registry = Registry(settings)

    # --- barre du haut : espaces, recherche, voyants
    topbar = QWidget()
    topbar.setObjectName("topbar")
    bar = QHBoxLayout(topbar)
    bar.setContentsMargins(14, 7, 12, 7)
    bar.setSpacing(6)
    mark = QLabel("●")
    mark.setObjectName("brandMark")
    bar.addWidget(mark)
    brand = QLabel("PhoneCam")
    brand.setObjectName("brand")
    bar.addWidget(brand)
    bar.addSpacing(18)
    space_group = QButtonGroup(win)
    spaces = QStackedWidget()
    spaces.setObjectName("workspaces")
    for i, (ic, text) in enumerate((("live", "Direct"), ("film", "Studio"))):
        b = QToolButton()
        b.setObjectName("space")
        b.setText(text)
        b.setIcon(icon(ic, 16, C["accent"]))
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.setCheckable(True)
        b.setCursor(Qt.PointingHandCursor)
        space_group.addButton(b, i)
        bar.addWidget(b)
    bar.addSpacing(18)
    search_btn = QPushButton("  Chercher un réglage                                      Ctrl+K")
    search_btn.setObjectName("search")
    search_btn.setIcon(icon("search", 15))
    search_btn.setMinimumWidth(320)
    bar.addWidget(search_btn)
    bar.addStretch()
    chips = {}
    for key, tip in (("phone", "Téléphone : ouvrir la connexion"), ("engine", "Moteur : ouvrir ses réglages"),
                     ("vcam", "Caméra virtuelle : ouvrir la sortie"), ("mic", "Micro : ouvrir ses réglages")):
        b = QToolButton()
        b.setObjectName("state")
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        chips[key] = b
        bar.addWidget(b)
    bar.addSpacing(8)

    def top_tool(name, tip, fn):
        b = QPushButton()
        b.setObjectName("icon")
        b.setIcon(icon(name, 17))
        b.setIconSize(QSize(17, 17))
        b.setToolTip(tip)
        b.clicked.connect(fn)
        bar.addWidget(b)
        return b
    outer.addWidget(topbar)
    outer.addWidget(spaces, 1)

    live = LiveSpace(pipeline, server, settings, registry, lambda t: toasts.show(t), make_qr)
    studio_space = StudioSpace(settings, registry, lambda t: toasts.show(t))
    spaces.addWidget(live)
    spaces.addWidget(studio_space)

    def persist():
        settings["ui_geometry"] = bytes(win.saveGeometry().toHex()).decode()
        settings["ui_space"] = spaces.currentIndex()
        live.state()
        studio_space.state()
        save_settings(settings)
    live.on_persist = persist

    def undo():
        text = registry.undo()
        toasts.show(f"Annulé : {text}" if text else "Rien à annuler")

    def redo():
        text = registry.redo()
        toasts.show(f"Rétabli : {text}" if text else "Rien à rétablir")

    live.presets = PresetHub(settings, live, studio_space)

    def open_presets():
        (live if spaces.currentIndex() == 0 else studio_space).open_presets()

    def preset_menu():
        m = QMenu(win)
        m.addAction("Par défaut", live.presets.apply_default)
        for group, items in live.presets.groups.items():
            sub = m.addMenu(group)
            for name in items:
                sub.addAction(name, lambda g=group, n=name: live.presets.apply(g, n))
        m.addSeparator()
        m.addAction("Gérer les presets…", open_presets)
        m.exec(profile_btn.mapToGlobal(profile_btn.rect().bottomLeft()))
    profile_btn = top_tool("layers", "Presets : changer de rendu d'un clic", preset_menu)
    top_tool("undo", "Annuler le dernier réglage (Ctrl+Z)", undo)
    top_tool("redo", "Rétablir (Ctrl+Y)", redo)

    def show_help():
        lines = [(n, k) for n, k, _f in actions() if k]
        text = "<table cellpadding='4'>" + "".join(f"<tr><td><b>{k}</b></td><td>{n}</td></tr>" for n, k in lines) + "</table>"
        text += ("<p>Curseurs : glisser n'importe où, Maj pour un réglage fin, double-clic pour la valeur d'origine, clic droit pour "
                 "saisir une valeur ou ajouter aux favoris. Studio : Espace lecture, I / O début et fin, ← → image par image, "
                 "J / K / L reculer, pause, avancer. Molette sur l'image : zoom.</p>")
        box = QMessageBox(win)
        box.setWindowTitle("Raccourcis")
        box.setText(text)
        box.exec()
    top_tool("help", "Raccourcis (F1)", show_help)

    def go(i):
        space_group.button(i).setChecked(True)
        spaces.setCurrentIndex(i)

    def full_screen():
        on = not win.isFullScreen()
        topbar.setVisible(not on)
        if spaces.currentIndex() == 0:
            live.bottom.setVisible(not on)
            live.rail.setVisible(not on)
        win.showFullScreen() if on else win.showNormal()

    live.on_full = full_screen

    def actions():
        return [("Caméra virtuelle : démarrer ou arrêter", "F5", lambda: live.vbtn.toggle()),
                ("Micro : démarrer ou arrêter", "F6", lambda: live.run_btn.toggle()),
                ("Remplacement du visage : activer ou couper", "F7", lambda: live.swap_on.toggle()),
                ("Pause (image figée, micro coupé)", "F3", live.toggle_pause),
                ("Couper l'image", "F4", live.toggle_black),
                ("Photo de la sortie", "F8", live.snapshot),
                ("Enregistrer la sortie", "F9", live.toggle_record),
                ("Comparer avec l'image du téléphone", "F10", live.toggle_compare),
                ("Plein écran", "F11", full_screen),
                ("Afficher ou masquer le panneau", "Ctrl+B", lambda: (live if spaces.currentIndex() == 0 else studio_space).rail.toggle_panel()),
                ("Aperçu : couper ou rétablir", "Ctrl+P", live.toggle_preview),
                ("Aller au Direct", "Ctrl+1", lambda: go(0)),
                ("Aller au Studio", "Ctrl+2", lambda: go(1)),
                ("Ouvrir les presets", "", open_presets),
                ("Preset : Par défaut", "", live.presets.apply_default),
                ("Studio : lancer ou arrêter le moteur", "", lambda: studio_space.run_btn.toggle()),
                *[(f"Preset : {g} / {n}", "", lambda g=g, n=n: live.presets.apply(g, n))
                  for g, items in live.presets.groups.items() for n in items],
                ("Ouvrir le dossier des captures", "", live.open_captures),
                ("Ouvrir Snapchat Web", "", pipeline.snap.open),
                ("Connecter le téléphone par câble", "", live._usb_connect)]
    palette = Palette(win, registry, actions)
    search_btn.clicked.connect(palette.open)
    shortcuts = [("Ctrl+K", palette.open), ("Ctrl+Z", undo), ("Ctrl+Y", redo), ("Ctrl+Shift+Z", redo), ("F1", show_help)]
    shortcuts += [(k, f) for _n, k, f in actions() if k]
    for keys, fn in shortcuts:
        QShortcut(QKeySequence(keys), win, activated=fn, context=Qt.ApplicationShortcut)

    def escape():
        if palette.isVisible():
            palette.hide()
        elif win.isFullScreen():
            full_screen()
    QShortcut(QKeySequence("Esc"), win, activated=escape)

    class Keys(QObject):
        """Touches simples du Studio, sauf quand on écrit quelque part."""

        def eventFilter(self, obj, ev):
            from PySide6.QtWidgets import QAbstractItemView, QComboBox, QLineEdit
            if ev.type() == QEvent.KeyPress and spaces.currentIndex() == 1 and not palette.isVisible() and win.isActiveWindow():
                focus = app.focusWidget()
                arrows = ev.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down)
                if not isinstance(focus, (QLineEdit, QComboBox, QAbstractItemView)) and not (isinstance(focus, ValueBar) and arrows):
                    return studio_space.handle_key(ev)
            return False
    keys_filter = Keys(app)
    app.installEventFilter(keys_filter)

    for key, name in (("phone", "Connexion"), ("engine", "Moteur"), ("vcam", "Sortie"), ("mic", "Micro")):
        chips[key].clicked.connect(lambda _c=False, name=name: open_page(name))

    def open_page(name):
        go(0)
        for i, page in enumerate(live.rail.pages):
            if page.title == name:
                live.rail.select(i)

    def tick():
        for key, (on, text) in live.states.items():
            b = chips[key]
            color = C["good"] if on else C["faint"]
            b.setText(f"● {text}")
            b.setStyleSheet(f"QToolButton#state {{ color: {color if on else C['faint']}; }}")
        status = getattr(live, "engine_text", "")
        chips["engine"].setToolTip(status or "Moteur")
    timer = QTimer(win)
    timer.timeout.connect(tick)
    timer.start(500)

    space_group.idClicked.connect(go)
    go(min(1, int(settings.get("ui_space", 0))))
    win.resize(1600, 940)
    if settings.get("ui_geometry"):
        win.restoreGeometry(QByteArray.fromHex(settings["ui_geometry"].encode()))
    win.show()
    dark_titlebar(win)
    if keep_alive is not None:
        keep_alive(dict(app=app, win=win, live=live, studio=studio_space, registry=registry, palette=palette, go=go, actions=actions))
    code = app.exec()
    persist()
    pipeline.snap.close()
    pipeline.audio.stop()
    pipeline.stop()
    usb.disconnect()
    time.sleep(0.2)
    return code


def main():
    server = PhoneServer()
    server.start()
    pipeline = Pipeline(server)
    pipeline.start()
    sys.exit(build_ui(pipeline, server))


if __name__ == "__main__":
    main()
