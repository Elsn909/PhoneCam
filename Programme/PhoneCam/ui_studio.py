"""Espace Studio : médias, visionneuse, frise, inspecteur, file d'export."""
import glob
import hashlib
import os
import queue
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

from effects import AMBIANCES, DEFAULTS as FX_DEFAULTS, GRADE_KEYS, WEBCAM_PRESETS
import reframe
from reshape import DEFAULTS as SHAPE_DEFAULTS, MAKEUP_KEYS, SHADE_KEYS, SHAPE_GROUPS, SHAPE_KEYS, SHAPE_PRESETS
from studio_conf import CODECS, ENGINE_DEFAULTS, LOOK_DEFAULTS, LOOK_PRESETS, MEDIA_FILTER, QUALITIES, SIZES, TRACK_DEFAULTS
from ui_kit import C, FavoritesPage, FlowBar, Form, Page, PresetBar, Rail, Segmented, Viewer, icon, note, to_pixmap
from ui_live import SHADES
from ui_studio_voix import AUDIO_EXT, StudioVoice

PERSON_COLORS = ["#ff6a3d", "#4fce9b", "#f5b94a", "#5cc2f2", "#f2545b", "#d98cf0"]
EXPORT_KINDS = {     # indices dans CODECS, QUALITIES, SIZES et reframe.FORMATS
    "YouTube, 1080p": dict(codec=0, quality=1, size=3, format=0),
    "TikTok, Reels, Shorts (9:16)": dict(codec=0, quality=1, size=3, format=2),
    "Instagram carré": dict(codec=0, quality=1, size=3, format=3),
    "Montage (ProRes, résolution d'origine)": dict(codec=5, quality=0, size=0, format=0),
    "Archive compacte (H.265)": dict(codec=1, quality=1, size=0, format=0),
}
PREVIEW_WIDTH = 1280
NO_FACE = "Aucun visage trouvé sur la photo choisie : essaie une autre photo, de face et bien éclairée."


def clock(t):
    t = max(0.0, t)
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}.{int(t * 100) % 100:02d}"


class Timeline(QWidget):
    """Frise : temps, zone exportée, présence de chaque personne, changements de plan."""
    seek = Signal(float)
    zone = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(74)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.duration, self.pos, self.a, self.b = 0.0, 0.0, 0.0, 0.0
        self.spans, self.cuts = [], []
        self.ready = False
        self.drag = None

    def set_media(self, duration, a, b):
        self.duration, self.a, self.b = duration, a, b
        self.spans, self.cuts = [], []
        self.update()

    def set_analysis(self, spans, cuts):
        self.spans, self.cuts = spans, cuts
        self.update()

    def set_pos(self, t):
        self.pos = t
        self.update()

    def _x(self, t):
        return 10 + (self.width() - 20) * (t / self.duration if self.duration else 0)

    def _seek(self, ev):
        if self.duration:
            t = (ev.position().x() - 10) / max(1, self.width() - 20) * self.duration
            self.seek.emit(float(min(self.duration, max(0.0, t))))

    def _handle(self, x):
        """Poignée sous la souris : début, fin ou rien."""
        if not self.duration:
            return None
        da, db = abs(x - self._x(self.a)), abs(x - self._x(self.b))
        return None if min(da, db) > 9 else ("a" if da <= db else "b")

    def _time(self, ev):
        return min(self.duration, max(0.0, (ev.position().x() - 10) / max(1, self.width() - 20) * self.duration))

    def mousePressEvent(self, ev):
        self.drag = self._handle(ev.position().x())
        if self.drag is None:
            self._seek(ev)

    def mouseMoveEvent(self, ev):
        if not ev.buttons() & Qt.LeftButton:
            self.setCursor(Qt.SizeHorCursor if self._handle(ev.position().x()) else Qt.PointingHandCursor)
        elif self.drag == "a":
            self.a = min(self._time(ev), self.b - 0.1)
            self.seek.emit(self.a)
        elif self.drag == "b":
            self.b = max(self._time(ev), self.a + 0.1)
            self.seek.emit(self.b)
        else:
            self._seek(ev)

    def mouseReleaseEvent(self, _ev):
        if self.drag:
            self.drag = None
            self.zone.emit(self.a, self.b)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C["bg"]))
        if not self.duration:
            return
        # graduations
        p.setPen(QColor(C["faint"]))
        font = p.font()
        font.setPointSizeF(7.5)
        p.setFont(font)
        step = next((s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600) if self.duration / s * 62 < w), 1200)
        t = 0.0
        while t <= self.duration:
            x = self._x(t)
            p.setPen(QColor(C["line2"]))
            p.drawLine(QPointF(x, 14), QPointF(x, 19))
            p.setPen(QColor(C["faint"]))
            p.drawText(QPointF(x + 3, 11), f"{int(t) // 60}:{int(t) % 60:02d}")
            t += step
        top, bar = 22, h - 30
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["panel"]))
        p.drawRoundedRect(QRectF(10, top, w - 20, bar), 5, 5)
        xa, xb = self._x(self.a), self._x(self.b)
        p.setBrush(QColor(255, 106, 61, 34))
        p.drawRect(QRectF(xa, top, xb - xa, bar))
        rows = max([s[2] for s in self.spans] + [0]) + 1
        rh = min(9.0, (bar - 8) / rows)
        for t0, t1, row in self.spans:
            p.setBrush(QColor(PERSON_COLORS[row % len(PERSON_COLORS)]))
            p.drawRoundedRect(QRectF(self._x(t0), top + 4 + row * (rh + 1), max(1.5, self._x(t1) - self._x(t0)), rh - 1), 2, 2)
        if self.ready:                    # rendu calculé : lecture fluide
            p.setBrush(QColor(C["good"]))
            p.drawRect(QRectF(xa, top + bar - 3, xb - xa, 3))
        p.setPen(QPen(QColor(C["warn"]), 1))
        for t in self.cuts:
            p.drawLine(QPointF(self._x(t), top), QPointF(self._x(t), top + bar))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(7, 7, 10, 170))     # parties coupées
        p.drawRect(QRectF(10, top, xa - 10, bar))
        p.drawRect(QRectF(xb, top, w - 10 - xb, bar))
        p.setBrush(QColor(C["accent2"]))      # poignées
        p.drawRoundedRect(QRectF(xa - 3, top - 1, 6, bar + 2), 2, 2)
        p.drawRoundedRect(QRectF(xb - 3, top - 1, 6, bar + 2), 2, 2)
        x = self._x(self.pos)
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.drawLine(QPointF(x, 12), QPointF(x, h - 6))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QPointF(x, 12), 4, 4)


class Scrubber:
    """Images d'aperçu, décodées à part : déplacement dans la frise et lecture."""

    def __init__(self, emit):
        self.emit = emit
        self.info = None
        self.target = None
        self.playing = False
        self.offset = 0.0
        self.cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True, name="studio-scrub").start()

    def go(self, info, t):
        with self.cv:
            self.info, self.target, self.playing = info, t, False
            self.cv.notify_all()

    def play(self, info, t, end, offset=0.0):
        """offset : début de la zone quand info est le rendu calculé (qui commence là)."""
        with self.cv:
            self.info, self.target, self.playing, self.end, self.offset = info, t, True, end, offset
            self.cv.notify_all()

    def stop(self):
        with self.cv:
            self.playing = False

    def _run(self):
        while True:
            with self.cv:
                while self.target is None:
                    self.cv.wait()
                info, t, playing = self.info, self.target, self.playing
                off = self.offset if playing else 0.0
                self.target = None
            try:
                import studio
                r = studio.Reader(info, max(0.0, t - off), None if playing else t + 2 / info["fps"],
                                  width=min(PREVIEW_WIDTH, info["width"]))
                f = r.read()
                if f is not None:
                    self.emit(t, f, False)
                if playing:
                    t0, n = time.perf_counter(), 0
                    while self.playing and self.target is None:
                        f = r.read()
                        n += 1
                        now = t + n / info["fps"]
                        if f is None or now >= self.end:
                            self.emit(now, None, True)
                            break
                        delay = t0 + n / info["fps"] - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                        elif delay < -0.2:         # en retard : on rattrape sans afficher
                            continue
                        self.emit(now, f, False)
                r.close()
            except Exception:  # noqa: BLE001 — la frise reste utilisable
                pass


class Bridge(QObject):
    media = Signal(object, object)         # infos, vignette
    shown = Signal(float, object, bool)    # temps, image, fin de lecture
    rendered = Signal(int, object)         # numéro de demande, image
    analysed = Signal(object, object)      # média, analyse
    progress = Signal(float, str)
    job = Signal(int, str)                 # numéro de tâche, état
    live = Signal(int, object)             # image en cours de calcul
    calculated = Signal(object, object)    # empreinte des réglages, infos du rendu calculé
    error = Signal(str)


EMPTY_TEXT = ("Glisse une vidéo ou une image ici (ou Médias → +).\n\n"
              "Puis : « Changer le visage » ou « Effets seuls » en bas, les réglages à gauche, et Exporter.\n"
              "Un son à transformer ? Glisse-le aussi, ou ouvre la page Voix.")
STUDIO_TIPS = {
    "Favoris": "Les réglages marqués d'une étoile (clic droit sur un réglage), réunis ici.",
    "Presets": "Des rendus complets en un clic, partagés avec le Direct.",
    "Médias": "Les vidéos et images à retoucher : importe-les ou glisse-les dans la fenêtre.",
    "Visages": "La photo du visage à appliquer, et qui remplacer dans la vidéo.",
    "Suivi": "Comment les visages sont suivis d'une image à l'autre.",
    "Moteur": "Qualité du remplacement de visage (modèle, réalisme, netteté).",
    "Retouche": "Retouches du visage : forme, peau, maquillage.",
    "Webcam": "Les défauts d'une vraie webcam (compression, bruit, flou…) par-dessus l'image.",
    "Ambiance": "La lumière d'une pièce : chaude, froide, sombre, ampoule de couleur.",
    "Couleur": "Étalonnage complet : style, LUT, lumière, couleurs, roues.",
    "Export": "Format, qualité et cadrage de la vidéo exportée.",
    "Voix": "Transformer la voix d'un fichier audio ou d'un enregistrement, puis l'exporter.",
}


class StudioSpace(QWidget):
    def __init__(self, settings, registry, toast, quick=False):
        """quick : lancement rapide (rapide.py), effets seuls et rien d'autre."""
        super().__init__()
        self.settings, self.registry, self.toast = settings, registry, toast
        self.quick = quick
        self.setObjectName("app")
        self.cfg = settings.setdefault("studio", {})
        if quick:
            self.cfg["swap"] = False      # non réenregistré : le logiciel complet garde son choix
        dlc = settings.get("dlc", {})
        self.q = self.cfg.setdefault("quality", {
            "model": dlc.get("plus_model", "hyperswap_1a_256"),
            "expression": dlc.get("plus_expression", 70), "weight": dlc.get("plus_weight", 50),
            "parser": True, "occlusion": dlc.get("plus_occlusion", True), "color": dlc.get("plus_color", 70),
            "grain": dlc.get("plus_grain", 100), "boost": 0, "detail": 80, "mask_padding": 6, "mask_blur": 4,
            "mask_soften": 5, "enhancer": 0, "enhancer_mix": 50})
        for key, value in ENGINE_DEFAULTS.items():
            self.q.setdefault(key, value)
        self.tr = dict(TRACK_DEFAULTS, **self.cfg.get("track", {}))
        self.cfg["track"] = self.tr
        self.shape = dict(SHAPE_DEFAULTS, **self.cfg.get("shape", {}))
        self.cfg["shape"] = self.shape
        self.look = dict(LOOK_DEFAULTS, **self.cfg.get("look", {}))
        self.cfg["look"] = self.look
        self.fx = dict(FX_DEFAULTS, **self.cfg.get("fx", {}))
        self.cfg["fx"] = self.fx
        self.frame = dict(reframe.DEFAULTS, **self.cfg.get("frame", {}))
        self.cfg["frame"] = self.frame
        self.last_before = None
        self.user_presets = settings.setdefault("user_presets", {})
        self.panels = {"dlc": {}, "filtres": {}, "effets": {}}
        self.proxy, self.computing, self.pv = None, False, None
        self.exp = self.cfg.setdefault("export", {"codec": 0, "quality": 1, "size": 0})
        self.photos = self.cfg.setdefault("photos", [])
        self.cfg.setdefault("swap", True)  # False : visage d'origine gardé, seulement retouches et effets
        self.stale = self.stale_photos = False   # calculs remis au moment où le Studio est affiché
        self.running = False              # bouton Lancer : pas de modèles ni de rendu automatique avant
        self.folded = self.cfg.setdefault("folded", {})
        self.medias, self.cur, self.analysis, self.assign = [], None, {}, {}
        self.restored = {m["path"]: m for m in self.cfg.get("session", [])}
        self.t = 0.0
        self.mode = 1                     # 0 original, 1 rendu, 2 comparer
        self.last_source, self.last_render = None, None
        self.req = 0
        self.jobs, self.cancelled = [], False
        self.models, self.render = None, None
        self.tasks = queue.Queue()
        self.b = Bridge()
        self.b.media.connect(self._media_added)
        self.b.shown.connect(self._shown)
        self.b.rendered.connect(self._rendered)
        self.b.analysed.connect(self._analysed)
        self.b.progress.connect(self._progress)
        self.b.job.connect(self._job_state)
        self.b.live.connect(self._live)
        self.b.calculated.connect(self._calculated)
        self.b.error.connect(self._error)
        self.setAcceptDrops(True)
        self._build()
        self.scrub = Scrubber(lambda t, f, end: self.b.shown.emit(t, f, end))
        threading.Thread(target=self._work, daemon=True, name="studio-work").start()
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(30)
        self.debounce.timeout.connect(self._request_preview)
        self.add_files([p for p in self.restored if os.path.isfile(p)])       # la liste de la dernière fois

    # ------------------------------------------------------------ construction
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.rail = self.insp = Rail(420)
        self.rail.set_width(self.cfg.get("panel_w", 420))
        self.rail.on_width = lambda w: self.cfg.__setitem__("panel_w", w)
        root.addWidget(self.rail)
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        root.addLayout(col, 1)
        self.view = Viewer(EMPTY_TEXT)
        col.addWidget(self.view, 1)
        bar = QWidget()
        bar.setObjectName("bottom")
        b = QVBoxLayout(bar)
        b.setContentsMargins(12, 8, 12, 8)
        b.setSpacing(5)
        flow = FlowBar()                 # en fenêtre étroite, les groupes passent à la ligne
        tr = flow.group(spacing=3)
        self.play_btn = self._icon_button("play", "Lecture (Espace)", self.toggle_play)
        back = self._icon_button("step_b", "Image précédente (←)", lambda: self.step(-1))
        fwd = self._icon_button("step_f", "Image suivante (→)", lambda: self.step(1))
        cut_in = self._icon_button("in", "Début de la zone ici (I)", self.mark_in)
        cut_out = self._icon_button("out", "Fin de la zone ici (O)", self.mark_out)
        cut_all = self._icon_button("reset", "Tout garder : annule la coupe", self.keep_all)
        self.video_only = [self.play_btn, back, fwd, cut_in, cut_out, cut_all]
        for w in (back, self.play_btn, fwd):
            tr.addWidget(w)
        tr.addSpacing(8)
        self.clock = QLabel("00:00.00")
        self.clock.setObjectName("mono")
        self.clock.setMinimumWidth(self.clock.fontMetrics().horizontalAdvance("00:00.00 / 00:00.00") + 6)
        tr.addWidget(self.clock)
        tr = flow.group(spacing=3)
        for w in (cut_in, cut_out, cut_all):
            tr.addWidget(w)
        tr = flow.group(right=True)
        self.kind = Segmented(["Changer le visage", "Effets seuls"])
        self.kind.setFixedWidth(270)
        self.kind.setToolTip("Effets seuls : le visage d'origine est gardé, sans retouche. Seulement Webcam, "
                             "Ambiance, Couleur et Format, puis Exporter.")
        self.kind.setCurrentIndex(0 if self.cfg["swap"] else 1)
        self.kind.currentIndexChanged.connect(lambda i: self._set_swap(i == 0))
        tr.addWidget(self.kind)
        tr = flow.group(right=True)
        self.view_mode = Segmented(["Original", "Rendu", "Avant / après"])
        self.view_mode.setFixedWidth(280)
        self.view_mode.setCurrentIndex(1)
        self.view_mode.currentIndexChanged.connect(self._set_mode)
        tr.addWidget(self.view_mode)
        tr.addSpacing(8)
        grid = self._icon_button("grid", "Grille de cadrage", self._toggle_grid)
        grid.setCheckable(True)
        tr.addWidget(grid)
        tr = flow.group(right=True, spacing=6)
        self.run_btn = QPushButton("Lancer")
        self.run_btn.setObjectName("primary")
        self.run_btn.setIcon(icon("play", 16))
        self.run_btn.setCheckable(True)
        self.run_btn.setToolTip("Charge les modèles et affiche le rendu. Arrêter les libère.")
        self.run_btn.toggled.connect(self.set_running)
        tr.addWidget(self.run_btn)
        self.an_btn = QPushButton("Analyser")
        self.an_btn.setIcon(icon("track", 16))
        self.an_btn.setToolTip("Suit tous les visages sur toute la vidéo. Fait automatiquement à l'export.")
        self.an_btn.clicked.connect(self.analyse)
        tr.addWidget(self.an_btn)
        self.exp_btn = QPushButton("Exporter")
        self.exp_btn.setObjectName("primary")
        self.exp_btn.clicked.connect(self.export)
        tr.addWidget(self.exp_btn)
        b.addWidget(flow)
        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek)
        self.timeline.zone.connect(self.set_zone)
        b.addWidget(self.timeline)
        st = QHBoxLayout()
        self.status = QLabel("Prêt.")
        self.status.setObjectName("status")
        st.addWidget(self.status, 1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self.bar.setFixedWidth(220)
        self.bar.hide()
        st.addWidget(self.bar)
        self.stop_btn = QPushButton("Annuler")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.cancel)
        self.stop_btn.hide()
        st.addWidget(self.stop_btn)
        b.addLayout(st)
        col.addWidget(bar)

        self.fav = FavoritesPage(self.registry, "Studio")
        self.rail.add("star", "Favoris", self.fav)
        self._page_presets()
        self._page_media()
        self.rail.add_gap()
        n0 = self.rail.stack.count()
        self._page_faces()
        self._page_track()
        self._page_quality()
        self.rail.add_gap()
        self._page_retouch()
        self.face_pages = list(range(n0, self.rail.stack.count()))   # Visages, Suivi, Moteur, Retouche
        self.webcam_index = self.rail.stack.count()
        self._page_webcam()
        self._page_ambiance()
        self._page_look()
        self.rail.add_gap()
        self._page_export()
        self.rail.add_gap()
        self.voice = StudioVoice(self)                 # post-production de la voix (fichier, enregistrement)
        self.fav.rebuild()
        self.rail.select(int(self.cfg.get("page", 1)) % self.rail.stack.count())
        self.rail.describe(STUDIO_TIPS)
        self._apply_kind()

    def _toggle_grid(self):
        self.view.guides = not self.view.guides
        self.view.update()

    def _icon_button(self, name, tip, fn):
        b = QPushButton()
        b.setObjectName("icon")
        b.setIcon(icon(name, 18))
        b.setIconSize(QSize(18, 18))
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(fn)
        return b

    def _form(self, title, panel=None, subtitle="", builtin=None, presets=True):
        page = Page(title, subtitle)
        f = Form(page, self.folded, self.panels.setdefault(panel, {}) if panel else None, self.registry, "Studio")
        if builtin and not self.quick:     # les réglages à soi vont dans la page Presets : ici, seulement les modèles fournis
            page.extra.addWidget(PresetBar(self.panels[panel], {}, builtin, self, models_only=True))
        return page, f

    def _page_presets(self):
        page, f = self._form("Presets", subtitle="Les mêmes que dans le Direct : un preset s'applique aux deux espaces.")
        f.section("Profil")
        self.presets_form = f.form
        self.presets_index = self._add("layers", "Presets", page, f)

    def open_presets(self):
        self.rail.select(self.presets_index)

    def shot_all(self):
        """Les réglages de l'espace, sous la forme de ceux du Direct."""
        shot = {panel: {k: g() for k, (g, _p) in items.items()} for panel, items in self.panels.items()}
        shot["visage"] = {"source": self.photos[0], "source_more": list(self.photos[1:])} if self.photos else {}
        return shot

    def load_preset(self, preset):
        """Les réglages propres au Direct (anti-fuite, sortie, micro…) ne concernent pas le Studio."""
        for section, values in preset.items():
            if section == "visage":
                photos = ([values["source"]] if values.get("source") else []) + list(values.get("source_more", []))
                photos = [p for p in photos if os.path.isfile(p)]     # une photo disparue n'enlève pas le visage en place
                if photos and photos != self.photos:
                    self.photos[:] = photos
                    self._photos_changed()
                continue
            panel = self.panels.get("effets" if section in ("webcam", "ambiance") else section, {})
            for k, v in values.items():
                if k in panel and panel[k][0]() != v:
                    panel[k][1](v)

    def _add(self, name, text, page, f):
        i = self.rail.add(name, text, page)
        f.bind(self.rail, i)
        return i

    def _page_media(self):
        page = Page("Médias", "Les fichiers à retoucher. Clique sur l'un d'eux pour le voir et le régler.")
        add = page.add_tool("plus", "Importer des vidéos ou des images", self.import_dialog)
        drop = page.add_tool("trash", "Retirer le média choisi de la liste", self.remove_media)
        self.bin = QListWidget()
        self.bin.setIconSize(QSize(112, 63))
        self.bin.setSpacing(2)
        self.bin.currentRowChanged.connect(self.select)
        page.tab("Liste")
        lay = page.tab("Liste")
        lay.insertWidget(0, self.bin, 1)
        self.meta = note("Glisse des vidéos ou des images ici.\nMP4, MOV, MKV, WebM, ProRes, PNG, JPEG…")
        lay.insertWidget(1, self.meta)
        lay.takeAt(lay.count() - 1)
        self.rail.add("film", "Médias", page)
        self.rail.media_buttons = (add, drop)

    def _page_faces(self):
        page, f = self._form("Visages", subtitle="Le visage à appliquer sur la vidéo, et à qui.")
        f.section("Visage à appliquer")
        self.thumb = QLabel("Aucune photo")
        self.thumb.setObjectName("viewer")
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setFixedHeight(128)
        f.form.addRow(self.thumb)
        self.ph_label = f.note("")
        row = QHBoxLayout()
        pick = QPushButton("Choisir des photos…")
        pick.clicked.connect(self.pick_photos)
        row.addWidget(pick)
        same = QPushButton("Comme le Direct")
        same.setToolTip("Reprendre les photos choisies dans l'espace Direct")
        same.clicked.connect(self.same_as_live)
        row.addWidget(same)
        f.form.addRow(row)
        f.section("Personnes dans la vidéo")
        self.people = QVBoxLayout()
        self.people.setSpacing(6)
        host = QWidget()
        host.setLayout(self.people)
        f.form.addRow(host)
        self.no_people = f.note("Tout le monde reçoit le visage ci-dessus. « Analyser » permet de choisir qui remplacer "
                                "et donne le meilleur suivi.")
        self.faces_page = self._add("face", "Visages", page, f)
        self._show_photos()

    def _set_swap(self, on):
        self.cfg["swap"] = bool(on)
        self._apply_kind()
        self.pv = None
        self._schedule()

    def _apply_kind(self):
        """Effets seuls : les pages du visage (Visages, Suivi, Moteur, Retouche) et Analyser disparaissent,
        rien à lancer : l'aperçu se calcule tout de suite, sans modèle."""
        swap = self.cfg["swap"]
        for i in self.face_pages:
            self.rail.group.button(i).setVisible(swap)
        if self.quick:                    # ni presets ni favoris, ni retour au visage
            self.kind.hide()
            self.rail.group.button(self.presets_index).hide()
            self.rail.group.button(0).hide()          # Favoris
            self.rail.group.button(self.voice.index).hide()
        self.an_btn.setVisible(swap)
        self.run_btn.setVisible(swap)
        if not swap and (self.rail.stack.currentIndex() in self.face_pages or
                         (self.quick and self.rail.stack.currentIndex() in (0, self.presets_index))):
            self.rail.select(self.webcam_index)
        if not swap:
            self.status.setText("Effets seuls : visage d'origine gardé. Règle Webcam, Ambiance, Couleur, puis Exporter.")

    def _live_ok(self):
        """Aperçu automatique : après « Lancer », ou tout de suite en effets seuls (aucun modèle)."""
        return self.running or not self.cfg["swap"]

    @staticmethod
    def _needs_faces(snap):
        """Les visages de la vidéo ne servent qu'à les changer : en effets seuls, ni analyse ni modèle."""
        return bool(snap["swap"])

    @staticmethod
    def _bare(info):
        """Analyse vide : aucun visage suivi, pour un rendu qui n'en a pas besoin."""
        import studio
        frames = 1 if info.get("image") else max(1, int(round(info["duration"] * info["fps"])))
        return studio.Analysis(info, 0.0, None, frames)

    def _page_track(self):
        page, f = self._form("Suivi", subtitle="Comment chaque visage est retrouvé d'une image à l'autre.")
        f.section("Stabilité")
        f.slider("Lissage du visage", 0, 100, self.tr["smooth"], lambda x: self._set_track("smooth", x),
                 lambda x: "coupé" if x == 0 else f"{x} %", default=TRACK_DEFAULTS["smooth"],
                 tip="Toute la vidéo étant connue, le lissage se fait dans les deux sens du temps : sans retard. "
                     "S'applique aussitôt, sans refaire l'analyse.")
        f.section("Analyse", key="track-analysis")
        sizes = [640, 1024, 1536]
        f.combo("Précision", ["Rapide (640)", "Élevée (1024, conseillé)", "Maximale (1536, petits visages)"],
                sizes.index(self.tr["det_size"]) if self.tr["det_size"] in sizes else 1,
                lambda i: self._set_track("det_size", sizes[i]), default=1)
        f.slider("Confiance minimale", 20, 70, self.tr["det_score"], lambda x: self._set_track("det_score", x),
                 lambda x: f"{x} %", default=TRACK_DEFAULTS["det_score"])
        f.check("Rechercher les visages perdus", self.tr["rescue"], lambda on: self._set_track("rescue", on), default=True)
        f.note("Une analyse est gardée en cache tant que la vidéo et ces réglages ne changent pas. "
               "Les changements de plan sont repérés : le suivi ne les traverse pas.")
        self._add("track", "Suivi", page, f)

    def _page_quality(self):
        page, f = self._form("Moteur", "dlc", "Le rendu du nouveau visage : modèle, réalisme, netteté.")
        q = self.q

        def set_(key):
            return lambda v: self._set(q, key, v)
        plus_only, classic_only = [], []

        def refresh_engine():
            classic = q["engine"] == "classic"
            for w in plus_only:
                w.setEnabled(not classic)
            for w in classic_only:
                w.setEnabled(classic)
        engines = ["plus", "classic"]

        def set_engine(i):
            self._set(q, "engine", engines[i])
            refresh_engine()
        f.tab("Base")
        f.section("Moteur")
        engine = f.combo("", ["Moteur Plus", "Moteur d'origine"], engines.index(q["engine"]), set_engine, name="Moteur", default=0,
                         tip="Le moteur Plus donne le meilleur rendu. Le moteur d'origine (inswapper 128 px) est plus simple : "
                             "Mouth mask et Poisson Blend lui sont propres.")
        f.track("plus_enabled", lambda: engine.currentIndex() == 0, lambda on: engine.setCurrentIndex(0 if on else 1))
        models = ["hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256", "inswapper"]
        model = f.combo("Modèle", ["HyperSwap 1a, 256 px (le plus naturel)", "HyperSwap 1b, 256 px", "HyperSwap 1c, 256 px",
                                   "inswapper 128 (ressemblance maximale)"],
                        models.index(q["model"]) if q["model"] in models else 0, lambda i: self._set(q, "model", models[i]), default=0)
        f.track("plus_model", model.currentIndex, model.setCurrentIndex)
        boost = f.combo("Résolution", ["Automatique (selon le visage)", "256 px", "512 px", "768 px", "1024 px"], int(q["boost"]),
                        set_("boost"), default=0,
                        tip="Pixel boost : le modèle repasse 4, 9 ou 16 fois sur l'image du visage, recomposée en haute "
                            "résolution. En automatique, adapté à chaque plan.")
        f.track("plus_boost", lambda: max(0, boost.currentIndex() - 1), lambda i: boost.setCurrentIndex(i + 1))
        plus_only += [model, boost]
        f.section("Visage généré")
        plus_only.append(f.slider("Suivi des expressions", 0, 100, q["expression"], set_("expression"),
                                  lambda x: "coupé" if x == 0 else f"{x} %", default=70, key="plus_expression"))
        plus_only.append(f.check("Suivre le haut du visage (sourcils, yeux)", q["expr_upper"], set_("expr_upper"), key="plus_expr_upper", default=True))
        plus_only.append(f.check("Suivre le bas du visage (bouche, mâchoire)", q["expr_lower"], set_("expr_lower"), key="plus_expr_lower", default=True))
        plus_only.append(f.slider("Détail conservé", 0, 100, q["detail"], set_("detail"), lambda x: f"{x} %", default=80,
                                  tip="Reporte le détail de la haute résolution sur le visage animé par les expressions."))
        plus_only.append(f.slider("Ressemblance à la photo", 0, 100, q["weight"], set_("weight"),
                                  lambda x: "neutre" if x == 50 else f"{x - 50:+d}", default=50, key="plus_weight"))
        landmarkers = ["detector", "hrffa"]
        lmk = f.combo("Repères", ["Détecteur InsightFace", "HRFFA (tient mieux de profil)"], landmarkers.index(q["landmarker"]),
                      lambda i: self._set(q, "landmarker", landmarkers[i]), key="plus_landmarker", default=1,
                      tip="Points qui placent le visage généré. Tous deux sont lissés sur toute la vidéo.")
        f.section("Fondu")
        plus_only.append(f.slider("Accord des couleurs", 0, 100, q["color"], set_("color"), lambda x: f"{x} %", default=70, key="plus_color"))
        plus_only.append(f.slider("Accord du grain", 0, 100, q["grain"], set_("grain"), lambda x: f"{x} %", default=100, key="plus_grain"))
        plus_only.append(f.check("Contour précis du visage", q["parser"], set_("parser"), key="plus_parser", default=True))
        f.slider("Opacité", 0, 100, q["opacity"], set_("opacity"), lambda x: f"{x} %", default=100, key="opacity")
        f.slider("Netteté", 0, 50, q["sharpness"], set_("sharpness"), lambda x: f"{x / 10:.1f}", default=0, key="sharpness")

        f.tab("Masque")
        f.section("Objets devant le visage (XSeg)")
        plus_only.append(f.check("Garder ce qui passe devant (main, micro…)", q["occlusion"], set_("occlusion"), key="plus_occlusion", default=True))
        occluders = ["xseg_1", "xseg_2", "xseg_3"]
        plus_only.append(f.combo("Détecteur", ["XSeg 1", "XSeg 2", "XSeg 3 (le plus récent)"], occluders.index(q["occluder"]),
                                 lambda i: self._set(q, "occluder", occluders[i]), key="plus_occluder", default=0,
                                 tip="XSeg 2 et 3 (~67 Mo chacun) sont téléchargés au premier choix."))
        plus_only.append(f.slider("Ignorer les petites taches", 0, 15, q["occ_islands"], set_("occ_islands"),
                                  lambda x: "coupé" if x == 0 else f"{x} %", default=0, key="plus_occ_islands",
                                  tip="Moustache ou barbiche : commence vers 8 % (en % du visage). Un objet plus petit "
                                      "que ça (doigt, cigarette) passe derrière le visage généré."))
        plus_only.append(f.slider("Stabilité du masque", 0, 100, q["occ_steady"], set_("occ_steady"),
                                  lambda x: "coupée" if x == 0 else f"{x} %", default=0, key="plus_occ_steady",
                                  tip="Évite le clignotement. S'applique d'une image à la suivante : visible à l'export, "
                                      "pas sur une image isolée de l'aperçu."))
        plus_only.append(f.slider("Seuil de détection", 10, 90, q["occ_threshold"], set_("occ_threshold"),
                                  lambda x: "d'origine" if x == 50 else f"{x} %", default=50, key="plus_occ_threshold",
                                  tip="Plus bas : XSeg réagit moins, partout. Plus haut : au moindre doute."))
        f.section("Zones remplacées")
        region_names = [("skin", "Peau"), ("brows", "Sourcils"), ("eyes", "Yeux"), ("glasses", "Lunettes"),
                        ("nose", "Nez"), ("lips", "Lèvres"), ("mouth", "Intérieur de la bouche")]
        region_cbs = {}

        def regions_now():
            return [k for k, c in region_cbs.items() if c.isChecked()]
        for key, text in region_names:
            region_cbs[key] = f.check(text, key in q["regions"], lambda _on: self._set(q, "regions", regions_now()),
                                      tip="Décocher une zone garde la vraie.", default=True)
            plus_only.append(region_cbs[key])
        f.track("plus_regions", regions_now, lambda v: [c.setChecked(k in v) for k, c in region_cbs.items()])
        f.section("Bords")
        plus_only.append(f.slider("Marge", 0, 20, q["mask_padding"], set_("mask_padding"), lambda x: f"{x} %", default=6, key="plus_mask_padding"))
        plus_only.append(f.slider("Fondu", 0, 10, q["mask_blur"], set_("mask_blur"), lambda x: f"{x} %", default=4, key="plus_mask_blur"))
        plus_only.append(f.slider("Adoucissement", 0, 15, q["mask_soften"], set_("mask_soften"), lambda x: f"{x} px", default=5, key="plus_mask_soften"))

        f.tab("Finition")
        f.section("Amélioration du visage")
        f.combo("Modèle", ["Aucune", "GFPGAN", "GPEN 512", "GPEN 256"], int(q["enhancer"]), set_("enhancer"), default=0,
                tip="Ajoute du détail, mais peut lisser la peau ou trembler légèrement. À doser.")
        f.slider("Force", 0, 100, q["enhancer_mix"], set_("enhancer_mix"), lambda x: f"{x} %", default=50)
        f.section("Image")
        f.check("Miroir", q["mirror"], set_("mirror"), key="mirror", default=False)
        f.check("Corriger les couleurs (Fix Blueish Cam)", q["color_correction"], set_("color_correction"), key="color_correction",
                tip="Inverse le rouge et le bleu d'une source aux couleurs bleutées.", default=False)
        f.check("Lissage temporel (Interpolation)", q["interpolation"], set_("interpolation"), key="enable_interpolation",
                tip="Mêle chaque image à la précédente. Fort, il laisse des traînées sur les mouvements.", default=False)
        f.slider("Force du lissage", 0, 90, q["interpolation_force"], set_("interpolation_force"), lambda x: f"{x} %", default=0)

        f.tab("Origine")
        f.section("Moteur d'origine")
        classic_only.append(f.check("Poisson Blend (fondu des bords)", q["poisson_blend"], set_("poisson_blend"), key="poisson_blend", default=False))
        classic_only.append(f.slider("Mouth mask : taille", 0, 100, q["mouth_mask_size"], set_("mouth_mask_size"),
                                     lambda x: "désactivé" if x == 0 else str(x), default=0, key="mouth_mask_size"))
        classic_only.append(f.slider("Adoucissement de la bouche", 2, 40, q["mask_feather_ratio"], set_("mask_feather_ratio"),
                                     default=12, key="mask_feather_ratio"))
        classic_only.append(f.slider("Lèvre inférieure", 0, 50, q["mask_down_size"], set_("mask_down_size"),
                                     lambda x: f"{x / 100:.2f}", default=10, key="mask_down_size"))
        classic_only.append(f.slider("Lèvre supérieure", 50, 200, q["mask_size"], set_("mask_size"),
                                     lambda x: f"{x / 100:.2f}", default=100, key="mask_size"))
        refresh_engine()
        self._add("swap", "Moteur", page, f)

    def _page_retouch(self):
        p = self.shape
        sliders = {}

        def preset(name):
            values = {k: 0 for k in SHAPE_KEYS + MAKEUP_KEYS + SHADE_KEYS}
            values.update(SHAPE_PRESETS[name])
            for k in SHAPE_KEYS + MAKEUP_KEYS + SHADE_KEYS:
                if k in self.panels["filtres"]:
                    self.panels["filtres"][k][1](values[k])
        page, f = self._form("Retouche", "filtres", "Forme du visage, peau et maquillage, sur le visage final.", {n: lambda n=n: preset(n) for n in SHAPE_PRESETS})
        for title, items in SHAPE_GROUPS:
            f.tab(title.split(" ")[0] if title != "Maquillage et peau" else "Peau")
            f.section(title)
            for key, text, lo in items:
                fmt = (lambda x: "neutre" if x == 0 else f"{x:+d}") if lo < 0 else (lambda x: "coupé" if x == 0 else f"{x} %")
                sliders[key] = f.slider(text, lo, 100, p[key], lambda x, k=key: self._set(p, k, x), fmt, default=0, key=key)
                if key in SHADES:
                    k, label, shades = SHADES[key]
                    f.combo(label, [n for n, _ in shades], int(p[k]) % len(shades), lambda i, k=k: self._set(p, k, i), key=k, default=0)
        self._add("wand", "Retouche", page, f)

    def _page_webcam(self):
        p = self.fx
        sliders = {}

        def preset(name):
            values = dict(FX_DEFAULTS, **WEBCAM_PRESETS[name])
            for k, s in sliders.items():
                s.setValue(int(values[k]))
        page, f = self._form("Webcam", "effets", "Les défauts d'une vraie webcam, pour un rendu crédible.", {n: lambda n=n: preset(n) for n in WEBCAM_PRESETS})
        for title, specs in (
                ("Flux", (("scale", "Résolution du flux", 20, 100, " %"), ("jpeg", "Qualité de compression", 5, 100, ""),
                          ("glitch", "Chutes de débit", 0, 50, " ‰"), ("adaptive", "Compression selon le mouvement", 0, 100, " %"))),
                ("Capteur", (("blur", "Flou", 0, 30, ""), ("sharpen", "Sur-accentuation", 0, 150, " %"),
                             ("noise", "Bruit capteur", 0, 40, ""), ("chroma_noise", "Bruit de couleur", 0, 30, ""),
                             ("exposure", "Dérive exposition / WB", 0, 100, " %"),
                             ("contrast", "Contraste petit capteur", 0, 100, " %"),
                             ("motion_blur", "Flou de bougé", 0, 100, " %"),
                             ("vignette", "Vignettage / aberration", 0, 100, " %"))),
                ("Cadence", (("fps", "Cadence max", 5, 60, " img/s"), ("freeze", "Micro-gels", 0, 30, " ‰")))):
            f.tab(title)
            f.section(title, key="webcam-" + title)
            for key, text, lo, hi, unit in specs:
                sliders[key] = f.slider(text, lo, hi, p[key], lambda x, k=key: self._set(p, k, x),
                                        lambda x, unit=unit: f"{x}{unit}", default=FX_DEFAULTS[key], key=key)
            if title == "Cadence":
                f.note("Cadence et micro-gels s'appliquent au rendu calculé et à l'export, pas à l'image fixe.")
        self._add("webcam", "Webcam", page, f)

    def _page_ambiance(self):
        p = self.fx
        sliders = {}

        def preset(name):
            values = {k: FX_DEFAULTS[k] for k in GRADE_KEYS}
            values.update(AMBIANCES[name])
            for k, s in sliders.items():
                s.setValue(int(values[k]))
        page, f = self._form("Ambiance", "effets", "La lumière d'une pièce, en quelques curseurs.", {n: lambda n=n: preset(n) for n in AMBIANCES})
        f.section("Lumière", key="ambiance-room")
        for key, text, lo, hi, fmt in (
                ("grade_temp", "Lumière chaude ↔ froide", 2000, 10000, lambda x: f"{x} K"),
                ("grade_dark", "Pièce sombre", 0, 100, lambda x: "coupée" if x == 0 else f"{x} %"),
                ("grade_exposure", "Exposition", -100, 100, lambda x: f"{x:+d}" if x else "0"),
                ("grade_contrast", "Contraste", -100, 100, lambda x: f"{x:+d}" if x else "0"),
                ("grade_saturation", "Saturation", -100, 100, lambda x: f"{x:+d}" if x else "0")):
            sliders[key] = f.slider(text, lo, hi, p[key], lambda x, k=key: self._set(p, k, x), fmt, default=FX_DEFAULTS[key], key=key)
        f.section("Ampoule RGB")
        for key, text, lo, hi, fmt in (("grade_rgb", "Intensité", 0, 100, lambda x: "coupée" if x == 0 else f"{x} %"),
                                       ("grade_hue", "Couleur", 0, 359, lambda x: f"{x}°")):
            sliders[key] = f.slider(text, lo, hi, p[key], lambda x, k=key: self._set(p, k, x), fmt, default=FX_DEFAULTS[key], key=key)
        self._add("sun", "Ambiance", page, f)

    def _page_look(self):
        page, f = self._form("Couleur", subtitle="Étalonnage : un style tout prêt ou chaque réglage à la main.")
        p = self.look
        sliders = {}
        names = list(LOOK_PRESETS)

        def preset(i):
            values = dict(LOOK_DEFAULTS, **LOOK_PRESETS[names[i]])
            for k, s in sliders.items():
                s.setValue(int(values[k]))

        def sld(key, text, lo, hi, fmt=lambda x: f"{x:+d}" if x else "0"):
            sliders[key] = f.slider(text, lo, hi, p[key], lambda x, k=key: self._set(p, k, x), fmt, default=LOOK_DEFAULTS[key])
        pct = lambda x: "coupé" if x == 0 else f"{x} %"
        deg = lambda x: f"{x}°"
        f.tab("Style")
        f.section("Style", key="look-preset")
        c = f.combo("", names, 0, lambda _i: None, name="Style de couleur")
        c.activated.connect(preset)
        self.lut_name = QLabel(os.path.basename(p["lut_path"]) or "Aucune table LUT")
        self.lut_name.setObjectName("value")
        row = QHBoxLayout()
        lut = QPushButton("Importer une LUT (.cube)…")
        lut.clicked.connect(self.pick_lut)
        clear = QPushButton("Retirer")
        clear.clicked.connect(lambda: self._set_lut(""))
        row.addWidget(lut, 1)
        row.addWidget(clear)
        f.form.addRow(row)
        f.form.addRow(self.lut_name)
        sld("lut_mix", "Force de la LUT", 0, 100, lambda x: f"{x} %")
        f.section("Finitions")
        sld("sharpen", "Netteté", 0, 100, pct)
        sld("bloom", "Lueur", 0, 100, pct)
        sld("vignette", "Vignettage", -100, 100)
        sld("grain", "Grain de film", 0, 100, pct)
        f.tab("Lumière")
        f.section("Lumière")
        for k, t in (("exposure", "Exposition"), ("contrast", "Contraste"), ("highlights", "Hautes lumières"),
                     ("shadows", "Ombres"), ("whites", "Blancs"), ("blacks", "Noirs")):
            sld(k, t, -100, 100)
        f.tab("Couleur")
        f.section("Couleur")
        for k, t in (("temperature", "Température"), ("tint", "Teinte"), ("saturation", "Saturation"),
                     ("vibrance", "Vibrance")):
            sld(k, t, -100, 100)
        sld("teal_orange", "Bleu-vert et orangé", 0, 100, pct)
        sld("mono", "Noir et blanc", 0, 100, pct)
        sld("fade", "Noirs délavés", 0, 100, pct)
        f.tab("Roues")
        for key, t in (("shadow", "Ombres"), ("mid", "Tons moyens"), ("high", "Hautes lumières")):
            f.section(t, key="look-" + key)
            sld(key + "_tint", f"{t} : intensité", 0, 100, pct)
            sld(key + "_hue", f"{t} : teinte", 0, 359, deg)
        self._add("palette", "Couleur", page, f)

    def _page_export(self):
        page, f = self._form("Export", subtitle="Réglages du fichier vidéo, puis « Exporter ».")
        e = self.exp
        fr = self.frame

        def set_(key):
            return lambda v: self._set(fr, key, v)
        f.tab("Fichier")
        f.section("Modèle d'export", key="export-kind")
        kinds = list(EXPORT_KINDS)
        kind = f.combo("", ["Choisir un modèle…"] + kinds, 0, lambda _i: None, name="Modèle d'export")

        def apply_kind(i):
            if i:
                k = EXPORT_KINDS[kinds[i - 1]]
                self.exp_widgets["codec"].setCurrentIndex(k["codec"])
                self.exp_widgets["quality"].setCurrentIndex(k["quality"])
                self.exp_widgets["size"].setCurrentIndex(k["size"])
                self.exp_widgets["format"].setCurrentIndex(k["format"])
        kind.activated.connect(apply_kind)
        f.section("Fichier", key="export-file")
        self.exp_widgets = {}
        self.exp_widgets["codec"] = f.combo("Encodage", list(CODECS), int(e["codec"]), lambda i: e.__setitem__("codec", i), default=0)
        self.exp_widgets["quality"] = f.combo("Qualité", list(QUALITIES), int(e["quality"]), lambda i: e.__setitem__("quality", i), default=1)
        self.exp_widgets["size"] = f.combo("Résolution", list(SIZES), int(e["size"]), lambda i: (e.__setitem__("size", i), self._frame_info()), default=0)
        self.frame_label = f.note("")
        f.note("Le son d'origine et la fréquence d'images sont conservés. ProRes : fichiers lourds, idéaux pour le montage. "
               "Le fichier d'origine n'est jamais modifié.")
        f.button("Exporter", self.export, primary=True)
        f.button("Exporter tous les médias", self.export_all)
        f.section("File d'export", key="export-queue")
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(110)
        self.queue_list.setToolTip("Double-clic : ouvrir le dossier du fichier")
        self.queue_list.itemDoubleClicked.connect(lambda it: self._open_job(self.queue_list.row(it)))
        f.form.addRow(self.queue_list)

        f.tab("Cadre")
        f.section("Format", key="export-frame")
        self.exp_widgets["format"] = f.combo("Format", list(reframe.FORMATS), int(fr["format"]), set_("format"), default=0)
        f.combo("Ajustement", reframe.FITS, int(fr["fit"]), set_("fit"), default=0)
        f.section("Position")
        zoom = f.slider("Zoom", 100, 400, fr["zoom"], set_("zoom"), lambda x: "aucun" if x == 100 else f"×{x / 100:.2f}", default=100)
        f.slider("Position horizontale", -100, 100, fr["x"], set_("x"),
                 lambda x: "centre" if x == 0 else (f"← {-x}" if x < 0 else f"{x} →"), default=0)
        f.slider("Position verticale", -100, 100, fr["y"], set_("y"),
                 lambda x: "centre" if x == 0 else (f"↑ {-x}" if x < 0 else f"{x} ↓"), default=0)
        f.section("Suivi du visage")
        f.check("Suivre le visage (après l'analyse)", fr["follow"], set_("follow"), default=True,
                tip="Le cadre suit la personne la plus présente, en douceur, et repart à chaque changement de plan.")
        f.slider("Douceur du suivi", 0, 100, fr["follow_smooth"], set_("follow_smooth"), lambda x: f"{x} %", default=50)
        self.export_page = self._add("export", "Export", page, f)

    def _open_job(self, n):
        if 0 <= n < len(self.jobs):
            os.startfile(os.path.dirname(self.jobs[n]) or ".")

    # ------------------------------------------------------------ médias
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        self.add_files([u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()])

    def import_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Importer des médias", self.cfg.get("folder", ""), MEDIA_FILTER)
        if paths:
            self.cfg["folder"] = os.path.dirname(paths[0])
            self.add_files(paths)

    def add_files(self, paths):
        for path in paths:
            if path.lower().endswith(AUDIO_EXT) and not self.quick:     # un son : page Voix
                self.rail.select(self.voice.index)
                self.voice.open(os.path.normpath(path))
                continue
            self.tasks.put(("import", path))

    def _do_import(self, path):
        import engine  # noqa: F401  (bibliothèques de la carte graphique)
        import studio
        try:
            info = studio.probe(path)
        except Exception as e:  # noqa: BLE001
            self.b.error.emit(f"{os.path.basename(path)} : {e}")
            return
        r = studio.Reader(info, 0.0 if info.get("image") else min(info["duration"] * 0.2, 5.0), None, width=192)
        thumb = r.read()
        r.close()
        self.b.media.emit(info, thumb)

    def _media_added(self, info, thumb):
        m = dict(info=info, a=0.0, b=info["duration"], analysis=None, name=os.path.basename(info["path"]))
        old = self.restored.pop(info["path"], None)
        if old is not None and not info.get("image"):
            m["a"], m["b"] = max(0.0, old["a"]), min(info["duration"], old["b"])
        self.medias.append(m)
        it = QListWidgetItem(m["name"])
        it.setToolTip(info["path"])
        if thumb is not None:
            it.setIcon(to_pixmap(thumb))
        self.bin.addItem(it)
        self.bin.setCurrentRow(len(self.medias) - 1)

    def remove_media(self):
        i = self.bin.currentRow()
        if not (0 <= i < len(self.medias)):
            return
        self.scrub.stop()
        self._play_ui(False)
        gone = self.medias.pop(i)
        self.bin.takeItem(i)
        if gone is self.cur:
            self.cur = None
        if not self.medias:
            self.view.clear_image(EMPTY_TEXT)
            self.timeline.set_media(0.0, 0.0, 0.0)
            self.meta.setText("Glisse des vidéos ou des images ici.")

    def step(self, frames):
        if self.cur is not None and not self.cur["info"].get("image"):
            self.scrub.stop()
            self._play_ui(False)
            self.seek(min(self.cur["info"]["duration"], max(0.0, self.t + frames / self.cur["info"]["fps"])))

    def handle_key(self, ev):
        """Touches simples, depuis tout l'espace (voir phonecam.py). Vrai si la touche est prise."""
        k, shift = ev.key(), bool(ev.modifiers() & Qt.ShiftModifier)
        if self.rail.current() == self.voice.index and not ev.modifiers() & Qt.ControlModifier:
            return self.voice.handle_key(ev)
        if ev.modifiers() & Qt.ControlModifier or self.cur is None:
            return False
        if k in (Qt.Key_Space, Qt.Key_K):
            self.toggle_play()
        elif k == Qt.Key_I:
            self.mark_in()
        elif k == Qt.Key_O:
            self.mark_out()
        elif k in (Qt.Key_Left, Qt.Key_J):
            self.step(-(round(self.cur["info"]["fps"]) if shift or k == Qt.Key_J else 1))
        elif k in (Qt.Key_Right, Qt.Key_L):
            self.step(round(self.cur["info"]["fps"]) if shift or k == Qt.Key_L else 1)
        elif k == Qt.Key_Home:
            self.seek(self.cur["a"])
        elif k == Qt.Key_End:
            self.seek(self.cur["b"])
        else:
            return False
        return True

    def select(self, i):
        if not (0 <= i < len(self.medias)):
            return
        self.scrub.stop()
        self._play_ui(False)
        self.cur = self.medias[i]
        info = self.cur["info"]
        if info.get("image"):
            self.meta.setText(f"Image {info['codec']} · {info['width']} × {info['height']}")
        else:
            self.meta.setText(f"{info['width']} × {info['height']} · {info['fps']:.2f} img/s · {clock(info['duration'])}\n"
                              f"{info['codec']}" + (" · HDR ramené en SDR" if info["hdr"] else "")
                              + (f" · son {info['audio']}" if info["audio"] else " · sans son"))
        for b in self.video_only:
            b.setEnabled(not info.get("image"))
        self.voice._refresh(keep_text=True)
        self.timeline.set_media(info["duration"], self.cur["a"], self.cur["b"])
        self.last_render, self.last_source, self.last_before = None, None, None
        self._frame_info()
        self.seek(self.cur["a"])
        self.tasks.put(("cached", self.cur))

    def seek(self, t):
        if self.cur is None:
            return
        if self.computing:                 # se déplacer arrête le calcul en cours
            self.cancelled = True
        if self.cur["info"].get("image"):
            t = 0.0
        self.t = t
        self.timeline.set_pos(t)
        self.clock.setText(f"{clock(t)} / {clock(self.cur['info']['duration'])}")
        self.last_render = None
        self.scrub.go(self.cur["info"], t)
        self._schedule()

    def toggle_play(self):
        if self.cur is None or self.cur["info"].get("image"):
            return
        if self.computing:                 # pause pendant le calcul : on l'arrête
            self.cancelled = True
            return
        if self.play_btn.property("playing"):
            self.scrub.stop()
            self._play_ui(False)
            self._schedule()
            return
        start = self.t if self.cur["a"] <= self.t < self.cur["b"] - 0.05 else self.cur["a"]
        if self.mode == 0 or (self.cfg["swap"] and not self.photos):
            self.scrub.play(self.cur["info"], start, self.cur["b"])
            self._play_ui(True)
            return
        info = self._proxy_info()
        if info is not None:
            self.scrub.play(info, start, self.cur["b"], self.cur["a"])
            self._play_ui(True)
            return
        self.computing = True
        self.play_btn.setIcon(icon("pause", 18))
        self.tasks.put(("proxy", self.cur, self._signature(), dict(self._snapshot(), zone=(self.cur["a"], self.cur["b"]))))

    def _signature(self):
        """Empreinte de ce qui change le rendu : un même réglage retrouve son calcul."""
        m = self.cur
        data = repr((m["info"]["path"], m["a"], m["b"], self.q, self.shape, self.fx, self.look, self.frame, self.assign,
                     self.photos, self.tr, self.cfg["swap"]))
        return hashlib.sha1(data.encode("utf-8", "replace")).hexdigest()[:16]

    def _proxy_path(self, sig):
        import models_plus
        return os.path.join(os.path.dirname(models_plus.MODELS_DIR), "studio-cache", f"rendu_{sig}.mp4")

    def _proxy_info(self):
        if self.proxy is None or self.cur is None or self.proxy[0] != self._signature():
            return None
        return self.proxy[1]

    def _live(self, i, img):
        info = self.cur["info"] if self.cur else None
        if info is None:
            return
        self.t = self.cur["a"] + i / info["fps"]
        self.timeline.set_pos(self.t)
        self.clock.setText(f"{clock(self.t)} / {clock(info['duration'])}")
        self.view.show_image(img)

    def _calculated(self, sig, info):
        self.computing = False
        self._play_ui(False)
        if info is None or self.cur is None:
            return
        self.proxy = (sig, info)
        self.timeline.ready = sig == self._signature()
        self.timeline.update()
        if self.timeline.ready:            # calcul terminé : lecture depuis le début de la zone
            self.scrub.play(info, self.cur["a"], self.cur["b"], self.cur["a"])
            self._play_ui(True)

    def _play_ui(self, on):
        self.play_btn.setProperty("playing", on)
        self.play_btn.setIcon(icon("pause" if on else "play", 18))

    def mark_in(self):
        if self.cur:
            self.set_zone(min(self.t, self.cur["b"] - 0.1), self.cur["b"])

    def mark_out(self):
        if self.cur:
            self.set_zone(self.cur["a"], max(self.t, self.cur["a"] + 0.1))

    def keep_all(self):
        if self.cur:
            self.set_zone(0.0, self.cur["info"]["duration"])

    def set_zone(self, a, b):
        """Couper le début et la fin. L'analyse porte sur toute la vidéo : rien à recalculer."""
        if self.cur is None or self.cur["info"].get("image"):
            return
        self.cur["a"], self.cur["b"] = a, b
        self.timeline.a, self.timeline.b = a, b
        self.timeline.update()
        self._zone_label()
        self._schedule()

    def _zone_label(self):
        info = self.cur["info"]
        if info.get("image"):
            return
        kept = self.cur["b"] - self.cur["a"]
        self.status.setText(f"Gardé : {clock(self.cur['a'])} → {clock(self.cur['b'])} ({clock(kept)} sur {clock(info['duration'])})")

    def _shown(self, t, img, ended):
        if ended:
            self._play_ui(False)
            self._schedule()
            return
        if img is None:
            return
        self.last_source = img
        if self.play_btn.property("playing"):
            self.t = t
            self.timeline.set_pos(t)
            self.clock.setText(f"{clock(t)} / {clock(self.cur['info']['duration'])}")
        self._paint()

    def _paint(self):
        src, out = self.last_source, self.last_render
        if src is None:
            return
        if self.mode == 0 or out is None or self.play_btn.property("playing"):
            self.view.show_image(self._with_frame(src) if self.mode == 0 else src)
            return
        if self.mode == 2:
            src = self.last_before if self.last_before is not None else src
        if out.shape[:2] != src.shape[:2]:
            src = cv2.resize(src, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
        if self.mode == 2:
            self.view.set_compare(src, out)
        else:
            self.view.show_image(out)

    def _with_frame(self, img):
        """Mode Original : ce qui restera dans la sortie reste clair, le reste est assombri."""
        info = self.cur["info"] if self.cur else None
        if info is None or int(self.frame["fit"]) or not reframe.is_active(self.frame, info["width"], info["height"]):
            return img
        i = int(round(self.t * info["fps"]))
        x0, y0, cw, ch = reframe.window(self.frame, info["width"], info["height"], self.cur["analysis"], i)
        s = img.shape[1] / info["width"]
        x0, y0, x1, y1 = int(x0 * s), int(y0 * s), int((x0 + cw) * s), int((y0 + ch) * s)
        out = (img * 0.35).astype(np.uint8)
        out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 2)
        return out

    def _set_mode(self, i):
        self.mode = i
        self._paint()
        self._schedule()

    # ------------------------------------------------------------ réglages
    def _set(self, d, key, value):
        d[key] = value
        # Le visage est gardé pour l'image affichée : seule la qualité oblige à le recalculer.
        self.debounce.setInterval(220 if d is self.q else 30)
        self._schedule()

    def _set_track(self, key, value):
        self.tr[key] = value
        if key == "smooth":
            if self.cur and self.cur["analysis"] is not None:
                self.tasks.put(("smooth", self.cur["analysis"], value))
            self._schedule()
        elif self.cur:
            self.cur["analysis"] = None
            self.timeline.set_analysis([], [])

    def _frame_info(self):
        """Taille finale, sous les réglages du cadre."""
        if self.cur is None:
            self.frame_label.setText("Importe une vidéo pour voir la taille de sortie.")
            return
        info = self.cur["info"]
        w, h = reframe.output_size(self.frame, info["width"], info["height"])
        height = list(SIZES.values())[int(self.exp["size"])]
        if height and height != min(w, h):
            s = height / min(w, h)
            w, h = reframe.even(w * s), reframe.even(h * s)
        self.frame_label.setText(f"Sortie : {w} × {h}. En mode Original, le cadre final s'affiche sur l'image.")

    def _schedule(self):
        self._frame_info()
        if self.mode == 0:
            self._paint()
        self.timeline.ready = bool(self.cur and self._proxy_info() is not None)
        self.timeline.update()
        if self.cur is not None and self.mode != 0 and not self.play_btn.property("playing"):
            self.debounce.start()

    def _request_preview(self):
        if not self.isVisible():          # un preset appliqué depuis le Direct ne charge pas la carte graphique
            self.stale = True
            return
        if not self._live_ok():
            return
        self.req += 1
        self.tasks.put(("preview", self.req))

    def pick_lut(self):
        path, _ = QFileDialog.getOpenFileName(self, "Table de couleurs", "", "Tables LUT (*.cube)")
        if path:
            self._set_lut(path)

    def _set_lut(self, path):
        self.lut_name.setText(os.path.basename(path) or "Aucune table LUT")
        self._set(self.look, "lut_path", path)

    def pick_photos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Photos du visage à appliquer", "",
                                                "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if paths:
            self.photos[:] = paths
            self._photos_changed()

    def same_as_live(self):
        src = self.settings.get("source")
        self.photos[:] = ([src] if src else []) + list(self.settings.get("source_more", []))
        self._photos_changed()

    def _photos_changed(self):
        self._show_photos()
        if not self.isVisible():
            self.stale = self.stale_photos = True
            return
        self.tasks.put(("sources",))
        self._schedule()

    def showEvent(self, ev):
        super().showEvent(ev)
        if self.stale_photos:
            self.tasks.put(("sources",))
        if self.stale:
            self.stale = self.stale_photos = False
            self._schedule()

    def _show_photos(self):
        n = len(self.photos)
        if n and os.path.isfile(self.photos[0]):
            img = cv2.imdecode(np.fromfile(self.photos[0], np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.thumb.setPixmap(to_pixmap(img).scaled(200, 124, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.ph_label.setText(f"{n} photo(s) : leurs empreintes sont moyennées." if n else
                              "Choisis une ou plusieurs photos de la même personne.")

    def _analysed(self, media, a):
        if a is None:
            return
        media["analysis"] = a
        if media is not self.cur:
            return
        fps = a.info["fps"]
        spans = []
        for t in a.tracks:
            spans.append((a.start + t.frames[0] / fps, a.start + (t.frames[-1] + 1) / fps, t.person))
        self.timeline.set_analysis(spans, [a.start + c / fps for c in a.cuts])
        self._fill_people(a)
        self._schedule()

    def _fill_people(self, a):
        while self.people.count():
            w = self.people.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.no_people.setVisible(not a.persons)
        if not a.persons:
            self.no_people.setText("Aucun visage trouvé dans cette vidéo.")
        total = max(1, a.frames)
        for pr in a.persons:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            pic = QLabel()
            pic.setFixedSize(52, 52)
            if pr["thumb"] is not None:
                pic.setPixmap(to_pixmap(pr["thumb"]).scaled(52, 52, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            h.addWidget(pic)
            txt = QVBoxLayout()
            name = QLabel(f"Personne {pr['id'] + 1}")
            name.setStyleSheet(f"color:{PERSON_COLORS[pr['id'] % len(PERSON_COLORS)]}; font-weight:600;")
            pct = QLabel(f"{min(100, pr['frames'] * 100 // total)} % de la vidéo")
            pct.setObjectName("note")
            txt.addWidget(name)
            txt.addWidget(pct)
            h.addLayout(txt)
            c = QComboBox()
            c.addItems(["Remplacer", "Laisser tel quel", "Autre visage…"])
            state = self.assign.get(pr["id"], "default")
            c.setCurrentIndex(0 if state == "default" else 1 if state == "none" else 2)
            c.activated.connect(lambda i, pid=pr["id"]: self._assign(pid, i))
            h.addWidget(c, 1)
            self.people.addWidget(row)

    def _assign(self, pid, i):
        if i == 2:
            path, _ = QFileDialog.getOpenFileName(self, f"Visage pour la personne {pid + 1}", "",
                                                  "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
            self.assign[pid] = path or "default"
        else:
            self.assign[pid] = "default" if i == 0 else "none"
        self.tasks.put(("sources",))
        self._schedule()

    # ------------------------------------------------------------ actions
    def analyse(self):
        if self.cur is not None:
            self.an_btn.setEnabled(False)
            self.tasks.put(("analyse", self.cur))

    def _snapshot(self):
        codec = list(CODECS)[int(self.exp["codec"])]
        return dict(codec=codec, quality=list(QUALITIES)[int(self.exp["quality"])],
                    height=list(SIZES.values())[int(self.exp["size"])], q=dict(self.q), shape=dict(self.shape),
                    look=dict(self.look), fx=dict(self.fx), frame=dict(self.frame), track=dict(self.tr), assign=dict(self.assign), photos=list(self.photos),
                    swap=bool(self.cfg["swap"]))

    def export(self):
        if self.cur is None:
            return
        if self.cfg["swap"] and not self.photos:
            self.insp.select(self.faces_page)
            self.status.setText("Choisis d'abord le visage à appliquer.")
            return
        codec = list(CODECS)[int(self.exp["codec"])]
        ext = CODECS[codec][1]
        stem = os.path.splitext(self.cur["info"]["path"])[0] + "_studio."
        if self.cur["info"].get("image"):
            path, _ = QFileDialog.getSaveFileName(self, "Exporter l'image", stem + "png",
                                                  "Image PNG (*.png);;Image JPEG (*.jpg)")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Exporter la vidéo", stem + ext, f"Vidéo (*.{ext})")
        if path:
            self._queue(self.cur, path)

    def export_all(self):
        if self.cfg["swap"] and not self.photos:
            self.insp.select(self.faces_page)
            self.status.setText("Choisis d'abord le visage à appliquer.")
            return
        ext = CODECS[list(CODECS)[int(self.exp["codec"])]][1]
        for m in self.medias:
            self._queue(m, os.path.splitext(m["info"]["path"])[0] + "_studio." + ("png" if m["info"].get("image") else ext))

    def _queue(self, media, path):
        n = len(self.jobs)
        self.jobs.append(path)
        self.queue_list.addItem(f"{os.path.basename(path)} : en attente")
        self.tasks.put(("export", n, media, path, dict(self._snapshot(), zone=(media["a"], media["b"]))))
        self.insp.select(self.export_page)

    def cancel(self):
        self.cancelled = True

    def _progress(self, value, text):
        self.status.setText(text)
        if value < 0:
            self.bar.hide()
            self.stop_btn.hide()
            self.an_btn.setEnabled(True)
        else:
            self.bar.show()
            self.stop_btn.show()
            self.bar.setValue(int(value * 1000))

    def _job_state(self, n, text):
        self.queue_list.item(n).setText(f"{os.path.basename(self.jobs[n])} : {text}")
        if text.startswith("terminé"):
            self.toast(f"Export terminé : {os.path.basename(self.jobs[n])}")

    def _error(self, text):
        self.status.setText(text)
        self.an_btn.setEnabled(True)

    def _rendered(self, req, img):
        if req == self.req and img is not None:
            self.last_render = img
            self._paint()

    # ------------------------------------------------------------ fil de calcul (un seul : la carte graphique)
    def _work(self):
        while True:
            task = self.tasks.get()
            try:
                if task[0] == "preview" and self.tasks.qsize():       # une plus récente attend peut-être
                    if any(t[0] == "preview" for t in list(self.tasks.queue)):
                        continue
                getattr(self, "_t_" + task[0])(*task[1:])
            except Exception as e:  # noqa: BLE001
                self.b.progress.emit(-1, "")
                self.b.error.emit(f"Erreur : {e}")

    def _t_import(self, path):
        self._do_import(path)

    def _models(self, det_size):
        import engine  # noqa: F401
        import studio
        if self.models is None:
            self.models = studio.Models(log=lambda m: self.b.progress.emit(0.0, m))
        self.b.progress.emit(0.0, "Préparation des modèles (la 1re fois : quelques minutes)…")
        self.models.load(det_size)
        self.b.progress.emit(-1, "Modèles prêts.")
        return self.models

    def _t_cached(self, media):
        import engine  # noqa: F401
        import studio
        info = media["info"]
        a = studio.load_cached(info, 0.0, None, self.tr)
        self.b.analysed.emit(media, a)

    def _t_smooth(self, a, value):
        a.finish(value)
        self._schedule_safe()

    def _schedule_safe(self):
        if not self._live_ok():
            return
        self.req += 1
        self.tasks.put(("preview", self.req))

    def set_running(self, on):
        self.running = on
        self.run_btn.setText("Arrêter" if on else "Lancer")
        self.run_btn.setIcon(icon("stop" if on else "play", 16))
        self.run_btn.setObjectName("danger" if on else "primary")
        self.run_btn.style().unpolish(self.run_btn)
        self.run_btn.style().polish(self.run_btn)
        if on:
            self._schedule()
            return
        self.debounce.stop()
        if self.computing:
            self.cancelled = True
        self.last_render = None
        self.tasks.put(("unload",))
        self._paint()
        self.status.setText("Moteur arrêté : modèles libérés.")

    def _t_unload(self):
        if self.running:              # relancé entre-temps
            return
        self.models = self.render = self.pv = None
        self._src_cache = {}
        import gc
        gc.collect()

    def _analyse(self, media, track):
        import studio
        info = media["info"]
        a = studio.load_cached(info, 0.0, None, track)
        if a is None:
            m = self._models(track["det_size"])
            self.cancelled = False
            a = studio.analyse(m, info, 0.0, None, track, self.b.progress.emit, lambda: self.cancelled)
            if a is not None:
                studio.save_cached(a, track)
        self.b.analysed.emit(media, a)
        return a

    def _t_analyse(self, media):
        a = self._analyse(media, dict(self.tr))
        self.b.progress.emit(-1, "Analyse annulée." if a is None else
                             f"Analyse terminée : {len(a.persons)} personne(s), {len(a.cuts) + 1} plan(s).")

    def _sources(self, m, photos, assign, a):
        """Personne → visage source."""
        cache = getattr(self, "_src_cache", {})
        def face(paths):
            key = tuple(paths)
            if key not in cache:
                cache[key] = m.source(paths)
            return cache[key]
        self._src_cache = cache
        main = face(photos) if photos else None
        out = {}
        ids = [p["id"] for p in a.persons] if a is not None else [0]
        for pid in ids:
            state = assign.get(pid, "default")
            out[pid] = None if state == "none" else (main if state == "default" else face([state]))
        return out

    def _enhancer(self, q):
        i = int(q.get("enhancer", 0))
        if not i or not q.get("enhancer_mix"):
            return None
        import importlib
        import modules.globals as G
        G.many_faces = True          # pas de résultat réutilisé d'une image à l'autre (réservé au direct)
        name = {1: "face_enhancer", 2: "face_enhancer_gpen512", 3: "face_enhancer_gpen256"}[i]
        mod = importlib.import_module(f"modules.processors.frame.{name}")
        if not mod.pre_check():
            return None
        return mod, q["enhancer_mix"] / 100

    def _make_render(self, snap, m, a):
        import studio
        q = {k: v for k, v in snap["q"].items() if k not in ("enhancer", "enhancer_mix")}
        if not snap["swap"]:              # visage d'origine : ni photo, ni remplacement, ni amélioration
            render = studio.Render(m, q, {}, snap["look"], None, {}, snap["fx"], snap["frame"], swap=False)
            render.no_face = False
            return render
        sources = self._sources(m, snap["photos"], snap["assign"], a)
        render = studio.Render(m, q, snap["shape"], snap["look"], self._enhancer(snap["q"]), sources, snap["fx"], snap["frame"])
        # Personne à remplacer sans visage utilisable : on le dit au lieu de ne rien changer en silence.
        render.no_face = any(v is None and snap["assign"].get(k, "default") != "none" for k, v in sources.items())
        return render

    def _t_sources(self):
        pass                          # relues au prochain aperçu

    def _t_preview(self, req):
        media = self.cur
        if media is None or self.mode == 0:
            return
        snap = self._snapshot()
        if snap["swap"] and not self.photos:
            self.b.error.emit("Étape suivante : choisis le visage à appliquer dans « Visages ».")
            return
        import studio
        m = self._models(self.tr["det_size"]) if self._needs_faces(snap) else None
        a = media["analysis"]
        info = media["info"]
        t = self.t
        # Image gardée tant qu'on ne bouge pas dans la vidéo : relire une image 4K prend du temps.
        frame = self.pv[3] if self.pv is not None and self.pv[0][:2] == (info["path"], t) else None
        if frame is None:
            r = studio.Reader(info, t, t + 2 / info["fps"])
            frame = r.read()
            r.close()
            if frame is None:
                return
        key = (info["path"], t, repr(snap["q"]), repr(snap["assign"]), tuple(snap["photos"]), id(a), self.tr["smooth"],
               snap["swap"], m is None)
        if self.pv is not None and self.pv[0] == key:      # même image, même visage : fin de chaîne seule
            self.render.set_post(snap["shape"], snap["fx"], snap["look"], snap["frame"])
            swapped, faces = self.pv[1], self.pv[2]
        else:
            self.b.error.emit("Calcul du visage…" if snap["swap"] else "Calcul de l'image…")
            self.render = self._make_render(snap, m, a)
            i = int(round(t * info["fps"]))
            swapped, faces = studio.swap_frame(m, self.render, a, i, frame)
            self.pv = (key, swapped, faces, frame)
        i = int(round(t * info["fps"]))
        self.render.a = a
        out = self.render.finish(swapped, faces, i)
        self.last_before = reframe.apply(self.pv[3], snap["frame"], a, i)
        if self.render.no_face:
            self.b.error.emit(NO_FACE)
        elif not snap["swap"]:
            self.b.error.emit("Aperçu à jour, visage d'origine gardé. Lecture : calcule le rendu, puis le rejoue.")
        elif not faces:
            self.b.error.emit("Aperçu à jour. Aucun visage à cet endroit de la vidéo.")
        else:
            self.b.error.emit("Aperçu à jour. Lecture : calcule le rendu, puis le rejoue en continu.")
        if out.shape[1] > PREVIEW_WIDTH * 1.5:
            s = PREVIEW_WIDTH * 1.5 / out.shape[1]
            out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self.b.rendered.emit(req, np.ascontiguousarray(out))

    @staticmethod
    def _kept(a, snap):
        """Analyse réduite à la partie gardée de la vidéo."""
        if a is None or a.info.get("image"):
            return a
        start, end = snap["zone"]
        whole = start <= 1e-3 and end >= a.info["duration"] - 1e-3
        return a if whole else a.sub(start, end)

    def _t_proxy(self, media, sig, snap):
        """Rendu léger de la zone (720p, visage en 256 px) : montré pendant le calcul, puis rejoué en continu."""
        import studio
        path = self._proxy_path(sig)
        self.cancelled = False
        try:
            if self._needs_faces(snap):
                a = media["analysis"] or self._analyse(media, snap["track"])
            else:
                a = media["analysis"] or self._bare(media["info"])
            a = self._kept(a, snap)
            if a is None or self.cancelled:
                self.b.progress.emit(-1, "Calcul arrêté.")
                self.b.calculated.emit(sig, None)
                return
            if not os.path.isfile(path):
                m = self._models(snap["track"]["det_size"]) if self._needs_faces(snap) else None
                q = dict(snap["q"], boost=1)
                render = self._make_render(dict(snap, q=q), m, a)
                if render.no_face:
                    self.b.progress.emit(-1, NO_FACE)
                    self.b.calculated.emit(sig, None)
                    return
                last = [0.0]

                def show(i, img):
                    now = time.perf_counter()
                    if now - last[0] > 0.08:
                        last[0] = now
                        s = min(1.0, PREVIEW_WIDTH / img.shape[1])
                        self.b.live.emit(i, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
                for old in sorted(glob.glob(path.replace(sig, "*")), key=os.path.getmtime)[:-4]:
                    os.remove(old)               # on garde les derniers calculs
                done = studio.export(m, render, a, path, None, 23, 720 if min(a.info["width"], a.info["height"]) > 720 else None,
                                     self.b.progress.emit,
                                     lambda: self.cancelled, show)
                if done is None:
                    self.b.progress.emit(-1, "Calcul arrêté.")
                    self.b.calculated.emit(sig, None)
                    return
            self.b.progress.emit(-1, "Rendu calculé : lecture en continu tant que les réglages ne changent pas.")
            self.b.calculated.emit(sig, studio.probe(path))
        except Exception:
            self.b.calculated.emit(sig, None)
            raise

    def _t_export(self, n, media, path, snap):
        import studio
        self.cancelled = False
        need = self._needs_faces(snap)
        if need:
            self.b.job.emit(n, "analyse…")
            a = media["analysis"] or self._analyse(media, snap["track"])
        else:
            a = media["analysis"] or self._bare(media["info"])
        a = self._kept(a, snap)
        if a is None:
            self.b.job.emit(n, "annulé")
            self.b.progress.emit(-1, "Export annulé.")
            return
        m = self._models(snap["track"]["det_size"]) if need else None
        render = self._make_render(snap, m, a)
        if render.no_face:
            self.b.job.emit(n, "arrêté : aucun visage sur la photo")
            self.b.progress.emit(-1, NO_FACE)
            return
        self.b.job.emit(n, "rendu…")
        t0 = time.perf_counter()
        if a.info.get("image"):
            res = studio.export_image(m, render, a, path)
        else:
            res = studio.export(m, render, a, path, snap["codec"], snap["quality"], snap["height"],
                                self.b.progress.emit, lambda: self.cancelled)
        took = time.perf_counter() - t0
        self.b.job.emit(n, f"terminé en {int(took) // 60} min {int(took) % 60:02d} s" if res else "annulé")
        self.b.progress.emit(-1, f"Exporté : {path}" if res else "Export annulé.")

    def state(self):
        self.cfg["page"] = self.insp.current()
        self.cfg["session"] = [dict(path=m["info"]["path"], a=m["a"], b=m["b"]) for m in self.medias]
