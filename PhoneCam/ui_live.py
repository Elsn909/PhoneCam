"""Espace Direct : aperçu, commandes d'antenne et pages de réglages."""
import os
import subprocess
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QMenu, QPushButton, QVBoxLayout, QWidget

import phonemic
import usb
import webcam
from effects import AMBIANCES, DEFAULTS, WEBCAM_PRESETS
from framing import DEFAULTS as FRAMING_DEFAULTS
from guard import AFTER_CHOICES, DEFAULTS as GUARD_DEFAULTS
from phonemic import CODECS, DEFAULTS as MIC_DEFAULTS
from reshape import (DEFAULTS as SHAPE_DEFAULTS, EYE_SHADES, IRIS_SHADES, LIP_SHADES, MAKEUP_KEYS, SHADE_KEYS, SHAPE_GROUPS,
                     SHAPE_KEYS, SHAPE_PRESETS)
from server import DATA_DIR
from snapweb import find_browser
from ui_kit import (C, FavoritesPage, Form, Meter, Page, PresetBar, Rail, Sparkline, Toggle, Viewer, face_tile, icon, note,
                    to_qimage, to_pixmap)

IMAGES = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SHADES = {"lip_color": ("lip_shade", "Teinte des lèvres", LIP_SHADES), "eyeshadow": ("eyeshadow_shade", "Teinte du fard", EYE_SHADES),
          "iris": ("iris_shade", "Teinte des yeux", IRIS_SHADES)}
REC_TIMES = [("10 s", 10), ("30 s", 30), ("1 min", 60), ("5 min", 300)]
MIC_PRESETS = {
    "Appel 4G, téléphone en main": {"codec": 1, "bitrate": 70, "distance": 12, "room": 45,
                                    "noise_suppression": 60, "agc": 60, "handling": 22},
    "Appel posé sur le bureau": {"codec": 1, "bitrate": 70, "distance": 60, "room": 100,
                                 "noise_suppression": 75, "agc": 80, "handling": 5, "fan": 25},
    "WhatsApp (appel ou vocal)": {"codec": 3, "bitrate": 55, "distance": 12, "room": 45,
                                  "noise_suppression": 50, "agc": 55, "handling": 18},
    "Discord / Teams sur téléphone": {"codec": 4, "bitrate": 60, "distance": 35, "room": 70,
                                      "noise_suppression": 70, "agc": 65, "air": 40},
    "Mémo vocal / story": {"codec": 5, "bitrate": 95, "distance": 25, "room": 80,
                           "noise_suppression": 0, "agc": 45, "mic_noise": 40, "air": 60},
    "Vieux réseau (3G)": {"codec": 0, "bitrate": 40, "packet_loss": 8, "noise_suppression": 70},
    "Mauvais réseau": {"codec": 1, "bitrate": 25, "packet_loss": 30, "noise_suppression": 80, "agc": 75},
}


def pct(off="coupé"):
    return lambda x: off if x == 0 else f"{x} %"


def signed(x):
    return "neutre" if x == 0 else f"{x:+d}"


class Bridge(QObject):
    frame = Signal(QImage, object)
    usb = Signal(str)
    gpu = Signal(str)


class LiveSpace(QWidget):
    def __init__(self, pipeline, server, settings, registry, toast, make_qr):
        super().__init__()
        self.pipeline, self.server, self.settings, self.registry, self.toast = pipeline, server, settings, registry, toast
        self.setObjectName("app")
        self.dlc = dict(settings.get("dlc", {}))
        self.panels = {}
        self.mine = settings.setdefault("user_presets", {})
        self.folded = settings.setdefault("ui_folded", {})
        self.preview_on = True
        self.visible_now = True
        self.compare = False
        self.last = 0.0
        self.target = (960, 540)
        self.synced = False
        self.states = {}                  # état des quatre voyants de la barre du haut
        self.pending_gpu = False
        self.bridge = Bridge()
        self.bridge.frame.connect(self._frame)
        self.bridge.usb.connect(self._usb_done)
        self.bridge.gpu.connect(lambda t: self.gpu_label.setText(t))
        pipeline.on_preview = self._on_preview
        self.setAcceptDrops(True)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.rail = Rail(420)
        root.addWidget(self.rail)
        col = QVBoxLayout()
        col.setSpacing(0)
        root.addLayout(col, 1)
        self.view = Viewer("")
        self.view.overlay = self._overlay
        col.addWidget(self.view, 1)
        self.bottom = QWidget()
        self.bottom.setObjectName("bottom")
        self.bar = QVBoxLayout(self.bottom)
        self.bar.setContentsMargins(12, 9, 12, 8)
        self.bar.setSpacing(6)
        col.addWidget(self.bottom)

        self.fav = FavoritesPage(registry, "Direct")
        self.rail.add("star", "Favoris", self.fav)
        self._page_presets()
        self._page_face()
        self.rail.add_gap()
        self._page_engine()
        self._page_retouch()
        self.rail.add_gap()
        self._page_webcam()
        self._page_ambiance()
        self.rail.add_gap()
        self._page_output()
        self._page_mic()
        self._page_link(make_qr)
        self._page_diag()
        self._controls()
        self.fav.rebuild()
        self.rail.select(int(settings.get("ui_tab", 1)) % self.rail.stack.count())
        if settings.get("source"):
            self._show_thumb(settings["source"])
        self._refresh_pairs()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(250)
        self.remote_sent = {}
        self.remote_timer = QTimer(self)
        self.remote_timer.timeout.connect(self._remote_tick)
        self.remote_timer.start(60)

    # ------------------------------------------------------------ outils
    def form(self, panel, title, subtitle="", presets=None, builtin=None):
        page = Page(title, subtitle)
        f = Form(page, self.folded, self.panels.setdefault(panel, {}), self.registry, "Direct")
        if builtin:     # les réglages à soi vont dans la page Presets : ici, seulement les modèles fournis
            page.extra.addWidget(PresetBar(self.panels[panel], {}, builtin, self.window(), models_only=True))
        return page, f

    def persist_cb(self):
        if self.on_persist:
            self.on_persist()
    on_persist = None
    on_full = None

    def add(self, name, text, page, f):
        f.bind(self.rail, self.rail.add(name, text, page))

    def engine_set(self, name, value):
        self.settings.setdefault("dlc", {})[name] = value
        if self.pipeline.engine:
            self.pipeline.engine.set_option(name, value)

    # ------------------------------------------------------------ Visage
    def _page_face(self):
        s = self.settings
        page, f = self.form("visage", "Visage", "Glisse une photo n'importe où dans cette fenêtre.", "visage")
        self.face_form = f
        card = f.section("Visage appliqué")
        top = QHBoxLayout()
        top.setSpacing(12)
        self.thumb = QLabel()
        self.thumb.setFixedSize(112, 112)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setObjectName("viewer")
        self.thumb.setText("Aucun")
        top.addWidget(self.thumb)
        side = QVBoxLayout()
        side.setSpacing(5)
        pick = QPushButton("Choisir une photo…")
        pick.clicked.connect(self._choose_face)
        more = QPushButton("Ajouter des photos…")
        more.setToolTip("3 à 10 photos variées de la même personne : une identité plus fidèle et plus stable.")
        more.clicked.connect(self._add_more)
        self.more_clear = QPushButton("Retirer les photos en plus")
        self.more_clear.clicked.connect(lambda: self._set_more([]))
        for b in (pick, more, self.more_clear):
            side.addWidget(b)
        top.addLayout(side, 1)
        card.addRow(top)
        self.more_label = QLabel()         # sur une ligne : une ligne repliable étirerait la rangée au-dessus
        self.more_label.setObjectName("note")
        card.addRow(self.more_label)
        self._more_text()
        rc = f.section("Récents")
        self.recent_host = QWidget()
        self.recent = QGridLayout(self.recent_host)
        self.recent.setContentsMargins(0, 0, 0, 0)
        self.recent.setSpacing(6)
        rc.addRow(self.recent_host)
        rec = s.setdefault("recent_faces", [])
        for path in [s.get("source")] + list(s.get("source_more", [])):
            if path and path not in rec:
                rec.append(path)
        self._fill_recent()
        f.section("Qui remplacer")
        self.mode_keys = ["single", "many", "map"]
        self.mode = f.combo("", ["Le plus à gauche", "Tous", "Par personne"],
                            self.mode_keys.index(s.get("mode", "single")) if s.get("mode") in self.mode_keys else 0,
                            self._mode_changed, name="Qui remplacer",
                            tip="Un visage, tous les visages avec la même source, ou chaque personne reconnue reçoit le sien.")
        self.pair_card = f.section("Correspondances")
        f.note("Pour chaque personne filmée : une photo d'elle et le visage qui doit la remplacer.")
        self.pairs = list(s.get("pairs", []))
        self.pair_list = QListWidget()
        self.pair_list.setMaximumHeight(130)
        self.pair_card.addRow(self.pair_list)
        row = QHBoxLayout()
        for text, fn in (("Ajouter…", self._add_pair), ("Retirer", self._remove_pair)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        self.pair_card.addRow(row)
        f.slider("Seuil de reconnaissance", 10, 70, int(self.dlc.get("map_threshold", 0.3) * 100),
                 lambda x: self.engine_set("map_threshold", x / 100), lambda x: f"{x / 100:.2f}", default=30, key="map_threshold",
                 tip="Plus haut : la personne doit ressembler davantage à sa photo pour être remplacée.")
        self.pair_card.setVisible(s.get("mode") == "map")
        f.track("mode", lambda: s.get("mode", "single"),
                lambda m: m in self.mode_keys and self.mode.setCurrentIndex(self.mode_keys.index(m)))
        f.track("source_more", lambda: list(s.get("source_more", [])), self._set_more)
        f.track("source", lambda: s.get("source", ""), self._set_source)
        f.track("pairs", lambda: [list(p) for p in self.pairs], self._set_pairs)
        self.add("face", "Visage", page, f)

    def _mode_changed(self, i):
        m = self.mode_keys[i]
        self.settings["mode"] = m
        self.engine_set("many_faces", m == "many")
        self.engine_set("map_faces", m == "map")
        self.pair_card.setVisible(m == "map")

    def _show_thumb(self, path):
        pm = face_tile(path, 112) if os.path.isfile(path) else None
        if pm is not None:
            self.thumb.setPixmap(pm)

    def _more_text(self):
        n = len(self.settings.get("source_more", []))
        self.more_label.setText(f"+ {n} autre(s) photo(s) : empreinte moyenne." if n else "Plusieurs photos de la même personne : identité plus stable.")
        self.more_clear.setEnabled(bool(n))

    def _set_source(self, path):
        s = self.settings
        s["source"] = path
        self._show_thumb(path)
        if path and os.path.isfile(path):
            rec = s.setdefault("recent_faces", [])
            if path in rec:
                rec.remove(path)
            rec.insert(0, path)
            del rec[12:]
            self._fill_recent()
            if self.pipeline.engine:
                self.pipeline.engine.set_source(path, s.get("source_more", []))

    def _set_more(self, paths):
        self.settings["source_more"] = list(paths)
        self._more_text()
        if self.settings.get("source"):
            self._set_source(self.settings["source"])

    def _add_more(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Autres photos de la même personne", "", IMAGES)
        if paths:
            self._set_more(self.settings.get("source_more", []) + paths)

    def _choose_face(self):
        path, _ = QFileDialog.getOpenFileName(self, "Visage à appliquer", "", IMAGES)
        if path:
            self._set_source(path)

    def _fill_recent(self):
        while self.recent.count():
            w = self.recent.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        cur = self.settings.get("source")
        shown = 0
        for path in self.settings.get("recent_faces", []):
            pm = face_tile(path, 72) if os.path.isfile(path) else None
            if pm is None:
                continue
            b = QPushButton()
            b.setObjectName("icon")
            b.setIcon(pm)
            b.setIconSize(QSize(72, 72))
            b.setCheckable(True)
            b.setChecked(path == cur)
            b.setToolTip(path)
            b.clicked.connect(lambda _c=False, p=path: self._set_source(p))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _pos, p=path, w=b: self._recent_menu(p, w))
            self.recent.addWidget(b, shown // 4, shown % 4)
            shown += 1
        if not shown:
            self.recent.addWidget(note("Les visages utilisés arrivent ici : un clic pour changer."), 0, 0, 1, 4)

    def _recent_menu(self, path, w):
        m = QMenu(self)
        m.addAction("Retirer de la liste", lambda: (self.settings["recent_faces"].remove(path), self._fill_recent()))
        m.exec(w.mapToGlobal(w.rect().center()))

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile() and u.toLocalFile().lower().endswith(IMAGE_EXT)]
        if paths:
            self._set_source(paths[0])
            if len(paths) > 1:
                self._set_more(paths[1:])
            self.toast(f"Visage appliqué : {os.path.basename(paths[0])}")

    def _refresh_pairs(self):
        self.pair_list.clear()
        for src, ref in self.pairs:
            self.pair_list.addItem(f"{os.path.basename(ref)}  →  {os.path.basename(src)}")
        self.settings["pairs"] = self.pairs
        if self.pipeline.engine:
            self.pipeline.engine.set_pairs([tuple(p) for p in self.pairs])

    def _add_pair(self):
        ref, _ = QFileDialog.getOpenFileName(self, "Photo de la personne filmée", "", IMAGES)
        if not ref:
            return
        src, _ = QFileDialog.getOpenFileName(self, "Visage qui la remplace", "", IMAGES)
        if src:
            self.pairs.append([src, ref])
            self._refresh_pairs()

    def _remove_pair(self):
        row = self.pair_list.currentRow()
        if row >= 0:
            self.pairs.pop(row)
            self._refresh_pairs()

    def _set_pairs(self, value):
        self.pairs[:] = [list(p) for p in value]
        self._refresh_pairs()

    # ------------------------------------------------------------ Moteur
    def _page_engine(self):
        d, es = self.dlc, self.engine_set
        page, f = self.form("dlc", "Moteur", "", "dlc")
        self.engine_form = f

        def check(text, key, default, tip=None):
            return f.check(text, d.get(key, default), lambda on, key=key: es(key, on), key=key, tip=tip, default=default)

        def num(text, key, lo, hi, value, default, fmt=str, tip=None, scale=1):
            return f.slider(text, lo, hi, value, lambda x, key=key, scale=scale: es(key, x / scale), fmt, default, key, tip)

        f.tab("Base")
        f.section("Remplacement")
        self.swap_on = check("Remplacer le visage", "enabled", True)
        self.plus_on = check("Moteur Plus", "plus_enabled", False,
                             "Réalisme, expressions, objets devant le visage. ~1,7 Go téléchargés à la 1re activation (data\\models-plus).")
        plus_models = [("hyperswap_1a_256", "HyperSwap 1a, 256 px (le plus naturel)"), ("hyperswap_1b_256", "HyperSwap 1b, 256 px"),
                       ("hyperswap_1c_256", "HyperSwap 1c, 256 px"), ("inswapper", "inswapper 128 (ressemblance maximale)")]
        keys = [k for k, _ in plus_models]
        if d.get("plus_model") not in keys:   # modèle retiré depuis : retour au modèle conseillé
            d["plus_model"] = keys[0]
        f.combo("Modèle", [t for _, t in plus_models], keys.index(d["plus_model"]), lambda i: es("plus_model", keys[i]),
                key="plus_model", default=0)
        self.plus_status = f.label()
        f.section("Rendu")
        num("Suivi des expressions", "plus_expression", 0, 100, int(d.get("plus_expression", 80)), 80, pct("désactivé"))
        num("Accord couleur et lumière", "plus_color", 0, 100, int(d.get("plus_color", 70)), 70, lambda x: f"{x} %")
        num("Accord grain et netteté", "plus_grain", 0, 100, int(d.get("plus_grain", 100)), 100, lambda x: f"{x} %")
        check("Garder ce qui passe devant (main, micro…)", "plus_occlusion", True)
        check("Contour précis du visage", "plus_parser", True)
        f.section("Image")
        num("Opacité", "opacity", 0, 100, int(d.get("opacity", 1.0) * 100), 100, lambda x: f"{x} %", scale=100)
        num("Netteté", "sharpness", 0, 50, int(d.get("sharpness", 0.0) * 10), 0, lambda x: f"{x / 10:.1f}", scale=10)
        check("Miroir", "mirror", False)
        check("Corriger les couleurs (Fix Blueish Cam)", "color_correction", False)
        f.button("Réglages d'origine du moteur Plus", self._plus_best)
        f.button("Remettre les réglages avancés d'origine", self._plus_advanced)

        f.tab("Qualité")
        f.section("Détail")
        boost = f.combo("Pixel boost", ["256 px (d'origine)", "512 px : plus de détails (~+15 ms)", "768 px : trop lent pour le direct"],
                        max(0, min(2, int(d.get("plus_boost", 1)) - 1)), lambda i: es("plus_boost", i + 1), key="plus_boost", default=0,
                        tip="Le modèle passe 4 (ou 9) fois sur des images entrelacées, assemblées en 512 (ou 768) px : peau, barbe "
                            "et sourcils plus fins, surtout visage proche de la caméra.")
        num("Ressemblance à la photo", "plus_weight", 0, 100, int(d.get("plus_weight", 50)), 50,
            lambda x: "d'origine" if x == 50 else (f"+{x - 50} (plus fidèle)" if x > 50 else f"{x - 50} (plus filmé)"),
            tip="Au-dessus : plus fidèle à la photo. En dessous : un peu de la personne filmée est gardé (plus naturel).")
        f.section("Expressions")
        check("Suivre le haut du visage (sourcils, yeux)", "plus_expr_upper", True)
        check("Suivre le bas du visage (bouche, mâchoire)", "plus_expr_lower", True)
        f.section("Amélioration du visage")
        for key, text in (("face_enhancer", "GFPGAN"), ("face_enhancer_gpen256", "GPEN 256"), ("face_enhancer_gpen512", "GPEN 512")):
            check(text, key, False)

        f.tab("Masque")
        f.section("Objets devant le visage (XSeg)")
        occluders = ["xseg_1", "xseg_2", "xseg_3"]
        f.combo("Détecteur", ["XSeg 1 (d'origine)", "XSeg 2", "XSeg 3 (le plus récent)"],
                occluders.index(d.get("plus_occluder")) if d.get("plus_occluder") in occluders else 0,
                lambda i: es("plus_occluder", occluders[i]), key="plus_occluder", default=0,
                tip="XSeg 2 et 3 (~67 Mo chacun) sont téléchargés au 1er choix. Pour les grimaces, garder XSeg 1 : "
                    "les autres prennent une langue tirée pour un objet.")
        num("Ignorer les petites taches", "plus_occ_islands", 0, 15, int(d.get("plus_occ_islands", 0)), 0, pct(),
            tip="Moustache ou barbiche qui apparaît et disparaît : commence vers 8 % (en % du visage). "
                "Un objet plus petit que ça (doigt, cigarette) passe derrière le visage généré.")
        num("Stabilité du masque", "plus_occ_steady", 0, 100, int(d.get("plus_occ_steady", 0)), 0, pct("coupée"),
            tip="Contre le clignotement. Une main qui arrive passe sans retard.")
        num("Seuil de détection", "plus_occ_threshold", 10, 90, int(d.get("plus_occ_threshold", 50)), 50,
            lambda x: "d'origine" if x == 50 else f"{x} %", tip="Plus bas : XSeg réagit moins, partout. Plus haut : au moindre doute.")
        f.section("Zones remplacées")
        region_names = [("skin", "Peau"), ("brows", "Sourcils"), ("eyes", "Yeux"), ("glasses", "Lunettes"), ("nose", "Nez"),
                        ("lips", "Lèvres"), ("mouth", "Intérieur de la bouche")]
        all_regions = [k for k, _ in region_names]
        current = set(d.get("plus_regions", all_regions))
        cbs = {}

        def regions_now():
            return [k for k, c in cbs.items() if c.isChecked()]
        for key, text in region_names:
            cbs[key] = f.check(text, key in current, lambda _on: es("plus_regions", regions_now()), default=True,
                               tip="Décocher une zone garde la vraie (dents, regard…). La 1re fois, ~1 min de préparation en arrière-plan.")
        f.track("plus_regions", regions_now, lambda v: [c.setChecked(k in v) for k, c in cbs.items()])
        f.section("Bords")
        num("Marge du bord", "plus_mask_padding", 0, 20, int(d.get("plus_mask_padding", 6)), 6, lambda x: f"{x} %")
        num("Fondu du bord", "plus_mask_blur", 0, 10, int(d.get("plus_mask_blur", 4)), 4, lambda x: "net" if x == 0 else f"{x} %")
        num("Adoucissement des contours", "plus_mask_soften", 0, 15, int(d.get("plus_mask_soften", 5)), 5,
            lambda x: "net" if x == 0 else f"{x} px")

        f.tab("Stabilité")
        f.section("Cadrage du visage")
        num("Stabilisation du cadrage", "plus_stabilize", 0, 100, int(d.get("plus_stabilize", 50)), 50, lambda x: f"{x} %")
        num("Stabilisation de profil", "plus_profile_smooth", 0, 100, int(d.get("plus_profile_smooth", 70)), 70, pct("coupée"))
        check("Visage manqué : garder le précédent", "plus_hold", True)
        check("Assouplir les masques quand la tête bouge vite", "plus_motion_mask", True,
              "Sur une image floue, XSeg et le masque du modèle ouvrent des trous sur le vrai visage. Tête immobile : aucun changement.")
        landmarkers = ["detector", "hrffa"]
        f.combo("Repères du visage", ["Détecteur InsightFace (d'origine)", "HRFFA : tient mieux de profil"],
                landmarkers.index(d.get("plus_landmarker")) if d.get("plus_landmarker") in landmarkers else 0,
                lambda i: es("plus_landmarker", landmarkers[i]), key="plus_landmarker", default=0,
                tip="Points qui placent le visage généré (~36 Mo, téléchargé au 1er choix).")
        f.section("Charge et fluidité")
        check("Deux images en chevauchement (plus fluide)", "parallel_render", True)
        f.slider("Images/s traitées par l'IA", 8, 30, int(d.get("ai_fps", 0) or 30), lambda x: es("ai_fps", 0 if x >= 30 else x),
                 lambda x: "toutes" if x >= 30 else f"{x} img/s", 30, "ai_fps",
                 "Moins d'images traitées : moins de charge sur la carte graphique, mouvement moins fluide.")
        check("Lissage temporel (Interpolation)", "enable_interpolation", False)
        num("Force du lissage", "interpolation_weight", 0, 90, int(d.get("interpolation_weight", 0.0) * 100), 0, lambda x: f"{x} %", scale=100)
        det_sizes = [640, 320, 160]
        f.combo("Taille de détection", ["640 × 640 (précis)", "320 × 320", "160 × 160 (rapide)"],
                det_sizes.index(d.get("det_size", 640)), lambda i: es("det_size", det_sizes[i]), key="det_size", default=0)
        num("Détection toutes les", "det_interval", 1, 4, int(d.get("det_interval", 1)), 1, lambda x: f"{x} image(s)")

        f.tab("Anti-fuite")
        f.section("Ne jamais montrer le vrai visage")
        gp = self.pipeline.guard.p
        gp.update({k: v for k, v in self.settings.get("guard", {}).items() if k in GUARD_DEFAULTS})
        self.guard_on = f.check("Figer avant que le vrai visage apparaisse", gp["enabled"], lambda on: gp.__setitem__("enabled", on),
                                key="guard_enabled", default=GUARD_DEFAULTS["enabled"],
                                tip="Une image ne sort que si elle et les suivantes sont bien remplacées, sinon la dernière image sûre est répétée.")
        check("Surveiller la tête qui tourne vite", "plus_guard_check", True,
              "Moteur Plus : fige dès que deux systèmes de repères se contredisent. Utilise HRFFA (~36 Mo, téléchargé au 1er usage).")
        f.slider("Anticipation", 0, 8, gp["lookahead"], lambda x: gp.__setitem__("lookahead", x),
                 lambda x: "aucune" if x == 0 else f"{x} img (~{x * 33} ms)", GUARD_DEFAULTS["lookahead"], "guard_lookahead")
        f.slider("Confiance minimale", 0, 95, gp["min_score"], lambda x: gp.__setitem__("min_score", x), lambda x: f"{x} %",
                 GUARD_DEFAULTS["min_score"], "guard_min_score", "Plus haut : fige plus tôt (visage qui tourne).")
        f.slider("Gel maximal", 2, 100, gp["max_freeze"] // 100, lambda x: gp.__setitem__("max_freeze", x * 100),
                 lambda x: f"{x / 10:.1f} s", GUARD_DEFAULTS["max_freeze"] // 100, "guard_max_freeze")
        f.combo("Ensuite", AFTER_CHOICES, gp["after"], lambda i: gp.__setitem__("after", i), key="guard_after", default=GUARD_DEFAULTS["after"])

        f.tab("Origine")
        f.section("Fondu")
        check("Poisson Blend (fondu des bords)", "poisson_blend", False, "Avec le moteur d'origine seulement.")
        f.section("Mouth mask")

        def mouth(x):
            es("mouth_mask_size", float(x))
            es("mouth_mask", x > 0)
        ms = f.slider("Taille", 0, 100, int(d.get("mouth_mask_size", 0)), mouth, lambda x: "désactivé" if x == 0 else str(x),
                      0, "mouth_mask_size", "Avec le moteur d'origine seulement.")
        ms.sliderPressed.connect(lambda: es("show_mouth_mask_box", ms.value() > 0))
        ms.sliderReleased.connect(lambda: es("show_mouth_mask_box", False))
        num("Adoucissement", "mask_feather_ratio", 2, 40, int(d.get("mask_feather_ratio", 12)), 12)
        num("Lèvre inférieure", "mask_down_size", 0, 50, int(d.get("mask_down_size", 0.1) * 100), 10, lambda x: f"{x / 100:.2f}", scale=100)
        num("Lèvre supérieure", "mask_size", 50, 200, int(d.get("mask_size", 1.0) * 100), 100, lambda x: f"{x / 100:.2f}", scale=100)
        self.add("swap", "Moteur", page, f)
        self.plus_adv = {"plus_boost": 0, "plus_occluder": 0, "plus_landmarker": 0, "plus_weight": 50, "plus_regions": all_regions,
                         "plus_expr_upper": True, "plus_expr_lower": True, "plus_mask_padding": 6, "plus_mask_blur": 4,
                         "plus_mask_soften": 5, "plus_occ_threshold": 50, "plus_occ_islands": 0, "plus_occ_steady": 0,
                         "plus_guard_check": True, "plus_motion_mask": True}
        self.plus_best = dict(self.plus_adv, plus_model=0, plus_expression=80, plus_occlusion=True, plus_parser=True, plus_color=70,
                              plus_grain=100, plus_stabilize=50, plus_profile_smooth=70, plus_hold=True)

    def _plus_best(self):
        for key, value in self.plus_best.items():
            self.panels["dlc"][key][1](value)
        self.toast("Moteur Plus : réglages d'origine")

    def _plus_advanced(self):
        for key, value in self.plus_adv.items():
            self.panels["dlc"][key][1](value)
        self.toast("Réglages avancés d'origine")

    # ------------------------------------------------------------ Retouche
    def _page_retouch(self):
        p = self.pipeline.shape.p
        p.update({k: v for k, v in self.settings.get("reshape", {}).items() if k in SHAPE_DEFAULTS})
        self.shape_sliders = {}

        def preset(name):
            values = {k: 0 for k in SHAPE_KEYS + MAKEUP_KEYS + SHADE_KEYS}
            values.update(SHAPE_PRESETS[name])
            for k in SHAPE_KEYS + MAKEUP_KEYS + SHADE_KEYS:
                if k in self.panels["filtres"]:
                    self.panels["filtres"][k][1](values[k])
        page, f = self.form("filtres", "Retouche", "Appliquée au visage final, avant les effets webcam. Fonctionne aussi sans remplacement.",
                            "filtres", {n: lambda n=n: preset(n) for n in SHAPE_PRESETS})
        for title, items in SHAPE_GROUPS:
            f.tab(title.split(" ")[0] if title != "Maquillage et peau" else "Peau")
            f.section(title)
            for key, text, lo in items:
                fmt = signed if lo < 0 else pct()
                self.shape_sliders[key] = f.slider(text, lo, 100, int(p[key]), lambda x, key=key: p.__setitem__(key, x), fmt, 0, key)
                if key in SHADES:
                    k, label, shades = SHADES[key]
                    f.combo(label, [n for n, _ in shades], int(p[k]) % len(shades), lambda i, k=k: p.__setitem__(k, i), key=k, default=0)
        f.tab("Suivi")
        f.section("Stabilité des filtres")
        self.shape_sliders["smooth"] = f.slider("Stabilisation des repères", 0, 100, int(p["smooth"]), lambda x: p.__setitem__("smooth", x),
                                                lambda x: f"{x} %", SHAPE_DEFAULTS["smooth"], "smooth")
        f.combo("Repères du visage", ["InsightFace (d'origine)", "HRFFA : tient mieux de profil"], 1 if p["hrffa"] else 0,
                lambda i: p.__setitem__("hrffa", i), key="hrffa", default=0)
        f.check("Ne déformer que la personne (décor droit)", bool(p["protect_bg"]), lambda on: p.__setitem__("protect_bg", int(on)),
                key="protect_bg", default=True, tip="« Corps » suit alors la silhouette détectée (sinon : tout ce qui est sous le menton).")
        self.seg_status = f.label()
        self.add("wand", "Retouche", page, f)

    # ------------------------------------------------------------ Webcam et ambiance
    def _page_webcam(self):
        fx = self.pipeline.fx
        fx.p.update({k: v for k, v in self.settings.get("effects", {}).items() if k in DEFAULTS})
        sliders = {}

        def preset(name):
            values = dict(DEFAULTS, **WEBCAM_PRESETS[name])
            for k, s in sliders.items():
                s.setValue(values[k])
        page, f = self.form("effets", "Webcam", "Les défauts d'une vraie caméra, pour un rendu crédible.", "effets",
                            {n: lambda n=n: preset(n) for n in WEBCAM_PRESETS})
        for tab, title, specs in (
                ("Flux", "Compression", (("scale", "Résolution du flux", 20, 100, " %"), ("jpeg", "Qualité de compression", 5, 100, ""),
                                         ("glitch", "Chutes de débit", 0, 50, " ‰"), ("adaptive", "Compression selon le mouvement", 0, 100, " %"))),
                ("Capteur", "Capteur et objectif", (("blur", "Flou", 0, 30, ""), ("sharpen", "Sur-accentuation", 0, 150, " %"),
                                                    ("noise", "Bruit capteur", 0, 40, ""), ("chroma_noise", "Bruit de couleur", 0, 30, ""),
                                                    ("exposure", "Dérive exposition / WB", 0, 100, " %"),
                                                    ("contrast", "Contraste petit capteur", 0, 100, " %"),
                                                    ("motion_blur", "Flou de bougé", 0, 100, " %"),
                                                    ("vignette", "Vignettage / aberration", 0, 100, " %"))),
                ("Temps", "Latence et cadence", (("latency", "Latence", 0, 1000, " ms"), ("jitter", "Gigue", 0, 300, " ms"),
                                                 ("fps", "Cadence max", 5, 60, " img/s"), ("freeze", "Micro-gels", 0, 30, " ‰")))):
            f.tab(tab)
            f.section(title)
            for key, text, lo, hi, unit in specs:
                sliders[key] = f.slider(text, lo, hi, fx.p[key], lambda x, key=key: fx.p.__setitem__(key, x),
                                        lambda x, unit=unit: f"{x}{unit}", DEFAULTS[key], key)
        self.add("webcam", "Webcam", page, f)

    def _page_ambiance(self):
        gp = self.pipeline.fx.p
        page, f = self.form("effets", "Ambiance", "Simple étalonnage : la lumière d'une pièce, ~1 à 3 ms.")
        self.grade_sliders = {}
        f.section("Pièce")
        amb = f.combo("Ambiance", list(AMBIANCES), 0, lambda _i: None)

        def apply(i):
            values = {k: DEFAULTS[k] for k in self.grade_sliders}
            values.update(AMBIANCES[amb.itemText(i)])
            for k, s in self.grade_sliders.items():
                s.setValue(int(values[k]))
        amb.activated.connect(apply)
        f.section("Lumière")
        for key, text, lo, hi, fmt in (
                ("grade_temp", "Lumière chaude ↔ froide", 2000, 10000, lambda x: f"{x} K"),
                ("grade_dark", "Pièce sombre", 0, 100, pct("désactivée")),
                ("grade_exposure", "Exposition", -100, 100, lambda x: f"{x:+d}"),
                ("grade_contrast", "Contraste", -100, 100, lambda x: f"{x:+d}"),
                ("grade_saturation", "Saturation", -100, 100, lambda x: f"{x:+d}")):
            self.grade_sliders[key] = f.slider(text, lo, hi, int(gp[key]), lambda x, key=key: gp.__setitem__(key, x), fmt, DEFAULTS[key], key)
        f.section("Ampoule RGB")
        for key, text, lo, hi, fmt in (("grade_rgb", "Intensité", 0, 100, pct("désactivée")), ("grade_hue", "Couleur", 0, 359, lambda x: f"{x}°")):
            self.grade_sliders[key] = f.slider(text, lo, hi, int(gp[key]), lambda x, key=key: gp.__setitem__(key, x), fmt, DEFAULTS[key], key)
        self.add("sun", "Ambiance", page, f)

    # ------------------------------------------------------------ Sortie
    def _page_output(self):
        s, pipe = self.settings, self.pipeline
        OUTPUT_SIZES = pipe.sizes
        page, f = self.form("sortie", "Sortie", "", "sortie")
        f.tab("Caméra")
        f.section("Caméra virtuelle")
        self.vbtn = Toggle("Caméra virtuelle active")
        self.vbtn.toggled.connect(lambda on: setattr(pipe, "vcam_on", on))
        f.form.addRow(self.vbtn)
        self.vstatus = f.label()
        names = list(OUTPUT_SIZES)
        name = s.get("size", "1280×720")

        def size_changed(i):
            s["size"] = names[i]
            pipe.size = OUTPUT_SIZES[names[i]]
            pipe.restart_vcam()
        pipe.size = OUTPUT_SIZES[name]
        f.combo("Résolution", names, names.index(name), size_changed, key="size", default=0,
                tip="Pour Snapchat : 1280×720. Au-delà, Snapchat n'en garde qu'une partie (image zoomée).")
        pipe.vcam_fps = int(s.get("vcam_fps", 30))

        def fps_changed(i):
            s["vcam_fps"] = (30, 60)[i]
            pipe.vcam_fps = s["vcam_fps"]
            pipe.restart_vcam()
        f.combo("Images/s annoncées", ["30", "60"], 0 if pipe.vcam_fps == 30 else 1, fps_changed, key="vcam_fps", default=0)
        pipe.fill = int(s.get("framing", 0)) == 0

        def fit_changed(i):
            s["framing"] = i
            pipe.fill = i == 0
        f.combo("Cadrage", ["Remplir (recadrer)", "Ajuster (bandes noires)"], int(s.get("framing", 0)), fit_changed, key="framing", default=0)

        f.tab("Cadrage")
        f.section("Zoom et position")
        fp = pipe.framing.p
        fp.update({k: v for k, v in s.get("framing_out", {}).items() if k in FRAMING_DEFAULTS})
        self.frame_sliders = {
            "zoom": f.slider("Zoom", 100, 400, int(fp["zoom"]), lambda x: fp.__setitem__("zoom", x),
                             lambda x: "aucun" if x == 100 else f"×{x / 100:.2f}", 100, "zoom",
                             "Zoom numérique avant les effets webcam. Au-delà de ×2, l'image devient plus floue (720p reçu)."),
            "pan_x": f.slider("Position horizontale", -100, 100, int(fp["pan_x"]), lambda x: fp.__setitem__("pan_x", x),
                              lambda x: "centre" if x == 0 else (f"← {-x}" if x < 0 else f"{x} →"), 0, "pan_x"),
            "pan_y": f.slider("Position verticale", -100, 100, int(fp["pan_y"]), lambda x: fp.__setitem__("pan_y", x),
                              lambda x: "centre" if x == 0 else (f"↑ {-x}" if x < 0 else f"{x} ↓"), 0, "pan_y"),
        }
        f.button("Recentrer (zoom ×1)", self._recenter)
        f.section("Suivi du visage")
        f.check("Suivre le visage (cadrage automatique)", bool(fp["follow"]), lambda on: fp.__setitem__("follow", on), key="follow",
                default=bool(FRAMING_DEFAULTS["follow"]), tip="Avec le suivi, la position décale le visage dans le cadre.")
        f.slider("Douceur du suivi", 0, 100, int(fp["follow_smooth"]), lambda x: fp.__setitem__("follow_smooth", x), lambda x: f"{x} %",
                 FRAMING_DEFAULTS["follow_smooth"], "follow_smooth")

        f.tab("Captures")
        f.section("Enregistrer la sortie")
        self.rec_seconds = next((i for i, (_t, sec) in enumerate(REC_TIMES) if sec == s.get("rec_seconds", 10)), 0)
        f.combo("Durée", [t for t, _ in REC_TIMES], self.rec_seconds, self._rec_time)
        self.rec_page = f.button("Enregistrer", self.toggle_record)
        self.rec_label = f.label()
        row = QHBoxLayout()
        shot = QPushButton("Photo de la sortie")
        shot.clicked.connect(self.snapshot)
        folder = QPushButton("Ouvrir le dossier")
        folder.clicked.connect(self.open_captures)
        row.addWidget(shot)
        row.addWidget(folder)
        f.form.addRow(row)
        self.add("output", "Sortie", page, f)

    def _recenter(self):
        self.frame_sliders["zoom"].setValue(100)
        self.frame_sliders["pan_x"].setValue(0)
        self.frame_sliders["pan_y"].setValue(0)

    def _rec_time(self, i):
        self.rec_seconds = i
        self.settings["rec_seconds"] = REC_TIMES[i][1]

    def toggle_record(self):
        if self.pipeline.recording():
            self.pipeline.stop_recording()
            return
        self.pipeline.start_recording(REC_TIMES[self.rec_seconds][1])
        self.toast(f"Enregistrement : {REC_TIMES[self.rec_seconds][0]}")

    def snapshot(self):
        path = self.pipeline.snapshot()
        self.toast(f"Photo enregistrée : {os.path.basename(path)}" if path else "Aucune image à enregistrer pour l'instant")

    def open_captures(self):
        folder = os.path.join(DATA_DIR, "captures")
        os.makedirs(folder, exist_ok=True)
        os.startfile(folder)

    # ------------------------------------------------------------ Micro
    def _page_mic(self):
        s, pipe = self.settings, self.pipeline
        mic = pipe.mic
        mp = mic.p
        mp.update({k: v for k, v in s.get("mic", {}).items() if k in MIC_DEFAULTS})
        mic.extra_delay_ms = lambda: pipe.latency_ms
        self.mdev = s.setdefault("mic_devices", {})
        keys = []

        def preset(name):
            values = dict(MIC_DEFAULTS, **MIC_PRESETS[name])
            for k in keys:
                self.panels["micro"][k][1](int(values[k]))
        page, f = self.form("micro", "Micro", "Une voix de smartphone simulée, du micro à l'appel.", "micro",
                            {n: lambda n=n: preset(n) for n in MIC_PRESETS})

        def sl(key, text, lo, hi, fmt=str, sound=True):
            if sound:
                keys.append(key)
            return f.slider(text, lo, hi, int(mp[key]), lambda x, key=key: mp.__setitem__(key, x), fmt, int(MIC_DEFAULTS[key]), key)

        f.tab("Appareils")
        f.section("Entrée et sortie")
        from PySide6.QtWidgets import QComboBox
        self.dev_in, self.dev_out, self.dev_mon = QComboBox(), QComboBox(), QComboBox()
        self.mic_status = f.label()
        self._fill_devices()
        for combo, text in ((self.dev_in, "Entrée"), (self.dev_out, "Sortie (câble)"), (self.dev_mon, "Écoute")):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.currentIndexChanged.connect(lambda _i: pipe.audio.running and self._mic_run(True))
            f.form.addRow(text, combo)
        self.run_btn = Toggle("Micro actif")
        self.run_btn.toggled.connect(self._mic_run)
        f.form.addRow(self.run_btn)
        f.check("Démarrer au lancement", s.get("mic_autostart", False), lambda on: s.__setitem__("mic_autostart", on))
        rescan = QPushButton("Actualiser la liste")
        rescan.clicked.connect(self._rescan)
        f.form.addRow(rescan)
        f.section("Niveaux")
        self.meters = {"in": Meter("Entrée"), "out": Meter("Sortie")}
        for m in self.meters.values():
            f.form.addRow(m)
        f.note("Micro du téléphone : entrée « Micro du téléphone (PhoneCam) », puis « Micro » sur la page du téléphone. "
               "Les autres logiciels prennent « CABLE-B Output » comme micro.")

        f.tab("Voix")
        f.section("Téléphone et réseau")
        f.combo("Type d'appel", [c[0] for c in CODECS], int(mp["codec"]), lambda i: mp.__setitem__("codec", i), key="codec",
                default=int(MIC_DEFAULTS["codec"]))
        keys.append("codec")
        sl("bitrate", "Qualité du codec (débit)", 5, 100, lambda x: f"{x} %")
        sl("packet_loss", "Pertes de paquets", 0, 80, lambda x: f"{x} ‰")
        sl("noise_suppression", "Réduction de bruit", 0, 100, lambda x: f"{x} %")
        sl("agc", "Gain automatique (AGC)", 0, 100, lambda x: f"{x} %")
        f.section("Micro du téléphone")
        sl("distance", "Distance bouche → téléphone", 3, 150, lambda x: f"{x} cm")
        sl("saturation", "Saturation (voix forte)", 0, 100, lambda x: f"{x} %")
        sl("mic_noise", "Souffle de la capsule", 0, 100, lambda x: f"{x} %")
        sl("tone", "Timbre (chaud ↔ brillant)", -100, 100, signed)
        keys_air = sl("air", "Recréer les aigus", 0, 100, lambda x: f"{x} %")
        keys_air.setToolTip("Écouteurs Bluetooth : le micro s'arrête à 8 kHz, ceci aide pour les modes 48 kHz.")

        f.tab("Pièce")
        f.section("Chambre")
        sl("room", "Réverbération", 0, 150, lambda x: f"{x} %")
        sl("room_size", "Taille de la pièce", 6, 30, lambda x: f"{x} m²")
        sl("damping", "Amortissement (meubles, rideaux)", 0, 100, lambda x: f"{x} %")
        f.section("Bruits de fond (légers, aléatoires)")
        for key, text in (("room_tone", "Ambiance de la pièce"), ("fan", "Ventilateur / PC"), ("street", "Rue au loin (voitures)"),
                          ("house", "Maison (porte, pas, objet)"), ("handling", "Manipulation du téléphone")):
            sl(key, text, 0, 100, pct())

        f.tab("Réglage")
        f.section("Gains et synchro")
        sl("input_gain", "Gain d'entrée", -24, 24, lambda x: f"{x:+d} dB", False)
        sl("output_gain", "Gain de sortie", -24, 12, lambda x: f"{x:+d} dB", False)
        sl("delay", "Retard ajouté", 0, 1000, lambda x: f"{x} ms", False)
        f.check("Ajouter la latence de la vidéo (synchro lèvres)", mp["sync_video"], lambda on: mp.__setitem__("sync_video", on),
                key="sync_video", default=bool(MIC_DEFAULTS["sync_video"]))
        f.section("Écoute et test")
        f.slider("Volume de l'écoute", 0, 100, int(s.get("mic_monitor", 80)),
                 lambda x: (s.__setitem__("mic_monitor", x), setattr(pipe.audio, "monitor_gain", x / 100)), lambda x: f"{x} %", 80)
        pipe.audio.monitor_gain = s.get("mic_monitor", 80) / 100
        f.check("Son brut (comparer sans la simulation)", not mp["enabled"], lambda on: mp.__setitem__("enabled", not on))
        self.add("mic", "Micro", page, f)
        if s.get("mic_autostart") and self.dev_in.count() and self.dev_out.count():
            self.run_btn.setChecked(True)

    def _fill_devices(self):
        try:
            ins, outs = [phonemic.PHONE_INPUT] + phonemic.devices("input"), phonemic.devices("output")
        except Exception as e:  # noqa: BLE001
            ins, outs = [], []
            self.mic_status.setText(f"Audio indisponible : {e}")
        for combo, names, key, default, none in ((self.dev_in, ins, "input", "CABLE-A Output", False),
                                                 (self.dev_out, outs, "output", "CABLE-B Input", False),
                                                 (self.dev_mon, outs, "monitor", None, True)):
            combo.blockSignals(True)
            combo.clear()
            if none:
                combo.addItem("Aucune")
            combo.addItems(names)
            want = self.mdev.get(key) or next((n for n in names if default and n.startswith(default)), None)
            if want and combo.findText(want) >= 0:
                combo.setCurrentText(want)
            combo.blockSignals(False)

    def _mic_run(self, on):
        a = self.pipeline.audio
        if on:
            self.mdev.update(input=self.dev_in.currentText(), output=self.dev_out.currentText(),
                             monitor=self.dev_mon.currentText() if self.dev_mon.currentIndex() > 0 else "")
            a.start(self.mdev["input"], self.mdev["output"], self.mdev["monitor"] or None)
            if not a.running:
                self.run_btn.setChecked(False)
        else:
            a.stop()

    def _rescan(self):
        if self.pipeline.audio.running:
            self.mic_status.setText("Arrête le micro avant d'actualiser la liste.")
            return
        phonemic.rescan()
        self._fill_devices()

    # ------------------------------------------------------------ Connexion
    def _page_link(self, make_qr):
        server, snap = self.server, self.pipeline.snap
        page, f = self.form("connexion", "Connexion")
        vin = self.vin = self.settings.setdefault("video_input", {"source": "phone", "cam": "", "size": "1280×720"})
        f.section("Source vidéo")
        f.combo("", ["Téléphone", "Webcam du PC"], int(vin["source"] == "webcam"), self._source_changed, name="Source vidéo",
                tip="L'un ou l'autre : avec la webcam, les images d'un téléphone connecté sont ignorées.")
        self.cam_combo = QComboBox()
        self.cam_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cam_combo.setMinimumContentsLength(8)
        self.cam_combo.activated.connect(lambda _i: self._webcam_apply())
        f.form.addRow("Webcam", self.cam_combo)
        names = list(webcam.SIZES)
        self.cam_size = f.combo("Résolution", names, names.index(vin["size"]) if vin["size"] in names else 0,
                                lambda i: (vin.__setitem__("size", names[i]), self._webcam_apply()))
        rescan = QPushButton("Actualiser la liste")
        rescan.clicked.connect(self._fill_cams)
        f.form.addRow(rescan)
        self.cam_status = f.label()
        self.cam_rows = [self.cam_combo, self.cam_size, rescan]
        self._fill_cams()
        self._source_changed(int(vin["source"] == "webcam"))
        self.phone_card = f.section("Téléphone")
        self.phone_status = f.label()
        f.section("Câble USB (le plus fluide)")
        f.note("Débogage USB activé, câble branché : la page s'ouvre seule sur le téléphone.")
        self.usb_btn = QPushButton("Connecter par câble")
        self.usb_btn.setObjectName("primary")
        self.usb_btn.setEnabled(usb.available())
        self.usb_btn.clicked.connect(self._usb_connect)
        f.form.addRow(self.usb_btn)
        self.usb_status = f.label()
        self.usb_status.setText("" if usb.available() else "ADB absent (dossier tools).")
        f.section("Wi-Fi", open_=False)
        qr = QLabel()
        qr.setPixmap(to_pixmap(make_qr(server.url)))
        qr.setAlignment(Qt.AlignCenter)
        f.form.addRow(qr)
        url = QLabel(server.url)
        url.setObjectName("note")
        url.setTextInteractionFlags(Qt.TextSelectableByMouse)
        url.setWordWrap(True)
        f.form.addRow(url)
        f.note("Scanne le QR, accepte l'avertissement de certificat (généré localement).")
        f.section("Snapchat Web")
        b = QPushButton("Ouvrir Snapchat Web")
        b.setEnabled(find_browser() is not None)
        b.clicked.connect(snap.open)
        f.form.addRow(b)
        self.snap_status = f.label()
        f.note("Un Edge séparé (data\\snap-browser), ouvert seul quand l'appli passe en Navigation. La première fois : se connecter, "
               "choisir la caméra « OBS Virtual Camera » et le micro « CABLE-B Output » dans Snapchat, puis garder la caméra "
               "virtuelle et le micro démarrés.")
        self.add("plug", "Connexion", page, f)

    def _fill_cams(self):
        try:
            names = webcam.cameras()
        except Exception as e:  # noqa: BLE001
            names = []
            self.cam_status.setText(f"Liste des webcams indisponible : {e}")
        c = self.cam_combo
        c.blockSignals(True)
        c.clear()
        c.addItems(names)
        if self.vin["cam"] in names:
            c.setCurrentText(self.vin["cam"])
        c.blockSignals(False)
        if not names:
            self.cam_status.setText("Aucune webcam trouvée (les caméras virtuelles ne sont pas proposées).")
        elif self.vin["source"] == "webcam" and c.currentText() != self.vin["cam"]:
            self._webcam_apply()

    def _source_changed(self, i):
        self.vin["source"] = self.server.source = "webcam" if i else "phone"
        for w in self.cam_rows:
            w.setEnabled(bool(i))
        self._webcam_apply()

    def _webcam_apply(self):
        cam = self.pipeline.webcam
        name = self.cam_combo.currentText()
        if self.vin["source"] != "webcam" or not name:
            cam.stop()
            return
        self.vin["cam"] = name
        cam.start(name, webcam.SIZES.get(self.vin["size"], (1280, 720)))

    def _usb_connect(self):
        self.usb_btn.setEnabled(False)
        self.usb_status.setText("Connexion…")
        server = self.server
        threading.Thread(target=lambda: self.bridge.usb.emit(usb.connect(server.usb_port, server.usb_url, server.app_usb_url)),
                         daemon=True).start()

    def _usb_done(self, msg):
        self.usb_status.setText(msg)
        self.usb_btn.setEnabled(True)

    # ------------------------------------------------------------ Presets
    def _page_presets(self):
        page, f = self.form("presets", "Presets", "Des profils, et dans chacun plusieurs presets qui règlent toutes les branches d'un coup.")
        f.section("Profil")
        self.presets_form = f.form
        self.presets_index = self.rail.add("layers", "Presets", page)
        f.bind(self.rail, self.presets_index)

    def open_presets(self):
        self.rail.select(self.presets_index)

    def load_preset(self, preset):
        """Seuls les réglages différents sont touchés : changer de preset ne recharge pas un modèle ou un visage déjà en place."""
        for section, values in preset.items():
            panel = self.panels["effets" if section in ("webcam", "ambiance") else section]
            for k, v in values.items():
                if k in panel and panel[k][0]() != v:
                    panel[k][1](v)

    # ------------------------------------------------------------ Diagnostic
    def _page_diag(self):
        page, f = self.form("diag", "Diagnostic", "Où passe le temps de chaque image, et l'état du PC.")
        f.section("Latence")
        self.spark = Sparkline()
        f.form.addRow(self.spark)
        self.rate_label = f.label()
        f.section("Étapes")
        self.stage_label = f.label()
        f.section("Carte graphique")
        self.gpu_label = f.label()
        self.gpu_label.setText("Mesure à l'ouverture de cette page.")
        f.section("Capture de diagnostic")
        f.button("Capturer 15 s (image rendue et valeurs du moteur)", lambda: self.pipeline.start_diag())
        self.diag_label = f.label()
        f.section("Fichiers")
        row = QHBoxLayout()
        log = QPushButton("Journal d'erreurs")
        log.clicked.connect(self._open_log)
        row.addWidget(log)
        folder = QPushButton("Captures")
        folder.clicked.connect(self.open_captures)
        row.addWidget(folder)
        f.form.addRow(row)
        self.diag_index = self.rail.add("gauge", "Diagnostic", page)
        f.bind(self.rail, self.diag_index)

    def _open_log(self):
        path = os.path.join(DATA_DIR, "phonecam-errors.log")
        if os.path.isfile(path):
            os.startfile(path)
        else:
            self.toast("Aucune erreur enregistrée")

    def _poll_gpu(self):
        """nvidia-smi seulement quand la page est ouverte : rien en temps normal."""
        if self.pending_gpu:
            return
        self.pending_gpu = True

        def work():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                                     creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip().split(",")
                u, used, total, temp = (x.strip() for x in out[:4])
                self.bridge.gpu.emit(f"Charge {u} % · mémoire {int(used) / 1024:.1f} / {int(total) / 1024:.1f} Go · {temp} °C")
            except Exception:  # noqa: BLE001
                self.bridge.gpu.emit("Mesure indisponible.")
            self.pending_gpu = False
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ commandes d'antenne
    def _controls(self):
        self.quick = []
        row = QHBoxLayout()
        row.setSpacing(6)

        def toggle(text, widget, tip):
            b = QPushButton(text)
            b.setObjectName("air")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            b.setChecked(widget.isChecked())
            b.toggled.connect(lambda on: widget.isChecked() != on and widget.setChecked(on))
            row.addWidget(b)
            self.quick.append((b, widget))
            return b
        toggle("Remplacement", self.swap_on, "Remplacer le visage (F7)")
        toggle("Moteur Plus", self.plus_on, "Réalisme, expressions, objets devant le visage")
        toggle("Caméra virtuelle", self.vbtn, "Démarrer / arrêter la caméra virtuelle (F5)")
        toggle("Micro", self.run_btn, "Démarrer / arrêter le micro (F6)")
        row.addStretch()

        def tool(name, tip, fn, checkable=False):
            b = QPushButton()
            b.setObjectName("icon")
            b.setIcon(icon(name, 19))
            b.setIconSize(QSize(19, 19))
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(fn)
            row.addWidget(b)
            return b
        self.pause_btn = tool("pause", "Pause : fige l'image et coupe le micro, comme une coupure réseau (F3)", self.toggle_pause, True)
        self.black_btn = tool("webcam", "Couper l'image : noir sur la caméra virtuelle (F4)", self.toggle_black, True)
        sep = QFrame()
        sep.setFixedSize(1, 22)
        sep.setStyleSheet(f"background:{C['line2']};")
        row.addWidget(sep)
        self.preview_btn = tool("output", "Couper l'aperçu : un peu moins de travail pour le PC (Ctrl+P)", self.toggle_preview, True)
        self.compare_btn = tool("compare", "Comparer avec l'image du téléphone, dans l'aperçu seulement (F10)", self.toggle_compare, True)
        self.grid_btn = tool("grid", "Grille de cadrage (ne sort pas sur la caméra)", self.toggle_guides, True)
        tool("shot", "Photo de la sortie (F8)", self.snapshot)
        self.rec_btn = tool("rec", "Enregistrer la sortie (F9)", self.toggle_record, True)
        tool("expand", "Plein écran (F11)", lambda: self.on_full and self.on_full())
        self.eye_btn = tool("sidebar", "Afficher ou masquer le panneau (Ctrl+B)", self.rail.toggle_panel)
        self.bar.addLayout(row)
        info = QHBoxLayout()
        self.stats = QLabel()
        self.stats.setObjectName("mono")
        info.addWidget(self.stats, 1)
        self.bar.addLayout(info)

    def toggle_pause(self):
        self.pipeline.snap._set_out("paused", not self.pipeline.snap.out_state["paused"])

    def toggle_black(self):
        self.pipeline.snap._set_out("black", not self.pipeline.snap.out_state["black"])

    def toggle_compare(self):
        self.compare = not self.compare
        self.compare_btn.setChecked(self.compare)

    def toggle_guides(self):
        self.view.guides = not self.view.guides
        self.grid_btn.setChecked(self.view.guides)
        self.view.update()

    def toggle_preview(self):
        self.preview_on = not self.preview_on
        self.preview_btn.setChecked(not self.preview_on)
        if not self.preview_on:
            self.view.clear_image("Aperçu coupé")
        self.toast("Aperçu coupé : un peu moins de travail pour le PC" if not self.preview_on else "Aperçu rétabli")

    def _on_preview(self, img):
        """Fil de sortie : rien du tout quand l'aperçu ne se voit pas."""
        if not (self.preview_on and self.visible_now):
            return
        now = time.perf_counter()
        if now - self.last < 1 / 61:
            return
        self.last = now
        pw, ph = self.target
        h, w = img.shape[:2]
        s = min(pw / w, ph / h)
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR) if s < 1 else img
        before = None
        if self.compare:
            raw = self.server.latest()[1]
            if raw is not None:        # image reçue du téléphone : dans l'aperçu local, jamais sur la sortie
                before = to_qimage(self.pipeline.fitted(raw, (small.shape[1], small.shape[0])))
        self.bridge.frame.emit(to_qimage(small), before)

    def _frame(self, qimg, before):
        self.view.set_qimage(qimg, before)

    def _overlay(self, p):
        """Liste de départ tant qu'aucune image n'arrive ; états importants par-dessus l'image."""
        pipe = self.pipeline
        if not self.view.has_image():
            if not self.preview_on:
                return
            first = (("Webcam active", self.server.fps() > 0, "Connexion") if self.server.source == "webcam"
                     else ("Téléphone connecté", self.server.connected, "Connexion"))
            steps = [first, ("Visage choisi", bool(self.settings.get("source")), "Visage"),
                     ("Caméra virtuelle démarrée", pipe.vcam_on, "bouton ci-dessous"), ("Micro démarré", pipe.audio.running, "bouton ci-dessous")]
            w, h = self.view.width(), self.view.height()
            x, y = (w - 320) / 2, (h - 170) / 2
            p.setRenderHint(p.RenderHint.Antialiasing)
            font = p.font()
            big = type(font)(font)
            big.setFamily("Bahnschrift")
            big.setPointSizeF(15)
            p.setFont(big)
            p.setPen(QColor(C["text"]))
            p.drawText(QRectF(x, y, 320, 30), Qt.AlignLeft, "Prêt pour l'antenne ?")
            p.setFont(font)
            for i, (text, ok, hint) in enumerate(steps):
                cy = y + 54 + i * 30
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(C["good"] if ok else C["line2"]))
                p.drawEllipse(QPointF(x + 8, cy), 6, 6)
                p.setPen(QColor(C["text"] if ok else C["dim"]))
                p.drawText(QRectF(x + 26, cy - 12, 200, 24), Qt.AlignVCenter, text)
                if not ok:
                    p.setPen(QColor(C["faint"]))
                    p.drawText(QRectF(x + 200, cy - 12, 120, 24), Qt.AlignVCenter | Qt.AlignRight, hint)
            return
        tags = []
        if pipe.recording():
            tags.append("● Enregistrement")
        if pipe.guard.frozen:
            tags.append("Image figée (anti-fuite)")
        st = pipe.snap.out_state
        if st["paused"]:
            tags.append("Pause")
        if st["black"]:
            tags.append("Image coupée")
        x = 12
        for t in tags:
            Viewer._tag(p, x, 12, t)
            x += p.fontMetrics().horizontalAdvance(t) + 24

    def sync_engine(self):
        e, s = self.pipeline.engine, self.settings
        for k, val in s.get("dlc", {}).items():
            e.set_option(k, val)
        mode = s.get("mode", "single")
        e.set_option("many_faces", mode == "many")
        e.set_option("map_faces", mode == "map")
        if s.get("source") and os.path.isfile(s["source"]):
            e.set_source(s["source"], s.get("source_more", []))
        if self.pairs:
            e.set_pairs([tuple(p) for p in self.pairs])

    def tick(self):
        pipe, server = self.pipeline, self.server
        win = self.window()
        self.visible_now = self.isVisible() and not win.isMinimized()
        k = self.view.devicePixelRatioF()       # l'aperçu en vrais pixels d'écran : net sur un écran à 150 %
        self.target = (max(160, int(self.view.width() * k)), max(90, int(self.view.height() * k)))
        if pipe.restart_wanted:
            self.persist_cb()
            pipe.restart_now()
        if server.error:
            text = f"Erreur serveur : {server.error}"
        elif server.connected:
            text = f"Téléphone connecté ({server.link})."
        else:
            text = "Téléphone non connecté."
        cam_on = server.source == "webcam"
        if cam_on and server.connected:
            text += " Ses images sont ignorées : la source vidéo est la webcam du PC."
        self.phone_status.setText(text)
        if cam_on or self.cam_combo.count():
            self.cam_status.setText(pipe.webcam.status if cam_on else "Source : téléphone.")
        e = pipe.engine
        self.engine_text = e.status if e else "Démarrage du moteur…"
        shape = pipe.shape
        seg, h3, hr = shape.segmenter, shape.head3d, shape.hrffa
        self.seg_status.setText(("Détourage : modèle MODNet absent (activer le moteur Plus le télécharge)."
                                 if not seg.available() else seg.status + (f" {seg.ms:.1f} ms" if seg.ms else ""))
                                + (f"\n{h3.status} {h3.ms:.1f} ms" if h3.status else "")
                                + (f"\n{hr.status} {hr.ms:.1f} ms" if shape.p["hrffa"] and hr.status else ""))
        plus = e._plus if e and e._plus is not None and e.plus_enabled else None
        if e and e.plus_status:
            note_ = getattr(plus, "note", "") if plus else ""
            self.plus_status.setText(e.plus_status + (f"\n{note_}" if note_ else ""))
            if not e.plus_enabled and self.plus_on.isChecked():
                self.plus_on.setChecked(False)  # échec du chargement
        if e and pipe.engine_ready and not self.synced:
            self.synced = True
            self.sync_engine()
        dec = server.decoder
        got, done, out = server.fps(), pipe.process_rate.value(), pipe.output_rate.value()
        self.stats.setText(f"{got:.0f} reçues   ·   {done:.0f} traitées   ·   {out:.0f} sorties   ·   {pipe.latency_ms:.0f} ms de latence"
                           + (f"   ·   {pipe.guard.frozen_total} images retenues" if pipe.guard.frozen_total else ""))
        self.vstatus.setText(pipe.vcam_status)
        self.rec_label.setText(pipe.rec_status)
        self.diag_label.setText(pipe.diag_status)
        self.rec_page.setText("Arrêter" if pipe.recording() else "Enregistrer")
        self.rec_btn.setChecked(pipe.recording())
        st = pipe.snap.out_state
        self.pause_btn.setChecked(st["paused"])
        self.black_btn.setChecked(st["black"])
        self.snap_status.setText(pipe.snap.status + (" · appli en mode " + {"call": "Appel"}.get(pipe.snap.mode, "Caméra"))
                                 + "".join(f" · {t}" for k, t in (("paused", "PAUSE"), ("black", "image coupée"), ("muted", "micro coupé")) if st[k]))
        mic = pipe.mic
        running = pipe.audio.running
        self.meters["in"].setValue(mic.in_db if running else -60)
        self.meters["out"].setValue(mic.out_db if running else -60)
        phone_wait = running and self.mdev.get("input") == phonemic.PHONE_INPUT and not server.audio_active()
        self.mic_status.setText(pipe.audio.status + (f" · traitement {mic.ms:.1f} ms / 10 ms · coupures {pipe.audio.underruns()}" if running else "")
                                + ("\nEn attente du son du téléphone : active « Micro » sur sa page et appuie sur Démarrer." if phone_wait else ""))
        if self.run_btn.isChecked() != running:
            self.run_btn.blockSignals(True)
            self.run_btn.setChecked(running)
            self.run_btn.blockSignals(False)
        if self.vbtn.isChecked() != pipe.vcam_on:
            self.vbtn.setChecked(pipe.vcam_on)
        for b, w in self.quick:
            if b.isChecked() != w.isChecked():
                b.blockSignals(True)
                b.setChecked(w.isChecked())
                b.blockSignals(False)
        source = ((got > 0, "Webcam") if cam_on else
                  (server.connected, server.link if server.connected else "Téléphone"))
        self.states = {"phone": source,
                       "engine": (pipe.engine_ready, "Moteur Plus" if e and e.plus_enabled else "Moteur"),
                       "vcam": (pipe.vcam_on, "Caméra"), "mic": (running, "Micro")}
        if not self.view.has_image():
            self.view.update()
        elif got == 0 and out == 0 and not pipe.snap.out_state["paused"]:
            self.view.clear_image()         # téléphone parti : retour à la liste de départ
        if self.rail.current() == self.diag_index and self.visible_now:
            self.spark.push(pipe.latency_ms)
            self.rate_label.setText(f"Actuelle : {pipe.latency_ms:.0f} ms. Reçu {got:.0f} · traité {done:.0f} · sortie {out:.0f} img/s.")
            lines = [f"Décodage ({dec.codec})   {dec.ms:.1f} ms", f"Moteur   {e.ms if e else 0:.1f} ms",
                     f"Filtres   {shape.ms:.1f} ms", f"Effets   {pipe.fx.ms:.1f} ms"]
            if plus is not None:
                lines += [f"   {k}   {v:.1f} ms" for k, v in plus.ms.items()]
            self.stage_label.setText("\n".join(lines))
            if int(time.time()) % 2 == 0:
                self._poll_gpu()

    def _remote_tick(self):
        """Menu Paramètres de l'appli : il agit sur les vrais curseurs, PC et téléphone restent d'accord."""
        snap = self.pipeline.snap
        remote = {"zoom": self.frame_sliders["zoom"], "dark": self.grade_sliders["grade_dark"],
                  "blush": self.shape_sliders["blush"], "guard": self.guard_on}
        while snap.pc_wanted:
            key, value = snap.pc_wanted.popitem()
            w = remote.get(key)
            if w is self.guard_on:
                w.setChecked(bool(value))
            elif w is not None:
                w.setValue(int(value))
        values = {k: w.isChecked() if w is self.guard_on else w.value() for k, w in remote.items()}
        if values != self.remote_sent or snap.pc_ask:
            self.remote_sent.update(values)
            snap.pc_ask = False
            self.server.send_json({"t": "pcset", **values}, key="pcset")

    def state(self):
        p = self.pipeline
        s = self.settings
        s["ui_tab"] = self.rail.current()
        s["effects"] = dict(p.fx.p)
        s["reshape"] = dict(p.shape.p)
        s["framing_out"] = dict(p.framing.p)
        s["guard"] = dict(p.guard.p)
        s["mic"] = dict(p.mic.p)

    def shot_all(self):
        """Tous les réglages de l'espace, pour un profil."""
        return {panel: {k: g() for k, (g, _p) in items.items()} for panel, items in self.panels.items()}

    def load_all(self, data):
        for panel, items in data.items():
            for k, v in items.items():
                if k in self.panels.get(panel, {}):
                    self.panels[panel][k][1](v)
