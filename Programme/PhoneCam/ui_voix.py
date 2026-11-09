"""Page Voix du Direct : changeur de voix (RVC) avec Smart, un modèle .pth (+ .index) à la fois.

Le son passe : micro → voix transformée → simulation du micro de téléphone (page Micro) → sortie.
"""
import os

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QWidget

from ui_kit import C, Meter, Toggle, note
from voix import DEFAULTS as VOICE_DEFAULTS, find_index

MODELS = "Modèles de voix RVC (*.pth)"


def signed(x, unit=""):
    return f"{x:+d}{unit}" if x else f"0{unit}"


def voice_folders(s):
    """Les dossiers de voix choisis (le premier est l'ancien « voice_library »), sans doublon."""
    out, seen = [], set()
    for f in [s.get("voice_library", "")] + list(s.get("voice_folders", [])):
        if f and os.path.normcase(f) not in seen:
            seen.add(os.path.normcase(f))
            out.append(f)
    return out


def default_voices(s, vp=None):
    """Version distribuée, première fois : le dossier « Voix » fourni (à côté du lanceur) devient le dossier
    de voix, et sa première voix est choisie (le changeur reste coupé tant qu'on ne l'active pas)."""
    from server import VOICES_DIR
    if not VOICES_DIR or not os.path.isdir(VOICES_DIR) or s.get("voices_seeded"):
        return
    s["voices_seeded"] = True
    if not any(os.path.normcase(f) == os.path.normcase(VOICES_DIR) for f in voice_folders(s)):
        set_voice_folders(s, [VOICES_DIR] + voice_folders(s))
    found = sorted(p for _l, p in scan_voices(s) if os.path.normcase(p).startswith(os.path.normcase(VOICES_DIR)))
    if found and vp is not None and not (vp.get("pth") and os.path.isfile(vp["pth"])):
        vp["pth"], vp["index"] = found[0], find_index(found[0])
        s.setdefault("voix", {}).update(pth=vp["pth"], index=vp["index"])


def set_voice_folders(s, folders):
    s["voice_library"] = folders[0] if folders else ""
    s["voice_folders"] = list(folders[1:])


def scan_voices(s):
    """(nom affiché, chemin) de chaque modèle .pth dans les dossiers de voix (3 niveaux au plus)."""
    found, seen = [], set()
    for folder in voice_folders(s):
        if not os.path.isdir(folder):
            continue
        for root, _dirs, files in os.walk(folder):
            if root[len(folder):].count(os.sep) > 2:
                continue
            for fn in files:
                if fn.lower().endswith(".pth") and not fn.startswith(("G_", "D_", "f0G", "f0D")):
                    path = os.path.join(root, fn)
                    if os.path.normcase(path) in seen:
                        continue
                    seen.add(os.path.normcase(path))
                    rel = os.path.relpath(path, folder)
                    label = os.path.splitext(fn)[0] if os.path.dirname(rel) in ("", os.path.splitext(fn)[0]) else \
                        f"{os.path.dirname(rel)} · {os.path.splitext(fn)[0]}"
                    found.append((label, path))
    return found


def folders_menu(button, s, changed):
    """Menu « Dossiers… » : ajouter un dossier de voix, en ouvrir ou en retirer un."""
    from PySide6.QtWidgets import QMenu

    def add():
        start = (voice_folders(s) or [os.path.expanduser("~")])[0]
        d = QFileDialog.getExistingDirectory(button.window(), "Dossier où sont rangées tes voix", start)
        if d:
            d = os.path.normpath(d)
            fs = voice_folders(s)
            if os.path.normcase(d) not in {os.path.normcase(f) for f in fs}:
                set_voice_folders(s, fs + [d])
            changed()

    def remove(f):
        set_voice_folders(s, [x for x in voice_folders(s) if os.path.normcase(x) != os.path.normcase(f)])
        changed()
    m = QMenu(button)
    m.addAction("Ajouter un dossier de voix…", add)
    fs = voice_folders(s)
    if fs:
        m.addSeparator()
    for f in fs:
        sub = m.addMenu(f + ("" if os.path.isdir(f) else "   (introuvable)"))
        sub.addAction("Ouvrir dans l'explorateur", lambda f=f: os.path.isdir(f) and os.startfile(f))
        sub.addAction("Retirer de la liste (rien n'est supprimé)", lambda f=f: remove(f))
    m.exec(button.mapToGlobal(button.rect().bottomLeft()))


class _Bridge(QObject):
    loaded = Signal(bool, str)


class Activity(QWidget):
    """Les dernières secondes : niveau du micro, en couleur quand ta voix passe, en gris quand Smart bloque."""

    def __init__(self, n=60):
        super().__init__()
        self.hist = [(-90.0, False)] * n
        self.setFixedHeight(46)

    def push(self, db, open_):
        self.hist = self.hist[1:] + [(db, open_)]
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["bg"]))
        p.drawRoundedRect(QRectF(0, 0, w, h), 7, 7)
        n = len(self.hist)
        bw = (w - 12) / n
        for i, (db, on) in enumerate(self.hist):
            v = max(0.0, min(1.0, (db + 60) / 60))
            bh = max(2.0, v * (h - 12))
            p.setBrush(QColor(C["accent"] if on else C["line2"]))
            p.drawRoundedRect(QRectF(6 + i * bw + 0.6, h - 6 - bh, max(1.0, bw - 1.2), bh), 1.2, 1.2)


class VoicePage:
    def __init__(self, live):
        self.live = live
        self.pipe = live.pipeline
        self.voice = self.pipe.voice
        s = live.settings
        vp = self.voice.p
        vp.update({k: v for k, v in s.get("voix", {}).items() if k in VOICE_DEFAULTS})
        if s.get("voix_rev", 1) < 2:     # Smart réglé sur une voix douce : anciens réglages d'origine relevés
            old = s.get("voix", {})
            if old.get("sensitivity", 50) in (50, 40):
                vp["sensitivity"] = VOICE_DEFAULTS["sensitivity"]
            if old.get("release_ms", 220) == 220:
                vp["release_ms"] = VOICE_DEFAULTS["release_ms"]
        if s.get("voix_rev", 1) < 3:
            for k, old_v in (("lift_floor", 20), ("lift_amount", 70), ("lift_floor", 10), ("lift_amount", 80), ("lively", 20)):
                if vp[k] == old_v:
                    vp[k] = VOICE_DEFAULTS[k]
            s["voix_rev"] = 3
        s.setdefault("voice_library", "")
        default_voices(s, vp)
        if "voice_favorites" not in s:      # 1er lancement : la voix utilisée d'habitude devient favorite
            s["voice_favorites"] = [vp["pth"]] if vp["pth"] and os.path.isfile(vp["pth"]) else []
        self.favs = s["voice_favorites"]
        if s.get("voix_rev", 1) < 4:        # hauteur normale : réglée à la main (la mesure apprise était faussée)
            vp["lift_ref"] = VOICE_DEFAULTS["lift_ref"]
            vp["lift_auto"] = False
            s["voix_rev"] = 4
        if s.get("voix_rev", 1) < 5:        # nettoyage moins fort : les fins de mots (« mal ») ne sont plus rognées
            if vp.get("denoise_db") == 18:
                vp["denoise_db"] = VOICE_DEFAULTS["denoise_db"]
            s["voix_rev"] = 5
        self.bridge = _Bridge()
        self.bridge.loaded.connect(self._loaded)
        self.restart_timer = QTimer()
        self.restart_timer.setSingleShot(True)
        self.restart_timer.timeout.connect(self._restart_audio)

        page, f = live.form("voix", "Voix", "Ta voix transformée en temps réel (RVC), sans les bruits autour.")
        self.page = page
        self._tab_model(f)
        self._tab_voice(f)
        self._tab_smart(f)
        self._tab_perf(f)
        live.add("voice", "Voix", page, f)
        live.rail.changed.connect(lambda i: live.rail.pages[i] is page and self._fill_library())   # dossiers changés au Studio
        # Un modèle déjà choisi et la voix active : chargé tout de suite (en arrière-plan).
        if vp["enabled"] and vp["pth"] and os.path.isfile(vp["pth"]):
            self._load(vp["pth"], vp["index"] if os.path.isfile(vp["index"]) else find_index(vp["pth"]), quiet=True)
        self._show_model()
        self._show_fav()

    # ------------------------------------------------------------ Modèle
    def _tab_model(self, f):
        vp = self.voice.p
        f.tab("Modèle")
        card = f.section("Changeur de voix")
        self.on = f.check("Voix transformée", vp["enabled"], self._toggle,
                          tip="Le micro sort avec la voix du modèle. Jamais ta voix d'origine : en attendant le modèle, c'est le silence.")
        head = QHBoxLayout()
        self.name = QLabel()
        self.name.setObjectName("h1")
        self.name.setMinimumWidth(10)           # un long nom ne doit pas élargir le panneau
        self.fav_btn = QPushButton("☆")
        self.fav_btn.setObjectName("icon")
        self.fav_btn.setToolTip("Voix favorite (en tête de la bibliothèque)")
        self.fav_btn.clicked.connect(lambda: self._toggle_fav(self.voice.p["pth"]))
        head.addWidget(self.name, 1)
        head.addWidget(self.fav_btn)
        self.info = note("")
        card.addRow(head)
        card.addRow(self.info)
        self.activity = Activity()
        card.addRow(self.activity)
        self.meters = {"in": Meter("Micro"), "out": Meter("Voix")}
        for m in self.meters.values():
            card.addRow(m)
        self.status = f.label()
        row = QHBoxLayout()
        pick = QPushButton("Choisir un modèle…")
        pick.clicked.connect(self._choose)
        idx = QPushButton("Index…")
        idx.setToolTip("Le fichier .index du modèle (trouvé tout seul s'il est dans le même dossier).")
        idx.clicked.connect(self._choose_index)
        row.addWidget(pick, 1)
        row.addWidget(idx)
        card.addRow(row)
        save = QPushButton("Enregistrer mes réglages")
        save.setToolTip("Tout ce que tu as réglé (voix, Smart, effets…) est repris tel quel au prochain démarrage. "
                        "PhoneCam enregistre aussi tout seul toutes les 20 s.")
        save.clicked.connect(self.save_now)
        card.addRow(save)

        lib = f.section("Bibliothèque")
        self.lib_label = note("")
        lib.addRow(self.lib_label)
        row = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filtrer…")
        self.filter.textChanged.connect(self._fill_library)
        folder = QPushButton("Dossiers…")
        folder.setToolTip("Les dossiers où tu ranges tes modèles de voix (un sous-dossier par voix, .pth et .index "
                          "ensemble) : ajoute les tiens, ouvre-les ou retire-les de la liste. Partagés avec le Studio.")
        folder.clicked.connect(lambda: folders_menu(folder, self.live.settings, self._fill_library))
        row.addWidget(self.filter, 1)
        row.addWidget(folder)
        lib.addRow(row)
        self.list = QListWidget()
        self.list.setMinimumHeight(220)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.ElideMiddle)
        self.list.itemClicked.connect(lambda it: it.data(Qt.UserRole) and
                                      self._load(it.data(Qt.UserRole), find_index(it.data(Qt.UserRole))))
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._list_menu)
        lib.addRow(self.list)
        f.note("Un clic charge la voix (1 à 3 s). Clic droit : favori. On peut aussi glisser un .pth dans la fenêtre.")
        self._fill_library()
        f.track("pth", lambda: vp["pth"], lambda v: v and v != vp["pth"] and os.path.isfile(v) and self._load(v, find_index(v)))

    def _fill_library(self, *_a):
        self.list.clear()
        folders = voice_folders(self.live.settings)
        if not folders and not self.favs:
            self.lib_label.setText("Aucun dossier de voix : « Dossiers… » → « Ajouter un dossier de voix… ».")
            return
        want = self.filter.text().strip().lower()
        found = scan_voices(self.live.settings)
        favs = {os.path.normcase(p) for p in self.favs}
        # Favoris en tête (même hors du dossier), puis toutes les voix du dossier.
        extra = [(os.path.splitext(os.path.basename(p))[0], p) for p in self.favs
                 if os.path.isfile(p) and not any(os.path.normcase(q) == os.path.normcase(p) for _l, q in found)]
        found = sorted(found + extra, key=lambda t: (os.path.normcase(t[1]) not in favs, t[0].lower()))
        current = self.voice.p["pth"]
        shown_sep = False
        for label, path in found:
            if want and want not in label.lower():
                continue
            fav = os.path.normcase(path) in favs
            if not fav and favs and not shown_sep:
                sep = QListWidgetItem("Toutes les voix")
                sep.setFlags(Qt.NoItemFlags)
                sep.setForeground(QColor(C["faint"]))
                self.list.addItem(sep)
                shown_sep = True
            mark = "● " if os.path.normcase(path) == os.path.normcase(current) else "    "
            it = QListWidgetItem(mark + ("★ " if fav else "") + label + ("" if find_index(path) else "   (sans index)"))
            it.setData(Qt.UserRole, path)
            it.setToolTip(path)
            self.list.addItem(it)
        self.lib_label.setText((f"Dossier « {os.path.basename(folders[0]) or folders[0]} »" if len(folders) == 1
                                else f"{len(folders)} dossiers") + f"  ·  {len(found)} voix"
                               if folders else f"{len(found)} voix favorites (aucun dossier)")
        self.lib_label.setToolTip("\n".join(folders))

    def _list_menu(self, pos):
        it = self.list.itemAt(pos)
        path = it.data(Qt.UserRole) if it else None
        if not path:
            return
        from PySide6.QtWidgets import QMenu
        m = QMenu(self.list)
        fav = any(os.path.normcase(p) == os.path.normcase(path) for p in self.favs)
        m.addAction("Retirer des favoris" if fav else "★ Ajouter aux favoris", lambda: self._toggle_fav(path))
        m.addAction("Charger cette voix", lambda: self._load(path, find_index(path)))
        m.exec(self.list.mapToGlobal(pos))

    def _toggle_fav(self, path):
        if not path:
            self.live.toast("Charge d'abord une voix")
            return
        keep = [p for p in self.favs if os.path.normcase(p) != os.path.normcase(path)]
        if len(keep) == len(self.favs):
            keep.insert(0, path)
            self.live.toast(f"★ {os.path.splitext(os.path.basename(path))[0]} : voix favorite")
        self.favs[:] = keep
        self._fill_library()
        self._show_fav()
        self.live.persist_cb()

    def _show_fav(self):
        pth = self.voice.p["pth"]
        on = bool(pth) and any(os.path.normcase(p) == os.path.normcase(pth) for p in self.favs)
        self.fav_btn.setText("★" if on else "☆")
        self.fav_btn.setEnabled(bool(pth))

    def _choose(self):
        start = os.path.dirname(self.voice.p["pth"]) if self.voice.p["pth"] else \
            (voice_folders(self.live.settings) or [os.path.expanduser("~")])[0]
        path, _ = QFileDialog.getOpenFileName(self.live.window(), "Modèle de voix", start, MODELS)
        if path:
            self._load(os.path.normpath(path), find_index(path))

    def _choose_index(self):
        if not self.voice.p["pth"]:
            self.live.toast("Choisis d'abord un modèle")
            return
        path, _ = QFileDialog.getOpenFileName(self.live.window(), "Index du modèle", os.path.dirname(self.voice.p["pth"]),
                                              "Index RVC (*.index)")
        if path:
            self._load(self.voice.p["pth"], os.path.normpath(path))

    def drop(self, path):
        self._load(os.path.normpath(path), find_index(path))

    def _load(self, pth, index, quiet=False):
        vp = self.voice.p
        if self.voice.ready() and os.path.normcase(pth) == os.path.normcase(vp["pth"]) and (index or "") == (vp["index"] or ""):
            return False                # déjà là
        if self.voice.loading:
            self.live.toast("Un modèle est déjà en cours de chargement")
            return False
        self.pending = (pth, quiet)
        self.name.setText(os.path.splitext(os.path.basename(pth))[0])
        self.info.setText("Chargement…")
        if not voice_folders(self.live.settings):     # premier modèle : son dossier parent devient un dossier de voix
            parent = os.path.dirname(os.path.dirname(pth))
            if parent and any(os.path.isdir(os.path.join(parent, d)) for d in os.listdir(parent)):
                set_voice_folders(self.live.settings, [parent])
        self.voice.load(pth, index, lambda ok, msg: self.bridge.loaded.emit(ok, msg))
        return True

    def _loaded(self, ok, msg):
        pth, quiet = getattr(self, "pending", ("", True))
        self._show_model(msg if ok else "")
        self._show_fav()
        self._fill_library()
        if ok:
            if not quiet:
                self.live.toast(f"Voix : {os.path.splitext(os.path.basename(pth))[0]}")
            self.live.persist_cb()
        else:
            self.info.setText(msg)

    def _show_model(self, info=""):
        vp = self.voice.p
        if not vp["pth"]:
            self.name.setText("Aucune voix")
            self.info.setText("Choisis un modèle .pth dans ta bibliothèque ou avec « Choisir un modèle… ».")
            return
        self.name.setText(os.path.splitext(os.path.basename(vp["pth"]))[0])
        if info:
            self.info.setText(info + ("" if vp["index"] else "   ·   ajoute son .index pour un timbre plus fidèle"))
        elif not self.voice.ready():
            self.info.setText("Chargé quand la voix transformée est activée.")

    def _toggle(self, on):
        vp = self.voice.p
        self.voice.set("enabled", bool(on))
        if on and vp["pth"] and not self.voice.ready() and not self.voice.loading and os.path.isfile(vp["pth"]):
            self._load(vp["pth"], vp["index"] if os.path.isfile(vp["index"]) else find_index(vp["pth"]), quiet=True)
        live = self.live
        if self.pipe.audio.running:
            self._restart_audio()
        elif on and live.dev_in.count() and live.dev_out.count():
            live.run_btn.setChecked(True)       # la voix passe par le micro : on le démarre
        if on and not vp["pth"]:
            live.toast("Choisis un modèle de voix")

    def _measure(self):
        eng = self.voice.engine
        a = self.pipe.audio
        if eng is None or not (a.running and a.vox and self.voice.ready()):
            self.live.toast("Démarre le micro avec la voix transformée, puis parle normalement")
            return
        eng.lively.measure = []
        self.measuring = True
        self.live.toast("Parle normalement, avec ta voix joyeuse habituelle, pendant ~5 s")

    def save_now(self):
        self.live.persist_cb()
        self.live.toast("Réglages enregistrés : ils seront là au prochain démarrage")

    def _restart_audio(self):
        if self.pipe.audio.running:
            self.live._mic_run(True)

    # ------------------------------------------------------------ Voix
    def _sl(self, f, key, text, lo, hi, fmt, tip=None, restart=False):
        vp = self.voice.p

        def changed(x):
            self.voice.set(key, x)
            if restart and self.pipe.audio.running and self.pipe.audio.vox:
                self.restart_timer.start(700)       # tampons de sortie à refaire (un court silence)
        return f.slider(text, lo, hi, int(vp[key]), changed, fmt, int(VOICE_DEFAULTS[key]), key, tip)

    def _tab_voice(self, f):
        vp = self.voice.p
        f.tab("Voix")
        f.section("Hauteur et timbre")
        self._sl(f, "pitch", "Hauteur", -24, 24, lambda x: signed(x, " demi-tons"),
                 "+12 : une octave au-dessus (homme → femme), −12 : une octave en dessous.")
        self._sl(f, "formant", "Timbre (formants)", -40, 40, lambda x: f"{x / 10:+.1f}" if x else "0",
                 "Plus haut : voix plus fine, plus « petite » ; plus bas : plus grave et plus large. Sans changer la hauteur.")
        f.section("Voix vivante (ton bas)")
        f.check("Relever mon ton quand je parle bas", vp["lift_on"], lambda on: self.voice.set("lift_on", bool(on)),
                key="lift_on", default=True,
                tip="Ta hauteur habituelle est apprise toute seule. Quand tu passes nettement en dessous, la voix est remontée, "
                    "plus mélodieuse et plus claire : enjouée plutôt qu'éteinte.")
        self.ref_bar = self._sl(f, "lift_ref", "Ta hauteur normale", 70, 300, lambda x: f"{x} Hz",
                                "Ta voix habituelle, quand tu parles normalement et joyeusement. En dessous, la voix est relevée. "
                                "Règle à l'oreille, ou « Mesurer ma voix normale ».")
        self._sl(f, "lift_floor", "Seuil (sous ta hauteur normale)", 0, 60, lambda x: f"{x / 10:.1f} demi-ton",
                 "À partir de combien sous ta hauteur normale on remonte. Plus bas : agit plus souvent.")
        self._sl(f, "lift_amount", "Remonter de", 0, 100, lambda x: f"{x} % de l'écart",
                 "100 % : ramené à ton ton habituel ; 50 % : à mi-chemin.")
        self._sl(f, "lively", "Expressivité", 0, 80, lambda x: f"+{x} %",
                 "La mélodie de la voix (montées, descentes) est élargie : moins monotone.")
        f.check("Expressivité tout le temps", vp["lively_always"], lambda on: self.voice.set("lively_always", bool(on)),
                key="lively_always", default=False, tip="Sinon, pleine seulement quand le ton est relevé (un peu le reste du temps).")
        self._sl(f, "lift_bright", "Clarté ajoutée", 0, 100, lambda x: f"{x} %",
                 "Présence et air ajoutés seulement pendant que la voix est relevée.")
        self.lift_state = f.label()
        measure = f.button("Mesurer ma voix normale (5 s)", self._measure)
        measure.setToolTip("Parle normalement pendant ~5 s : « Ta hauteur normale » est réglée sur ta voix.")
        f.check("L'apprendre toute seule en continu", vp["lift_auto"], lambda on: self.voice.set("lift_auto", bool(on)),
                key="lift_auto", default=False,
                tip="Le curseur suit lentement ta voix. Coupé (conseillé) : il ne bouge que si tu le règles.")
        f.section("Fidélité au modèle", key="voix-fidelite")
        self._sl(f, "index_rate", "Index (accent, timbre du modèle)", 0, 100, lambda x: f"{x} %",
                 "Plus haut : plus proche de la voix d'origine du modèle, mais articulation un peu moins nette.")
        self._sl(f, "protect", "Protection des consonnes", 0, 50, lambda x: "coupée" if x >= 50 else f"{x} %",
                 "Garde les « s », « ch », souffles et respirations nets face à l'index. Plus bas = plus protégé.")
        f.section("Volume")
        self._sl(f, "follow", "Suivre le volume de ta voix", 0, 100, lambda x: f"{x} %",
                 "100 % : la voix transformée monte et descend exactement comme la tienne.")
        self._sl(f, "gain", "Gain de la voix", -18, 12, lambda x: f"{x:+d} dB")

    # ------------------------------------------------------------ Smart
    def _tab_smart(self, f):
        vp = self.voice.p
        f.tab("Smart")
        f.section("Smart")
        f.check("Smart", vp["smart"], lambda on: self.voice.set("smart", bool(on)), key="smart", default=True,
                tip="Seule ta voix entre dans le modèle : clavier, souris, souffle, respirations, ventilateur… sortent en silence.")
        f.note("Rien qui ne soit pas de la voix n'entre dans le changeur : il ne peut donc pas en faire des sons bizarres. "
               "La hauteur n'existe que dans la parole, et le générateur de sinus ne transforme plus le bruit en voix. À laisser activé.")
        self.smart_state = f.label()
        f.section("Nettoyage du micro")
        f.check("Retirer les bruits avant la conversion", vp["denoise"], lambda on: self.voice.set("denoise", bool(on)),
                key="denoise", default=True, tip="Réseau de débruitage (GTCRN) : clavier, ventilateur, souffle. +20 ms de latence.")
        self._sl(f, "denoise_db", "Atténuation maximale", 6, 40, lambda x: f"{x} dB",
                 "Jusqu'où le nettoyage peut baisser un son. Plus haut : bruits plus effacés pendant que tu parles, "
                 "mais chuchotements et « p », « f », « s » risquent d'être abîmés.")
        f.section("Détection de la voix", open_=False)
        self._sl(f, "sensitivity", "Seuil de détection", 10, 95, lambda x: f"{x} %",
                 "Plus haut : il faut une voix plus nette pour ouvrir (moins de bruits, risque de couper un chuchotement).")
        self._sl(f, "release_ms", "Maintien après la voix", 60, 600, lambda x: f"{x} ms",
                 "Temps pendant lequel on reste ouvert après la dernière syllabe (fins de mots, respirations entre deux mots).")
        self._sl(f, "preroll_ms", "Ouverture anticipée", 0, 50, lambda x: f"{x} ms",
                 "Rouvre un peu avant le début du mot : la première consonne n'est pas mangée.")
        self._sl(f, "floor_db", "Silence en dessous de", -90, -30, lambda x: f"{x} dB",
                 "Sous ce niveau (après nettoyage), rien ne passe, même si cela ressemble à une voix.")
        f.check("Garder les voyelles tenues", vp["hold_voiced"], lambda on: self.voice.set("hold_voiced", bool(on)),
                key="hold_voiced", default=True, tip="« Aaaaah », « mmmm », chant : la porte reste ouverte tant que la voix tient.")
        f.check("Garder les respirations dans les phrases", vp["breath"], lambda on: self.voice.set("breath", bool(on)),
                key="breath", default=True,
                tip="Souffles, respirations et chuchotements entre tes mots (moins de 1,5 s après ta voix). "
                    "Hors des phrases, rien sans voix ne passe. Coupé : seule la voix avec une hauteur passe.")
        f.check("Chuchotements seuls (sans voix autour)", vp["whisper"], lambda on: self.voice.set("whisper", bool(on)),
                key="whisper", default=False,
                tip="Une phrase entièrement chuchotée passe aussi. Moins strict : quelques souffles et petits bruits peuvent passer.")
        self._sl(f, "voicing", "Confiance de la hauteur", 1, 60, lambda x: f"{x} %",
                 "Plus haut : seules les voyelles bien nettes reçoivent une hauteur (moins de bruits « chantés »).")

    # ------------------------------------------------------------ Réglage
    def _tab_perf(self, f):
        f.tab("Réglage")
        f.section("Latence et qualité")
        self._sl(f, "block", "Taille de bloc", 10, 50, lambda x: f"{x / 100:.2f} s",
                 "Le son est transformé par morceaux de cette durée. Plus petit = moins de retard, plus de calcul.", True)
        self._sl(f, "extra", "Contexte analysé", 10, 40, lambda x: f"{x / 10:.1f} s",
                 "Ce que le modèle « entend » avant chaque bloc. Plus long = voix plus stable et naturelle (sans retard en plus).")
        self._sl(f, "crossfade", "Fondu entre les blocs", 1, 4, lambda x: f"{x * 10} ms",
                 "Raccord entre deux blocs. Plus long = raccords plus doux, un peu plus de retard.")
        self._sl(f, "margin_ms", "Réserve anti-coupures", 10, 150, lambda x: f"{x} ms",
                 "Marge contre les à-coups de la carte graphique (vidéo en même temps). Monte si tu entends des coupures.", True)
        self.perf = f.label()
        f.note("La voix est calculée sur la carte graphique dans un flux prioritaire (elle passe devant la vidéo), avec CUDA Graphs. "
               "Hors de ta voix, la sortie est un silence exact.")
        card = f.section("Dépannage", open_=False, key="voix-depannage")
        keep = QPushButton("Ma voix a coupé : garder les 30 dernières secondes")
        keep.setToolTip("Juste après une coupure : ton micro brut, la voix sortie et ce que Smart a décidé "
                        "sont gardés dans data/coupures-voix, pour trouver la cause. Rien d'autre n'est envoyé.")
        keep.clicked.connect(lambda: self.pipe.audio.keep_last())
        card.addRow(keep)
        self.bb_label = note("")
        card.addRow(self.bb_label)
        f.note("Pour transformer un fichier ou un enregistrement de ta voix : Studio, page Voix.")

    # ------------------------------------------------------------ chaque tic
    def tick(self):
        v, a = self.voice, self.pipe.audio
        on = a.running and a.vox and v.ready()
        speaking = on and bool(v.stat("speaking", False))
        self.activity.push(v.stat("in_db", -90.0) if on else -90.0, speaking)
        self.meters["in"].setValue(v.stat("in_db", -60.0) if on else -60)
        self.meters["out"].setValue(v.stat("out_db", -60.0) if on else -60)
        if not v.p["enabled"]:
            st = "Voix transformée coupée : le micro sort tel quel."
        elif not v.p["pth"]:
            st = "Choisis un modèle de voix."
        elif not a.running:
            st = "Le micro est arrêté : démarre-le (page Micro ou bouton « Micro » en bas)."
        elif not a.vox:
            st = "Redémarre le micro pour passer par la voix transformée."
        else:
            st = v.status
        self.status.setText(st)
        if on and v.p["lift_on"]:
            sh = v.stat("lift_shift", 0.0)
            cur = v.stat("cur_hz", 0.0)
            eng = v.engine
            if getattr(self, "measuring", False) and eng is not None and eng.lively.measure is None:
                self.measuring = False
                self.ref_bar.setValue(int(v.p["lift_ref"]))
                self.live.toast(f"Ta hauteur normale : {v.p['lift_ref']} Hz")
            if v.p["lift_auto"] and self.ref_bar.value() != int(v.p["lift_ref"]):
                self.ref_bar.blockSignals(True)
                self.ref_bar.setValue(int(v.p["lift_ref"]))
                self.ref_bar.blockSignals(False)
            now = f"en ce moment : {cur:.0f} Hz" if cur else "parle pour voir ta hauteur"
            if getattr(self, "measuring", False):
                n = len(eng.lively.measure or []) if eng is not None else 0
                now = f"mesure… {min(100, n // 3)} %"
            self.lift_state.setText((f"● Ton bas : relevé de +{sh:.1f} demi-ton" if sh > 0.3 else "○ Ton normal : rien à relever")
                                    + f"   ·   {now}")
        else:
            self.lift_state.setText("")
        if self.bb_label.text() != self.pipe.audio.bb_status:
            self.bb_label.setText(self.pipe.audio.bb_status)
        if on:
            self.smart_state.setText(("● Ta voix passe" if speaking else "○ Silence : rien d'autre que ta voix n'entre")
                                     if v.p["smart"] else "Smart coupé : tout ce que capte le micro est transformé.")
            self.perf.setText(f"Calcul : {v.stat('ms'):.0f} ms par bloc (pointe {v.stat('ms_max'):.0f} ms)   ·   "
                              f"retard de la voix ≈ {v.latency_ms() + 60:.0f} ms (cartes son comprises)   ·   coupures {a.underruns()}")
        else:
            self.smart_state.setText("")
            self.perf.setText("")
