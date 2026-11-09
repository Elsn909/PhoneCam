"""Studio, page Voix : transformer la voix d'un fichier audio (ou d'un enregistrement), l'écouter, l'exporter.

Post-production : rien ne passe en direct. On choisit un son (fichier, enregistrement au micro ou son
de la vidéo choisie dans Médias), une voix, on transforme, on compare avec l'original au même endroit
(A/B), puis on exporte un fichier audio, ou la vidéo avec la nouvelle voix. Le moteur est à part de
celui du Direct : la voix en direct n'est jamais coupée ni ralentie par une conversion ici.
"""
import os
import queue
import subprocess
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

from entree_micro import DEFAULTS as IN_DEFAULTS
from phonemic import DEFAULTS as MIC_DEFAULTS
from server import DATA_DIR
from ui_kit import C, Meter, Segmented, icon
from voix.fichier import KEYS, find_index
from voix.reglages import DEFAULTS as VOICE_DEFAULTS

SR = 48000
RECORDS = os.path.join(DATA_DIR, "studio-voix")
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aif", ".aiff")
OPEN_FILTER = ("Son ou vidéo (*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.aac *.wma *.aif *.aiff *.mp4 *.mov *.mkv *.webm *.avi);;"
               "Tous les fichiers (*)")
FORMATS = {      # nom → (extension, arguments ffmpeg)
    "WAV (sans perte, pour le montage)": (".wav", ["-c:a", "pcm_s24le"]),
    "FLAC (sans perte, plus léger)": (".flac", ["-c:a", "flac"]),
    "MP3 (320 kb/s)": (".mp3", ["-c:a", "libmp3lame", "-b:a", "320k"]),
    "M4A / AAC (256 kb/s)": (".m4a", ["-c:a", "aac", "-b:a", "256k"]),
}
LIMIT_S = 1800                    # 30 min par fichier
AUTO_S = 180                      # en dessous, un réglage changé est réentendu tout seul
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def apply_input(x, mp):
    """Entrée du micro (sensibilité, niveau automatique, moins d'écho, graves de loin) sur un fichier entier,
    comme dans le Direct avant le changeur de voix. « Moins d'écho » retarde de 5 ms : compensé."""
    from entree_micro import BLOCK, H, MicInput
    inp = MicInput(dict(mp))
    lat = H if mp["dereverb"] > 0 else 0
    src = np.concatenate([np.asarray(x, np.float32), np.zeros(lat, np.float32)])
    src = np.concatenate([src, np.zeros(-len(src) % BLOCK, np.float32)])
    out = np.empty(len(src), np.float32)
    for i in range(0, len(src), BLOCK):
        out[i:i + BLOCK] = inp.process(src[i:i + BLOCK])
    return out[lat:lat + len(x)]


def apply_mic(y, mp, progress=None, cancel=None):
    """Effets de la page Micro (égaliseur, de-esser, compresseur, chaleur) puis réalisme d'un smartphone
    (pièce, bruit de fond, couleur du micro), comme dans le Direct, sans retard ajouté. None si annulé."""
    import phonemic
    mic = phonemic.PhoneMic()
    mic.p.update({k: mp[k] for k in MIC_DEFAULTS if k in mp})
    mic.p.update(delay=0, sync_video=False)
    key = (mic.p["room_size"], mic.p["damping"], mic.p["distance"])
    mic.conv.set_ir(phonemic.room_ir(key[0], key[1], key[2] / 100, np.random.default_rng(7)))   # tout de suite
    mic._room_key = key
    B = phonemic.BLOCK
    src = np.concatenate([np.asarray(y, np.float32), np.zeros(-len(y) % B, np.float32)])
    out = np.empty(len(src), np.float32)
    n = len(src) // B
    for i in range(n):
        if cancel is not None and i % 200 == 0 and cancel():
            return None
        out[i * B:(i + 1) * B] = mic.process(src[i * B:(i + 1) * B])
        if progress is not None and i % 500 == 0:
            progress(i / n)
    return out[:len(y)]


class _LiveShim:
    """Ce dont l'onglet Effets de la page Micro (ui_micfx.FxTab) a besoin, côté Studio."""

    def __init__(self, st, panel):
        self.st, self.settings, self.panels = st, st.settings, {"micro": panel}

    def window(self):
        return self.st.window()

    def persist_cb(self):
        pass


def clock(t):
    return f"{int(t) // 60:02d}:{t % 60:04.1f}"


def ffmpeg_write(path, y, args, video=None):
    """Son 48 kHz mono → fichier (ou vidéo d'origine + ce son, image copiée telle quelle)."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
    if video:
        cmd += ["-i", video]
    cmd += ["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-"]
    if video:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest"]
    else:
        cmd += args
    cmd.append(path)
    p = subprocess.run(cmd, input=np.ascontiguousarray(y, np.float32).tobytes(), capture_output=True,
                       creationflags=NO_WINDOW)
    if p.returncode != 0:
        lines = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(lines[-1] if lines else "ffmpeg a échoué")


class Bridge(QObject):
    loaded = Signal(object, str)          # son (ou None), message
    converted = Signal(object, int, object)  # son transformé (ou None), numéro de demande, (clé, message)
    progress = Signal(float, str)
    recorded = Signal(object)
    done = Signal(str)


class Wave(QWidget):
    """Forme d'onde : l'original en gris, la voix transformée en couleur ; un clic déplace la lecture."""
    seek = Signal(float)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(84)
        self.setCursor(Qt.PointingHandCursor)
        self.a = self.b = None
        self.pos = 0.0                     # 0…1
        self.hint = "Aucun son : ouvre un fichier ou enregistre ta voix."

    @staticmethod
    def peaks(x, n=600):
        if x is None or not len(x):
            return None
        k = max(1, len(x) // n)
        m = len(x) // k * k
        return np.abs(x[:m]).reshape(-1, k).max(1) if m else np.abs(x)

    def set_audio(self, a, b=None):
        self.a, self.b = self.peaks(a), self.peaks(b)
        self.update()

    def mousePressEvent(self, ev):
        if self.a is not None:
            self.seek.emit(max(0.0, min(1.0, ev.position().x() / max(1, self.width()))))

    mouseMoveEvent = mousePressEvent

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["bg"]))
        p.drawRoundedRect(QRectF(0, 0, w, h), 7, 7)
        if self.a is None:
            p.setPen(QColor(C["faint"]))
            p.drawText(QRectF(8, 0, w - 16, h), Qt.AlignCenter | Qt.TextWordWrap, self.hint)
            return
        mid = h / 2
        for data, color, scale in ((self.a, C["line2"], 1.0), (self.b, C["accent"], 0.8)):
            if data is None:
                continue
            n = len(data)
            bw = (w - 8) / n
            p.setBrush(QColor(color))
            top = max(1e-3, float(data.max()))
            for i in range(0, n, max(1, int(n / (w - 8)))):
                v = min(1.0, float(data[i]) / top) * (mid - 6) * scale
                p.drawRect(QRectF(4 + i * bw, mid - v - 0.5, max(1.0, bw), 2 * v + 1))
        x = 4 + (w - 8) * self.pos
        p.setBrush(QColor(C["text"]))
        p.drawRect(QRectF(x - 1, 3, 2, h - 6))


class Player:
    """Lecture sur la sortie son de Windows. Original et transformé gardent la même position : on passe
    de l'un à l'autre sans revenir au début."""

    def __init__(self):
        self.tracks = [None, None]          # original, transformé
        self.which = 1
        self.pos = 0
        self.playing = False
        self.stream = None

    def length(self):
        t = self.tracks[0]
        return len(t) if t is not None else 0

    def _cb(self, out, frames, _t, _s):
        t = self.tracks[self.which] if self.tracks[self.which] is not None else None
        if not self.playing or t is None:
            out[:] = 0
            return
        chunk = t[self.pos:self.pos + frames]
        out[:len(chunk), 0] = chunk
        out[len(chunk):, 0] = 0
        self.pos += frames
        if self.pos >= len(t):
            self.pos = 0
            self.playing = False

    def play(self):
        import sounddevice as sd
        if self.stream is None:
            self.stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=1024, callback=self._cb)
            self.stream.start()
        self.playing = True

    def stop(self):
        self.playing = False

    def close(self):
        self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
            self.stream = None


class StudioVoice:
    """La page Voix du Studio (construite dans le rail du Studio)."""

    def __init__(self, studio):
        self.st = studio
        cfg = studio.cfg.setdefault("voix", {})
        live = studio.settings.get("voix", {})
        for k in KEYS:                      # au premier usage : les réglages de la voix du Direct
            cfg.setdefault(k, live.get(k, VOICE_DEFAULTS[k]))
        cfg.setdefault("pth", live.get("pth", ""))
        cfg.setdefault("format", 0)
        cfg.setdefault("rec_device", "")
        cfg.setdefault("convert", True)     # False : seulement les effets du micro, sans modèle de voix
        mic = cfg.setdefault("mic", {})
        live_mic = studio.settings.get("mic", {})
        for k in MIC_DEFAULTS:              # au premier usage : les réglages de la page Micro du Direct
            mic.setdefault(k, live_mic.get(k, MIC_DEFAULTS[k]))
        self.mic = mic
        self.mic_panel = {}
        self._mic_seen = None
        self._vcache = None                 # (clé, voix avant les effets du micro) : un effet changé ne refait pas la voix
        self.cfg = cfg
        self.engine = None                  # voix.fichier.FileVoice, créé à la première conversion
        self.src = None                     # son d'origine (48 kHz mono)
        self.src_name, self.src_video = "", ""
        self.out = None                     # voix transformée
        self.out_key = None
        self.want = 0                       # numéro de la dernière demande de conversion
        self.busy = False
        self.cancel = False
        self.rec = None                     # enregistrement en cours : [morceaux, flux, début]
        self.level = -90.0
        self.player = Player()
        self.tasks = queue.Queue()
        self.b = Bridge()
        self.b.loaded.connect(self._loaded)
        self.b.converted.connect(self._converted)
        self.b.progress.connect(self._progress)
        self.b.recorded.connect(self._recorded)
        self.b.done.connect(self._done)
        self.auto = QTimer()
        self.auto.setSingleShot(True)
        self.auto.setInterval(600)
        self.auto.timeout.connect(lambda: self.convert(auto=True))
        self.idle = QTimer()                # mémoire de la carte graphique rendue après 10 min sans servir
        self.idle.setSingleShot(True)
        self.idle.setInterval(10 * 60 * 1000)
        self.idle.timeout.connect(lambda: self.tasks.put(("release",)))
        self.ticker = QTimer()
        self.ticker.timeout.connect(self.tick)
        self.ticker.start(50)
        self._build()
        threading.Thread(target=self._work, daemon=True, name="studio-voix").start()

    # ------------------------------------------------------------ page
    def _build(self):
        st, cfg = self.st, self.cfg
        page, f = st._form("Voix", subtitle="Post-production : un son (fichier ou ta voix enregistrée) transformé "
                                            "avec un modèle de voix, à écouter puis exporter. Rien ne sort en direct.")
        self.page = page
        f.tab("Transformer")
        card = f.section("1 · Le son à transformer", key="sv-son")
        row = QHBoxLayout()
        b_open = QPushButton("Fichier…")
        b_open.setIcon(icon("folder", 15))
        b_open.setToolTip("Un fichier audio (WAV, MP3, M4A, FLAC…) ou le son d'une vidéo. Tu peux aussi le glisser dans le Studio.")
        b_open.clicked.connect(self.open_dialog)
        self.rec_btn = QPushButton("Enregistrer")
        self.rec_btn.setIcon(icon("rec", 13, C["bad"]))
        self.rec_btn.setToolTip("Enregistre ton micro jusqu'à ce que tu cliques de nouveau (30 min au plus).")
        self.rec_btn.clicked.connect(self.toggle_record)
        self.media_btn = QPushButton("Son du média")
        self.media_btn.setToolTip("Le son de la vidéo choisie dans Médias : à l'export, la vidéo peut ressortir avec la nouvelle voix.")
        self.media_btn.clicked.connect(self.from_media)
        for w in (b_open, self.rec_btn, self.media_btn):
            row.addWidget(w)
        card.addRow(row)
        self.src_label = QLabel("Aucun son choisi")
        self.src_label.setObjectName("value")
        self.src_label.setMinimumWidth(10)
        card.addRow(self.src_label)
        self.rec_meter = Meter("Micro")
        self.rec_meter.hide()
        card.addRow(self.rec_meter)
        self.dev = QComboBox()
        self.dev.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.dev.setMinimumContentsLength(12)
        self.dev.setToolTip("Le micro utilisé par « Enregistrer ».")
        card.addRow("Micro", self.dev)
        self._fill_devices()
        self.dev.currentIndexChanged.connect(lambda _i: cfg.__setitem__("rec_device", self.dev.currentData() or ""))

        card = f.section("2 · Écouter", key="sv-ecoute")
        self.wave = Wave()
        self.wave.seek.connect(self.seek)
        card.addRow(self.wave)
        row = QHBoxLayout()
        self.play_btn = QPushButton("Écouter")
        self.play_btn.setIcon(icon("play", 14))
        self.play_btn.setObjectName("primary")
        self.play_btn.setToolTip("Lecture / pause (Espace quand cette page est ouverte)")
        self.play_btn.clicked.connect(self.toggle_play)
        self.ab = Segmented(["Original", "Transformée"])
        self.ab.setCurrentIndex(1)
        self.ab.setToolTip("Passe de l'un à l'autre pendant la lecture, au même endroit (touche A).")
        self.ab.currentIndexChanged.connect(self._set_ab)
        self.time = QLabel("00:00.0")
        self.time.setObjectName("mono")
        row.addWidget(self.play_btn)
        row.addWidget(self.ab, 1)
        card.addRow(row)
        card.addRow(self.time)
        self.state = QLabel("")
        self.state.setObjectName("status")
        self.state.setWordWrap(True)
        self.state.setMinimumWidth(10)
        card.addRow(self.state)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self.bar.hide()
        card.addRow(self.bar)

        f.section("3 · La voix", key="sv-voix")
        self.convert_on = f.check("Changer la voix avec un modèle", cfg["convert"], self._set_convert, "sv_convert",
                                  "Coupé : la voix d'origine est gardée, seulement les effets du micro (égaliseur, "
                                  "compresseur, pièce…) sont appliqués.", True)
        self.model = QComboBox()
        self.model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model.setMinimumContentsLength(12)
        self.model.activated.connect(self._model_picked)
        f.form.addRow(self.model)
        row = QHBoxLayout()
        other = QPushButton("Autre fichier…")
        other.setToolTip("Choisir un modèle .pth ailleurs sur le PC (son .index est trouvé tout seul).")
        other.clicked.connect(self._choose_model)
        row.addWidget(other, 1)
        folders = QPushButton("Dossiers…")
        folders.setToolTip("Les dossiers où tu ranges tes voix : ajoute les tiens. Partagés avec la page Voix du Direct.")
        folders.clicked.connect(lambda: __import__("ui_voix").folders_menu(folders, self.st.settings, self._fill_models))
        row.addWidget(folders, 1)
        f.form.addRow(row)
        self._fill_models()
        self.voice_widgets = [self.model, other, folders]
        self.sliders = {}
        self._sl(f, "pitch", "Hauteur", -24, 24, lambda x: f"{x:+d} demi-tons" if x else "0",
                 "+12 : une octave au-dessus (homme → femme), −12 : une octave en dessous.")
        self._sl(f, "formant", "Timbre (formants)", -40, 40, lambda x: f"{x / 10:+.1f}" if x else "0",
                 "Plus haut : voix plus fine ; plus bas : plus grave et plus large. Sans changer la hauteur.")
        self._sl(f, "index_rate", "Ressemblance au modèle (index)", 0, 100, lambda x: f"{x} %",
                 "Plus haut : plus proche de la voix d'origine du modèle, articulation un peu moins nette.")
        self._sl(f, "protect", "Protection des consonnes", 0, 50, lambda x: "coupée" if x >= 50 else f"{x} %",
                 "Garde les « s », « ch », souffles nets. Plus bas = plus protégé.")
        same = f.button("Comme la voix du Direct", self.same_as_live)
        same.setToolTip("Modèle, hauteur, timbre, nettoyage… comme dans la page Voix du Direct.")
        self.voice_widgets += [self.sliders[k] for k in ("pitch", "formant", "index_rate", "protect")] + [same]

        f.section("4 · Nettoyage et volume", key="sv-nettoyage")
        self._ck(f, "smart", "Garder seulement la voix (Smart)",
                 "Bruits, clavier, musique, souffles hors des phrases : rien d'autre que la voix n'est transformé.")
        self._ck(f, "denoise", "Retirer les bruits avant la conversion", "Débruitage (GTCRN) du son d'origine.")
        self._sl(f, "denoise_db", "Force du nettoyage", 6, 40, lambda x: f"{x} dB",
                 "Plus haut : bruits plus effacés, mais chuchotements et « s » risquent d'être abîmés.")
        self._sl(f, "follow", "Suivre le volume de l'original", 0, 100, lambda x: f"{x} %",
                 "100 % : la voix transformée monte et descend exactement comme l'original.")
        self._sl(f, "gain", "Volume", -18, 12, lambda x: f"{x:+d} dB" if x else "0 dB")

        f.section("5 · Effets du micro", key="sv-micro")
        f.note("Les mêmes que la page Micro du Direct, appliqués au son exporté : égaliseur, sifflantes, "
               "compresseur, chaleur (onglet Effets) ; son de smartphone, pièce, bruit de fond (onglet Ambiance) ; "
               "niveau d'entrée et gain de sortie (onglet Réglages fins).")
        b = f.button("Comme le micro du Direct", self.same_as_live_mic)
        b.setToolTip("Reprend tous les réglages de la page Micro du Direct (effets, ambiance, niveaux).")

        card = f.section("6 · Exporter", key="sv-export")
        f.combo("Format", list(FORMATS), int(cfg["format"]), lambda i: cfg.__setitem__("format", i), default=0)
        self.exp_btn = f.button("Exporter le son…", self.export_audio, primary=True)
        self.vid_btn = f.button("Exporter la vidéo avec cette voix…", self.export_video)
        self.vid_btn.setToolTip("L'image d'origine copiée telle quelle (aucune perte), le son remplacé par la voix transformée.")
        self.exp_note = f.note("Le fichier d'origine n'est jamais modifié.")

        # --- onglets de la page Micro du Direct, sur les réglages du Studio
        import types

        from effets_micro import VoiceFx
        from ui_micfx import FxTab
        f.panel = self.mic_panel                    # « Styles » de l'égaliseur : réglages de cette page
        self.fx_view = types.SimpleNamespace(p=self.mic, fx=VoiceFx(self.mic))
        self.fx_view.fx.spec[:] = -200              # spectre : celui du son en lecture (voir tick)
        self.fx_tab = FxTab(_LiveShim(st, self.mic_panel), f, self.fx_view)

        f.tab("Ambiance")
        f.section("Réalisme d'un smartphone", key="sv-smartphone")
        self._mck(f, "enabled", "Son de smartphone", "Coupé : seulement la voix et ses effets, sans pièce ni bruit de fond.")
        self._msl(f, "color", "Couleur du micro de téléphone", 0, 100, lambda x: f"{x} %")
        self._msl(f, "tone", "Timbre (chaud ↔ brillant)", -100, 100, lambda x: "neutre" if x == 0 else f"{x:+d}")
        self._msl(f, "distance", "Distance bouche → téléphone", 5, 120, lambda x: f"{x} cm")
        f.section("Pièce", key="sv-piece")
        self._msl(f, "room", "Réverbération", 0, 120, lambda x: f"{x} %")
        self._msl(f, "room_size", "Taille de la pièce", 6, 30, lambda x: f"{x} m²")
        self._msl(f, "damping", "Amortissement (meubles, rideaux)", 0, 100, lambda x: f"{x} %")
        f.section("Bruit de fond (très léger)", key="sv-bruit")
        pct = lambda x: "coupé" if x == 0 else f"{x} %"   # noqa: E731
        self._msl(f, "room_tone", "Ambiance de la pièce", 0, 100, pct)
        self._msl(f, "mic_noise", "Souffle du micro", 0, 100, pct)
        self._msl(f, "fan", "Ventilateur / PC", 0, 100, pct)

        f.tab("Réglages fins")
        f.section("Niveau d'entrée (avant la voix)", key="sv-entree")
        self._msl(f, "input_gain", "Sensibilité", -24, 30, lambda x: f"{x:+d} dB")
        self._mck(f, "auto_level", "Niveau automatique", "La voix est ramenée à son niveau « près du micro » (prises de loin).")
        self._msl(f, "auto_max", "Remontée maximale", 0, 36, lambda x: f"+{x} dB")
        self._msl(f, "auto_target", "Niveau de voix visé", -36, -12, lambda x: f"{x} dB")
        self._msl(f, "dereverb", "Moins d'écho de la pièce", 0, 100, pct)
        self._msl(f, "proximity", "Graves rendus de loin", 0, 100, pct)
        f.section("Sortie", key="sv-sortie")
        self._msl(f, "output_gain", "Gain de sortie", -24, 12, lambda x: f"{x:+d} dB")
        f.section("Détection de la voix (Smart)", key="sv-detection")
        self._sl(f, "sensitivity", "Seuil de détection", 10, 95, lambda x: f"{x} %",
                 "Plus haut : il faut une voix plus nette (moins de bruits, risque de couper un chuchotement).")
        self._sl(f, "release_ms", "Maintien après la voix", 60, 600, lambda x: f"{x} ms",
                 "Temps pendant lequel la voix reste ouverte après la dernière syllabe.")
        self._sl(f, "preroll_ms", "Ouverture anticipée", 0, 50, lambda x: f"{x} ms",
                 "Rouvre un peu avant le début du mot : la première consonne n'est pas mangée.")
        self._sl(f, "floor_db", "Silence en dessous de", -90, -30, lambda x: f"{x} dB")
        self._ck(f, "hold_voiced", "Garder les voyelles tenues", "« Aaaaah », « mmmm », chant.")
        self._ck(f, "breath", "Garder les respirations dans les phrases", "Souffles et chuchotements entre les mots.")
        self._ck(f, "whisper", "Chuchotements seuls (sans voix autour)", "Moins strict : quelques souffles peuvent passer.")
        self._sl(f, "voicing", "Confiance de la hauteur", 1, 60, lambda x: f"{x} %",
                 "Plus haut : seules les voyelles bien nettes reçoivent une hauteur.")
        f.section("Voix vivante", key="sv-vivante", open_=False)
        self._ck(f, "lift_on", "Relever le ton quand la voix est basse",
                 "Sous « Hauteur normale », la voix est remontée : enjouée plutôt qu'éteinte.")
        self._sl(f, "lift_ref", "Hauteur normale", 70, 300, lambda x: f"{x} Hz")
        self._sl(f, "lift_floor", "Seuil (sous la hauteur normale)", 0, 60, lambda x: f"{x / 10:.1f} demi-ton")
        self._sl(f, "lift_amount", "Remonter de", 0, 100, lambda x: f"{x} % de l'écart")
        self._sl(f, "lively", "Expressivité", 0, 80, lambda x: f"+{x} %")
        self._ck(f, "lively_always", "Expressivité tout le temps", "Sinon, pleine seulement quand le ton est relevé.")
        self._sl(f, "lift_bright", "Clarté ajoutée", 0, 100, lambda x: f"{x} %")
        f.note("Le son est transformé par blocs de 0,5 s avec 4 s de contexte (plus stable que le direct), "
               "puis recalé exactement sur l'original : même durée, synchro avec la vidéo.")
        for w in self.voice_widgets:
            w.setEnabled(bool(cfg["convert"]))
        self.index = st._add("voice", "Voix", page, f)
        st.rail.changed.connect(lambda i: i == self.index and self._fill_models())       # dossiers changés au Direct
        self._refresh()

    def _msl(self, f, key, text, lo, hi, fmt, tip=None):
        f.slider(text, lo, hi, int(self.mic[key]), lambda x: self.mic.__setitem__(key, x), fmt, int(MIC_DEFAULTS[key]), key, tip)

    def _mck(self, f, key, text, tip=None):
        f.check(text, self.mic[key], lambda on: self.mic.__setitem__(key, bool(on)), key, tip, bool(MIC_DEFAULTS[key]))

    def _set_convert(self, on):
        self.cfg["convert"] = bool(on)
        for w in self.voice_widgets:
            w.setEnabled(bool(on))
        self._changed(force=True)

    def same_as_live_mic(self):
        live = self.st.settings.get("mic", {})
        for k in MIC_DEFAULTS:
            v = live.get(k, MIC_DEFAULTS[k])
            if k in self.mic_panel:
                self.mic_panel[k][1](v if isinstance(v, bool) else int(v))
            else:
                self.mic[k] = v
        self.fx_tab.curve.update()
        self.fx_tab._band_to_ui(self.fx_tab.curve.sel)
        self.st.toast("Réglages du micro du Direct repris")

    def _sl(self, f, key, text, lo, hi, fmt, tip=None):
        def changed(x):
            self.cfg[key] = x
            self._changed()
        self.sliders[key] = f.slider(text, lo, hi, int(self.cfg[key]), changed, fmt, int(VOICE_DEFAULTS[key]), "sv_" + key, tip)

    def _ck(self, f, key, text, tip=None):
        def changed(on):
            self.cfg[key] = bool(on)
            self._changed()
        self.sliders[key] = f.check(text, self.cfg[key], changed, "sv_" + key, tip, bool(VOICE_DEFAULTS[key]))

    # ------------------------------------------------------------ modèles
    def _library(self):
        from ui_voix import scan_voices
        s = self.st.settings
        found = [p for _l, p in scan_voices(s)]
        favs = [p for p in s.get("voice_favorites", []) if os.path.isfile(p)]
        rest = sorted((p for p in found if os.path.normcase(p) not in {os.path.normcase(q) for q in favs}),
                      key=lambda p: os.path.basename(p).lower())
        return favs, rest

    def _fill_models(self):
        favs, rest = self._library()
        cur = self.cfg["pth"]
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem("Choisir une voix…", "")
        for p in favs:
            self.model.addItem("★ " + os.path.splitext(os.path.basename(p))[0], p)
        for p in rest:
            self.model.addItem(os.path.splitext(os.path.basename(p))[0], p)
        if cur and self.model.findData(cur) < 0 and os.path.isfile(cur):
            self.model.addItem(os.path.splitext(os.path.basename(cur))[0], cur)
        i = self.model.findData(cur)
        self.model.setCurrentIndex(max(0, i))
        self.model.blockSignals(False)

    def _model_picked(self, i):
        path = self.model.itemData(i) or ""
        if path and path != self.cfg["pth"]:
            self.cfg["pth"] = path
            self._changed(force=True)

    def _choose_model(self):
        start = os.path.dirname(self.cfg["pth"]) if self.cfg["pth"] else self.st.settings.get("voice_library", "")
        start = start or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self.st.window(), "Modèle de voix", start or os.path.expanduser("~"),
                                              "Modèles de voix RVC (*.pth)")
        if path:
            self.cfg["pth"] = os.path.normpath(path)
            self._fill_models()
            self._changed(force=True)

    def same_as_live(self):
        live = self.st.settings.get("voix", {})
        for k in KEYS:
            if k in live and k in self.sliders:
                w = self.sliders[k]
                (w.setChecked if hasattr(w, "setChecked") else w.setValue)(live[k] if isinstance(live[k], bool) else int(live[k]))
        if live.get("pth") and os.path.isfile(live["pth"]):
            self.cfg["pth"] = live["pth"]
            self._fill_models()
        self._changed(force=True)
        self.st.toast("Réglages de la voix du Direct repris")

    # ------------------------------------------------------------ son d'origine
    def _fill_devices(self):
        self.dev.blockSignals(True)
        self.dev.clear()
        self.dev.addItem("Micro par défaut de Windows", "")
        try:
            import phonemic
            for name in phonemic.devices("input"):
                self.dev.addItem(name, name)
        except Exception:  # noqa: BLE001
            pass
        i = self.dev.findData(self.cfg["rec_device"])
        self.dev.setCurrentIndex(max(0, i))
        self.dev.blockSignals(False)

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self.st.window(), "Son à transformer", self.cfg.get("folder", "") or
                                              os.path.expanduser("~"), OPEN_FILTER)
        if path:
            self.cfg["folder"] = os.path.dirname(path)
            self.open(os.path.normpath(path))

    def open(self, path, video=""):
        self.player.stop()
        self.src_label.setText(f"Lecture de {os.path.basename(path)}…")
        self.tasks.put(("load", path, video))

    def from_media(self):
        m = self.st.cur
        if m is None or m["info"].get("image"):
            self.st.toast("Choisis d'abord une vidéo dans Médias")
            return
        if not m["info"].get("audio"):
            self.st.toast("Cette vidéo n'a pas de son")
            return
        self.open(m["info"]["path"], video=m["info"]["path"])

    def toggle_record(self):
        if self.rec is not None:
            self._stop_record()
            return
        import sounddevice as sd
        self.player.stop()
        name = self.dev.currentData() or ""
        chunks = []

        def cb(data, _frames, _t, _s):
            x = data.mean(1) if data.ndim > 1 else data
            chunks.append(x.astype(np.float32).copy())
            self.level = 20 * np.log10(float(np.abs(x).max()) + 1e-9)
        try:
            if name:
                apis = [h["name"] for h in sd.query_hostapis()]
                wasapi = apis.index("Windows WASAPI")
                idx = next(i for i, d in enumerate(sd.query_devices())
                           if d["name"] == name and d["hostapi"] == wasapi and d["max_input_channels"] > 0)
                ch = min(2, sd.query_devices(idx)["max_input_channels"])
                stream = sd.InputStream(device=idx, channels=ch, samplerate=SR, dtype="float32", blocksize=960,
                                        extra_settings=sd.WasapiSettings(auto_convert=True), callback=cb)
            else:
                stream = sd.InputStream(channels=1, samplerate=SR, dtype="float32", blocksize=960, callback=cb)
            stream.start()
        except Exception as e:  # noqa: BLE001
            self.st.toast(f"Micro indisponible : {e}")
            return
        self.rec = [chunks, stream, time.time()]
        self.rec_btn.setText("Arrêter l'enregistrement")
        self.rec_meter.show()

    def _stop_record(self):
        chunks, stream, _t0 = self.rec
        self.rec = None
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001
            pass
        self.rec_btn.setText("Enregistrer")
        self.rec_meter.hide()
        x = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        if len(x) < SR // 2:
            self.st.toast("Enregistrement trop court")
            return
        self.tasks.put(("save_rec", x))

    def _recorded(self, path):
        self.st.toast("Enregistrement gardé dans data/studio-voix")
        self.open(path)

    def _loaded(self, x, msg):
        if x is None:
            self.src_label.setText("Aucun son choisi")
            self.st.toast(msg)
            return
        self.src = x
        self.src_name, self.src_video = msg.split("\n", 1) if "\n" in msg else (msg, "")
        self.out, self.out_key = None, None
        self.player.tracks = [x, None]
        self.player.pos = 0
        self.wave.set_audio(x)
        self.src_label.setText(f"{os.path.basename(self.src_name)}  ·  {clock(len(x) / SR)}"
                               + ("  ·  son de la vidéo" if self.src_video else ""))
        self._refresh()
        if self._model_ok():
            self.convert(auto=True)

    # ------------------------------------------------------------ conversion
    def _params(self):
        return {k: self.cfg[k] for k in KEYS}

    def _model_ok(self):
        return not self.cfg["convert"] or bool(self.cfg["pth"] and os.path.isfile(self.cfg["pth"]))

    def _key(self):
        return (id(self.src), self.cfg["convert"], self.cfg["pth"] if self.cfg["convert"] else "",
                tuple(self._params().items()), tuple(sorted(self.mic.items())))

    def _changed(self, force=False):
        self._refresh()
        if self.src is not None and self._model_ok() and (force or self.out is not None) and len(self.src) <= AUTO_S * SR:
            self.auto.start()

    def convert(self, auto=False):
        if self.src is None:
            if not auto:
                self.st.toast("Choisis d'abord un son")
            return
        if not self._model_ok():
            if not auto:
                self.st.toast("Choisis une voix (ou coupe « Changer la voix avec un modèle »)")
            return
        if self.out is not None and self.out_key == self._key():
            return
        self.want += 1
        if self.busy:
            self.cancel = True              # la conversion en cours s'arrête : la nouvelle la remplace
        self.idle.stop()
        self.tasks.put(("convert", self.want, self.src, self.cfg["pth"] if self.cfg["convert"] else "", self._params(),
                        dict(self.mic), self._key()))
        self._refresh()

    def _converted(self, y, n, info):
        if n != self.want:
            return
        if y is None:
            self._refresh()
            return
        key, msg = info
        self.out, self.out_key = y, key
        self.player.tracks = [self.src, y]
        self.wave.set_audio(self.src, y)
        self.idle.start()
        self._refresh()
        if self.out_key == self._key():
            self.state.setText(msg)

    def _progress(self, v, text):
        if v < 0:
            self.bar.hide()
        else:
            self.bar.show()
            self.bar.setValue(int(v * 1000))
        if text:
            self.state.setText(text)

    def _work(self):
        while True:
            job = self.tasks.get()
            try:
                getattr(self, "_t_" + job[0])(*job[1:])
            except Exception as e:  # noqa: BLE001
                self.busy = False
                self.b.progress.emit(-1, "")
                self.b.done.emit(f"Erreur : {e}")

    def _t_load(self, path, video):
        import phonemic
        try:
            x = phonemic.load_audio(path, LIMIT_S)
        except Exception as e:  # noqa: BLE001
            self.b.loaded.emit(None, f"Son illisible : {e}")
            return
        if not len(x):
            self.b.loaded.emit(None, "Ce fichier n'a pas de son")
            return
        self.b.loaded.emit(x, path + ("\n" + video if video else ""))

    def _t_save_rec(self, x):
        from scipy.io import wavfile
        os.makedirs(RECORDS, exist_ok=True)
        path = os.path.join(RECORDS, time.strftime("enregistrement_%Y-%m-%d_%H-%M-%S.wav"))
        wavfile.write(path, SR, x.astype(np.float32))
        self.b.recorded.emit(path)

    def _t_convert(self, n, x, pth, params, mic, key):
        if n != self.want:
            return
        self.busy, self.cancel = True, False
        stop = lambda: self.cancel or n != self.want     # noqa: E731
        try:
            t0 = time.perf_counter()
            secs = len(x) / SR
            vkey = (id(x), pth, tuple(params.items()) if pth else (), tuple((k, mic[k]) for k in IN_DEFAULTS))
            if self._vcache is not None and self._vcache[0] == vkey:
                v = self._vcache[1]                       # seuls les effets du micro ont changé
            else:
                self.b.progress.emit(0, "Niveau d'entrée…")
                v = apply_input(x, mic)
                if pth:
                    if self.engine is None:
                        from voix.fichier import FileVoice
                        self.engine = FileVoice()
                    idx = find_index(pth)
                    if self.engine.loaded() != (pth, idx):
                        self.b.progress.emit(0, f"Chargement de la voix {os.path.splitext(os.path.basename(pth))[0]}… "
                                                "(10 à 20 s la première fois)")
                        self.engine.load(pth, idx)
                    t0 = time.perf_counter()
                    v = self.engine.convert(v, params, lambda q: self.b.progress.emit(0.85 * q, f"Transformation… {int(85 * q)} %"),
                                            stop)
                    if v is None:
                        self.b.converted.emit(None, n, "")
                        return
                self._vcache = (vkey, v)
            y = apply_mic(v, mic, lambda q: self.b.progress.emit(0.85 + 0.15 * q, f"Effets du micro… {int(85 + 15 * q)} %"),
                          stop)
            if y is None:
                self.b.converted.emit(None, n, "")
                return
            dt = time.perf_counter() - t0
            self.b.progress.emit(-1, "")
            self.b.converted.emit(y, n, (key, f"Transformé : {clock(secs)} de son en {dt:.1f} s. "
                                              "« Original » / « Transformée » pour comparer au même endroit."))
        finally:
            self.busy = False

    def _t_release(self):
        if self.engine is not None and not self.busy:
            self.engine.release()
            self.engine = None

    def _t_export(self, path, y, args, video):
        self.b.progress.emit(0.5, "Écriture du fichier…")
        ffmpeg_write(path, y, args, video)
        self.b.progress.emit(-1, "")
        self.b.done.emit(f"__exp__{path}")

    def _done(self, text):
        if text.startswith("__exp__"):
            path = text[7:]
            self.st.toast(f"Exporté : {os.path.basename(path)}")
            self.state.setText(f"Exporté : {os.path.basename(path)}\ndans {os.path.dirname(path)}")
            self.state.setToolTip(path)
        else:
            self.state.setText(text)
            self.st.toast(text)
            self._refresh(keep_text=True)

    # ------------------------------------------------------------ export
    def _ready_out(self):
        if self.src is None:
            self.st.toast("Choisis d'abord un son")
            return None
        if not self._model_ok():
            self.st.toast("Choisis une voix (ou coupe « Changer la voix avec un modèle »)")
            return None
        if self.out is None or self.out_key != self._key() or self.busy:
            self.convert()
            self.st.toast("Transformation en cours : exporte quand elle est finie")
            return None
        return self.out

    def export_audio(self):
        y = self._ready_out()
        if y is None:
            return
        name = list(FORMATS)[int(self.cfg["format"])]
        ext, args = FORMATS[name]
        voice = os.path.splitext(os.path.basename(self.cfg["pth"]))[0] if self.cfg["convert"] else "effets"
        base = os.path.splitext(os.path.basename(self.src_name))[0]
        start = os.path.join(self.cfg.get("out_folder") or os.path.dirname(self.src_name), f"{base} - {voice}{ext}")
        path, _ = QFileDialog.getSaveFileName(self.st.window(), "Exporter le son", start, f"{name} (*{ext})")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(self.src_name)):
            self.st.toast("Choisis un autre nom : le fichier d'origine n'est jamais remplacé")
            return
        self.cfg["out_folder"] = os.path.dirname(path)
        self.tasks.put(("export", path, y, args, None))

    def export_video(self):
        if not self.src_video:
            self.st.toast("Pour cela, prends le son d'une vidéo avec « Son du média »")
            return
        y = self._ready_out()
        if y is None:
            return
        voice = os.path.splitext(os.path.basename(self.cfg["pth"]))[0] if self.cfg["convert"] else "effets"
        base, ext = os.path.splitext(os.path.basename(self.src_video))
        ext = ext if ext.lower() in (".mp4", ".mov", ".mkv") else ".mp4"
        start = os.path.join(self.cfg.get("out_folder") or os.path.dirname(self.src_video), f"{base} - {voice}{ext}")
        path, _ = QFileDialog.getSaveFileName(self.st.window(), "Exporter la vidéo avec cette voix", start, f"Vidéo (*{ext})")
        if not path:
            return
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(self.src_video)):
            self.st.toast("Choisis un autre nom : le fichier d'origine n'est jamais remplacé")
            return
        self.cfg["out_folder"] = os.path.dirname(path)
        self.tasks.put(("export", path, y, None, self.src_video))

    # ------------------------------------------------------------ lecture
    def toggle_play(self):
        p = self.player
        if self.src is None:
            self.open_dialog()
            return
        if p.playing:
            p.stop()
        else:
            if self.ab.currentIndex() == 1 and self.out is None:
                self.convert()
            try:
                p.play()
            except Exception as e:  # noqa: BLE001
                self.st.toast(f"Sortie son indisponible : {e}")

    def _set_ab(self, i):
        self.player.which = i

    def seek(self, v):
        n = self.player.length()
        if n:
            self.player.pos = int(v * n)

    def handle_key(self, ev):
        k = ev.key()
        if k in (Qt.Key_Space, Qt.Key_K):
            self.toggle_play()
        elif k == Qt.Key_A:
            self.ab.setCurrentIndex(1 - self.ab.currentIndex())
        elif k in (Qt.Key_Left, Qt.Key_Right) and self.player.length():
            self.player.pos = max(0, min(self.player.length() - 1, self.player.pos + (SR * 2 if k == Qt.Key_Right else -SR * 2)))
        elif k == Qt.Key_Home:
            self.player.pos = 0
        else:
            return False
        return True

    # ------------------------------------------------------------ affichage
    def _refresh(self, keep_text=False):
        has_src = self.src is not None
        self.vid_btn.setVisible(bool(self.src_video))
        m = self.st.cur
        self.media_btn.setEnabled(m is not None and not m["info"].get("image") and bool(m["info"].get("audio")))
        if keep_text:
            return
        if not has_src:
            text = "① Choisis un son (fichier, enregistrement ou son d'une vidéo), ② une voix, puis écoute et exporte."
        elif not self._model_ok():
            text = "Choisis une voix dans « 3 · La voix » (ou coupe « Changer la voix » pour les effets seuls)."
        elif self.busy or (self.out is None or self.out_key != self._key()) and self.auto.isActive():
            text = "Transformation…"
        elif self.out is None or self.out_key != self._key():
            text = ("Réglages changés : écoute « Transformée » pour réentendre." if self.out is not None
                    else "Prêt : « Écouter » transforme le son avec cette voix.")
        else:
            text = "À jour : « Original » / « Transformée » pour comparer au même endroit."
        self.state.setText(text)

    def tick(self):
        p = self.player
        n = p.length()
        self.wave.pos = p.pos / n if n else 0.0
        self.wave.update()
        if self.rec is not None:
            t = time.time() - self.rec[2]
            self.time.setText(clock(t))
            self.rec_meter.setValue(self.level)
            if t > LIMIT_S:
                self._stop_record()
        else:
            self.time.setText(f"{clock(p.pos / SR)} / {clock(n / SR)}" if n else "00:00.0")
        playing = p.playing
        if getattr(self, "_was_playing", None) != playing:
            self._was_playing = playing
            self.play_btn.setText("Pause" if playing else "Écouter")
            self.play_btn.setIcon(icon("pause" if playing else "play", 14))
        if p.which == 1 and playing and self.out is None:
            if not self.busy and not self.auto.isActive() and self.src is not None and self._model_ok():
                self.convert()
        seen = tuple(sorted(self.mic.items()))      # un effet du micro changé (curseur, courbe, style…)
        if seen != self._mic_seen:
            first = self._mic_seen is None
            self._mic_seen = seen
            if not first:
                self._changed()
        t = p.tracks[p.which] if playing else None   # spectre du son en lecture, sur la courbe de l'égaliseur
        fx = self.fx_view.fx
        if t is not None and p.pos + 256 <= len(t):
            X = np.abs(np.fft.rfft(t[p.pos:p.pos + 256] * np.hanning(256)))
            fx.spec = (0.6 * fx.spec + 0.4 * 20 * np.log10(X + 1e-9)) if fx.spec.max() > -150 else 20 * np.log10(X + 1e-9)
        elif fx.spec.max() > -150:
            fx.spec = np.full(129, -200.0)
