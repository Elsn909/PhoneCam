"""Accueil du direct : les appareils et quelques actions, reglages experts a part."""
import os
from pathlib import Path
import threading

from PySide6.QtCore import QObject, Qt, Signal, QSignalBlocker
from PySide6.QtWidgets import QComboBox, QFileDialog, QPushButton

from audio_routing import virtual_microphone
from server import VOICES_DIR
from ui_kit import Form, Page, Toggle, note
from ui_voix import scan_voices, set_voice_folders, voice_folders
from voix import find_index


class _ImportBridge(QObject):
    done = Signal(str, str)


class SimpleDashboard(Page):
    def __init__(self, live):
        super().__init__('Ton studio', 'Choisis ton image, ta voix, puis lance le direct.')
        self.live = live
        self._importing = False
        self._import_bridge = _ImportBridge(self)
        self._import_bridge.done.connect(self._imported)
        self._photo_path = None
        self._voice_signature = None
        f = Form(self)

        f.section('Caméra et visage')
        self.camera = self._combo(live.cam_combo, f, 'Caméra')
        self.camera.activated.connect(lambda i: live.cam_combo.activated.emit(i))
        self.photo = note('Aucune photo choisie')
        f.form.addRow(self.photo)
        choose = QPushButton('Choisir le visage…')
        choose.clicked.connect(live._choose_face)
        f.form.addRow(choose)
        self._toggle(f, 'Remplacer le visage', live.swap_on)

        f.section('Voix')
        self.voices = QComboBox()
        self.voices.setMinimumContentsLength(8)
        self.voices.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.voices.activated.connect(self._choose_voice)
        f.form.addRow('Modèle', self.voices)
        self.import_button = QPushButton('Ajouter un pack de voix (.zip)…')
        self.import_button.clicked.connect(self._import_pack)
        f.form.addRow(self.import_button)
        self._toggle(f, 'Transformer ma voix', live.voice_page.on)
        pitch = live.panels['voix']['pitch']
        self.pitch = f.slider('Hauteur', -24, 24, int(pitch[0]()), pitch[1],
                              lambda x: 'Naturelle' if x == 0 else f'{x:+d}', default=0)
        self.voice_state = note('')
        f.form.addRow(self.voice_state)

        f.section('Micro')
        self.microphone = self._combo(live.dev_in, f, 'Ton micro')
        self.microphone.currentIndexChanged.connect(lambda i: self._audio_changed(live.dev_in, i))
        self.monitor = self._combo(live.dev_mon, f, 'Retour casque')
        self.monitor.currentIndexChanged.connect(lambda i: self._audio_changed(live.dev_mon, i))
        self._toggle(f, 'Micro actif', live.run_btn)
        self.routing = note('')
        self.routing.setTextFormat(Qt.PlainText)
        f.form.addRow(self.routing)
        route = QPushButton('Configurer automatiquement le micro')
        route.clicked.connect(self._route)
        f.form.addRow(route)
        self.install_button = QPushButton('Installer les prérequis manquants')
        self.install_button.clicked.connect(self._install)
        f.form.addRow(self.install_button)

        f.section('Direct')
        self._toggle(f, 'Caméra virtuelle', live.vbtn)
        self.start = QPushButton('Lancer caméra + micro')
        self.start.setObjectName('primary')
        self.start.clicked.connect(self._start)
        f.form.addRow(self.start)
        self.state = note('')
        f.form.addRow(self.state)
        self.advanced_button = QPushButton('Réglages avancés')
        self.advanced_button.clicked.connect(lambda: live.set_simple_mode(not live.simple_mode))
        f.form.addRow(self.advanced_button)
        self.refresh_voices()
        self.refresh()

    @staticmethod
    def _combo(source, form, title):
        combo = QComboBox()
        combo.setModel(source.model())
        combo.setCurrentIndex(source.currentIndex())
        combo.setMinimumContentsLength(8)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        form.form.addRow(title, combo)
        return combo

    @staticmethod
    def _toggle(form, title, source):
        toggle = Toggle(title)
        toggle.setChecked(source.isChecked())
        toggle.toggled.connect(source.setChecked)
        source.toggled.connect(toggle.setChecked)
        form.form.addRow(toggle)
        return toggle

    def _audio_changed(self, source, index):
        source.setCurrentIndex(index)

    def refresh_voices(self):
        current = self.live.pipeline.voice.p.get('pth', '')
        self.voices.blockSignals(True)
        self.voices.clear()
        self.voices.addItem('Choisir une voix', '')
        for label, path in sorted(scan_voices(self.live.settings)):
            self.voices.addItem(label, path)
        if current and self.voices.findData(current) < 0:
            self.voices.addItem(Path(current).stem, current)
        self.voices.setCurrentIndex(max(0, self.voices.findData(current)))
        self.voices.blockSignals(False)

    def _choose_voice(self, index):
        path = self.voices.itemData(index)
        if path:
            self.live.voice_page._load(path, find_index(path))

    def _import_pack(self):
        if self._importing:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Ajouter une voix RVC', '', 'Pack de voix (*.zip)')
        if not path:
            return
        self._importing = True
        self.import_button.setEnabled(False)
        self.import_button.setText('Vérification et import…')

        def work():
            try:
                from voice_import import import_voice_pack
                root = VOICES_DIR or str(Path(__file__).parent / 'data' / 'voix')
                pth = import_voice_pack(path, root)
                self._import_bridge.done.emit(pth, '')
            except Exception as exc:
                self._import_bridge.done.emit('', str(exc))
        threading.Thread(target=work, daemon=True, name='voice-import').start()

    def _imported(self, pth, error):
        self._importing = False
        self.import_button.setEnabled(True)
        self.import_button.setText('Ajouter un pack de voix (.zip)…')
        if error:
            self.live.toast(f'Import impossible : {error}')
            return
        root = VOICES_DIR or str(Path(pth).parent)
        folders = voice_folders(self.live.settings)
        if root not in folders:
            set_voice_folders(self.live.settings, folders + [root])
        self.live.voice_page._fill_library()
        self.refresh_voices()
        self.voices.setCurrentIndex(max(0, self.voices.findData(pth)))
        self.live.toast('Voix ajoutée. Choisis-la pour la charger.')

    def _route(self):
        live = self.live
        if live.pipeline.audio.running:
            live.run_btn.setChecked(False)
        live._rescan()
        self.refresh()
        name = virtual_microphone(live.dev_out.currentText())
        live.toast(f'Micro à choisir dans Discord / Snap : {name}' if name else
                   'Le câble audio virtuel est absent. Lance l’installation des prérequis.')

    def _install(self):
        launcher = Path(__file__).resolve().parents[2] / 'Installer les pilotes.bat'
        if launcher.is_file():
            os.startfile(str(launcher))
            self.live.toast('Installateur ouvert. Windows peut demander une autorisation administrateur.')

    def _start(self):
        self._route()
        self.live.vbtn.setChecked(True)
        if virtual_microphone(self.live.dev_out.currentText()):
            self.live.run_btn.setChecked(True)

    def refresh(self):
        live = self.live
        for target, source in ((self.camera, live.cam_combo), (self.microphone, live.dev_in), (self.monitor, live.dev_mon)):
            with QSignalBlocker(target):
                target.setCurrentIndex(source.currentIndex())
        path = live.settings.get('source', '')
        if path != self._photo_path:
            self._photo_path = path
            self.photo.setText(Path(path).name if path else 'Aucune photo choisie')
            self.photo.setTextFormat(Qt.PlainText)
        vp = live.pipeline.voice.p
        path = (getattr(live.voice_page, 'pending', ('', False))[0]
                if live.pipeline.voice.loading else vp.get('pth', ''))
        if path and self.voices.findData(path) < 0:
            self.voices.addItem(Path(path).stem, path)
        with QSignalBlocker(self.voices):
            self.voices.setCurrentIndex(max(0, self.voices.findData(path)))
        with QSignalBlocker(self.pitch):
            self.pitch.setValue(int(vp.get('pitch', 0)))
        self.voice_state.setText(live.pipeline.voice.status)
        name = virtual_microphone(live.dev_out.currentText())
        self.routing.setText(f'Dans Discord / Snapchat, choisis : {name}' if name else
                             'Micro virtuel manquant. Le retour casque seul ne transmet pas ta voix aux autres applications.')
        self.install_button.setVisible(not bool(name))
        camera_ok = live.server.fps() > 0
        self.state.setText(('Caméra reçue' if camera_ok else 'En attente de la caméra') + ' · '
                           + ('Micro en marche' if live.pipeline.audio.running else 'Micro arrêté'))
        self.advanced_button.setText('Réglages avancés' if live.simple_mode else 'Revenir au mode simple')
