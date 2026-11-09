"""Page Presets : des profils (« Visage 1 »), et dans chacun plusieurs presets (« Visage Finnyzyy »…).
La même page dans le Direct et le Studio : les presets sont communs et s'appliquent aux deux espaces.

Un preset garde plusieurs branches d'un coup : un dictionnaire {branche: {réglage: valeur}} dans
settings["preset_groups"][profil][preset]. Seules les branches gardées changent quand on l'applique.
Webcam et Ambiance sont des branches séparées, mais partagent le panneau « effets ».

Presets fournis (presets_fournis.json) : « Par défaut », en tête de chaque profil, ni modifiable ni
supprimable, et appliqué au premier lancement ; les autres sont rangés dans « Visage 1 » au premier lancement."""
import json
import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from ui_kit import icon, note

SECTIONS = (("visage", "Visage"), ("dlc", "Moteur"), ("filtres", "Retouche"), ("webcam", "Webcam"), ("ambiance", "Ambiance"),
            ("sortie", "Sortie"), ("micro", "Micro"), ("voix", "Voix"))
LOOK = ("visage", "dlc", "filtres", "webcam", "ambiance")   # branches cochées pour un nouveau preset
OLD_GROUP = "Anciens préréglages"
FIRST_GROUP = "Visage 1"
DEFAULT = "Par défaut"
def _builtin_path():
    """Version distribuée : « Presets\\Presets fournis.json » à côté du lanceur ; sinon, à côté du code."""
    from server import PRESETS_DIR
    top = os.path.join(PRESETS_DIR, "Presets fournis.json") if PRESETS_DIR else ""
    return top if top and os.path.isfile(top) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets_fournis.json")


with open(_builtin_path(), encoding="utf-8") as _f:
    BUILTIN = json.load(_f)


def _effects(values):
    """Sépare les réglages du panneau « effets » en Webcam et Ambiance."""
    return ({k: v for k, v in values.items() if not k.startswith("grade_")},
            {k: v for k, v in values.items() if k.startswith("grade_")})


def split(shot):
    """Toutes les valeurs par panneau → par branche."""
    out = {k: dict(shot[k]) for k, _t in SECTIONS if k in shot}
    out["webcam"], out["ambiance"] = _effects(shot.get("effets", {}))
    return out


def seed(settings):
    """Au premier lancement : « Visage 1 » avec les presets fournis, et les préréglages par branche
    d'avant rangés dans leur propre profil, pour n'en perdre aucun."""
    if "preset_groups" in settings:
        return settings["preset_groups"]
    groups = settings["preset_groups"] = {}
    groups[FIRST_GROUP] = {name: preset for name, preset in BUILTIN.items() if name != DEFAULT}
    titles = dict(SECTIONS)
    old = {}
    for panel, saved in settings.get("user_presets", {}).items():
        for name, values in saved.items():
            if panel == "effets":
                web, amb = _effects(values)
                old[f"Effets · {name}"] = {k: v for k, v in (("webcam", web), ("ambiance", amb)) if v}
            elif panel in titles:
                old[f"{titles[panel]} · {name}"] = {panel: dict(values)}
    if old:
        groups[OLD_GROUP] = old
    return groups


class NewPreset(QDialog):
    """Nom du preset et branches qu'il garde."""

    def __init__(self, parent, name="", chosen=LOOK):
        super().__init__(parent)
        self.setWindowTitle("Nouveau preset")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Nom"))
        self.name = QLineEdit(name)
        lay.addWidget(self.name)
        lay.addWidget(QLabel("Branches gardées (les autres ne changeront pas)"))
        self.boxes = {}
        for key, title in SECTIONS:
            box = QCheckBox(title)
            box.setChecked(key in chosen)
            self.boxes[key] = box
            lay.addWidget(box)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def sections(self):
        return [k for k, box in self.boxes.items() if box.isChecked()]


class PresetHub:
    """Les presets, communs aux deux espaces ; une page dans chacun."""

    def __init__(self, settings, live, studio):
        self.live = live
        self.spaces = (live, studio)
        first = "preset_groups" not in settings
        self.groups = seed(settings)
        self.active = settings.setdefault("preset_active", [])
        self.views = []
        for space in self.spaces:
            view = PresetsPanel(self, space)
            space.presets_form.addRow(view)
            self.views.append(view)
        if first:       # une installation neuve démarre sur « Par défaut »
            QTimer.singleShot(0, lambda: self.apply(FIRST_GROUP, DEFAULT))

    def names(self, group):
        return [DEFAULT] + list(self.groups.get(group, {}))

    def get(self, group, name):
        return BUILTIN[DEFAULT] if name == DEFAULT else self.groups[group][name]

    def apply_default(self):
        self.apply(self.views[0].group() if self.views else "", DEFAULT)

    def apply(self, group, name):
        for space in self.spaces:
            space.load_preset(self.get(group, name))
        self.active[:] = [group, name]
        self.refresh(group)
        self.changed()
        self.live.toast(f"{group} : {name}")

    def snapshot(self, space):
        """L'état actuel : celui de cet espace, complété par les réglages propres à l'autre."""
        now = {}
        for s in sorted(self.spaces, key=lambda x: x is space):
            for key, values in split(s.shot_all()).items():
                now.setdefault(key, {}).update(values)
        return now

    def refresh(self, group=None):
        for view in self.views:
            view.reload(group if group is not None else view.group())

    def changed(self):
        self.live.persist_cb()


class PresetsPanel(QWidget):
    def __init__(self, hub, space):
        super().__init__()
        self.hub, self.space = hub, space
        self.groups, self.active = hub.groups, hub.active

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(8)
        self.combo.currentTextChanged.connect(lambda _t: self.fill())
        row.addWidget(self.combo, 1)
        for name, tip, fn in (("plus", "Nouveau profil", self.add_group), ("pencil", "Renommer ce profil", self.rename_group),
                              ("trash", "Supprimer ce profil", self.drop_group)):
            row.addWidget(self._tool(name, tip, fn))
        lay.addLayout(row)
        self.rows = QVBoxLayout()
        self.rows.setSpacing(4)
        lay.addLayout(self.rows)
        add = QPushButton("Nouveau preset")
        add.setToolTip("Depuis les réglages actuels")
        add.clicked.connect(self.add_preset)
        lay.addWidget(add)
        hint = note("Un clic applique le preset. La disquette le remplace par les réglages actuels, "
                    "seulement pour les branches qu'il garde.")
        hint.setMinimumWidth(1)
        lay.addWidget(hint)
        self.reload(self.active[0] if self.active else None)

    @staticmethod
    def _tool(name, tip, fn):
        b = QPushButton()
        b.setObjectName("icon")
        b.setIcon(icon(name, 16))
        b.setToolTip(tip)
        b.clicked.connect(lambda: fn())
        return b

    def group(self):
        return self.combo.currentText()

    def reload(self, select=None):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(self.groups))
        if select in self.groups:
            self.combo.setCurrentText(select)
        self.combo.blockSignals(False)
        self.fill()

    def fill(self):
        while self.rows.count():
            w = self.rows.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        group = self.group()
        for name in self.hub.names(group):
            preset = self.hub.get(group, name)
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            on = self.active == [group, name]
            apply_btn = QPushButton(("✓ " if on else "") + name)
            apply_btn.setToolTip(name + " : " + " · ".join(t for k, t in SECTIONS if k in preset))
            apply_btn.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)   # un nom long ne pousse pas les boutons dehors
            apply_btn.setStyleSheet("text-align: left; padding-left: 10px;" + ("font-weight: 600;" if on else ""))
            apply_btn.clicked.connect(lambda _c=False, n=name: self.hub.apply(group, n))
            h.addWidget(apply_btn, 1)
            if name == DEFAULT:
                lock = self._tool("lock", "Preset fourni : toujours là, ni modifiable ni supprimable", lambda: None)
                lock.setEnabled(False)
                h.addWidget(lock)
                self.rows.addWidget(row)
                continue
            h.addWidget(self._tool("save", "Remplacer par les réglages actuels", lambda n=name: self.update_preset(n)))
            h.addWidget(self._tool("pencil", "Renommer", lambda n=name: self.rename_preset(n)))
            h.addWidget(self._tool("trash", "Supprimer", lambda n=name: self.drop_preset(n)))
            self.rows.addWidget(row)

    def _ask(self, title, label, text=""):
        name, ok = QInputDialog.getText(self.window(), title, label, text=text)
        return name.strip() if ok else ""

    @staticmethod
    def _keep_order(items, old, new):
        """Renommer sans changer la place dans la liste."""
        renamed = {(new if k == old else k): v for k, v in items.items()}
        items.clear()
        items.update(renamed)

    # ---------- profils
    def add_group(self):
        name = self._ask("Profil", "Nom du profil :")
        if name and name not in self.groups:
            self.groups[name] = {}
            self.hub.refresh(name)
            self.hub.changed()

    def rename_group(self):
        old = self.group()
        name = self._ask("Profil", "Nouveau nom :", old)
        if name and name != old and name not in self.groups:
            self._keep_order(self.groups, old, name)
            if self.active and self.active[0] == old:
                self.active[0] = name
            self.hub.refresh(name)
            self.hub.changed()

    def drop_group(self):
        old = self.group()
        if old and QMessageBox.question(self.window(), "Profil", f"Supprimer « {old} » et ses presets ?") == QMessageBox.Yes:
            self.groups.pop(old, None)
            if self.active and self.active[0] == old:
                self.active.clear()
            self.hub.refresh("")
            self.hub.changed()

    # ---------- presets
    def add_preset(self):
        group = self.group() or self._first_group()
        if not group:
            return
        box = NewPreset(self.window())
        if box.exec() != QDialog.Accepted:
            return
        name = box.name.text().strip()
        if not name or not box.sections():
            return
        if name == DEFAULT:
            QMessageBox.information(self.window(), "Preset", f"« {DEFAULT} » est réservé : choisis un autre nom.")
            return
        if name in self.groups[group] and QMessageBox.question(self.window(), "Preset", f"Remplacer « {name} » ?") != QMessageBox.Yes:
            return
        now = self.hub.snapshot(self.space)
        self.groups[group][name] = {k: now[k] for k in box.sections()}
        self.hub.refresh()
        self.hub.changed()

    def _first_group(self):
        name = self._ask("Profil", "Nom du profil :")
        if name:
            self.groups.setdefault(name, {})
            self.hub.refresh(name)
        return name

    def update_preset(self, name):
        if QMessageBox.question(self.window(), "Preset", f"Remplacer « {name} » par les réglages actuels ?") != QMessageBox.Yes:
            return
        now = self.hub.snapshot(self.space)
        preset = self.groups[self.group()][name]
        for key in list(preset):
            preset[key] = now[key]
        self.hub.changed()
        self.hub.live.toast(f"« {name} » mis à jour")

    def rename_preset(self, old):
        group = self.group()
        name = self._ask("Preset", "Nouveau nom :", old)
        if name and name != old and name != DEFAULT and name not in self.groups[group]:
            self._keep_order(self.groups[group], old, name)
            if self.active == [group, old]:
                self.active[1] = name
            self.hub.refresh()
            self.hub.changed()

    def drop_preset(self, name):
        group = self.group()
        if QMessageBox.question(self.window(), "Preset", f"Supprimer « {name} » ?") == QMessageBox.Yes:
            self.groups[group].pop(name, None)
            if self.active == [group, name]:
                self.active.clear()
            self.hub.refresh()
            self.hub.changed()
