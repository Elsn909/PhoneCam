"""Briques de l'interface : thème « Régie », icônes, commandes, cartes, pages, recherche, annulation."""
import os
import time
import unicodedata

import numpy as np
from PySide6.QtCore import (QByteArray, QEasingCurve, QEvent, QObject, QPointF, QPropertyAnimation, QRectF,
                            QSize, Qt, QTimer, Signal)
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QLinearGradient, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QFrame,
                               QGraphicsOpacityEffect, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
                               QToolButton, QVBoxLayout, QWidget)

ACCENT = "#ff6a3d"
C = dict(bg="#121110", panel="#191816", raised="#23211e", hover="#2c2926", line="#2b2926", line2="#3b3834",
         text="#f2eee8", dim="#a9a39a", faint="#756f66", accent=ACCENT, accent2="#ff9570", tint="#4a2a1e",
         good="#4fce9b", warn="#f5b94a", bad="#f2545b")
DISPLAY = '"Bahnschrift", "Segoe UI Variable Display", "Segoe UI"'

THEME = f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI"; }}
QWidget {{ color: {C['text']}; font-size: 9.5pt; }}
QWidget#app, QStackedWidget#workspaces {{ background: {C['bg']}; }}
QWidget#topbar {{ background: {C['bg']}; border-bottom: 1px solid {C['line']}; }}
QWidget#rail {{ background: {C['bg']}; border-right: 1px solid {C['line']}; }}
QWidget#inspector, QWidget#inspectorBody, QWidget#page, QWidget#tabBody {{ background: {C['panel']}; }}
QWidget#panelEdge {{ background: {C['panel']}; border-right: 1px solid {C['line']}; }}
QWidget#bottom {{ background: {C['bg']}; border-top: 1px solid {C['line']}; }}
QLabel#brand {{ font-family: {DISPLAY}; font-size: 12pt; font-weight: 600; }}
QLabel#brandMark {{ color: {C['accent']}; font-family: {DISPLAY}; font-size: 12pt; font-weight: 700; }}
QLabel#h1 {{ font-family: {DISPLAY}; font-size: 14.5pt; font-weight: 600; }}
QLabel#h2 {{ font-family: {DISPLAY}; font-size: 8.5pt; font-weight: 600; color: {C['dim']}; }}
QLabel#note {{ color: {C['faint']}; font-size: 8.5pt; }}
QLabel#status {{ color: {C['dim']}; font-size: 8.5pt; }}
QLabel#value {{ color: {C['dim']}; font-size: 8.5pt; }}
QLabel#mono {{ font-family: {DISPLAY}; color: {C['dim']}; font-size: 9pt; }}
QLabel#viewer {{ background: #0a0a09; color: {C['faint']}; border-radius: 10px; }}
QFrame#card {{ background: {C['raised']}; border: none; border-radius: 10px; }}
QFrame#sep {{ background: {C['line']}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#popup {{ background: {C['raised']}; border: 1px solid {C['line2']}; border-radius: 12px; }}
QToolButton#space {{ background: transparent; border: none; border-radius: 7px; padding: 5px 15px; color: {C['dim']};
                     font-family: {DISPLAY}; font-size: 10pt; font-weight: 600; }}
QToolButton#space:hover {{ color: {C['text']}; }}
QToolButton#space:checked {{ background: {C['raised']}; color: {C['text']}; }}
QToolButton#nav {{ background: transparent; border: none; border-radius: 9px; padding: 8px 0 6px 0; color: {C['faint']};
                   font-size: 7.5pt; }}
QToolButton#nav:hover {{ background: {C['panel']}; color: {C['text']}; }}
QToolButton#nav:checked {{ background: {C['raised']}; color: {C['text']}; }}
QToolButton#chip {{ background: transparent; border: none; border-radius: 12px; padding: 3px 8px; color: {C['dim']};
                    font-size: 8.8pt; font-weight: 600; }}
QToolButton#chip:hover {{ color: {C['text']}; background: {C['raised']}; }}
QToolButton#chip:checked {{ color: #1a0d07; background: {C['accent']}; }}
QToolButton#state {{ background: transparent; border: none; border-radius: 7px; padding: 4px 9px; color: {C['dim']};
                     font-size: 8.5pt; font-weight: 600; }}
QToolButton#state:hover {{ background: {C['raised']}; color: {C['text']}; }}
QToolButton#sectionHead {{ background: transparent; border: none; color: {C['dim']}; font-family: {DISPLAY};
                           font-size: 8.5pt; font-weight: 600; padding: 2px 0; text-align: left; }}
QToolButton#sectionHead:hover {{ color: {C['text']}; }}
QToolButton#ghost {{ background: transparent; border: none; border-radius: 6px; padding: 2px 5px; color: {C['faint']}; font-size: 8pt; }}
QToolButton#ghost:hover {{ background: {C['hover']}; color: {C['text']}; }}
QPushButton {{ background: {C['raised']}; border: none; border-radius: 8px; padding: 7px 13px; font-weight: 600; }}
QPushButton:hover {{ background: {C['hover']}; }}
QPushButton:pressed {{ background: {C['line']}; }}
QPushButton:checked {{ background: {C['tint']}; color: {C['accent2']}; }}
QPushButton:disabled {{ color: {C['faint']}; background: {C['panel']}; }}
QFrame#card QPushButton {{ background: {C['hover']}; }}
QFrame#card QPushButton:hover {{ background: {C['line2']}; }}
QPushButton#primary, QFrame#card QPushButton#primary {{ background: {C['accent']}; color: #1a0d07; padding: 9px 20px; font-weight: 700; }}
QPushButton#primary:hover, QFrame#card QPushButton#primary:hover {{ background: {C['accent2']}; }}
QPushButton#primary:disabled {{ background: {C['raised']}; color: {C['faint']}; }}
QPushButton#danger {{ color: {C['bad']}; }}
QPushButton#icon, QFrame#card QPushButton#icon {{ background: transparent; border: none; padding: 6px; border-radius: 8px; }}
QPushButton#icon:hover, QFrame#card QPushButton#icon:hover {{ background: {C['hover']}; }}
QPushButton#icon:checked, QFrame#card QPushButton#icon:checked {{ background: {C['tint']}; }}
QPushButton#pill {{ background: {C['panel']}; border-radius: 9px; padding: 8px 14px; font-weight: 600; font-size: 9pt; color: {C['dim']}; }}
QPushButton#pill:hover {{ background: {C['raised']}; color: {C['text']}; }}
QPushButton#pill:checked {{ background: {C['tint']}; color: {C['accent2']}; }}
QPushButton#air {{ background: {C['panel']}; border-radius: 9px; padding: 8px 16px; font-family: {DISPLAY}; font-weight: 600; color: {C['dim']}; }}
QPushButton#air:hover {{ background: {C['raised']}; color: {C['text']}; }}
QPushButton#air:checked {{ background: {C['accent']}; color: #1a0d07; }}
QPushButton#search {{ background: {C['panel']}; color: {C['faint']}; border-radius: 8px; padding: 6px 12px; font-weight: 400; text-align: left; }}
QPushButton#search:hover {{ background: {C['raised']}; color: {C['dim']}; }}
QComboBox, QLineEdit, QSpinBox {{ background: {C['hover']}; border: none; border-radius: 7px; padding: 5px 9px; min-height: 18px; }}
QComboBox:hover, QLineEdit:hover {{ background: {C['line2']}; }}
QLineEdit:focus {{ background: {C['line2']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {C['raised']}; border: 1px solid {C['line2']}; border-radius: 6px;
                               selection-background-color: {C['tint']}; selection-color: {C['text']}; outline: none; padding: 3px; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ padding: 6px; border-radius: 7px; }}
QListWidget::item:hover {{ background: {C['raised']}; }}
QListWidget::item:selected {{ background: {C['tint']}; color: {C['text']}; }}
QMenu {{ background: {C['raised']}; border: 1px solid {C['line2']}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {C['tint']}; }}
QMenu::separator {{ height: 1px; background: {C['line2']}; margin: 4px 6px; }}
QProgressBar {{ background: {C['hover']}; border: none; border-radius: 3px; text-align: center; font-size: 8pt; color: {C['dim']}; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 3px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {C['line2']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QToolTip {{ background: {C['raised']}; color: {C['text']}; border: 1px solid {C['line2']}; padding: 5px; }}
QSplitter::handle {{ background: {C['line']}; }}
QInputDialog, QMessageBox, QDialog {{ background: {C['panel']}; }}
"""

# Icônes (traits 24 × 24), dessinées ici pour ne dépendre d'aucune police ni d'aucun fichier.
ICONS = {
    "live": '<circle cx="12" cy="12" r="3"/><path d="M5 5a10 10 0 0 0 0 14M19 5a10 10 0 0 1 0 14M8 8a5.5 5.5 0 0 0 0 8M16 8a5.5 5.5 0 0 1 0 8"/>',
    "film": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 4v16M16 4v16M3 9h5M3 15h5M16 9h5M16 15h5"/>',
    "plug": '<path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-12 0zM12 17v4"/>',
    "face": '<circle cx="12" cy="12" r="9"/><circle cx="9" cy="10" r=".8"/><circle cx="15" cy="10" r=".8"/><path d="M8.5 15c1 1.2 2.2 1.8 3.5 1.8s2.5-.6 3.5-1.8"/>',
    "swap": '<path d="M4 8h13l-3-3M20 16H7l3 3"/>',
    "wand": '<path d="M5 19 16 8l3 3L8 22zM14 4v3M21 4h-3M19 7l-2-2"/>',
    "palette": '<path d="M12 3a9 9 0 1 0 0 18c1.5 0 2-1 2-2s-1-2 0-3 2-.5 3.500-.5A3.500 3.500 0 0 0 21 12a9 9 0 0 0-9-9z"/><circle cx="8" cy="11" r=".9"/><circle cx="12" cy="8" r=".9"/><circle cx="16" cy="11" r=".9"/>',
    "webcam": '<circle cx="12" cy="10" r="7"/><circle cx="12" cy="10" r="2.500"/><path d="M8 21h8M12 17v4"/>',
    "output": '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M9 21h6M12 17v4"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    "track": '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="3.500"/>',
    "export": '<path d="M12 15V4M8 8l4-4 4 4M5 14v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1.500" fill="currentColor"/>',
    "play": '<path d="M7 4.500v15l12-7.500z" fill="currentColor"/>',
    "pause": '<rect x="6.500" y="5" width="3.500" height="14" rx="1" fill="currentColor"/><rect x="14" y="5" width="3.500" height="14" rx="1" fill="currentColor"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "trash": '<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>',
    "compare": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 2v20"/>',
    "folder": '<path d="M3 6a1 1 0 0 1 1-1h5l2 3h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/>',
    "in": '<path d="M7 4H4v16h3M12 8v8M8 12h8"/>',
    "out": '<path d="M17 4h3v16h-3M8 12h8"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.600 5.600 7 7M17 17l1.400 1.400M5.600 18.400 7 17M17 7l1.400-1.400"/>',
    "search": '<circle cx="11" cy="11" r="6.500"/><path d="m20 20-4.300-4.300"/>',
    "star": '<path d="m12 3.500 2.600 5.400 5.900.8-4.300 4.100 1 5.900-5.200-2.800-5.200 2.800 1-5.900L3.500 9.700l5.900-.8z"/>',
    "gauge": '<path d="M4 18a9 9 0 1 1 16 0"/><path d="m12 14 4-5"/><circle cx="12" cy="14" r="1.300"/>',
    "shot": '<path d="M4 8a1 1 0 0 1 1-1h3l1.500-2.500h5L16 7h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1z"/><circle cx="12" cy="13" r="3.500"/>',
    "rec": '<circle cx="12" cy="12" r="6.500" fill="currentColor"/>',
    "expand": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    "grid": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16M15 4v16M3 9.300h18M3 14.700h18"/>',
    "sidebar": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
    "undo": '<path d="M8 6 4 10l4 4"/><path d="M4 10h10a5 5 0 0 1 0 10h-2"/>',
    "redo": '<path d="m16 6 4 4-4 4"/><path d="M20 10H10a5 5 0 0 0 0 10h2"/>',
    "reset": '<path d="M4 12a8 8 0 1 0 2.500-5.800L4 8.500"/><path d="M4 4v4.500h4.500"/>',
    "save": '<path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.500 9.500a2.500 2.500 0 1 1 3.700 2.200c-.8.500-1.200 1-1.200 2"/><circle cx="12" cy="17" r=".6"/>',
    "pencil": '<path d="M4 20l1-4L16 5l3 3L8 19z"/><path d="m14 7 3 3"/>',
    "layers": '<path d="m12 4 9 5-9 5-9-5z"/><path d="m3 14 9 5 9-5"/>',
    "cut": '<circle cx="6" cy="7" r="2.500"/><circle cx="6" cy="17" r="2.500"/><path d="M8 8.500 20 18M8 15.500 20 6"/>',
    "first": '<path d="M6 5v14M19 5 9 12l10 7z"/>',
    "last": '<path d="M18 5v14M5 5l10 7-10 7z"/>',
    "step_b": '<path d="M15 6l-6 6 6 6"/>',
    "step_f": '<path d="m9 6 6 6-6 6"/>',
}


def icon_pixmap(name, color=None, size=20):
    color = color or C["dim"]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" color="{color}">'
           f'{ICONS[name]}</svg>').replace("currentColor", color)
    r = QSvgRenderer(QByteArray(svg.encode()))
    screen = QApplication.primaryScreen()
    ratio = screen.devicePixelRatio() if screen else 1.0
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    r.render(p, QRectF(0, 0, pm.width(), pm.height()))
    p.end()
    pm.setDevicePixelRatio(ratio)
    return pm


def icon(name, size=20, on=None, off=None):
    ic = QIcon()
    ic.addPixmap(icon_pixmap(name, off or C["dim"], size), QIcon.Normal, QIcon.Off)
    ic.addPixmap(icon_pixmap(name, on or C["accent"], size), QIcon.Normal, QIcon.On)
    ic.addPixmap(icon_pixmap(name, on or C["accent"], size), QIcon.Active, QIcon.On)
    return ic


def app_icon():
    """Point « à l'antenne » dans un viseur : aussi l'icône de la barre des tâches."""
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(C["raised"]))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(2, 2, 60, 60), 15, 15)
    p.setBrush(QColor(ACCENT))
    p.drawEllipse(QPointF(32, 32), 13, 13)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(C["text"]), 3.5))
    p.drawEllipse(QPointF(32, 32), 21, 21)
    p.end()
    return QIcon(pm)


def apply_theme(app):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, C["bg"]), (QPalette.WindowText, C["text"]), (QPalette.Base, C["raised"]),
                        (QPalette.AlternateBase, C["panel"]), (QPalette.Text, C["text"]), (QPalette.Button, C["raised"]),
                        (QPalette.ButtonText, C["text"]), (QPalette.Highlight, C["accent"]),
                        (QPalette.HighlightedText, "#1a0d07"), (QPalette.ToolTipBase, C["raised"]),
                        (QPalette.ToolTipText, C["text"]), (QPalette.PlaceholderText, C["faint"])):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(THEME)
    app.setWindowIcon(app_icon())


def dark_titlebar(win):
    """Barre de titre Windows assortie à l'interface (Windows 11 ; sans effet ailleurs)."""
    try:
        import ctypes
        hwnd = int(win.winId())
        dark, caption = ctypes.c_int(1), ctypes.c_int(0x00101112)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark), 4)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(caption), 4)
    except Exception:  # noqa: BLE001
        pass


class WheelGuard(QObject):
    """La molette fait défiler le panneau au lieu de changer une liste survolée (sauf celle cliquée)."""

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel and isinstance(obj, QComboBox) and not obj.hasFocus():
            w = obj.parentWidget()
            while w is not None and not isinstance(w, QScrollArea):
                w = w.parentWidget()
            if w is not None:
                bar = w.verticalScrollBar()
                bar.setValue(bar.value() - ev.angleDelta().y())
            return True
        return False


def to_pixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QPixmap.fromImage(QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy())


def to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()


def rounded(pm: QPixmap, radius=8) -> QPixmap:
    out = QPixmap(pm.size())
    out.setDevicePixelRatio(pm.devicePixelRatio())
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, pm.width(), pm.height()), radius, radius)
    p.setClipPath(path)
    p.drawPixmap(0, 0, pm)
    p.end()
    return out


class Wrapped(QLabel):
    """Texte sur plusieurs lignes : annonce sa vraie hauteur à la largeur du panneau, sinon sa carte
    garde un vide dans la rangée d'au-dessus."""
    WIDTH = 350

    def __init__(self, text=""):
        super().__init__(text)
        self.setWordWrap(True)

    def sizeHint(self):
        return QSize(self.WIDTH, self.heightForWidth(self.WIDTH))

    def minimumSizeHint(self):
        return QSize(120, self.heightForWidth(self.WIDTH))


def note(text):
    lab = Wrapped(text)
    lab.setObjectName("note")
    return lab


def plain(text):
    """Sans accent ni majuscule : « netteté » se trouve en tapant « nettete »."""
    return "".join(c for c in unicodedata.normalize("NFD", str(text).lower()) if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------- commandes
class ValueBar(QWidget):
    """Curseur d'une seule barre : nom à gauche, valeur à droite, remplissage entre les deux. On glisse
    de n'importe où (Maj : réglage fin), sans saut au clic ; double-clic : valeur d'origine."""
    valueChanged = Signal(int)
    sliderPressed = Signal()
    sliderReleased = Signal()

    def __init__(self, text, lo, hi, value, fmt=str, default=None):
        super().__init__()
        self.text, self.lo, self.hi, self.fmt, self.default = text, lo, hi, fmt, default
        self._v = max(lo, min(hi, int(value)))
        self._drag = None
        self._k = 1.0
        self._hover = False
        self.menu_extra = None             # appelé avec le menu du clic droit
        self.setFixedHeight(30)
        self.setMinimumWidth(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.SizeHorCursor)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover)

    def value(self):
        return self._v

    def minimum(self):
        return self.lo

    def maximum(self):
        return self.hi

    def setValue(self, v):
        v = max(self.lo, min(self.hi, int(v)))
        if v != self._v:
            self._v = v
            self.update()
            self.valueChanged.emit(v)

    def reset(self):
        if self.default is not None:
            self.setValue(self.default)

    def _origin(self):
        """Le remplissage part du neutre : 0 ou la valeur d'origine pour les réglages à deux sens."""
        if self.lo < 0 < self.hi:
            return 0
        if self.default is not None and self.lo < self.default < self.hi and self.hi - self.lo > 1000:
            return self.default
        return self.lo

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.isEnabled():
            self._drag = (ev.position().x(), float(self._v))
            self.sliderPressed.emit()

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        x0, v0 = self._drag
        k = 0.2 if ev.modifiers() & Qt.ShiftModifier else 1.0
        if k != self._k:                                        # changer de vitesse ne fait pas sauter la valeur
            self._drag = (ev.position().x(), float(self._v))
            self._k = k
            return
        # Toute la plage demande presque 1,2 largeur de barre : précis, pas trop sensible.
        span = (self.hi - self.lo) / max(1.0, self.width() * 1.2)
        self.setValue(int(round(v0 + (ev.position().x() - x0) * span * k)))

    def mouseReleaseEvent(self, ev):
        if self._drag is not None:
            self._drag = None
            self.sliderReleased.emit()

    def mouseDoubleClickEvent(self, ev):
        self._drag = None
        self.reset()

    def keyPressEvent(self, ev):
        step = {Qt.Key_Left: -1, Qt.Key_Down: -1, Qt.Key_Right: 1, Qt.Key_Up: 1, Qt.Key_PageDown: -10, Qt.Key_PageUp: 10}
        if ev.key() in step:
            self.setValue(self._v + step[ev.key()] * max(1, (self.hi - self.lo) // 200))
        else:
            super().keyPressEvent(ev)

    def contextMenuEvent(self, ev):
        m = QMenu(self)
        if self.default is not None:
            m.addAction(f"Revenir à « {self.fmt(self.default)} »", self.reset)
        m.addAction("Saisir une valeur…", self._ask)
        if self.menu_extra:
            self.menu_extra(m)
        m.exec(ev.globalPos())

    def _ask(self):
        v, ok = QInputDialog.getInt(self, self.text, f"Valeur brute, de {self.lo} à {self.hi} :", self._v, self.lo, self.hi)
        if ok:
            self.setValue(v)

    def event(self, ev):
        if ev.type() in (QEvent.HoverEnter, QEvent.HoverLeave):
            self._hover = ev.type() == QEvent.HoverEnter
            self.update()
        return super().event(ev)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        on = self.isEnabled()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["hover"] if self._hover and on else C["line"]))
        p.drawRoundedRect(QRectF(0, 0, w, h), 7, 7)
        span = max(1, self.hi - self.lo)
        x0 = (self._origin() - self.lo) / span * w
        x1 = (self._v - self.lo) / span * w
        a, b = min(x0, x1), max(x0, x1)
        changed = self.default is not None and self._v != self.default
        if b - a > 0.5:
            clip = QPainterPath()
            clip.addRoundedRect(QRectF(0, 0, w, h), 7, 7)
            p.setClipPath(clip)
            p.setBrush(QColor(C["tint"] if on and changed else C["line2"]))
            p.drawRect(QRectF(a, 0, b - a, h))
            p.setBrush(QColor(ACCENT if on and changed else C["dim"]))
            p.drawRect(QRectF(x1 - 1, 0, 2, h))
            p.setClipping(False)
        p.setPen(QColor(C["text"] if on else C["faint"]))
        font = p.font()
        p.setFont(font)
        value = self.fmt(self._v)
        fm = p.fontMetrics()
        vw = fm.horizontalAdvance(value) + 12
        label = fm.elidedText(self.text, Qt.ElideRight, int(w - vw - 16))
        p.drawText(QRectF(10, 0, w - vw - 12, h), Qt.AlignVCenter | Qt.AlignLeft, label)
        p.setPen(QColor(C["accent2"] if on and changed else C["dim"] if on else C["faint"]))
        p.drawText(QRectF(0, 0, w - 10, h), Qt.AlignVCenter | Qt.AlignRight, value)


class Toggle(QCheckBox):
    """Interrupteur sur toute la rangée : même usage qu'une case à cocher."""

    def __init__(self, text=""):
        super().__init__(text)
        self.setFixedHeight(30)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.menu_extra = None

    def hitButton(self, pos):
        return self.rect().contains(pos)

    def sizeHint(self):
        return QSize(180, 30)

    def contextMenuEvent(self, ev):
        if self.menu_extra:
            m = QMenu(self)
            self.menu_extra(m)
            m.exec(ev.globalPos())

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        on, en = self.isChecked(), self.isEnabled()
        p.setPen(QColor(C["text"] if en else C["faint"]))
        fm = p.fontMetrics()
        p.drawText(QRectF(10, 0, w - 54, h), Qt.AlignVCenter | Qt.AlignLeft, fm.elidedText(self.text(), Qt.ElideRight, w - 58))
        tw, th = 32, 18
        x, y = w - tw - 8, (h - th) / 2
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACCENT if on and en else C["line2"] if en else C["line"]))
        p.drawRoundedRect(QRectF(x, y, tw, th), th / 2, th / 2)
        p.setBrush(QColor("#1a0d07" if on and en else C["text"] if en else C["faint"]))
        p.drawEllipse(QPointF(x + (tw - 9 if on else 9), y + 9), 6.5, 6.5)


class Segmented(QWidget):
    """Choix court toujours visible : même usage qu'une liste déroulante."""
    currentIndexChanged = Signal(int)
    activated = Signal(int)

    def __init__(self, items):
        super().__init__()
        self.items = list(items)
        self._i = 0
        self.setFixedHeight(30)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.menu_extra = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def count(self):
        return len(self.items)

    def itemText(self, i):
        return self.items[i]

    def currentIndex(self):
        return self._i

    def currentText(self):
        return self.items[self._i]

    def setCurrentIndex(self, i):
        i = max(0, min(len(self.items) - 1, int(i)))
        if i != self._i:
            self._i = i
            self.update()
            self.currentIndexChanged.emit(i)

    def mousePressEvent(self, ev):
        if self.isEnabled() and ev.button() == Qt.LeftButton:
            i = int(ev.position().x() / max(1, self.width()) * len(self.items))
            self.setCurrentIndex(i)
            self.activated.emit(self._i)

    def contextMenuEvent(self, ev):
        if self.menu_extra:
            m = QMenu(self)
            self.menu_extra(m)
            m.exec(ev.globalPos())

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, n = self.width(), self.height(), len(self.items)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["line"]))
        p.drawRoundedRect(QRectF(0, 0, w, h), 7, 7)
        cw = w / n
        en = self.isEnabled()
        p.setBrush(QColor(C["tint"] if en else C["hover"]))
        p.drawRoundedRect(QRectF(self._i * cw + 2, 2, cw - 4, h - 4), 6, 6)
        for i, t in enumerate(self.items):
            p.setPen(QColor((C["accent2"] if i == self._i else C["dim"]) if en else C["faint"]))
            p.drawText(QRectF(i * cw + 3, 0, cw - 6, h), Qt.AlignCenter, p.fontMetrics().elidedText(t, Qt.ElideRight, int(cw - 8)))


class Meter(QWidget):
    """Niveau sonore, de −60 à 0 dB."""

    def __init__(self, text):
        super().__init__()
        self.text, self.db = text, -60.0
        self.setFixedHeight(22)

    def setValue(self, db):
        db = max(-60.0, min(0.0, float(db)))
        if abs(db - self.db) > 0.4:
            self.db = db
            self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QColor(C["dim"]))
        p.drawText(QRectF(0, 0, 60, h), Qt.AlignVCenter, self.text)
        bar = QRectF(62, 8, w - 62 - 48, 6)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["line2"]))
        p.drawRoundedRect(bar, 3, 3)
        g = QLinearGradient(bar.left(), 0, bar.right(), 0)
        g.setColorAt(0, QColor(C["good"]))
        g.setColorAt(0.75, QColor(C["good"]))
        g.setColorAt(0.9, QColor(C["warn"]))
        g.setColorAt(1, QColor(C["bad"]))
        p.setBrush(g)
        p.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * (self.db + 60) / 60, bar.height()), 3, 3)
        p.setPen(QColor(C["dim"]))
        p.drawText(QRectF(0, 0, w, h), Qt.AlignVCenter | Qt.AlignRight, f"{self.db:.0f} dB" if self.db > -60 else "–")


class Sparkline(QWidget):
    """Barres de durée (ms) des étages du flux, ou courbe récente."""

    def __init__(self, n=120):
        super().__init__()
        self.values = [0.0] * n
        self.setFixedHeight(46)

    def push(self, v):
        self.values = self.values[1:] + [float(v)]
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        top = max(max(self.values), 60.0)
        path = QPainterPath()
        n = len(self.values)
        for i, v in enumerate(self.values):
            pt = QPointF(i / (n - 1) * w, h - 2 - v / top * (h - 6))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.setPen(QPen(QColor(ACCENT), 1.6))
        p.drawPath(path)


class Viewer(QWidget):
    """Zone d'image : proportions gardées, zoom à la molette, glisser pour se déplacer quand on a zoomé,
    double-clic pour revenir à l'image entière. En comparaison, le trait se déplace à la souris."""

    def __init__(self, text=""):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.text = text
        self.overlay = None                 # appelé avec le peintre, par-dessus l'image
        self._img, self._before = None, None
        self.split = 0.5
        self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5
        self.guides = False
        self._grab = None

    def show_image(self, img):
        self.set_qimage(to_qimage(img))

    def set_qimage(self, qimg, before=None):
        self._img, self._before = qimg, before
        self.update()

    def set_compare(self, before, after):
        self.set_qimage(to_qimage(after), to_qimage(before))

    def set_pixmap(self, pm):
        self.set_qimage(pm.toImage())

    def clear_image(self, text=""):
        self._img, self._before = None, None
        self.text = text
        self.update()

    def setText(self, text):
        self.text = text
        self.update()

    def has_image(self):
        return self._img is not None

    def fit_rect(self):
        """Rectangle de l'image dans le widget, zoom compris."""
        if self._img is None:
            return QRectF()
        iw, ih = self._img.width(), self._img.height()
        w, h = self.width(), self.height()
        s = min(w / iw, h / ih) * self.zoom
        dw, dh = iw * s, ih * s
        x = (w - dw) / 2 if dw <= w else max(w - dw, min(0.0, w / 2 - self.cx * dw))
        y = (h - dh) / 2 if dh <= h else max(h - dh, min(0.0, h / 2 - self.cy * dh))
        return QRectF(x, y, dw, dh)

    def wheelEvent(self, ev):
        if self._img is None:
            return
        r = self.fit_rect()
        pos = ev.position()
        fx, fy = (pos.x() - r.x()) / max(1.0, r.width()), (pos.y() - r.y()) / max(1.0, r.height())
        k = 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
        self.zoom = max(1.0, min(8.0, self.zoom * k))
        if self.zoom > 1.0:
            self.cx, self.cy = max(0.0, min(1.0, fx)), max(0.0, min(1.0, fy))
        self.update()

    def mousePressEvent(self, ev):
        self.setFocus()
        if ev.button() != Qt.LeftButton or self._img is None:
            return
        r = self.fit_rect()
        if self._before is not None and abs(ev.position().x() - (r.x() + r.width() * self.split)) < 14:
            self._grab = "split"
        elif self.zoom > 1:
            self._grab = (ev.position(), self.cx, self.cy)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        r = self.fit_rect()
        if self._grab == "split":
            self.split = max(0.0, min(1.0, (ev.position().x() - r.x()) / max(1.0, r.width())))
            self.update()
        elif self._grab is not None:
            p0, cx, cy = self._grab
            d = ev.position() - p0
            self.cx = max(0.0, min(1.0, cx - d.x() / max(1.0, r.width())))
            self.cy = max(0.0, min(1.0, cy - d.y() / max(1.0, r.height())))
            self.update()
        elif self._before is not None and abs(ev.position().x() - (r.x() + r.width() * self.split)) < 14:
            self.setCursor(Qt.SplitHCursor)
        else:
            self.setCursor(Qt.OpenHandCursor if self.zoom > 1 else Qt.ArrowCursor)

    def mouseReleaseEvent(self, _ev):
        self._grab = None

    def mouseDoubleClickEvent(self, _ev):
        self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a0a09"))
        if self._img is None:
            if self.text:
                p.setPen(QColor(C["faint"]))
                p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.text)
            if self.overlay:
                self.overlay(p)
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = self.fit_rect()
        p.drawImage(r, self._img)
        if self._before is not None:
            x = r.x() + r.width() * self.split
            p.save()
            p.setClipRect(QRectF(r.x(), r.y(), x - r.x(), r.height()))
            p.drawImage(r, self._before)
            p.restore()
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor("#ffffff"), 1.6))
            p.drawLine(QPointF(x, r.y()), QPointF(x, r.bottom()))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#ffffff"))
            cy = r.y() + r.height() / 2
            p.drawEllipse(QPointF(x, cy), 10, 10)
            p.setPen(QPen(QColor("#1a0d07"), 1.6))
            p.drawLine(QPointF(x - 3, cy - 4), QPointF(x - 6, cy))
            p.drawLine(QPointF(x - 6, cy), QPointF(x - 3, cy + 4))
            p.drawLine(QPointF(x + 3, cy - 4), QPointF(x + 6, cy))
            p.drawLine(QPointF(x + 6, cy), QPointF(x + 3, cy + 4))
            self._tag(p, r.x() + 10, r.y() + 10, "Avant")
            self._tag(p, r.right() - 62, r.y() + 10, "Après")
        if self.guides:
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setPen(QPen(QColor(255, 255, 255, 70), 1))
            for k in (1 / 3, 2 / 3):
                p.drawLine(QPointF(r.x() + r.width() * k, r.y()), QPointF(r.x() + r.width() * k, r.bottom()))
                p.drawLine(QPointF(r.x(), r.y() + r.height() * k), QPointF(r.right(), r.y() + r.height() * k))
        if self.zoom > 1:
            self._tag(p, 10, self.height() - 34, f"×{self.zoom:.1f}")
        if self.overlay:
            self.overlay(p)

    @staticmethod
    def _tag(p, x, y, text):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        w = p.fontMetrics().horizontalAdvance(text) + 16
        p.drawRoundedRect(QRectF(x, y, w, 24), 6, 6)
        p.setPen(QColor("#ffffff"))
        p.drawText(QRectF(x, y, w, 24), Qt.AlignCenter, text)


class Card(QFrame):
    """Groupe de réglages. S'emploie comme un formulaire : addRow(widget) ou addRow(libellé, widget)."""

    def __init__(self, title, folded_state=None, key=None, opened=True, foldable=False):
        super().__init__()
        self.setObjectName("card")
        self.title = title
        self.controls = []                  # réglages avec valeur d'origine : bouton Réinitialiser
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 9, 12, 12)
        outer.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.head = QToolButton() if foldable else QLabel()
        self.head.setObjectName("sectionHead" if foldable else "h2")
        self.head.setText(title)
        self.head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.head, 1)
        self.reset_btn = QToolButton()
        self.reset_btn.setObjectName("ghost")
        self.reset_btn.setText("Réinitialiser")
        self.reset_btn.setToolTip("Remet ce groupe à ses valeurs d'origine")
        self.reset_btn.clicked.connect(self.reset)
        self.reset_btn.hide()
        row.addWidget(self.reset_btn)
        self.head_row = row
        outer.addLayout(row)
        self.body = QWidget()
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(0, 2, 0, 0)
        self.lay.setSpacing(5)
        outer.addWidget(self.body)
        self.foldable = foldable
        k = key or title
        state = folded_state if folded_state is not None else {}
        if foldable:
            self.head.setCheckable(True)
            self.head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

            def show(on):
                self.body.setVisible(on)
                self.head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
                state[k] = not on
            self.head.toggled.connect(show)
            self.head.setChecked(not state.get(k, not opened))
            show(self.head.isChecked())

    def open(self):
        if self.foldable:
            self.head.setChecked(True)

    def reset(self):
        for w in self.controls:
            w.reset()

    def refresh(self):
        self.reset_btn.setVisible(any(w.is_changed() for w in self.controls))

    def addRow(self, a, b=None):
        if b is None:
            self._add(a)
            return
        row = QHBoxLayout()
        row.setSpacing(10)
        lab = QLabel(a) if isinstance(a, str) else a
        lab.setMinimumWidth(110)
        lab.setContentsMargins(10, 0, 0, 0)
        row.addWidget(lab)
        if isinstance(b, QWidget):
            row.addWidget(b, 1)
        else:
            row.addLayout(b, 1)
        self.lay.addLayout(row)

    def _add(self, a):
        if isinstance(a, QWidget):
            self.lay.addWidget(a)
        else:
            self.lay.addLayout(a)

    addWidget = _add
    addLayout = _add


class Page(QWidget):
    """Page du panneau : titre, préréglages, petits onglets. Une seule série de réglages à la fois."""

    def __init__(self, title, subtitle=""):
        super().__init__()
        self.setObjectName("page")
        self.title = title
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 0)
        root.setSpacing(8)
        head = QHBoxLayout()
        lab = QLabel(title)
        lab.setObjectName("h1")
        head.addWidget(lab)
        head.addStretch()
        self.tools = head
        root.addLayout(head)
        if subtitle:
            root.addWidget(note(subtitle))
        self.extra = QVBoxLayout()
        self.extra.setSpacing(6)
        root.addLayout(self.extra)
        self.chips_host = QWidget()
        self.chips = QHBoxLayout(self.chips_host)
        self.chips.setContentsMargins(0, 0, 0, 2)
        self.chips.setSpacing(2)
        self.chips.addStretch()
        self.chips_host.hide()
        root.addWidget(self.chips_host)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.idClicked.connect(self.stack.setCurrentIndex)
        self.tabs = {}

    def tab(self, name="Réglages"):
        if name in self.tabs:
            return self.tabs[name]
        inner = QWidget()
        inner.setObjectName("tabBody")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 6, 6, 16)
        lay.setSpacing(10)
        lay.addStretch()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setWidget(inner)
        i = self.stack.addWidget(sc)
        chip = QToolButton()
        chip.setObjectName("chip")
        chip.setText(name)
        chip.setCheckable(True)
        chip.setCursor(Qt.PointingHandCursor)
        chip.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        bold = QFont(chip.font())
        bold.setPointSizeF(8.8)
        bold.setBold(True)
        chip.setMinimumWidth(QFontMetrics(bold).horizontalAdvance(name) + 20)
        self.group.addButton(chip, i)
        self.chips.insertWidget(self.chips.count() - 1, chip)
        if i == 0:
            chip.setChecked(True)
        self.chips_host.setVisible(i > 0)
        self.tabs[name] = lay
        return lay

    def show_tab(self, name):
        lay = self.tabs.get(name)
        if lay is not None:
            i = list(self.tabs).index(name)
            self.group.button(i).setChecked(True)
            self.stack.setCurrentIndex(i)

    def scroll_to(self, w):
        sc = self.stack.currentWidget()
        if isinstance(sc, QScrollArea):
            sc.ensureWidgetVisible(w, 10, 60)

    def add_tool(self, name, tip, fn):
        b = QPushButton()
        b.setObjectName("icon")
        b.setIcon(icon(name, 16))
        b.setIconSize(QSize(16, 16))
        b.setToolTip(tip)
        b.clicked.connect(fn)
        self.tools.addWidget(b)
        return b


class Rail(QWidget):
    """Colonne d'icônes et pages du panneau : une catégorie à la fois."""
    changed = Signal(int)

    def __init__(self, width=380):
        super().__init__()
        self.setObjectName("inspector")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.rail = QWidget()
        self.rail.setObjectName("rail")
        self.rail.setFixedWidth(74)
        self.rail_lay = QVBoxLayout(self.rail)
        self.rail_lay.setContentsMargins(7, 10, 7, 10)
        self.rail_lay.setSpacing(3)
        self.rail_lay.addStretch()
        self.stack = QStackedWidget()
        self.stack.setObjectName("panelEdge")
        self.stack.setFixedWidth(width)
        row.addWidget(self.rail)
        row.addWidget(self.stack, 1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.idClicked.connect(self._pick)
        self.pages = []

    def _pick(self, i):
        self.stack.show()
        self.stack.setCurrentIndex(i)
        self.changed.emit(i)

    def add(self, name, text, page, bottom=False):
        i = self.stack.addWidget(page)
        self.pages.append(page)
        b = QToolButton()
        b.setObjectName("nav")
        b.setIcon(icon(name, 21, C["accent"], C["faint"]))
        b.setIconSize(QSize(21, 21))
        b.setText(text)
        b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        b.setCheckable(True)
        b.setFixedWidth(60)
        b.setCursor(Qt.PointingHandCursor)
        self.group.addButton(b, i)
        if bottom:
            self.rail_lay.addWidget(b)
        else:
            self.rail_lay.insertWidget(self.rail_lay.count() - 1, b)
        if i == 0:
            b.setChecked(True)
        return i

    def add_gap(self):
        gap = QFrame()
        gap.setObjectName("sep")
        self.rail_lay.insertSpacing(self.rail_lay.count() - 1, 4)
        self.rail_lay.insertWidget(self.rail_lay.count() - 1, gap)
        self.rail_lay.insertSpacing(self.rail_lay.count() - 1, 4)

    def select(self, i):
        if 0 <= i < self.stack.count():
            self.group.button(i).setChecked(True)
            self.stack.show()
            self.stack.setCurrentIndex(i)

    def current(self):
        return self.stack.currentIndex()

    def toggle_panel(self):
        self.stack.setVisible(not self.stack.isVisible())


# ---------------------------------------------------------------- réglages : registre commun
class Setting:
    """Un réglage de l'interface : de quoi le retrouver, le remettre, l'annuler, l'épingler."""
    __slots__ = ("id", "text", "tab", "page", "card", "space", "kind", "widget", "last", "extra", "rail", "index", "tip")

    def get(self):
        k = self.kind
        return self.widget.value() if k == "slider" else self.widget.isChecked() if k == "check" else self.widget.currentIndex()

    def put(self, v):
        k = self.kind
        (self.widget.setValue if k == "slider" else self.widget.setChecked if k == "check" else self.widget.setCurrentIndex)(v)

    def where(self):
        parts = [self.space, self.page.title]
        if self.tab and self.tab != "Réglages":
            parts.append(self.tab)
        return " › ".join(parts)


class Registry(QObject):
    """Tous les réglages : recherche, annuler / rétablir (Ctrl+Z, Ctrl+Y), favoris."""
    changed = Signal()

    def __init__(self, settings):
        super().__init__()
        self.items = {}
        self.favorites = settings.setdefault("favorites", [])
        self.done, self.undone = [], []
        self.busy = False
        self._t = 0.0

    def add(self, s):
        s.last = s.get()
        while s.id in self.items:
            s.id += "+"
        self.items[s.id] = s
        w = s.widget
        sig = w.valueChanged if s.kind == "slider" else w.toggled if s.kind == "check" else w.currentIndexChanged
        sig.connect(lambda _v=None: self._changed(s))
        extra = lambda m: self.menu(m, s)
        if hasattr(w, "menu_extra"):
            w.menu_extra = extra
        else:
            w.setContextMenuPolicy(Qt.CustomContextMenu)

            def popup(pos):
                menu = QMenu(w)
                extra(menu)
                menu.exec(w.mapToGlobal(pos))
            w.customContextMenuRequested.connect(popup)

    def menu(self, m, s):
        if m.actions():
            m.addSeparator()
        on = s.id in self.favorites
        m.addAction("Retirer des favoris" if on else "Ajouter aux favoris", lambda: self.pin(s, not on))

    def pin(self, s, on):
        if on and s.id not in self.favorites:
            self.favorites.append(s.id)
        elif not on and s.id in self.favorites:
            self.favorites.remove(s.id)
        self.changed.emit()

    def _changed(self, s):
        old, s.last = s.last, s.get()
        if self.busy or old == s.last:
            return
        now = time.perf_counter()
        if self.done and now - self._t < 0.45:      # un préréglage ou un même geste : une seule étape
            self.done[-1].setdefault(s, old)
        else:
            self.done.append({s: old})
            del self.done[:-200]
        self._t = now
        self.undone.clear()

    def _swap(self, src, dst):
        if not src:
            return None
        step = src.pop()
        back = {}
        self.busy = True
        for s, v in step.items():
            back[s] = s.get()
            s.put(v)
            s.last = v
        self.busy = False
        dst.append(back)
        self._t = 0.0
        return ", ".join(s.text for s in list(step)[:2]) + ("…" if len(step) > 2 else "")

    def undo(self):
        return self._swap(self.done, self.undone)

    def redo(self):
        return self._swap(self.undone, self.done)

    def search(self, text, limit=40):
        words = plain(text).split()
        if not words:
            return []
        out = []
        for s in self.items.values():
            hay = plain(f"{s.text} {s.where()} {s.card.title if s.card else ''} {s.tip or ''}")
            if all(w in hay for w in words):
                rank = 0 if plain(s.text).startswith(words[0]) else 1 if words[0] in plain(s.text) else 2
                out.append((rank, s.where(), s))
        out.sort(key=lambda r: (r[0], r[1]))
        return [r[2] for r in out[:limit]]

    def jump(self, s):
        """Montre le réglage et le fait clignoter."""
        s.rail.select(s.index)
        s.page.show_tab(s.tab)
        if s.card is not None:
            s.card.open()
        QTimer.singleShot(40, lambda: (s.page.scroll_to(s.widget), flash(s.widget)))


def flash(w):
    fx = QGraphicsOpacityEffect(w)
    w.setGraphicsEffect(fx)
    a = QPropertyAnimation(fx, b"opacity", w)
    a.setDuration(900)
    a.setKeyValueAt(0, 0.2)
    a.setKeyValueAt(0.25, 1)
    a.setKeyValueAt(0.5, 0.2)
    a.setKeyValueAt(0.75, 1)
    a.setKeyValueAt(1, 1)
    a.finished.connect(lambda: w.setGraphicsEffect(None))
    a.start(QPropertyAnimation.DeleteWhenStopped)


class StatusLabel(Wrapped):
    """Ligne d'état : ne prend pas de place quand elle est vide."""

    def __init__(self):
        super().__init__()
        self.hide()

    def setText(self, text):
        if text != self.text():
            super().setText(text)
            self.setVisible(bool(text))


class Form:
    """Remplit une page. Tout réglage déclaré avec une clé peut être relu et réécrit (préréglages) par
    le panneau passé à la création ; tout réglage entre dans la recherche, l'annulation et les favoris."""

    def __init__(self, page, folded=None, panel=None, registry=None, space="", rail=None, index=None):
        self.page, self.folded = page, folded if folded is not None else {}
        self.panel = panel if panel is not None else {}
        self.registry, self.space, self.rail, self.index = registry, space, rail, index
        self.tab_name = None
        self.form = None
        self.pending = []

    def tab(self, name):
        self.tab_name = name
        self.page.tab(name)
        self.form = None
        return self

    def section(self, title, open_=True, key=None, foldable=None):
        """open_=False : carte repliée, pour ce qui sert rarement."""
        if self.tab_name is None:
            self.tab("Réglages")
        lay = self.page.tab(self.tab_name)
        card = Card(title, self.folded, key, open_, foldable if foldable is not None else not open_)
        lay.insertWidget(lay.count() - 1, card)
        self.form = card
        return card

    def track(self, key, get, put):
        if key:
            self.panel[key] = (get, put)

    def _note_setting(self, widget, text, kind, tip=None):
        if self.registry is None:
            return
        s = Setting()
        s.id = f"{self.space}/{self.page.title}/{self.tab_name}/{text}"
        s.text, s.tab, s.page, s.card, s.space = text, self.tab_name, self.page, self.form, self.space
        s.kind, s.widget, s.tip, s.rail, s.index = kind, widget, tip, self.rail, self.index
        self.pending.append(s)

    def bind(self, rail, index):
        """Fin de la page : son numéro dans le rail est connu, ses réglages entrent au registre."""
        for s in self.pending:
            s.rail, s.index = rail, index
            self.registry.add(s)
        self.pending = []

    def _resettable(self, widget, default, changed):
        card = self.form
        widget.reset_value = default
        widget.is_changed = changed
        card.controls.append(widget)
        return card

    def slider(self, text, lo, hi, value, on_change, fmt=str, default=None, key=None, tip=None):
        s = ValueBar(text, lo, hi, value, fmt, default)
        if tip:
            s.setToolTip(tip)
        s.valueChanged.connect(on_change)
        self.form.addRow(s)
        self.track(key, s.value, s.setValue)
        self._note_setting(s, text, "slider", tip)
        if default is not None:
            card = self._resettable(s, default, lambda: s.value() != default)
            s.valueChanged.connect(lambda _v: card.refresh())
            card.refresh()
        return s

    def check(self, text, value, on_change, key=None, tip=None, default=None):
        c = Toggle(text)
        c.setChecked(bool(value))
        c.toggled.connect(on_change)
        if tip:
            c.setToolTip(tip)
        self.form.addRow(c)
        self.track(key, c.isChecked, c.setChecked)
        self._note_setting(c, text, "check", tip)
        if default is not None:
            c.reset = lambda: c.setChecked(default)
            card = self._resettable(c, default, lambda: c.isChecked() != default)
            c.toggled.connect(lambda _v: card.refresh())
            card.refresh()
        return c

    def combo(self, text, items, index, on_change, key=None, tip=None, default=None, name=None):
        items = list(items)
        short = len(items) <= 3 and sum(len(t) for t in items) <= 36 and not text
        short = short or (len(items) <= 3 and sum(len(t) for t in items) <= 22)
        c = Segmented(items) if short else QComboBox()
        if not short:
            c.addItems(items)
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(8)
        c.setCurrentIndex(max(0, index))
        c.currentIndexChanged.connect(on_change)
        if tip:
            c.setToolTip(tip)
        if text:
            self.form.addRow(text, c)
        else:
            self.form.addRow(c)
        self.track(key, c.currentIndex, c.setCurrentIndex)
        self._note_setting(c, text or name or (items[0] if items else "Choix"), "combo", tip)
        if default is not None:
            c.reset = lambda: c.setCurrentIndex(default)
            card = self._resettable(c, default, lambda: c.currentIndex() != default)
            c.currentIndexChanged.connect(lambda _v: card.refresh())
            card.refresh()
        return c

    def button(self, text, fn, primary=False):
        b = QPushButton(text)
        if primary:
            b.setObjectName("primary")
        b.clicked.connect(fn)
        self.form.addRow(b)
        return b

    def note(self, text):
        lab = note(text)
        self.form.addRow(lab)
        return lab

    def label(self):
        lab = StatusLabel()
        lab.setObjectName("status")
        self.form.addRow(lab)
        return lab


class PresetBar(QWidget):
    """Préréglages fournis et les tiens (★). Enregistrer complète un préréglage déjà là : il garde les
    réglages propres à l'autre espace."""
    MARK = "★ "

    def __init__(self, panel, mine, builtin=None, parent_win=None, after=None, models_only=False):
        super().__init__()
        self.panel, self.mine, self.builtin, self.win, self.after = panel, mine, builtin or {}, parent_win, after
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(8)
        row.addWidget(self.combo, 1)
        self.save_btn = QPushButton()
        self.save_btn.setObjectName("icon")
        self.save_btn.setIcon(icon("save", 16))
        self.save_btn.setToolTip("Enregistrer ces réglages comme préréglage")
        self.del_btn = QPushButton()
        self.del_btn.setObjectName("icon")
        self.del_btn.setIcon(icon("trash", 16))
        self.del_btn.setToolTip("Supprimer ce préréglage")
        row.addWidget(self.save_btn)
        row.addWidget(self.del_btn)
        self.combo.activated.connect(self.chosen)
        self.save_btn.clicked.connect(self.save)
        self.del_btn.clicked.connect(self.delete)
        self.refill()
        self.save_btn.setVisible(not models_only)
        self.del_btn.setVisible(not models_only)

    def refill(self, select=None):
        c = self.combo
        c.blockSignals(True)
        c.clear()
        c.addItem("Préréglages…")
        c.addItems(list(self.builtin))
        if self.builtin and self.mine:
            c.insertSeparator(c.count())
        c.addItems([self.MARK + n for n in self.mine])
        if select:
            c.setCurrentText(select)
        c.blockSignals(False)
        self.del_btn.setEnabled(c.currentText().startswith(self.MARK))

    def chosen(self, _i):
        name = self.combo.currentText()
        self.del_btn.setEnabled(name.startswith(self.MARK))
        if name in self.builtin:
            self.builtin[name]()
        elif name.startswith(self.MARK):
            for key, val in self.mine.get(name[len(self.MARK):], {}).items():
                if key in self.panel:
                    self.panel[key][1](val)

    def save(self):
        cur = self.combo.currentText()
        name, ok = QInputDialog.getText(self.win or self, "Enregistrer le préréglage", "Nom :",
                                        text=cur[len(self.MARK):] if cur.startswith(self.MARK) else "")
        name = name.strip()
        if not ok or not name:
            return
        if name in self.mine and QMessageBox.question(self.win or self, "Préréglage", f"Remplacer « {name} » ?") != QMessageBox.Yes:
            return
        self.mine[name] = dict(self.mine.get(name, {}), **{k: g() for k, (g, _p) in self.panel.items()})
        self.refill(self.MARK + name)
        if self.after:
            self.after()

    def delete(self):
        name = self.combo.currentText()
        if name.startswith(self.MARK) and QMessageBox.question(
                self.win or self, "Préréglage", f"Supprimer « {name[len(self.MARK):]} » ?") == QMessageBox.Yes:
            self.mine.pop(name[len(self.MARK):], None)
            self.refill()
            if self.after:
                self.after()


class FavoritesPage(Page):
    """Tes réglages épinglés, d'un clic droit : copies liées aux vrais réglages, dans les deux sens."""

    def __init__(self, registry, space):
        super().__init__("Favoris", "Clic droit sur un réglage : « Ajouter aux favoris ». Il arrive ici.")
        self.registry, self.space = registry, space
        self.lay = self.tab()
        self.links = []
        self.empty = note("Aucun favori pour l'instant.")
        self.lay.insertWidget(0, self.empty)
        registry.changed.connect(self.rebuild)

    def rebuild(self):
        for sig, fn in self.links:
            try:
                sig.disconnect(fn)
            except (RuntimeError, TypeError):
                pass
        self.links = []
        while self.lay.count() > 2:
            it = self.lay.takeAt(1)
            if it.widget():
                it.widget().deleteLater()
        items = [self.registry.items[k] for k in self.registry.favorites
                 if k in self.registry.items and self.registry.items[k].space == self.space]
        self.empty.setVisible(not items)
        for s in items:
            self.lay.insertWidget(self.lay.count() - 1, self._row(s))

    def _row(self, s):
        w = s.widget
        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        if s.kind == "slider":
            m = ValueBar(s.text, w.lo, w.hi, w.value(), w.fmt, w.default)
            m.valueChanged.connect(w.setValue)
            self.links.append((w.valueChanged, m.setValue))
            w.valueChanged.connect(m.setValue)
        elif s.kind == "check":
            m = Toggle(s.text)
            m.setChecked(w.isChecked())
            m.toggled.connect(w.setChecked)
            self.links.append((w.toggled, m.setChecked))
            w.toggled.connect(m.setChecked)
        else:
            names = [w.itemText(i) for i in range(w.count())]
            m = QComboBox()
            m.addItems(names)
            m.setCurrentIndex(w.currentIndex())
            m.currentIndexChanged.connect(w.setCurrentIndex)
            self.links.append((w.currentIndexChanged, m.setCurrentIndex))
            w.currentIndexChanged.connect(m.setCurrentIndex)
        m.setToolTip(s.tip or "")
        col.addWidget(m)
        row = QHBoxLayout()
        row.setSpacing(4)
        where = note(s.where())
        row.addWidget(where, 1)
        go = QToolButton()
        go.setObjectName("ghost")
        go.setText("Aller")
        go.clicked.connect(lambda: self.registry.jump(s))
        off = QToolButton()
        off.setObjectName("ghost")
        off.setText("Retirer")
        off.clicked.connect(lambda: self.registry.pin(s, False))
        row.addWidget(go)
        row.addWidget(off)
        col.addLayout(row)
        return host


class Palette(QFrame):
    """Recherche (Ctrl+K) : tous les réglages des deux espaces et toutes les actions."""

    def __init__(self, win, registry, actions):
        super().__init__(win)
        self.setObjectName("popup")
        self.win, self.registry, self.actions = win, registry, actions
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Chercher un réglage ou une action…")
        self.edit.setClearButtonEnabled(True)
        lay.addWidget(self.edit)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(self.list, 1)
        self.edit.textChanged.connect(self.fill)
        self.edit.returnPressed.connect(self.run)
        self.edit.installEventFilter(self)
        self.list.itemActivated.connect(lambda _it: self.run())
        self.list.itemClicked.connect(lambda _it: self.run())
        self.hide()

    def open(self):
        w = min(620, self.win.width() - 80)
        self.setGeometry((self.win.width() - w) // 2, 70, w, min(460, self.win.height() - 120))
        self.edit.clear()
        self.fill("")
        self.show()
        self.raise_()
        self.edit.setFocus()

    def fill(self, text):
        self.list.clear()
        q = plain(text)
        for name, keys, fn in self.actions():
            if all(w in plain(name) for w in q.split()):
                it = QListWidgetItem(f"{name}" + (f"      {keys}" if keys else ""))
                it.setData(Qt.UserRole, fn)
                it.setIcon(icon("wand", 15))
                self.list.addItem(it)
        for s in self.registry.search(text):
            it = QListWidgetItem(f"{s.text}      {s.where()}")
            it.setData(Qt.UserRole, s)
            it.setIcon(icon("track", 15))
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def run(self):
        it = self.list.currentItem()
        self.hide()
        if it is None:
            return
        target = it.data(Qt.UserRole)
        if isinstance(target, Setting):
            self.registry.jump(target)
        else:
            target()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress:
            if ev.key() == Qt.Key_Escape:
                self.hide()
                return True
            if ev.key() in (Qt.Key_Down, Qt.Key_Up):
                self.list.setCurrentRow(max(0, min(self.list.count() - 1, self.list.currentRow() + (1 if ev.key() == Qt.Key_Down else -1))))
                return True
        return False


class Toasts:
    """Message bref en bas de la fenêtre, sans fenêtre à fermer."""

    def __init__(self, win):
        self.win = win
        self.lab = QLabel(win)
        self.lab.setObjectName("toast")
        self.lab.setStyleSheet(f"background:{C['raised']}; border:1px solid {C['line2']}; border-radius:10px; padding:9px 16px; font-weight:600;")
        self.lab.setAlignment(Qt.AlignCenter)
        self.lab.hide()
        self.fx = QGraphicsOpacityEffect(self.lab)
        self.lab.setGraphicsEffect(self.fx)
        self.anim = QPropertyAnimation(self.fx, b"opacity")
        self.anim.setDuration(350)
        self.anim.setEasingCurve(QEasingCurve.InQuad)
        self.anim.finished.connect(lambda: self.anim.endValue() == 0 and self.lab.hide())
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._out)

    def show(self, text, seconds=2.6):
        self.lab.setText(text)
        self.lab.adjustSize()
        self.lab.move((self.win.width() - self.lab.width()) // 2, self.win.height() - self.lab.height() - 110)
        self.anim.stop()
        self.fx.setOpacity(1)
        self.lab.show()
        self.lab.raise_()
        self.timer.start(int(seconds * 1000))

    def _out(self):
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.start()


def face_tile(path, size=64):
    """Vignette ronde d'un visage récent."""
    import cv2
    try:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = max(0, (h - s) // 3), (w - s) // 2
    return rounded(to_pixmap(cv2.resize(img[y:y + s, x:x + s], (size * 2, size * 2), interpolation=cv2.INTER_AREA))
                   .scaled(size, size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation), 10)


def fix_file_dialogs():
    """Première ouverture d'une fenêtre : le dossier Vidéos ou Images plutôt que le dossier du logiciel."""
    from PySide6.QtWidgets import QFileDialog
    from home_fix import REAL_HOME

    def wrap(fn):
        def call(parent=None, caption="", directory="", filter="", *args, **kw):
            if not directory:
                sub = "Videos" if "*.mp4" in filter else "Pictures" if "*.png" in filter else ""
                folder = os.path.join(REAL_HOME, sub)
                directory = folder if os.path.isdir(folder) else REAL_HOME
            return fn(parent, caption, directory, filter, *args, **kw)
        return call
    for name in ("getOpenFileName", "getOpenFileNames", "getSaveFileName"):
        setattr(QFileDialog, name, staticmethod(wrap(getattr(QFileDialog, name))))
