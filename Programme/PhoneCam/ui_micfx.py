"""Onglet Effets de la page Micro : égaliseur visuel, de-esser, compresseur, chaleur, styles enregistrables."""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

import effets_micro as FX
from ui_kit import C, PresetBar, Segmented, ValueBar

F_LO, F_HI, G_MAX = 20.0, 20000.0, 18.0
FREQS = np.geomspace(F_LO, F_HI, 240)
SPEC_F = np.fft.rfftfreq(256, 1 / 48000)


def hz(f):
    return f"{f / 1000:.1f} kHz".replace(".0 ", " ") if f >= 1000 else f"{f:.0f} Hz"


class EqCurve(QWidget):
    """Courbe de l'égaliseur : glisser un point (fréquence, gain), molette = largeur (Q), double-clic = 0 dB."""
    changed = Signal(int)
    picked = Signal(int)

    def __init__(self, p, fx):
        super().__init__()
        self.p, self.fx = p, fx
        self.sel = 3
        self.drag = None
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Glisse un point pour régler la bande · molette : plus large / plus étroit · double-clic : remettre à 0")

    # --- coordonnées
    def _area(self):
        return QRectF(8, 8, self.width() - 16, self.height() - 26)

    def _x(self, f):
        a = self._area()
        return a.left() + a.width() * math.log(f / F_LO) / math.log(F_HI / F_LO)

    def _f(self, x):
        a = self._area()
        return F_LO * (F_HI / F_LO) ** min(1, max(0, (x - a.left()) / a.width()))

    def _y(self, g):
        a = self._area()
        return a.center().y() - a.height() / 2 * max(-1.0, min(1.0, g / G_MAX))

    def _g(self, y):
        a = self._area()
        return max(-G_MAX, min(G_MAX, (a.center().y() - y) / (a.height() / 2) * G_MAX))

    def _node(self, i):
        kind = FX.BANDS[i][0]
        g = 0 if kind in ("hp", "lp") else self.p[f"eq{i}_g"]
        return QPointF(self._x(self.p[f"eq{i}_f"]), self._y(g))

    def _active(self, i):
        kind = FX.BANDS[i][0]
        return self.p["hp_on"] if kind == "hp" else self.p["lp_on"] if kind == "lp" else True

    # --- souris
    def _hit(self, pos):
        best, bd = None, 14.0
        for i in range(len(FX.BANDS)):
            d = (self._node(i) - pos).manhattanLength()
            if d < bd:
                best, bd = i, d
        return best

    def mousePressEvent(self, ev):
        i = self._hit(ev.position())
        if i is not None:
            self.sel = i
            self.drag = i
            self.picked.emit(i)
            self.update()

    def mouseMoveEvent(self, ev):
        if self.drag is None:
            self.setCursor(Qt.SizeAllCursor if self._hit(ev.position()) is not None else Qt.PointingHandCursor)
            return
        i, pos = self.drag, ev.position()
        f = self._f(pos.x())
        if FX.BANDS[i][0] == "hp":
            f = min(f, 600)
        elif FX.BANDS[i][0] == "lp":
            f = max(f, 2000)
        self.p[f"eq{i}_f"] = int(round(f))
        if FX.BANDS[i][0] not in ("hp", "lp"):
            self.p[f"eq{i}_g"] = int(round(self._g(pos.y())))
        self.changed.emit(i)
        self.update()

    def mouseReleaseEvent(self, _ev):
        self.drag = None

    def mouseDoubleClickEvent(self, ev):
        i = self._hit(ev.position())
        if i is not None and FX.BANDS[i][0] not in ("hp", "lp"):
            self.p[f"eq{i}_g"] = 0
            self.changed.emit(i)
            self.update()

    def wheelEvent(self, ev):
        i = self._hit(ev.position())
        i = self.sel if i is None else i
        if FX.BANDS[i][0] in ("hp", "lp"):
            return
        q = self.p[f"eq{i}_q"] * (1.12 if ev.angleDelta().y() < 0 else 1 / 1.12)
        self.p[f"eq{i}_q"] = int(round(min(1000, max(20, q))))
        self.sel = i
        self.changed.emit(i)
        self.picked.emit(i)
        self.update()
        ev.accept()

    # --- dessin
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a = self._area()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["bg"]))
        p.drawRoundedRect(QRectF(self.rect()), 8, 8)
        small = QFont(self.font())
        small.setPointSizeF(7.5)
        p.setFont(small)
        # grille
        for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
            x = self._x(f)
            p.setPen(QPen(QColor(C["line"]), 1))
            p.drawLine(QPointF(x, a.top()), QPointF(x, a.bottom()))
            p.setPen(QColor(C["faint"]))
            p.drawText(QRectF(x - 24, a.bottom() + 2, 48, 14), Qt.AlignCenter, hz(f).replace(" Hz", "").replace(" kHz", "k"))
        for g in (-12, -6, 0, 6, 12):
            y = self._y(g)
            p.setPen(QPen(QColor(C["line2"] if g == 0 else C["line"]), 1))
            p.drawLine(QPointF(a.left(), y), QPointF(a.right(), y))
            if g:
                p.setPen(QColor(C["faint"]))
                p.drawText(QRectF(a.left() + 2, y - 13, 30, 12), Qt.AlignLeft, f"{g:+d}")
        # spectre de la voix (sortie des effets)
        spec = self.fx.spec
        if spec.max() > -150:
            path = QPainterPath()
            top = max(-20.0, float(spec.max()))
            pts = [(self._x(f), a.bottom() - a.height() * 0.9 * max(0.0, (s - top + 70) / 70))
                   for f, s in zip(SPEC_F[1:], spec[1:]) if F_LO <= f <= F_HI]
            if pts:
                path.moveTo(pts[0][0], a.bottom())
                for x, y in pts:
                    path.lineTo(x, y)
                path.lineTo(pts[-1][0], a.bottom())
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(255, 255, 255, 18))
                p.drawPath(path)
        # courbe de l'égaliseur
        resp = FX.response(self.p, FREQS) if self.p["eq_on"] else np.zeros(len(FREQS))
        path = QPainterPath()
        for k, (f, g) in enumerate(zip(FREQS, resp)):
            pt = QPointF(self._x(f), self._y(g))
            path.moveTo(pt) if k == 0 else path.lineTo(pt)
        fill = QPainterPath(path)
        fill.lineTo(self._x(F_HI), self._y(0))
        fill.lineTo(self._x(F_LO), self._y(0))
        grad = QLinearGradient(0, a.top(), 0, a.bottom())
        acc = QColor(C["accent"])
        acc.setAlpha(55)
        grad.setColorAt(0, acc)
        acc.setAlpha(10)
        grad.setColorAt(1, acc)
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawPath(fill)
        p.setPen(QPen(QColor(C["accent"] if self.p["eq_on"] else C["faint"]), 2))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        # points
        for i in range(len(FX.BANDS)):
            c = self._node(i)
            on = self._active(i) and self.p["eq_on"]
            r = 7 if i == self.sel else 5.5
            p.setPen(QPen(QColor(C["text"] if i == self.sel else C["bg"]), 1.5))
            p.setBrush(QColor(C["accent"] if on else C["line2"]))
            p.drawEllipse(c, r, r)
            p.setPen(QColor("#1a0d07" if on else C["dim"]))
            p.drawText(QRectF(c.x() - 6, c.y() - 6, 12, 12), Qt.AlignCenter, str(i) if 0 < i < 7 else "")
        name = FX.BAND_NAMES[self.sel]
        p.setPen(QColor(C["dim"]))
        p.drawText(QRectF(a.right() - 220, a.top() + 2, 216, 14), Qt.AlignRight, name)


class FxTab:
    def __init__(self, live, f, mic):
        self.live, self.mic, self.p = live, mic, mic.p
        s = live.settings
        p = self.p
        f.tab("Effets")
        card = f.section("Égaliseur")
        self.fx_panel = {}

        styles = {n: (lambda n=n: self.apply_style(FX.PRESETS[n])) for n in FX.PRESETS}
        self.styles = PresetBar(self.fx_panel, s.setdefault("fx_styles", {}), styles, live.window(), after=live.persist_cb)
        self.styles.combo.setItemText(0, "Styles…")
        card.addRow(self.styles)
        self.on = f.check("Effets de voix", p["fx_on"], lambda v: self._set("fx_on", bool(v)), key="fx_on", default=True,
                          tip="Égaliseur, de-esser, compresseur et chaleur, avant la simulation du téléphone.")
        self.curve = EqCurve(p, mic.fx)
        self.curve.changed.connect(lambda i: (self._band_to_ui(i), self.live.persist_cb()))
        self.curve.picked.connect(self._band_to_ui)
        card.addRow(self.curve)
        self.anim = QTimer(self.curve)          # spectre en direct, seulement quand la courbe est visible
        self.anim.timeout.connect(lambda: self.curve.isVisible() and self.curve.update())
        self.anim.start(70)
        row = QHBoxLayout()
        self.band = Segmented(["1", "2", "3", "4", "5", "6"])
        self.band.setToolTip("Bande à régler (ou clique un point de la courbe)")
        self.band.currentIndexChanged.connect(lambda k: self._pick(k + 1))
        row.addWidget(QLabel("Bande"))
        row.addWidget(self.band, 1)
        card.addRow(row)
        self.f_bar = ValueBar("Fréquence", 0, 1000, 500, lambda v: hz(self._log_f(v)))
        self.g_bar = ValueBar("Gain", -18, 18, 0, lambda v: f"{v:+d} dB" if v else "0 dB", 0)
        self.q_bar = ValueBar("Largeur (Q)", 20, 1000, 100, lambda v: f"{v / 100:.2f}", 100)
        self.f_bar.valueChanged.connect(lambda v: self._band_set("f", int(round(self._log_f(v)))))
        self.g_bar.valueChanged.connect(lambda v: self._band_set("g", v))
        self.q_bar.valueChanged.connect(lambda v: self._band_set("q", v))
        for b in (self.f_bar, self.g_bar, self.q_bar):
            card.addRow(b)
        self.hp = f.check("Coupe-bas", p["hp_on"], lambda v: self._set("hp_on", bool(v)), key="hp_on", default=True,
                          tip="Retire les graves inutiles (souffle, vibrations du bureau). Point de gauche sur la courbe.")
        self.lp = f.check("Coupe-haut", p["lp_on"], lambda v: self._set("lp_on", bool(v)), key="lp_on", default=False,
                          tip="Retire le haut du spectre (son plus « petit » ou plus doux). Point de droite sur la courbe.")
        self.eq = f.check("Égaliseur actif", p["eq_on"], lambda v: self._set("eq_on", bool(v)), key="eq_on", default=True)
        for key, c in (("fx_on", self.on), ("hp_on", self.hp), ("lp_on", self.lp), ("eq_on", self.eq)):
            self.fx_panel[key] = (c.isChecked, c.setChecked)
        for k in [k for k in FX.DEFAULTS if k.startswith("eq") and k[2].isdigit()]:
            f.track(k, lambda k=k: self.p[k], lambda v, k=k: self._put(k, v))
            self.fx_panel[k] = (lambda k=k: self.p[k], lambda v, k=k: self._put(k, v))
        self._pick(3)

        f.section("Sifflantes (de-esser)")
        self._check(f, "deess_on", "Adoucir les « s » et « ch »", "Baisse seulement les sifflantes trop fortes, à la volée.")
        self._sl(f, "deess_freq", "À partir de", 3000, 11000, lambda v: hz(v))
        self._sl(f, "deess_thr", "Seuil", -60, -6, lambda v: f"{v} dB")
        self._sl(f, "deess_amt", "Intensité", 0, 100, lambda v: f"{v} %")

        f.section("Compresseur")
        self._check(f, "comp_on", "Compresser la voix", "Rapproche les passages doux et forts : une voix plus régulière, plus « produite ».")
        self._sl(f, "comp_thr", "Seuil", -50, 0, lambda v: f"{v} dB")
        self._sl(f, "comp_ratio", "Taux", 10, 100, lambda v: f"{v / 10:.1f} : 1")
        self._sl(f, "comp_att", "Attaque", 1, 60, lambda v: f"{v} ms")
        self._sl(f, "comp_rel", "Relâchement", 20, 600, lambda v: f"{v} ms")
        self._sl(f, "comp_makeup", "Gain de compensation", 0, 18, lambda v: f"+{v} dB")
        self.gr_label = f.label()

        f.section("Couleur")
        self._sl(f, "warm", "Chaleur (saturation douce)", 0, 100, lambda v: f"{v} %",
                 "Ajoute des harmoniques comme un préampli à lampe : voix plus ronde et présente.")
        f.note("Les styles (★) s'enregistrent avec la disquette en haut ; ils entrent aussi dans les presets (branche Micro).")

    # --- outils
    def _check(self, f, key, text, tip):
        c = f.check(text, self.p[key], lambda v, k=key: self._set(k, bool(v)), key=key, default=FX.DEFAULTS[key], tip=tip)
        self.fx_panel[key] = (c.isChecked, c.setChecked)
        return c

    def _sl(self, f, key, text, lo, hi, fmt, tip=None):
        s = f.slider(text, lo, hi, int(self.p[key]), lambda v, k=key: self._set(k, v), fmt, int(FX.DEFAULTS[key]), key, tip)
        self.fx_panel[key] = (s.value, s.setValue)
        return s

    def _set(self, key, value):
        self.p[key] = value
        self.curve.update()

    def _put(self, key, value):
        self.p[key] = value
        self.curve.update()
        self._band_to_ui(self.curve.sel)

    @staticmethod
    def _log_f(v):
        return F_LO * (F_HI / F_LO) ** (v / 1000)

    @staticmethod
    def _log_v(f):
        return int(round(1000 * math.log(max(F_LO, f) / F_LO) / math.log(F_HI / F_LO)))

    def _pick(self, i):
        self.curve.sel = i
        self.curve.update()
        self._band_to_ui(i)

    def _band_to_ui(self, i):
        p = self.p
        self._sync = True
        kind = FX.BANDS[i][0]
        for b in (self.f_bar, self.g_bar, self.q_bar):
            b.blockSignals(True)
        self.f_bar.setValue(self._log_v(p[f"eq{i}_f"]))
        self.g_bar.setValue(p[f"eq{i}_g"])
        self.q_bar.setValue(p[f"eq{i}_q"])
        for b in (self.f_bar, self.g_bar, self.q_bar):
            b.blockSignals(False)
        self.g_bar.setEnabled(kind not in ("hp", "lp"))
        self.q_bar.setEnabled(kind not in ("hp", "lp"))
        self.g_bar.text = "Gain" if kind not in ("hp", "lp") else "Gain (coupure : sans objet)"
        self.f_bar.text = f"Fréquence · {FX.BAND_NAMES[i]}"
        if 1 <= i <= 6 and self.band.currentIndex() != i - 1:
            self.band.blockSignals(True)
            self.band.setCurrentIndex(i - 1)
            self.band.blockSignals(False)
        for b in (self.f_bar, self.g_bar, self.q_bar):
            b.update()

    def _band_set(self, what, v):
        i = self.curve.sel
        self.p[f"eq{i}_{what}"] = v
        self.curve.update()

    def apply_style(self, values):
        vals = dict(FX.DEFAULTS, **values)
        for k, v in vals.items():
            if k in self.live.panels["micro"]:
                self.live.panels["micro"][k][1](v)
            else:
                self.p[k] = v
        self.curve.update()
        self._band_to_ui(self.curve.sel)

    def tick(self):
        if self.p["comp_on"] or self.p["deess_on"]:
            fx = self.mic.fx
            self.gr_label.setText(f"Réduction : compresseur {fx.gr:.1f} dB · sifflantes {fx.de_gr:.1f} dB")
        else:
            self.gr_label.setText("")
