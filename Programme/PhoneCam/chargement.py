"""Écran de chargement de PhoneCam, dans son propre processus.

Le chargement des modèles d'IA (sessions ONNX Runtime, TensorRT) bloque tout le processus Python
pendant qu'il a lieu : une fenêtre dans le même processus resterait figée. Ici l'écran tourne à
part et reçoit l'avancement par son entrée standard, une ligne « pourcentage<TAB>étape<TAB>détail ».
Il se ferme sur « fin », ou tout seul si PhoneCam s'arrête (entrée standard fermée).

Splash (côté PhoneCam) n'importe que la bibliothèque standard : il se lance avant les gros imports.
"""
import os
import subprocess
import sys


class Splash:
    """Côté PhoneCam : lance l'écran et lui envoie l'avancement. Ne fait jamais échouer le démarrage."""

    def __init__(self):
        self._last = None
        try:
            self._p = subprocess.Popen([sys.executable, os.path.abspath(__file__)], stdin=subprocess.PIPE,
                                       text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            self._p = None

    def step(self, pct: int, text: str, detail: str = "") -> None:
        line = f"{int(pct)}\t{text}\t{' '.join(str(detail).split())}"
        if self._p is None or line == self._last:
            return
        self._last = line
        try:
            self._p.stdin.write(line + "\n")
            self._p.stdin.flush()
        except (OSError, ValueError):
            self._p = None

    def close(self) -> None:
        if self._p is None:
            return
        try:
            self._p.stdin.write("fin\n")
            self._p.stdin.close()
        except (OSError, ValueError):
            pass
        self._p = None


def run() -> None:
    import queue
    import threading
    import time

    from PySide6.QtCore import QRectF, Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
    from PySide6.QtWidgets import QApplication, QWidget

    from ui_kit import C

    inbox = queue.Queue()
    sys.stdin.reconfigure(encoding="utf-8")     # sinon la page de code de Windows : accents abîmés

    def read():
        for line in sys.stdin:
            inbox.put(line.rstrip("\n"))
        inbox.put("fin")                    # PhoneCam fermé ou arrêté

    threading.Thread(target=read, daemon=True).start()

    class Screen(QWidget):
        def __init__(self):
            super().__init__(None, Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setWindowTitle("PhoneCam Studio : chargement")
            self.setFixedSize(500, 210)
            self.target, self.shown, self.text, self.detail = 3.0, 0.0, "Démarrage…", ""
            self.t0 = time.perf_counter()
            self.closing = None
            tm = QTimer(self)
            tm.timeout.connect(self.frame)
            tm.start(16)

        def frame(self):
            while True:
                try:
                    msg = inbox.get_nowait()
                except queue.Empty:
                    break
                if msg == "fin":
                    self.target, self.text, self.detail = 100, "Prêt", ""
                    self.closing = self.closing or time.perf_counter()
                    continue
                parts = (msg.split("\t") + ["", ""])[:3]
                try:
                    self.target = max(self.target, float(parts[0]))
                except ValueError:
                    continue
                self.text, self.detail = parts[1], parts[2]
            # Avance en douceur vers la cible ; sans nouvelle, continue un peu (jamais au-delà).
            self.shown += (self.target - self.shown) * 0.12
            if self.closing:
                fade = 1 - (time.perf_counter() - self.closing) / 0.25
                if fade <= 0:
                    QApplication.quit()
                    return
                self.setWindowOpacity(fade)
            self.update()

        def paintEvent(self, ev):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            path = QPainterPath()
            path.addRoundedRect(r, 14, 14)
            p.fillPath(path, QColor(C["panel"]))
            p.setPen(QPen(QColor(C["line2"]), 1))
            p.drawPath(path)
            display = QFont("Bahnschrift", 15)
            display.setWeight(QFont.DemiBold)
            p.setFont(display)
            p.setPen(QColor(C["accent"]))
            p.drawText(QRectF(32, 30, 30, 32), Qt.AlignVCenter, "●")
            p.setPen(QColor(C["text"]))
            p.drawText(QRectF(56, 30, 300, 32), Qt.AlignVCenter, "PhoneCam Studio")
            p.setFont(QFont("Segoe UI", 10))
            p.setPen(QColor(C["text"]))
            p.drawText(QRectF(32, 92, 360, 24), Qt.AlignVCenter, self.text)
            p.setPen(QColor(C["dim"]))
            p.drawText(QRectF(392, 92, 76, 24), Qt.AlignVCenter | Qt.AlignRight, f"{round(self.shown)} %")
            # Barre : fond, partie faite, reflet qui passe (montre que tout tourne même sans nouvelle).
            bar = QRectF(32, 124, 436, 8)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(C["raised"]))
            p.drawRoundedRect(bar, 4, 4)
            done = QRectF(bar.x(), bar.y(), max(8.0, bar.width() * self.shown / 100), bar.height())
            p.setBrush(QColor(C["accent"]))
            p.drawRoundedRect(done, 4, 4)
            phase = ((time.perf_counter() - self.t0) % 1.6) / 1.6
            x = done.x() - 80 + (done.width() + 160) * phase
            g = QLinearGradient(x - 60, 0, x + 60, 0)
            g.setColorAt(0, QColor(255, 255, 255, 0))
            g.setColorAt(0.5, QColor(255, 255, 255, 90))
            g.setColorAt(1, QColor(255, 255, 255, 0))
            p.setBrush(g)
            p.drawRoundedRect(done, 4, 4)
            p.setFont(QFont("Segoe UI", 9))
            p.setPen(QColor(C["faint"]))
            detail = p.fontMetrics().elidedText(self.detail, Qt.ElideRight, int(bar.width()))
            p.drawText(QRectF(32, 144, 436, 22), Qt.AlignVCenter, detail)
            secs = int(time.perf_counter() - self.t0)
            p.drawText(QRectF(32, 170, 436, 20), Qt.AlignVCenter | Qt.AlignRight, f"{secs} s")

    app = QApplication(sys.argv)
    w = Screen()
    w.show()
    app.exec()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run()
