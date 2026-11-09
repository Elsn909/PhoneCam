"""Accueil : ce que fait chaque espace et par où commencer. Montré au premier lancement, puis avec le bouton « ? »."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ui_kit import C, Toggle, icon

SPACES = (
    ("live", "Direct", "En direct : visio, stream, appels.",
     [("Connexion", "Branche le téléphone (ou une webcam)."),
      ("Visage", "Choisis la photo du visage à appliquer."),
      ("Caméra virtuelle", "Démarre-la (bouton en bas) : les autres logiciels la voient comme une webcam."),
      ("Micro et Voix", "Démarre le micro ; la voix transformée se règle dans la page Voix.")]),
    ("film", "Studio", "Après coup : retoucher des fichiers, puis exporter.",
     [("Médias", "Importe des vidéos ou des images (ou glisse-les dans la fenêtre)."),
      ("Changer le visage ou Effets seuls", "Le sélecteur en bas : le visage remplacé, ou seulement des effets par-dessus."),
      ("Voix", "Un fichier audio ou ta voix enregistrée, transformé avec un modèle de voix."),
      ("Exporter", "Le fichier d'origine n'est jamais modifié.")]),
)
TIPS = ("Ctrl+K : chercher n'importe quel réglage par son nom.",
        "Sur un curseur : double-clic pour revenir à la valeur d'origine, clic droit pour saisir une valeur ou l'ajouter aux favoris.",
        "Ctrl+Z annule le dernier réglage. Les voyants en haut à droite ouvrent la page qui va avec.",
        "Survole un bouton ou une icône de la colonne de gauche : une bulle dit à quoi il sert.")


class Welcome(QDialog):
    def __init__(self, parent, settings, go, shortcuts):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Bienvenue dans PhoneCam")
        self.setObjectName("app")
        self.setMinimumWidth(820)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(12)
        title = QLabel("Bienvenue dans PhoneCam")
        title.setObjectName("h1")
        lay.addWidget(title)
        sub = QLabel("Deux espaces, en haut à gauche. Voici par où commencer dans chacun.")
        sub.setObjectName("note")
        lay.addWidget(sub)
        row = QHBoxLayout()
        row.setSpacing(12)
        for i, (ic, name, what, steps) in enumerate(SPACES):
            card = QFrame()
            card.setObjectName("card")
            c = QVBoxLayout(card)
            c.setContentsMargins(14, 12, 14, 14)
            c.setSpacing(8)
            head = QHBoxLayout()
            ico = QLabel()
            ico.setPixmap(icon(ic, 20, C["accent"]).pixmap(20, 20))
            head.addWidget(ico)
            h = QLabel(name)
            h.setObjectName("h2")
            head.addWidget(h, 1)
            c.addLayout(head)
            w = QLabel(what)
            w.setWordWrap(True)
            c.addWidget(w)
            for n, (step, text) in enumerate(steps, 1):
                line = QLabel(f"<b>{n}. {step}</b><br><span style='color:{C['dim']}'>{text}</span>")
                line.setWordWrap(True)
                line.setTextFormat(Qt.RichText)
                c.addWidget(line)
            c.addStretch()
            btn = QPushButton(f"Ouvrir le {name}")
            if i == 0:
                btn.setObjectName("primary")
            btn.clicked.connect(lambda _c=False, i=i: (go(i), self.accept()))
            c.addWidget(btn)
            row.addWidget(card, 1)
        lay.addLayout(row)
        tips = QLabel("<br>".join("• " + t for t in TIPS))
        tips.setObjectName("note")
        tips.setWordWrap(True)
        tips.setTextFormat(Qt.RichText)
        lay.addWidget(tips)
        bottom = QHBoxLayout()
        self.again = Toggle("Montrer cet accueil au démarrage")
        self.again.setChecked(not settings.get("welcome_seen", False))
        bottom.addWidget(self.again)
        bottom.addStretch()
        keys = QPushButton("Raccourcis clavier")
        keys.clicked.connect(shortcuts)
        bottom.addWidget(keys)
        close = QPushButton("Fermer")
        close.clicked.connect(self.accept)
        bottom.addWidget(close)
        lay.addLayout(bottom)

    def done(self, r):
        self.settings["welcome_seen"] = not self.again.isChecked()
        super().done(r)
