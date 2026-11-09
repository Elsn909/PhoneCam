"""Réglages du changeur de voix (sans dépendance lourde : lisible avant le chargement de PyTorch)."""
import os

DEFAULTS = {
    "enabled": False,       # voix transformée (sinon : micro tel quel)
    "pth": "",
    "index": "",
    # voix
    "pitch": 0,             # demi-tons
    "formant": 0,           # timbre, dixièmes de demi-ton (-40…40)
    "index_rate": 50,       # % d'index
    "protect": 33,          # protection des consonnes (50 = coupée)
    "follow": 40,           # % : le volume de sortie suit celui de ta voix
    "gain": 0,              # dB
    # voix vivante (ton bas relevé)
    "lift_on": True,
    "lift_floor": 5,        # dixièmes de demi-ton sous ta hauteur habituelle avant d'agir
    "lift_amount": 100,     # % de l'écart rattrapé
    "lift_ref": 120,        # ta hauteur normale (Hz) : en dessous, la voix est relevée
    "lift_auto": False,     # l'apprendre toute seule en continu (sinon : le curseur, ou « Mesurer »)
    "lively": 35,           # % d'expressivité ajoutée (mélodie élargie)
    "lively_always": False, # expressivité pleine même quand le ton n'est pas bas
    "lift_bright": 50,      # % de clarté ajoutée quand la voix est relevée
    # Smart
    "smart": True,
    "denoise": True,
    "denoise_db": 10,       # atténuation maximale du nettoyage (dB)
    "sensitivity": 35,      # % : seuil de détection de voix (voix douce : assez bas)
    "release_ms": 300,      # maintien après la dernière syllabe (occlusions p, t, k du français)
    "hold_voiced": True,    # une voyelle tenue (« aaaah ») garde la porte ouverte
    "breath": True,         # souffles, respirations, chuchotements : ils passent DANS les phrases
    "whisper": False,       # chuchotements seuls (sans voix autour) : ils passent aussi (moins strict)
    "preroll_ms": 50,       # ouverture avant le mot (≤ fondu + recherche)
    "floor_db": -62,        # en dessous : silence quoi qu'il arrive
    "voicing": 12,          # % : confiance minimale de la hauteur
    # performance
    "block": 20,            # centièmes de seconde
    "extra": 25,            # dixièmes de seconde de contexte
    "crossfade": 4,         # centièmes de seconde
    "margin_ms": 40,        # réserve contre les à-coups de la carte graphique
}
# Changent la taille des tampons : le moteur se reconfigure (un court silence).
RESTART_KEYS = ("block", "extra", "crossfade", "denoise", "smart", "margin_ms")


def find_index(pth):
    """L'index qui va avec un .pth : même dossier, nom le plus proche (added_… de préférence)."""
    folder = os.path.dirname(pth)
    try:
        cands = [f for f in os.listdir(folder) if f.lower().endswith(".index") and not f.lower().startswith("trained_")]
    except OSError:
        return ""
    if not cands:
        return ""
    stem = os.path.splitext(os.path.basename(pth))[0].lower().split("_e")[0]
    cands.sort(key=lambda f: (stem not in f.lower(), not f.lower().startswith("added"), f))
    return os.path.join(folder, cands[0])
