"""Tour de passage sur la carte graphique : la voix d'abord, la vidéo ensuite.

Mesuré (RTX 5080, voix + amélioration de visage GPEN en continu) : chaque opération de la voix
attend derrière les gros calculs de la vidéo, même sur un flux CUDA prioritaire (la carte ne coupe
pas un calcul en cours). Un bloc de voix passait de 20 ms à 100–200 ms pour 200 ms de son : à la
moindre pointe, un bloc arrive en retard et le son se coupe (petits à-coups réguliers, ou plusieurs
secondes de silence le temps de rattraper).
Ici, pendant les ~20 ms où la voix calcule (toutes les 200 ms), la vidéo ne lance pas d'étape
nouvelle : elle attend, au plus 60 ms. Résultat mesuré : bloc de voix 100 → 54 ms, et la vidéo
perd ~5 % de ses images en usage réel.
"""
import threading

_free = threading.Event()
_free.set()
MAX_WAIT = 0.06           # secondes : la vidéo n'attend jamais plus que cela


class voice_turn:
    """with voice_turn(): … — le calcul de la voix (la vidéo patiente pendant ce temps)."""

    def __enter__(self):
        _free.clear()
        return self

    def __exit__(self, *a):
        _free.set()
        return False


def video_turn():
    """À appeler avant chaque étape lourde de la vidéo (détection, échange, amélioration)."""
    if not _free.is_set():
        _free.wait(MAX_WAIT)
