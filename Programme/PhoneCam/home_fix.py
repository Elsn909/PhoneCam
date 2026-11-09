"""À importer avant tout le reste. Le lanceur redirige USERPROFILE vers data\\home pour garder modèles et
caches dans le dossier du logiciel, mais Windows en déduit aussi le Bureau, les Images, les
Téléchargements… et le mémorise au démarrage du programme : les fenêtres d'ouverture cherchaient
alors « data\\home\\Desktop » (« Emplacement non disponible »).

USERPROFILE reprend donc sa vraie valeur pour Windows, et « ~ » reste data\\home pour les bibliothèques.
"""
import ctypes
import ntpath
import os
from ctypes import wintypes

DATA_HOME = os.environ.get("USERPROFILE", "")
REAL_HOME = DATA_HOME


def _real_profile():
    token = wintypes.HANDLE()
    adv, kernel = ctypes.windll.advapi32, ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    if not adv.OpenProcessToken(wintypes.HANDLE(kernel.GetCurrentProcess()), 8, ctypes.byref(token)):
        return None
    size = wintypes.DWORD(1024)
    buf = ctypes.create_unicode_buffer(1024)
    ok = ctypes.windll.userenv.GetUserProfileDirectoryW(token, buf, ctypes.byref(size))
    kernel.CloseHandle(token)
    return buf.value if ok else None


def _install():
    global REAL_HOME
    real = _real_profile()
    if not real or not DATA_HOME or os.path.normcase(real) == os.path.normcase(DATA_HOME):
        return
    REAL_HOME = real
    expand = ntpath.expanduser

    def expanduser(path):
        text = os.fspath(path)
        if text == "~" or text[:2] in ("~/", "~\\"):
            return DATA_HOME + text[1:]
        return expand(path)
    ntpath.expanduser = os.path.expanduser = expanduser
    os.environ["USERPROFILE"] = real


if os.name == "nt":
    _install()


def data_env():
    """Environnement d'un programme qui range ses fichiers dans ~ (ADB et ses clés)."""
    return dict(os.environ, USERPROFILE=DATA_HOME)
