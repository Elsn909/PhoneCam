"""Connexion par câble USB via ADB (« adb reverse ») : le téléphone joint le PC sur localhost."""
import os
import subprocess

import home_fix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADB = os.path.join(ROOT, "tools", "platform-tools", "adb.exe")
_NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW : pas de console qui clignote


def _adb(*args: str, timeout: float = 15) -> tuple[int, str]:
    try:
        r = subprocess.run([ADB, *args], capture_output=True, text=True, timeout=timeout, env=home_fix.data_env(),
                           creationflags=_NO_WINDOW)
        return r.returncode, (r.stdout + r.stderr).strip()
    except FileNotFoundError:
        return 1, "adb.exe introuvable (dossier tools\\platform-tools)."
    except subprocess.TimeoutExpired:
        return 1, "ADB ne répond pas."


def available() -> bool:
    return os.path.isfile(ADB)


APP_PACKAGE = "com.phonecam.app"


def connect(port: int, url: str, app_url: str = "") -> str:
    """Relie le port du téléphone au PC et ouvre l'appli PhoneCam (sinon la page web). Renvoie un message."""
    _adb("start-server", timeout=30)
    _, out = _adb("devices")
    devices = [line.split("\t") for line in out.splitlines()[1:] if "\t" in line]
    if not devices:
        return ("Aucun téléphone détecté. Branche le câble et active le débogage USB "
                "(Options pour les développeurs).")
    serial, state = devices[0]
    if state == "unauthorized":
        return "Accepte « Autoriser le débogage USB ? » sur le téléphone, puis réessaie."
    if state != "device":
        return f"Téléphone dans l'état « {state} »."
    code, out = _adb("-s", serial, "reverse", f"tcp:{port}", f"tcp:{port}")
    if code != 0:
        return f"Échec du relais USB : {out}"
    _, model = _adb("-s", serial, "shell", "getprop", "ro.product.model")
    _, pkg = _adb("-s", serial, "shell", "pm", "path", APP_PACKAGE)
    if app_url and pkg.startswith("package:"):
        # L'appli reçoit le lien d'appairage et démarre l'envoi (guillemets : le « & » passe au shell).
        _adb("-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", f"'{app_url}'")
        return f"Relié par câble à {model or serial}. L'appli PhoneCam s'est ouverte sur le téléphone."
    # Chrome de préférence (WebCodecs/H.264 matériel), sinon le navigateur par défaut.
    code, out = _adb("-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW",
                     "-n", "com.android.chrome/com.google.android.apps.chrome.Main", "-d", url)
    if code != 0 or "Error" in out:
        _adb("-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url)
    return f"Relié par câble à {model or serial}. La page s'est ouverte sur le téléphone."


def disconnect() -> None:
    """Retire le relais et arrête le serveur ADB (aucun processus ne reste en arrière-plan)."""
    if available():
        _adb("reverse", "--remove-all", timeout=5)
        _adb("kill-server", timeout=5)
