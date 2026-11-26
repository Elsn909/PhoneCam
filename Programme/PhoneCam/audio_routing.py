"""Choix du cable virtuel et nom du micro a selectionner dans les autres apps."""
import re


def virtual_microphone(output):
    match = re.match(r"^(CABLE(?:-[A-D])?) Input\b", output or "", re.IGNORECASE)
    return f"{match.group(1)} Output" if match else ""


def preferred_output(outputs, saved=""):
    """Conserve un cable choisi ; prefere un cable a une ancienne sortie casque."""
    if saved in outputs and virtual_microphone(saved):
        return saved
    for prefix in ("CABLE-B Input", "CABLE Input", "CABLE-A Input", "CABLE-C Input", "CABLE-D Input"):
        found = next((name for name in outputs if name.startswith(prefix)), "")
        if found:
            return found
    return saved if saved in outputs else ""
