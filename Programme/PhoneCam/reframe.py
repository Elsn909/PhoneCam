"""Studio : format de sortie (16:9, 9:16…), recadrage, zoom, position, cadre qui suit le visage."""
import cv2
import numpy as np

FORMATS = {"Format d'origine": None, "16:9 (paysage)": 16 / 9, "9:16 (vertical)": 9 / 16, "1:1 (carré)": 1.0,
           "4:5 (portrait)": 4 / 5}
FITS = ["Remplir (recadrer)", "Ajuster (bandes noires)", "Ajuster (fond flouté)"]
DEFAULTS = {"format": 0, "fit": 0, "zoom": 100, "x": 0, "y": 0, "follow": True, "follow_smooth": 50}


def even(x):
    return max(2, int(round(x / 2)) * 2)


def output_size(p, w, h):
    """Le petit côté de la sortie reste celui de la vidéo : 1920 × 1080 en 9:16 donne 1080 × 1920."""
    aspect = list(FORMATS.values())[int(p["format"])]
    if aspect is None:
        return w, h
    short = min(w, h)
    return (even(short * aspect), even(short)) if aspect >= 1 else (even(short), even(short / aspect))


def is_active(p, w, h):
    return output_size(p, w, h) != (w, h) or p["zoom"] != 100 or p["x"] or p["y"]


def _gaussian(values, sigma):
    r = max(1, int(sigma * 3))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
    return np.convolve(np.pad(values, r, mode="edge"), k / k.sum(), mode="valid")


def follow_path(a, smooth):
    """Centre du cadre pour chaque image : la personne la plus présente, lissée sur chaque plan dans les deux
    sens du temps (aucun retard), jamais à travers un changement de plan. None si personne n'est suivi."""
    cached = getattr(a, "_path", None)
    if cached is not None and cached[0] == smooth:
        return cached[1]
    tracks = [t for t in a.tracks if a.persons and t.person == a.persons[0]["id"]] or a.tracks
    if not tracks:
        return None
    n = max(1, a.frames)
    xy = np.full((n, 2), np.nan)
    for t in tracks:
        box = np.stack(t.bbox)
        xy[t.frames[0]:t.frames[-1] + 1] = (box[:, :2] + box[:, 2:4]) / 2
    sigma = a.info["fps"] * (0.15 + 1.5 * smooth / 100)
    edges = [0] + list(a.cuts) + [n]
    for s, e in zip(edges, edges[1:]):
        seg = xy[s:e]
        ok = ~np.isnan(seg[:, 0])
        if not ok.any():
            xy[s:e] = np.nan
            continue
        idx = np.arange(e - s)
        xy[s:e] = np.stack([_gaussian(np.interp(idx, idx[ok], seg[ok, c]), sigma) for c in (0, 1)], 1)
    a._path = (smooth, xy)
    return xy


def window(p, w, h, a=None, i=0):
    """Mode Remplir : rectangle (x0, y0, largeur, hauteur) de l'image gardé dans la sortie."""
    W, H = output_size(p, w, h)
    aspect = W / H
    cw = min(float(w), h * aspect) * 100 / p["zoom"]
    ch = cw / aspect
    cx, cy = w / 2, h / 2
    path = follow_path(a, p["follow_smooth"]) if a is not None and p["follow"] else None
    if path is not None and 0 <= i < len(path) and not np.isnan(path[i, 0]):
        cx, cy = float(path[i, 0]), float(path[i, 1])
    cx += p["x"] / 100 * (w - cw) / 2
    cy += p["y"] / 100 * (h - ch) / 2
    return min(max(cx - cw / 2, 0.0), w - cw), min(max(cy - ch / 2, 0.0), h - ch), cw, ch


def _place(img, s, tx, ty, W, H, border, base=None):
    if s < 0.75:                       # forte réduction : moyenne des pixels, pas de crénelage
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        tx, ty, s = tx, ty, 1.0
    M = np.array([[s, 0, tx], [0, s, ty]], np.float32)
    if base is None:
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=border)
    return cv2.warpAffine(img, M, (W, H), base, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)


def apply(img, p, a=None, i=0):
    h, w = img.shape[:2]
    if not is_active(p, w, h):
        return img
    W, H = output_size(p, w, h)
    if int(p["fit"]) == 0:
        x0, y0, cw, _ch = window(p, w, h, a, i)
        s = W / cw
        return _place(img, s, -x0 * s, -y0 * s, W, H, cv2.BORDER_REPLICATE)
    s = min(W / w, H / h) * p["zoom"] / 100
    tx = (W - w * s) / 2 + p["x"] / 100 * W / 2
    ty = (H - h * s) / 2 + p["y"] / 100 * H / 2
    if int(p["fit"]) == 1:
        back = np.zeros((H, W, 3), np.uint8)
    else:                              # fond : la même image agrandie et floutée
        k = max(W / w, H / h)
        small = cv2.resize(img, (even(w * k / 8), even(h * k / 8)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 7)
        big = cv2.resize(small, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
        y, x = (big.shape[0] - H) // 2, (big.shape[1] - W) // 2
        back = (big[y:y + H, x:x + W] * 0.65).astype(np.uint8)
    return _place(img, s, tx, ty, W, H, cv2.BORDER_CONSTANT, back)
