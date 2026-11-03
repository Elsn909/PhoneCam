"""Étalonnage du Studio : couleurs et ambiance à pleine résolution, sur la carte graphique.

Toutes les retouches de couleur sont cuites dans une table 3D (33³) recalculée seulement quand un
réglage change, puis appliquée en quelques ms même en 4K. Les effets qui dépendent de l'image
(netteté, lueur, vignettage, grain) viennent ensuite. Une table .cube importée s'insère dans la chaîne.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from studio_conf import LOOK_DEFAULTS as DEFAULTS, LOOK_PRESETS as PRESETS  # noqa: F401

LUT_SIZE = 33


def read_cube(path):
    """Table .cube → tableau (N, N, N, 3) indexé [b, g, r]."""
    size, rows = None, []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
            elif line[0].isdigit() or line[0] in "-.":
                parts = line.split()
                if len(parts) >= 3:
                    rows.append([float(x) for x in parts[:3]])
    if size is None or len(rows) < size ** 3:
        raise ValueError("table .cube 3D invalide")
    return np.array(rows[:size ** 3], np.float32).reshape(size, size, size, 3)


def _hue_rgb(h):
    h = (h % 360) / 60
    x = 1 - abs(h % 2 - 1)
    i = int(h) % 6
    return np.array([(1, x, 0), (x, 1, 0), (0, 1, x), (0, x, 1), (x, 0, 1), (1, 0, x)][i], np.float32)


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def grade(rgb, p, cube=None):
    """Couleurs : rgb (…, 3) entre 0 et 1, valeurs d'affichage (gamma 2,2)."""
    x = np.clip(rgb, 0, 1).astype(np.float32) ** 2.2             # lumière linéaire
    x = x * 2.0 ** (p["exposure"] / 50)
    t, m = p["temperature"] / 100, p["tint"] / 100
    x = x * np.array([1 + 0.35 * t + 0.1 * m, 1 - 0.2 * m, 1 - 0.35 * t + 0.1 * m], np.float32)
    x = np.clip(x, 0, None) ** (1 / 2.2)                         # le reste en valeurs perçues
    y = (x * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)
    # Hautes lumières, ombres, blancs, noirs : déplacent la luminosité sans toucher au reste.
    low = (1 - _smooth(0.0, 0.5, y)) ** 2
    high = _smooth(0.5, 1.0, y) ** 2
    d = (p["shadows"] / 100 * 0.3 * low + p["highlights"] / 100 * 0.3 * high
         + p["blacks"] / 100 * 0.15 * (1 - _smooth(0.0, 0.25, y)) + p["whites"] / 100 * 0.15 * _smooth(0.75, 1.0, y))
    x = x * (y + d).clip(0, None) / np.maximum(y, 1e-4)
    c = p["contrast"] / 100
    if c:                                                        # courbe en S autour du gris moyen
        k = 1 + 2.4 * abs(c)
        s = 0.5 + np.tanh((np.clip(x, 0, 1.2) - 0.5) * k) / (2 * np.tanh(0.5 * k))
        x = x + (s - x) * min(1.0, abs(c) * 2) if c > 0 else 0.5 + (x - 0.5) * (1 + c * 0.7)
    y = np.clip((x * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True), 0, 1)
    to = p["teal_orange"] / 100
    if to:
        warm = _smooth(0.3, 0.7, y)
        tone = np.array([0.0, 0.42, 0.55], np.float32) * (1 - warm) + np.array([0.95, 0.55, 0.28], np.float32) * warm
        x = x + (tone - 0.5) * 2 * 0.16 * to * (0.35 + 0.65 * (1 - np.abs(y * 2 - 1)))
    for hue, amount, w in ((p["shadow_hue"], p["shadow_tint"], (1 - _smooth(0.05, 0.5, y))),
                           (p["mid_hue"], p["mid_tint"], 1 - np.abs(y * 2 - 1)),
                           (p["high_hue"], p["high_tint"], _smooth(0.5, 0.95, y))):
        if amount:
            col = _hue_rgb(hue)
            x = x + (col - col.mean()) * 0.28 * (amount / 100) * w
    y = (x * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)
    sat = 1 + p["saturation"] / 100
    v = p["vibrance"] / 100
    if v:                                       # surtout les couleurs peu saturées, peu les peaux
        cur = (x.max(-1, keepdims=True) - x.min(-1, keepdims=True))
        sat = sat * (1 + v * (1 - np.clip(cur * 1.6, 0, 1)))
    x = y + (x - y) * sat
    if cube is not None and p["lut_mix"] > 0:
        x = x + (_sample_cube(cube, np.clip(x, 0, 1)) - x) * (p["lut_mix"] / 100)
    fade = p["fade"] / 100
    if fade:
        x = 0.07 * fade + x * (1 - 0.13 * fade)
    mono = p["mono"] / 100
    if mono:
        y = (x * [0.25, 0.65, 0.10]).sum(-1, keepdims=True)
        x = x + (y - x) * mono
    return np.clip(x, 0, 1).astype(np.float32)


def _sample_cube(cube, rgb):
    t = torch.from_numpy(cube).permute(3, 0, 1, 2)[None]
    g = torch.from_numpy(rgb.reshape(1, 1, 1, -1, 3).astype(np.float32)) * 2 - 1
    out = F.grid_sample(t, g, mode="bilinear", padding_mode="border", align_corners=True)
    return out[0, :, 0, 0].T.numpy().reshape(rgb.shape).astype(np.float32)


class Look:
    def __init__(self, device=None):
        self.p = dict(DEFAULTS)
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._key = None
        self._lut = None
        self._cube = (None, None)
        self._gen = torch.Generator(self.dev).manual_seed(1234)

    def active(self):
        return any(self.p[k] != v for k, v in DEFAULTS.items()
                   if k not in ("shadow_hue", "mid_hue", "high_hue", "lut_mix"))

    def _bake(self):
        color_keys = [k for k in DEFAULTS if k not in ("sharpen", "bloom", "vignette", "grain")]
        key = tuple(self.p[k] for k in color_keys)
        if key == self._key:
            return self._lut
        path = self.p["lut_path"]
        if path != self._cube[0]:
            cube = read_cube(path) if path and os.path.isfile(path) else None
            self._cube = (path, cube)
        g = np.linspace(0, 1, LUT_SIZE, dtype=np.float32)
        b, gg, r = np.meshgrid(g, g, g, indexing="ij")
        out = grade(np.stack([r, gg, b], -1), self.p, self._cube[1])
        self._lut = torch.from_numpy(out).permute(3, 0, 1, 2)[None].to(self.dev)
        self._key = key
        return self._lut

    @torch.no_grad()
    def apply(self, img: np.ndarray) -> np.ndarray:
        """img : BGR uint8 ; renvoie une nouvelle image BGR uint8."""
        if not self.active():
            return img
        p = self.p
        h, w = img.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(img)).to(self.dev, non_blocking=True).float() / 255
        x = x[..., [2, 1, 0]]                                     # (H, W, 3) rouge, vert, bleu
        lut = self._bake()
        x = F.grid_sample(lut, x[None, None] * 2 - 1, mode="bilinear", padding_mode="border",
                          align_corners=True)[0, :, 0]            # (3, H, W)
        scale = max(w, h) / 1920
        if p["sharpen"]:
            b = _blur(x, 1.1 * scale)
            x = (x + (x - b) * p["sharpen"] / 100 * 1.4).clamp(0, 1)
        if p["bloom"]:
            s = max(2, int(8 * scale))
            small = F.avg_pool2d(x[None], s)
            y = (small * torch.tensor([0.21, 0.72, 0.07], device=self.dev)[None, :, None, None]).sum(1, keepdim=True)
            glow = _blur((small * ((y - 0.62) / 0.3).clamp(0, 1))[0], 6.0)
            glow = F.interpolate(glow[None], size=(h, w), mode="bilinear", align_corners=False)[0]
            x = 1 - (1 - x) * (1 - glow * p["bloom"] / 100 * 1.2)   # mode « écran »
        if p["vignette"]:
            yy = torch.linspace(-1, 1, h, device=self.dev)[:, None]
            xx = torch.linspace(-1, 1, w, device=self.dev)[None, :]
            d = ((xx ** 2 + yy ** 2) / 2).sqrt()
            m = ((d - 0.45) / 0.55).clamp(0, 1) ** 2
            x = x * (1 - m * p["vignette"] / 100 * 0.8) if p["vignette"] > 0 else \
                x + (1 - x) * m * -p["vignette"] / 100 * 0.5
        if p["grain"]:
            gs = max(1, round(scale))
            n = torch.randn((1, 1, h // gs + 1, w // gs + 1), device=self.dev, generator=self._gen)
            if gs > 1:
                n = F.interpolate(n, size=(h, w), mode="bilinear", align_corners=False)
            else:
                n = n[..., :h, :w]
            y = x.mean(0, keepdim=True)
            x = x + n[0] * p["grain"] / 100 * 0.11 * (1 - (y * 2 - 1) ** 2).clamp(0.15, 1)
        out = (x.clamp(0, 1)[[2, 1, 0]] * 255 + 0.5).byte().permute(1, 2, 0)
        return out.cpu().numpy()


def _blur(x, sigma):
    """Flou gaussien séparable de x (3, H, W)."""
    r = max(1, int(sigma * 2.5))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())[None, None]
    c = x[:, None]
    c = F.conv2d(F.pad(c, (r, r, 0, 0), mode="replicate"), k[..., None, :])
    c = F.conv2d(F.pad(c, (0, 0, r, r), mode="replicate"), k[..., :, None])
    return c[:, 0]
