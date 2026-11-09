"""Petits outils de signal sur la carte graphique : rééchantillonnage polyphase (sinc fenêtré Kaiser)."""
import math

import torch
import torch.nn.functional as F


class Resampler:
    """Change la fréquence d'échantillonnage d'un bloc entier (filtre sinc, coupure à 97 % de Nyquist).

    Même principe que torchaudio.functional.resample, avec une fenêtre de Kaiser plus sélective :
    rien ne se replie dans le grave quand on descend de 48 à 16 kHz.
    """

    def __init__(self, orig, new, device, width=16, rolloff=0.97, beta=8.6):
        g = math.gcd(int(orig), int(new))
        self.orig, self.new = int(orig) // g, int(new) // g
        self.same = self.orig == self.new
        if self.same:
            return
        base = min(self.orig, self.new) * rolloff
        self.width = math.ceil(width * self.orig / base)
        idx = torch.arange(-self.width, self.width + self.orig, dtype=torch.float64)[None, None] / self.orig
        t = torch.arange(0, -self.new, -1, dtype=torch.float64)[:, None, None] / self.new + idx
        t = (t * base).clamp(-width, width)
        win = torch.special.i0(beta * torch.sqrt(1 - (t / width) ** 2)) / torch.special.i0(torch.tensor(beta, dtype=torch.float64))
        t = t * math.pi
        k = torch.where(t == 0, torch.ones_like(t), torch.sin(t) / t) * win * (base / self.orig)
        self.kernel = k.to(device=device, dtype=torch.float32)

    def __call__(self, x):
        """x : (n,) → (ceil(n·new/orig),)."""
        if self.same:
            return x
        n = x.shape[-1]
        y = F.conv1d(F.pad(x.float().view(1, 1, -1), (self.width, self.width + self.orig)), self.kernel, stride=self.orig)
        return y.transpose(1, 2).reshape(-1)[: math.ceil(self.new * n / self.orig)]
