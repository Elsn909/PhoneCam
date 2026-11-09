"""HuBERT / ContentVec (le « contenu » de la voix pour RVC), réécrit en PyTorch pur.

Même calcul que transformers.HubertModel (configuration « base », normalisation de groupe, couches
post-LN) avec les poids convertis fournis par RVC 2.3 (data/models-voix/hubert_base). Sans dépendance
à transformers ni fairseq ; l'attention passe par scaled_dot_product_attention (noyaux rapides).
RVC v2 lit la 12e couche (768), RVC v1 la 9e passée par final_proj (256).
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self, d, heads, ff, eps):
        super().__init__()
        self.heads = heads
        self.attention = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.attention, n, nn.Linear(d, d))
        self.layer_norm = nn.LayerNorm(d, eps=eps)
        self.feed_forward = nn.Module()
        self.feed_forward.intermediate_dense = nn.Linear(d, ff)
        self.feed_forward.output_dense = nn.Linear(ff, d)
        self.final_layer_norm = nn.LayerNorm(d, eps=eps)

    def forward(self, x):
        b, t, d = x.shape
        a = self.attention
        q, k, v = (p(x).view(b, t, self.heads, -1).transpose(1, 2) for p in (a.q_proj, a.k_proj, a.v_proj))
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, d)
        x = self.layer_norm(x + a.out_proj(h))
        f = self.feed_forward
        return self.final_layer_norm(x + f.output_dense(F.gelu(f.intermediate_dense(x))))


class Hubert(nn.Module):
    def __init__(self, folder, device, half=True):
        super().__init__()
        cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
        assert cfg["feat_extract_norm"] == "group" and not cfg["do_stable_layer_norm"], "configuration HuBERT inattendue"
        d, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        convs, c_in = [], 1
        for i, (c, k, s) in enumerate(zip(cfg["conv_dim"], cfg["conv_kernel"], cfg["conv_stride"])):
            m = nn.Module()
            m.conv = nn.Conv1d(c_in, c, k, s, bias=cfg["conv_bias"])
            if i == 0:
                m.layer_norm = nn.GroupNorm(c, c, affine=True)
            convs.append(m)
            c_in = c
        self.feature_extractor = nn.Module()
        self.feature_extractor.conv_layers = nn.ModuleList(convs)
        self.feature_projection = nn.Module()
        self.feature_projection.layer_norm = nn.LayerNorm(c_in, eps=eps)
        self.feature_projection.projection = nn.Linear(c_in, d)
        k = cfg["num_conv_pos_embeddings"]
        self.pos_conv = nn.Conv1d(d, d, k, padding=k // 2, groups=cfg["num_conv_pos_embedding_groups"])
        self.encoder = nn.Module()
        self.encoder.layer_norm = nn.LayerNorm(d, eps=eps)
        self.encoder.layers = nn.ModuleList(Layer(d, cfg["num_attention_heads"], cfg["intermediate_size"], eps)
                                            for _ in range(cfg["num_hidden_layers"]))
        self.final_proj = nn.Linear(d, cfg["classifier_proj_size"])
        self.trim = 1 if k % 2 == 0 else 0

        sd = torch.load(os.path.join(folder, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        sd.pop("masked_spec_embed", None)
        # Convolution de position : normalisation de poids (dim=2) appliquée une fois pour toutes.
        g = sd.pop("encoder.pos_conv_embed.conv.parametrizations.weight.original0").float()
        v = sd.pop("encoder.pos_conv_embed.conv.parametrizations.weight.original1").float()
        sd["pos_conv.weight"] = v * (g / v.norm(dim=(0, 1), keepdim=True))
        sd["pos_conv.bias"] = sd.pop("encoder.pos_conv_embed.conv.bias")
        self.load_state_dict({k: t.float() for k, t in sd.items()}, strict=True)
        self.eval().to(device)
        self.dtype_ = torch.float16 if half else torch.float32
        self.to(self.dtype_)

    @torch.no_grad()
    def forward(self, wav, version="v2"):
        """wav : (n,) à 16 kHz → (1, n/320, 768 ou 256), une trame toutes les 20 ms."""
        x = wav.to(self.dtype_).view(1, 1, -1)
        for i, m in enumerate(self.feature_extractor.conv_layers):
            x = m.conv(x)
            if i == 0:
                x = m.layer_norm(x)
            x = F.gelu(x)
        x = x.transpose(1, 2)
        x = self.feature_projection.projection(self.feature_projection.layer_norm(x))
        p = self.pos_conv(x.transpose(1, 2))
        if self.trim:
            p = p[:, :, :-self.trim]
        x = self.encoder.layer_norm(x + F.gelu(p).transpose(1, 2))
        last = 9 if version == "v1" else len(self.encoder.layers)
        for layer in self.encoder.layers[:last]:
            x = layer(x)
        return self.final_proj(x) if version == "v1" else x
