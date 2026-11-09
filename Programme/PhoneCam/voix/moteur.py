"""Changeur de voix en temps réel, basé sur RVC 2.3 (Retrieval-based-Voice-Conversion-WebUI).

Par bloc (0,20 s par défaut), à 16 kHz pour l'analyse :
  48 kHz → 16 kHz (FIR exact) → [Smart : nettoyage GTCRN → détection de voix Silero]
  → HuBERT (contenu, 2,5 s de contexte) → index du modèle (recherche exacte sur la carte graphique)
  → RMVPE (hauteur ; Smart : seulement dans la parole) → synthétiseur RVC (NSF)
  → 48 kHz → [Smart : silence hors parole] → suivi du volume → fondu SOLA → sortie.

Différences avec RVC d'origine, pour un meilleur rendu :
  · Smart (voir smart.py) : bruits, souffles et clics n'entrent plus dans le modèle, et le
    générateur de sinus ne transforme plus le bruit en voix ;
  · « protection des consonnes » aussi en temps réel (RVC ne l'applique qu'aux fichiers) ;
  · index cherché exactement (tous les vecteurs, sur la carte graphique) au lieu d'IVF approché ;
  · rééchantillonnage d'entrée sans erreur aux jointures des blocs ;
  · calculs dans un flux CUDA prioritaire : la voix passe devant le traitement vidéo ;
  · modèles chargés en « poids seuls » : un .pth piégé ne peut pas exécuter de code.
"""
import math
import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import smart
from .dsp import Resampler
from .graph import Graphs
from .reglages import DEFAULTS, RESTART_KEYS, find_index  # noqa: F401
from .vivante import Lively

try:
    from tour_gpu import voice_turn
except ImportError:                     # moteur utilisé seul, hors de PhoneCam
    from contextlib import nullcontext as voice_turn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "models-voix")
SR = 48000
ZC = SR // 100              # une trame = 10 ms
F0_MIN, F0_MAX = 50.0, 1100.0
MEL_MIN, MEL_MAX = 1127 * math.log(1 + F0_MIN / 700), 1127 * math.log(1 + F0_MAX / 700)


class Model:
    """Un modèle RVC (.pth) et son index (.index) facultatif, prêts sur la carte graphique."""
    count = 0

    def __init__(self, pth, index, device, half):
        from .rvc.models import (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono,
                                 SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono)
        cpt = torch.load(pth, map_location="cpu", weights_only=True)    # jamais de code exécuté
        if not isinstance(cpt, dict) or "weight" not in cpt or "config" not in cpt:
            raise ValueError("ce fichier n'est pas un modèle de voix RVC (le petit .pth d'inférence, ~50 Mo)")
        vocoder = cpt.get("vocoder", "HiFi-GAN")
        if vocoder != "HiFi-GAN":
            raise ValueError(f"modèle Applio avec le vocodeur « {vocoder} » : seuls les modèles HiFi-GAN (RVC) sont pris en charge")
        if cpt.get("embedder_model", "contentvec") not in ("contentvec", "hubert"):
            raise ValueError(f"modèle entraîné avec « {cpt['embedder_model']} » : seul ContentVec/HuBERT est pris en charge")
        cfg = list(cpt["config"])
        cfg[-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.f0 = int(cpt.get("f0", 1))
        self.version = cpt.get("version", "v1")
        self.sr = int(cfg[-1]) if not isinstance(cfg[-1], str) else {"32k": 32000, "40k": 40000, "48k": 48000}[cfg[-1]]
        cls = {("v1", 1): SynthesizerTrnMs256NSFsid, ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
               ("v2", 1): SynthesizerTrnMs768NSFsid, ("v2", 0): SynthesizerTrnMs768NSFsid_nono}[(self.version, self.f0)]
        net = cls(*cfg, is_half=half) if self.f0 else cls(*cfg)
        del net.enc_q
        net.load_state_dict(cpt["weight"], strict=False)
        net.eval().to(device)
        net.remove_weight_norm()
        self.net = net.half() if half else net.float()
        self.name = os.path.splitext(os.path.basename(pth))[0]
        self.pth = pth
        Model.count += 1
        self.serial = Model.count           # clé des graphes CUDA (un id() peut être réutilisé)
        self.big = self.big_sq = None
        self.index = ""
        if index and os.path.isfile(index):
            import faiss
            ix = faiss.read_index(index)
            vec = ix.reconstruct_n(0, ix.ntotal).astype(np.float32)
            self.big = torch.from_numpy(vec).to(device, torch.float16)
            self.big_sq = (torch.from_numpy(vec) ** 2).sum(1).to(device)
            self.index = index
            self.n_index = ix.ntotal
        dim = 768 if self.version == "v2" else 256
        if self.big is not None and self.big.shape[1] != dim:
            raise ValueError(f"l'index ({self.big.shape[1]} dim.) ne correspond pas au modèle ({self.version})")

    def describe(self):
        bits = [self.version.upper(), f"{self.sr // 1000} kHz", "avec hauteur" if self.f0 else "sans hauteur"]
        bits.append(f"index : {self.n_index:,} vecteurs".replace(",", " ") if self.big is not None else "sans index")
        return " · ".join(bits)


class VoiceChanger:
    def __init__(self, data_dir=DATA):
        self.p = dict(DEFAULTS)
        self.data = data_dir
        self.lock = threading.RLock()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.dev.type == "cuda"
        self.model = None
        self.base = False
        self.ready = False           # moteur configuré, prêt à traiter
        self.status = "Aucun modèle chargé."
        self.loading = False
        self.stream = None           # flux CUDA prioritaire, créé au premier chargement
        self.graphs = Graphs(self.dev.type == "cuda")
        self.ms = self.ms_max = 0.0
        self.in_db = self.out_db = -120.0
        self.speaking = False
        self.last_silent = False
        self.skipped = 0
        self.lively = Lively()
        self.lift_shift = 0.0
        self._cfg_key = None
        self._acc = np.zeros(0, np.float32)
        self._t_hist = []
        # Boîte noire : pour chaque trame sortie, niveau brut, hauteur, détection, porte (30 s).
        import collections
        self.diag = collections.deque(maxlen=150)

    # ------------------------------------------------------------ chargement
    def _load_base(self):
        if self.base:
            return
        if self.dev.type == "cuda":
            self.stream = torch.cuda.Stream(priority=-1)     # passe devant la vidéo sur la carte
        from .hubert import Hubert
        from .rmvpe import RMVPE
        self.hubert = Hubert(os.path.join(self.data, "hubert_base"), self.dev, self.half)
        self.rmvpe = RMVPE(os.path.join(self.data, "rmvpe.pt"), self.dev, self.half)
        self.vad = smart.Vad(os.path.join(self.data, "silero_vad.onnx"))
        self.vad_raw = smart.Vad(os.path.join(self.data, "silero_vad.onnx"))     # sur le son brut
        self.denoiser = smart.Denoiser(os.path.join(self.data, "gtcrn_simple.onnx"))
        self.gate = smart.Gate()
        self.breath = smart.Breath()
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.base = True

    def load(self, pth, index, done=None):
        """Charge un modèle en arrière-plan ; done(ok, message) est appelé à la fin (dans ce fil)."""
        def work():
            self.loading = True
            self.status = "Chargement…"
            try:
                t0 = time.perf_counter()
                self._load_base()
                with self._ctx():
                    model = Model(pth, index, self.dev, self.half)
                with self.lock:
                    self.ready = False
                    self.graphs.clear()             # graphes de l'ancien modèle : mémoire rendue
                    old, self.model = self.model, model
                    self.p["pth"], self.p["index"] = pth, model.index
                    self._configure()
                    self._warmup()
                    self._configure()
                    self.ready = True
                del old
                torch.cuda.empty_cache()
                self.status = f"Prêt : {model.name} ({time.perf_counter() - t0:.1f} s)"
                ok, msg = True, model.describe()
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"Impossible de charger le modèle : {e}"
                self.status = msg
            finally:
                self.loading = False
            if done:
                done(ok, msg)
        threading.Thread(target=work, daemon=True, name="voix-chargement").start()

    def _ctx(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else _Null()

    def set(self, key, value):
        self.p[key] = value
        if key in RESTART_KEYS:
            self._cfg_key = None     # reconfiguré au prochain bloc (dans le fil audio)

    # ------------------------------------------------------------ état temps réel
    def _key(self):
        return tuple(self.p[k] for k in RESTART_KEYS) + (getattr(self.model, "serial", 0),)

    def _configure(self):
        p, m, dev = self.p, self.model, self.dev
        q = lambda s: int(round(s * SR / ZC)) * ZC          # noqa: E731
        self.block = q(p["block"] / 100)
        self.cross = q(p["crossfade"] / 100)
        self.sola_buf = min(self.cross, 4 * ZC)
        self.search = ZC
        self.extra = q(p["extra"] / 10)
        total = self.extra + self.cross + self.search + self.block
        self.n = total // ZC
        self.n16 = total // 3
        self.b16 = self.block // 3
        self.bf = self.block // ZC
        self.skip = self.extra // ZC
        self.ret = (self.block + self.sola_buf + self.search) // ZC
        self.x16 = torch.zeros(self.n16, device=dev)
        self.f0_raw = np.zeros(self.n)
        self.conf = np.zeros(self.n)
        self.prob = np.zeros(self.n)
        self.db = np.full(self.n, -120.0)
        self.mask = np.zeros(self.n, np.float32)
        self.click = np.zeros(self.n, bool)
        self.breathy = np.zeros(self.n, bool)
        self.breath.tail[:] = 0
        self.breath.floor, self.breath.known = None, 0
        self.sola = torch.zeros(self.sola_buf, device=dev)
        t = torch.linspace(0, 1, self.sola_buf, device=dev)
        self.fade_in = torch.sin(0.5 * math.pi * t) ** 2
        self.fade_out = 1 - self.fade_in
        self.den_kernel = torch.ones(1, 1, self.sola_buf, device=dev)
        self.down = smart.Down3()
        self.denoiser.reset(self.b16)
        self.raw_line = np.zeros(self.denoiser.delay, np.float32)
        self.vad.reset()
        self.vad_raw.reset()
        self.gate.reset()
        self.abs16 = 0
        self.commit = 0
        self.f0_len = 5120 * ((self.b16 + 800 - 1) // 5120 + 1) - 160
        self.up = Resampler(m.sr, SR, dev)
        self._formant = {}
        self._acc = np.zeros(0, np.float32)
        self.lively.reset()
        self._cfg_key = self._key()
        self.skipped = 0

    def latency_ms(self):
        """Retard total ajouté par la voix (hors cartes son)."""
        if self.model is None:
            return 0.0
        p = self.p
        d = p["block"] * 10 + p["crossfade"] * 10 + 10 + 81 / 48 + p["margin_ms"]
        if p["denoise"] and p["smart"]:
            d += getattr(self.denoiser, "delay", 384) / 16
        return d + self.ms

    # ------------------------------------------------------------ traitement
    def feed(self, x):
        """Morceaux de son à 48 kHz (n'importe quelle taille) → son transformé, par blocs entiers
        (tableau vide tant qu'un bloc n'est pas complet). Jamais la voix d'origine : en attendant un
        modèle, c'est du silence."""
        if not self.ready or not self.lock.acquire(blocking=False):
            return np.zeros(len(x), np.float32)
        try:
            if self._cfg_key != self._key():
                with self._ctx():
                    self._configure()
            self._acc = np.concatenate([self._acc, x.astype(np.float32)])
            outs = []
            while len(self._acc) >= self.block:
                blk, self._acc = self._acc[:self.block], self._acc[self.block:]
                t0 = time.perf_counter()
                with voice_turn(), self._ctx():        # la vidéo patiente ces ~20 ms (tour_gpu.py)
                    y = self._step(blk)
                dt = (time.perf_counter() - t0) * 1000
                self._t_hist = (self._t_hist + [dt])[-25:]
                self.ms = float(np.mean(self._t_hist))
                self.ms_max = float(np.max(self._t_hist))
                outs.append(y)
            return np.concatenate(outs) if outs else np.zeros(0, np.float32)
        except Exception as e:  # noqa: BLE001  (jamais la voix d'origine, même en cas d'erreur)
            self.status = f"Erreur : {e}"
            self._cfg_key = None
            return np.zeros(len(x), np.float32)
        finally:
            self.lock.release()

    def _step(self, blk):
        p = self.p
        use_smart = bool(p["smart"])
        self.in_db = 20 * math.log10(float(np.abs(blk).max()) + 1e-9)
        # 1. 48 → 16 kHz, nettoyage (le son brut est retardé d'autant pour pouvoir doser le mélange)
        raw = self.down(blk)
        if use_smart and p["denoise"]:
            self.denoiser.floor = 10 ** (-p["denoise_db"] / 20)
            clean = self.denoiser(raw)
            line = np.concatenate([self.raw_line, raw])           # son brut aligné (détection de voix)
            raw_d, self.raw_line = line[:len(raw)], line[len(raw):]
        else:
            clean = raw_d = raw
        B = self.bf
        for a in (self.f0_raw, self.conf, self.prob, self.db, self.mask, self.click, self.breathy):
            a[:-B] = a[B:]
            a[-B:] = 0
        # Niveau mesuré sur le son brut : après nettoyage, une voix douce passerait sous le seuil.
        fr = raw_d.reshape(B, -1)
        self.db[-B:] = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
        # Chocs : crête très au-dessus du niveau moyen de la trame, ou niveau qui bondit d'un coup.
        crest = 20 * np.log10(np.abs(fr).max(axis=1) + 1e-9) - self.db[-B:]
        prev = np.array([self.db[-B - 3 + j:-B + j].max() for j in range(B)])
        self.click[-B:] = (crest > 15) | (self.db[-B:] - prev > 18)
        floor = self.breath.track(self.db[-B:])
        self.breathy[-B:] = self.breath(raw_d, self.db[-B:], floor)
        # pas de souffle à moins de 120 ms d'un choc (frappes de clavier : du bruit entre les clics)
        near = np.convolve(self.click.astype(np.float32), np.ones(25), "same") > 0
        self.breathy &= ~near
        self.x16[:-self.b16] = self.x16[self.b16:].clone()
        self.x16[-self.b16:] = torch.from_numpy(clean).to(self.dev, non_blocking=True)
        self.abs16 += self.b16
        A = self.abs16 // 160                                       # trames depuis le départ
        # 2. hauteur (toujours : elle sert aussi à la détection de voix)
        m = self.model
        f0, conf = self.graphs.run("rmvpe", lambda a: self.rmvpe(a, thr=0.0), self.x16[-self.f0_len:])
        f0, conf = f0.cpu().numpy(), conf.cpu().numpy()
        f0 = np.where(conf > (0.01 if use_smart else 0.03), f0, 0.0)
        k = len(f0) - 4
        self.f0_raw[-k:] = f0[3:-1]
        f_ok = (f0[3:-1] >= 60) & (f0[3:-1] <= 600)              # hauteur d'une voix humaine
        self.conf[-k:] = np.where(f_ok, conf[3:-1], 0.0)
        if use_smart and p["denoise"]:
            self._keep_vowels(clean, raw_d)
        # 3. détection de voix → masque
        if use_smart:
            self._vad(clean, raw_d, A)
        else:
            self.mask[:] = 1
        self.speaking = bool(self.mask[-B:].any())
        s0 = self.skip
        self.diag.append(np.stack([self.db[s0:s0 + B], self.conf[s0:s0 + B], self.prob[s0:s0 + B],
                                   self.mask[s0:s0 + B], self.click[s0:s0 + B], self.breathy[s0:s0 + B]], 1)
                         .astype(np.float32))
        # Rien à dire dans ce qui va sortir (ni juste avant) : ce bloc sortira en silence. Le calcul
        # est fait quand même (~5 % de la carte) : sans travail, la carte baisse ses fréquences et
        # le premier mot après un silence prendrait 10 fois plus de temps (coupure).
        self.last_silent = bool(use_smart and not self._soft()[self.skip - 2:].any())
        if self.last_silent:
            self.skipped += 1
        x_in = self.x16
        if use_smart:
            env16 = F.interpolate(torch.from_numpy(self._soft()).to(self.dev).view(1, 1, -1), size=self.n16,
                                  mode="linear", align_corners=False).view(-1)
            x_in = self.x16 * env16
        # 4. hauteur utilisée par le modèle
        factor = 2 ** (p["formant"] / 120)
        ret2 = int(math.ceil(self.ret * factor))
        if m.f0:
            if use_smart:
                f0u = smart.smart_f0(self.f0_raw, self.conf, self.mask, p["voicing"] / 100)
            else:
                f0u = smart.classic_f0(np.where(self.conf > 0.03, self.f0_raw, 0))
            if p["lift_on"]:            # voix vivante : ton bas relevé, mélodie élargie
                f0u = self.lively.pitch(f0u, p, self.skip, self.skip + self.ret, self.bf)
            self.lift_shift = self.lively.shift if p["lift_on"] else 0.0
            self.cur_hz = self.lively.cur_hz
            f0u = f0u * 2 ** ((p["pitch"] - p["formant"] / 10) / 12)
            mel = 1127 * np.log(1 + f0u / 700)
            coarse = np.where(mel > 0, (mel - MEL_MIN) * 254 / (MEL_MAX - MEL_MIN) + 1, 1)
            coarse = np.clip(np.round(coarse), 1, 255).astype(np.int64)
            pitch = torch.from_numpy(coarse).to(self.dev)[None]
            pitchf = torch.from_numpy((f0u * ret2 / self.ret).astype(np.float32)).to(self.dev)[None]
        # 5. contenu (HuBERT) + index
        feats = self.graphs.run(("hubert", m.version), lambda a: self.hubert(a, m.version), x_in)
        feats = torch.cat([feats, feats[:, -1:]], 1)
        rate = p["index_rate"] / 100
        protect = p["protect"] / 100
        feats0 = feats.clone() if (m.f0 and protect < 0.5 and rate > 0) else None
        if m.big is not None and rate > 0:
            h = self.skip // 2
            ret_feats = self.graphs.run(("index", m.serial), lambda q: _search(m, q), feats[0, h:].to(torch.float16))
            feats[0, h:] = (ret_feats * rate + feats[0, h:].float() * (1 - rate)).to(feats.dtype)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)[:, :self.n]
        if feats0 is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)[:, :self.n]
            pf = torch.where(pitchf > 0, 1.0, protect).to(feats.dtype)[..., None]
            feats = feats * pf + feats0 * (1 - pf)
        # 6. synthèse
        n_len = torch.tensor([self.n], device=self.dev)
        sid = torch.tensor([0], device=self.dev)
        with torch.no_grad():
            key = ("synth", m.serial, self.skip, self.ret, ret2)
            if m.f0:
                audio = self.graphs.run(key, lambda a, b, c, d, e: m.net.infer(a, b, c, d, e, self.skip, self.ret, ret2)[0],
                                        feats, n_len, pitch, pitchf, sid)
            else:
                audio = self.graphs.run(key, lambda a, b, e: m.net.infer(a, b, e, self.skip, self.ret, ret2)[0],
                                        feats, n_len, sid)
        audio = audio.view(-1).float()
        upp = int(math.floor(factor * m.sr // 100))
        if upp != m.sr // 100:
            if upp not in self._formant:
                self._formant[upp] = Resampler(upp, m.sr // 100, self.dev)
            audio = self._formant[upp](audio[: self.ret * upp])
        out = self.up(audio)[: self.ret * ZC]
        if len(out) < self.ret * ZC:
            out = F.pad(out, (0, self.ret * ZC - len(out)))
        # 7. silence hors parole, suivi du volume
        seg = torch.from_numpy(self._soft()[self.skip:self.skip + self.ret]).to(self.dev)
        if use_smart:
            out = out * F.interpolate(seg.view(1, 1, -1), size=len(out), mode="linear", align_corners=False).view(-1)
        follow = p["follow"] / 100
        if follow > 0:
            src = x_in[self.skip * 160:(self.skip + self.ret) * 160]
            r1 = _rms(src, 160, len(out))
            r2 = _rms(out, ZC, len(out))
            out = out * torch.clamp(r1 / r2.clamp_min(1e-3), max=4.0) ** follow
        out = out * 10 ** (p["gain"] / 20)
        # 8. SOLA (recherche du meilleur point de raccord, puis fondu)
        n = self.sola_buf + self.search
        head = out[None, None, :n]
        num = F.conv1d(head, self.sola[None, None, :])
        den = torch.sqrt(F.conv1d(head ** 2, self.den_kernel) + 1e-8)
        off = int(torch.argmax(num[0, 0] / den[0, 0]))
        out = out[off:]
        out[:self.sola_buf] = out[:self.sola_buf] * self.fade_in + self.sola * self.fade_out
        self.sola[:] = out[self.block:self.block + self.sola_buf]
        y = out[:self.block].clamp(-1, 1).cpu().numpy()
        if p["lift_on"] and p["lift_bright"] > 0:
            y = np.clip(self.lively.brighten(y, p), -1, 1)
        if self.last_silent:
            y[:] = 0.0                          # silence exact (PhoneCam peut le raccourcir)
            self.sola.zero_()
        self.out_db = 20 * math.log10(float(np.abs(y).max()) + 1e-9)
        return y

    def _soft(self, frames=6):
        """Masque de Smart avec une fermeture en fondu (60 ms) : la fin d'une phrase s'éteint au lieu
        d'être coupée net. L'ouverture reste immédiate (l'ouverture anticipée s'en charge)."""
        m = self.mask.astype(np.float32)
        out = m.copy()
        for k in range(1, frames):
            out[k:] = np.maximum(out[k:], m[:-k] * (1 - k / frames))
        return out

    def _keep_vowels(self, clean, raw):
        """Le débruiteur prend parfois une voyelle tenue pour du bruit de fond (« aaaah » dont le niveau
        baisse un peu : −20 dB en 1 s). Là où une vraie hauteur de voix est détectée, il ne retire
        jamais plus de 3 dB : le son brut est remélangé ce qu'il faut (le bruit reste masqué par la voix).
        Juste après la voix (fin de mot), jamais plus de 6 dB."""
        B = self.bf
        c = self.conf[-B - 2:]
        cv = np.maximum(np.maximum(c[:-2], c[1:-1]), c[2:])        # ±1 trame (hauteur un peu décalée)
        # Fins de mots (« mal », « bon », « oui ») : la consonne ou la voyelle qui s'éteint juste après
        # la voix (150 ms) perd au plus 6 dB, au lieu d'être effacée comme un bruit de fond.
        v = (self.conf[-B - 17:] > 0.5).astype(np.float32)
        recent = np.convolve(v, np.ones(16, np.float32))[:len(v)][-B:] > 0
        tail = recent & (self.db[-B:] > self.p["floor_db"] + 6)
        gmin = np.where(cv > 0.5, 0.7, np.where(tail, 0.5, 0.0))
        ec = np.mean(clean.reshape(B, -1) ** 2, axis=1)
        er = np.mean(raw.reshape(B, -1) ** 2, axis=1) + 1e-12
        g = np.sqrt(ec / er)
        a = np.where(g < gmin, np.clip((gmin - g) / np.maximum(1 - g, 1e-3), 0, 1), 0.0)
        if not a.any():
            return
        a = np.interp(np.arange(len(clean)), np.arange(B) * 160 + 80, a)
        self.x16[-self.b16:] = torch.from_numpy((clean + a * (raw - clean)).astype(np.float32)).to(self.dev)

    def _vad(self, clean, raw, A):
        n0 = A - self.n                                             # trame absolue de self.prob[0]
        res_r = self.vad_raw(raw)
        res = [(s, e, max(pc, pr)) for (s, e, pc), (_s, _e, pr) in zip(self.vad(clean), res_r)]
        last = self.prob[-self.bf - 1] if not res else res[-1][2]
        for s, e, pr in res:
            f0 = max(-(-(s - 80) // 160), n0)
            f1 = min(-(-(e - 80) // 160), A)
            if f1 > f0:
                self.prob[f0 - n0:f1 - n0] = pr
        done = -(-(self.vad.pos - 80) // 160)                       # trames dont la proba est définitive
        if done < A:
            self.prob[done - n0:] = last
        start = max(self.commit, n0)
        pre = max(0, int(round(self.p["preroll_ms"] / 10)))
        gp = dict(self.p, sensitivity=self.p["sensitivity"] / 100)
        self.mask[start - n0:] = self.gate.run(self.prob[start - n0:], self.db[start - n0:], self.conf[start - n0:], gp,
                                               done - start, self.click[start - n0:], self.breathy[start - n0:])
        lo = max(0, start - n0 - pre - 1)
        self.mask[lo:] = smart.preroll(self.mask[lo:], pre)
        self.commit = max(self.commit, done)

    def _warmup(self):
        """Quelques blocs factices : les noyaux CUDA sont choisis avant le direct (sinon 1er bloc lent)."""
        t = np.arange(self.block * 4) / SR
        x = (0.2 * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        smart_on, self.p["smart"] = self.p["smart"], False
        lift_on, self.p["lift_on"] = self.p["lift_on"], False      # le son de test n'est pas ta voix
        try:
            with self._ctx():
                for i in range(4):
                    self._step(x[i * self.block:(i + 1) * self.block])
                # Smart, détection forcée ouverte : nettoyage, détection (ONNX) et chemin « parole »
                # (le 1er passage de chacun est lent, il ne doit pas tomber sur la 1re phrase).
                saved = {k: self.p[k] for k in ("sensitivity", "floor_db")}
                self.p.update(smart=True, sensitivity=0, floor_db=-300)
                try:
                    for i in range(4):
                        self._step(x[i * self.block:(i + 1) * self.block])
                finally:
                    self.p.update(saved)
            if self.stream is not None:
                self.stream.synchronize()
        finally:
            self.p["smart"] = smart_on
            self.p["lift_on"] = lift_on


def _search(m, q):
    """Les 8 plus proches voisins de chaque trame dans l'index (distance euclidienne exacte), pondérés."""
    d = (q.float() ** 2).sum(1, keepdim=True) - 2 * (q @ m.big.T).float() + m.big_sq[None]
    score, ix = torch.topk(d, 8, dim=1, largest=False)
    w = 1 / score.clamp_min(1e-6) ** 2
    w = w / w.sum(1, keepdim=True)
    r = (m.big[ix].float() * w[..., None]).sum(1)
    # Lissé sur ±40 ms : sur une note tenue (« mmmm », « eeeuh »), les voisins trouvés changent d'un
    # bloc à l'autre et le timbre « sautait » toutes les 200 ms. Mesuré : ce rythme ÷ 2, articulation
    # gardée (ressemblance du contenu entrée / sortie 0,78 → 0,80).
    p = torch.cat([r[:1], r[:1], r, r[-1:], r[-1:]], 0)
    return (p[:-4] + 2 * p[1:-3] + 3 * p[2:-2] + 2 * p[3:-1] + p[4:]) / 9


def _rms(x, hop, size):
    """Niveau efficace sur 40 ms toutes les 10 ms, ramené à « size » échantillons."""
    fr = F.pad(x.view(1, 1, -1), (2 * hop, 2 * hop), mode="reflect").view(-1).unfold(0, 4 * hop, hop)
    r = torch.sqrt((fr ** 2).mean(1) + 1e-12)
    return F.interpolate(r.view(1, 1, -1), size=size, mode="linear", align_corners=False).view(-1)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
