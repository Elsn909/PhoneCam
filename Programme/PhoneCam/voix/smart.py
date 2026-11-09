"""Smart : rien d'autre que ta voix n'entre dans le changeur de voix.

Trois étages, tous en flux, à 16 kHz (la fréquence de travail de HuBERT et RMVPE) :

1. Nettoyage (GTCRN, réseau de débruitage léger, licence MIT) : retire le souffle, le ventilateur,
   le clavier… avant que HuBERT et RMVPE ne les « entendent ». Latence propre : 16 à 24 ms.
2. Détection de voix (Silero VAD, licence MIT), sur le son nettoyé ET sur le son brut (le plus
   sûr des deux : une voix douce ou chuchotée, que le nettoyage atténue, reste reconnue ; le
   clavier ne trompe ni l'un ni l'autre). Hystérésis, maintien après la dernière syllabe (les
   occlusions du français : p, t, k) et ouverture anticipée (le moteur a 50 ms d'avance sur ce
   qu'il joue). Ce qui sort de ta bouche passe (voix, souffles, respirations, chuchotements :
   Silero les reconnaît un peu, même sans hauteur) ; ce qui n'a rien de vocal reste dehors :
   les chocs (clavier, clic, objet posé : attaque brutale) ne peuvent pas ouvrir la porte, et
   une hauteur hors de la voix humaine (grincement aigu) ne compte pas comme une voix. Silero décroche sur une voyelle tenue (« aaaah », « mmmm ») : tant que la
   porte est ouverte, la hauteur la garde ouverte. La hauteur seule ne l'ouvre pas (un bip, une
   musique ne passent pas). Tout ce qui n'est pas de la voix sort en silence exact.
3. Hauteur « intelligente » : RVC d'origine comble les trous de hauteur en interpolant, si bien que
   le générateur de sinus transforme bruits et souffles en voix chantée. Ici, la hauteur n'existe
   que dans la parole, avec une confiance suffisante ; seuls les trous très courts (< 40 ms,
   au milieu d'une voyelle) sont comblés, les fausses détections isolées sont effacées et les
   sauts d'octave sont lissés. Les consonnes restent des consonnes (bruit, pas de sinus).
"""
import numpy as np
import onnxruntime as ort
from scipy import signal


def _session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class Down3:
    """48 kHz → 16 kHz en flux, sans erreur aux bords des blocs (FIR 163 points, retard 81 éch. = 1,7 ms)."""
    TAPS = 163

    def __init__(self):
        self.h = signal.firwin(self.TAPS, 7300, window=("kaiser", 8.0), fs=48000)
        self.hist = np.zeros(self.TAPS - 1)

    def __call__(self, x):
        """x : bloc à 48 kHz, longueur multiple de 3 → x/3 échantillons à 16 kHz."""
        full = np.concatenate([self.hist, x])
        self.hist = full[-(self.TAPS - 1):]
        return signal.upfirdn(self.h, full, 1, 3)[54:54 + len(x) // 3].astype(np.float32)


class Denoiser:
    """GTCRN image par image (trames de 32 ms, pas de 16 ms), état conservé d'un bloc à l'autre."""

    def __init__(self, path):
        self.sess = _session(path)
        self.win = np.sqrt(np.hanning(513)[:512]).astype(np.float32)    # sqrt-Hann périodique
        self.block = None
        # Atténuation maximale : le réseau prend parfois un chuchotement ou l'explosion d'un « p »
        # pour du souffle et l'efface. On ne descend jamais sous ce gain (les bruits seuls, eux,
        # sont coupés par la détection de voix).
        self.floor = 10 ** (-18 / 20)

    def reset(self, block):
        self.block = block
        self.c1 = np.zeros((2, 1, 16, 16, 33), np.float32)
        self.c2 = np.zeros((2, 3, 1, 1, 16), np.float32)
        self.c3 = np.zeros((2, 1, 33, 16), np.float32)
        self.inbuf = np.zeros(256, np.float32)
        self.ola = np.zeros(512, np.float32)
        # Réserve de départ : de quoi rendre exactement « block » échantillons à chaque appel.
        pad = max((k * block) % 256 for k in range(1, 257))
        self.out = np.zeros(pad, np.float32)
        self.delay = 256 + pad

    def __call__(self, x):
        self.inbuf = np.concatenate([self.inbuf, x])
        outs = [self.out]
        while len(self.inbuf) >= 512:
            spec = np.fft.rfft(self.inbuf[:512] * self.win)
            mix = np.stack([spec.real, spec.imag], -1).astype(np.float32)[None, :, None, :]
            enh, self.c1, self.c2, self.c3 = self.sess.run(None, {"mix": mix, "conv_cache": self.c1,
                                                                  "tra_cache": self.c2, "inter_cache": self.c3})
            e = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
            low = np.abs(e) < self.floor * np.abs(spec)
            e = np.where(low, spec * self.floor, e)
            y = np.fft.irfft(e, 512).astype(np.float32) * self.win
            self.ola += y
            outs.append(self.ola[:256].copy())
            self.ola = np.concatenate([self.ola[256:], np.zeros(256, np.float32)])
            self.inbuf = self.inbuf[256:]
        out = np.concatenate(outs)
        self.out = out[len(x):]
        return out[:len(x)]


class Vad:
    """Silero VAD (v5/v6) en flux : une probabilité de voix par tranche de 32 ms."""

    def __init__(self, path):
        self.sess = _session(path)
        self.sr = np.array(16000, np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros(64, np.float32)
        self.buf = np.zeros(0, np.float32)
        self.pos = 0                         # échantillons déjà analysés (début de self.buf)

    def __call__(self, x):
        """→ liste de (début, fin, probabilité), en échantillons depuis le départ du flux."""
        self.buf = np.concatenate([self.buf, x])
        res = []
        while len(self.buf) >= 512:
            chunk = self.buf[:512]
            inp = np.concatenate([self.ctx, chunk])[None]
            out, self.state = self.sess.run(None, {"input": inp, "state": self.state, "sr": self.sr})
            self.ctx = chunk[-64:]
            res.append((self.pos, self.pos + 512, float(out[0, 0])))
            self.buf = self.buf[512:]
            self.pos += 512
        return res


class Breath:
    """Souffles et respirations, que Silero ne reconnaît pas : un bruit large (spectre plat), centré
    entre 400 et 4 000 Hz, nettement au-dessus du bruit de fond, ni choc ni grondement grave.
    Frottements de bonnette et manipulations : surtout des graves (< 300 Hz) → refusés ;
    ventilateur : constant, au niveau du bruit de fond → refusé ; clavier : des chocs, ou des bouffées
    de moins de 60 ms → refusé (il faut 150 ms de souffle continu pour ouvrir)."""

    def __init__(self):
        self.tail = np.zeros(160, np.float32)
        self.win = np.hanning(320)
        f = np.fft.rfftfreq(320, 1 / 16000)
        self.band = (f >= 100) & (f <= 6000)
        self.lo, self.mid = (f > 100) & (f < 1000), f >= 1000
        self.f = f
        self.floor = None           # bruit de fond (dB), suivi en continu
        self.known = 0              # trames de son réel vues (le micro peut sortir du silence numérique)
        self.prev_db = -120.0

    def track(self, db):
        """Bruit de fond : descend vite vers les creux, ne remonte que lentement (≈ 4 s pour +10 dB)."""
        for d in db:
            if d < -100:            # silence numérique du micro : rien à apprendre
                continue
            self.known += 1
            if self.floor is None:
                self.floor = d
            elif d < self.floor:
                self.floor += (d - self.floor) * 0.3
            else:
                self.floor += min(d - self.floor, 0.025)
        return self.floor if self.known >= 100 else None

    def __call__(self, x, db, floor_db):
        """x : nouveau son brut (16 kHz, multiple de 160) ; db : niveaux de ces trames ; floor_db : bruit
        de fond estimé (None : pas encore connu, aucun souffle accepté). → booléens par trame de 10 ms."""
        full = np.concatenate([self.tail, x])
        self.tail = x[-160:]
        n = len(x) // 160
        fr = np.lib.stride_tricks.sliding_window_view(full, 320)[::160][:n] * self.win
        P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-14
        flat = np.exp(np.mean(np.log(P[:, self.band]), 1)) / np.mean(P[:, self.band], 1)
        cen = (P * self.f).sum(1) / P.sum(1)
        r_lo = 10 * np.log10(P[:, self.lo].sum(1) / P[:, self.mid].sum(1))
        # régularité : un souffle monte et descend doucement ; un clavier, un frottement sautillent
        jit = np.abs(np.diff(np.concatenate([[self.prev_db], db])))
        self.prev_db = float(db[-1])
        smooth = np.convolve(jit, np.ones(8) / 8, "same") < 3.5
        if floor_db is None:
            return np.zeros(n, bool)
        return (flat > 0.06) & (cen > 400) & (cen < 4000) & (r_lo < 10) & (db > floor_db + 9) & smooth


PHRASE = 150       # trames (1,5 s) : au-delà de 1,5 s sans voix, on n'est plus dans une phrase


class Gate:
    """Probabilités de voix par trame de 10 ms → masque 0/1 (hystérésis, maintien, ouverture anticipée)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.on = False
        self.hang = 0
        self.run_v = 0
        self.since_v = 1000
        self.run_b = 0

    def run(self, probs, db, conf, p, commit, click=None, breathy=None):
        """probs, db, conf : trames à décider (les « commit » premières sont définitives, l'état est
        gardé là). conf : confiance d'une hauteur de voix humaine (RMVPE, 60–600 Hz). click : trames
        de choc (attaque brutale). → masque (float32) de même longueur."""
        thr = p["sensitivity"]
        thr_on, thr_off = thr, max(0.05, thr - 0.15)
        release = max(1, int(round(p["release_ms"] / 10)))
        floor = p["floor_db"]
        hold = p.get("hold_voiced", True)
        breath = p.get("breath", True)
        whisper = breath and p.get("whisper", False)
        thr_u = max(0.12, thr - 0.17)               # sans hauteur (souffle, chuchotement) : plus bas
        if click is None:
            click = np.zeros(len(probs), bool)
        if breathy is None:
            breathy = np.zeros(len(probs), bool)
        shock = 0
        run_b = self.run_b
        on, hang, run_v, since_v = self.on, self.hang, self.run_v, self.since_v
        mask = np.zeros(len(probs), np.float32)
        for i, (pr, lv, c, bz) in enumerate(zip(probs, db, conf, breathy)):
            run_b = run_b + 1 if (bz and breath) else max(0, run_b - 4)
            voiced = c > 0.5 and lv >= floor + 6
            shock = 4 if click[i] else max(0, shock - 1)      # un choc et les 30 ms qui suivent
            run_v = run_v + 1 if voiced else 0
            since_v = 0 if voiced else since_v + 1
            if lv < floor:
                pr = 0.0
            # Souffles, respirations, chuchotements : seulement DANS une phrase (vraie voix il y a moins
            # de 1,5 s). Hors des phrases, rien sans hauteur de voix : c'est là que passaient les petits
            # bruits (frottements, souffles, raclements pris pour de la voix).
            phrase = since_v <= PHRASE
            # ouvrir : détection + vraie voix récente (ou détection très sûre)
            opens = (pr >= thr_on and (since_v <= 30 or pr >= 0.7)) or \
                (breath and phrase and not shock and (pr >= thr_u or run_b >= 15)) or                 (whisper and pr >= thr_u and not shock)
            keeps = on and (pr >= thr_off or (hold and voiced) or (phrase and run_b >= 3))
            if opens or keeps or (run_v >= 8 and pr >= 0.2):
                on, hang = True, release
            elif hang > 0:
                hang -= 1
            else:
                on = False
            mask[i] = on
            if i + 1 == commit:
                self.on, self.hang, self.run_v, self.since_v, self.run_b = on, hang, run_v, since_v, run_b
        return mask


def preroll(mask, frames):
    """Ouvre « frames » trames avant chaque début de parole."""
    if frames <= 0:
        return mask
    m = mask.copy()
    rise = np.flatnonzero((mask[1:] > 0) & (mask[:-1] == 0)) + 1
    for r in rise:
        m[max(0, r - frames):r] = 1
    return m


def smart_f0(f0, conf, mask, conf_thr, gap=4, min_run=3):
    """Hauteur propre : seulement dans la parole, trous courts comblés, détections isolées effacées."""
    v = (f0 > 0) & (conf > conf_thr) & (mask > 0)
    out = np.where(v, f0, 0.0)
    if not v.any():
        return out
    # Petites séquences voisées isolées (≤ 20 ms) : presque toujours un bruit pris pour une voyelle.
    edges = np.flatnonzero(np.diff(np.concatenate([[0], v.astype(np.int8), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < min_run:
            v[s:e] = False
    out = np.where(v, out, 0.0)
    # Sauts d'octave d'une seule trame : médiane sur 3 trames voisées.
    lf = np.log2(np.maximum(out, 1.0))
    med = signal.medfilt(lf, 3)
    jump = v & (np.abs(lf - med) > 0.6) & np.concatenate([[False], v[:-2] & v[2:], [False]])
    lf[jump] = med[jump]
    # Trous courts entre deux voyelles : interpolation (évite les craquements dans les voyelles).
    edges = np.flatnonzero(np.diff(np.concatenate([[1], v.astype(np.int8), [1]])))
    for s, e in zip(edges[::2], edges[1::2]):
        if 0 < s and e < len(v) and e - s <= gap:
            lf[s:e] = np.linspace(lf[s - 1], lf[e], e - s + 2)[1:-1]
            v[s:e] = True
    return np.where(v, 2.0 ** lf, 0.0)


def classic_f0(f0):
    """Comportement de RVC d'origine (pour comparer) : tous les trous comblés par interpolation."""
    v = f0 > 0
    if v.any() and not v.all():
        idx = np.arange(len(f0))
        f0 = f0.copy()
        f0[~v] = np.interp(idx[~v], idx[v], f0[v])
    return f0
