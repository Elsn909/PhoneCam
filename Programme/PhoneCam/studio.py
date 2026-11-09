"""Studio : vidéos importées (1080p à 4K, tout ce que lit FFmpeg), sans contrainte de temps réel.

Deux passes :
 1. Analyse : chaque image est analysée à pleine résolution. Le détecteur est complété par une
    recherche ciblée là où un visage vient d'être perdu, les plans sont séparés, les visages sont
    suivis et regroupés par personne. Comme toute la vidéo est connue, les repères sont lissés
    dans les deux sens du temps : aucun retard, et les sauts d'une image sont gommés.
 2. Rendu : mêmes modèles que le direct, en qualité maximale (pixel boost selon la taille du visage),
    puis filtres de forme, étalonnage, et encodage FFmpeg avec le son d'origine.
"""
import copy
import hashlib
import json
import os
import pickle
import queue
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from effects import DEFAULTS as FX_DEFAULTS, Effects
import reframe
from landmarks import hrffa_68, to_5
from look import Look
from models_plus import MODELS_DIR
from reshape import DEFAULTS as SHAPE_DEFAULTS, FaceReshape
from studio_conf import CODECS, ENGINE_DEFAULTS, IMAGE_EXT, QUALITIES, TRACK_DEFAULTS
from swap_plus import DEFAULTS as PLUS_DEFAULTS, OneEuro, SwapPlus

CACHE_DIR = os.path.join(os.path.dirname(MODELS_DIR), "studio-cache")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Qualité d'analyse et de rendu (remplace les réglages du moteur Plus).
QUALITY = dict(PLUS_DEFAULTS, boost=0, parser=True, occlusion=True, expression=70,
               stabilize=0, profile_smooth=0, hold=False, detail=80)
QUALITY.update(ENGINE_DEFAULTS)
CLASSIC_LOCK = threading.Lock()   # le moteur d'origine lit ses réglages dans des variables globales


PROXY_CODEC = ("h264", "mp4", ["-c:v", "h264_nvenc", "-preset", "p2", "-rc", "vbr", "-b:v", "0", "-cq", "{q}"], "yuv420p")
BIG_FACE_SIZE = 512       # taille d'analyse de secours, pour les visages qui remplissent l'image


def find_ffmpeg():
    return "ffmpeg", "ffprobe"


def read_image(path):
    """Orientation EXIF comprise ; chemins accentués acceptés."""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def probe(path):
    """Taille (rotation comprise), cadence, durée, audio, HDR. Une image compte pour une vidéo d'une image."""
    if os.path.splitext(path)[1].lower() in IMAGE_EXT:
        img = read_image(path)
        if img is None:
            raise ValueError("image illisible")
        h, w = img.shape[:2]
        return dict(path=path, width=w, height=h, fps=25.0, fps_text="25/1", duration=0.04, frames=1,
                    codec=os.path.splitext(path)[1][1:].upper(), pix_fmt="", audio=None, hdr=False, matrix="",
                    image=True)
    out = subprocess.run([find_ffmpeg()[1], "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                         creationflags=NO_WINDOW)
    if out.returncode:
        raise ValueError(out.stderr.strip() or "fichier illisible")
    info = json.loads(out.stdout)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise ValueError("aucune piste vidéo")
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)

    def rate(text):
        try:
            n, d = text.split("/")
            return float(n) / float(d) if float(d) else 0.0
        except (AttributeError, ValueError):
            return 0.0
    w, h = int(v["width"]), int(v["height"])
    rot = int(v.get("tags", {}).get("rotate", 0) or 0)
    for sd in v.get("side_data_list", []):
        rot = int(sd.get("rotation", rot) or 0)
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps_text = v.get("avg_frame_rate") if rate(v.get("avg_frame_rate")) else v.get("r_frame_rate")
    fps = rate(fps_text) or 30.0
    duration = float(info.get("format", {}).get("duration") or v.get("duration") or 0)
    return dict(path=path, width=w, height=h, fps=fps, fps_text=fps_text, duration=duration,
                frames=int(round(duration * fps)), codec=v.get("codec_name", "?"), pix_fmt=v.get("pix_fmt", ""),
                audio=a.get("codec_name") if a else None,
                hdr=v.get("color_transfer") in ("smpte2084", "arib-std-b67"),
                matrix=v.get("color_space") or "")


def _in_filter(info, width=None):
    """Conversion vers BGR 8 bits : matrice de couleur de la source, HDR ramené en SDR."""
    parts = []
    if info["hdr"]:
        parts.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
                     "zscale=t=bt709:m=bt709:r=tv")
    matrix = info["matrix"] if info["matrix"] in ("bt709", "bt601", "smpte170m", "bt470bg", "bt2020nc") else \
        ("bt709" if info["height"] >= 720 else "smpte170m")
    matrix = {"bt2020nc": "bt2020", "smpte170m": "bt601", "bt470bg": "bt601"}.get(matrix, matrix)
    scale = f"scale={width}:-2:flags=area:" if width else "scale=flags=accurate_rnd+full_chroma_int:"
    parts.append(f"{scale}in_color_matrix={matrix}")
    return ",".join(parts)


class Reader:
    """Images BGR de la vidéo, entre start et end (secondes), à cadence constante."""

    def __init__(self, info, start=0.0, end=None, width=None):
        self.w, self.h = info["width"], info["height"]
        if width:
            self.w, self.h = width, int(round(info["height"] * width / info["width"] / 2)) * 2
        self.p, self.still = None, None
        if info.get("image"):
            self.still = read_image(info["path"])
            if width:
                self.still = cv2.resize(self.still, (self.w, self.h), interpolation=cv2.INTER_AREA)
            return
        cmd = [find_ffmpeg()[0], "-v", "error", "-nostdin"]
        if start:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", info["path"]]
        if end:
            cmd += ["-t", f"{end - start:.6f}"]
        cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", _in_filter(info, width),
                "-fps_mode", "cfr", "-r", info["fps_text"], "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  bufsize=self.w * self.h * 3, creationflags=NO_WINDOW)

    def read(self):
        if self.p is None:
            img, self.still = self.still, None
            return img
        size = self.w * self.h * 3
        buf = bytearray(size)
        view, got = memoryview(buf), 0
        while got < size:
            n = self.p.stdout.readinto(view[got:])
            if not n:
                return None
            got += n
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        if self.p is None:
            return
        try:
            self.p.kill()
            self.p.stdout.close()
            self.p.wait(timeout=2)
        except Exception:  # noqa: BLE001
            pass


class Writer:
    def __init__(self, info, out_path, codec, quality, height, start, end, size):
        _family, _ext, args, pix = PROXY_CODEC if codec is None else CODECS[codec]
        w, h = size
        tw, th = w, h
        if height and height != min(w, h):      # le petit côté : 1080 × 1920 reste « 1080p »
            s = height / min(w, h)
            tw, th = reframe.even(w * s), reframe.even(h * s)
        ff = find_ffmpeg()[0]
        vf = f"scale={tw}:{th}:flags=lanczos+accurate_rnd+full_chroma_int:out_color_matrix=bt709:out_range=tv,format={pix}"
        cmd = [ff, "-y", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", info["fps_text"], "-i", "pipe:0"]
        ext = os.path.splitext(out_path)[1].lower()
        if info["audio"]:
            if start:
                cmd += ["-ss", f"{start:.6f}"]
            cmd += ["-i", info["path"]]
            if end:
                cmd += ["-t", f"{end - start:.6f}"]
            cmd += ["-map", "0:v:0", "-map", "1:a:0?"]
            compatible = ext == ".mkv" or (ext == ".mov" and info["audio"] in ("aac", "pcm_s16le", "pcm_s24le", "alac"))\
                or (ext == ".mp4" and info["audio"] in ("aac", "mp3", "ac3", "eac3", "opus", "alac"))
            cmd += ["-c:a", "copy"] if compatible else ["-c:a", "aac", "-b:a", "320k"]
        cmd += ["-vf", vf] + [a.format(q=quality) for a in args]
        cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
        if ext in (".mp4", ".mov"):
            cmd += ["-movflags", "+faststart"]
        cmd += ["-shortest", out_path] if info["audio"] else [out_path]
        self.err = os.path.join(CACHE_DIR, "encode.log")
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._log = open(self.err, "wb")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log, bufsize=w * h * 3,
                                  creationflags=NO_WINDOW)

    def write(self, img):
        self.p.stdin.write(np.ascontiguousarray(img).data)

    def close(self):
        self.p.stdin.close()
        code = self.p.wait()
        self._log.close()
        if code:
            with open(self.err, "rb") as f:
                raise RuntimeError(f.read()[-400:].decode("utf-8", "replace").strip() or f"FFmpeg : erreur {code}")

    def kill(self):
        try:
            self.p.kill()
            self.p.wait(timeout=2)
            self._log.close()
        except Exception:  # noqa: BLE001
            pass


class Models:
    """Modèles du Studio, séparés du direct (autres réglages de détection et de qualité)."""

    def __init__(self, log=lambda _m: None):
        self.log = log
        self.sp = None
        self.det = None
        self.det_size = None
        self.det_big = None
        self.rec = None
        self.seg, self.h3d = None, None

    def load(self, det_size=1024):
        import models_plus
        import trt_insight
        from insightface.model_zoo import get_model
        home = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.sp is None:
            if models_plus.missing():
                models_plus.download(log=self.log)
            models_plus.prepare_parser(log=self.log)
            models_plus.prepare_labels(log=self.log)
            self.log("Préparation des modèles…")
            self.sp = SwapPlus(MODELS_DIR)
            self.sp.p.update(QUALITY)
            self.sp.warmup()
            if "hrffa" in models_plus.OPTIONAL:
                models_plus.download_optional("hrffa", log=self.log)
            self.sp.session("hrffa")
            self.rec = get_model(os.path.join(home, "w600k_r50.onnx"), providers=providers)
            self.rec.prepare(0)
        if self.det is None or self.det_size != det_size:
            self.det = get_model(os.path.join(home, "det_10g.onnx"), providers=providers)
            self.det.prepare(0, input_size=(det_size, det_size), det_thresh=0.3)
            self.det_size = det_size
            trt_insight.upgrade(self.det, (1, 3, det_size, det_size), "studio-det")
        if self.det_big is None:
            self.det_big = get_model(os.path.join(home, "det_10g.onnx"), providers=providers)
            self.det_big.prepare(0, input_size=(BIG_FACE_SIZE, BIG_FACE_SIZE), det_thresh=0.3)
            trt_insight.upgrade(self.det_big, (1, 3, BIG_FACE_SIZE, BIG_FACE_SIZE), "studio-det-big")
        return self

    def shape_models(self):
        """Détourage et repères 3D des filtres de forme (chargés en arrière-plan à la 1re demande)."""
        if self.seg is None:
            from head3d import Head3D
            from segment import PersonSegmenter
            self.seg, self.h3d = PersonSegmenter(MODELS_DIR), Head3D()
            for _ in range(600):
                self.seg.mask(np.zeros((64, 64, 3), np.uint8))
                self.h3d.add(np.zeros((64, 64, 3), np.uint8), [])
                if (self.seg._session is not None or not self.seg.available())                         and (self.h3d._model is not None or not self.h3d.available()):
                    break
                time.sleep(0.1)
        return self.seg, self.h3d

    def detect(self, frame, thresh):
        """Le détecteur a une plage de tailles : réglé pour les petits visages, il ne voit plus un visage
        qui remplit l'image (très gros plan, portrait). Il est alors repris sur l'image réduite."""
        for det in (self.det, self.det_big):
            bboxes, kpss = det.detect(frame, max_num=0, metric="default")
            if len(bboxes):
                keep = bboxes[:, 4] >= thresh
                if keep.any():
                    return bboxes[keep], kpss[keep]
        return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)

    def source(self, paths):
        """Empreinte moyenne de photos de la même personne ; None si aucun visage."""
        from insightface.app.common import Face
        faces = []
        for path in paths:
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) if os.path.isfile(path) else None
            if img is None:
                continue
            bb, kp = self.detect(img, 0.3)
            if not len(bb):
                continue
            i = int(np.argmax((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])))
            f = Face(bbox=bb[i, :4], kps=kp[i], det_score=bb[i, 4])
            self.rec.get(img, f)
            faces.append(f)
        if not faces:
            return None
        ref = faces[0]
        embs = [f.normed_embedding for f in faces
                if float(np.dot(f.normed_embedding, ref.normed_embedding)) >= 0.35]
        mean = np.mean(embs, 0)
        ref.embedding = (mean / np.linalg.norm(mean) * float(np.linalg.norm(ref["embedding"]))).astype(np.float32)
        return ref


class Track:
    def __init__(self, tid, shot):
        self.id, self.shot = tid, shot
        self.frames, self.bbox, self.pts, self.score = [], [], [], []
        self.kps5 = []                 # 5 points du détecteur (repères « détecteur »)
        self.embs = []
        self.thumb, self.thumb_q = None, -1.0
        self.person = None
        self.smooth = None             # (n, 68, 2) après finish()
        self.smooth5 = None            # (n, 5, 2) après finish()
        self.first = 0
        self.embedding = None
        self.first_emb = 0

    def add(self, i, bbox, pts, score, kps=None):
        self.kps5.append(to_5(pts) if kps is None else np.asarray(kps, np.float32))
        self.frames.append(i)
        self.bbox.append(np.asarray(bbox[:4], np.float32))
        self.pts.append(pts)
        self.score.append(float(score))

    @property
    def last(self):
        return self.frames[-1]


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _smooth_both_ways(pts, strength, fps):
    """Lissage sans retard : filtre One Euro dans les deux sens du temps, moyenné. Les pointes d'une
    seule image sont retirées avant (médiane sur 3 images)."""
    n = len(pts)
    if n < 3 or strength <= 0:
        return pts.copy()
    med = np.median(np.stack([np.r_[pts[:1], pts[:-1]], pts, np.r_[pts[1:], pts[-1:]]]), 0)
    # Une médiane écrête aussi les vrais mouvements : on ne garde que l'écart anormal.
    jump = np.linalg.norm(pts - med, axis=-1, keepdims=True)
    size = np.linalg.norm(pts[:, 36:42].mean(1) - pts[:, 42:48].mean(1), axis=-1)[:, None, None]
    a = np.clip((jump / np.maximum(size, 1) - 0.03) / 0.05, 0, 1)
    clean = pts + (med - pts) * a
    s = strength / 100
    mc, beta = float(np.interp(s, [0, 1], [4.0, 0.5])), float(np.interp(s, [0, 1], [3.0, 0.4]))
    norm = float(np.median(size)) or 1.0

    def run(seq):
        f = OneEuro(min_cutoff=mc, beta=beta / norm)
        return np.stack([f(p, t / fps) for t, p in enumerate(seq)])
    return ((run(clean) + run(clean[::-1])[::-1]) / 2).astype(np.float32)


class Analysis:
    """Résultat de l'analyse : visages suivis, plans, personnes."""

    def __init__(self, info, start, end, frames):
        self.info, self.start, self.end, self.frames = info, start, end, frames
        self.tracks, self.persons, self.cuts = [], [], []
        self.index = {}                # image → [(piste, position)]

    def finish(self, smooth):
        self.index = {}
        for t in self.tracks:
            pts = np.stack(t.pts)
            t.smooth = _smooth_both_ways(pts, smooth, self.info["fps"])
            # Le détecteur reçoit la même correction que les repères HRFFA.
            t.smooth5 = np.stack(t.kps5) + np.stack([to_5(p) for p in t.smooth]) - np.stack([to_5(p) for p in pts])
            t.first = t.frames[0]
            for k, i in enumerate(t.frames):
                self.index.setdefault(i, []).append((t, k))

    def sub(self, start, end):
        """Partie de la vidéo (couper le début et la fin) : l'analyse complète resservant, couper ne
        demande aucun nouveau calcul. Les repères restent ceux lissés sur toute la vidéo."""
        fps = self.info["fps"]
        i0 = int(round(start * fps))
        i1 = self.frames if end is None else min(self.frames, int(round(end * fps)))
        out = Analysis(self.info, start, end, max(0, i1 - i0))
        out.persons = self.persons
        out.cuts = [c - i0 for c in self.cuts if i0 < c < i1]
        for t in self.tracks:
            f0, f1 = max(t.frames[0], i0), min(t.frames[-1] + 1, i1)
            if f0 >= f1:
                continue
            k0, k1 = f0 - t.frames[0], f1 - t.frames[0]
            c = copy.copy(t)
            c.frames = [f - i0 for f in t.frames[k0:k1]]
            c.bbox, c.pts, c.score, c.smooth = t.bbox[k0:k1], t.pts[k0:k1], t.score[k0:k1], t.smooth[k0:k1]
            c.kps5, c.smooth5 = t.kps5[k0:k1], t.smooth5[k0:k1]
            out.tracks.append(c)
            for k, i in enumerate(c.frames):
                out.index.setdefault(i, []).append((c, k))
        return out

    def faces(self, i, person_filter=None):
        out = []
        for t, k in self.index.get(i, []):
            if person_filter is None or t.person in person_filter:
                out.append((t, k))
        return out


def analyse(models, info, start=0.0, end=None, params=None, progress=lambda *_: None, cancel=lambda: False):
    p = dict(TRACK_DEFAULTS, **(params or {}))
    if info.get("image"):
        p["min_len"] = 0
    fps = info["fps"]
    total = int(round(((end or info["duration"]) - start) * fps)) or 1
    reader = Reader(info, start, end)
    hr = models.sp.session("hrffa")
    thresh = p["det_score"] / 100
    tracks, active = [], []
    cuts, prev_small = [], None
    shot, i, t0 = 0, 0, time.perf_counter()
    q = queue.Queue(8)                 # le décodage 4K tourne à part

    def feed():
        while True:
            f = reader.read()
            q.put(f)
            if f is None:
                return
    threading.Thread(target=feed, daemon=True, name="studio-read").start()
    try:
        while True:
            frame = q.get()
            if frame is None:
                break
            if cancel():
                return None
            small = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if prev_small is not None and np.abs(small.astype(np.int16) - prev_small).mean() > 28:
                shot += 1                      # changement de plan : on ne suit pas au travers
                cuts.append(i)
                active = []
            prev_small = small.astype(np.int16)
            bb, kp = models.detect(frame, thresh)
            used, matched = set(), set()
            order = sorted(((_iou(tr.bbox[-1], bb[j, :4]), a, j) for a, tr in enumerate(active)
                            for j in range(len(bb))), reverse=True)
            pairs = []
            for iou, a, j in order:
                if iou < 0.15 or a in matched or j in used:
                    continue
                matched.add(a)
                used.add(j)
                pairs.append((active[a], j))
            for tr, j in pairs:
                _add(models, tr, i, frame, bb[j], kp[j], hr)
            for a, tr in enumerate(active):
                if a not in matched and p["rescue"]:
                    _rescue(models, tr, i, frame, hr, thresh)
            for j in range(len(bb)):
                if j not in used and bb[j, 4] >= 0.5 and not any(_iou(tr.bbox[-1], bb[j, :4]) > 0.3 for tr in active):
                    tr = Track(len(tracks), shot)
                    tracks.append(tr)
                    active.append(tr)
                    _add(models, tr, i, frame, bb[j], kp[j], hr)
            active = [tr for tr in active if i - tr.last <= p["gap"] * fps]
            i += 1
            if i % 5 == 0:
                done = time.perf_counter() - t0
                progress(min(i / total, 1.0), f"Analyse : image {i} / {total} ({i / done:.1f} img/s)")
    finally:
        reader.close()
        _drain(q)
    a = Analysis(info, start, end, i)
    a.cuts = cuts
    a.tracks = _clean(tracks, p, fps)
    a.persons = _group(a.tracks, i)
    a.finish(p["smooth"])
    return a


def _drain(q):
    """Débloque le fil de lecture quand on s'arrête avant la fin de la vidéo."""
    def run():
        while q.get() is not None:
            pass
    threading.Thread(target=run, daemon=True).start()


def _add(models, tr, i, frame, bbox, kps, hr):
    from insightface.app.common import Face
    pts = hrffa_68(hr, frame, bbox[:4])
    tr.add(i, bbox, pts, bbox[4], kps)
    size = float(bbox[2] - bbox[0]) * float(bbox[3] - bbox[1])
    q = float(bbox[4]) * size
    if bbox[4] >= 0.55 and (not tr.embs or (i - tr.first_emb) % 12 == 0):
        f = Face(bbox=bbox[:4], kps=kps, det_score=bbox[4])
        models.rec.get(frame, f)
        if not tr.embs:
            tr.first_emb = i
        tr.embs.append(f.normed_embedding)
    if q > tr.thumb_q:
        tr.thumb_q = q
        h, w = frame.shape[:2]
        cx, cy, r = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.75
        x0, y0, x1, y1 = int(max(0, cx - r)), int(max(0, cy - r)), int(min(w, cx + r)), int(min(h, cy + r))
        if x1 > x0 and y1 > y0:
            tr.thumb = cv2.resize(frame[y0:y1, x0:x1], (112, 112), interpolation=cv2.INTER_AREA)


def _rescue(models, tr, i, frame, hr, thresh):
    """Visage perdu par la détection globale : recherche autour de sa dernière position (le visage y
    est bien plus grand pour le détecteur), sinon repères seuls sur la dernière position, peu de temps."""
    if i - tr.last > 12:
        return
    b = tr.bbox[-1]
    h, w = frame.shape[:2]
    cx, cy, r = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) * 1.4
    x0, y0, x1, y1 = int(max(0, cx - r)), int(max(0, cy - r)), int(min(w, cx + r)), int(min(h, cy + r))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return
    bb, kp = models.detect(frame[y0:y1, x0:x1], max(0.2, thresh - 0.15))
    if len(bb):
        j = int(np.argmax(bb[:, 4]))
        box = bb[j].copy()
        box[[0, 2]] += x0
        box[[1, 3]] += y0
        _add(models, tr, i, frame, box, kp[j] + (x0, y0), hr)
    elif i - tr.last <= 2:
        tr.add(i, b, hrffa_68(hr, frame, b), 0.2)


def _clean(tracks, p, fps):
    """Comble les trous de chaque piste, retire les fausses détections très brèves."""
    keep = []
    for t in tracks:
        if len(t.frames) < p["min_len"] * fps or max(t.score) < 0.5:
            continue
        frames = list(range(t.frames[0], t.frames[-1] + 1))
        if len(frames) != len(t.frames):
            idx = np.array(t.frames)
            pts, bbox = np.stack(t.pts), np.stack(t.bbox)
            fill = lambda arr: np.stack([np.interp(frames, idx, c) for c in arr.reshape(len(idx), -1).T], 1)\
                .reshape(len(frames), *arr.shape[1:]).astype(np.float32)
            t.pts, t.bbox = list(fill(pts)), list(fill(bbox))
            t.score = list(np.interp(frames, idx, t.score))
            t.frames = frames
        keep.append(t)
    return keep


def _group(tracks, frames):
    """Regroupe les pistes par personne (empreintes moyennes), les plus présentes d'abord."""
    persons = []
    for t in sorted(tracks, key=lambda t: -len(t.frames)):
        emb = None
        if t.embs:
            emb = np.mean(t.embs, 0)
            emb = emb / np.linalg.norm(emb)
            t.embedding = emb
        best, score = None, 0.42
        for pr in persons:
            if emb is not None and pr["emb"] is not None and float(emb @ pr["emb"]) > score:
                best, score = pr, float(emb @ pr["emb"])
        if best is None:
            best = dict(id=len(persons), emb=emb, thumb=t.thumb, frames=0, tracks=[], q=t.thumb_q)
            persons.append(best)
        best["tracks"].append(t.id)
        best["frames"] += len(t.frames)
        t.person = best["id"]
        if t.thumb_q > best["q"]:
            best["thumb"], best["q"] = t.thumb, t.thumb_q
    persons.sort(key=lambda pr: -pr["frames"])
    remap = {pr["id"]: n for n, pr in enumerate(persons)}
    for pr in persons:
        pr["id"] = remap[pr["id"]]
    for t in tracks:
        t.person = remap[t.person]
    return persons


def cache_path(info, start, end, params):
    st = os.stat(info["path"])
    key = json.dumps(["kps5", os.path.abspath(info["path"]), st.st_size, st.st_mtime, start, end,
                      {k: v for k, v in dict(TRACK_DEFAULTS, **(params or {})).items() if k != "smooth"}, 2])
    return os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest()[:20] + ".pkl")


def load_cached(info, start, end, params):
    path = cache_path(info, start, end, params)
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                a = pickle.load(f)
            a.info = info
            a.finish(dict(TRACK_DEFAULTS, **(params or {}))["smooth"])
            return a
        except Exception:  # noqa: BLE001 — analyse à refaire
            pass
    return None


def save_cached(a, params):
    os.makedirs(CACHE_DIR, exist_ok=True)
    a.index = {}
    with open(cache_path(a.info, a.start, a.end, params), "wb") as f:
        pickle.dump(a, f)
    a.finish(dict(TRACK_DEFAULTS, **(params or {}))["smooth"])


def auto_boost(size):
    """Pixel boost selon la taille du visage dans l'image : le modèle de 256 px recomposé en 512, 768
    ou 1024 px, pour ne jamais agrandir le visage généré."""
    return 1 if size < 200 else 2 if size < 420 else 3 if size < 650 else 4


class TrackedLandmarks:
    """Pour les filtres de forme : les repères HRFFA viennent du suivi, déjà posés sur chaque visage."""

    @staticmethod
    def ready():
        return True

    @staticmethod
    def add(_frame, _faces):
        pass


class Render:
    """Chaîne de rendu d'une image : remplacement, amélioration, filtres de forme, étalonnage."""

    def __init__(self, models, quality=None, shape=None, look=None, enhancer=None, sources=None, fx=None, frame=None,
                 swap=True):
        """swap=False : visage d'origine gardé, seulement retouche, cadrage, effets et étalonnage. models peut
        alors être None quand rien ne demande les visages (ni retouche, ni cadre qui suit le visage)."""
        self.m = models
        self.swap = swap
        self.q = dict(QUALITY, **(quality or {}))
        self.sp = models.sp if models is not None else None
        if self.sp is not None:
            self.sp.p.update({k: v for k, v in self.q.items() if k in self.sp.p and k != "boost"})
            self.sp._color, self.sp._fixed, self.sp._occ = {}, set(), {}
        self.boost = int(self.q.get("boost") or 0)
        self.classic = self.q["engine"] == "classic"
        self._prev = None                   # lissage temporel : image précédente
        self._last = None                   # dernière image calculée seule (aperçu)
        self._ready_models()
        self.shape = FaceReshape()
        self.shape.hrffa = TrackedLandmarks()
        self.fx = Effects()
        self.fx.ref_width = 1280            # mêmes effets que le direct, quelle que soit la résolution
        self.look = Look()
        self._freeze, self._next = 0.0, 0.0
        self.frame = dict(reframe.DEFAULTS)
        self.a = None
        self.set_post(shape or {}, fx or {}, look or {}, frame or {})
        self.enhancer = enhancer if swap else None      # (module, force) : GFPGAN / GPEN
        self.sources = (sources or {}) if swap else {}  # personne → visage source

    def _ready_models(self):
        """Modèles facultatifs du réglage choisi, attendus ici : en direct le rendu continue sans
        eux, mais une vidéo exportée doit les avoir dès la première image."""
        if not self.swap:
            return
        import models_plus
        from modules.processors.frame import face_swapper
        if self.classic:
            face_swapper.get_face_swapper()
            return
        if self.q["model"] == "inswapper":
            self.sp.inswapper = face_swapper.get_face_swapper()
        name = self.q["occluder"]
        if self.q["occlusion"] and name in models_plus.OPTIONAL:
            models_plus.download_optional(name, log=self.m.log)
            self.sp.session(name)
        if set(self.q["regions"]) != set(ENGINE_DEFAULTS["regions"]):
            self.sp.session("bisenet_labels_256")

    def set_post(self, shape, fx, look, frame=None):
        """Réglages de la fin de chaîne : changés sans refaire le remplacement du visage."""
        self.shape.p.update({k: v for k, v in shape.items() if k in SHAPE_DEFAULTS})
        self.shape.p["smooth"] = 0                       # déjà lissé sur toute la vidéo
        self.shape.p["hrffa"] = 1
        self.use_shape = self.shape.active()
        if self.use_shape and self.m is not None:
            self.shape.segmenter, self.shape.head3d = self.m.shape_models()
        self.fx.p.update({k: v for k, v in fx.items() if k in FX_DEFAULTS})
        self.look.p.update(look)
        self.frame.update(frame or {})

    def hold(self, t):
        """Cadence maximale et micro-gels des effets webcam : l'image précédente est-elle répétée ?"""
        p = self.fx.p
        if t < self._freeze:
            return True
        if p["freeze"] > 0 and random.random() < p["freeze"] / 1000:
            self._freeze = t + random.uniform(0.15, 0.6)
            return True
        step = 1 / max(1, p["fps"])
        if t < self._next - step * 0.25:
            return True
        self._next = (self._next if t - self._next < step else t) + step
        return False

    def faces_at(self, a, i):
        from insightface.app.common import Face
        faces, pairs = [], []
        for t, k in a.index.get(i, []):
            src = self.sources.get(t.person)
            pts = t.smooth[k]
            box = t.bbox[k]
            kps = t.smooth5[k] if self.q["landmarker"] == "detector" else to_5(pts)
            f = Face(bbox=np.asarray(box, np.float32), kps=kps, det_score=float(t.score[k]), track_id=t.id)
            f["landmark_hrffa_68"] = pts
            faces.append(f)
            if src is not None:
                f["fixed_kps"] = f.kps
                if t.embedding is not None:
                    f.embedding = t.embedding
                pairs.append((src, f))
        return faces, pairs

    def _set_boost(self, pairs):
        if self.boost:
            self.sp.p["boost"] = self.boost
        elif pairs:
            eye = max(float(np.linalg.norm(t.kps[1] - t.kps[0])) for _, t in pairs)
            self.sp.p["boost"] = auto_boost(eye * 256 / (0.42 * 256))    # taille du cadre de 256 px dans l'image

    def pre(self, frame):
        """« Corriger les couleurs » : comme en direct, avant le remplacement."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.swap and self.q["color_correction"] else frame

    def begin(self, frame, pairs):
        """Première partie du remplacement, dans l'ordre des images."""
        if not pairs:
            return None
        self._set_boost(pairs)
        return None if self.classic or not pairs else self.sp.prepare(frame, pairs)

    def run(self, frame, pairs, job):
        """Fin du remplacement : deux images peuvent se chevaucher avec le moteur Plus."""
        if not pairs:
            return frame
        out = self._classic(frame, pairs) if self.classic else self.sp.render(frame, job)[0]
        k = self.q["opacity"] / 100
        return cv2.addWeighted(out, k, frame, 1 - k, 0) if k < 1 else out

    def _classic(self, frame, pairs):
        """Moteur d'origine. Ses réglages sont des variables globales, communes avec le direct : ils
        ne sont posés que le temps de cette image."""
        import modules.globals as G
        from engine import ensure_landmarks
        from modules.processors.frame import face_swapper
        q = self.q
        options = dict(opacity=1.0, poisson_blend=bool(q["poisson_blend"]), mouth_mask=q["mouth_mask_size"] > 0,
                       mouth_mask_size=float(q["mouth_mask_size"]), mask_feather_ratio=q["mask_feather_ratio"],
                       mask_down_size=q["mask_down_size"] / 100, mask_size=q["mask_size"] / 100)
        with CLASSIC_LOCK:
            if options["mouth_mask"]:
                ensure_landmarks(frame, [t for _, t in pairs])
            kept = {k: getattr(G, k, None) for k in options}
            try:
                for k, v in options.items():
                    setattr(G, k, v)
                out = frame.copy()
                for src, target in pairs:
                    out = face_swapper.swap_face(src, target, out)
            finally:
                for k, v in kept.items():
                    setattr(G, k, v)
        return out

    def post(self, out, faces):
        """Netteté du visage et lissage temporel, mêmes calculs que le direct mais sans ses variables."""
        amount = self.q["sharpness"] / 10
        if amount > 0:
            from modules.gpu_processing import gpu_sharpen
            h, w = out.shape[:2]
            out = out.copy()
            for f in faces:
                if f.get("fixed_kps") is None:
                    continue
                x0, y0, x1, y1 = f.bbox.astype(int)
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                if x1 > x0 and y1 > y0:
                    out[y0:y1, x0:x1] = gpu_sharpen(out[y0:y1, x0:x1], strength=amount, sigma=3)
        force = self.q["interpolation_force"] / 100
        if self.swap and self.q["interpolation"] and force > 0:
            if self._prev is not None and self._prev.shape == out.shape:
                out = cv2.addWeighted(self._prev, force, out, 1 - force, 0)
            self._prev = out
        return out

    def enhance(self, out, faces):
        out = self.post(out, faces)
        if self.enhancer and faces:
            mod, force = self.enhancer
            enhanced = out
            for f in faces:
                enhanced = mod.process_frame(None, enhanced, detected_faces=[f])
            out = cv2.addWeighted(out, 1 - force, enhanced, force, 0) if force < 1 else enhanced
        return out

    def finish(self, out, faces, i=0):
        """Retouche, cadrage, effets webcam et ambiance, étalonnage : comme une caméra, le grain et le
        vignettage se font sur l'image cadrée."""
        if self.use_shape and faces:
            out = self.shape.apply(out, faces)
        out = reframe.apply(out, self.frame, self.a, i)
        if self.q["mirror"]:
            out = cv2.flip(out, 1)
        return self.look.apply(self.fx.apply(out))


def swap_frame(models, render, a, i, frame):
    """Une image jusqu'au visage remplacé (sans repères de suivi : détection sur place). Renvoie
    aussi les visages : la fin de chaîne peut être refaite seule (Render.finish)."""
    frame = render.pre(frame)
    render._prev = None                 # image isolée : rien à lisser avec la précédente
    if models is not None and (render._last is None or i != render._last + 1):
        models.sp._occ = {}             # masque stabilisé : seulement d'une image à la suivante
    render._last = i
    if a is not None and i in a.index:
        faces, pairs = render.faces_at(a, i)
    elif models is None:                # visage d'origine sans retouche : aucun visage à chercher
        faces, pairs = [], []
    else:
        from insightface.app.common import Face
        faces, pairs = [], []
        bb, kp = models.detect(frame, 0.35)
        hr = models.sp.session("hrffa")
        src = next(iter(render.sources.values()), None)
        for j in range(len(bb)):
            pts = hrffa_68(hr, frame, bb[j, :4])
            f = Face(bbox=bb[j, :4], kps=to_5(pts), det_score=float(bb[j, 4]), track_id=10000 + j)
            f["landmark_hrffa_68"] = pts
            f["fixed_kps"] = f.kps
            faces.append(f)
            if src is not None:
                pairs.append((src, f))
    return render.enhance(render.run(frame, pairs, render.begin(frame, pairs)), faces), faces


def preview(models, render, a, i, frame):
    out, faces = swap_frame(models, render, a, i, frame)
    render.a = a
    return render.finish(out, faces, i)


def export_image(models, render, a, out_path):
    """Image importée : même chaîne, enregistrée en PNG ou JPEG."""
    reader = Reader(a.info)
    out = preview(models, render, a, 0, reader.read())
    ext = os.path.splitext(out_path)[1].lower()
    ok, data = cv2.imencode(ext, out, [cv2.IMWRITE_JPEG_QUALITY, 96] if ext in (".jpg", ".jpeg") else [])
    if not ok:
        raise RuntimeError("enregistrement de l'image impossible")
    data.tofile(out_path)
    return out_path


def export(models, render, a, out_path, codec, quality, height, progress=lambda *_: None, cancel=lambda: False,
           on_frame=None):
    """Rendu complet. Écrit d'abord un fichier temporaire : l'original n'est jamais touché.
    on_frame(i, image) : pour montrer le calcul pendant qu'il avance."""
    info = a.info
    reader = Reader(info, a.start, a.end)
    root, ext = os.path.splitext(out_path)
    tmp = root + ".en-cours" + ext
    render.a = a
    writer = Writer(info, tmp, codec, QUALITIES.get(quality, quality), height, a.start, a.end,
                    reframe.output_size(render.frame, reader.w, reader.h))
    decoded = queue.Queue(6)
    pending = queue.Queue(4)
    pool = ThreadPoolExecutor(2, thread_name_prefix="studio-render")
    failure = []
    t0, done, last = time.perf_counter(), [0], [None]
    total = max(1, a.frames)

    def feed():
        while True:
            f = reader.read()
            decoded.put(f)
            if f is None:
                return

    def finish_loop():
        """Dans l'ordre des images : filtres, étalonnage, encodage."""
        try:
            while True:
                item = pending.get()
                if item is None:
                    return
                frame, fut, faces = item
                out = fut.result() if fut is not None else frame
                out = render.finish(render.enhance(out, faces), faces, done[0])
                if render.hold(done[0] / info["fps"]) and last[0] is not None:
                    out = last[0]
                last[0] = out
                writer.write(out)
                if on_frame is not None:
                    on_frame(done[0], out)
                done[0] += 1
                if done[0] % 3 == 0:
                    rate = done[0] / (time.perf_counter() - t0)
                    left = (total - done[0]) / max(rate, 0.01)
                    progress(min(done[0] / total, 1.0), f"Rendu : image {done[0]} / {total} · {rate:.1f} img/s · "
                                                        f"reste {int(left) // 60} min {int(left) % 60:02d} s")
        except Exception as e:  # noqa: BLE001
            failure.append(e)
            while pending.get() is not None:   # libère la boucle principale
                pass
    threading.Thread(target=feed, daemon=True, name="studio-read").start()
    finisher = threading.Thread(target=finish_loop, daemon=True, name="studio-write")
    finisher.start()
    i = 0
    ok = False
    try:
        while True:
            frame = decoded.get()
            if frame is None or cancel() or failure:
                break
            frame = render.pre(frame)
            faces, pairs = render.faces_at(a, i)
            job = render.begin(frame, pairs)
            fut = pool.submit(render.run, frame, pairs, job) if pairs else None
            pending.put((frame, fut, faces))
            i += 1
        pending.put(None)
        finisher.join()
        if failure:
            raise failure[0]
        if cancel():
            return None
        ok = True
        writer.close()
        os.replace(tmp, out_path)
        return out_path
    finally:
        reader.close()
        _drain(decoded)
        pool.shutdown(wait=False)
        if not ok:
            writer.kill()
            if os.path.isfile(tmp):
                os.remove(tmp)
