"""Serveur HTTPS local : le téléphone ouvre la page, envoie sa caméra et son micro par WebSocket."""
import asyncio
import datetime
import hashlib
import ipaddress
import json
import os
import queue
import secrets
import socket
import ssl
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
from aiohttp import WSMsgType, web

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
WEB_DIR = os.path.join(HERE, "web")


def lan_ip() -> str:
    # Connexion UDP « à blanc » : aucun paquet n'est envoyé, Windows indique juste
    # l'interface de la route par défaut (celle qui mène à la box).
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 9))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    if ip.startswith("169.254.") or ip == "127.0.0.1":
        # Pas de route par défaut : première adresse privée trouvée.
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            cand = info[4][0]
            if ipaddress.ip_address(cand).is_private and not cand.startswith(("169.254.", "127.")):
                return cand
    return ip


def ensure_certificate(ip: str) -> tuple[str, str]:
    """Certificat auto-signé local (le navigateur exige HTTPS pour accéder à la caméra)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    os.makedirs(DATA_DIR, exist_ok=True)
    cert_path = os.path.join(DATA_DIR, "cert.pem")
    key_path = os.path.join(DATA_DIR, "key.pem")
    ip_path = os.path.join(DATA_DIR, "cert_ip.txt")
    if os.path.isfile(cert_path) and os.path.isfile(key_path) and os.path.isfile(ip_path):
        with open(ip_path, encoding="utf-8") as f:
            if f.read().strip() == ip:
                return cert_path, key_path

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "PhoneCam local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address(ip)), x509.DNSName("localhost")]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(ip_path, "w", encoding="utf-8") as f:
        f.write(ip)
    return cert_path, key_path


def load_token() -> str:
    """Jeton gardé d'un lancement à l'autre : l'appli du téléphone se reconnecte toute seule."""
    path = os.path.join(DATA_DIR, "token.txt")
    try:
        with open(path, encoding="utf-8") as f:
            tok = f.read().strip()
        if len(tok) >= 16:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(18)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tok)
    return tok


def cert_fingerprint(cert_path: str) -> str:
    """Empreinte SHA-256 du certificat : l'appli n'accepte que celui-ci (épinglage)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    with open(cert_path, "rb") as f:
        der = x509.load_pem_x509_certificate(f.read()).public_bytes(serialization.Encoding.DER)
    return hashlib.sha256(der).hexdigest()


ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


MAX_BACKLOG = 45                # images en attente de décodage (1,5 s à 30 img/s)
MAX_STREAM = 12                 # images d'appel en attente d'envoi vers le téléphone


def log_error(where: str) -> None:
    """Écrit l'erreur dans data\\phonecam-errors.log : l'appli tourne sans console."""
    try:
        with open(os.path.join(DATA_DIR, "phonecam-errors.log"), "a", encoding="utf-8") as f:
            f.write(f"--- {time.strftime('%d/%m %H:%M:%S')} {where}\n{traceback.format_exc()}\n")
    except OSError:
        pass


class _Decoder:
    """Décode le flux du téléphone dans son propre thread (H.264 via FFmpeg/PyAV, ou JPEG)."""

    def __init__(self, on_frame, request_key):
        self._q: queue.Queue = queue.Queue()
        self._on_frame = on_frame
        self._request_key = request_key
        self._ctx = None
        self.ms = 0.0
        self.codec = "—"
        threading.Thread(target=self._run, daemon=True, name="decoder").start()

    def reset(self):
        self._drain()
        self._q.put(None)

    def _drain(self):
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def feed(self, data: bytes):
        self._q.put(data)

    def _new_h264(self):
        import av
        ctx = av.CodecContext.create("h264", "r")
        # Multithread par tranche seulement : le multithread par image ajoute des images de retard.
        ctx.thread_type = "SLICE"
        ctx.options = {"flags": "+low_delay"}
        return ctx

    def _run(self):
        # Ce fil ne doit jamais mourir : sans lui, l'image reste figée alors que le téléphone est connecté.
        while True:
            try:
                self._decode_loop()
            except Exception:
                log_error("décodeur vidéo")
                self._ctx = None
                time.sleep(0.05)

    def _decode_loop(self):
        import av
        waiting_key = True
        last_pub = 0.0
        while True:
            data = self._q.get()
            if self._q.qsize() > MAX_BACKLOG:   # trop de retard : on repart de la prochaine image clé
                self._drain()
                waiting_key = True
                self._request_key()
            if data is None:
                self._ctx, waiting_key = None, True
                continue
            kind, payload = data[0], data[1:]
            t0 = time.perf_counter()
            img = None
            if kind == 3:
                self.codec = "JPEG"
                img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            elif kind in (1, 2):
                self.codec = "H.264"
                if waiting_key and kind != 1:
                    continue  # impossible de décoder sans image clé : on attend la suivante
                if self._ctx is None or kind == 1 and waiting_key:
                    self._ctx = self._new_h264()
                waiting_key = False
                try:
                    for frame in self._ctx.decode(av.Packet(payload)):
                        img = frame.to_ndarray(format="bgr24")
                except (av.FFmpegError, ValueError):
                    waiting_key = True
                    self._request_key()
            if img is not None:
                self.ms = (time.perf_counter() - t0) * 1000
                # Si le décodage prend du retard, on ne publie que la dernière image (10 par seconde
                # au minimum : sinon une file jamais vide figerait la sortie).
                now = time.perf_counter()
                if self._q.qsize() == 0 or kind == 3 or now - last_pub > 0.1:
                    last_pub = now
                    self._on_frame(img)


class PhoneServer:
    """Reçoit le flux du téléphone ; garde seulement l'image la plus récente (latence minimale).

    Deux accès : Wi-Fi (HTTPS sur le réseau local) et câble USB (HTTP sur 127.0.0.1 seulement,
    relayé par « adb reverse » : rien n'est exposé sur le réseau).
    """

    def __init__(self, port: int = 8443, usb_port: int = 8444):
        self.port = port
        self.usb_port = usb_port
        self.ip = lan_ip()
        # Jeton aléatoire (gardé dans data\token.txt) : un autre appareil ne peut pas envoyer d'images.
        self.token = load_token()
        self.url = f"https://{self.ip}:{port}/?k={self.token}"
        self.usb_url = f"http://localhost:{usb_port}/?k={self.token}"
        self.cert = ensure_certificate(self.ip)
        fp = cert_fingerprint(self.cert[0])
        # Liens d'appairage de l'application Android (le Wi-Fi épingle le certificat du PC).
        self.app_url = f"phonecam://connect?h={self.ip}:{port}&s=1&k={self.token}&fp={fp}"
        self.app_usb_url = f"phonecam://connect?h=localhost:{usb_port}&s=0&k={self.token}"
        self.rotation = 0               # l'appli indique comment redresser l'image (degrés, sens horaire)
        self.source = "phone"           # « phone » ou « webcam » (webcam du PC, voir webcam.py)
        self.on_json = None             # appelé avec chaque commande JSON de l'appli (modes Navigation / Appel)
        self.on_disconnect = None
        self._out_lock = threading.Lock()
        self._out_busy: set = set()
        self._out_pending: dict = {}
        self._streams: dict = {}
        self._cv = threading.Condition()
        self._frame = None
        self._frame_t = 0.0
        self._frame_id = 0
        self._recv_times: list[float] = []
        self._ws = None
        self._loop = None
        self.connected = False
        self.link = ""
        self.error = None
        self.audio_sink = None          # appelé avec chaque bloc du micro (float32 mono 48 kHz)
        self.audio_rate = 0
        self._audio_t = 0.0
        self.decoder = _Decoder(self._publish, self._request_key)

    def _publish(self, img):
        if self.source != "phone":      # webcam du PC choisie : le téléphone connecté n'entre plus
            return
        if self.rotation in ROTATIONS:
            img = cv2.rotate(img, ROTATIONS[self.rotation])
        self._store(img)

    def publish_local(self, img):
        """Image de la webcam du PC (webcam.py)."""
        if self.source == "webcam":
            self._store(img)

    def _store(self, img):
        with self._cv:
            self._frame = img
            self._frame_t = time.perf_counter()
            self._frame_id += 1
            self._recv_times.append(self._frame_t)
            self._cv.notify_all()

    # --- canal retour PC → téléphone (écran de Snapchat Web, vidéos de l'appel, état)
    def send_binary(self, kind: int, payload: bytes) -> None:
        """Envoie au téléphone un message binaire (1er octet = kind). Thread-safe.
        Si le précédent du même type part encore, on ne garde que le plus récent (jamais de
        file qui s'allonge, et la dernière image arrive toujours)."""
        self._out_slot(kind, bytes([kind]) + payload, binary=True)

    def send_stream(self, kind: int, payload: bytes) -> bool:
        """Vidéo H.264 vers le téléphone : dans l'ordre, sans en perdre. Faux si la liaison ne suit
        plus : la file est vidée et il faut repartir d'une image clé."""
        ws, loop = self._ws, self._loop
        if ws is None or loop is None:
            return False
        data = bytes([kind]) + payload
        with self._out_lock:
            q = self._streams.get(kind)
            if q is not None and q[0] is ws:
                if len(q[1]) >= MAX_STREAM:
                    q[1].clear()
                    return False
                q[1].append(data)
                return True
            self._streams[kind] = (ws, deque([data]))

        async def go():
            while True:
                with self._out_lock:
                    q = self._streams.get(kind)
                    if q is None or q[0] is not ws:
                        return
                    if not q[1]:
                        del self._streams[kind]
                        return
                    item = q[1].popleft()
                try:
                    await ws.send_bytes(item)
                except Exception:
                    with self._out_lock:
                        if self._streams.get(kind, (None,))[0] is ws:
                            del self._streams[kind]
                    return
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(go()))
        return True

    def send_json(self, obj: dict, key: str = "json") -> None:
        self._out_slot(key, json.dumps(obj, ensure_ascii=False), binary=False)

    def _out_slot(self, key, data, binary: bool) -> None:
        ws, loop = self._ws, self._loop
        if ws is None or loop is None:
            return
        with self._out_lock:
            if key in self._out_busy:
                self._out_pending[key] = (ws, data, binary)
                return
            self._out_busy.add(key)

        async def go(ws, data, binary):
            while True:
                try:
                    await (ws.send_bytes(data) if binary else ws.send_str(data))
                except Exception:
                    pass
                with self._out_lock:
                    nxt = self._out_pending.pop(key, None)
                    if nxt is None or nxt[0] is not self._ws:
                        self._out_busy.discard(key)
                        return
                ws, data, binary = nxt
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(go(ws, data, binary)))

    def _request_key(self):
        ws, loop = self._ws, self._loop
        if ws is not None and loop is not None:
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(ws.send_str("key")))

    def _audio(self, data: bytes):
        self.audio_rate = int.from_bytes(data[1:5], "little")
        self._audio_t = time.perf_counter()
        sink = self.audio_sink
        if sink is not None and self.audio_rate == 48000:
            sink(np.frombuffer(data[5:], "<i2").astype(np.float32) / 32768)

    def audio_active(self) -> bool:
        """Le micro du téléphone a envoyé du son dans la dernière seconde."""
        return time.perf_counter() - self._audio_t < 1.0

    def wait_frame(self, last_id: int, timeout: float = 0.1):
        """Attend une image plus récente que last_id → (id, image, instant de réception)."""
        with self._cv:
            if self._frame_id == last_id:
                self._cv.wait(timeout)
            return self._frame_id, self._frame, self._frame_t

    def latest(self):
        with self._cv:
            return self._frame_id, self._frame

    def fps(self) -> float:
        with self._cv:
            now = time.perf_counter()
            self._recv_times = [t for t in self._recv_times if now - t < 1.0]
            return float(len(self._recv_times))

    async def _index(self, request: web.Request):
        if request.query.get("k") != self.token:
            raise web.HTTPForbidden(text="Lien invalide : scanne le QR code affiché sur le PC.")
        return web.FileResponse(os.path.join(WEB_DIR, "index.html"),
                                headers={"Cache-Control": "no-store"})

    async def _applink(self, request: web.Request):
        """Lien « ouvrir dans l'application » pour la page web (même accès que la page)."""
        if request.query.get("k") != self.token:
            raise web.HTTPForbidden()
        usb = request.host.startswith("localhost")
        return web.Response(text=self.app_usb_url if usb else self.app_url,
                            headers={"Cache-Control": "no-store"})

    async def _ws_handler(self, request: web.Request):
        if request.query.get("k") != self.token:
            raise web.HTTPForbidden()
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=6)
        await ws.prepare(request)
        if self._ws is not None:    # un seul téléphone à la fois : le plus récent gagne
            try:                    # une ancienne liaison morte ne doit pas bloquer la nouvelle
                await asyncio.wait_for(self._ws.close(), 2)
            except Exception:
                pass
        self._ws = ws
        self.rotation = 0
        self.decoder.reset()
        self.connected = True
        self.link = "Câble USB" if request.host.startswith("localhost") else "Wi-Fi"
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY and len(msg.data) > 5 and msg.data[0] == 4:
                    self._audio(msg.data)
                elif msg.type == WSMsgType.BINARY and len(msg.data) > 1:
                    self.decoder.feed(msg.data)
                    if msg.data[0] == 3:
                        await ws.send_str("ok")  # accusé JPEG (au plus 2 images en vol)
                elif msg.type == WSMsgType.TEXT and msg.data.startswith("{"):
                    cb = self.on_json
                    if cb is not None:
                        try:
                            cb(json.loads(msg.data))
                        except Exception as e:
                            print("commande de l'appli ignorée :", e)
                elif msg.type == WSMsgType.TEXT and msg.data.startswith("rot:"):
                    try:
                        self.rotation = int(msg.data[4:]) % 360
                    except ValueError:
                        pass
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            if self._ws is ws:
                self._ws = None
                self.connected = False
                if self.on_disconnect is not None:
                    self.on_disconnect()
        return ws

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="phone-server").start()

    def _run(self):
        try:
            cert, key = self.cert
            ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ctx.load_cert_chain(cert, key)
            app = web.Application()
            app.router.add_get("/", self._index)
            app.router.add_get("/ws", self._ws_handler)
            app.router.add_get("/applink", self._applink)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            runner = web.AppRunner(app, access_log=None)
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", self.usb_port).start())
            loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", self.port, ssl_context=ctx).start())
            loop.run_forever()
        except Exception as e:  # affiché dans l'interface
            self.error = str(e)
