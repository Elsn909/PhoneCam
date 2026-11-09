"""Appels Snapchat Web pilotés depuis l'appli du téléphone.

PhoneCam utilise son propre Edge (profil séparé dans data\\snap-browser : ton navigateur habituel
n'est pas touché) par le protocole de débogage de Chromium (CDP), en local. La page reste telle
quelle sur le PC : aucune taille n'est imposée, les chats s'y utilisent normalement.

Un petit script dans la page lit l'état de l'appel, envoie les deux vidéos et repère les boutons
(décrocher, refuser, raccrocher, caméra, filtres). L'appli envoie ses commandes, rejouées comme de
vrais clics. Rien n'est automatisé : seules les actions demandées depuis l'appli sont faites.
"""
import asyncio
import base64
import json
import os
import subprocess
import threading
import time

import aiohttp

EDGE_PATHS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe")]
PORT = 9333
HOME = "https://www.snapchat.com/web"
ORIGIN = "https://www.snapchat.com"

# Messages binaires vers l'appli (1er octet) : vidéo entrante, vidéo sortante, en H.264.
# Puis : 1 = image clé, largeur et hauteur sur 2 octets, image.
CALL_REMOTE, CALL_LOCAL = 21, 22
KINDS = {"R": CALL_REMOTE, "L": CALL_LOCAL}
AVATAR_SCALE = "scale=3"         # 640 px au lieu de 100 (les URL de Snapchat portent la taille)

PAGE_JS = r"""
(() => {
  const VERSION = 25;          // une version plus récente de ce script remplace celle déjà dans la page
  if (window.__pc && window.__pc.version === VERSION) return;
  if (window.__pc) window.__pc.pump(false);
  const LOCAL = /obs|virtual|virtuel|phonecam/i;
  const KEYS = {
    hangup: /raccroch|terminer l.appel|quitter l.appel|fin de l.appel|end call|leave call|hang ?up/i,
    answer: /^r[ée]pondre$|d[ée]crocher|rejoindre|accepter|^answer|join call|accept call/i,
    decline: /pas maintenant|not now|refuser|ignorer|d[ée]cliner|decline|ignore/i,
    camera: /activer\/d[ée]sactiver la cam[ée]ra|toggle camera/i,
    mic: /activer\/d[ée]sactiver le micro|toggle mic/i,
  };
  function videos() {
    const out = [];
    for (const v of document.querySelectorAll('video')) {
      // Seuls les flux en direct (appel) comptent, pas les vidéos de présentation du site.
      if (!v.srcObject || !v.videoWidth || v.readyState < 2) continue;
      const r = v.getBoundingClientRect();
      const t = v.srcObject.getVideoTracks()[0];
      // Avec les filtres Snapchat, ta propre vidéo sort d'un canvas.
      const local = !!t && (LOCAL.test(t.label) || t.constructor.name === 'CanvasCaptureMediaStreamTrack');
      out.push({ v, area: r.width * r.height, local });
    }
    return out;
  }
  function pick(local) {
    let best = null;
    for (const x of videos()) if (x.local === local && (!best || x.area > best.area)) best = x;
    return best && best.v;
  }
  function visible(el) {
    const r = el.getBoundingClientRect();
    return r.width > 4 && r.height > 4 && r.bottom > 0 && r.right > 0 &&
           r.top < innerHeight && r.left < innerWidth && getComputedStyle(el).visibility !== 'hidden';
  }
  function label(el) {
    return (el.getAttribute('aria-label') || el.getAttribute('title') || el.innerText || '').trim().slice(0, 60);
  }
  function find(kind) {
    const re = KEYS[kind];
    for (const el of document.querySelectorAll('button,[role=button],[aria-label],[title]')) {
      if (!visible(el)) continue;
      // « Répondre » existe aussi sur les snaps et dans les conversations : ce n'est pas un appel.
      if ((kind === 'answer' || kind === 'decline') && el.closest('ul,li,[role=list],[role=listitem]')) continue;
      const l = label(el);
      if (!l || !re.test(l)) continue;
      if (kind !== 'hangup' && KEYS.hangup.test(l)) continue;
      const r = el.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2, label: l };
    }
    return null;
  }
  function box(el) {
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2, l: r.left, t: r.top, w: r.width, h: r.height };
  }
  function rowEls() {
    const list = document.querySelector('[role=list][aria-label]');
    return list ? [...list.querySelectorAll('[role=listitem] [role=button]')].filter(visible) : [];
  }
  function rows() {
    return rowEls().map((b, i) => {
      const lines = b.innerText.split('\n').map(s => s.trim()).filter(s => s && !/^(·|Voir|View|Répondre|Reply|Fermer le Chat|Close Chat)$/.test(s));
      // L'émoji d'ami peut précéder le nom sur sa propre ligne.
      const n = Math.max(0, lines.findIndex(l => /[\p{L}\p{N}]/u.test(l)));
      return { i, name: lines[n] || '', sub: lines.slice(n + 1).join(' · ') };
    }).filter(r => r.name);
  }
  function closer() {
    const all = [...document.querySelectorAll('[title="Fermer le Chat"],[title="Close Chat"]')];
    return all.find(e => !e.closest('[role=listitem]') && visible(e)) || null;
  }
  function peer() {
    const c = closer();
    return c ? (c.parentElement.innerText.split('\n')[0] || '').trim() : '';
  }
  function timer() {     // appel décroché : un minuteur s'affiche sous le nom
    const c = closer();
    if (!c) return '';
    const m = c.parentElement.innerText.match(/\b\d{1,2}:\d{2}(:\d{2})?\b/);
    return m ? m[0] : '';
  }
  // Les deux vidéos : chaque image décodée par la page, réencodée en H.264 par la carte graphique
  // sur une copie de la piste (l'appel lui-même n'est pas touché).
  const streams = {};
  const RATE = { R: 0.16, L: 0.16 };     // bits par pixel
  function b64(u8) {
    let s = '';
    for (let i = 0; i < u8.length; i += 32768) s += String.fromCharCode.apply(null, u8.subarray(i, i + 32768));
    return btoa(s);
  }
  function stopStream(k) {
    const s = streams[k];
    if (!s) return;
    delete streams[k];
    s.off = true;
    try { s.reader.cancel(); } catch (e) {}
    try { s.track.stop(); } catch (e) {}
    try { if (s.enc && s.enc.state !== 'closed') s.enc.close(); } catch (e) {}
  }
  function encoder(k, s, w, h) {
    try { if (s.enc && s.enc.state !== 'closed') s.enc.close(); } catch (e) {}
    s.enc = new VideoEncoder({
      output: c => {
        if (s.off) return;
        const u = new Uint8Array(c.byteLength);
        c.copyTo(u);
        window.pcFrame(k + (c.type === 'key' ? '1' : '0') + w + ',' + h + ',' + b64(u));
      },
      error: () => { s.w = 0; },
    });
    s.enc.configure({ codec: 'avc1.42E033', width: w, height: h, framerate: 30, latencyMode: 'realtime',
                      bitrate: Math.round(Math.max(1.5e6, w * h * 30 * RATE[k] * s.q)),
                      hardwareAcceleration: 'prefer-hardware', avc: { format: 'annexb' } });
    s.w = w; s.h = h; s.key = true;
    s.canvas = new OffscreenCanvas(w, h);
    s.ctx = s.canvas.getContext('2d', { alpha: false });
  }
  function feed(k, s, f) {
    const w = f.displayWidth & ~1, h = f.displayHeight & ~1;
    if (!w || !h) return;
    if (w !== s.w || h !== s.h) encoder(k, s, w, h);
    if (s.enc.state !== 'configured' || s.enc.encodeQueueSize > 1) return;   // en retard : image sautée
    const key = s.key || ++s.n % 120 === 0;
    s.key = false;
    // Passage par le RVB : les images arrivent en plage complète ou limitée, 601 ou 709, sans le
    // dire au téléphone. Ainsi elles sortent toujours en BT.709 limité, comme il les décode.
    s.ctx.drawImage(f, 0, 0, w, h);
    const v = new VideoFrame(s.canvas, { timestamp: f.timestamp });
    try { s.enc.encode(v, { keyFrame: key }); } finally { v.close(); }
  }
  function startStream(k, track, q) {
    const clone = track.clone();
    const reader = new MediaStreamTrackProcessor({ track: clone, maxBufferSize: 2 }).readable.getReader();
    const s = streams[k] = { id: track.id, track: clone, reader, enc: null, w: 0, h: 0, n: 0, key: true, off: false, q };
    (async () => {
      while (!s.off) {
        const { value: f, done } = await reader.read();
        if (done) break;
        try { feed(k, s, f); } catch (e) {} finally { f.close(); }
      }
    })().catch(() => {}).finally(() => { if (streams[k] === s) stopStream(k); });
  }
  function syncStreams(q) {
    for (const [k, v] of [['R', pick(false)], ['L', pick(true)]]) {
      const t = v && v.srcObject.getVideoTracks()[0];
      if (streams[k] && (!t || streams[k].id !== t.id)) stopStream(k);
      if (t && !streams[k]) startStream(k, t, q);
    }
  }
  function lensEls() {
    const seen = new Set(), out = [];
    for (const b of document.querySelectorAll('button[title]')) {
      if (!b.querySelector('img') || !b.title || seen.has(b.title) || !b.getBoundingClientRect().width) continue;
      seen.add(b.title);
      out.push(b);
    }
    return out;
  }
  let timer_ = null;
  window.__pc = {
    version: VERSION,
    find,
    rows,
    size() { return [innerWidth, innerHeight]; },
    isOpen(name) { return peer() === name; },
    row(i) { const b = rowEls()[i]; return b ? box(b) : null; },
    avatar(i) { const b = rowEls()[i], img = b && b.querySelector('img'); return img ? img.src : null; },
    key(k) { if (streams[k]) streams[k].key = true; },
    dial(video) {
      const bs = [...document.querySelectorAll('button')].filter(visible);
      const i = bs.findIndex(b => /^(appeler|call)$/i.test(b.innerText.trim()));
      const b = i < 0 ? null : bs[video ? i : i + 1];
      return b ? box(b) : null;
    },
    micArrow() {   // la petite flèche à côté du bouton micro
      const m = find('mic');
      if (!m) return null;
      let best = null, dist = 40;
      for (const b of document.querySelectorAll('button')) {
        const r = b.getBoundingClientRect();
        if (!r.width || r.width > 22 || b.title) continue;
        const d = Math.hypot(r.left + r.width / 2 - m.x, r.top + r.height / 2 - m.y);
        if (d < dist) { dist = d; best = b; }
      }
      return best && box(best);
    },
    micOptions() {  // le menu ouvert : le micro actif est celui qui porte la coche
      return [...document.querySelectorAll('[role=option]')].filter(visible).map(o => ({
        name: o.innerText.trim().split('\n').join(' '), on: !!o.querySelector('svg'), ...box(o) }));
    },
    lenses() { return lensEls().map(b => ({ name: b.title, src: b.querySelector('img').src })); },
    lens(name) {
      const b = lensEls().find(x => x.title === name);
      if (!b) return null;
      b.scrollIntoView({ block: 'nearest', inline: 'center' });
      return box(b);
    },
    lensOff() {   // la petite croix posée sur le filtre choisi, jamais les menus caméra et micro
      const ls = lensEls().map(b => b.getBoundingClientRect());
      const x = [...document.querySelectorAll('button')].find(b => {
        if (b.title || b.querySelector('img') || b.innerText.trim() || !visible(b)) return false;
        const r = b.getBoundingClientRect();
        return r.width <= 26 && ls.some(l => r.right > l.left && r.left < l.right && r.bottom > l.top && r.top < l.bottom);
      });
      return x ? box(x) : null;
    },
    streams() { return Object.keys(streams).map(k => [k, streams[k].w, streams[k].h]); },
    state() {
      const remote = pick(false), local = pick(true);
      const hang = find('hangup'), ans = find('answer'), t = timer();
      return { remote: !!remote, local: !!local, hangup: !!hang, answer: !!ans, incoming: !!ans && !hang,
               inCall: !!hang, live: !!hang && (!!t || !!remote), time: t, peer: peer(), camOff: !!hang && !local,
               buttons: [...document.querySelectorAll('button,[role=button]')].filter(visible)
                          .map(label).filter(Boolean).slice(0, 30) };
    },
    pump(on, q) {
      clearInterval(timer_); timer_ = null;
      stopStream('R'); stopStream('L');
      if (!on || !window.pcFrame) return;
      timer_ = setInterval(() => { try { syncStreams(q || 1); } catch (e) {} }, 300);
    },
  };
})();
"""


def find_browser() -> str | None:
    return next((p for p in EDGE_PATHS if os.path.isfile(p)), None)


class SnapBridge:
    # Débit vidéo selon le niveau d'économie du téléphone (0, 1, 2).
    CALL_QUALITY = (1, 0.7, 0.5)

    def __init__(self, server, data_dir: str):
        self.server = server
        self.profile = os.path.join(data_dir, "snap-browser")
        self.status = "Snapchat Web fermé."
        self.on_output = None           # (action, valeur) : micro coupé, pause, volume côté PhoneCam
        self.out_state = {"paused": False, "black": False, "muted": False}
        self.mic_running = lambda: False  # le micro de PhoneCam tourne-t-il ?
        self.mic_gain = lambda: 0         # volume de sortie du micro de PhoneCam, dB
        self.mode = "cam"
        self.pc_wanted: dict = {}       # réglages de PhoneCam demandés par l'appli, appliqués par l'interface
        self.pc_ask = False             # l'appli attend leurs valeurs
        self._proc = None
        self._ws = None
        self._session = None
        self._id = 0
        self._futures: dict = {}
        self._last_state = None
        self._last_extra = None
        self.received: dict = {}        # commandes reçues de l'appli, par type (diagnostic)
        self.eco = 0                    # niveau d'économie demandé par le téléphone (chaleur)
        self._need_key: set = set()
        self._rows_sent = None
        self._avatars: dict = {}
        self._thumbs: dict = {}
        self._lens_sent = 0
        self._lens_busy = False
        self._mic_direct = False
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="snapweb").start()
        self._submit(self._poll_loop())
        server.on_json = lambda m: self._submit(self._handle(m))
        server.on_disconnect = lambda: self._submit(self._set_mode("cam"))

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ------------------------------------------------------------ navigateur
    def open(self) -> None:
        self._submit(self._ensure())

    def close(self) -> None:
        """Se détache de la page sans fermer Edge : les chats continuent sur le PC."""
        fut = self._submit(self._close())
        try:
            fut.result(3)
        except Exception:
            pass

    async def _close(self):
        if self._ws is not None:
            try:
                await self._eval("window.__pc && window.__pc.pump(false)")
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self.status = "Détaché de Snapchat Web."

    async def _ensure(self) -> bool:
        """Lance Edge si besoin et se connecte à l'onglet Snapchat. Vrai si prêt."""
        if self._ws is not None and not self._ws.closed:
            return True
        if self._session is None:
            self._session = aiohttp.ClientSession()
        tabs = await self._tabs()
        if tabs is None:
            exe = find_browser()
            if exe is None:
                self.status = "Edge introuvable."
                return False
            os.makedirs(self.profile, exist_ok=True)
            self.status = "Ouverture de Snapchat Web…"
            self._proc = subprocess.Popen([
                exe, f"--user-data-dir={self.profile}", f"--remote-debugging-port={PORT}",
                "--no-first-run", "--no-default-browser-check",
                # La fenêtre peut être cachée derrière PhoneCam : ne pas ralentir la page.
                "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows", "--autoplay-policy=no-user-gesture-required",
                "--window-size=1000,900", HOME])
            for _ in range(60):
                await asyncio.sleep(0.25)
                tabs = await self._tabs()
                if tabs:
                    break
        if not tabs:
            self.status = "Snapchat Web ne répond pas."
            return False
        private = await self._private_targets()
        tabs = [t for t in tabs if t.get("id") not in private]
        page = next((t for t in tabs if t.get("type") == "page" and "snapchat" in t.get("url", "")),
                    next((t for t in tabs if t.get("type") == "page"), None))
        if page is None:
            self.status = "Aucun onglet ouvert dans Snapchat Web."
            return False
        self._ws = await self._session.ws_connect(page["webSocketDebuggerUrl"], max_msg_size=64 << 20)
        asyncio.ensure_future(self._reader(self._ws))
        await self._cmd("Page.enable")
        await self._cmd("Runtime.enable")
        await self._cmd("Runtime.addBinding", name="pcFrame")
        await self._cmd("Page.addScriptToEvaluateOnNewDocument", source=PAGE_JS)
        await self._cmd("Runtime.evaluate", expression=PAGE_JS)
        await self._cmd("Emulation.clearDeviceMetricsOverride")     # la page garde la taille de sa fenêtre
        await self._cmd("Emulation.setFocusEmulationEnabled", enabled=True)
        try:   # caméra et micro autorisés pour Snapchat (sinon une fenêtre de demande sur le PC)
            await self._cmd("Browser.grantPermissions", origin=ORIGIN,
                            permissions=["videoCapture", "audioCapture", "notifications"])
        except Exception:
            pass
        self.status = "Snapchat Web ouvert."
        if "snapchat" not in page.get("url", ""):
            await self._cmd("Page.navigate", url=HOME)
        await self._apply_mode()
        return True

    async def _private_targets(self):
        """Onglets des fenêtres privées : jamais pilotés, ce sont des usages à part sur le PC."""
        try:
            async with self._session.get(f"http://127.0.0.1:{PORT}/json/version") as r:
                url = (await r.json())["webSocketDebuggerUrl"]
            async with self._session.ws_connect(url, max_msg_size=16 << 20) as ws:
                async def ask(i, method):
                    await ws.send_json({"id": i, "method": method})
                    async for msg in ws:
                        d = json.loads(msg.data)
                        if d.get("id") == i:
                            return d.get("result", {})
                default = (await ask(1, "Target.getBrowserContexts")).get("defaultBrowserContextId")
                targets = (await ask(2, "Target.getTargets")).get("targetInfos", [])
            return {t["targetId"] for t in targets if default and t.get("browserContextId") != default}
        except Exception:
            return set()

    async def _tabs(self):
        try:
            async with self._session.get(f"http://127.0.0.1:{PORT}/json", timeout=aiohttp.ClientTimeout(total=1)) as r:
                return await r.json()
        except Exception:
            return None

    async def _cmd(self, method, timeout=5, **params):
        ws = self._ws
        if ws is None:
            raise RuntimeError("Snapchat Web non connecté")
        self._id += 1
        i = self._id
        fut = self._loop.create_future()
        self._futures[i] = fut
        await ws.send_json({"id": i, "method": method, "params": params})
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._futures.pop(i, None)

    async def _eval(self, expr):
        r = await self._cmd("Runtime.evaluate", expression=expr, returnByValue=True)
        return (r.get("result") or {}).get("value")

    async def _value(self, expr):
        s = await self._eval(f"JSON.stringify({expr})")
        return json.loads(s) if s else None

    async def _reader(self, ws):
        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                if "id" in m:
                    fut = self._futures.get(m["id"])
                    if fut is not None and not fut.done():
                        fut.set_result(m.get("result", {}))
                    continue
                meth, prm = m.get("method"), m.get("params", {})
                if meth == "Runtime.bindingCalled" and prm.get("name") == "pcFrame":
                    p = prm.get("payload", "")
                    if self.mode == "call" and len(p) > 2:
                        self._video(p)
                elif meth == "Page.frameNavigated" and not prm.get("frame", {}).get("parentId"):
                    asyncio.ensure_future(self._apply_mode())   # nouvelle page : on relance le mode
        except Exception:
            pass
        if self._ws is ws:
            self._ws = None
            self.status = "Snapchat Web fermé."

    def _video(self, p):
        """Une image H.264 de la page → le téléphone. Si la liaison décroche, on saute jusqu'à une image clé."""
        k, key = p[0], p[1] == "1"
        if k in self._need_key and not key:
            return
        w, h, data = p[2:].split(",", 2)
        head = bytes([key]) + int(w).to_bytes(2, "big") + int(h).to_bytes(2, "big")
        if self.server.send_stream(KINDS[k], head + base64.b64decode(data)):
            self._need_key.discard(k)
        else:
            self._need_key.add(k)
            asyncio.ensure_future(self._eval(f"window.__pc.key('{k}')"))

    # ------------------------------------------------------------ modes
    async def _set_mode(self, mode):
        self.mode = mode
        self._last_state = None
        self._rows_sent = None
        self._avatars = {}
        # Snapchat reste ouvert tant que le téléphone est connecté : un appel est vu depuis n'importe quel onglet.
        if self.server.connected and not await self._ensure():
            self.server.send_json({"t": "web", "status": self.status})
            return
        await self._apply_mode()

    async def _mic_menu(self, name=None):
        """Le menu de la petite flèche du micro de Snapchat : ouvert le temps de le lire, ou d'y
        choisir name, exactement comme à la souris. Il n'existe que pendant un appel."""
        await self._usable_window()
        arrow = await self._value("window.__pc.micArrow()")
        if not arrow:
            self.server.send_json({"t": "mics", "items": [], "current": ""}, key="mics")
            return
        await self._click(arrow["x"], arrow["y"])
        await asyncio.sleep(0.3)
        opts = await self._value("window.__pc.micOptions()") or []
        pick = next((o for o in opts if o["name"] == name), None)
        if pick and not pick["on"]:
            await self._click(pick["x"], pick["y"])
            current = name
        else:
            await self._click(arrow["x"], arrow["y"])
            current = next((o["name"] for o in opts if o["on"]), "")
        await asyncio.sleep(0.4)
        if await self._value("window.__pc.micOptions().length"):      # menu resté ouvert
            for kind in ("keyDown", "keyUp"):
                await self._cmd("Input.dispatchKeyEvent", type=kind, key="Escape", code="Escape",
                                windowsVirtualKeyCode=27)
        self._mic_direct = bool(current) and "VB-AUDIO" not in current.upper()
        self.server.send_json({"t": "mics", "items": [o["name"] for o in opts], "current": current}, key="mics")

    async def _apply_mode(self):
        if self._ws is None:
            return
        try:
            await self._eval(f"window.__pc && window.__pc.pump({'true' if self.mode == 'call' else 'false'}, "
                             f"{self.CALL_QUALITY[self.eco]})")
        except Exception as e:
            self.status = f"Snapchat Web : {e}"

    async def _usable_window(self):
        """Fenêtre Edge réduite : la page est mise en pause, ta vidéo se fige pour l'autre personne et
        les boutons ne répondent plus. La fenêtre reprend sa taille, sans rien imposer à la page."""
        info = await self._cmd("Browser.getWindowForTarget")
        if info.get("bounds", {}).get("windowState") == "minimized":
            await self._cmd("Browser.setWindowBounds", windowId=info["windowId"], bounds={"windowState": "normal"})
            await asyncio.sleep(0.5)

    async def _poll_loop(self):
        """État de l'appel et contacts, envoyés à l'appli quand ils changent."""
        while True:
            await asyncio.sleep(0.7)
            if self._ws is None or not self.server.connected:
                continue
            try:
                st = await self._value("window.__pc ? window.__pc.state() : null")
                if st:
                    if st["inCall"] or st["incoming"]:
                        await self._usable_window()
                    await self._push_state(st)
                    if not st["inCall"]:
                        await self._push_contacts()
                    elif not self._lens_busy:     # les filtres apparaissent après le début de l'appel
                        n = await self._value("window.__pc.lenses().length")
                        if n != self._lens_sent:
                            self._lens_busy = True
                            asyncio.ensure_future(self._send_lenses())
            except Exception:
                continue

    async def _push_state(self, st):
        keys = ("remote", "local", "hangup", "answer", "incoming", "inCall", "live", "peer", "camOff")
        short = {k: st[k] for k in keys}
        extra = {"time": st["time"], **self.out_state, "gain": self.mic_gain(), "micOn": self.mic_running()}
        if short != self._last_state or extra != self._last_extra:
            if short != self._last_state:
                self._log_call(st)
            if not short["inCall"]:
                self._lens_sent = 0
            self._last_state, self._last_extra = short, extra
            self.server.send_json({"t": "call", **short, **extra}, key="call")

    def _log_call(self, st):
        """Boutons vus à chaque changement d'état d'appel (data\\snap-call.log) : la page de Snapchat
        change, et c'est ce qui permet d'adapter la détection du décrochage."""
        try:
            with open(os.path.join(os.path.dirname(self.profile), "snap-call.log"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%H:%M:%S"), **st}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    async def _push_contacts(self):
        rows = await self._value("window.__pc.rows()") or []
        if rows != self._rows_sent:
            self._rows_sent = rows
            self.server.send_json({"t": "contacts", "rows": rows}, key="contacts")
            for r in rows:
                if r["name"] not in self._avatars:
                    self._avatars[r["name"]] = ""
                    asyncio.ensure_future(self._send_avatar(r))

    async def _send_avatar(self, r):
        try:
            src = await self._value(f"window.__pc.avatar({r['i']})")
            if not src or not src.startswith("http"):
                return
            src = src.replace("scale=0", AVATAR_SCALE)
            async with self._session.get(src, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                data = base64.b64encode(await resp.read()).decode()
            self._avatars[r["name"]] = data
            self.server.send_json({"t": "avatar", "name": r["name"], "b64": data}, key="avatar" + r["name"])
        except Exception:
            pass                               # sans avatar, la liste reste utilisable

    async def _send_lenses(self):
        """Filtres Snapchat proposés pendant l'appel, avec leurs vignettes."""
        try:
            items = (await self._value("window.__pc.lenses()") or [])
            self._lens_sent = len(items)
            for it in items[:30]:
                if it["name"] not in self._thumbs and it["src"].startswith("http"):
                    try:
                        async with self._session.get(it["src"], timeout=aiohttp.ClientTimeout(total=4)) as r:
                            self._thumbs[it["name"]] = base64.b64encode(await r.read()).decode()
                    except Exception:
                        self._thumbs[it["name"]] = ""
            self.server.send_json({"t": "lenses", "items": [{"name": i["name"], "b64": self._thumbs.get(i["name"], "")}
                                                            for i in items[:30]]}, key="lenses")
        finally:
            self._lens_busy = False

    # ------------------------------------------------------------ commandes de l'appli
    async def _handle(self, m):
        t = m.get("t")
        self.received[t] = self.received.get(t, 0) + 1
        try:
            if t == "mode":
                await self._set_mode(m.get("m", "cam"))
            elif t == "eco":
                self.eco = max(0, min(2, int(m.get("level", 0))))
                if self.mode == "call":
                    await self._apply_mode()
            elif t == "pc":                        # menu Paramètres de l'appli
                if "key" in m:
                    self.pc_wanted[m["key"]] = m.get("value")
                else:
                    self.pc_ask = True
            elif t == "gain" and self.on_output:   # volume de sortie du micro de PhoneCam
                self.on_output("gain", max(-24, min(12, int(m.get("db", 0)))))
            elif t == "key" and self._ws is not None:   # décodeur du téléphone prêt ou décroché
                await self._eval(f"window.__pc.key({json.dumps(m.get('k', 'R'))})")
            elif self._ws is None:
                return
            elif t == "dial":
                await self._dial(m.get("i", 0), bool(m.get("video")))
            elif t == "mics":                      # liste du menu micro de Snapchat, ou choix
                await self._mic_menu(m.get("name"))
            elif t == "lens":
                name = m.get("name")
                await self._usable_window()
                pos = await self._value(f"window.__pc.lens({json.dumps(name)})" if name else "window.__pc.lensOff()")
                if pos:
                    await self._click(pos["x"], pos["y"])
            elif t == "call":
                await self._call_action(m.get("a"))
        except Exception as e:
            self.server.send_json({"t": "web", "status": f"Erreur : {e}"})

    async def _dial(self, i, video):
        """Ouvre la conversation sur le PC (comme à la souris) et lance l'appel."""
        await self._usable_window()
        rows = await self._value("window.__pc.rows()") or []
        name = next((r["name"] for r in rows if r["i"] == int(i)), None)
        if name is None:
            self.server.send_json({"t": "web", "status": "Contact introuvable."})
            return
        if not await self._value(f"window.__pc.isOpen({json.dumps(name)})"):
            pos = await self._value(f"window.__pc.row({int(i)})")
            await self._click(pos["x"], pos["y"])
            await asyncio.sleep(1.0)
        pos = await self._value(f"window.__pc.dial({'true' if video else 'false'})")
        if pos:
            await self._click(pos["x"], pos["y"])
        else:
            self.server.send_json({"t": "web", "status": "Bouton d'appel introuvable sur la page."})

    async def _click(self, x, y):
        await self._cmd("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        await self._cmd("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left",
                        buttons=1, clickCount=1)
        await self._cmd("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left",
                        buttons=0, clickCount=1)

    async def _call_action(self, a):
        """Boutons de l'appel : ceux de Snapchat. Le micro passe par PhoneCam, c'est là qu'on le coupe."""
        await self._usable_window()
        if a == "mic":
            self._set_out("muted", not self.out_state["muted"])
            # Un micro branché directement dans Snapchat ne passe pas par PhoneCam.
            if self.mic_running() and not self._mic_direct:
                return
        pos = await self._value(f"window.__pc.find({json.dumps(a)})")
        if pos:
            await self._click(pos["x"], pos["y"])
        else:
            self.server.send_json({"t": "web", "status": "Bouton introuvable sur la page."})

    def _set_out(self, key, value):
        self.out_state[key] = value
        if self.on_output:
            self.on_output(key, value)
