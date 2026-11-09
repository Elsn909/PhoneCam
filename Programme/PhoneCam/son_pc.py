"""Son du PC vers le téléphone : tout ce que le PC joue (haut-parleurs par défaut) part en direct vers
l'appli PhoneCam du téléphone, qui le joue avec son propre réglage de volume.

Capture « loopback » de Windows (WASAPI, mode partagé) sur la sortie par défaut : exactement ce que tu
entends, sans câble virtuel ni pilote en plus (comtypes, déjà présent). Si tu changes de sortie par
défaut, la capture suit toute seule.
Envoi : PCM 16 bits stéréo 48 kHz par paquets de 20 ms (message binaire de type 30), dans l'ordre,
sur la même connexion que la caméra (câble USB ou Wi-Fi). ~190 ko/s. Rien n'est capté ni envoyé tant
que l'appli ne l'a pas demandé (case « 🔊 Son du PC »).
"""
import ctypes
import threading
import time
from ctypes import POINTER, c_int64, c_uint64, c_void_p
from ctypes.wintypes import DWORD, HANDLE, LPWSTR, UINT, WORD

import numpy as np

KIND = 30                      # type du message binaire vers le téléphone
SR = 48000
CHUNK = 960                    # 20 ms

try:
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

    class WAVEFORMATEX(ctypes.Structure):
        _pack_ = 1                      # comme dans mmreg.h (sinon le reste du format est décalé)
        _fields_ = [("wFormatTag", WORD), ("nChannels", WORD), ("nSamplesPerSec", DWORD),
                    ("nAvgBytesPerSec", DWORD), ("nBlockAlign", WORD), ("wBitsPerSample", WORD), ("cbSize", WORD)]

    class WAVEFORMATEXTENSIBLE(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("Format", WAVEFORMATEX), ("Samples", WORD), ("dwChannelMask", DWORD), ("SubFormat", GUID)]

    class IAudioCaptureClient(IUnknown):
        _iid_ = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
        _methods_ = [
            COMMETHOD([], HRESULT, "GetBuffer", (["out"], POINTER(c_void_p), "ppData"),
                      (["out"], POINTER(UINT), "pNumFramesToRead"), (["out"], POINTER(DWORD), "pdwFlags"),
                      (["out"], POINTER(c_uint64), "pu64DevicePosition"), (["out"], POINTER(c_uint64), "pu64QPCPosition")),
            COMMETHOD([], HRESULT, "ReleaseBuffer", (["in"], UINT, "NumFramesRead")),
            COMMETHOD([], HRESULT, "GetNextPacketSize", (["out"], POINTER(UINT), "pNumFramesInNextPacket")),
        ]

    class IAudioClient(IUnknown):
        _iid_ = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
        _methods_ = [
            COMMETHOD([], HRESULT, "Initialize", (["in"], DWORD, "ShareMode"), (["in"], DWORD, "StreamFlags"),
                      (["in"], c_int64, "hnsBufferDuration"), (["in"], c_int64, "hnsPeriodicity"),
                      (["in"], POINTER(WAVEFORMATEX), "pFormat"), (["in"], POINTER(GUID), "AudioSessionGuid")),
            COMMETHOD([], HRESULT, "GetBufferSize", (["out"], POINTER(UINT), "pNumBufferFrames")),
            COMMETHOD([], HRESULT, "GetStreamLatency", (["out"], POINTER(c_int64), "phnsLatency")),
            COMMETHOD([], HRESULT, "GetCurrentPadding", (["out"], POINTER(UINT), "pNumPaddingFrames")),
            COMMETHOD([], HRESULT, "IsFormatSupported", (["in"], DWORD, "ShareMode"), (["in"], POINTER(WAVEFORMATEX), "pFormat"),
                      (["out"], POINTER(POINTER(WAVEFORMATEX)), "ppClosestMatch")),
            COMMETHOD([], HRESULT, "GetMixFormat", (["out"], POINTER(POINTER(WAVEFORMATEX)), "ppDeviceFormat")),
            COMMETHOD([], HRESULT, "GetDevicePeriod", (["out"], POINTER(c_int64), "phnsDefaultDevicePeriod"),
                      (["out"], POINTER(c_int64), "phnsMinimumDevicePeriod")),
            COMMETHOD([], HRESULT, "Start"),
            COMMETHOD([], HRESULT, "Stop"),
            COMMETHOD([], HRESULT, "Reset"),
            COMMETHOD([], HRESULT, "SetEventHandle", (["in"], HANDLE, "eventHandle")),
            COMMETHOD([], HRESULT, "GetService", (["in"], POINTER(GUID), "riid"), (["out"], POINTER(c_void_p), "ppv")),
        ]

    class IMMDevice(IUnknown):
        _iid_ = GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
        _methods_ = [
            COMMETHOD([], HRESULT, "Activate", (["in"], POINTER(GUID), "iid"), (["in"], DWORD, "dwClsCtx"),
                      (["in"], c_void_p, "pActivationParams"), (["out"], POINTER(c_void_p), "ppInterface")),
            COMMETHOD([], HRESULT, "OpenPropertyStore", (["in"], DWORD, "stgmAccess"), (["out"], POINTER(c_void_p), "ppProperties")),
            COMMETHOD([], HRESULT, "GetId", (["out"], POINTER(LPWSTR), "ppstrId")),
            COMMETHOD([], HRESULT, "GetState", (["out"], POINTER(DWORD), "pdwState")),
        ]

    class IMMDeviceEnumerator(IUnknown):
        _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        _methods_ = [
            COMMETHOD([], HRESULT, "EnumAudioEndpoints", (["in"], DWORD, "dataFlow"), (["in"], DWORD, "dwStateMask"),
                      (["out"], POINTER(c_void_p), "ppDevices")),
            COMMETHOD([], HRESULT, "GetDefaultAudioEndpoint", (["in"], DWORD, "dataFlow"), (["in"], DWORD, "role"),
                      (["out"], POINTER(POINTER(IMMDevice)), "ppEndpoint")),
        ]

    CLSID_ENUM = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
    FLOAT = GUID("{00000003-0000-0010-8000-00AA00389B71}")
    AVAILABLE = True
except Exception:  # noqa: BLE001
    AVAILABLE = False

LOOPBACK = 0x00020000
SILENT = 0x2
CLSCTX_ALL = 23


class _Capture:
    """Une capture loopback ouverte sur la sortie par défaut du moment."""

    def __init__(self):
        enum = comtypes.CoCreateInstance(CLSID_ENUM, IMMDeviceEnumerator, comtypes.CLSCTX_ALL)
        self.enum = enum
        dev = enum.GetDefaultAudioEndpoint(0, 0)            # eRender, eConsole
        self.dev_id = self._id(dev)
        p = dev.Activate(ctypes.byref(IAudioClient._iid_), CLSCTX_ALL, None)
        self.client = ctypes.cast(p, POINTER(IAudioClient))
        fmt = self.client.GetMixFormat()
        f = fmt.contents
        self.rate, self.ch, self.align, self.bits = f.nSamplesPerSec, f.nChannels, f.nBlockAlign, f.wBitsPerSample
        self.is_float = f.wFormatTag == 3
        if f.wFormatTag == 0xFFFE:
            ext = ctypes.cast(fmt, POINTER(WAVEFORMATEXTENSIBLE)).contents
            self.is_float = ext.SubFormat == FLOAT
        self.client.Initialize(0, LOOPBACK, 1_000_000, 0, fmt, None)     # partagé, tampon 100 ms
        ctypes.windll.ole32.CoTaskMemFree(fmt)
        p = self.client.GetService(ctypes.byref(IAudioCaptureClient._iid_))
        self.cap = ctypes.cast(p, POINTER(IAudioCaptureClient))
        self.client.Start()

    @staticmethod
    def _id(dev):
        s = dev.GetId()
        return str(s)

    def default_changed(self):
        try:
            return self._id(self.enum.GetDefaultAudioEndpoint(0, 0)) != self.dev_id
        except Exception:  # noqa: BLE001
            return True

    def read(self):
        """→ (frames, canaux) float32, éventuellement vide."""
        out = []
        n = self.cap.GetNextPacketSize()
        while n:
            data, frames, flags, _p, _q = self.cap.GetBuffer()
            nbytes = frames * self.align
            if flags & SILENT or not data:
                a = np.zeros((frames, self.ch), np.float32)
            else:
                raw = ctypes.string_at(data, nbytes)
                if self.is_float and self.bits == 32:
                    a = np.frombuffer(raw, "<f4").reshape(frames, self.ch)
                elif self.bits == 16:
                    a = np.frombuffer(raw, "<i2").reshape(frames, self.ch).astype(np.float32) / 32768
                elif self.bits == 32:
                    a = np.frombuffer(raw, "<i4").reshape(frames, self.ch).astype(np.float32) / 2 ** 31
                else:
                    a = np.zeros((frames, self.ch), np.float32)
            self.cap.ReleaseBuffer(frames)
            out.append(a.copy())
            n = self.cap.GetNextPacketSize()
        return np.concatenate(out) if out else np.zeros((0, self.ch), np.float32)

    def close(self):
        try:
            self.client.Stop()
        except Exception:  # noqa: BLE001
            pass


class PcSound:
    """Envoie le son du PC au téléphone tant que l'appli le demande (server.pc_audio_wanted)."""

    def __init__(self, server):
        self.server = server
        self.allowed = True             # case du PC (page Connexion)
        self.status = "Arrêté."
        self.sent = 0                   # paquets envoyés
        self.level_db = -120.0
        self._stop = threading.Event()
        self._pos = 0.0                 # rééchantillonnage (si la sortie n'est pas à 48 kHz)
        self._last = np.zeros(2, np.float32)
        threading.Thread(target=self._loop, daemon=True, name="son-pc").start()

    def stop(self):
        self._stop.set()

    def active(self):
        return bool(self.allowed and AVAILABLE and self.server.connected and getattr(self.server, "pc_audio_wanted", False))

    def _resample(self, a, rate):
        if rate == SR or not len(a):
            return a
        step = rate / SR                # échantillons d'entrée par échantillon de sortie (linéaire, avec état)
        src = np.vstack([self._last[None], a])
        t = np.arange(self._pos, len(a), step)
        i = np.floor(t).astype(int)
        f = (t - i)[:, None]
        out = src[i] * (1 - f) + src[i + 1] * f
        self._pos = t[-1] + step - len(a) if len(t) else self._pos - len(a)
        self._last = a[-1]
        return out.astype(np.float32)

    def _loop(self):
        if not AVAILABLE:
            self.status = "Indisponible (comtypes absent)."
            return
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        except OSError:                 # déjà initialisé pour ce fil
            pass
        cap, acc, t_check = None, np.zeros((0, 2), np.float32), 0.0
        while not self._stop.is_set():
            if not self.active():
                if cap is not None:
                    cap.close()
                    cap = None
                acc = acc[:0]
                self.status = ("En attente : « 🔊 Son du PC » dans l'appli du téléphone." if self.allowed else "Coupé.")
                self.level_db = -120.0
                time.sleep(0.2)
                continue
            try:
                if cap is None or (time.perf_counter() - t_check > 2 and cap.default_changed()):
                    if cap is not None:
                        cap.close()
                    cap = _Capture()
                    self._pos, self._last = 0.0, np.zeros(2, np.float32)
                    self.status = f"En cours : son du PC → téléphone ({cap.rate // 1000} kHz)."
                if time.perf_counter() - t_check > 2:
                    t_check = time.perf_counter()
                a = cap.read()
            except Exception as e:  # noqa: BLE001 (sortie débranchée, changée…) : on rouvre
                self.status = f"Capture du son du PC impossible : {e}"
                cap = None
                time.sleep(1.0)
                continue
            if len(a):
                if cap.ch == 1:
                    a = np.repeat(a, 2, axis=1)
                a = self._resample(a[:, :2], cap.rate)
                acc = np.concatenate([acc, a])
                while len(acc) >= CHUNK:
                    blk, acc = acc[:CHUNK], acc[CHUNK:]
                    pk = float(np.abs(blk).max())
                    self.level_db = 20 * np.log10(pk + 1e-9)
                    pcm = (np.clip(blk, -1, 1) * 32767).astype("<i2").tobytes()
                    if self.server.send_stream(KIND, pcm):
                        self.sent += 1
                if len(acc) > CHUNK * 10:           # ne jamais accumuler de retard
                    acc = acc[-CHUNK:]
            time.sleep(0.005)
        if cap is not None:
            cap.close()
