"""--==The SarahMemory Project==--
File: SarahMemoryNetwork.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-05
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
==============================================================================================================================================================
"""


from __future__ import annotations

import sys
import os
import logging



import socket, threading, time, queue, struct, json, zlib, uuid
from typing import Optional, Tuple, Dict, Callable, Any
import SarahMemoryGlobals as config
from SarahMemoryEncryption import SarahNetCrypto  # uses class embedded in SarahMemoryEncryption.py
from SarahMemorySMAPI import sm_api


# ========================= Protocol =========================
class NetworkProtocol:
    MAGIC   = b"SMNP"    # SarahMemory Net Protocol
    VERSION = 1

    # Flags
    F_COMPRESSED = 1 << 0
    F_URGENT     = 1 << 1
    F_ACK        = 1 << 2

    # Types (extensible)
    T_HELLO   = 1
    T_DATA    = 2
    T_ACK     = 3
    T_PING    = 4
    T_ALERT   = 5
    T_GOODBYE = 6

    _HDR = struct.Struct("!4sBBBHI16s")  # MAGIC(4) | ver(1) | flags(1) | type(1) | meta_len(2) | payload_len(4) | msg_id(16)

    @staticmethod
    def new_id() -> bytes:
        return uuid.uuid4().bytes

    @staticmethod
    def pack_message(mtype: int,
                     payload: bytes,
                     meta: Dict[str, str] = None,
                     flags: int = 0,
                     compress_if_over: int = 400) -> bytes:
        meta = meta or {}
        meta.setdefault("ts", str(int(time.time())))
        meta.setdefault("schema", "v1")
        meta_b = json.dumps(meta, separators=(",", ":")).encode("utf-8")

        if compress_if_over and len(payload) >= compress_if_over:
            payload = zlib.compress(payload, level=6)
            flags |= NetworkProtocol.F_COMPRESSED

        msg_id = NetworkProtocol.new_id()
        header = NetworkProtocol._HDR.pack(NetworkProtocol.MAGIC, NetworkProtocol.VERSION, flags, mtype,
                                           len(meta_b), len(payload), msg_id)
        return header + meta_b + payload

    @staticmethod
    def unpack_message(blob: bytes) -> Tuple[int, Dict[str, str], bytes, int, bytes]:
        (magic, ver, flags, mtype, meta_len, payload_len, msg_id) = NetworkProtocol._HDR.unpack(
            blob[:NetworkProtocol._HDR.size]
        )
        assert magic == NetworkProtocol.MAGIC and ver == NetworkProtocol.VERSION, "Bad protocol header"
        off = NetworkProtocol._HDR.size
        meta = json.loads(blob[off:off+meta_len].decode("utf-8"))
        off += meta_len
        payload = blob[off:off+payload_len]
        if flags & NetworkProtocol.F_COMPRESSED:
            payload = zlib.decompress(payload)
        return mtype, meta, payload, flags, msg_id

# ========================= IDS =========================
class NetworkIDS:
    from collections import defaultdict, deque

    class RateLimiter:
        def __init__(self, rps: float = 20.0, burst: int = 40):
            from collections import defaultdict
            import time as _t
            self.capacity = float(burst)
            self.refill   = float(rps)
            self.tokens: Dict[str, float] = defaultdict(lambda: self.capacity)
            self.ts: Dict[str, float]     = defaultdict(_t.time)
            self.lock = threading.Lock()

        def allow(self, key: str, now: float = None) -> bool:
            import time as _t
            now = now or _t.time()
            with self.lock:
                last = self.ts[key]
                self.tokens[key] = min(self.capacity, self.tokens[key] + (now - last) * self.refill)
                self.ts[key] = now
                if self.tokens[key] >= 1.0:
                    self.tokens[key] -= 1.0
                    return True
                return False

    class TinyAnomaly:
        def __init__(self, alpha: float = 0.2):
            from collections import defaultdict, deque
            self.alpha = alpha
            self.means: Dict[str, float] = defaultdict(float)
            self.vars:  Dict[str, float] = defaultdict(lambda: 1.0)
            self.last_ts: Dict[str, float] = {}
            self.flags = deque(maxlen=1024)  # (ts, peer, reason)

        def update(self, peer: str, msg_size: int, now: float = None) -> Optional[str]:
            import time as _t, math
            now = now or _t.time()
            mu, var = self.means[peer], self.vars[peer]
            if mu == 0.0:
                mu, var = float(msg_size), max(1.0, (msg_size**2)/4)
            else:
                mu = self.alpha*msg_size + (1-self.alpha)*mu
                var = self.alpha*((msg_size - mu)**2) + (1-self.alpha)*var

            itv_reason = None
            if peer in self.last_ts:
                itv = now - self.last_ts[peer]
                if itv < 0.005:
                    itv_reason = "burst-interval"
                elif itv > 30 and msg_size > 8*mu:
                    itv_reason = "idle-then-large"

            self.means[peer], self.vars[peer] = mu, max(var, 1.0)
            self.last_ts[peer] = now

            z = abs((msg_size - mu)/max(1e-6, (var**0.5)))
            if z > 6.0:
                reason = "size-z>6"
            elif itv_reason:
                reason = itv_reason
            else:
                return None
            self.flags.append((now, peer, reason))
            return reason

    def __init__(self, rps=20.0, burst=40):
        from collections import deque
        self.limiter = NetworkIDS.RateLimiter(rps, burst)
        self.detector = NetworkIDS.TinyAnomaly()
        self.audit_log = deque(maxlen=5000)
        self.callbacks: Dict[str, Callable[[str, str], None]] = {}

    def on(self, event: str, fn: Callable[[str, str], None]):
        self.callbacks[event] = fn

    def check(self, peer: str, msg_size: int) -> bool:
        ok = self.limiter.allow(peer)
        if not ok:
            self._emit("rate_limit", peer, f"rate limit exceeded: {msg_size}")
            return False
        reason = self.detector.update(peer, msg_size)
        if reason:
            self._emit("anomaly", peer, reason)
        return True

    def _emit(self, event: str, peer: str, info: str):
        from time import strftime
        line = f"{strftime('%Y-%m-%d %H:%M:%S')} [{event}] {peer} :: {info}"
        self.audit_log.append(line)
        if event in self.callbacks:
            try: self.callbacks[event](peer, info)
            except Exception: pass

# ========================= Transport (TCP/UDP Node) =========================
class NetworkNode:
    """
    Threaded, low-dependency comms node.
    - send() enqueues; background sender batches, encrypts, signs.
    - receiver validates AEAD/HMAC, and hands messages to on_message().
    - adaptive timeouts: grows to MAX on repeated timeouts, shrinks on success.
    """
    _DEFAULT_TIMEOUT   = 4.0
    _MAX_TIMEOUT       = 30.0
    _BATCH_BYTES_SOFT  = 32_768
    _BATCH_LATENCY_MS  = 60
    _UDP_CHUNK         = 1200

    def __init__(self,
                 node_id: str,
                 shared_secret: bytes,
                 bind: Tuple[str,int],
                 prefer_tcp: bool = True,
                 allow_udp: bool  = True):
        self.node_id = node_id
        self.shared_key = SarahNetCrypto.hkdf_256(shared_secret, info=b"sarah-net-"+node_id.encode())
        self.bind_addr = bind
        self.prefer_tcp = prefer_tcp
        self.allow_udp  = allow_udp

        self._txq: "queue.Queue[bytes]" = queue.Queue(maxsize=4096)
        self._stop   = threading.Event()
        self._timeout = self._DEFAULT_TIMEOUT
        self._peers: Dict[str, Tuple[str,int]] = {}  # peer_id -> (host,port)

        self.ids = NetworkIDS(rps=float(config.SARAHNET_RPS), burst=float(config.SARAHNET_BURST))
        self.ids.on("anomaly",     lambda p,i: self._log(f"[IDS] anomaly {p} :: {i}"))
        self.ids.on("rate_limit",  lambda p,i: self._log(f"[IDS] ratelimit {p} :: {i}"))

        # sockets
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_sock.settimeout(0.5)
        self.tcp_sock.bind(bind)
        try:
            self.tcp_sock.listen(8)
        except Exception:
            pass

        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_sock.settimeout(0.2)
        try:
            self.udp_sock.bind(bind)
        except Exception:
            pass

        # threads
        self._threads = [
            threading.Thread(target=self._accept_loop, name="SMN-TCP-Accept", daemon=True),
            threading.Thread(target=self._udp_rx_loop, name="SMN-UDP-Rx", daemon=True),
            threading.Thread(target=self._tx_loop, name="SMN-Tx", daemon=True),
        ]
        for t in self._threads: t.start()

        # Callbacks
        self.on_message: Callable[[str, dict, bytes], None] = lambda p, m, b: None
        self.on_log:     Callable[[str], None] = print

    def _log(self, s: str):
        try: self.on_log(s)
        except Exception: pass

    # ---- public API ----
    def add_peer(self, peer_id: str, addr: Tuple[str,int]):
        self._peers[peer_id] = addr
        self._log(f"[peers] {peer_id} -> {addr}")

    def stop(self):
        self._stop.set()
        try: self.tcp_sock.close()
        except Exception: pass
        try: self.udp_sock.close()
        except Exception: pass

    def send(self, peer_id: str, payload: bytes, meta: dict = None, flags: int = 0):
        blob = NetworkProtocol.pack_message(NetworkProtocol.T_DATA, payload, meta or {}, flags)
        self._txq.put((peer_id, blob), block=True)

    # ---- internal loops ----
    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self.tcp_sock.accept()
                conn.settimeout(self._timeout)
                threading.Thread(target=self._tcp_conn_loop, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                pass
            except Exception:
                time.sleep(0.05)


    def _tcp_conn_loop(self, conn: socket.socket, addr):
        peer = f"{addr[0]}:{addr[1]}"
        try:
            while not self._stop.is_set():
                # Read 4-byte big-endian length prefix
                szb = b""
                while len(szb) < 4:
                    chunk = conn.recv(4 - len(szb))
                    if not chunk:
                        return
                    szb += chunk
                (n,) = struct.unpack("!I", szb)
                if n <= 0 or n > 8_388_608:  # sanity (<= 8 MB)
                    return
                # Read exactly n bytes (ciphertext)
                ct = b""
                while len(ct) < n:
                    chunk = conn.recv(n - len(ct))
                    if not chunk:
                        return
                    ct += chunk

                if not self.ids.check(peer, len(ct)):
                    continue

                # Decrypt â†’ inner protocol frame â†’ unpack
                try:
                    inner = SarahNetCrypto.open(self.shared_key, ct, aad=b"")
                except Exception:
                    self._log("[crypto] decrypt/auth failed")
                    continue
                try:
                    mtype, meta, payload, flags, mid = NetworkProtocol.unpack_message(inner)
                except Exception:
                    self._log("[proto] unpack failed")
                    continue

                self.on_message(peer, meta, payload)
                # Acknowledge (best-effort)
                try:
                    ack = NetworkProtocol.pack_message(NetworkProtocol.T_ACK, b"", {"ok":"1","mid": meta.get("mid","")}, flags=NetworkProtocol.F_ACK)
                    ct_ack = SarahNetCrypto.seal(self.shared_key, ack, aad=meta.get("ts","").encode())
                    conn.sendall(struct.pack("!I", len(ct_ack)) + ct_ack)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try: conn.close()
            except Exception: pass



    def _udp_rx_loop(self):
        # UDP: a datagram is exactly one ciphertext (no length prefix)
        while not self._stop.is_set():
            try:
                data, addr = self.udp_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.01); continue

            peer = f"{addr[0]}:{addr[1]}"
            if not self.ids.check(peer, len(data)):
                continue
            try:
                inner = SarahNetCrypto.open(self.shared_key, data, aad=b"")
                mtype, meta, payload, flags, mid = NetworkProtocol.unpack_message(inner)
            except Exception:
                continue
            self.on_message(peer, meta, payload)


    def _tx_loop(self):
        batch: Dict[str, bytearray] = {}
        last_flush = time.time()

        def flush():
            nonlocal batch
            now = time.time()
            for peer_id, buf in list(batch.items()):
                self._send_raw(peer_id, bytes(buf))
                del batch[peer_id]
            return now

        while not self._stop.is_set():
            try:
                peer_id, blob = self._txq.get(timeout=0.05)
            except queue.Empty:
                pass
            else:
                try:
                    try:
                        _, meta, payload, flags, _ = NetworkProtocol.unpack_message(blob)
                        aad = meta.get("ts","").encode()
                    except Exception:
                        aad = b""
                    ct = SarahNetCrypto.seal(self.shared_key, blob, aad=aad)
                except Exception:
                    self._log("[crypto] seal failed; dropping")
                    ct = None
                if ct:
                    agg = batch.setdefault(peer_id, bytearray())
                    agg += struct.pack("!I", len(ct)) + ct

            now = time.time()
            if (now - last_flush)*1000 >= self._BATCH_LATENCY_MS:
                last_flush = flush()

            for peer_id, buf in list(batch.items()):
                if len(buf) >= self._BATCH_BYTES_SOFT:
                    self._send_raw(peer_id, bytes(buf))
                    del batch[peer_id]

    def _send_raw(self, peer_id: str, blob: bytes):
        addr = self._peers.get(peer_id)
        if not addr:
            self._log(f"[warn] no route to {peer_id}")
            return

        ok = self._send_tcp(addr, blob)
        if not ok and self.allow_udp and len(blob) <= self._UDP_CHUNK:
            self._send_udp(addr, blob)

    def _send_tcp(self, addr, blob: bytes) -> bool:
        try:
            with socket.create_connection(addr, timeout=self._timeout) as s:
                s.sendall(blob)
                self._timeout = max(1.0, self._timeout * 0.7)
                return True
        except Exception:
            self._timeout = min(self._MAX_TIMEOUT, max(1.0, self._timeout * 1.5))
            return False


    def _send_udp(self, addr, blob: bytes):
        try:
            # blob may contain a single framed message: [len][ct]
            if len(blob) >= 4:
                (n,) = struct.unpack("!I", blob[:4])
                if 4 + n == len(blob):
                    self.udp_sock.sendto(blob[4:], addr)  # send just ciphertext
                    return
            # Fallback: send as-is (best effort)
            self.udp_sock.sendto(blob, addr)
        except Exception:
            pass
# ========================= Server Script Wrapper & Cloud Connector =========================
# NOTE: This class is additive and does not change any existing defs. Itâ€™s importable from
# SarahMemoryNetwork and can be used by SarahMemoryAPI, SarahMemoryAiFunctions, Integration, etc.



class ServerConnection:
    """
    Central wrapper for local server script + cloud hub connector.
    - Non-blocking heartbeat to web hub (config.SARAH_WEB_*)
    - Simple local HTTP relay (optional, no Flask dependency here)
    - Async-friendly methods: ping(), health(), register_node(), send_embedding(), context_update(), post_job_result()
    - Uses aiohttp if available, else urllib fallback
    """
    def __init__(self, session=None):
        import asyncio
        self.base = getattr(config, "SARAH_WEB_BASE", "https://www.sarahmemory.com")
        self.api_prefix = getattr(config, "SARAH_WEB_API_PREFIX", "/api")
        self.timeout = float(getattr(config, "REMOTE_HTTP_TIMEOUT", 6.0) or 6.0)
        self.heartbeat_sec = int(getattr(config, "REMOTE_HEARTBEAT_SEC", 30) or 30)
        self.node_id = getattr(config, "SARAHNET_NODE_ID", "local-node")
        self.api_key = getattr(config, "REMOTE_API_KEY", None)
        self._hb_task = None
        self._loop = None
        self._session = session  # optional externally-managed aiohttp.ClientSession
        # Mesh bootstrap (best-effort)
        try:
            shared = config.sarahnet_shared_secret()
            self.mesh = NetworkNode(
                node_id=self.node_id,
                shared_secret=shared,
                bind=(getattr(config, "SARAHNET_BIND_HOST", "0.0.0.0"),
                      int(getattr(config, "SARAHNET_BIND_PORT", 9876))),
                prefer_tcp=bool(getattr(config, "SARAHNET_PREFER_TCP", True)),
                allow_udp=bool(getattr(config, "SARAHNET_ALLOW_UDP", True)),
            )
            # load peers from globals
            peers = getattr(config, "SARAHNET_PEERS", {}) or {}
            for pid, (h,p) in peers.items():
                try: self.mesh.add_peer(pid, (h, int(p)))
                except Exception: pass
        except Exception:
            self.mesh = None

    # --------- HTTP helpers ---------
    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _url(self, path):
        base = self.base.rstrip("/")
        return f"{base}{self.api_prefix}{path}"

    async def _fetch(self, method, path, payload=None):
        data = json.dumps(payload or {}, separators=(",",":")).encode("utf-8")
        try:
            import aiohttp
            close_me = False
            if self._session is None:
                close_me = True
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self._session = aiohttp.ClientSession(timeout=timeout)
            async with self._session.request(method.upper(), self._url(path), data=data, headers=self._headers()) as resp:
                txt = await resp.text()
                try: return resp.status, json.loads(txt)
                except Exception: return resp.status, {"text": txt}
        except Exception as e:
            # urllib fallback
            try:
                import urllib.request
                req = urllib.request.Request(self._url(path), data=data, headers=self._headers(), method=method.upper())
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    txt = r.read().decode("utf-8","ignore")
                    try: return r.getcode(), json.loads(txt)
                    except Exception: return r.getcode(), {"text": txt}
            except Exception as ee:
                return 599, {"error": str(ee)}
        finally:
            try:
                if close_me and self._session:
                    await self._session.close()
                    self._session = None
            except Exception:
                pass

    # --------- Public async API ---------
    async def ping(self):
        return await self._fetch("GET", getattr(config, "SARAH_WEB_PING_PATH", "/health"))

    async def health(self):
        return await self._fetch("GET", getattr(config, "SARAH_WEB_HEALTH_PATH", "/health"))

    async def register_node(self, node_id=None, meta=None):
        node_id = node_id or self.node_id
        payload = {"node_id": node_id, "meta": meta or {"version": getattr(config, "PROJECT_VERSION","unknown")}}
        return await self._fetch("POST", getattr(config, "SARAH_WEB_REGISTER_PATH", "/register-node"), payload)

    async def send_embedding(self, vector, context_id=None):
        payload = {"node_id": self.node_id, "embedding": vector, "context_id": context_id}
        return await self._fetch("POST", getattr(config, "SARAH_WEB_EMBED_PATH", "/receive-embedding"), payload)

    async def context_update(self, text, tags=None):
        payload = {"node_id": self.node_id, "text": text, "tags": tags or []}
        return await self._fetch("POST", getattr(config, "SARAH_WEB_CONTEXT_PATH", "/context-update"), payload)

    async def post_job_result(self, job_id, result):
        payload = {"node_id": self.node_id, "job_id": job_id, "result": result}
        return await self._fetch("POST", getattr(config, "SARAH_WEB_JOBS_PATH", "/jobs"), payload)

    # --------- Heartbeat ---------
    def start_heartbeat(self):
        import asyncio, threading
        if self._hb_task is not None:
            return
        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._hb_task = self._loop.create_task(self._hb_loop())
            try:
                self._loop.run_until_complete(self._hb_task)
            except Exception:
                pass
        threading.Thread(target=_runner, name="SM-Heartbeat", daemon=True).start()

    async def _hb_loop(self):
        import asyncio
        while True:
            try:
                await self.register_node(self.node_id, {"ts": int(time.time())})
                await asyncio.sleep(self.heartbeat_sec)
            except Exception:
                await asyncio.sleep(min(10, self.heartbeat_sec))

    # --------- Convenience (sync wrappers) ---------
    def _ensure_loop(self):
        import asyncio
        try: loop = asyncio.get_event_loop()
        except Exception:
            loop = None
        return loop

    def ping_sync(self):
        loop = self._ensure_loop()
        if loop and loop.is_running():
            import asyncio
            return asyncio.create_task(self.ping())
        return __import__("asyncio").run(self.ping())

    def health_sync(self):
        loop = self._ensure_loop()
        if loop and loop.is_running():
            import asyncio
            return asyncio.create_task(self.health())
        return __import__("asyncio").run(self.health())

    def register_sync(self, node_id=None, meta=None):
        loop = self._ensure_loop()
        if loop and loop.is_running():
            import asyncio
            return asyncio.create_task(self.register_node(node_id, meta))
        return __import__("asyncio").run(self.register_node(node_id, meta))

def get_default_server_connection() -> "ServerConnection":
    """
    Convenience factory used by modules that want a ready-to-go connector
    without importing aiohttp at import-time.
    """
    return ServerConnection()
    # --- Device Communications (optional dependencies, safe to import) ---

def list_bluetooth_devices(self) -> list[dict]:
    """Scan nearby Bluetooth devices. Requires PyBluez (optional)."""
    try:
        import bluetooth  # PyBluez
        devices = bluetooth.discover_devices(duration=6, lookup_names=True)
        return [{"address": addr, "name": name or ""} for addr, name in devices]
    except Exception:
        return []

def pair_bluetooth(self, address: str, pin: str | None = None) -> bool:
    """Best-effort pairing; on Windows this may require OS pairing UI."""
    try:
        # Note: programmatic pairing varies by OS; we surface intent and log.
        # Future: use Bleak for BLE characteristics if needed.
        return True
    except Exception:
        return False

def wifi_scan(self) -> list[dict]:
    """Return nearby Wi-Fi SSIDs. Windows: netsh; Linux: nmcli/iw."""
    import subprocess, sys, re
    try:
        if sys.platform.startswith("win"):
            out = subprocess.check_output(["netsh", "wlan", "show", "networks", "mode=Bssid"], encoding="utf-8", errors="ignore")
            ssids = re.findall(r"SSID \d+ : (.+)", out)
            return [{"ssid": s.strip()} for s in ssids if s.strip()]
        else:
            out = subprocess.check_output(["nmcli", "-t", "-f", "SSID", "device", "wifi", "list"], encoding="utf-8", errors="ignore")
            return [{"ssid": s.strip()} for s in out.splitlines() if s.strip()]
    except Exception:
        return []

def wifi_connect(self, ssid: str, password: str | None = None) -> bool:
    """Connect to Wi-Fi SSID using OS tools (best effort)."""
    import subprocess, sys
    try:
        if sys.platform.startswith("win"):
            # Expect an existing profile; adding profiles programmatically requires XML
            subprocess.check_call(["netsh", "wlan", "connect", f"name={ssid}"])
            return True
        else:
            args = ["nmcli", "device", "wifi", "connect", ssid]
            if password: args += ["password", password]
            subprocess.check_call(args)
            return True
    except Exception:
        return False

def nfc_poll_and_exchange(self, payload: bytes = b"hello") -> bytes | None:
    """NFC handshake (if nfcpy available and reader attached)."""
    try:
        import nfc
        # Minimal demo pattern; requires hardware and correct permissions
        # In practice, use nfc.ContactlessFrontend('usb') and exchange NDEF
        return payload  # placeholder echo
    except Exception:
        return None

def usb_tethering_status(self) -> dict:
    """Detect common USB tether NICs and trigger sync if they appear."""
    try:
        import psutil, time
        nics = psutil.net_if_addrs().keys()
        usb_like = [n for n in nics if "usb" in n.lower() or "rndis" in n.lower()]
        return {"detected": bool(usb_like), "interfaces": list(usb_like)}
    except Exception:
        return {"detected": False, "interfaces": []}
# RFID/NFC functionality (reading and writing)
# NFC and RFID reader (requires hardware and nfcpy)
try:
    import nfc as _nfc  # optional; may be absent
except Exception:
    _nfc = None

def read_rfid_tag(self) -> Optional[str]:
    if _nfc is None:
        return None
    try:
        clf = _nfc.ContactlessFrontend('usb')
        tag = clf.connect(rdwr={'on-connect': lambda tag: False})
        return getattr(tag, "identifier", b"").hex()
    except Exception:
        return None

def write_rfid_tag(self, data: str) -> bool:
    if _nfc is None:
        return False
    try:
        clf = _nfc.ContactlessFrontend('usb')
        clf.connect(rdwr={'on-connect': lambda tag: write_tag(tag, data)})
        return True
    except Exception:
        return False


def write_tag(tag, data: str) -> None:
    if _nfc is None:
        return
    try:
        rec = _nfc.ndef.TextRecord(data)
        tag.ndef.records = [rec]
    except Exception as e:
        pass  # or: print(f"Error while writing to tag: {e}")
# ========================= Additive Protocol Extensions =========================
# We don't edit the original class; we extend it at runtime.
def _extend_protocol():
    np = NetworkProtocol
    # Modalities / control / safety
    setattr(np, "T_NEGOTIATE",        getattr(np, "T_NEGOTIATE",        20))
    setattr(np, "T_AUTH",             getattr(np, "T_AUTH",             21))
    setattr(np, "T_AUDIT",            getattr(np, "T_AUDIT",            22))
    # Media / telephony
    setattr(np, "T_VOICE_FRAME",      getattr(np, "T_VOICE_FRAME",      30))
    setattr(np, "T_VOICE_CTRL",       getattr(np, "T_VOICE_CTRL",       31))
    setattr(np, "T_DTMF",             getattr(np, "T_DTMF",             32))
    setattr(np, "T_FAX_MODEM_TONE",   getattr(np, "T_FAX_MODEM_TONE",   33))
    # Mobility / sensors
    setattr(np, "T_GPS",              getattr(np, "T_GPS",              40))
    setattr(np, "T_GEOFENCE_EVENT",   getattr(np, "T_GEOFENCE_EVENT",   41))
    # Universal payloads
    setattr(np, "T_TEXT",             getattr(np, "T_TEXT",             50))
    setattr(np, "T_IMAGE",            getattr(np, "T_IMAGE",            51))
    setattr(np, "T_FILE",             getattr(np, "T_FILE",             52))
    setattr(np, "T_SENSOR",           getattr(np, "T_SENSOR",           53))

_extend_protocol()

# Ensure meta receives `_mtype` for handler dispatch without changing receiver loops.
_ORIG_UNPACK = NetworkProtocol.unpack_message
def _patched_unpack_message(blob: bytes):
    mtype, meta, payload, flags, mid = _ORIG_UNPACK(blob)
    if isinstance(meta, dict) and "_mtype" not in meta:
        meta = dict(meta)
        meta["_mtype"] = mtype
    return mtype, meta, payload, flags, mid
NetworkProtocol.unpack_message = staticmethod(_patched_unpack_message)  # safe patch


# ========================= Service Registry & Node Monkey-Patches =========================
class ServiceRegistry:
    """Maps message types to callables: (peer:str, meta:dict, payload:bytes) -> None"""
    def __init__(self):
        self._handlers: Dict[int, Callable[[str, dict, bytes], None]] = {}

    def register(self, mtype: int, fn: Callable[[str, dict, bytes], None]):
        self._handlers[mtype] = fn

    def handle(self, mtype: int, peer: str, meta: dict, payload: bytes):
        fn = self._handlers.get(mtype)
        if fn:
            fn(peer, meta, payload)

def _nn_register_handler(self: "NetworkNode", mtype: int, fn: Callable[[str,dict,bytes],None]):
    if not hasattr(self, "_services"):
        self._services = ServiceRegistry()
    self._services.register(mtype, fn)

def _nn_on_message(self: "NetworkNode", peer: str, meta: dict, payload: bytes):
    # Uniform dispatch, defaulting to T_DATA for legacy senders
    mtype = int(meta.get("_mtype", getattr(NetworkProtocol, "T_DATA")))
    # Audit & IDS hook (non-fatal)
    try:
        if hasattr(self, "_audit"):
            self._audit.log("rx", {"peer": peer, "mtype": mtype, "len": len(payload), "meta": meta})
    except Exception:
        pass
    # Dispatch if any service is registered; otherwise do nothing (back-compat)
    if hasattr(self, "_services"):
        try:
            self._services.handle(mtype, peer, meta, payload)
        except Exception:
            pass

# Monkey-patch onto NetworkNode without editing its class body
setattr(NetworkNode, "register_handler", _nn_register_handler)
setattr(NetworkNode, "on_message", _nn_on_message)


# ========================= Compliance & Audit =========================
class AuditTrail:
    """Writes comms events with anomaly hints to system_logs.db (best effort)."""
    def __init__(self, node_id: str):
        self.node_id = node_id

    def log(self, event: str, details: dict):
        try:
            import sqlite3, os
            from datetime import datetime
            db_path = os.path.abspath(os.path.join(config.DATASETS_DIR, "system_logs.db"))
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS comms_audit(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT, node_id TEXT, event TEXT, details TEXT
            )""")
            cur.execute("INSERT INTO comms_audit(ts,node_id,event,details) VALUES(?,?,?,?)",
                        (datetime.now().isoformat(), self.node_id, event, json.dumps(details)[:1000]))
            conn.commit(); conn.close()
        except Exception:
            pass

class SafetyPolicy:
    """Simple allowlist/denylist gates to avoid misuse (spam/surveillance/unauth)."""
    def __init__(self, allow_external_sip: bool = False):
        self.allow_external_sip = allow_external_sip
        self.blocked_prefixes = ["*99#", "*#06#"]   # sample dialcodes you might decide to block
    def permit_dial(self, number: str) -> bool:
        number = (number or "").strip()
        return not any(number.startswith(p) for p in self.blocked_prefixes)


# ========================= Universal Negotiation =========================
class ProtocolNegotiator:
    """
    Exchanges capability sets during HELLO and chooses optimal format per peer.
    Keeps a peer->selection map visible to other services.
    """
    def __init__(self, node: "NetworkNode"):
        self.node = node
        self.peer_caps: Dict[str, dict] = {}
        self.selection: Dict[str, dict] = {}

        # Register handlers
        node.register_handler(NetworkProtocol.T_HELLO, self._on_hello)
        node.register_handler(NetworkProtocol.T_NEGOTIATE, self._on_negotiate)

    @staticmethod
    def capabilities() -> dict:
        return {
            "v": 1,
            "voice": {"codecs": ["opus", "pcm"], "ptime_ms": [20, 40, 60]},
            "telephony": {"dtmf": True, "p2p": True, "sip_fallback": True},
            "media": {"text": True, "image": True, "file": True, "sensor": True},
            "mobility": {"gps": True, "geofence": True, "bt": True, "wifi": True, "nfc": True, "usb": True},
            "crypto": {"aead": True, "zlib": True},
        }

    def send_hello(self, peer_id: str):
        caps = self.capabilities()
        payload = json.dumps(caps).encode("utf-8")
        self.node.send(peer_id, payload, meta={"_mtype": NetworkProtocol.T_HELLO})

    def _on_hello(self, peer: str, meta: dict, payload: bytes):
        try:
            caps = json.loads(payload.decode("utf-8", "ignore"))
            self.peer_caps[peer] = caps
        except Exception:
            return
        # reply with our chosen selection
        choice = self._choose(caps)
        self.selection[peer] = choice
        self.node.send(peer, json.dumps(choice).encode("utf-8"),
                       meta={"_mtype": NetworkProtocol.T_NEGOTIATE})

    def _on_negotiate(self, peer: str, meta: dict, payload: bytes):
        try:
            self.selection[peer] = json.loads(payload.decode("utf-8", "ignore"))
        except Exception:
            pass

    def _choose(self, caps: dict) -> dict:
        """Greedy but simple: prefer Opus @ 20ms if possible."""
        use_codec = "opus" if "opus" in caps.get("voice", {}).get("codecs", []) else "pcm"
        ptime = 20 if 20 in caps.get("voice", {}).get("ptime_ms", []) else min(caps.get("voice", {}).get("ptime_ms", [60]))
        return {"voice": {"codec": use_codec, "ptime_ms": ptime}}


# ========================= VoIP (RTP-lite over UDP/TCP) =========================
class JitterBuffer:
    """Tiny adaptive jitter buffer for voice frames."""
    def __init__(self, target_ms=60):
        from collections import deque
        self.q = deque()
        self.target_ms = target_ms
        self._last_ts = None
    def push(self, frame_ts: float, data: bytes):
        self.q.append((frame_ts, data))
    def pop(self) -> Optional[bytes]:
        return self.q.popleft()[1] if self.q else None

class OpusCodec:
    """Opus wrapper; falls back to PCM passthrough if libs missing."""
    def __init__(self, sample_rate=16000, channels=1, frame_ms=20):
        self.sample_rate = sample_rate; self.channels = channels; self.frame_ms = frame_ms
        self.ok = False
        try:
            # Try pyogg/opuslib lightly; if missing, passthrough will be used.
            import opuslib  # type: ignore
            self._enc = opuslib.Encoder(sample_rate, channels, opuslib.APPLICATION_AUDIO)
            self._dec = opuslib.Decoder(sample_rate, channels)
            self.ok = True
        except Exception:
            self._enc = self._dec = None

    def encode(self, pcm16: bytes) -> bytes:
        if not self.ok:
            return pcm16  # passthrough
        try:
            import array, opuslib
            pcm = array.array('h', pcm16)
            frame_size = int(self.sample_rate * self.frame_ms / 1000)
            return self._enc.encode(pcm, frame_size, 4000)
        except Exception:
            return pcm16

    def decode(self, data: bytes) -> bytes:
        if not self.ok:
            return data
        try:
            import array
            frame_size = int(self.sample_rate * self.frame_ms / 1000)
            pcm = self._dec.decode(data, frame_size, decode_fec=False)
            out = array.array('h', pcm).tobytes()
            return out
        except Exception:
            return data

class VoIPEngine:
    """
    Sends/receives voice frames (Opus preferred). Tracks latency/jitter/loss and adapts ptime.
    Provides endpoint classification hints (human/fax/modem/IVR) via tone heuristics.
    """
    def __init__(self, node: "NetworkNode", negotiator: ProtocolNegotiator, audit: AuditTrail):
        self.node, self.neg, self.audit = node, negotiator, audit
        self.codec = OpusCodec()
        self.jb = JitterBuffer()
        self.stats = {"rtt_ms": 0.0, "jitter_ms": 0.0, "loss": 0.0}
        # register handlers
        node.register_handler(NetworkProtocol.T_VOICE_FRAME, self._on_voice_frame)
        node.register_handler(NetworkProtocol.T_VOICE_CTRL,  self._on_voice_ctrl)

    def send_frame(self, peer_id: str, pcm16: bytes, ts_ms: int):
        encoded = self.codec.encode(pcm16)
        meta = {"_mtype": NetworkProtocol.T_VOICE_FRAME, "ts": ts_ms, "codec": "opus" if self.codec.ok else "pcm"}
        self.node.send(peer_id, encoded, meta=meta)

    def _on_voice_frame(self, peer: str, meta: dict, payload: bytes):
        try:
            pcm = self.codec.decode(payload)
            self.jb.push(time.time(), pcm)
            self.audit.log("voice_rx", {"peer": peer, "len": len(payload), "codec": meta.get("codec")})
        except Exception:
            pass

    def _on_voice_ctrl(self, peer: str, meta: dict, payload: bytes):
        # future: jitter reports, RTCP-lite feedback
        pass

    # Fax/Modem/IVR endpoint hints (very lightweight heuristic)
    @staticmethod
    def classify_endpoint(pcm16: bytes) -> str:
        # simple spectral peek: detect 2100 Hz fax CNG tone or DTMF presence
        try:
            import math, array
            pcm = array.array('h', pcm16)
            N = min(len(pcm), 16000)  # short window
            # Goertzel for 2100 Hz (fax CNG)
            def goertzel(freq, sr=8000):
                k = int(0.5 + (N*freq)/sr)
                w = 2*math.pi*k/N
                cos_w, sin_w = math.cos(w), math.sin(w)
                s_prev = s_prev2 = 0.0
                for n in range(N):
                    s = pcm[n] + 2*cos_w*s_prev - s_prev2
                    s_prev2, s_prev = s_prev, s
                power = s_prev2**2 + s_prev**2 - 2*cos_w*s_prev*s_prev2
                return power
            fax_power = goertzel(2100.0)
            if fax_power > 1e9: return "fax"
            # quick DTMF check at 697/770/852/941 & 1209/1336/1477
            lows = [697, 770, 852, 941]; highs = [1209, 1336, 1477]
            if any(goertzel(f) > 5e8 for f in lows) and any(goertzel(f) > 5e8 for f in highs):
                return "ivr_or_phone"
            return "human"
        except Exception:
            return "unknown"


# ========================= Telephony (P2P + SIP fallback) =========================
class DTMFDetector:
    """Goertzel-based dual-tone detector returning (symbol or None)."""
    _map = {
        (697,1209): "1",(697,1336): "2",(697,1477): "3",
        (770,1209): "4",(770,1336): "5",(770,1477): "6",
        (852,1209): "7",(852,1336): "8",(852,1477): "9",
        (941,1209): "*",(941,1336): "0",(941,1477): "#",
    }
    def __init__(self, sample_rate=8000):
        self.sr = sample_rate
    def detect(self, pcm16: bytes) -> Optional[str]:
        try:
            import math, array
            pcm = array.array('h', pcm16)
            N = min(len(pcm), 1600)
            def goertzel(freq):
                k = int(0.5 + (N*freq)/self.sr)
                w = 2*math.pi*k/N
                cos_w, sin_w = math.cos(w), math.sin(w)
                s_prev = s_prev2 = 0.0
                for n in range(N):
                    s = pcm[n] + 2*cos_w*s_prev - s_prev2
                    s_prev2, s_prev = s_prev, s
                return s_prev2**2 + s_prev**2 - 2*cos_w*s_prev*s_prev2
            lows  = [697,770,852,941]
            highs = [1209,1336,1477]
            lp = {f: goertzel(f) for f in lows}
            hp = {f: goertzel(f) for f in highs}
            low = max(lp, key=lp.get); high = max(hp, key=hp.get)
            if lp[low] > 5e8 and hp[high] > 5e8:
                return self._map.get((low, high))
        except Exception:
            pass
        return None

class SIPGatewayClient:
    """Optional SIP fallback via pjsua; only used if installed and allowed by SafetyPolicy."""
    def __init__(self, safety: SafetyPolicy):
        self.enabled = False; self.safety = safety
        try:
            import pjsua as pj  # type: ignore
            self.pj = pj; self.enabled = True
        except Exception:
            self.pj = None

    def call(self, sip_uri: str) -> bool:
        if not (self.enabled and self.safety.allow_external_sip):
            return False
        # Minimal placeholder; your full PJ setup would go here.
        return False

class TelephonyController:
    """P2P 'calls' over mesh; SIP gateway fallback if permitted."""
    def __init__(self, node: "NetworkNode", audit: AuditTrail, safety: SafetyPolicy):
        self.node, self.audit, self.safety = node, audit, safety
        self.dtmf = DTMFDetector()
        self.sip  = SIPGatewayClient(safety)
        node.register_handler(NetworkProtocol.T_DTMF, self._on_dtmf)

    def dial_p2p(self, peer_id: str) -> bool:
        self.audit.log("dial_p2p", {"peer": peer_id})
        # send a simple call-setup over VOICE_CTRL (ringing etc.)
        self.node.send(peer_id, b"INVITE", meta={"_mtype": NetworkProtocol.T_VOICE_CTRL})
        return True

    def send_dtmf(self, peer_id: str, symbol: str):
        self.node.send(peer_id, symbol.encode("ascii"), meta={"_mtype": NetworkProtocol.T_DTMF})

    def _on_dtmf(self, peer: str, meta: dict, payload: bytes):
        sym = payload.decode("ascii","ignore")[:1]
        self.audit.log("dtmf_rx", {"peer": peer, "symbol": sym})


# ========================= GPS, Location & Geofencing =========================
class GPSService:
    """Parses NMEA, shares signed location, and emits geofence triggers."""
    def __init__(self, node: "NetworkNode", audit: AuditTrail):
        self.node, self.audit = node, audit
        self.geofences: Dict[str, dict] = {}  # {name: {"lat": .., "lon": .., "radius_m": ..}}
        node.register_handler(NetworkProtocol.T_GPS, self._on_gps)

    @staticmethod
    def parse_nmea(nmea: str) -> Optional[dict]:
        try:
            # Very small GPRMC/GPGGA parser (lat/lon in decimal degrees)
            fields = nmea.strip().split(',')
            if fields[0].endswith("GPRMC") and fields[2] == "A":
                lat = _nmea_to_deg(fields[3], fields[4])
                lon = _nmea_to_deg(fields[5], fields[6])
                return {"lat": lat, "lon": lon, "src": "GPRMC"}
            if fields[0].endswith("GPGGA"):
                lat = _nmea_to_deg(fields[2], fields[3])
                lon = _nmea_to_deg(fields[4], fields[5])
                return {"lat": lat, "lon": lon, "src": "GPGGA"}
        except Exception:
            pass
        return None

    def broadcast(self, peer_id: str, location: dict):
        payload = json.dumps(location).encode("utf-8")
        self.node.send(peer_id, payload, meta={"_mtype": NetworkProtocol.T_GPS})

    def add_geofence(self, name: str, lat: float, lon: float, radius_m: float):
        self.geofences[name] = {"lat": lat, "lon": lon, "radius_m": radius_m}

    def _on_gps(self, peer: str, meta: dict, payload: bytes):
        try:
            loc = json.loads(payload.decode("utf-8","ignore"))
            self.audit.log("gps_rx", {"peer": peer, "loc": loc})
            for name, gf in self.geofences.items():
                if _haversine(loc["lat"], loc["lon"], gf["lat"], gf["lon"]) <= gf["radius_m"]:
                    self.node.send(peer, json.dumps({"name": name}).encode("utf-8"),
                                   meta={"_mtype": NetworkProtocol.T_GEOFENCE_EVENT})
        except Exception:
            pass

def _nmea_to_deg(coord: str, hemi: str) -> float:
    # "ddmm.mmmm" or "dddmm.mmmm"
    if not coord: return 0.0
    raw = float(coord)
    deg = int(raw/100); minutes = raw - deg*100
    val = deg + minutes/60.0
    if hemi in ("S","W"): val = -val
    return val

def _haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6371000.0
    dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))


# ========================= Mobile IO (BT/Wi-Fi/NFC/USB) =========================
class MobileIO:
    """
    Scanning/pairing stubs with graceful degradation. Each fn returns a small dict.
    """
    def scan_bluetooth(self) -> dict:
        try:
            import bluetooth  # PyBluez
            devs = bluetooth.discover_devices(duration=4, lookup_names=True) or []
            return {"ok": True, "devices": devs}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    def scan_wifi(self) -> dict:
        try:
            import pywifi  # pywifi
            wifi = pywifi.PyWiFi(); ifaces = wifi.interfaces()
            nets = []
            for iface in ifaces:
                iface.scan(); time.sleep(1.0)
                for p in iface.scan_results(): nets.append((p.ssid, p.signal))
            return {"ok": True, "networks": nets}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    def nfc_poll(self) -> dict:
        try:
            import nfc  # nfcpy
            return {"ok": True, "device": "present"}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    def usb_tether_status(self) -> dict:
        try:
            import psutil
            nics = psutil.net_if_addrs().keys()
            usb_like = [n for n in nics if "usb" in n.lower() or "rndis" in n.lower()]
            return {"ok": True, "interfaces": list(usb_like)}
        except Exception as e:
            return {"ok": False, "reason": str(e)}


# ========================= Universal Router (Text/Image/File/Sensor) =========================
class UniversalRouter:
    """High-level helpers that pick types, compress, and encrypt automatically."""
    def __init__(self, node: "NetworkNode", audit: AuditTrail):
        self.node, self.audit = node, audit

    def send_text(self, peer_id: str, text: str):
        self.node.send(peer_id, text.encode("utf-8"), meta={"_mtype": NetworkProtocol.T_TEXT})

    def send_image(self, peer_id: str, blob: bytes, fmt: str="jpg"):
        self.node.send(peer_id, blob, meta={"_mtype": NetworkProtocol.T_IMAGE, "fmt": fmt, "ts": int(time.time()*1000)})

    def send_file(self, peer_id: str, blob: bytes, name: str):
        self.node.send(peer_id, blob, meta={"_mtype": NetworkProtocol.T_FILE, "name": name})

    def send_sensor(self, peer_id: str, sensor: str, reading: dict):
        self.node.send(peer_id, json.dumps({"sensor": sensor, "reading": reading}).encode("utf-8"),
                       meta={"_mtype": NetworkProtocol.T_SENSOR})


# ========================= Attachment Helper =========================
def attach_extended_services(node: "NetworkNode",
                             allow_external_sip: bool = False) -> dict:
    """
    One-call extender that wires all new capabilities into an existing node instance.
    Does not alter existing defs. Idempotent (safe to call multiple times).
    """
    if getattr(node, "_extended_ok", False):
        return {"ok": True, "note": "already attached"}

    node._audit = AuditTrail(getattr(node, "node_id", "unknown"))
    node._safety = SafetyPolicy(allow_external_sip=allow_external_sip)
    node._neg = ProtocolNegotiator(node)
    node._voip = VoIPEngine(node, node._neg, node._audit)
    node._tel  = TelephonyController(node, node._audit, node._safety)
    node._gps  = GPSService(node, node._audit)
    node._mob  = MobileIO()
    node._u    = UniversalRouter(node, node._audit)

    node._extended_ok = True
    # Optional: send capabilities to all known peers we already have
    try:
        for pid in getattr(node, "_peers", {}).keys():
            node._neg.send_hello(pid)
    except Exception:
        pass
    return {"ok": True}
# --- Legacy export shims (keep older modules working) ---
try:
    # Old class name â†’ new implementation
    SarahSocketNode = NetworkNode

    # Old top-level imports that used to come from a separate socket module
    pack_message = NetworkProtocol.pack_message
    T_DATA       = NetworkProtocol.T_DATA

    # (Optional) expose a few other common protocol types for convenience
    T_HELLO   = NetworkProtocol.T_HELLO
    T_ACK     = NetworkProtocol.T_ACK
    T_PING    = NetworkProtocol.T_PING
    T_TEXT    = NetworkProtocol.T_TEXT
except Exception:
    pass

# If you maintain an __all__, extend it so â€œfrom SarahMemoryNetwork import â€¦â€ always works
try:
    __all__ = list(set((__all__ if "__all__" in globals() else []) + [
        "SarahSocketNode", "NetworkNode",
        "pack_message", "T_DATA",
        "NetworkProtocol", "ServerConnection",
    ]))
except Exception:
    pass


# [PATCH v7.7.2] Crypto envelope utilities (Ed25519 + X25519) â€” opt-in via NETWORK_SHARING_ENABLED
def sign_message(private_key_bytes: bytes, message: bytes) -> bytes:
    try:
        from nacl.signing import SigningKey
        sk = SigningKey(private_key_bytes)
        return sk.sign(message).signature
    except Exception:
        return b""

def verify_signature(public_key_bytes: bytes, message: bytes, signature: bytes) -> bool:
    try:
        from nacl.signing import VerifyKey
        vk = VerifyKey(public_key_bytes)
        vk.verify(message, signature)
        return True
    except Exception:
        return False

def derive_shared_key(priv_key_bytes: bytes, peer_pub_bytes: bytes) -> bytes:
    try:
        from nacl.public import PrivateKey, PublicKey, Box
        sk = PrivateKey(priv_key_bytes)
        pk = PublicKey(peer_pub_bytes)
        box = Box(sk, pk)
        # derive by boxing zeros
        nonce = b"\x00" * 24
        return box.encrypt(b"seed", nonce).ciphertext[:32]
    except Exception:
        import hashlib
        return hashlib.sha256(priv_key_bytes + peer_pub_bytes).digest()

def pack_envelope(payload: dict, nonce: str, sender_pub: bytes, signature: bytes) -> dict:
    return {"nonce": nonce, "from": sender_pub.hex(), "sig": signature.hex(), "payload": payload}


# --- injected: simple heartbeat ping to remote API ---
def heartbeat_server(base_url=None, timeout=4.0, allow_insecure_env="SARAHMEMORY_ALLOW_INSECURE_API"):
    import time, logging, os
    try:
        import requests
    except Exception:
        logging.warning("[NET] requests not available; skipping heartbeat")
        return False, None
    try:
        import SarahMemoryGlobals as config
    except Exception:
        class config: pass
    if base_url is None:
        base_url = getattr(config, "REMOTE_API_BASE", "https://www.sarahmemory.com/api")
    allow_insecure = str(os.getenv(allow_insecure_env, "1")).strip().lower() in ("1","true","yes","on")
    url = base_url.rstrip("/") + "/health"
    t0 = time.time()
    try:
        r = requests.get(url, timeout=timeout, verify=not allow_insecure)
        ok = (200 <= r.status_code < 300)
    except Exception as e:
        logging.warning("[NET] Heartbeat failed: %s", e)
        return False, None
    rtt = int((time.time()-t0)*1000)
    if ok:
        logging.info("[NET] Server heartbeat OK (%sms): %s", rtt, url)
    else:
        logging.warning("[NET] Server heartbeat DOWN (%sms): %s %s", rtt, url, r.status_code)
    return ok, rtt
# ============================
# PHASE A: Identity & Device Awareness (v7.7.5â€“8)
# ============================

def execute_agent_action(action: dict):
    print("[Phase A] execute_agent_action stub:", action)
    return {"status": "noop"}

# =====================================
# insert all of PHASE A: ABOVE THIS LINE
# =====================================

# =============================================================================
# PHASE D: DISTRIBUTED MULTI-INSTANCE MESH NETWORK (v7.7.5-8)
# =============================================================================
"""
Phase D Implementation: Complete peer-to-peer mesh network for distributed
SarahMemory instances to communicate, share knowledge, and collaborate.

Features:
- Peer-to-peer mesh networking
- Token-based knowledge economy (non-monetary)
- 5-tier reputation system
- Secure RSA/AES encryption
- Distributed transaction ledger
- Automatic peer discovery
- Offline operation with sync
- Anti-abuse protection
"""

# Try to import Phase D mesh module
try:
    import importlib.util

    phase_d_module_path = os.path.join(os.path.dirname(__file__), "SarahMemory_PhaseD_MeshNetwork.py")

    if os.path.exists(phase_d_module_path):
        spec = importlib.util.spec_from_file_location("SarahMemory_PhaseD_MeshNetwork", phase_d_module_path)
        phase_d_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(phase_d_module)

        # Import Phase D components
        MeshNode = phase_d_module.MeshNode
        MeshDatabase = phase_d_module.MeshDatabase
        MeshCrypto = phase_d_module.MeshCrypto
        PeerInfo = phase_d_module.PeerInfo
        MeshMessage = phase_d_module.MeshMessage
        MessageType = phase_d_module.MessageType
        TransactionRecord = phase_d_module.TransactionRecord
        ReputationTier = phase_d_module.ReputationTier
        get_mesh_node = phase_d_module.get_mesh_node
        shutdown_mesh_node = phase_d_module.shutdown_mesh_node

        PHASE_D_ENABLED = True
        logging.info("✅ Phase D Mesh Network loaded successfully")
    else:
        PHASE_D_ENABLED = False
        logging.info("ℹ️  Phase D module not found. Mesh networking disabled.")

except Exception as e:
    PHASE_D_ENABLED = False
    logging.warning(f"⚠️  Phase D mesh not available: {e}")

# Phase D Integration Functions

def mesh_network_available() -> bool:
    """Check if Phase D mesh network is available."""
    return PHASE_D_ENABLED

def start_mesh_node(port: int = 9999, node_id: Optional[str] = None) -> Optional[Any]:
    """
    Start a mesh network node.

    Args:
        port: Port to listen on
        node_id: Optional node identifier

    Returns:
        MeshNode instance if successful, None otherwise
    """
    if not PHASE_D_ENABLED:
        logging.warning("Phase D mesh network not available")
        return None

    try:
        node = get_mesh_node(port=port)
        logging.info(f"✅ Mesh node started: {node.node_id} on port {port}")
        return node
    except Exception as e:
        logging.error(f"Failed to start mesh node: {e}")
        return None

def stop_mesh_node():
    """Stop the global mesh node."""
    if not PHASE_D_ENABLED:
        return

    try:
        shutdown_mesh_node()
        logging.info("✅ Mesh node stopped")
    except Exception as e:
        logging.error(f"Failed to stop mesh node: {e}")

def connect_to_mesh_peer(peer_address: Tuple[str, int]) -> bool:
    """
    Connect to a peer in the mesh network.

    Args:
        peer_address: Tuple of (host, port)

    Returns:
        True if connection successful
    """
    if not PHASE_D_ENABLED:
        logging.warning("Phase D mesh network not available")
        return False

    try:
        node = get_mesh_node()
        success = node.connect_to_peer(peer_address)
        if success:
            logging.info(f"✅ Connected to peer: {peer_address}")
        else:
            logging.warning(f"Failed to connect to peer: {peer_address}")
        return success
    except Exception as e:
        logging.error(f"Error connecting to peer {peer_address}: {e}")
        return False

def get_mesh_peers() -> List:
    """Get list of connected mesh peers."""
    if not PHASE_D_ENABLED:
        return []

    try:
        node = get_mesh_node()
        return node.get_peers()
    except Exception as e:
        logging.error(f"Failed to get mesh peers: {e}")
        return []

def get_mesh_stats() -> Dict:
    """Get mesh network statistics."""
    if not PHASE_D_ENABLED:
        return {"enabled": False}

    try:
        node = get_mesh_node()
        stats = node.get_network_stats()
        stats["enabled"] = True
        return stats
    except Exception as e:
        logging.error(f"Failed to get mesh stats: {e}")
        return {"enabled": True, "error": str(e)}

def get_local_node_status():
    return sm_api.get_system_status()