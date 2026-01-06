#!/usr/bin/env python3
"""
SarahNetMCP_Diagnostics.py
------------------------------------------------------------
Unified SarahNet / MCP broker diagnostics test suite.

Combines prior standalone tests:
- Test1: Ping + rendezvous announce + lookup
- Test2: Message send/poll/ack
- Test3: Message + optional file send/poll/ack + signal send/poll/ack
- Test4: "Telecom-like" messaging + WebRTC-style signaling envelopes (offer/answer/ICE/hangup)

IMPORTANT:
- This is signaling + store-and-forward only.
- Real duplex voice/video should be done P2P (e.g., WebRTC). The broker relays signaling envelopes privately.

Usage:
  python SarahNetMCP_Diagnostics.py --base https://api.sarahmemory.com --a BrianLaptopNode01 --b BeachLaptopNode02 --send-file .\test.png
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import sys
import time
from urllib.parse import urljoin

import requests


def _ms(t0: float) -> float:
    return (time.time() - t0) * 1000.0


def _req(session: requests.Session, method: str, base: str, path: str, timeout: float = 15.0, **kw):
    url = urljoin(base.rstrip("/") + "/", path.lstrip("/"))
    t0 = time.time()
    r = session.request(method, url, timeout=timeout, **kw)
    dt = _ms(t0)
    return r, dt


def _banner(base: str, a: str, b: str, send_file: str | None):
    print("\n" + "=" * 60)
    print(" SarahNet MCP Diagnostics (Unified Test1-4)")
    print("=" * 60)
    print(f"Base     : {base}")
    print(f"From node: {a}")
    print(f"To node  : {b}")
    if send_file:
        print(f"Send file: {send_file}")
    print("-" * 60)


def _announce(s: requests.Session, base: str, node_id: str, meta: dict, timeout: float):
    r, dt = _req(s, "POST", base, "/api/net/rendezvous/announce", timeout=timeout, json={"node_id": node_id, "meta": meta})
    print(f"ANNOUNCE node={node_id:<22} => {r.status_code} ({dt:.1f}ms) {r.text}")
    r.raise_for_status()


def _ping(s: requests.Session, base: str, timeout: float):
    r, dt = _req(s, "GET", base, "/api/net/ping", timeout=timeout)
    print(f"PING                       => {r.status_code} ({dt:.1f}ms) {r.text}")
    r.raise_for_status()


def _lookup(s: requests.Session, base: str, node_id: str, timeout: float):
    r, dt = _req(s, "GET", base, "/api/net/rendezvous/lookup", timeout=timeout, params={"node_id": node_id})
    print(f"LOOKUP node={node_id:<20}  => {r.status_code} ({dt:.1f}ms) {r.text}")
    r.raise_for_status()
    return r.json()


def _send_message(s: requests.Session, base: str, from_node: str, to_node: str, payload: dict, timeout: float):
    r, dt = _req(s, "POST", base, "/api/net/message/send", timeout=timeout, json={
        "from_node": from_node,
        "to_node": to_node,
        "message": payload,
    })
    print(f"SEND  MSG to={to_node:<22} => {r.status_code} ({dt:.1f}ms) {r.text}")
    r.raise_for_status()
    j = r.json()
    return j.get("id")


def _poll_messages(s: requests.Session, base: str, node_id: str, limit: int, timeout: float):
    r, dt = _req(s, "GET", base, "/api/net/message/poll", timeout=timeout, params={"node_id": node_id, "limit": limit})
    print(f"POLL  MSG node={node_id:<20} => {r.status_code} ({dt:.1f}ms)")
    r.raise_for_status()
    j = r.json()
    msgs = j.get("messages") or []
    print(f"    received_count={len(msgs)}")
    return msgs


def _ack_message(s: requests.Session, base: str, node_id: str, msg_id: str, timeout: float):
    r, dt = _req(s, "POST", base, "/api/net/message/ack", timeout=timeout, json={"to_node": node_id, "id": msg_id})
    print(f"ACK  MSG node={node_id:<20} => {r.status_code} ({dt:.1f}ms) {r.text}")
    r.raise_for_status()


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _send_file_small(s: requests.Session, base: str, from_node: str, to_node: str, file_path: str, timeout: float):
    data = open(file_path, "rb").read()
    b64 = base64.b64encode(data).decode("utf-8")
    payload = {
        "filename": os.path.basename(file_path),
        "content_b64": b64,
        "size_bytes": len(data),
        "sha256": _sha256_hex(data),
        "ts": time.time(),
    }
    r, dt = _req(s, "POST", base, "/api/net/file/send", timeout=timeout, json={
        "from_node": from_node,
        "to_node": to_node,
        "file": payload,
    })
    print(f"SEND  FILE to={to_node:<21} => {r.status_code} ({dt:.1f}ms) {r.text}")
    r.raise_for_status()
    j = r.json()
    return j.get("id")


def _poll_files(s: requests.Session, base: str, node_id: str, limit: int, timeout: float):
    r, dt = _req(s, "GET", base, "/api/net/file/poll", timeout=timeout, params={"node_id": node_id, "limit": limit})
    print(f"POLL  FILE node={node_id:<19} => {r.status_code} ({dt:.1f}ms)")
    r.raise_for_status()
    j = r.json()
    files = j.get("files") or []
    print(f"    files_count={len(files)}")
    return files


def _ack_file(s: requests.Session, base: str, node_id: str, file_id: str, timeout: float):
    r, dt = _req(s, "POST", base, "/api/net/file/ack", timeout=timeout, json={"to_node": node_id, "id": file_id})
    print(f"ACK  FILE node={node_id:<19} => {r.status_code} ({dt:.1f}ms) {r.text}")
    r.raise_for_status()


def _send_signal(s: requests.Session, base: str, from_node: str, to_node: str, sig_type: str, payload: dict, timeout: float):
    r, dt = _req(s, "POST", base, "/api/net/signal/send", timeout=timeout, json={
        "from_node": from_node,
        "to_node": to_node,
        "type": sig_type,
        "payload": payload,
    })
    print(f"SEND  SIG type={sig_type:<12} => {r.status_code} ({dt:.1f}ms) {r.text}")
    r.raise_for_status()
    j = r.json()
    return j.get("id")


def _poll_signals(s: requests.Session, base: str, node_id: str, limit: int, timeout: float):
    r, dt = _req(s, "GET", base, "/api/net/signal/poll", timeout=timeout, params={"node_id": node_id, "limit": limit})
    print(f"POLL  SIG node={node_id:<19} => {r.status_code} ({dt:.1f}ms)")
    r.raise_for_status()
    j = r.json()
    sigs = j.get("signals") or []
    print(f"    signals_count={len(sigs)}")
    return sigs


def _ack_signal(s: requests.Session, base: str, node_id: str, sig_id: str, timeout: float):
    r, dt = _req(s, "POST", base, "/api/net/signal/ack", timeout=timeout, json={"to_node": node_id, "id": sig_id})
    print(f"ACK  SIG node={node_id:<19} => {r.status_code} ({dt:.1f}ms) {r.text}")
    r.raise_for_status()


def run_all(base: str, a: str, b: str, send_file: str | None = None, timeout: float = 15.0, api_key: str | None = None) -> dict:
    """
    Programmatic entry point (used by SarahMemoryDiagnostics menu).
    Returns a dict report with ok + per-step results.
    """
    report = {"ok": True, "steps": [], "errors": []}
    base = base.rstrip("/")

    s = requests.Session()
    s.headers.update({
        "User-Agent": "SarahNetMCP_Diagnostics/8.0.0",
        "Accept": "application/json",
    })
    if api_key:
        s.headers["X-API-Key"] = api_key

    meta = {
        "host": socket.gethostname(),
        "platform": sys.platform,
        "ts": time.time(),
        "client": "SarahNetMCP_Diagnostics",
        "version": "8.0.0",
    }

    try:
        _banner(base, a, b, send_file)

        # Test1: ping + announce both + lookup
        _ping(s, base, timeout); report["steps"].append("ping_ok")
        _announce(s, base, a, meta, timeout); report["steps"].append("announce_a_ok")
        _announce(s, base, b, meta, timeout); report["steps"].append("announce_b_ok")
        _lookup(s, base, a, timeout); report["steps"].append("lookup_a_ok")
        _lookup(s, base, b, timeout); report["steps"].append("lookup_b_ok")

        # Test2: message A->B roundtrip
        msg_id = _send_message(s, base, a, b, {"text": f"Hello from {a} @ {time.time()}", "kind": "im"}, timeout)
        report["steps"].append("message_send_ok")
        msgs_b = _poll_messages(s, base, b, limit=10, timeout=timeout); report["steps"].append("message_poll_b_ok")
        if msg_id:
            _ack_message(s, base, b, msg_id, timeout); report["steps"].append("message_ack_b_ok")

        # Test3: optional file + signal
        file_id = None
        if send_file:
            file_id = _send_file_small(s, base, a, b, send_file, timeout); report["steps"].append("file_send_ok")
            _poll_files(s, base, b, limit=10, timeout=timeout); report["steps"].append("file_poll_b_ok")
            if file_id:
                _ack_file(s, base, b, file_id, timeout); report["steps"].append("file_ack_b_ok")

        sig_id = _send_signal(s, base, a, b, "poke", {"ts": time.time(), "note": "diagnostics_signal"}, timeout)
        report["steps"].append("signal_send_ok")
        _poll_signals(s, base, b, limit=10, timeout=timeout); report["steps"].append("signal_poll_b_ok")
        if sig_id:
            _ack_signal(s, base, b, sig_id, timeout); report["steps"].append("signal_ack_b_ok")

        # Test4: telecom-like signaling envelopes
        call_id = f"call_{hashlib.sha1(f'{a}->{b}:{time.time()}'.encode()).hexdigest()[:12]}"
        offer_id = _send_signal(s, base, a, b, "webrtc_offer", {"call_id": call_id, "sdp": "v=0 (stub)", "ts": time.time()}, timeout)
        report["steps"].append("webrtc_offer_sent")
        sigs_b = _poll_signals(s, base, b, limit=10, timeout=timeout); report["steps"].append("webrtc_offer_polled")
        if offer_id:
            _ack_signal(s, base, b, offer_id, timeout); report["steps"].append("webrtc_offer_acked")

        answer_id = _send_signal(s, base, b, a, "webrtc_answer", {"call_id": call_id, "sdp": "v=0 (stub)", "ts": time.time()}, timeout)
        report["steps"].append("webrtc_answer_sent")
        sigs_a = _poll_signals(s, base, a, limit=10, timeout=timeout); report["steps"].append("webrtc_answer_polled")
        if answer_id:
            _ack_signal(s, base, a, answer_id, timeout); report["steps"].append("webrtc_answer_acked")

        ice_a = _send_signal(s, base, a, b, "webrtc_ice", {"call_id": call_id, "candidate": "candidate:1 (stub)", "ts": time.time()}, timeout)
        report["steps"].append("webrtc_ice_a_sent")
        _poll_signals(s, base, b, limit=10, timeout=timeout); report["steps"].append("webrtc_ice_a_polled")
        if ice_a:
            _ack_signal(s, base, b, ice_a, timeout); report["steps"].append("webrtc_ice_a_acked")

        ice_b = _send_signal(s, base, b, a, "webrtc_ice", {"call_id": call_id, "candidate": "candidate:2 (stub)", "ts": time.time()}, timeout)
        report["steps"].append("webrtc_ice_b_sent")
        _poll_signals(s, base, a, limit=10, timeout=timeout); report["steps"].append("webrtc_ice_b_polled")
        if ice_b:
            _ack_signal(s, base, a, ice_b, timeout); report["steps"].append("webrtc_ice_b_acked")

        hang_id = _send_signal(s, base, a, b, "webrtc_hangup", {"call_id": call_id, "ts": time.time()}, timeout)
        report["steps"].append("webrtc_hangup_sent")
        _poll_signals(s, base, b, limit=10, timeout=timeout); report["steps"].append("webrtc_hangup_polled")
        if hang_id:
            _ack_signal(s, base, b, hang_id, timeout); report["steps"].append("webrtc_hangup_acked")

        print("\n✅ Unified SarahNet diagnostics complete.")
        return report

    except Exception as e:
        report["ok"] = False
        report["errors"].append(str(e))
        print("\n❌ Unified SarahNet diagnostics FAILED:", e)
        return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://api.sarahmemory.com", help="Broker base URL")
    ap.add_argument("--a", required=True, help="User A node_id (caller)")
    ap.add_argument("--b", required=True, help="User B node_id (callee)")
    ap.add_argument("--send-file", default=None, help="Optional file path to send as small broker payload")
    ap.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds")
    ap.add_argument("--api-key", default=None, help="Optional X-API-Key header")
    args = ap.parse_args()

    run_all(
        base=args.base,
        a=args.a.strip(),
        b=args.b.strip(),
        send_file=args.send_file,
        timeout=float(args.timeout),
        api_key=args.api_key.strip() if args.api_key else None,
    )


if __name__ == "__main__":
    main()
