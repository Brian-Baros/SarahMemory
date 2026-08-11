from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))

import SarahMemorySMLProtocol as sml  # noqa: E402


def test_sml_self_test() -> None:
    result = sml.sml_self_test()
    assert result['status'] == 'OK'
    assert result['serialization_roundtrip'] is True


def test_packet_validation_roundtrip() -> None:
    protocol = sml.get_protocol(reset=True)
    packet = protocol.create_packet(raw_request='Build a Python tool', identity={'primary': 'Developer'})
    assert packet.packet_id
    assert packet.pipeline
    assert protocol.validate_packet(packet).status in ('OK', 'WARNING')
    restored = protocol.deserialize_packet(protocol.serialize_packet(packet))
    assert restored.packet_id == packet.packet_id
