from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().with_name("SarahMemorySMUGCC.py")
spec = importlib.util.spec_from_file_location("SarahMemorySMUGCC", MODULE_PATH)
smugcc = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(smugcc)


def valid_envelope():
    return smugcc.build_smugcc_envelope(
        identity={
            "subject_id": "external:test-agent",
            "provider": "test_provider",
            "implementation": "unit_test_adapter",
            "version": "1.0",
            "origin": "external",
            "trust_level": "unverified",
        },
        protocol={
            "source_protocol": "unit-test",
            "source_version": "1.0",
            "adapter_id": "generic_rest_tool",
            "adapter_version": "1.0",
        },
        mission={
            "mission_id": "mission-1",
            "task_id": "task-1",
            "objective": "Validate SMUGCC contract only.",
            "intent": "contract_validation",
            "requested_by": "unit-test",
            "created_at": "2026-09-16T00:00:00.000+00:00",
        },
        capabilities={"declared": ["inspect"], "requested": ["inspect"], "granted": [], "denied": []},
        payload={"input": {"hello": "world"}, "normalized": {"hello": "world"}},
    )


class TestSarahMemorySMUGCC(unittest.TestCase):
    def test_valid_smugcc_envelope_passes_validation(self):
        result = smugcc.validate_smugcc_envelope(valid_envelope())
        self.assertTrue(result["ok"], result)

    def test_missing_identity_fails_validation(self):
        envelope = valid_envelope()
        envelope.pop("identity")
        self.assertFalse(smugcc.validate_smugcc_envelope(envelope)["ok"])

    def test_missing_mission_fails_validation(self):
        envelope = valid_envelope()
        envelope.pop("mission")
        self.assertFalse(smugcc.validate_smugcc_envelope(envelope)["ok"])

    def test_wildcard_resource_denied(self):
        envelope = valid_envelope()
        envelope["resources"]["allowed_sources"] = ["*"]
        result = smugcc.validate_smugcc_envelope(envelope)
        self.assertIn("wildcard_resource_denied", {e["code"] for e in result["errors"]})

    def test_shell_allowed_true_is_denied_by_default(self):
        envelope = valid_envelope()
        envelope["resources"]["shell_allowed"] = True
        result = smugcc.validate_smugcc_envelope(envelope)
        self.assertIn("shell_authority_denied_by_default", {e["code"] for e in result["errors"]})

    def test_filesystem_write_authority_denied_by_default(self):
        envelope = valid_envelope()
        envelope["resources"]["filesystem_allowed"] = True
        result = smugcc.validate_smugcc_envelope(envelope)
        self.assertIn("filesystem_write_authority_denied_by_default", {e["code"] for e in result["errors"]})

    def test_memory_allowed_true_denied_unless_governed(self):
        envelope = valid_envelope()
        envelope["resources"]["memory_allowed"] = True
        envelope["governance"].pop("memory_governed", None)
        result = smugcc.validate_smugcc_envelope(envelope)
        self.assertIn("memory_authority_requires_governance", {e["code"] for e in result["errors"]})

    def test_device_allowed_true_denied_without_hard_governance_path(self):
        envelope = valid_envelope()
        envelope["resources"]["device_allowed"] = True
        envelope["governance"].pop("hard_governance_path", None)
        result = smugcc.validate_smugcc_envelope(envelope)
        self.assertIn("device_authority_requires_hard_governance", {e["code"] for e in result["errors"]})

    def test_execution_authority_cannot_be_self_granted(self):
        envelope = valid_envelope()
        envelope["authority"]["execution_authority"] = True
        result = smugcc.validate_smugcc_envelope(envelope)
        self.assertIn("execution_authority_self_grant_denied", {e["code"] for e in result["errors"]})

    def test_adapter_declaration_validates_without_executing(self):
        result = smugcc.validate_adapter_declaration(smugcc.ADAPTER_DECLARATIONS["generic_rest_tool"])
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["execution_authority"])

    def test_passport_required_external_task_cannot_bypass_trust_registry(self):
        envelope = valid_envelope()
        envelope["passport"]["required"] = False
        result = smugcc.validate_smugcc_envelope(envelope)
        self.assertIn("passport_required_for_external_contract", {e["code"] for e in result["errors"]})

    def test_return_contract_requires_evidence_hash_and_receipt(self):
        envelope = valid_envelope()
        envelope["return_contract"]["evidence_required"] = False
        envelope["return_contract"]["hash_required"] = False
        envelope["return_contract"]["receipt_required"] = False
        result = smugcc.validate_smugcc_envelope(envelope)
        codes = {e["code"] for e in result["errors"]}
        self.assertIn("return_evidence_required", codes)
        self.assertIn("return_hash_required", codes)
        self.assertIn("return_receipt_required", codes)

    def test_validator_has_no_external_network_side_effects(self):
        envelope = valid_envelope()
        with patch("socket.socket", side_effect=AssertionError("network used")):
            result = smugcc.validate_smugcc_envelope(envelope)
        self.assertTrue(result["ok"], result)

    def test_validator_has_no_shell_execution_side_effects(self):
        envelope = valid_envelope()
        with patch("subprocess.Popen", side_effect=AssertionError("shell used")):
            result = smugcc.validate_smugcc_envelope(envelope)
        self.assertTrue(result["ok"], result)

    def test_invalid_contract_produces_deterministic_errors(self):
        bad = {"schema": "wrong", "contract_version": "0"}
        first = smugcc.validate_smugcc_envelope(bad)
        second = smugcc.validate_smugcc_envelope(bad)
        self.assertEqual(first["errors"], second["errors"])


if __name__ == "__main__":
    unittest.main()
