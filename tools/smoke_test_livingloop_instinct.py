import json
import sys

import SarahMemoryCognitiveServices as C

print("Living status:")
print(json.dumps(C.cognitive_living_loop_status(), indent=2)[:2000])

payload = {
    "hazard_type": "fire",
    "confidence": 0.88,
    "human_risk": True,
    "observation": "Grease fire on stove with person in home.",
    "capabilities": {
        "can_speak": True,
        "can_notify": True,
        "can_move": True,
        "has_gripper": True,
        "has_fire_extinguisher_access": True,
        "can_cut_power": True
    }
}
result = C.evaluate_emergency_instinct(payload, caller="smoke_test")
print("\nEmergency instinct result:")
print(json.dumps({
    "ok": result.get("ok"),
    "decision": result.get("decision"),
    "bounded_action_allowed": result.get("bounded_action_allowed"),
    "incident_id": result.get("incident_id"),
    "selected_action": ((result.get("bearing_packet") or {}).get("selected_action") or {}).get("action_id"),
    "contract_id": (result.get("action_contract") or {}).get("contract_id")
}, indent=2))

logs = C.list_emergency_instinct_logs(limit=5)
print("\nLedger events:", logs.get("count"))
