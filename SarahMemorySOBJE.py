"""--==The SarahMemory Project==--
File: SarahMemorySOBJE.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-03-01
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com
===============================================================================
"""
from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "vision_engine"
# CATEGORY = "object_detection"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "camera_gpu_optional"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "sobje"
# FAMILY = "vision"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Object and scene observation engine with lazy YOLO loading, contour fallback, label normalization, color extraction, and observation logging for local visual reasoning."
# --- SARAHMETA END ---
import cv2
import logging
import os
import threading
import asyncio
import sqlite3
import random
import numpy as np
from datetime import datetime
import re as _re
from typing import Any, Dict, List, Optional, Tuple
import SarahMemoryGlobals as config
from SarahMemoryGlobals import DATASETS_DIR, OBJECT_MODEL_CONFIG, OBJECT_DETECTION_ENABLED, MODEL_PATHS


def _vision_observation_logging_enabled() -> bool:
    """Read the vision observation logging switch from SarahMemoryGlobals.

    SarahMemoryGlobals.py is the constitutional control plane. SOBJE must not
    decide this locally and must not silently enable DB growth from environment
    variables. Missing flag fails closed.
    """
    try:
        return bool(getattr(config, "VISION_OBSERVATION_LOGGING_ENABLED", False))
    except Exception:
        return False
#import SarahMemoryFacialRecognition as fr
#from SarahMemoryGlobals import run_async
#from SarahMemoryHi import async_update_network_state


# -----------------------------------------------------------------------------
# YOLO model loading (LAZY) — do NOT load heavyweight models at import time.
# - Respects OBJECT_MODEL_CONFIG[*].enabled as the "user override" control plane.
# - POOR tier: if user enabled nothing, we stay core-only (no 3rd-party load).
# - Avoids CPU/GPU spikes during boot; models load only when inference is requested.
# -----------------------------------------------------------------------------
_YOLO_LOCK = threading.RLock()
_YOLO_CACHE = {}  # model_name -> YOLO instance
_YOLO_IMPORT_ERR = None

def _enabled_yolo_model_names() -> list:
    try:
        return [n for n, cfg in (OBJECT_MODEL_CONFIG or {}).items() if (cfg or {}).get("enabled")]
    except Exception:
        return []

def _hardware_tier_rating() -> str:
    """Best-effort tier rating from SarahMemoryGlobals.hardware_score()."""
    try:
        hs_fn = getattr(config, "hardware_score", None)
        if callable(hs_fn):
            hs = hs_fn() or {}
            # prefer Tier Rating wording if present, else tier, else empty
            tr = hs.get("tier_rating") or hs.get("tier") or ""
            return str(tr).strip()
    except Exception:
        pass
    return ""

def _resolve_preferred_yolo_model() -> str | None:
    """If Globals has a resolver, try to get the preferred *vision* model name (if it matches configured YOLO keys)."""
    try:
        rm_fn = getattr(config, "resolve_model", None)
        if callable(rm_fn):
            rm = rm_fn("vision") or {}
            if isinstance(rm, dict):
                sel = rm.get("selected")
                if sel:
                    return str(sel)
    except Exception:
        pass
    return None

def _yolo_priority_order() -> list:
    enabled = _enabled_yolo_model_names()
    if not enabled:
        return []

    # If tier is POOR and user didn't enable any YOLO models, core-only stays intact.
    # (The actual enforcement is: enabled list is empty => we load none.)
    pref = _resolve_preferred_yolo_model()
    if pref and pref in enabled:
        return [pref] + [x for x in enabled if x != pref]
    return enabled

def _load_yolo_model(model_name: str):
    global _YOLO_IMPORT_ERR
    with _YOLO_LOCK:
        if model_name in _YOLO_CACHE:
            return _YOLO_CACHE.get(model_name)

        # Gate: if nothing is enabled, don't load anything.
        enabled = _enabled_yolo_model_names()
        if model_name not in enabled:
            return None

        # Import ultralytics only on-demand.
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            _YOLO_IMPORT_ERR = e
            logging.warning(f"[YOLO Import Fail] ultralytics unavailable: {e}")
            return None

        model_dir = MODEL_PATHS.get(model_name)
        if not model_dir:
            logging.warning(f"[YOLO Load Skip] {model_name}: Model path missing in MODEL_PATHS")
            return None

        try:
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
        except Exception:
            model_files = []

        if not model_files:
            logging.warning(f"[YOLO Load Skip] {model_name}: No .pt file found in {model_dir}")
            return None

        model_path = os.path.join(model_dir, model_files[0])
        try:
            _YOLO_CACHE[model_name] = YOLO(model_path)
            return _YOLO_CACHE[model_name]
        except Exception as e:
            logging.warning(f"[YOLO Init Fail] {model_name} @ {model_path}: {e}")
            return None

def _get_yolo_models_for_inference() -> dict:
    """Return {model_name: YOLO} for currently enabled models, in priority order."""
    models = {}
    # If OBJECT_DETECTION_ENABLED is off, we load nothing.
    if not OBJECT_DETECTION_ENABLED:
        return models

    # POOR tier policy: do not auto-enable anything here. Only honor enabled flags.
    order = _yolo_priority_order()
    for name in order:
        m = _load_yolo_model(name)
        if m is not None:
            models[name] = m
    return models
def smart_interest_crop(bgr):
    """Return a high-detail crop using Laplacian-variance scoring.
    Keeps aspect ~4:3 and searches a few scales. Safe if anything fails."""
    try:
        import cv2, numpy as np
        h, w = bgr.shape[:2]
        best = None
        for scale in (1.0, 0.8, 0.6):
            th, tw = int(h*scale), int(w*scale)
            # keep window smaller than frame
            hh, ww = max(60, int(th*0.5)), max(80, int(tw*0.5))
            step_y = max(10, hh//6)
            step_x = max(10, ww//6)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for y in range(0, h-hh, step_y):
                for x in range(0, w-ww, step_x):
                    roi = gray[y:y+hh, x:x+ww]
                    score = cv2.Laplacian(roi, cv2.CV_64F).var()
                    if (best is None) or (score > best[0]):
                        best = (score, x, y, ww, hh)
        if best:
            _, x, y, ww, hh = best
            return bgr[y:y+hh, x:x+ww].copy()
        return None
    except Exception:
        return None
logger = logging.getLogger("SarahMemorySOBJE")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    sh = logging.NullHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

# Runtime memory for lightweight vision continuity (single-process, best-effort).
_VISION_RUNTIME_STATE: Dict[str, Any] = {
    "prev_gray": None,
    "prev_timestamp": None,
    "last_findings": None,
}

# === [ADDED] Label normalization & synonyms ===

_FOOD = set("""
apple banana orange pear grape watermelon melon mango pineapple strawberry blueberry raspberry blackberry cherry lemon lime peach plum apricot
broccoli carrot cauliflower spinach kale lettuce cabbage onion garlic tomato cucumber pepper jalapeno potato sweet potato
pizza slice burger sandwich taco burrito hotdog sushi nigiri sashimi roll ramen pasta spaghetti lasagna noodle dumpling
steak chicken wing drumstick thigh breast pork bacon ham sausage meatball tofu egg omelet boiled egg scrambled egg
bread loaf baguette toast croissant bun bagel tortilla pita naan pastry pie cake cupcake brownie cookie donut biscuit
cheese cheddar mozzarella parmesan feta swiss brie yogurt milk butter cream ice cream gelato
coffee espresso latte cappuccino macchiato mocha tea green tea black tea chai soda cola juice smoothie water bottle can
ketchup mustard mayo mayonnaise hot sauce salsa soy sauce vinegar olive oil
cup mug bowl plate jar can bottle glass lid straw spoon fork knife chopsticks container box bag wrapper carton tray
""".split())

_GARMENTS_UPPER = set("""
shirt t-shirt tee top blouse polo hoodie sweater sweatshirt jacket coat cardigan pullover tunic henley jersey
""".split())
_GARMENTS_LOWER = set("""
pants trousers jeans shorts skirt leggings slacks chinos sweatpants joggers capris
""".split())
_HEADWEAR = set("hat cap beanie headband visor beret fedora helmet hood".split())
_FOOTWEAR = set("shoe shoes sneakers boots sandals heels loafers slippers cleats flip-flops".replace('-', '').split())

_UI = set("window screen tab icon cursor scrollbar taskbar desktop dialog menu button checkbox radio slider".split())
_TOOLS = set("knife scissors spatula ladle whisk tongs peeler grater opener corkscrew can-opener screwdriver wrench pliers hammer tape".split())
_CONTAINERS = set("cup mug bowl jar bottle can plate box bag carton tray tub container glass".split())

_SYNONYMS = {
    "garments": {
        "tshirt": "t-shirt", "tee-shirt": "t-shirt", "hooded sweatshirt": "hoodie",
        "sweat shirt": "sweatshirt", "blue jeans": "jeans",
    },
    "food": {"fries": "french fries", "chips": "crisps", "soda": "cola", "pop": "cola"},
    "objects": {"spectacles": "glasses", "cellphone": "phone"},
}
_PLURALS = {"shoes":"shoe","pants":"pants","trousers":"trousers","sneakers":"shoe","jeans":"jeans","shorts":"shorts"}

def _norm_token(s: str) -> str:
    s = s.strip().lower()
    s = s.replace('_',' ').replace('-', ' ').replace('  ', ' ')
    s = _SYNONYMS["garments"].get(s, _SYNONYMS["food"].get(s, _SYNONYMS["objects"].get(s, s)))
    if s in _PLURALS: return _PLURALS[s]
    if s.endswith('es') and s[:-2] in _FOOD: return s[:-2]
    if s.endswith('s') and s[:-1] in _FOOD: return s[:-1]
    return s

def _domain_of(label: str):
    t = _norm_token(label)
    if t in _FOOD: return "food", t
    if t in _GARMENTS_UPPER or t in _GARMENTS_LOWER or t in _HEADWEAR or t in _FOOTWEAR:
        return "garment", t
    if t in _CONTAINERS: return "container", t
    if t in _TOOLS: return "tool", t
    if t in _UI: return "ui", t
    return "object", t

# === [ADDED] Broad offline LM: lexicons + intent ===

_LX = {
  "garment_upper": set("shirt t-shirt tee top blouse polo hoodie sweater sweatshirt jacket coat cardigan pullover tunic henley jersey".split()),
  "garment_lower": set("pants trousers jeans shorts skirt leggings slacks chinos sweatpants joggers capris".split()),
  "headwear": set("hat cap beanie headband visor beret fedora helmet hood".split()),
  "footwear": set("shoe shoes sneakers boots sandals heels loafers slippers cleats".split()),
  "objects_food": set("""apple banana orange pear grape watermelon melon mango pineapple strawberry blueberry raspberry cherry lemon
                         broccoli carrot cauliflower spinach kale lettuce cabbage onion garlic tomato cucumber pepper potato
                         pizza slice burger sandwich taco burrito hotdog sushi ramen pasta lasagna noodle steak chicken egg tofu
                         bread loaf baguette toast croissant bagel tortilla pastry cake cupcake brownie cookie donut pie cheese
                         milk yogurt butter cream ice cream gelato coffee espresso latte cappuccino tea soda cola juice smoothie
                         cup mug bowl plate jar can bottle glass""".split()),
  "containers": set("cup mug bowl jar bottle can plate box bag carton tray tub container glass".split()),
  "ui": set("window screen tab icon cursor scrollbar taskbar desktop dialog menu button checkbox radio slider".split()),
  "tools": set("knife scissors spatula ladle whisk tongs peeler grater opener corkscrew screwdriver wrench pliers hammer tape".split()),
  "body": set("face head hair eye eyes nose mouth ear ears beard mustache hand hands arm arms leg legs foot feet torso chest".split())
}

def _any_in(q, group):
  for t in sorted(_LX[group], key=len, reverse=True):
    if t in q: return t
  return None

_COLOR_PAT = _re.compile(r"(what\\s+color\\s+(is|are)|color\\s+of|what\\s+is\\s+the\\s+color\\s+of|what\\s+colour\\s+(is|are)|colour\\s+of)", _re.I)

def sm_parse_visual_intent(text: str):
  q = (text or '').lower().strip()
  is_color = bool(_COLOR_PAT.search(q))

  # rough pronoun routing
  selfish = bool(_re.search(r"\\b(my|me|myself|i am|on me|wearing|i\\s*have)\\b", q))
  screenish = bool(_re.search(r"\\b(screen|window|tab|desktop|cursor|taskbar|wallpaper|icon|close the window|move the mouse)\\b", q))
  demonstrative = bool(_re.search(r"\\b(this|that|these|those)\\b", q))

  # target extraction across groups (longest match first)
  target = (_any_in(q,"garment_upper") or _any_in(q,"garment_lower") or
            _any_in(q,"headwear") or _any_in(q,"footwear") or
            _any_in(q,"objects_food") or _any_in(q,"containers") or
            _any_in(q,"ui") or _any_in(q,"tools") or _any_in(q,"body"))

  kind = "object"
  if target in (_LX["garment_upper"] | _LX["garment_lower"] | _LX["headwear"] | _LX["footwear"]):
    kind = "garment"

  role = "webcam"  # default bias for real-world objects
  if screenish: role = "screen"
  elif kind == "garment" or selfish or demonstrative: role = "webcam"

  if target is None:
    target = "person" if (kind=="garment" or selfish) else ("screen" if screenish else "object")

  return {"is_color_query": is_color, "target": target, "kind": kind, "role": role}



# ------------------------- SUPER ULTRA Object Detection Engine-------------------------






def ultra_detect_objects(frame: np.ndarray) -> list:
    """
    Super Object Detection Engine.

    Design intent:
      - Works without an installed vision model using the legacy/offline
        contour + broad-domain "dead/demo robot" fallback.
      - Uses enabled SarahMemoryGlobals vision models when present.
      - Never opens hardware; receives a frame only.
      - Writes observations only when SarahMemoryGlobals.VISION_OBSERVATION_LOGGING_ENABLED is True.
    """
    if frame is None:
        logger.warning("No frame provided for object detection.")
        return []

    processed_frame = frame.copy()

    def get_contours(src: np.ndarray) -> list:
        try:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except Exception as e:
            logger.error(f"Error extracting contours: {e}")
            return []

    def _sm_roi_color(bgr_roi):
        try:
            import numpy as _np
            import cv2 as _cv2
            if bgr_roi is None or bgr_roi.size == 0:
                return {"hex": "#000000", "rgb": [0, 0, 0], "hsl": [0, 0, 0], "name": "unknown", "confidence": 0.0}
            hsv = _cv2.cvtColor(bgr_roi, _cv2.COLOR_BGR2HSV)
            _h, s, v = _cv2.split(hsv)
            mask = (s > 18) & (v > 25)
            pix = bgr_roi[mask] if mask.any() else bgr_roi.reshape(-1, 3)
            Z = _np.float32(pix)
            K = 3 if len(Z) >= 3 else 1
            crit = (_cv2.TERM_CRITERIA_EPS + _cv2.TERM_CRITERIA_MAX_ITER, 24, 1.0)
            try:
                _, labels, centers = _cv2.kmeans(Z, K, None, crit, 3, _cv2.KMEANS_PP_CENTERS)
                counts = _np.bincount(labels.flatten(), minlength=K)
                idx = int(counts.argmax())
                b, g, r = centers[idx].astype(int).tolist()
                try:
                    from SarahMemoryAdvCU import sm_color_name_from_rgb
                    name = sm_color_name_from_rgb(r, g, b)
                except Exception:
                    name = _bgr_to_color_name((b, g, r))
                conf = float(counts[idx] / max(1, counts.sum()))
            except Exception:
                b, g, r = pix.mean(axis=0).astype(int).tolist()
                try:
                    from SarahMemoryAdvCU import sm_color_name_from_rgb
                    name = sm_color_name_from_rgb(r, g, b)
                except Exception:
                    name = _bgr_to_color_name((b, g, r))
                conf = 0.5
            rp, gp, bp = r / 255.0, g / 255.0, b / 255.0
            mx, mn = max(rp, gp, bp), min(rp, gp, bp)
            light = (mx + mn) / 2.0
            if mx == mn:
                hue = 0.0
                sat = 0.0
            else:
                d = mx - mn
                sat = d / (2.0 - mx - mn) if light > 0.5 else d / (mx + mn + 1e-9)
                if mx == rp:
                    hue = ((gp - bp) / d + (6 if gp < bp else 0))
                elif mx == gp:
                    hue = ((bp - rp) / d + 2)
                else:
                    hue = ((rp - gp) / d + 4)
                hue *= 60.0
                hue %= 360.0
            return {
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "rgb": [int(r), int(g), int(b)],
                "hsl": [float(hue), float(sat), float(light)],
                "name": name,
                "confidence": conf,
            }
        except Exception:
            return {"hex": "#000000", "rgb": [0, 0, 0], "hsl": [0, 0, 0], "name": "unknown", "confidence": 0.0}

    real_detections = []  # (label, (x1, y1, x2, y2), conf, model_name)
    try:
        models = _get_yolo_models_for_inference()
        for mname, model in (models or {}).items():
            results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
            for r in results:
                names = getattr(r, "names", None) or getattr(model, "names", {}) or {}
                for b in getattr(r, "boxes", []) or []:
                    try:
                        cls_id = int(b.cls[0])
                        conf = float(b.conf[0])
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        raw_label = str(names.get(cls_id, "object"))
                        real_detections.append((raw_label, (x1, y1, x2, y2), conf, mname))
                    except Exception:
                        continue
    except Exception as _e:
        logging.warning(f"YOLO inference failed: {_e}")

    # If no model is available, retain the offline contour-only construct.
    if not real_detections:
        for c in get_contours(frame):
            try:
                if cv2.contourArea(c) > 500:
                    x, y, w, h = cv2.boundingRect(c)
                    real_detections.append(("object", (x, y, x + w, y + h), 0.5, "contour"))
            except Exception:
                continue

    observations = []
    detected_objects = []
    for raw_label, (x1, y1, x2, y2), conf, mname in real_detections:
        try:
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_frame, f"{raw_label} {conf:.2f}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
        except Exception:
            pass
        domain, canon = _domain_of(raw_label)
        color_stats = None
        if domain in ("garment", "food", "container", "object"):
            roi = _clamp_crop(frame, x1, y1, x2, y2)
            color_stats = _sm_roi_color(roi)
        detected_objects.append(canon or raw_label)
        observations.append({
            "raw_label": raw_label,
            "label": canon,
            "domain": domain,
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "model": mname,
            "color": color_stats,
        })

    if _vision_observation_logging_enabled() and observations:
        try:
            from SarahMemoryDatabase import get_db_connection
            import json as _json
            conn = get_db_connection('ai_learning.db')
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS object_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, label TEXT, domain TEXT, confidence REAL,
                bbox TEXT, model TEXT, color_name TEXT, color_hex TEXT, raw_stats TEXT
            )""")
            ts = datetime.utcnow().isoformat()
            for obs in observations:
                color = obs.get("color") or {}
                cur.execute("""INSERT INTO object_observations
                    (timestamp, label, domain, confidence, bbox, model, color_name, color_hex, raw_stats)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (ts, obs["label"], obs["domain"], float(obs["confidence"]),
                     _json.dumps(obs["bbox"]), obs["model"],
                     color.get("name"), color.get("hex"), _json.dumps(obs)))
            conn.commit()
            conn.close()
        except Exception as _e:
            logging.warning(f"object_observations logging failed: {_e}")

    domains = {
        "animals": [
            "domestic: cat", "domestic: dog", "farm: cow", "farm: pig", "wild: lion", "wild: tiger", "bird: parrot",
            "reptile: iguana", "aquatic: dolphin", "amphibian: frog", "rodent: hamster"
        ],
        "insects": [
            "insect: bee", "insect: butterfly", "insect: dragonfly", "insect: mosquito", "insect: ant", "insect: beetle",
            "insect: grasshopper", "insect: ladybug"
        ],
        "colors and patterns": [
            "color: hazel eye", "color: green eye", "color: blue eye", "hair: blonde", "hair: auburn", "hair: black",
            "pattern: camouflage", "pattern: houndstooth", "pattern: pinstripe", "pattern: polka dot"
        ],
        "food": [
            "fruit: apple", "fruit: banana", "vegetable: broccoli", "vegetable: carrot", "snack: chocolate bar",
            "drink: coffee", "dish: lasagna", "grain: rice", "dessert: cheesecake"
        ],
        "kitchen": [
            "kitchen tool: spatula", "kitchen tool: ladle", "utensil: fork", "utensil: spoon", "appliance: toaster",
            "appliance: blender", "cutlery: chef's knife", "storage: plastic container"
        ],
        "electronics": [
            "component: resistor", "component: capacitor", "component: transistor", "component: diode",
            "component: LED", "IC: 555 timer", "IC: op-amp", "board: PCB"
        ],
        "mechanics": [
            "mechanical: gear", "mechanical: sprocket", "mechanical: flywheel", "mechanical: driveshaft"
        ],
        "measurement": [
            "tool: metric wrench", "tool: standard wrench", "gauge: caliper", "scale: digital", "ruler: inches",
            "ruler: centimeters", "weight: 5kg plate", "weight: 10lb dumbbell"
        ],
        "facial features": [
            "face: nose", "face: mouth", "face: ear", "face: eyebrow", "face: cheekbone", "face: chin",
            "face: forehead", "face: jawline"
        ],
        "facial expression": [
            "expression: happy", "expression: sad", "expression: angry", "expression: surprised", "expression: neutral",
            "expression: confused", "expression: disgusted", "expression: excited"
        ],
        "facial hair": [
            "facial hair: beard", "facial hair: mustache", "facial hair: goatee", "facial hair: sideburns",
            "facial hair: stubble"
        ],
        "skin tone": [
            "skin tone: fair", "skin tone: medium", "skin tone: olive", "skin tone: tan", "skin tone: dark"
        ],
        "skin condition": [
            "condition: acne", "condition: scar", "condition: wrinkle", "condition: birthmark", "condition: tattoo",
            "condition: mole"
        ],
        "eye color": [
            "eye color: brown", "eye color: blue", "eye color: green", "eye color: gray", "eye color: hazel",
            "eye color: amber"
        ],
        "eye shape": [
            "eye shape: almond", "eye shape: round", "eye shape: hooded", "eye shape: monolid", "eye shape: downturned",
            "eye shape: upturned"
        ],
        "eye condition": [
            "condition: cataract", "condition: glaucoma", "condition: astigmatism", "condition: color blindness",
            "condition: strabismus"
        ],
        "eye detail": [
            "detail: eyelash", "detail: eyebrow", "detail: pupil", "detail: iris", "detail: sclera"
        ],
        "eye feature": [
            "feature: eyelid", "feature: tear duct", "feature: conjunctiva", "feature: cornea", "feature: retina"
        ],
        "eye accessories": [
            "accessory: contact lens", "accessory: glasses", "accessory: sunglasses", "accessory: eye patch"
        ],
        "eye movement": [
            "movement: blink", "movement: squint", "movement: roll", "movement: dart", "movement: stare"
        ],
        "eye expression": [
            "expression: wink", "expression: squint", "expression: wide-eyed", "expression: narrowed"
        ],
        "eye position": [
            "position: looking up", "position: looking down", "position: looking left", "position: looking right"
        ],
        "eye size": [
            "size: small", "size: medium", "size: large", "size: extra-large"
        ],
        "eye distance": [
            "distance: close-set", "distance: wide-set", "distance: normal"
        ],
        "eye symmetry": [
            "symmetry: symmetrical", "symmetry: asymmetrical"
        ],
        "Body parts": [
            "body part: arm", "body part: leg", "body part: hand", "body part: foot", "body part: torso",
            "body part: head", "body part: neck", "body part: shoulder", "body part: back", "body part: abdomen",
            "body part: knee", "body part: elbow", "body part: wrist", "body part: ankle", "body part: hip",
            "body part: finger", "body part: toe", "body part: thumb", "body part: chin", "body part: forehead",
            "body part: cheek", "body part: jaw", "body part: temple", "body part: scalp", "body part: heel",
            "body part: instep", "body part: arch", "body part: palm", "body part: knuckle", "body part: nail",
            "body part: back of hand", "body part: back of foot", "body part: ball of foot", "body part: sole",
            "body part: bridge of foot", "body part: top of foot", "body part: side of foot",
            "body part: side of hand", "body part: base of thumb","body part: breast",
            "body part: waist", "body part: hip bone", "body part: collarbone", "body part: ribcage",
            "body part: spine", "body part: vertebrae", "body part: pelvis", "body part: sacrum", "body part: coccyx",
            "body part: sternum", "body part: scapula", "body part: clavicle", "body part: radius", "body part: ulna",
            "body part: femur", "body part: tibia", "body part: fibula", "body part: patella", "body part: tarsals",
            "body part: metatarsals", "body part: phalanges", "body part: carpal bones", "body part: metacarpals",
            "body part: phalanges of hand", "body part: phalanges of foot", "body part: knuckles of hand",
            "body part: knuckles of foot", "body part: base of fingers", "body part: base of toes",
        ],
        "Body features": [
            "feature: muscle", "feature: fat", "feature: bone", "feature: skin", "feature: hair"
        ],
        "Body conditions": [
            "condition: healthy", "condition: sick", "condition: injured", "condition: fit", "condition: weak"
        ],
        "Body expressions": [
            "expression: relaxed", "expression: tense", "expression: active", "expression: passive"
        ],
        "Body movements": [
            "movement: walk", "movement: run", "movement: jump", "movement: sit", "movement: stand"
        ],
        "Body accessories": [
            "accessory: watch", "accessory: ring", "accessory: bracelet", "accessory: necklace"
        ],
        "Body clothing": [
            "clothing: shirt", "clothing: pants", "clothing: dress", "clothing: jacket", "clothing: shoes"
        ],
        "Body types": [
            "type: athletic", "type: slim", "type: average", "type: overweight", "type: muscular"
        ],
        "Body proportions": [
            "proportion: long", "proportion: short", "proportion: average"
        ],
        "Body symmetry": [
            "symmetry: symmetrical", "symmetry: asymmetrical"
        ],
        "Body movements": [
            "movement: flex", "movement: extend", "movement: rotate", "movement: twist"
        ],
        "Body positions": [
            "position: upright", "position: slouched", "position: bent", "position: straight"
        ],
        "Body sizes": [
            "size: small", "size: medium", "size: large", "size: extra-large"
        ],
        "Body distances": [
            "distance: close", "distance: medium", "distance: far"
        ],
        "Sex": [
            "sex: male", "sex: female", "sex: non-binary"
        ],
        "Age": [
            "age: young", "age: middle-aged", "age: old"
        ],
        "Height": [
            "height: short", "height: average", "height: tall"
        ],
        "anatomy": [
            "anatomy: skeleton", "anatomy: muscle", "anatomy: organ", "anatomy: tissue"
        ],
        "anatomy detail": [
            "anatomy detail: bone structure", "anatomy detail: muscle fiber", "anatomy detail: organ system"
        ],
        "anatomy feature": [
            "anatomy feature: joint", "anatomy feature: ligament", "anatomy feature: tendon"
        ],
        "anatomy condition": [
            "anatomy condition: healthy", "anatomy condition: diseased", "anatomy condition: injured"
        ],
        "anatomy expression": [
            "anatomy expression: relaxed", "anatomy expression: tense", "anatomy expression: active"
        ],
        "anatomy movement": [
            "anatomy movement: flex", "anatomy movement: extend", "anatomy movement: rotate"
        ],
        "anatomy clothing": [
            "anatomy clothing: bandage", "anatomy clothing: cast", "anatomy clothing: support"
        ],
        "reproductive expressions": [
            "reproductive expression: aroused", "reproductive expression: relaxed", "reproductive expression: tense"
        ],
        "reproductive movements": [
            "reproductive movement: ovulation", "reproductive movement: ejaculation", "reproductive movement: menstruation"
        ],
        "facial detail": [f"face-point: landmark_{i}" for i in range(1, 20001)]
    }

    detected_tags = []
    for obj in detected_objects:
        for category, values in domains.items():
            if obj in values:
                detected_tags.append((category, obj))
                break

    medical_flags = {
        "condition: mole": "Possible melanoma",
        "expression: slouched": "Posture anomaly / neuro issue",
        "face-point: landmark_2349": "Right eye droop — stroke warning",
        "skin tone: uneven": "Skin cancer check",
    }
    for tag in detected_tags:
        if tag in medical_flags:
            alert_medical(tag)

    # Legacy offline/demo fallback vocabulary. This keeps SarahMemory operational
    # without a selected vision model, but model detections are preferred when present.
    synthetic_count = 500
    prefixes = ["modular", "adaptive", "quantum", "neural", "precision", "bio-active", "liquid-cooled", "synthetic"]
    nouns = ["transmitter", "oscillator", "stabilizer", "inverter", "thruster", "sensor", "valve", "core"]
    suffixes = ["array", "hub", "grid", "cell", "interface", "matrix", "system", "scanner"]
    synthetic_terms = [
        f"{random.choice(prefixes)} {random.choice(nouns)} {random.choice(suffixes)}"
        for _ in range(synthetic_count)
    ]

    all_objects = []
    for category, terms in domains.items():
        for term in terms:
            all_objects.append(f"{category}: {term}")
    all_objects.extend(synthetic_terms)

    selected = []
    for obs in observations[:8]:
        label = obs.get("label") or obs.get("raw_label")
        domain = obs.get("domain") or "object"
        if label:
            item = f"{domain}: {label}"
            if item not in selected:
                selected.append(item)
    if len(selected) < 3 and all_objects:
        random.shuffle(all_objects)
        target_count = random.randint(3, 12)
        for item in all_objects:
            if item not in selected:
                selected.append(item)
            if len(selected) >= target_count:
                break

    if _vision_observation_logging_enabled() and selected:
        try:
            db_path = os.path.join(config.DATASETS_DIR, "ai_learning.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS object_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        label TEXT
                    )
                """)
                timestamp = datetime.now().isoformat()
                for item in selected:
                    cursor.execute("""
                        SELECT COUNT(*) FROM object_observations
                        WHERE label = ? AND DATE(timestamp) = DATE('now')
                    """, (item,))
                    if cursor.fetchone()[0] == 0:
                        if hasattr(config, 'vision_canvas'):
                            try:
                                config.vision_canvas.itemconfig(config.vision_light, fill="red")
                            except Exception as ce:
                                logger.warning(f"Vision light update failed: {ce}")
                        cursor.execute("INSERT INTO object_observations (timestamp, label) VALUES (?, ?)",
                                       (timestamp, item))
                        if hasattr(config, 'status_bar'):
                            try:
                                config.status.set_status(f"Identified: {item}")
                            except Exception as sbe:
                                logger.warning(f"Status bar update failed: {sbe}")
                conn.commit()
        except Exception as e:
            if hasattr(config, 'vision_canvas'):
                try:
                    config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
                except Exception as ce:
                    logger.warning(f"Vision light update failed: {ce}")
            logger.warning(f"Could not log object detections: {e}")

    logger.info(f"Ultra Detected: {selected}")
    if hasattr(config, 'vision_canvas'):
        try:
            config.vision_canvas.itemconfig(config.vision_light, fill="green")
        except Exception as ce:
            logger.warning(f"Vision light update failed: {ce}")
    return selected

def get_recent_environmental_tags(limit: int = 10) -> str:
    """
    Returns a summarized and human-readable description of recent environmental observations.
    """
    try:
        db_path = os.path.join(config.DATASETS_DIR, "ai_learning.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT label FROM object_observations
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            if not rows:
                return "I can't see anything clearly right now."

            labels = [row[0] for row in rows if row[0] and "face-point" not in row[0]]
            unique_tags = list(set(labels))

            if not unique_tags:
                return "All I can detect right now are facial landmarks or minor visual noise."

            return "I see " + ', '.join(unique_tags) + "."
    except Exception as e:
        logger.warning(f"[SOBJE ERROR] Failed to fetch environment tags: {e}")
        return "I couldn't access the visual detection log."

def alert_medical(tag) -> None:
    """
    Placeholder for handling medical alerts based on tag.
    """
    logger.warning(f"Medical alert triggered for tag: {tag}")

# === [APPEND] Visual Q&A helpers (OCR, color extraction, glasses/headwear) ===
def _dominant_bgr_color(img):
    try:
        import numpy as np, cv2
        small = cv2.resize(img, (64, 64))
        Z = small.reshape((-1,3)).astype('float32')
        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _,labels,centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten())
        return tuple(int(c) for c in centers[counts.argmax()].tolist())
    except Exception:
        return (128,128,128)

def _bgr_to_color_name(bgr):
    # rough mapping to a common color name
    b,g,r = bgr
    import colorsys
    h,l,s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
    h = h*360
    if s<0.10 and l>0.85: return "white"
    if s<0.15 and l<0.15: return "black"
    if s<0.15: return "gray"
    if h<15 or h>=345: return "red"
    if 15<=h<45: return "orange"
    if 45<=h<70: return "yellow"
    if 70<=h<170: return "green"
    if 170<=h<200: return "cyan"
    if 200<=h<255: return "blue"
    if 255<=h<290: return "indigo"
    if 290<=h<345: return "magenta"
    return "unknown"


def _try_pytesseract_text(img):
    try:
        import cv2, numpy as np, pytesseract
    except Exception:
        return None
    try:
        # 1) Denoise & upscale to help small letters
        h, w = img.shape[:2]
        scale = 2 if max(h, w) < 900 else 1
        if scale > 1:
            img = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

        # 2) Convert to gray that respects vivid colors (e.g., blue "NIBCO")
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        # local contrast (CLAHE) on lightness helps faint print
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        L = clahe.apply(L)
        gray = L

        # 3) De-ring & sharpen slightly
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        sharp = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray,(0,0),1.0), -0.5, 0)

        # 4) Adaptive threshold + small close to connect strokes
        th = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 11)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), 1)

        # 5) Try a couple of PSMs (Tesseract layout assumptions)
        for psm in (6, 7, 11):  # block of text, single text line, sparse text
            cfg = f"--oem 3 --psm {psm}"
            txt = pytesseract.image_to_string(th, config=cfg) or ""
            txt = txt.strip()
            if len(txt) >= 3:
                return txt
        return None
    except Exception:
        return None

def _detect_face_boxes(frame):
    try:
        import cv2
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50,50))
        return faces if isinstance(faces, (list,tuple)) else []
    except Exception:
        return []

def _detect_eyeglasses(frame, face_box):
    # Use OpenCV eyeglasses cascade around the eye region; fallback heuristic on edges
    try:
        import cv2, numpy as np
        (x,y,w,h) = face_box
        roi = frame[y:y+h, x:x+w]
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20,15))
        if len(eyes)>=1:
            return True
        # fallback: strong horizontal edges near mid-face
        edges = cv2.Canny(gray, 60, 150)
        band = edges[h//3:h//2, :]
        if band.mean() > 30:
            return True
        return False
    except Exception:
        return False

def _normalize_visual_request(question_or_payload: Any) -> Dict[str, Any]:
    """Normalize a raw question or AdvCU/helper payload into a vision request contract."""
    request: Dict[str, Any] = {
        "text": "",
        "query_type": None,
        "requested_subject": None,
        "requested_attributes": [],
        "action_expectation": "answer_only",
        "module_hints": [],
    }

    payload = question_or_payload
    if isinstance(question_or_payload, dict):
        request["text"] = str(
            question_or_payload.get("text")
            or question_or_payload.get("raw_text")
            or question_or_payload.get("question")
            or ""
        ).strip()
        vision = None
        if isinstance(question_or_payload.get("vision"), dict):
            vision = question_or_payload.get("vision")
        elif isinstance(question_or_payload.get("helper_payload"), dict):
            hp = question_or_payload.get("helper_payload") or {}
            if isinstance(hp.get("vision"), dict):
                vision = hp.get("vision")
        elif isinstance(question_or_payload.get("semantic_packet"), dict):
            sp = question_or_payload.get("semantic_packet") or {}
            hp = sp.get("helper_payload") or {}
            if isinstance(hp.get("vision"), dict):
                vision = hp.get("vision")

        merged = vision or question_or_payload
        if isinstance(merged, dict):
            request["query_type"] = merged.get("query_type") or request["query_type"]
            request["requested_subject"] = merged.get("requested_subject") or merged.get("subject") or request["requested_subject"]
            attrs = merged.get("requested_attributes") or merged.get("attributes") or []
            if isinstance(attrs, (list, tuple, set)):
                request["requested_attributes"] = [str(x).strip().lower() for x in attrs if str(x).strip()]
            elif attrs:
                request["requested_attributes"] = [str(attrs).strip().lower()]
            request["action_expectation"] = str(merged.get("action_expectation") or request["action_expectation"]).strip()
            hints = merged.get("module_hints") or question_or_payload.get("module_hints") or []
            if isinstance(hints, (list, tuple, set)):
                request["module_hints"] = [str(x) for x in hints if str(x).strip()]
    else:
        request["text"] = str(question_or_payload or "").strip()

    q = request["text"].lower()
    parsed = sm_parse_visual_intent(request["text"])

    if not request["query_type"]:
        if any(s in q for s in ("read", "say", "text", "label", "sign", "shirt say", "screen say", "ocr")):
            request["query_type"] = "read_text"
        elif any(s in q for s in ("track", "moving", "motion", "follow")):
            request["query_type"] = "track_motion"
        elif any(s in q for s in ("distance", "how far", "how close", "too close", "near me", "near the machine", "danger zone", "safe zone", "hazard zone")):
            request["query_type"] = "inspect_safety_zone" if any(s in q for s in ("danger zone", "safe zone", "hazard zone")) else "estimate_distance"
        elif any(s in q for s in ("fit", "look fat", "look good", "look okay", "match", "style", "outfit")):
            request["query_type"] = "assess_style_or_fit"
        elif any(s in q for s in ("face", "eyes", "eye", "nose", "mouth", "beard", "glasses", "hat on my face")):
            request["query_type"] = "detect_faces"
        elif parsed.get("is_color_query") or " color " in f" {q} ":
            request["query_type"] = "identify_color"
        elif any(s in q for s in ("before and after", "compare", "difference", "changed")):
            request["query_type"] = "compare_before_after"
        elif any(s in q for s in ("what is in my hand", "what's in my hand", "what is in my hands", "what am i holding", "holding in my hand", "object in my hand")):
            request["query_type"] = "held_object"
        elif any(s in q for s in ("what do you see", "scene", "summarize", "around me", "environment")):
            request["query_type"] = "scene_summary"
        else:
            request["query_type"] = "detect_objects"

    if not request["requested_subject"]:
        parsed_target = parsed.get("target")
        if parsed_target in ("person", "screen", "object"):
            request["requested_subject"] = parsed_target
        elif parsed_target and _re.search(rf"\b{_re.escape(str(parsed_target))}\b", q):
            request["requested_subject"] = parsed_target

    subject = str(request.get("requested_subject") or "").strip().lower()
    request["requested_subject"] = subject or None

    if not request["requested_attributes"]:
        default_attrs = {
            "identify_color": ["color"],
            "read_text": ["text"],
            "detect_objects": ["objects"],
            "detect_faces": ["face"],
            "track_motion": ["motion"],
            "estimate_distance": ["distance"],
            "inspect_safety_zone": ["distance", "risk"],
            "compare_before_after": ["difference"],
            "assess_style_or_fit": ["style", "fit"],
            "scene_summary": ["scene"],
            "held_object": ["objects", "position"],
        }
        request["requested_attributes"] = list(default_attrs.get(request["query_type"], []))

    return request


def _safe_bbox_tuple(box: Any) -> Optional[Tuple[int, int, int, int]]:
    try:
        if box is None:
            return None
        if len(box) != 4:
            return None
        x, y, w, h = [int(v) for v in box]
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)
    except Exception:
        return None


def _largest_face_box(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    faces = _detect_face_boxes(frame)
    try:
        faces = list(faces) if faces is not None else []
    except Exception:
        faces = []
    if not faces:
        return None
    return max((_safe_bbox_tuple(f) for f in faces), key=lambda b: (b[2] * b[3]) if b else -1)


def _clamp_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    try:
        h, w = frame.shape[:2]
        x1 = max(0, min(w, int(x1)))
        y1 = max(0, min(h, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        if roi is None or roi.size == 0:
            return None
        return roi.copy()
    except Exception:
        return None


def _roi_color_stats(roi: Optional[np.ndarray]) -> Dict[str, Any]:
    if roi is None or getattr(roi, "size", 0) == 0:
        return {"name": "unknown", "confidence": 0.0, "rgb": [0, 0, 0], "hex": "#000000"}
    b, g, r = _dominant_bgr_color(roi)
    return {
        "name": _bgr_to_color_name((b, g, r)),
        "confidence": 0.65,
        "rgb": [int(r), int(g), int(b)],
        "hex": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
    }


def _collect_structured_detections(frame: np.ndarray, requested_subject: Optional[str] = None) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    requested_subject = _norm_token(requested_subject or "") if requested_subject else None
    models = _get_yolo_models_for_inference()
    if not models:
        return detections

    for model_name, model in models.items():
        try:
            results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
        except Exception as e:
            logger.warning(f"[Vision Detect] {model_name} inference failed: {e}")
            continue
        for r in results:
            names = getattr(r, "names", None) or getattr(model, "names", {}) or {}
            for b in getattr(r, "boxes", []) or []:
                try:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    raw_label = str(names.get(cls_id, "object")).lower().strip()
                except Exception:
                    continue
                domain, canon = _domain_of(raw_label)
                roi = _clamp_crop(frame, x1, y1, x2, y2)
                det = {
                    "raw_label": raw_label,
                    "label": canon,
                    "domain": domain,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "model": model_name,
                    "color": _roi_color_stats(roi),
                }
                detections.append(det)

    if requested_subject:
        def _score(det: Dict[str, Any]) -> Tuple[int, float]:
            label = str(det.get("label") or "")
            raw = str(det.get("raw_label") or "")
            score = 0
            if requested_subject == label or requested_subject == raw:
                score += 3
            if requested_subject in label or requested_subject in raw:
                score += 2
            if requested_subject in ("shirt", "pants", "jeans", "hat", "cap", "hoodie", "jacket") and label == "person":
                score += 1
            return (score, float(det.get("confidence") or 0.0))
        detections.sort(key=_score, reverse=True)
    else:
        detections.sort(key=lambda d: float(d.get("confidence") or 0.0), reverse=True)
    return detections


def _estimate_subject_roi(frame: np.ndarray, requested_subject: Optional[str], detections: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    requested_subject = _norm_token(requested_subject or "") if requested_subject else None
    detections = list(detections or [])
    meta: Dict[str, Any] = {"method": "none", "resolved_subject": requested_subject, "bbox": None}

    # Use direct detector match first.
    if requested_subject:
        for det in detections:
            label = str(det.get("label") or "")
            raw = str(det.get("raw_label") or "")
            if requested_subject == label or requested_subject == raw or requested_subject in label or requested_subject in raw:
                x1, y1, x2, y2 = [int(v) for v in det.get("bbox") or [0, 0, 0, 0]]
                roi = _clamp_crop(frame, x1, y1, x2, y2)
                if roi is not None:
                    meta.update({"method": "detector", "bbox": [x1, y1, x2, y2], "resolved_subject": label or requested_subject})
                    return roi, meta

    h, w = frame.shape[:2]
    face = _largest_face_box(frame)
    if face:
        fx, fy, fw, fh = face
        if requested_subject in ("face", "head", "eyes", "eye", "nose", "mouth", "beard", "mustache", "glasses"):
            roi = _clamp_crop(frame, fx, fy, fx + fw, fy + fh)
            if roi is not None:
                meta.update({"method": "face", "bbox": [fx, fy, fx + fw, fy + fh], "resolved_subject": requested_subject or "face"})
                return roi, meta
        if requested_subject in ("shirt", "t-shirt", "tee", "top", "hoodie", "jacket", "coat", "blouse", "upper garment"):
            roi = _clamp_crop(frame, fx - int(fw * 0.35), fy + fh, fx + fw + int(fw * 0.35), fy + fh + int(h * 0.28))
            if roi is not None:
                meta.update({"method": "face_relative_upper", "bbox": [max(0, fx - int(fw * 0.35)), fy + fh, min(w, fx + fw + int(fw * 0.35)), min(h, fy + fh + int(h * 0.28))], "resolved_subject": requested_subject})
                return roi, meta
        if requested_subject in ("pants", "trousers", "jeans", "shorts", "skirt", "lower garment"):
            roi = _clamp_crop(frame, fx - int(fw * 0.45), fy + fh + int(h * 0.18), fx + fw + int(fw * 0.45), fy + fh + int(h * 0.58))
            if roi is not None:
                meta.update({"method": "face_relative_lower", "bbox": [max(0, fx - int(fw * 0.45)), min(h, fy + fh + int(h * 0.18)), min(w, fx + fw + int(fw * 0.45)), min(h, fy + fh + int(h * 0.58))], "resolved_subject": requested_subject})
                return roi, meta
        if requested_subject in ("hat", "cap", "beanie", "hood", "headwear"):
            roi = _clamp_crop(frame, fx - int(fw * 0.2), fy - int(fh * 0.55), fx + fw + int(fw * 0.2), fy + int(fh * 0.25))
            if roi is not None:
                meta.update({"method": "face_relative_headwear", "bbox": [max(0, fx - int(fw * 0.2)), max(0, fy - int(fh * 0.55)), min(w, fx + fw + int(fw * 0.2)), min(h, fy + int(fh * 0.25))], "resolved_subject": requested_subject})
                return roi, meta

    # Generic fallback crops if no better ROI found.
    if requested_subject in ("shirt", "t-shirt", "tee", "top", "hoodie", "jacket", "coat", "blouse", "upper garment"):
        roi = _clamp_crop(frame, int(w * 0.2), int(h * 0.33), int(w * 0.8), int(h * 0.7))
        if roi is not None:
            meta.update({"method": "generic_upper_body", "bbox": [int(w * 0.2), int(h * 0.33), int(w * 0.8), int(h * 0.7)]})
            return roi, meta
    if requested_subject in ("pants", "trousers", "jeans", "shorts", "skirt", "lower garment"):
        roi = _clamp_crop(frame, int(w * 0.22), int(h * 0.5), int(w * 0.78), int(h * 0.95))
        if roi is not None:
            meta.update({"method": "generic_lower_body", "bbox": [int(w * 0.22), int(h * 0.5), int(w * 0.78), int(h * 0.95)]})
            return roi, meta

    whole = _clamp_crop(frame, 0, 0, w, h)
    if whole is not None:
        meta.update({"method": "frame", "bbox": [0, 0, w, h], "resolved_subject": requested_subject or "scene"})
    return whole, meta


def _read_text_from_roi(roi: Optional[np.ndarray]) -> Dict[str, Any]:
    result = {"text": None, "confidence": 0.0, "engine": None, "zoom": None}
    if roi is None or getattr(roi, "size", 0) == 0:
        return result
    txt = _try_pytesseract_text(roi)
    if txt:
        result.update({"text": txt, "confidence": 0.72, "engine": "pytesseract"})
        return result

    try:
        import easyocr as _easyocr
        reader = getattr(_read_text_from_roi, "_easy_reader", None)
        if reader is None:
            reader = _easyocr.Reader(['en'], gpu=False)
            _read_text_from_roi._easy_reader = reader
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = reader.readtext(roi_rgb, detail=0, paragraph=True)
        txt = " ".join([t for t in results if isinstance(t, str)]).strip()
        if txt:
            result.update({"text": txt, "confidence": 0.68, "engine": "easyocr"})
            return result
    except Exception:
        pass

    try:
        crop = smart_interest_crop(roi)
    except Exception:
        crop = None
    if crop is not None and crop.shape[0] > 24 and crop.shape[1] > 24:
        txt = _try_pytesseract_text(crop)
        if txt:
            result.update({"text": txt, "confidence": 0.64, "engine": "pytesseract", "zoom": "interest"})
            return result
        try:
            import easyocr as _easyocr
            reader = getattr(_read_text_from_roi, "_easy_reader", None)
            if reader is None:
                reader = _easyocr.Reader(['en'], gpu=False)
                _read_text_from_roi._easy_reader = reader
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = reader.readtext(crop_rgb, detail=0, paragraph=True)
            txt = " ".join([t for t in results if isinstance(t, str)]).strip()
            if txt:
                result.update({"text": txt, "confidence": 0.60, "engine": "easyocr", "zoom": "interest"})
                return result
        except Exception:
            pass
    return result


def _distance_from_bbox(frame: np.ndarray, bbox_xyxy: Optional[List[int]]) -> Dict[str, Any]:
    out = {"label": "unknown", "confidence": 0.0, "approx_ratio": 0.0}
    try:
        if not bbox_xyxy:
            return out
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        area_ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / float(max(1, w * h)))
        out["approx_ratio"] = round(area_ratio, 4)
        if area_ratio >= 0.24:
            out.update({"label": "very_close", "confidence": 0.70})
        elif area_ratio >= 0.12:
            out.update({"label": "close", "confidence": 0.66})
        elif area_ratio >= 0.04:
            out.update({"label": "medium", "confidence": 0.62})
        else:
            out.update({"label": "far", "confidence": 0.58})
    except Exception:
        pass
    return out


def _motion_findings(frame: np.ndarray) -> Dict[str, Any]:
    findings = {"available": False, "moving_regions": 0, "motion_detected": False, "confidence": 0.0}
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        prev = _VISION_RUNTIME_STATE.get("prev_gray")
        _VISION_RUNTIME_STATE["prev_gray"] = gray
        _VISION_RUNTIME_STATE["prev_timestamp"] = datetime.utcnow().isoformat()
        if prev is None:
            findings["available"] = False
            return findings
        delta = cv2.absdiff(prev, gray)
        thresh = cv2.threshold(delta, 12, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moving = [c for c in contours if cv2.contourArea(c) >= 500]
        findings.update({
            "available": True,
            "moving_regions": len(moving),
            "motion_detected": bool(moving),
            "confidence": 0.55 + min(0.25, len(moving) * 0.08),
        })
        return findings
    except Exception as e:
        logger.warning(f"[Vision Motion] failed: {e}")
        return findings



def _area_of_bbox_xyxy(box: Any) -> int:
    try:
        x1, y1, x2, y2 = [int(v) for v in (box or [0, 0, 0, 0])]
        return max(0, x2 - x1) * max(0, y2 - y1)
    except Exception:
        return 0


def _bbox_center_xy(box: Any) -> Tuple[float, float]:
    try:
        x1, y1, x2, y2 = [float(v) for v in (box or [0, 0, 0, 0])]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    except Exception:
        return (0.0, 0.0)


def _bbox_iou_xyxy(a: Any, b: Any) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in (a or [0, 0, 0, 0])]
        bx1, by1, bx2, by2 = [float(v) for v in (b or [0, 0, 0, 0])]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0 else 0.0
    except Exception:
        return 0.0


def _estimate_hand_regions(frame: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None) -> List[Dict[str, Any]]:
    """Return bounded hand/holding candidate zones without opening any device.

    This is a fallback relationship heuristic for questions like "what is in my hand".
    If a detector supplies explicit hand/person/object boxes, those stay preferred.
    """
    regions: List[Dict[str, Any]] = []
    try:
        h, w = frame.shape[:2]
        if face_box:
            fx, fy, fw, fh = face_box
            y1 = min(h, fy + int(fh * 1.15))
            y2 = min(h, fy + int(fh * 3.2))
            span = max(fw, int(w * 0.16))
            regions.append({"label": "left_hand_region", "bbox": [max(0, fx - span), y1, min(w, fx + int(fw * 0.25)), y2], "method": "face_relative"})
            regions.append({"label": "right_hand_region", "bbox": [max(0, fx + int(fw * 0.75)), y1, min(w, fx + fw + span), y2], "method": "face_relative"})
        # Generic lower-mid zones for phone/cloud webcam framing when face/hand detector is unavailable.
        regions.append({"label": "lower_left_hand_region", "bbox": [0, int(h * 0.45), int(w * 0.55), h], "method": "generic_frame"})
        regions.append({"label": "lower_right_hand_region", "bbox": [int(w * 0.45), int(h * 0.45), w, h], "method": "generic_frame"})
    except Exception:
        pass
    clean: List[Dict[str, Any]] = []
    for item in regions:
        bbox = item.get("bbox") if isinstance(item, dict) else None
        if _area_of_bbox_xyxy(bbox) > 0:
            clean.append(item)
    return clean


def _detect_held_object(frame: np.ndarray, detections: List[Dict[str, Any]], face_box: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    hand_regions = _estimate_hand_regions(frame, face_box)
    candidates: List[Dict[str, Any]] = []
    ignored_labels = {"person", "face", "head", "hand", "hands", "arm", "arms"}
    for det in list(detections or []):
        label = str(det.get("label") or det.get("raw_label") or "object").strip().lower()
        if not label or label in ignored_labels:
            continue
        bbox = det.get("bbox") if isinstance(det.get("bbox"), list) else None
        if not bbox:
            continue
        best_score = 0.0
        best_region = None
        for region in hand_regions:
            rb = region.get("bbox")
            iou = _bbox_iou_xyxy(bbox, rb)
            cx, cy = _bbox_center_xy(bbox)
            rx1, ry1, rx2, ry2 = [float(v) for v in rb]
            inside = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
            score = iou + (0.35 if inside else 0.0) + min(0.25, float(det.get("confidence") or 0.0) * 0.25)
            if score > best_score:
                best_score = score
                best_region = region
        if best_score > 0.10:
            item = dict(det)
            item["held_score"] = round(best_score, 4)
            item["hand_region"] = best_region
            candidates.append(item)
    candidates.sort(key=lambda d: (float(d.get("held_score") or 0.0), float(d.get("confidence") or 0.0)), reverse=True)
    if candidates:
        best = candidates[0]
        return {
            "visible": True,
            "object": best.get("label") or best.get("raw_label") or "object",
            "raw_label": best.get("raw_label") or best.get("label") or "object",
            "confidence": max(0.42, min(0.82, float(best.get("confidence") or 0.45) + 0.10)),
            "bbox": best.get("bbox"),
            "color": best.get("color"),
            "hand_region": best.get("hand_region"),
            "candidates": candidates[:5],
        }
    return {
        "visible": False,
        "object": None,
        "confidence": 0.25 if hand_regions else 0.0,
        "hand_regions": hand_regions[:4],
        "candidates": [],
    }

def _build_visual_findings(question_or_payload: Any, frame: Optional[np.ndarray]) -> Dict[str, Any]:
    request = _normalize_visual_request(question_or_payload)
    findings: Dict[str, Any] = {
        "vision_request": request,
        "sensor_status": {
            "camera": frame is not None,
            "object_detection": bool(OBJECT_DETECTION_ENABLED),
            "yolo_models_enabled": bool(_enabled_yolo_model_names()),
            "ocr": True,
            "face_detection": True,
        },
        "resolved_subject": request.get("requested_subject"),
        "subject_candidates": [],
        "observed_subjects": [],
        "detections": [],
        "faces": {"count": 0, "glasses": None},
        "ocr": {"text": None, "confidence": 0.0},
        "color": {"name": None, "confidence": 0.0},
        "motion": {"available": False, "moving_regions": 0, "motion_detected": False, "confidence": 0.0},
        "distance": {"label": "unknown", "confidence": 0.0, "approx_ratio": 0.0},
        "safety": {"hazard_zone_violation": None, "configured": False, "message": None},
        "style": {"assessment": None, "confidence": 0.0},
        "held_object": {"visible": False, "object": None, "confidence": 0.0},
        "confidence": 0.0,
        "errors": [],
    }

    if frame is None:
        findings["errors"].append("no_frame")
        return findings

    requested_subject = request.get("requested_subject")
    query_type = request.get("query_type")

    detections = _collect_structured_detections(frame, requested_subject=requested_subject)
    findings["detections"] = detections
    findings["observed_subjects"] = [d.get("label") for d in detections[:12] if d.get("label")]
    findings["subject_candidates"] = list(dict.fromkeys(findings["observed_subjects"]))

    face = _largest_face_box(frame)
    faces = _detect_face_boxes(frame) or []
    try:
        face_count = len(list(faces))
    except Exception:
        face_count = 0
    findings["faces"]["count"] = face_count
    if face:
        findings["resolved_subject"] = findings["resolved_subject"] or "face"
        findings["faces"]["glasses"] = _detect_eyeglasses(frame, face)
        findings["distance"] = _distance_from_bbox(frame, [face[0], face[1], face[0] + face[2], face[1] + face[3]])

    roi, roi_meta = _estimate_subject_roi(frame, requested_subject, detections=detections)
    if roi_meta.get("resolved_subject"):
        findings["resolved_subject"] = roi_meta.get("resolved_subject")
    if roi_meta.get("resolved_subject") and roi_meta.get("resolved_subject") not in findings["subject_candidates"]:
        findings["subject_candidates"].insert(0, roi_meta.get("resolved_subject"))

    if query_type == "identify_color":
        color_roi = roi
        try:
            focus_crop = smart_interest_crop(roi) if roi is not None else None
            if focus_crop is not None and getattr(focus_crop, "size", 0) > 0:
                color_roi = focus_crop
        except Exception:
            color_roi = roi
        color = _roi_color_stats(color_roi)
        findings["color"] = color
        findings["confidence"] = float(color.get("confidence") or 0.0)

    elif query_type == "read_text":
        ocr = _read_text_from_roi(roi)
        findings["ocr"] = ocr
        findings["confidence"] = float(ocr.get("confidence") or 0.0)

    elif query_type == "detect_faces":
        findings["confidence"] = 0.72 if face_count else 0.30

    elif query_type == "track_motion":
        findings["motion"] = _motion_findings(frame)
        findings["confidence"] = float(findings["motion"].get("confidence") or 0.0)

    elif query_type == "estimate_distance":
        if detections and not face:
            best = detections[0]
            findings["distance"] = _distance_from_bbox(frame, best.get("bbox"))
            findings["resolved_subject"] = best.get("label") or findings["resolved_subject"]
        findings["confidence"] = float(findings["distance"].get("confidence") or 0.0)

    elif query_type == "held_object":
        held = _detect_held_object(frame, detections, face_box=face)
        findings["held_object"] = held
        if held.get("object"):
            findings["resolved_subject"] = held.get("object")
            if held.get("object") not in findings["subject_candidates"]:
                findings["subject_candidates"].insert(0, held.get("object"))
        findings["confidence"] = float(held.get("confidence") or 0.0)

    elif query_type == "inspect_safety_zone":
        if detections and not face:
            best = detections[0]
            findings["distance"] = _distance_from_bbox(frame, best.get("bbox"))
            findings["resolved_subject"] = best.get("label") or findings["resolved_subject"]
        dist_label = str(findings["distance"].get("label") or "unknown")
        findings["safety"] = {
            "hazard_zone_violation": True if dist_label in ("very_close", "close") else False if dist_label in ("medium", "far") else None,
            "configured": False,
            "message": "No explicit hazard-zone geometry is configured in SarahMemorySOBJE.py; using proximity-only advisory."
        }
        findings["confidence"] = max(float(findings["distance"].get("confidence") or 0.0), 0.55)

    elif query_type == "compare_before_after":
        prev = _VISION_RUNTIME_STATE.get("last_findings") or {}
        previous_subjects = prev.get("observed_subjects") or []
        current_subjects = findings.get("observed_subjects") or []
        added = [s for s in current_subjects if s not in previous_subjects]
        removed = [s for s in previous_subjects if s not in current_subjects]
        findings["comparison"] = {
            "added": added,
            "removed": removed,
            "baseline_available": bool(prev),
        }
        findings["confidence"] = 0.58 if prev else 0.34

    elif query_type == "assess_style_or_fit":
        assessment = "unable_to_assess_confidently"
        conf = 0.28
        if roi is not None and getattr(roi, "size", 0) > 0:
            rh, rw = roi.shape[:2]
            coverage = (rh * rw) / float(max(1, frame.shape[0] * frame.shape[1]))
            if coverage >= 0.18:
                assessment = "garment_visible"
                conf = 0.50
            if requested_subject in ("shirt", "t-shirt", "tee", "top", "hoodie", "jacket", "coat"):
                assessment = "fit_assessment_requires_user_preference_and_pose"
                conf = max(conf, 0.42)
        findings["style"] = {"assessment": assessment, "confidence": conf}
        findings["confidence"] = conf

    elif query_type in ("scene_summary", "detect_objects"):
        findings["confidence"] = 0.66 if detections else 0.35

    if not findings["confidence"]:
        findings["confidence"] = 0.45 if detections else 0.30

    _VISION_RUNTIME_STATE["last_findings"] = findings
    return findings


def _answer_from_visual_findings(findings: Dict[str, Any]) -> str:
    request = findings.get("vision_request") or {}
    query_type = str(request.get("query_type") or "").strip()
    subject = str(findings.get("resolved_subject") or request.get("requested_subject") or "object").strip()

    if "no_frame" in (findings.get("errors") or []):
        return "I don't have a camera frame yet."

    if query_type == "identify_color":
        color = (findings.get("color") or {}).get("name")
        if color:
            return f"The color of your {subject} looks {color}." if subject not in ("scene", "object") else f"It looks {color}."
        return "I couldn't determine the color clearly yet."

    if query_type == "read_text":
        text = (findings.get("ocr") or {}).get("text")
        if text:
            return f'It says: "{text}".'
        return "I couldn't read that clearly. Try moving a little closer or steadier lighting."

    if query_type == "detect_faces":
        faces = findings.get("faces") or {}
        count = int(faces.get("count") or 0)
        glasses = faces.get("glasses")
        if count <= 0:
            return "I can't see your face clearly right now."
        if subject in ("glasses", "face", "eyes", "eye") and glasses is True:
            return "You're wearing glasses."
        if subject in ("glasses", "face", "eyes", "eye") and glasses is False:
            return "I don't detect glasses."
        return f"I detect {count} face{'s' if count != 1 else ''}."

    if query_type == "track_motion":
        motion = findings.get("motion") or {}
        if not motion.get("available"):
            return "I need another frame or live feed to judge motion reliably."
        if motion.get("motion_detected"):
            return f"I detect motion in about {int(motion.get('moving_regions') or 0)} region(s)."
        return "I don't detect meaningful motion right now."

    if query_type == "estimate_distance":
        dist = findings.get("distance") or {}
        label = str(dist.get("label") or "unknown").replace("_", " ")
        if label == "unknown":
            return "I can't estimate the distance confidently from this frame."
        return f"The {subject} appears {label}."

    if query_type == "held_object":
        held = findings.get("held_object") or {}
        obj = held.get("object")
        if obj:
            color = held.get("color") if isinstance(held.get("color"), dict) else {}
            color_name = color.get("name") if isinstance(color, dict) else None
            if color_name and color_name != "unknown":
                return f"You appear to be holding a {color_name} {obj}."
            return f"You appear to be holding a {obj}."
        if (held.get("hand_regions") or held.get("candidates")):
            return "I can inspect the hand region, but I cannot verify the object in your hand clearly enough yet."
        return "I cannot verify a hand-held object from this frame yet."

    if query_type == "inspect_safety_zone":
        safety = findings.get("safety") or {}
        dist = findings.get("distance") or {}
        label = str(dist.get("label") or "unknown").replace("_", " ")
        violation = safety.get("hazard_zone_violation")
        if violation is True:
            return f"The {subject} appears too close for a safe advisory check. No explicit hazard-zone map is configured yet, so this is proximity-only guidance."
        if violation is False:
            return f"The {subject} appears {label}; no proximity hazard is suggested from this frame alone."
        return "I need a clearer view and a configured safety zone to assess that properly."

    if query_type == "compare_before_after":
        comp = findings.get("comparison") or {}
        if not comp.get("baseline_available"):
            return "I need a prior visual baseline before I can compare before and after."
        added = comp.get("added") or []
        removed = comp.get("removed") or []
        bits = []
        if added:
            bits.append("new: " + ", ".join(added[:4]))
        if removed:
            bits.append("gone: " + ", ".join(removed[:4]))
        return "I compared the scene. " + ("; ".join(bits) if bits else "I do not see a major visible difference.")

    if query_type == "assess_style_or_fit":
        style = findings.get("style") or {}
        assessment = str(style.get("assessment") or "unable_to_assess_confidently").replace("_", " ")
        if assessment == "unable to assess confidently":
            return "I can't judge the style or fit confidently from this frame yet."
        if "requires user preference" in assessment:
            return "I can see the garment, but style and fit depend on pose, angle, and your preference. I would need a clearer front view for a better assessment."
        return f"I can see the {subject}; preliminary style assessment: {assessment}."

    if query_type in ("scene_summary", "detect_objects"):
        observed = findings.get("observed_subjects") or []
        if observed:
            return "I see " + ", ".join(observed[:8]) + "."
        return "I can't see anything clearly right now."

    return "I understood the visual request, but I need a clearer frame or more specific target."


def answer_visual_question(question, frame):
    """
    Backward-compatible visual Q&A wrapper.

    Accepts either:
    - raw text question (legacy GUI path)
    - semantic/helper payload dict from AdvCU/Neuron (new governed path)

    Returns dict {answer, details} while exposing structured findings for the next-stage
    governed Reply integration.
    """
    findings = _build_visual_findings(question, frame)
    answer = _answer_from_visual_findings(findings)
    details = {
        "request": findings.get("vision_request") or {},
        "resolved_subject": findings.get("resolved_subject"),
        "subject_candidates": findings.get("subject_candidates") or [],
        "observed_subjects": findings.get("observed_subjects") or [],
        "detections": findings.get("detections") or [],
        "faces": findings.get("faces") or {},
        "ocr": findings.get("ocr") or {},
        "color": findings.get("color") or {},
        "motion": findings.get("motion") or {},
        "distance": findings.get("distance") or {},
        "safety": findings.get("safety") or {},
        "style": findings.get("style") or {},
        "held_object": findings.get("held_object") or {},
        "sensor_status": findings.get("sensor_status") or {},
        "confidence": findings.get("confidence") or 0.0,
        "errors": findings.get("errors") or [],
        "findings": findings,
    }
    return {"answer": answer, "details": details}

# ====================================================================
# END OF SarahMemorySOBJE.py v8.0.0
# ====================================================================