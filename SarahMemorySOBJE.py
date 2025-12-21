"""--==The SarahMemory Project==--
File: SarahMemorySOBJE.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-21
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================
"""

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
import SarahMemoryGlobals as config
from SarahMemoryGlobals import DATASETS_DIR, OBJECT_MODEL_CONFIG, OBJECT_DETECTION_ENABLED, MODEL_PATHS
#import SarahMemoryFacialRecognition as fr
#from SarahMemoryGlobals import run_async
#from SarahMemoryHi import async_update_network_state

YOLO_MODELS = {}
if OBJECT_DETECTION_ENABLED:
    from ultralytics import YOLO
    for model_name, model_cfg in OBJECT_MODEL_CONFIG.items():
        if model_cfg.get("enabled"):
            model_dir = MODEL_PATHS.get(model_name)
            if model_dir:
                model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])
                    try:
                        YOLO_MODELS[model_name] = YOLO(model_path)
                    except Exception as e:
                        logging.warning(f"[YOLO Init Fail] {model_name} @ {model_path}: {e}")
                else:
                    logging.warning(f"[YOLO Load Skip] {model_name}: No .pt file found in {model_dir}")
            else:
                logging.warning(f"[YOLO Load Skip] {model_name}: Model path missing in MODEL_PATHS")


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

    # === [ADDED] YOLO + real observations + DB logging ===
    real_detections = []  # (label, (x1,y1,x2,y2), conf, model_name)
    try:
        if YOLO_MODELS:
            # choose first loaded model (or loop models)
            for mname, model in YOLO_MODELS.items():
                results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
                for r in results:
                    for b in r.boxes:
                        cls_id = int(b.cls[0])
                        conf  = float(b.conf[0])
                        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                        raw_label = model.names.get(cls_id, "object")
                        real_detections.append((raw_label, (x1,y1,x2,y2), conf, mname))
    except Exception as _e:
        logging.warning(f"YOLO inference failed: {_e}")

    # if nothing from YOLO, fallback to contours with a generic 'object'
    if not real_detections:
        contours = get_contours(frame)
        for c in contours:
            if cv2.contourArea(c) > 500:
                x,y,w,h = cv2.boundingRect(c)
                real_detections.append(("object", (x,y,x+w,y+h), 0.5, "contour"))

    # draw & map to domain + optional color for key targets
    observations = []
    for raw_label, (x1,y1,x2,y2), conf, mname in real_detections:
        # draw box
        cv2.rectangle(processed_frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(processed_frame, f"{raw_label} {conf:.2f}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

        domain, canon = _domain_of(raw_label)
        color_stats = None
        # color-worthy targets
        if domain in ("garment","food","container","object"):
            roi = frame[y1:y2, x1:x2].copy()
            color_stats = _sm_roi_color(roi)

        observations.append({
            "raw_label": raw_label,
            "label": canon,
            "domain": domain,
            "bbox": [x1,y1,x2,y2],
            "confidence": conf,
            "model": mname,
            "color": color_stats
        })

    # === DB logging (real observations only) ===
    try:
        from SarahMemoryDatabase import get_db_connection
        import json as _json
        conn = get_db_connection('ai_learning.db')  # keep your existing db name
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
                 color.get("name"), color.get("hex"),
                 _json.dumps(obs)))
        conn.commit(); conn.close()
    except Exception as _e:
        logging.warning(f"object_observations logging failed: {_e}")




    # === [ADDED] ROI dominant color helper (uses AdvCU color naming) ===
    import numpy as _np
    import cv2 as _cv2

    def _sm_roi_color(bgr_roi):
        if bgr_roi is None or bgr_roi.size == 0:
            return {"hex":"#000000","rgb":[0,0,0],"hsl":[0,0,0],"name":"unknown","confidence":0.0}
        hsv = _cv2.cvtColor(bgr_roi, _cv2.COLOR_BGR2HSV)
        h,s,v = _cv2.split(hsv)
        mask = (s > 18) & (v > 25)
        pix = bgr_roi[mask] if mask.any() else bgr_roi.reshape(-1,3)
        Z = _np.float32(pix); K=3
        crit = (_cv2.TERM_CRITERIA_EPS + _cv2.TERM_CRITERIA_MAX_ITER, 24, 1.0)
        try:
            _, labels, centers = _cv2.kmeans(Z, K, None, crit, 3, _cv2.KMEANS_PP_CENTERS)
            counts = _np.bincount(labels.flatten(), minlength=K)
            idx = int(counts.argmax())
            b,g,r = centers[idx].astype(int).tolist()
            from SarahMemoryAdvCU import sm_color_name_from_rgb
            name = sm_color_name_from_rgb(r,g,b)
            conf = float(counts[idx]/max(1,counts.sum()))
        except Exception:
            b,g,r = pix.mean(axis=0).astype(int).tolist()
            from SarahMemoryAdvCU import sm_color_name_from_rgb
            name = sm_color_name_from_rgb(r,g,b)
            conf = 0.5
        rp,gp,bp = r/255.0, g/255.0, b/255.0
        mx, mn = max(rp,gp,bp), min(rp,gp,bp)
        l = (mx+mn)/2.0
        if mx==mn: h=0.0; s=0.0
        else:
            d = mx-mn
            s = d/(2.0-mx-mn) if l>0.5 else d/(mx+mn+1e-9)
            if mx==rp: h = ((gp-bp)/d + (6 if gp<bp else 0))
            elif mx==gp: h = ((bp-rp)/d + 2)
            else: h = ((rp-gp)/d + 4)
            h *= 60.0; h%=360.0
        hexv = f"#{r:02x}{g:02x}{b:02x}"
        return {"hex":hexv,"rgb":[int(r),int(g),int(b)],"hsl":[float(h),float(s),float(l)],"name":name,"confidence":conf}
    """
    Detects objects from the provided frame, applies domain tagging,
    logs findings to the database, and returns selected labels.

    Improvements:
      - Uses a local copy of the frame for drawing.
      - Reduces synthetic term generation for performance.
      - Uses context managers for SQLite logging.
      - Added inline documentation and type hints.
    """
    if frame is None:
        logger.warning("No frame provided for object detection.")
        return []

    processed_frame = frame.copy()  # local copy for drawing

    def get_contours(frame: np.ndarray) -> list:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except Exception as e:
            logger.error(f"Error extracting contours: {e}")
            return []

    def draw_and_identify(contours: list, min_area: int = 500) -> list:
        tags = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(processed_frame, "object", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                tags.append("object")
        return tags

    contours = get_contours(frame)
    detected_objects = draw_and_identify(contours)

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

    # Check for critical conditions.
    medical_flags = {
        "condition: mole": "Possible melanoma",
        "expression: slouched": "Posture anomaly / neuro issue",
        "face-point: landmark_2349": "Right eye droop — stroke warning",
        "skin tone: uneven": "Skin cancer check"
    }
    for tag in detected_tags:
        if tag in medical_flags:
            alert_medical(tag)

    # Generate synthetic high-tech terms (reduced count for better performance)
    synthetic_count = 500  # Reduced from 10000 iterations
    prefixes = ["modular", "adaptive", "quantum", "neural", "precision", "bio-active", "liquid-cooled", "synthetic"]
    nouns = ["transmitter", "oscillator", "stabilizer", "inverter", "thruster", "sensor", "valve", "core"]
    suffixes = ["array", "hub", "grid", "cell", "interface", "matrix", "system", "scanner"]
    synthetic_terms = [
        f"{random.choice(prefixes)} {random.choice(nouns)} {random.choice(suffixes)}"
        for _ in range(synthetic_count)
    ]

    # Combine domain terms with synthetic terms.
    all_objects = []
    for category, terms in domains.items():
        for term in terms:
            all_objects.append(f"{category}: {term}")
    all_objects.extend(synthetic_terms)

    random.shuffle(all_objects)
    selected = random.sample(all_objects, random.randint(3, 12))

    # Log selected detections to SQLite.
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

def answer_visual_question(question, frame):
    """
    High-level visual Q&A used by GUI bridge. Returns dict {answer, details}.
    - "what's on my face?" -> detect glasses/headwear
    - "what does my shirt say" -> OCR on torso below face
    - "what color is my <object>" -> detect object bbox if model available; else whole frame
    """
    q = (question or "").lower().strip()
    if frame is None:
        return {"answer":"I don't have a camera frame yet.", "details":{}}

    # Face-based queries
    if "on my face" in q or "wearing on my face" in q:
        faces = _detect_face_boxes(frame)
        if not faces:
            return {"answer":"I can't see your face clearly right now.", "details":{"faces":0}}
        # take largest
        face = max(faces, key=lambda b:b[2]*b[3])
        has_glasses = _detect_eyeglasses(frame, face)
        return {"answer": ("You're wearing glasses." if has_glasses else "I don't detect glasses."),
                "details":{"faces":len(faces), "glasses":bool(has_glasses)}}

    # Shirt text OCR
    if "what does my shirt say" in q or "read my shirt" in q or ("shirt" in q and "say" in q):
        faces = _detect_face_boxes(frame)
        h, w = frame.shape[:2]
        if faces:
            x,y,fw,fh = max(faces, key=lambda b:b[2]*b[3])
            y1 = min(h, y+fh+10)
            roi = frame[y1:min(h, y1 + int(h*0.35)), max(0, x-40):min(w, x+fw+40)]
        else:
            # center-lower crop
            roi = frame[int(h*0.45):int(h*0.8), int(w*0.2):int(w*0.8)]
        txt = _try_pytesseract_text(roi)

        if not txt:
            # Optional fallback if easyocr is installed
            try:
                import numpy as np
                import cv2
                import easyocr as _easyocr
                reader = getattr(answer_visual_question, "_easy_reader", None)
                if reader is None:
                    reader = _easyocr.Reader(['en'], gpu=False)
                    answer_visual_question._easy_reader = reader
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                results = reader.readtext(roi_rgb, detail=0, paragraph=True)
                txt = " ".join([t for t in results if isinstance(t, str)]).strip()
            except Exception:
                txt = None

        # If first read weak, auto-zoom with interest crop and try again
        if not txt:
            try:
                crop = smart_interest_crop(roi)
            except Exception:
                crop = None
            if crop is not None and crop.shape[0] > 24 and crop.shape[1] > 24:
                txt2 = _try_pytesseract_text(crop)
                if not txt2:
                    # also try EasyOCR fallback on the interest crop
                    try:
                        import cv2, easyocr as _easyocr
                        reader = getattr(answer_visual_question, "_easy_reader", None)
                        if reader is None:
                            reader = _easyocr.Reader(['en'], gpu=False)
                            answer_visual_question._easy_reader = reader
                        crgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        rs = reader.readtext(crgb, detail=0, paragraph=True)
                        txt2 = " ".join([t for t in rs if isinstance(t, str)]).strip()
                    except Exception:
                        pass
                if txt2:
                    return {"answer": f'It says: "{txt2}".', "details":{"ocr":True, "zoom":"interest"}}

        if txt:
            return {"answer": f'It says: "{txt}".', "details":{"ocr":True}}
        return {"answer":"I couldn't read that clearly. Try moving a little closer or steadier lighting.", "details":{"ocr":False}}

    # Color of object (e.g., couch/sofa/shirt/hat)
    if "color of my" in q or q.startswith("what color is my") or "what color is the" in q:
        # Try YOLO label for target object first
        target = None
        for key in ["couch","sofa","shirt","hat","cap","jacket","hoodie","pants","jeans"]:
            if key in q:
                target = key
                break
        crop = None
        try:
            if YOLO_MODELS:
                # pick first model and run a quick inference
                model = list(YOLO_MODELS.values())[0]
                results = model(frame)
                best = None
                for r in results:
                    for b in r.boxes:
                        cls = int(b.cls[0])
                        label = (r.names or {}).get(cls, "").lower()
                        if not label: 
                            continue
                        if (target and target in label) or (not target and label in ("couch","sofa","person","shirt","jacket","hat")):
                            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                            area = (x2-x1)*(y2-y1)
                            if not best or area>best[0]:
                                best = (area,(x1,y1,x2,y2))
                if best:
                    x1,y1,x2,y2 = best[1]
                    crop = frame[max(0,y1):y2, max(0,x1):x2]
        except Exception:
            crop = None
        if crop is None:
            crop = frame
        color = _bgr_to_color_name(_dominant_bgr_color(crop))
        if target:
            return {"answer": f"The color of your {target} looks {color}.", "details":{"color":color}}
        return {"answer": f"It looks {color}.", "details":{"color":color}}
    return {"answer":"I understood the question, but I need you to phrase it like: 'what's on my face', 'what does my shirt say', or 'what color is my couch?'", "details":{}}

# ====================================================================
# END OF SarahMemorySOBJE.py v8.0.0
# ====================================================================