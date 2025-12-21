"""--==The SarahMemory Project==--
File: SarahMemoryLLM.py
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

import os
import logging
import subprocess
import time
import urllib.request
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from SarahMemoryGlobals import BASE_DIR, MODEL_CONFIG, MODELS_DIR

# Updated object detection model config
OBJECT_MODEL_CONFIG = {
    "YOLOv8 <works>": {"enabled": True, "repo": "ultralytics_yolov8", "hf_repo": "ultralytics/yolov8", "require": "ultralytics"},
    "YOLOv5 <works>": {"enabled": True, "repo": "ultralytics_yolov5", "hf_repo": "ultralytics/yolov5", "require": None},
    "DETR <works>": {"enabled": True, "repo": "facebook_detr", "hf_repo": "facebook/detr-resnet-50", "require": None},
    "YOLOv7 <not tested>": {"enabled": False, "repo": "ultralytics_yolov7", "hf_repo": "WongKinYiu/yolov7", "require": None},
    "YOLO-NAS <no joy>": {
        "enabled": False,
        "repo": "Deci-AI_yolo-nas",
        "hf_repo": "https://github.com/naseemap47/YOLO-NAS",
        "require": "super-gradients",
        "weights": [
            {
                "url": "https://deci-pretrained-models.s3.amazonaws.com/yolo_nas/coco/yolo_nas_s.pth",
                "filename": "yolo_nas_s.pth"
            }
        ]
    },
    "YOLOX <not tested>": {"enabled": False, "repo": "MegviiBaseDetection_YOLOX", "hf_repo": "Megvii-BaseDetection/YOLOX", "require": None},
    "PP-YOLOv2 <not tested>": {"enabled": False, "repo": "PaddleDetection", "hf_repo": "PaddlePaddle/PaddleDetection", "require": "paddlepaddle"},
    "EfficientDet <not tested>": {"enabled": False, "repo": "automl_efficientdet", "hf_repo": "zylo117/Yet-Another-EfficientDet-Pytorch", "require": None},
    "DINO <works>": {"enabled": True, "repo": "facebook_dinov2", "hf_repo": "facebook/dinov2-base", "require": None},
    "CenterNet <not tested>": {"enabled": False, "repo": "CenterNet", "hf_repo": "xingyizhou/CenterNet", "require": None},
    "SSD<works>": {"enabled": True, "repo": "qfgaohao_pytorch-ssd", "hf_repo": "https://github.com/qfgaohao/pytorch-ssd", "require": None,
                   "weights": [
            {   "url": "https://github.com/qfgaohao/pytorch-ssd/releases/download/v1.0/mobilenet-v1-ssd-mp-0_675.pth",
                "filename": "mobilenet-v1-ssd-mp-0_675.pth"
            },
            {   "url": "https://github.com/qfgaohao/pytorch-ssd/releases/download/v1.0/voc-model-labels.txt",
                "filename": "voc-model-labels.txt"
            }
        ]
    },
    "Faster R-CNN <not tested>": {"enabled": False,"repo": "facebook_detectron2", "hf_repo": "https://github.com/facebookresearch/detectron2", "require": None, "weights": []
    },
    "RetinaNet <not tested>": {"enabled": False,"repo": "facebook_detectron2","hf_repo": "https://github.com/facebookresearch/detectron2","require": None,"weights": []
    },
}
# [patched] basicConfig moved to SarahMemoryMain.py
LOG_PATH = os.path.join(BASE_DIR, "data", "logs", "model_download.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

try:
    import hf_xet
except ImportError:
    subprocess.run(["pip", "install", "huggingface_hub[hf_xet]"], check=False)

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log_error(message):
    with open(LOG_PATH, 'a') as f:
        f.write(f"[ERROR] {message}\n")

def retry(func, retries=8, wait=20):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            log_error(f"Retry {attempt + 1} failed: {e}")
            time.sleep(wait)
    raise Exception("Max retries reached")

def model_files_valid(local_dir):
    try:
        if not os.path.isdir(local_dir):
            return False
        files = os.listdir(local_dir)
        if not any(f.endswith(('.bin', '.safetensors', '.pth')) for f in files):
            return False
        for f in files:
            full_path = os.path.join(local_dir, f)
            if os.path.isfile(full_path) and os.path.getsize(full_path) == 0:
                return False
        return True
    except Exception as e:
        log_error(f"VALIDATION ERROR {local_dir}: {e}")
        return False

def pip_install_if_needed(package_name):
    try:
        subprocess.run(["pip", "install", package_name], check=True)
        print(f"[INSTALLED] {package_name}")
    except subprocess.CalledProcessError as e:
        log_error(f"Pip install failed for {package_name}: {e}")

def download_model(model_name, local_dir, use_sentence_transformer=False):
    def inner():
        if model_files_valid(local_dir):
            print(f"[✔] {model_name} already downloaded.")
            return
        print(f"[→] Downloading {model_name} → {local_dir}")
        if use_sentence_transformer:
            SentenceTransformer(model_name, cache_folder=local_dir)
        else:
            AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
            AutoModel.from_pretrained(model_name, cache_dir=local_dir)
        print(f"[✔] Completed {model_name}")
    retry(inner)

def download_object_model(name, hf_repo_url, local_dir):
    def inner():
        ensure_directory(local_dir)
        config = OBJECT_MODEL_CONFIG.get(name, {})

        if hf_repo_url:
            print(f"[→] Downloading {name} from {hf_repo_url}")
            if os.path.exists(local_dir) and os.listdir(local_dir):
                print(f"[SKIPPED] {name} destination folder already exists and is not empty.")
            elif hf_repo_url.startswith("http"):
                subprocess.run(["git", "clone", hf_repo_url, local_dir], check=True)
            else:
                snapshot_download(hf_repo_url, local_dir=local_dir, ignore_patterns=['*.md', '*.txt'])
            print(f"[✔] Completed {name}")

        weights = config.get("weights", [])
        for w in weights:
            dest_path = os.path.join(local_dir, w["filename"])
            if not os.path.exists(dest_path):
                print(f"[↓] Downloading {w['filename']} for {name}")
                urllib.request.urlretrieve(w["url"], dest_path)
                print(f"[✔] Downloaded {w['filename']} to {dest_path}")
            else:
                print(f"[✔] Weight file {w['filename']} already exists")
    retry(inner)

def cli_menu():
    print("""
===== SarahMemory Model Setup Menu by Brian Baros =====
[1] Download ALL Models
[2] Download SELECTED LLM Models
[3] Download SELECTED Object Detection Models
[4] Install from requirements.txt
[5] Exit
""")
    return input("Enter your choice (1-5): ").strip()

def list_models(config_dict):
    print("\nAvailable Models:")
    selected = []
    for i, (name, enabled) in enumerate(config_dict.items(), 1):
        status = "✔" if model_files_valid(os.path.join(MODELS_DIR, name.replace('/', '_'))) else "❌"
        print(f"[{i}] {name} ({status})")
    print("[0] Done\n")
    while True:
        choice = input("Select model # (0 to finish): ").strip()
        if choice == '0':
            break
        try:
            idx = int(choice) - 1
            key = list(config_dict.keys())[idx]
            selected.append(key)
        except:
            print("Invalid selection.")
    return selected

def main():
    print("""
--- SarahMemory Unified Model Loader ---
""")

    choice = cli_menu()

    if choice == '1':
        selected_llms = [k for k, v in MODEL_CONFIG.items() if v]
        selected_objects = [k for k, v in OBJECT_MODEL_CONFIG.items() if isinstance(v, dict) and v.get("enabled")]
    elif choice == '2':
        selected_llms = list_models(MODEL_CONFIG)
        selected_objects = []
    elif choice == '3':
        selected_llms = []
        selected_objects = list_models(OBJECT_MODEL_CONFIG)
    elif choice == '4':
        os.system(f"pip install -r {os.path.join(BASE_DIR, 'requirements.txt')}")
        return
    else:
        print("[EXITED] No models downloaded.")
        return

    model_alias_map = {
        "phi-1_5": "microsoft/phi-1_5",
        "phi-2": "microsoft/phi-2",
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "multi-qa-MiniLM": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "paraphrase-MiniLM-L3-v2": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "distiluse-multilingual": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "allenai-specter": "allenai/specter",
        "e5-base": "intfloat/e5-base",
        "falcon-rw-1b": "tiiuae/falcon-rw-1b",
        "openchat-3.5": "openchat/openchat-3.5-0106",
        "Nous-Capybara-7B": "NousResearch/Nous-Capybara-7B",
        "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
        "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }

    for name in selected_llms:
        full_repo = model_alias_map.get(name, name)
        local_dir = os.path.join(MODELS_DIR, full_repo.replace('/', '_'))
        use_sentence_transformer = name in [
            "all-MiniLM-L6-v2", "multi-qa-MiniLM", "paraphrase-MiniLM-L3-v2",
            "distiluse-multilingual", "allenai-specter", "e5-base"
        ]
        download_model(full_repo, local_dir, use_sentence_transformer)

    for name in selected_objects:
        config = OBJECT_MODEL_CONFIG.get(name, {})
        repo = config.get("repo")
        hf_repo_url = config.get("hf_repo")
        pip_package = config.get("require")
        if not hf_repo_url:
            print(f"[SKIP] {name} has no hf_repo defined. Skipping.")
            continue
        if pip_package:
            pip_install_if_needed(pip_package)
        local_dir = os.path.join(MODELS_DIR, repo.replace('/', '_'))
        download_object_model(name, hf_repo_url, local_dir)

    print("""
✅ [FINISHED] All selected models processed.
""")

if __name__ == "__main__":
    main()

# ====================================================================
# END OF SarahMemoryLLM.py v8.0.0
# ====================================================================