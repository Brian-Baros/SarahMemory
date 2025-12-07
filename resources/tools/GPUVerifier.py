# GPUVerifier.py
# Checks for NVIDIA CUDA-compatible GPU and available drivers

import subprocess
import sys
import platform

def check_nvidia_gpu():
    try:
        if platform.system() == 'Windows':
            result = subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.STDOUT)
        else:
            result = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True, result.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return False, e.output.decode('utf-8')
    except FileNotFoundError:
        return False, "nvidia-smi not found. NVIDIA GPU may not be present."

def report():
    has_gpu, output = check_nvidia_gpu()
    print("\n[GPU VERIFIER] System GPU Check:")
    if has_gpu:
        print("\033[92m✔ NVIDIA GPU Detected\033[0m")
    else:
        print("\033[91m✘ No Compatible NVIDIA GPU Detected\033[0m")
    print(output)


if __name__ == '__main__':
    report()