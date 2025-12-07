# AutoRecoveryDaemon.py
# Watches SarahMemory Core Services and Restarts on Crash or Freeze

import time
import subprocess
import os
import psutil

MONITORED_PROCESSES = ["SarahMemoryGUI.py", "SarahMemoryVoice.py"]
RESTART_DELAY = 5  # seconds


def is_process_running(process_name):
    for proc in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            if process_name in proc.info['cmdline']:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False


def restart_process(script):
    print(f"[Recovery] Restarting: {script}")
    subprocess.Popen(["python", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def watchdog():
    print("[AutoRecoveryDaemon] Monitoring SarahMemory core processes...")
    while True:
        for script in MONITORED_PROCESSES:
            if not is_process_running(script):
                print(f"[Warning] Process {script} not running.")
                restart_process(script)
        time.sleep(RESTART_DELAY)


if __name__ == '__main__':
    watchdog()
