@echo off
title SarahMemory Initial Setup
color 0A

echo [STEP 1] Creating Virtual Environment (if not exists)...
if not exist venv (
    python -m venv venv
)

echo [STEP 2] Activating Virtual Environment...
call venv\Scripts\activate

echo [STEP 3] Installing Python Requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [STEP 4] Creating Databases...
python SarahMemoryDBCreate.py

echo [STEP 5] Indexing System Files...
python SarahMemorySystemIndexer.py

echo [STEP 6] Learning From Indexed Data...
python SarahMemorySystemLearn.py

echo [STEP 7] Downloading and Verifying Models...
python SarahMemoryLLM.py

echo [STEP 8] Launching Main Program...
python SarahMemoryMain.py

echo.
echo Setup Complete. SarahMemory is now running.
pause