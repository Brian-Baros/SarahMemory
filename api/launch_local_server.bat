@echo off
setlocal
cd /d %~dp0server
python - <<PY
from app import app
app.run(host="127.0.0.1", port=5055)
PY
