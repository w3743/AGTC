@echo off
setlocal
cd /d "%~dp0"
set "FIELD_PY=python"
if exist ".venv-field\Scripts\python.exe" set "FIELD_PY=.venv-field\Scripts\python.exe"
"%FIELD_PY%" -m field_sim serve %*
pause
