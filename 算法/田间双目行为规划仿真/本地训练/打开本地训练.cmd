@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0open_training.ps1"
if errorlevel 1 pause
