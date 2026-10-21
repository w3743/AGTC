@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\field_gazebo_web.ps1"
if errorlevel 1 pause
