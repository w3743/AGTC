@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\field_gazebo.ps1" -Gui -Seconds 3600 -NoMotion
if errorlevel 1 pause
