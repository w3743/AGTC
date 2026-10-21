@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\field_train_gazebo.ps1" -Command all -Backend gazebo -Mode mixed -Steps 512 -NSteps 128 -MonitorPort 8768
if errorlevel 1 pause
endlocal
