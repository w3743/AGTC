@echo off
chcp 65001 >nul
setlocal DisableDelayedExpansion
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& { & '%~dp0scripts\isaaclab_python.ps1' -Mode viewer -Arguments $args; exit $LASTEXITCODE }" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
exit /b %RC%
