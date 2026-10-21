@echo off
chcp 65001 >nul
setlocal
set "WSL_DISTRO=Ubuntu-24.04"
set "LINUX_PROJECT="
for /f "usebackq tokens=* delims=" %%P in (`wsl.exe -d %WSL_DISTRO% --exec wslpath -u "%~dp0." 2^>nul`) do set "LINUX_PROJECT=%%P"
if not defined LINUX_PROJECT (
    echo 无法定位项目的 WSL 路径。请确认已安装并启动 Ubuntu-24.04 WSL。
    pause
    exit /b 1
)
wsl.exe -d %WSL_DISTRO% --exec bash "%LINUX_PROJECT%/scripts/isaaclab_python.sh" -m isaac_field viewer %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
exit /b %RC%
