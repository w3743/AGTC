@echo off
setlocal
set "BOARD=b1371e33e9f6e49e"
adb -s %BOARD% get-state >nul 2>&1
if errorlevel 1 goto failed
if /i "%~1"=="stop" (
    adb -s %BOARD% shell sh /opt/rk-stereo/start.sh stop
    adb -s %BOARD% forward --remove tcp:8877
    exit /b
)
adb -s %BOARD% push "%~dp0stereo/server.py" /opt/rk-stereo/server.py >nul
if errorlevel 1 goto failed
adb -s %BOARD% push "%~dp0stereo/depth.py" /opt/rk-stereo/depth.py >nul
if errorlevel 1 goto failed
adb -s %BOARD% push "%~dp0stereo/calibration/current/parameters.npz" /opt/rk-stereo/calibration.npz >nul
if errorlevel 1 goto failed
adb -s %BOARD% push "%~dp0stereo/index.html" /opt/rk-stereo/index.html >nul
if errorlevel 1 goto failed
adb -s %BOARD% push "%~dp0stereo/start.sh" /opt/rk-stereo/start.sh >nul
if errorlevel 1 goto failed
adb -s %BOARD% shell sh /opt/rk-stereo/start.sh restart
if errorlevel 1 goto failed
adb -s %BOARD% forward tcp:8877 tcp:8088
if errorlevel 1 goto failed
start "" "http://127.0.0.1:8877"
exit /b 0
:failed
echo Cannot start stereo trial. Check USB ADB connection and installed dependencies.
pause
exit /b 1
