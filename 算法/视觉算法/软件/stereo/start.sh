#!/bin/sh
set -eu
cd /opt/rk-stereo
if [ -f /etc/systemd/system/rk-stereo.service ]; then
    case "${1:-start}" in
        start|stop|restart) systemctl "${1:-start}" rk-stereo.service ;;
        *) echo 'Usage: start.sh [start|stop|restart]'; exit 1 ;;
    esac
    exit 0
fi
if [ -f run.pid ]; then
    pid=$(cat run.pid)
    if kill -0 "$pid" 2>/dev/null && tr '\000' ' ' < /proc/"$pid"/cmdline | grep -q 'python3 -B server.py'; then
        if [ "${1:-}" = stop ] || [ "${1:-}" = restart ]; then
            kill "$pid"
            attempts=0
            while kill -0 "$pid" 2>/dev/null && tr '\000' ' ' < /proc/"$pid"/cmdline | grep -q 'python3 -B server.py'; do
                attempts=$((attempts + 1))
                if [ "$attempts" -ge 50 ]; then
                    echo 'Previous camera process did not stop; refusing a second owner.'
                    exit 1
                fi
                sleep 0.1
            done
            echo "Stopped: $pid"
            if [ "${1:-}" = stop ]; then exit 0; fi
        else
            echo "Already running: $pid"
            exit 0
        fi
    fi
fi
if [ "${1:-}" = stop ]; then
    echo 'Not running'
    exit 0
fi
setsid env PYTHONPATH=/opt/rk-stereo/vendor python3 -B server.py </dev/null >run.log 2>&1 &
echo $! >run.pid
sleep 1
kill -0 "$(cat run.pid)"
cat run.pid
