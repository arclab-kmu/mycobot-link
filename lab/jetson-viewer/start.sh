#!/bin/sh
# Start the RealSense web viewer in the background (127.0.0.1:8090). Extra args go to rs_web_viewer.py.
cd "$(dirname "$0")" || exit 1
if pgrep -f "python3 rs_web_viewer.py" >/dev/null; then
  echo "already running (pid $(pgrep -f 'python3 rs_web_viewer.py' | tr '\n' ' '))"
else
  nohup python3 rs_web_viewer.py "$@" > viewer.log 2>&1 < /dev/null &
  sleep 3
fi
curl -s localhost:8090/stats; echo
