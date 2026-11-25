#!/bin/sh
# Stop the viewer and wait until /dev/video0 and /dev/video2 are released.
if ! pkill -f "python3 rs_web_viewer.py"; then echo "not running"; exit 0; fi
i=0
while pgrep -f "python3 rs_web_viewer.py" >/dev/null && [ $i -lt 20 ]; do sleep 0.5; i=$((i+1)); done
if pgrep -f "python3 rs_web_viewer.py" >/dev/null; then
  pkill -9 -f "python3 rs_web_viewer.py"; pkill -9 -f "v4l2-ctl -d /dev/video"; echo "killed"
else
  echo "stopped"
fi
