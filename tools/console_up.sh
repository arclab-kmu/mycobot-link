#!/usr/bin/env bash
# Bring up everything the web console needs, from the workstation, in one go:
#   camera viewers on both Jetsons + tunnels (8090 top, 8092 side),
#   the Spark VLM tunnel (8100), the cuRobo plan server (8096 API, 8080 view),
#   and the console itself (http://127.0.0.1:8095).
# Arm servers are NOT started here: moving hardware stays a separate, explicit step
# (see README "웹 콘솔"). Ctrl-C stops what this script started.
#
#   tools/console_up.sh [workspace.json]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WS="${1:-$HOME/Documents/lab-jetson/console/workspace.json}"
CUROBO="${CUROBO:-$HOME/manipulation/mycobot_curobo}"
export SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-/run/user/$(id -u)/gcr/ssh}"
SSH=(ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o LogLevel=ERROR)
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

"${SSH[@]}" -p 22022 er@192.168.0.70 '~/rs_web_viewer/start.sh' >/dev/null
"${SSH[@]}" er@192.168.0.77 '~/rs_web_viewer/start.sh' >/dev/null
port_free() { ! ss -ltn "sport = :$1" | grep -q LISTEN; }
port_free 8090 && { "${SSH[@]}" -p 22022 -N -L 8090:127.0.0.1:8090 er@192.168.0.70 & pids+=($!); }
port_free 8092 && { "${SSH[@]}" -N -L 8092:127.0.0.1:8090 er@192.168.0.77 & pids+=($!); }
port_free 8100 && { "${SSH[@]}" -N -L 8100:127.0.0.1:8100 spark & pids+=($!); }
if port_free 8096; then
  "$CUROBO/scripts/run.sh" server > /tmp/curobo-plan-server.log 2>&1 & pids+=($!)
  echo "cuRobo warming up (about 60 s; log /tmp/curobo-plan-server.log)"
fi
cd "$HERE/brain"
uv run --extra console mycobot-brain -v console --workspace "$WS"
