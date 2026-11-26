#!/usr/bin/env bash
# Bring up everything the web console needs, from the workstation, in one go:
#   camera viewers on both Jetsons + tunnels (8090 top, 8092 side),
#   the VLM tunnel (8100, Gemma 4 on the AGX Orin; VLM_SSH=spark for the old host),
#   the cuRobo plan server (8096 API, 8080 view),
#   and the console itself (http://127.0.0.1:8095).
# Arm servers are NOT started here: moving hardware stays a separate, explicit step
# (see README "웹 콘솔"). Ctrl-C stops what this script started.
#
#   tools/console_up.sh [workspace.json]
#
# On the AGX Orin (exhibit setup, campus Wi-Fi since 2026-11-26) the Jetsons have other addresses and the VLM is
# local (8100 already listening, so no tunnel):
#   TOP_SSH=er@10.103.95.176 SIDE_SSH=er@10.102.90.89 MYCOBOT_LAB=~/Documents/lab-jetson/console tools/console_up.sh
# then open http://127.0.0.1:8095 in a browser on the AGX's monitor.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WS="${1:-${MYCOBOT_LAB:-$HOME/Documents/lab-jetson/console}/workspace.json}"
TOP_SSH="${TOP_SSH:-er@192.168.0.70}"; TOP_PORT="${TOP_PORT:-22022}"   # .70 Jetson: top D435i
SIDE_SSH="${SIDE_SSH:-er@192.168.0.77}"; SIDE_PORT="${SIDE_PORT:-22}"  # .77 Jetson: side D405
CUROBO="${CUROBO:-$HERE/curobo}"
VLM_SSH="${VLM_SSH:-agx@192.168.0.72}"   # vLLM gemma-4-26b-a4b-it on 127.0.0.1:8100 there (lab/agx-vlm/)
export SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-/run/user/$(id -u)/gcr/ssh}"
SSH=(ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o LogLevel=ERROR)
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

"${SSH[@]}" -p "$TOP_PORT" "$TOP_SSH" '~/rs_web_viewer/start.sh' >/dev/null
"${SSH[@]}" -p "$SIDE_PORT" "$SIDE_SSH" '~/rs_web_viewer/start.sh' >/dev/null
port_free() { ! ss -ltn "sport = :$1" | grep -q LISTEN; }
port_free 8090 && { "${SSH[@]}" -p "$TOP_PORT" -N -L 8090:127.0.0.1:8090 "$TOP_SSH" & pids+=($!); }
port_free 8092 && { "${SSH[@]}" -p "$SIDE_PORT" -N -L 8092:127.0.0.1:8090 "$SIDE_SSH" & pids+=($!); }
port_free 8100 && { "${SSH[@]}" -N -L 8100:127.0.0.1:8100 "$VLM_SSH" & pids+=($!); }
if port_free 8096; then
  "$CUROBO/scripts/run.sh" server > /tmp/curobo-plan-server.log 2>&1 & pids+=($!)
  echo "cuRobo warming up (about 60 s; log /tmp/curobo-plan-server.log)"
fi
# command-box scenarios that sit next to the workspace file (measure-and-plan generators, see planner/README.md)
scen=()
for f in matcha_live coffee_live matcha_grasp_test coffee_grasp_test matcha_move_test cal77_scenario park77 park70; do
  [ -f "$(dirname "$WS")/$f.json" ] && scen+=(--scenario "$(dirname "$WS")/$f.json")
done
cd "$HERE/brain"
uv run --extra console mycobot-brain -v console --workspace "$WS" "${scen[@]}"
