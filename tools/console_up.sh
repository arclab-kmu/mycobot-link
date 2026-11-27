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
#
# Since 2026-11-27 both cameras plug into the AGX itself. TOP_SSH=local / SIDE_SSH=local run the same viewer here on
# 8090 / 8092 (no Jetson, no tunnel); tools/rs_nodes.py picks the nodes by USB product (D435i top, D405 side), or set
# TOP_DEPTH/TOP_COLOR/SIDE_DEPTH/SIDE_COLOR yourself (and *_DEPTH_UNIT, mm per raw count: D405 0.1, D435i 1.0):
#   TOP_SSH=local SIDE_SSH=local MYCOBOT_LAB=~/Documents/lab-jetson/console tools/console_up.sh
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

port_free() { ! ss -ltn "sport = :$1" | grep -q LISTEN; }
camera() {  # ROLE HOST PORT LOCAL_PORT: a viewer on that Jetson + a tunnel, or (HOST=local) a viewer here
  local role=$1 host=$2 port=$3 lp=$4
  if [ "$host" = local ]; then
    port_free "$lp" || { echo "$role camera: $lp already listening, kept"; return 0; }
    eval "$(python3 "$HERE/tools/rs_nodes.py" env)"
    local dv="${role}_DEPTH" cv="${role}_COLOR" uv="${role}_DEPTH_UNIT"
    if [ -z "${!dv:-}" ] || [ -z "${!cv:-}" ]; then
      echo "$role camera: no RealSense nodes found here (tools/rs_nodes.py)"; return 1
    fi
    python3 "$HERE/lab/jetson-viewer/rs_web_viewer.py" --port "$lp" --depth-dev "${!dv}" --color-dev "${!cv}" \
      --depth-unit "${!uv:-1.0}" > "/tmp/rs-viewer-$lp.log" 2>&1 & pids+=($!)
    echo "$role camera: viewer here on $lp (depth ${!dv} x ${!uv:-1.0} mm, color ${!cv}; log /tmp/rs-viewer-$lp.log)"
  else
    "${SSH[@]}" -p "$port" "$host" '~/rs_web_viewer/start.sh' >/dev/null
    port_free "$lp" && { "${SSH[@]}" -p "$port" -N -L "$lp:127.0.0.1:8090" "$host" & pids+=($!); }
  fi
  return 0
}
camera TOP "$TOP_SSH" "$TOP_PORT" 8090
camera SIDE "$SIDE_SSH" "$SIDE_PORT" 8092
port_free 8100 && { "${SSH[@]}" -N -L 8100:127.0.0.1:8100 "$VLM_SSH" & pids+=($!); }
if port_free 8096; then
  "$CUROBO/scripts/run.sh" server > /tmp/curobo-plan-server.log 2>&1 & pids+=($!)
  echo "cuRobo warming up (about 20-60 s; log /tmp/curobo-plan-server.log)"
  # Wait for it: a /check during warmup's CUDA-graph capture killed the server on the AGX (2026-11-26,
  # "operation not permitted when stream is capturing"), and the console reports curobo ok before that.
  for _ in $(seq 1 180); do
    grep -q "plan_server ready" /tmp/curobo-plan-server.log 2>/dev/null && break
    kill -0 "${pids[-1]}" 2>/dev/null || { echo "cuRobo server exited; see /tmp/curobo-plan-server.log"; exit 1; }
    sleep 1
  done
  grep -q "plan_server ready" /tmp/curobo-plan-server.log || echo "cuRobo not ready after 180 s; starting the console anyway"
fi
# command-box scenarios that sit next to the workspace file (measure-and-plan generators, see planner/README.md)
scen=()
# lab-only extras (one scenario name per line, e.g. unfold77 / retreat77 on the AGX): $LAB/scenarios.extra
extra=()
[ -f "$(dirname "$WS")/scenarios.extra" ] && mapfile -t extra < <(grep -v '^\s*#' "$(dirname "$WS")/scenarios.extra" | tr -d ' ')
for f in matcha_live coffee_live matcha_grasp_test coffee_grasp_test matcha_move_test cal77_scenario park77 park70 wiggle77 "${extra[@]}"; do
  [ -n "$f" ] && [ -f "$(dirname "$WS")/$f.json" ] && scen+=(--scenario "$(dirname "$WS")/$f.json")
done
cd "$HERE/brain"
uv run --extra console mycobot-brain -v console --workspace "$WS" "${scen[@]}"
