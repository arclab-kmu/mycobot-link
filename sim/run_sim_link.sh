#!/usr/bin/env bash
# The sim side of the mycobot link, on the machine that runs the simulation:
#
#   sim-bridge (127.0.0.1:47950) <- fw_emu (pty) <- mycobot-arm --live --serial <pty> --brain <spark>:47900
#                                \- cam_push_sim -> <spark>:47901
#
# The brain on the Spark is started separately (it binds the two ports; this side dials in and
# keeps redialling every second, so the order does not matter). Ctrl-C stops everything: the
# arm-server first, so its stop() reaches the sim before the emulator goes away.
#
#   sim/run_sim_link.sh                 # against the real sim-bridge
#   sim/run_sim_link.sh --stub          # against sim/stub_sim.py (no Isaac Lab)
#   sim/run_sim_link.sh --no-camera
#
# Environment: BRAIN_HOST (210.123.42.157), ARM_PORT (47900), CAM_PORT (47901),
# BRIDGE (127.0.0.1:47950), LINK (/tmp/mycobot-sim-tty), LOGS (/tmp/mycobot-sim-link-logs),
# MYCOBOT_ARM (path of the arm-server binary; default arm-server/target/release, built if missing).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
BRAIN_HOST="${BRAIN_HOST:-210.123.42.157}"
ARM_PORT="${ARM_PORT:-47900}"
CAM_PORT="${CAM_PORT:-47901}"
BRIDGE="${BRIDGE:-127.0.0.1:47950}"
LINK="${LINK:-/tmp/mycobot-sim-tty}"
LOGS="${LOGS:-/tmp/mycobot-sim-link-logs}"
STUB=0
CAMERA=1
for arg in "$@"; do
  case "$arg" in
    --stub) STUB=1 ;;
    --no-camera) CAMERA=0 ;;
    -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOGS"
pids=()
names=()
start() { # name, command...
  local name="$1"; shift
  "$@" >"$LOGS/$name.out" 2>"$LOGS/$name.log" &
  pids+=("$!"); names+=("$name")
  echo "run_sim_link: $name started (pid $!, log $LOGS/$name.log)"
}

cleanup() {
  trap - EXIT
  # Reverse order. The arm-server leaves cleanly on SIGINT (ctrlc), which a background job of a
  # non-interactive shell ignores unless the child installs its own handler; the Python tools
  # leave on SIGTERM.
  for ((i = ${#pids[@]} - 1; i >= 0; i--)); do
    if [[ "${names[$i]}" == arm_server ]]; then kill -INT "${pids[$i]}" 2>/dev/null || true
    else kill -TERM "${pids[$i]}" 2>/dev/null || true; fi
    for _ in $(seq 30); do kill -0 "${pids[$i]}" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "${pids[$i]}" 2>/dev/null || true
  done
  echo "run_sim_link: stopped"
}
trap cleanup EXIT
trap 'exit 0' INT TERM  # leaving runs the EXIT trap: an intended stop is not a failure

ARM_BIN="${MYCOBOT_ARM:-$REPO/arm-server/target/release/mycobot-arm}"
if [[ ! -x "$ARM_BIN" ]]; then
  echo "run_sim_link: building $ARM_BIN"
  (cd "$REPO/arm-server" && cargo build --release)
fi

if [[ $STUB == 1 ]]; then
  start stub_sim python3 "$HERE/stub_sim.py" --host "${BRIDGE%:*}" --port "${BRIDGE##*:}"
  sleep 0.5
fi
if ! python3 - "$BRIDGE" <<'EOF'
import json, socket, sys
host, _, port = sys.argv[1].rpartition(":")
try:
    with socket.create_connection((host, int(port)), timeout=2) as s:
        s.sendall(b'{"op":"state"}\n')
        reply = json.loads(s.makefile().readline())
    sys.exit(0 if reply.get("ok") else 1)
except (OSError, ValueError):
    sys.exit(1)
EOF
then
  echo "run_sim_link: sim-bridge $BRIDGE does not answer yet; fw_emu stays silent (the arm-server will refuse to enable) until it does" >&2
fi

start fw_emu python3 "$HERE/fw_emu.py" --bridge "$BRIDGE" --link "$LINK"
for _ in $(seq 50); do [[ -L "$LINK" ]] && break; sleep 0.1; done
PTY="$(readlink -f "$LINK" 2>/dev/null || true)"
# --live below is only ever given a pty that fw_emu made: there is no hardware on this machine.
if [[ "$PTY" != /dev/pts/* ]]; then
  echo "run_sim_link: $LINK is not a pty link (got '$PTY'); refusing to start the arm-server with --live" >&2
  exit 1
fi
echo "run_sim_link: fw_emu pty $PTY linked at $LINK"

start arm_server "$ARM_BIN" --brain "$BRAIN_HOST:$ARM_PORT" --live --assume-yes --serial "$LINK"
if [[ $CAMERA == 1 ]]; then
  start cam_push_sim python3 "$HERE/cam_push_sim.py" --brain "$BRAIN_HOST:$CAM_PORT" --bridge "$BRIDGE"
fi

echo "run_sim_link: up. brain at $BRAIN_HOST:$ARM_PORT (arm) / :$CAM_PORT (camera). Ctrl-C to stop."
# Any child dying ends the whole link: a half-running chain is worse than none.
wait -n "${pids[@]}" || true
for i in "${!pids[@]}"; do
  if ! kill -0 "${pids[$i]}" 2>/dev/null; then
    echo "run_sim_link: ${names[$i]} exited; last lines of $LOGS/${names[$i]}.log:" >&2
    tail -n 5 "$LOGS/${names[$i]}.log" >&2 || true
  fi
done
exit 1
