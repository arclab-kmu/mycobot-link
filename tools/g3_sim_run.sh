#!/usr/bin/env bash
# One G3 run against the Isaac Lab sim: reset seed -> record on ->
# pick-place --execute -> truth -> record off.
#
#   tools/g3_sim_run.sh 0            # run g3_s00
#   tools/g3_sim_run.sh 3 _retry     # run g3_s03_retry
#
# The sim-bridge binds loopback on the sim PC, so its ops go over ssh. The
# truth reply is saved next to the run for scoring only; the brain never
# reads it (it gets the calib and the wrist camera, nothing else). The link
# chain (fw_emu + arm server --live on its pty + cam_push_sim) must already
# be dialling this host: see schema/SIM.md on the sim PC's checkout.
set -u
SEED=$1
RUN=$(printf 'g3_s%02d' "$SEED")${2:-}
SIM_SSH=${SIM_SSH:-"ssh -p 21402 -o ConnectTimeout=5 hkim@210.123.42.156"}
CALIB=${CALIB:-/home/hkim/mycobot-sim/sim_calib.json}
DIR=${RUNS:-/home/hkim/mycobot-sim/runs}/$RUN
INSTRUCTION=${INSTRUCTION:-"빨간 블록을 잡아서 갈색 상자 안에 넣어"}
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}

sim() { $SIM_SSH "printf '%s\n' '$1' | nc -q1 127.0.0.1 47950"; }

mkdir -p "$DIR"
echo "== $RUN"
sim "{\"op\":\"reset\",\"seed\":$SEED}"
sleep 3  # the reset puts the arm at the look pose; let the block settle
sim "{\"op\":\"record\",\"on\":true,\"run\":\"$RUN\"}"
(cd "$(dirname "$0")/../brain" && "$UV" run mycobot-brain pick-place "$INSTRUCTION" \
  --calib "$CALIB" --execute --record "$DIR") >"$DIR/cli.log" 2>&1
echo "exit $?" | tee -a "$DIR/cli.log"
sleep 1.5  # block at rest before it is scored
sim '{"op":"truth"}' | tee "$DIR/truth.json"
sim '{"op":"state"}' | tee "$DIR/state_after.json"
sim '{"op":"record","on":false}'
grep '"event": "result"' "$DIR/events.jsonl" | tail -1
