#!/usr/bin/env bash
# One D3 run against the sim drawer scene: reset -> truth before -> record on ->
# drawer --execute -> truth after -> record off.
#
#   tools/d3_sim_run.sh d3_find_s500 500 find "노란색 원뿔 찾아서 꺼내 줘"
#   tools/d3_sim_run.sh d3_take_s450_r2c0 450 find "왼쪽 위 서랍에 있는 물건을 꺼내 줘" r2c0
#   tools/d3_sim_run.sh d3_put_s460_r2c2 460 put "파란 큐브를 오른쪽 위 서랍에 넣어" r2c2
#
# LAYOUT is the bridge's reset layout (find, or put for one more object in the
# handover zone); DRAWER is the drawer the command names, for scoring. Both
# truth replies are saved beside the run for tools/d3_score.py only; the brain
# gets the instruction, the calib and the wrist camera, nothing else. The link
# chain (fw_emu + arm server --live + cam_push_sim) must already be dialling
# this host, as for tools/g3_sim_run.sh.
set -u
RUN=$1
SEED=$2
LAYOUT=$3
INSTRUCTION=$4
DRAWER=${5:-}
SIM_SSH=${SIM_SSH:-"ssh -p 21402 -o ConnectTimeout=5 hkim@210.123.42.156"}
CALIB=${CALIB:-/home/hkim/mycobot-sim/sim_calib_drawers.json}
DIR=${RUNS:-/home/hkim/mycobot-sim/runs}/$RUN
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}

sim() { $SIM_SSH "printf '%s\n' '$1' | nc -q1 127.0.0.1 47950"; }

mkdir -p "$DIR"
echo "== $RUN"
# EXPECT=refusal: the command is infeasible and the run shows the brain refusing it cleanly (d3_score.py).
printf '{"run": "%s", "seed": %s, "layout": "%s", "instruction": "%s", "drawer": "%s", "brain_args": "%s", "expect": "%s"}\n' \
  "$RUN" "$SEED" "$LAYOUT" "$INSTRUCTION" "$DRAWER" "${BRAIN_ARGS:-}" "${EXPECT:-}" >"$DIR/meta.json"
if [ "$LAYOUT" = find ]; then
  sim "{\"op\":\"reset\",\"seed\":$SEED}"
else
  sim "{\"op\":\"reset\",\"seed\":$SEED,\"layout\":\"$LAYOUT\"}"
fi
sleep 3  # the reset puts the arm at the cabinet look pose; let the objects settle
sim '{"op":"truth"}' >"$DIR/truth_before.json"
sim "{\"op\":\"record\",\"on\":true,\"run\":\"$RUN\"}"
# BRAIN_ARGS: extra drawer options, e.g. longer wall-clock waits when the bridge runs slower than real
# time (2026-11-02: ~0.7x with the new grip physics).
# shellcheck disable=SC2086
(cd "$(dirname "$0")/../brain" && "$UV" run mycobot-brain drawer "$INSTRUCTION" \
  --calib "$CALIB" --execute --record "$DIR" ${BRAIN_ARGS:-}) >"$DIR/cli.log" 2>&1
echo "exit $?" | tee -a "$DIR/cli.log"
# Objects at rest before they are scored: every one under 5 mm/s, 3 s at most (leader 18:5x: a cylinder
# rocking at 12 mm/s read 8 deg tilted at the snapshot).
for _ in 1 2 3 4 5 6; do
  sleep 0.5
  sim '{"op":"truth"}' >"$DIR/truth.json"
  python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); sys.exit(max(o["speed_mps"] for o in t["objects"].values()) >= 0.005)' \
    "$DIR/truth.json" && break
done
sim '{"op":"state"}' >"$DIR/state_after.json"
sim '{"op":"record","on":false}'
grep '"event": "result"' "$DIR/events.jsonl" | tail -1
