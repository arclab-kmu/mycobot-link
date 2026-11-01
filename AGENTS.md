# AGENTS.md

Guidance for coding agents working in this repository.
(`CLAUDE.md` is a symlink to this file. Edit `AGENTS.md`.)

The link between a **DGX Spark** running vLLM (the brain, Python) and the
**myCobot 280 JetsonNano** (the arm server, Rust), without ROS. The brain
proposes skill requests; the arm server owns every safety check and the UART.
Wire format: `schema/PROTOCOL.md` and `schema/messages.schema.json`.

## Commands

```bash
# arm server (any host; the binary for the Jetson is cross-built)
cd arm-server
cargo test                         # unit + schema examples + pymycobot golden frames
cargo clippy --all-targets
cargo run -- --brain 127.0.0.1:47900                    # dry-run
rustup target add aarch64-unknown-linux-musl            # once
cargo build --release --target aarch64-unknown-linux-musl   # static, for the Jetson

# brain (on the Spark; anywhere for tests)
cd brain
uv sync --extra dev
uv run pytest                      # includes test_e2e_rust.py if arm-server/target/debug/mycobot-arm exists
uv run mycobot-brain status
uv run mycobot-brain look                      # one camera frame -> vLLM description
uv run mycobot-brain act "centre the tape roll" # camera -> vLLM -> joint step (--execute to send)
uv run mycobot-brain pick-place "빨간 블록을 잡아서 갈색 상자 안에 넣어" --calib sim_calib.json  # --execute to move
uv run mycobot-brain drawer "노란색 원뿔 찾아서 꺼내 줘" --calib sim_calib_drawers.json  # put / take / find

# camera on the Jetson (python3 3.8 + ffmpeg, no venv)
python3 camera-push/camera_push.py --brain <spark>:47901
python3 camera-push/camera_push.py --brain 127.0.0.1:47901 --test-pattern   # no camera

# re-record the golden frames after a pymycobot bump (from the repo root)
uv run --no-project --with pymycobot==4.0.7 python tools/golden_frames.py
```

## Invariants

1. **The arm server has the last word.** The brain's schema check and the
   planner's prompt only shape proposals. Limits, the 20 deg jump check against
   *measured* angles, speed cap, freshness, gate and watchdog live in
   `arm-server/src/server.rs` + `safety.rs`. Never move a check to the brain
   side, and never add a brain-side path to the UART (e.g. pymycobot's
   `MyCobot280Socket`).
2. **Hardware is never the default.** Without `--live` the server builds
   `DryRunArm`. `--live` asks for a typed `yes` on a TTY or `--assume-yes`.
   Only `SerialArm` touches `/dev/ttyTHS1`, and it holds the vendor's
   `/tmp/mycobot_lock` flock for the whole session, like `mycobot-control`.
3. **Silence means stop.** Watchdog expiry, stale joint readings, link loss and
   shutdown all call `stop()` and close the gate; re-opening takes an explicit
   `enable`. A new failure path must end the same way.
4. **Frames are held to pymycobot's bytes**, not to a reading of its source:
   `arm-server/tests/golden/pymycobot-4.0.7.json` was recorded over a pty.
   Adding a serial command means adding it to `tools/golden_frames.py` first.
5. **Schema changes go to both sides at once.** `schema/examples.json` is read
   by `brain/tests/test_schema.py` (jsonschema) and
   `arm-server/tests/schema_examples.rs` (serde round trip). Bump `protocol`
   in the hello for any incompatible change.
6. **Claims are marked.** Anything not confirmed on this arm carries
   `TODO(unverified):`.

## Layout

- `schema/`: protocol doc, JSON Schema, shared examples.
- `arm-server/src/`: `frame.rs` (serial codec), `protocol.rs` (serde types),
  `safety.rs` (limits ported from `mycobot-control/safety/limits.py` - keep the
  numbers in step), `arm.rs` (`Arm` trait, `DryRunArm`, `SerialArm`),
  `server.rs` (gate/watchdog/requests, time passed in), `lines.rs`, `main.rs`
  (socket loop, reconnect, Ctrl-C).
- `brain/src/mycobot_brain/`: `link.py` (`ArmLink`: listen, request/reply by
  id, heartbeat thread, latest state), `planner.py` (one vLLM chat completion
  constrained to the move schema; `chat()` is the shared POST), `camera.py`
  (`CameraLink`: second listener, newest JPEG, age measured on this side),
  `vision.py` (image requests: `describe`, `propose_step` = one bounded joint
  delta), `act.py` (look → step → settle → look loop; its limits only stop a
  wandering model early), `kinematics.py` (FK/IK in firmware degrees, pixel →
  ray → known plane; no depth image), `calib.py` (scene geometry file),
  `grounding.py` (VLM locate/check requests, 0-1000 or pixel coordinates),
  `pickplace.py` (look → locate → close look → grasp → check → place → check,
  logged to `events.jsonl`), `fixture.py` (the drawer cabinet from the calib:
  handles, pull axis, floors, catalogue, handover; never contents),
  `drawer_vlm.py` (command parse, cabinet view, what is in a drawer),
  `drawers.py` (`DrawerTask`, a `PickPlace` subclass: handle grasp + straight
  pull/push, in-drawer pick, put, handover, find loop), `cli.py`.
- `camera-push/camera_push.py`: Jetson, Python 3.8 stdlib + `ffmpeg`. Grabs
  `/dev/video0` (MJPG), re-encodes every 1/rate-th frame, pushes the newest.
- `tools/golden_frames.py`: pymycobot over a pty.
- `tools/g1_grounding.py`: VLM pixels vs sim truth. `tools/g3_sim_run.sh` (one
  seed: reset → record → `pick-place --execute` → truth) and `tools/g3_score.py`
  (pass/fail from the run's raw files). Truth is read only by these.

## Relation to other repos

- `~/manipulation/mycobot-control`: the Python control package for the same
  arm. Source of the limit numbers and the dry-run/`--live`/gate conventions.
  Do not import it; copy numbers deliberately and say where from.
- `~/manipulation/nero-control-cuda`: the lab's C++ NERO controller; the
  "core without ROS, dry-run first" design is shared.
- Wiki record: `~/llm-wiki/queries/2026-10-23-spark-mycobot-link-and-language.md`.

## Conventions

Comments explain *why*. Tests assert behaviour and inject time (`now:
Instant` in Rust) instead of sleeping where they can. Rust: edition 2024,
clippy clean, no `unsafe`. Python: stdlib only at runtime, 3.10+ for the brain;
`camera_push.py` must stay 3.8-compatible (the Jetson's python3).
