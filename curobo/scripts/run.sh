#!/usr/bin/env bash
# Launch myCobot 280 cuRobo tools in the cuRobo venv.
#
#   scripts/run.sh viewer              # browser viewer at http://localhost:8080
#   scripts/run.sh server [args...]    # planning service: JSON API :8096 + viser :8080 (127.0.0.1), see plan_server.py
#   scripts/run.sh teleop [args...]    # keyboard EE jog in this terminal, drawn in viser (:8080)
#   scripts/run.sh demo [args...]      # plan + render to plans/ (see plan_demo.py -h)
#   scripts/run.sh check               # planner self-check
#   scripts/run.sh spheres             # inspect the fitted collision spheres (browser)
#   scripts/run.sh rebuild             # URDF -> cuRobo config again (both variants)
#
# Prefix VARIANT=camera for the camera flange + gripper model, e.g.
#   VARIANT=camera scripts/run.sh viewer
set -euo pipefail
CUROBO_DIR="${CUROBO_DIR:-$HOME/manipulation/curobo}"
HERE="$(cd "$(dirname "$0")/.." && pwd -P)"
PORT="${PORT:-8080}"
# VARIANT=camera selects the camera flange + gripper model; ROBOT= overrides with any yml.
case "${VARIANT:-gripper}" in
  camera)  ROBOT="${ROBOT:-$HERE/configs/mycobot_280_jn_camera_gripper.yml}" ;;
  *)       ROBOT="${ROBOT:-$HERE/configs/mycobot_280_jn.yml}" ;;
esac
# The configs and URDFs carry absolute paths (cuRobo joins relative ones onto its own asset
# folder), written for CANON. A checkout anywhere else runs on a copy rendered into .gen/.
CANON=/home/hkim/manipulation/mycobot-link/curobo
if [ "$HERE" != "$CANON" ] && [ "${1:-viewer}" != rebuild ]; then
  mkdir -p "$HERE/.gen/configs" "$HERE/.gen/urdf"
  for f in "$HERE"/configs/*.yml; do
    sed -e "s#$CANON/urdf#$HERE/.gen/urdf#g" -e "s#$CANON#$HERE#g" "$f" > "$HERE/.gen/configs/$(basename "$f")"
  done
  for f in "$HERE"/urdf/*.urdf; do sed "s#$CANON#$HERE#g" "$f" > "$HERE/.gen/urdf/$(basename "$f")"; done
  case "$ROBOT" in "$HERE"/configs/*) ROBOT="$HERE/.gen/configs/$(basename "$ROBOT")" ;; esac
fi
# shellcheck disable=SC1091
source "$CUROBO_DIR/.venv/bin/activate"
cd "$HERE"
case "${1:-viewer}" in
  viewer)  exec python -u scripts/viewer.py --port "$PORT" --robot "$ROBOT" ;;
  server)  shift; exec python -u scripts/plan_server.py --robot "$ROBOT" "$@" ;;
  teleop)  shift; exec python -u scripts/teleop_sim.py --port "$PORT" --robot "$ROBOT" "$@" ;;
  demo)    shift; exec python -u scripts/plan_demo.py --robot "$ROBOT" "$@" ;;
  check)   exec python -u mycobot_planner.py "$ROBOT" ;;
  spheres) exec python -u -m curobo.examples.getting_started.build_robot_model \
             --edit-config "$ROBOT" --visualize --viz-port "$PORT" ;;
  rebuild) (cd scripts && python build_urdf.py && python build_urdf.py --camera-flange) \
             && python scripts/build_config.py \
             && python scripts/build_config.py --urdf urdf/mycobot_280_jn_camera_gripper.urdf \
                  --output configs/mycobot_280_jn_camera_gripper.yml ;;
  *) echo "usage: $0 {viewer|server|teleop|demo|check|spheres|rebuild}" >&2; exit 2 ;;
esac
