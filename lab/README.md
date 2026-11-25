# lab

Lab-specific files the console and the planner read, kept here so the set-up can be rebuilt on another PC.

- `console/`: snapshot (2026-11-25) of `~/Documents/lab-jetson/console/` (`MYCOBOT_LAB`), the folder the console
  and `planner/` actually read. `workspace.json` = cameras, arm bases, tool, safety box, fixed obstacles
  (re-measure after moving an arm, the table or a camera). `workspace.before-recal-20261124.json` holds the
  calib_top3 top-camera model that `planner/locate.py` uses. The `*.json` scenarios are command-box triggers
  ("컵에 말차 가루를 타 줘", "말차 병 잡기 시험", "말차 병 옮기기", "카메라 보정") whose generators run
  `planner/` with `/home/hkim/IsaacLab/.venv/bin/python`; edit those paths on another PC.
  If a file is missing from `MYCOBOT_LAB`, the planner falls back to the copy here.
- `jetson-viewer/`: the RealSense web viewer that runs on both Jetsons as `~/rs_web_viewer/`
  (`start.sh` / `stop.sh`, 127.0.0.1:8090, reached through SSH tunnels; see `tools/console_up.sh`).

Not in any repository yet: the cuRobo plan server (`~/manipulation/mycobot_curobo`, `scripts/plan_server.py`,
robot configs and meshes), which `tools/console_up.sh` starts from `$CUROBO`.
