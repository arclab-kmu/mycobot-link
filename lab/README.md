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

- `teach/`: the hand-guided demonstration from 2026-11-22 on the `.77` arm (torque released, joints logged at
  ~15 Hz until `/tmp/teach_stop`): `teach_bg.py`, `teach_demo1.csv` (1152 rows, firmware degrees), its log.
  Copied from the Jetson's `~/mycobot-link/` where it was untracked.

The Jetsons' `~/mycobot-link` checkouts sit at `792ccde`; their `mycobot-arm` binaries were rebuilt on 2026-11-24
from the `25a8d50` source (`--query-guard-ms`). Rebuild there after pulling if the arm server changes again.

The cuRobo plan server, robot configs and meshes now live in `curobo/` of this repo (moved 2026-11-26);
`tools/console_up.sh` starts it from there unless `$CUROBO` says otherwise.
