# planner

Measure-and-plan for the myCobot pour-and-stir ("말차") demo, run by the web console when a command is typed.

- `matcha_dyn.py`: top camera colour + depth -> cup rim, matcha vial (greenest cap), other vials -> `.77` grasp,
  carry, 140-degree pour with two shakes, return (`plan77.py`) -> `.70` stir at the cup (`plan70.py`) -> the delay
  at which the `.70` may start during the `.77` return (two-arm sphere clearance on the time axis). Prints a
  console scenario (JSON). About 5 s on the lab PC. `--only grasp` and `--move-to X,Y` make test scenarios.
- `plan77.py`: two checked templates (`data/templates77.json`) moved to new spots by turning J1 by the change of
  bearing and walking the rest with DLS IK; candidates turn / tip / raise the grasp and turn the pour about the
  cup and the hand about the vial axis. Waypoints are checked for limits, branch jumps, gripper/vial clearance
  and with the cuRobo server (`/check`, densified); transits come from cuRobo `/plan`.
- `plan70.py` + `s70/`: the 2026-11-24 smooth stir planner as a function; aims at the measured cup plus a
  learned offset (`cal70.json`), with the top-camera centring step at the dip.
- `detect.py`, `locate.py`: cup rim circle fit (radius pinned to the rolled rim's outer edge), cap detection
  with a depth height filter; pixel -> `.77` via the calib_top3 camera and the `cal77` 2-D map.
- `align.py`: pre-grasp alignment in one image (gripper open/closed difference vs the vial cap).
- `verify.py`: after the lift, fail if the matcha cap still stands at table height.
- `cal77.py`: hover-and-toggle calibration of the top camera against the `.77` (experimental; see the wiki).
- `spheres.py`: local copy of cuRobo's collision spheres on the DH frames (within ~2 mm of `/spheres`).

Runs with `/home/hkim/IsaacLab/.venv/bin/python` (numpy, PIL); `brain/src` must be importable.
