"""Read fw_emu's replies with pymycobot 4.0.7 itself (not our decoder): the vendor SDK is the judge.

    uv run --no-project --with pymycobot==4.0.7 python sim/tests/pymycobot_check.py

Not part of ``unittest discover`` (pymycobot is not a dependency of this repo). Needs no arm: the
stub sim and fw_emu run in this process, and ``MyCobot280`` opens the emulator's pty.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fw_emu  # noqa: E402
from stub_sim import StubSim  # noqa: E402


def main() -> int:
    sim = StubSim(port=0, start_deg=[12.34, -45.68, 90.0, -0.01, 150.0, -179.99]).start()
    emu = fw_emu.Service(("127.0.0.1", sim.port), link=None).start()
    time.sleep(0.3)
    from pymycobot import MyCobot280

    arm = MyCobot280(emu.path, 1000000)
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: pymycobot read {got!r}" + ("" if good else f", wanted {want!r}"))

    check("get_angles", arm.get_angles(), [12.34, -45.68, 90.0, -0.01, 150.0, -179.99])
    check("get_gripper_value", arm.get_gripper_value(), 100)
    check("is_power_on", arm.is_power_on(), 1)
    check("get_error_information", arm.get_error_information(), 0)
    arm.set_gripper_value(30, 50)  # no reply is defined for it: pymycobot waits 0.5 s and resends, 3 writes
    time.sleep(0.2)
    check("set_gripper_value reached the sim", sim.grips[-1] if sim.grips else None, {"value": 30.0, "speed": 50})
    check("get_gripper_value follows", (time.sleep(0.5), arm.get_gripper_value())[1], 30)
    arm.close()
    emu.close()
    sim.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
