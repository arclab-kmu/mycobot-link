"""Taught motions (leader 2026-11-02 12:24, calib v3.2): sim-scene's verified joint waypoints for opening,
closing and moving between the views, played as given; the brain's checker only logs on them."""

import json

import cabinet_scene as cs
import pytest
from test_pickplace import START, camera, sent, teleporting  # noqa: F401 (fixtures)

from mycobot_brain import calib, collision, drawers, kinematics, pickplace
from mycobot_brain.drawers import _add


def _leg(qs, chain, tcp=True):
    qs = collision.densify(qs, 2.0)
    out = {"q_deg": [[round(a, 4) for a in q] for q in qs]}
    if tcp:
        out["tcp_m"] = [[round(x, 6) for x in kinematics.position(chain.pose(q, "tcp"))] for q in qs]
    return out


def with_taught(d, ids):
    """The brain's own plans for ``ids`` in this scene, written out as taught legs in the BOARD layout."""
    c = calib.from_dict(d)
    task = drawers.DrawerTask(None, None, c, None, pickplace.RunLog(_tmp()), drawers.DrawerSettings())
    fx, chain, look = c.fixture, c.chain, c.fixture.cabinet_look_pose_deg
    for row in d["cabinet"]["drawers"]:
        if row["id"] not in ids:
            continue
        dr = fx.drawers[row["id"]]
        (plan,) = task._handle_plan(dr)
        st = plan.stage
        a, j = plan.frames[st.from_m]
        pre = plan.seeds[st.from_m]
        to_handle = kinematics.line_path(chain, pre, fx.stage_tcp(st, st.from_m), step_m=0.02, approach=a, jaw=j)
        pull = kinematics.line_path(chain, to_handle[-1], fx.stage_tcp(st, st.to_m), step_m=0.005, approach=a, jaw=j)
        a1 = kinematics.column(chain.pose(pull[-1], "tcp"), 2)
        back = kinematics.line_path(chain, pull[-1], _add(fx.stage_tcp(st, st.to_m), a1, -st.standoff_m), step_m=0.02,
                                    approach=a1, jaw=j)
        grip = {"preshape": 55, "close": 0, "release": 55, "from_m": st.from_m, "to_m": st.to_m}
        rev = lambda qs: list(reversed(qs))  # noqa: E731
        row["taught"] = {
            "open": [dict(grip, **{"in": _leg([look, pre], chain), "to_handle": _leg([pre] + to_handle, chain),
                                  "pull": _leg([to_handle[-1]] + pull, chain), "back_off": _leg([pull[-1]] + back, chain)}),
                     {"out": _leg([back[-1], look], chain)}],
            "close": [dict(grip, **{"in": _leg([look, back[-1]], chain), "to_handle": _leg(rev([pull[-1]] + back), chain),
                                   "pull": _leg(rev([to_handle[-1]] + pull), chain),
                                   "back_off": _leg(rev([pre] + to_handle), chain)}),
                      {"out": _leg([pre, look], chain)}],
            "routes": {"cabinet_look>tilted": _leg([look, row["look_pose_deg"]], chain, tcp=False),
                       "tilted>cabinet_look": _leg([row["look_pose_deg"], look], chain, tcp=False),
                       "tilted>topdown": _leg([row["look_pose_deg"], row["topdown_pose_deg"]], chain, tcp=False),
                       "topdown>tilted": _leg([row["topdown_pose_deg"], row["look_pose_deg"]], chain, tcp=False)},
        }
    return d


def as_sim_scene(d):
    """The same motions in sim-scene's v3.2 segment layout (results/taught_motions_v32.json): flat lists of
    {stage, phase, gripper_before, speed, q_deg, tcp_m}, each leg's q_deg starting one step past where it
    begins; routes with from_q_deg."""
    names = {"in": "to_handle", "to_handle": "approach", "pull": "pull", "back_off": "back_off"}
    for row in d["cabinet"]["drawers"]:
        t = row.get("taught")
        if not t:
            continue
        out = {"routes": {}}
        for verb in ("open", "close"):
            segs = []
            for e in t[verb]:
                if "out" in e:
                    segs.append({"stage": None, "phase": "return", "gripper_before": None, "speed": 20,
                                 "q_deg": e["out"]["q_deg"][1:], "tcp_m": e["out"]["tcp_m"][1:]})
                    continue
                grip = {"in": e["preshape"], "pull": e["close"], "back_off": e["release"]}
                for key in ("in", "to_handle", "pull", "back_off"):
                    phase = names[key] if not (verb == "close" and key == "pull") else "push"
                    segs.append({"stage": 0, "phase": phase, "gripper_before": grip.get(key), "speed": 15,
                                 "q_deg": e[key]["q_deg"][1:], "tcp_m": e[key]["tcp_m"][1:]})
            out[verb] = segs
        for name, r in t["routes"].items():
            a, b = name.split(">")
            sim = {"cabinet_look": "cabinet", "tilted": "tilted", "topdown": "topdown"}
            out["routes"][f"{sim[a]}_to_{sim[b]}"] = {"phase": f"{sim[a]}_to_{sim[b]}", "q_deg": r["q_deg"][1:],
                                                      "from_q_deg": r["q_deg"][0]}
        row["taught"] = out
    return d


def _tmp():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp())


def test_taught_legs_load_and_agree_with_this_chain(tmp_path):
    d = with_taught(cs.sim_like_dict(), {"r2c1"})
    c = calib.from_dict(d)
    t = c.fixture.drawers["r2c1"].taught
    assert t is not None and c.fixture.drawers["r1c1"].taught is None
    assert [s.phase for s in t.open] == ["standoff", "approach", "pull", "back_off", "return"]
    assert [s.gripper for s in t.open] == [55, None, 0, 55, None]
    assert [s.opening_after for s in t.open] == [None, None, 0.1, None, None]
    assert [s.opening_after for s in t.close] == [None, None, 0.0, None, None]
    assert set(t.routes) == {"cabinet_look>tilted", "tilted>cabinet_look", "tilted>topdown", "topdown>tilted"}
    sim = calib.from_dict(as_sim_scene(with_taught(cs.sim_like_dict(), {"r2c1"}))).fixture.drawers["r2c1"].taught
    assert [s.phase for s in sim.close] == ["standoff", "approach", "pull", "back_off", "return"]
    assert [s.gripper for s in sim.open] == [55, None, 0, 55, None] and sim.open[2].opening_after == 0.1
    assert sim.routes["cabinet_to_tilted"].start == t.routes["cabinet_look>tilted"].start
    task = drawers.DrawerTask(None, None, c, None, pickplace.RunLog(tmp_path / "run"), drawers.DrawerSettings())
    assert task.taught_fk_mm() == pytest.approx(0.0, abs=0.01)  # its own FK, rounded to 1 um


def test_a_taught_leg_more_than_2_deg_apart_stops_the_load():
    d = with_taught(cs.sim_like_dict(), {"r2c1"})
    row = next(r for r in d["cabinet"]["drawers"] if r["id"] == "r2c1")
    q = row["taught"]["open"][0]["pull"]["q_deg"]  # brain-spark's layout
    del q[1:3]
    with pytest.raises(calib.CalibError, match="more than 2 deg apart"):
        calib.from_dict(d)


@pytest.mark.parametrize("layout", ["brain-spark", "sim-scene"])
def test_a_find_plays_the_taught_open_close_and_view_routes(link, camera, teleporting, tmp_path, monkeypatch, layout):
    from test_drawers import SIM_OBJECTS, events, run
    d = with_taught(cs.sim_like_dict(), {"r2c1"})
    if layout == "sim-scene":
        d = as_sim_scene(d)
    monkeypatch.setattr(cs, "load", lambda: calib.from_dict(d))
    cone = dict(SIM_OBJECTS["yellow_cone"], drawer="r2c1", xy=[0.289, 0.0])
    result, world, _ = run(link, camera, teleporting, tmp_path, objects={"yellow_cone": cone}, order=["r2c1"])
    assert result["ok"] is True, result
    assert world.in_handover("yellow_cone") and world.all_shut()
    ev = events(tmp_path)
    taught = [e for e in ev if e["event"] == "taught"]
    assert taught[0]["drawers"] == ["r2c1"] and taught[0]["fk_vs_calib_tcp_max_mm"] < 0.01
    assert any(e.get("route") in ("r2c1 cabinet_look>tilted", "r2c1 cabinet_to_tilted") for e in taught)
    # The pull sent exactly the taught waypoints (no IK), 2 deg apart at most.
    t = calib.from_dict(d).fixture.drawers["r2c1"].taught
    pulled = [e["angles_deg"] for e in ev if e["event"] == "move" and e["phase"] == "open r2c1"]
    assert pulled == [pytest.approx(list(q), abs=0.01) for q in next(s for s in t.open if s.phase == "pull").leg.q]
    assert not any(e["event"] == "skip" and "not a taught sequence" in e["why"] for e in ev)


def test_one_way_taught_routes_are_chained(tmp_path):
    """v3.2 (sim-scene 12:43) teaches cabinet look -> tilted -> top-down -> cabinet look; from the tilted
    look back to the cabinet look the brain goes on round by the top-down view."""
    d = with_taught(cs.sim_like_dict(), {"r2c1"})
    row = next(r for r in d["cabinet"]["drawers"] if r["id"] == "r2c1")
    chain = calib.from_dict(d).chain
    look = calib.from_dict(d).fixture.cabinet_look_pose_deg
    routes = row["taught"]["routes"]
    routes["topdown>cabinet_look"] = _leg([row["topdown_pose_deg"], look], chain, tcp=False)
    del routes["tilted>cabinet_look"], routes["topdown>tilted"]
    task = drawers.DrawerTask(None, None, calib.from_dict(d), None, pickplace.RunLog(tmp_path / "run"),
                              drawers.DrawerSettings())
    names = [n for n, _ in task._taught_route(row["look_pose_deg"], look)]
    assert names == ["r2c1 tilted>topdown", "r2c1 topdown>cabinet_look"]
    assert [n for n, _ in task._taught_route(look, row["look_pose_deg"])] == ["r2c1 cabinet_look>tilted"]
    assert task._taught_route(look, [0.0] * 6) is None
