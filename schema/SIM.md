# 시뮬레이션 연결 (`sim/`)

진짜 myCobot 대신 Isaac Lab 시뮬레이션을 붙이는 방법이다. **프로토콜 1과 Rust 팔 서버는 그대로**다. 바뀌는 것은 팔 서버 뒤쪽, 즉 UART 맞은편에 Atom 펌웨어 대신 `sim/fw_emu.py`가 있다는 점뿐이다. 그래서 관절 한계, **측정값**에서 20°, 속도 상한, 신선도, watchdog 같은 진짜 안전 검사가 시뮬레이션 실험에서도 매번 돈다.

```text
Spark                                        시뮬레이션 PC (rtx5090)
 brain :47900 (팔) ◀── TCP connect ───────── mycobot-arm --live --serial /tmp/mycobot-sim-tty   (바이너리 그대로)
       :47901 (카메라) ◀─ TCP connect ──┐              │ FE FE … FA, pty
                                         │         fw_emu.py ── TCP 127.0.0.1:47950 ──▶ sim-bridge (Isaac Lab)
                                         └─ cam_push_sim.py ─── TCP 127.0.0.1:47950 ──▶ (frame)
```

`--live`는 **fw_emu가 만든 pty에만** 쓴다. 이 PC에는 하드웨어가 없고, `sim/run_sim_link.sh`는 링크가 `/dev/pts/*`를 가리키지 않으면 팔 서버를 시작하지 않는다. (팔 서버의 옵션 이름은 `--serial`이다. `--port`가 아니다.)

## 실행

```bash
sim/run_sim_link.sh            # 진짜 sim-bridge(127.0.0.1:47950)에 연결: fw_emu + 팔 서버 + cam_push_sim
sim/run_sim_link.sh --stub     # sim/stub_sim.py(가짜 시뮬레이션)에 연결
# Spark에서 두뇌를 띄우면(둘 다 먼저 떠 있어도 된다: 이쪽이 1초마다 다시 접속한다) 연결된다
```

환경 변수: `BRAIN_HOST`(기본 `210.123.42.157`), `ARM_PORT`, `CAM_PORT`, `BRIDGE`, `LINK`, `LOGS`(로그, 기본 `/tmp/mycobot-sim-link-logs`), `MYCOBOT_ARM`(팔 서버 바이너리, 기본 `arm-server/target/release/mycobot-arm`이고 없으면 빌드한다). Ctrl-C나 SIGTERM이면 팔 서버를 먼저 끝내서(`stop()`이 시뮬레이션에 닿는다) 나머지를 정리한다.

네트워크 너머 G2 점검은 두뇌 쪽에서 `sim/g2_check.py`를 돌린다. 체크아웃은 건드리지 않는다.

```bash
ssh spark 'cd ~/manipulation/mycobot-link/brain && ~/.local/bin/uv run python -' < sim/g2_check.py
```

테스트(표준 라이브러리 `unittest`, 팔 서버를 `cargo build`로 만들어 둬야 e2e가 돈다):

```bash
python3 -m unittest discover -s sim/tests            # 골든 바이트, pty, 카메라, 한 PC e2e
uv run --no-project --with pymycobot==4.0.7 python sim/tests/pymycobot_check.py   # pymycobot이 응답을 읽는지
```

## fw_emu가 답하는 프레임

프레임은 `arm-server/src/frame.rs`와 같다(`FE FE LEN CMD DATA… FA`, 골든은 pymycobot 4.0.7이 쓴 바이트).

| CMD | 요청 | fw_emu가 하는 일 |
| --- | --- | --- |
| `0x20` | `get_angles` | 캐시한 sim `state`의 `angles_deg`를 i16 **센티도**(반올림) 6개로 답한다. sim을 기다리지 않는다(`--state-hz` 50으로 따로 읽어 둔다). 캐시가 `--max-state-age-ms`(200)보다 오래되면 **답하지 않는다**: 팔 서버는 `--reply-timeout-ms 50` 뒤 읽기 실패로 치고 신선도 검사로 gate를 닫는다 |
| `0x22` | `send_angles` | 6×i16 센티도 + speed(1 B)를 풀어서 `move_joints{angles_deg, speed}`로 보낸다. pymycobot처럼 센티도로 **버림**된 값(0.29° → 0.28°)이 그대로 간다 |
| `0x67` | `set_gripper_value` | `gripper{value, speed}`로 보낸다. 값은 그대로 0–100 |
| `0x29` | `stop` | `stop`을 **맨 앞으로** 보내고 아직 보내지 않은 이동 명령은 버린다(뒤에 줄 선 이동이 정지 뒤에 팔을 다시 움직이면 안 된다) |
| `0x16` | `set_fresh_mode` | 아무것도 안 한다. sim은 항상 가장 새 명령을 따른다 |
| `0x65` | `get_gripper_value` | sim `state.gripper`를 i16 하나로 답한다 |
| `0x12` | `is_power_on` | `1` |
| `0x07` | `get_error_information` | `0` |
| 그 밖 | | 로그만 남기고 무시한다 |

명령(`0x22`, `0x67`, `0x29`, `0x16`)에는 답하지 않는다. 팔 서버가 기다리지 않기 때문이다. pymycobot은 답이 없으면 0.5초 뒤 같은 프레임을 다시 써서 모두 3번 보낸다(`tools/golden_frames.py` 실측). 그 클라이언트로 붙으면 이동 명령이 3번 간다.

sim-bridge와는 연결을 둘 쓴다. 하나는 `state` 폴링 전용이고, 다른 하나는 명령 전용이라 느린 `move_joints` 응답 뒤에 읽기가 줄 서지 않는다. fw_emu가 끝날 때(예: SIGTERM)는 시뮬레이션에 `stop`을 한 번 보낸다.

## sim-bridge에 가정하는 것

CONTRACT.md의 sim-bridge v0 그대로(`state`, `move_joints`, `gripper`, `stop`, `frame`, …)이고, 각도는 **펌웨어 각도(도)**다. 그 위에 아래를 가정한다. 시뮬레이션 쪽은 이 표대로 구현한다.

| 항목 | 가정 |
| --- | --- |
| speed → 관절 속도 | `speed`(1–100) = 관절 최대 속도 **`speed` °/s**. 시간 동기: `max_j |Δ_j| / speed`초 걸리고 여섯 관절이 함께 도착한다(끝에서 1차 지연으로 부드럽게 멈추는 것은 괜찮다). 팔 서버가 `--max-speed 50`으로 막으므로 실제로는 50 °/s 이하다. 진짜 팔과 같은지 TODO(unverified): 공식 문서는 °/s를 말하지 않는다(`mycobot-control/safety/limits.py`도 같은 말) |
| 그리퍼 값 | 0–100 정수, 그대로 전달. **100 = 열림, 0 = 닫힘**으로 가정한다. pymycobot의 `set_gripper_value`는 방향을 적어 두지 않았고(`set_gripper_state`만 0 = 열기, 1 = 닫기), `mycobot-control`의 키맵도 같은 가정에 TODO(unverified)를 달았다. `sim_calib.json`의 열림/닫힘 값은 이 방향으로 쓴다. `state.gripper`도 같은 0–100 |
| 그리퍼 속도 | `2 × speed` 단위/초 (stub의 가정; 진짜는 TODO(unverified)) |
| `stop` | 지금 **측정한** 각도를 유지한다 |
| `state` 응답 | 몇 ms 안에. 렌더링 뒤에 줄 세우지 않는다. fw_emu는 50 Hz로 읽는다 |
| 각도 해상도 | `get_angles`는 0.01°로 반올림된다. 실제로 이 해상도 아래는 보이지 않는다 |

`stub_sim.py`는 위 표를 그대로 구현한 가짜 sim-bridge다. 관절은 `v = (목표 − q)/0.1 s`를 가장 빠른 관절이 `speed` °/s를 넘지 않게 함께 줄이고, 그리퍼는 선형이며, `frame`은 ffmpeg 시험 화면(5 Hz로 새 그림)이다. 진짜 sim과 이 숫자가 다르면 `TAU_S` 같은 상수가 아니라 이 표가 기준이다.

## 카메라

`cam_push_sim.py`는 sim-bridge `frame`을 5 Hz로 읽어 `camera_push.py`와 같은 틀(`hello` 한 줄, 프레임마다 머리 한 줄 + JPEG)로 두뇌 `:47901`에 보낸다. 두 가지가 진짜 카메라와 다르다. 둘 다 두뇌가 프레임 나이를 *도착 시각*부터 재기 때문이다.

- 같은 그림(`wall`, `sim_t`가 같은 응답)은 한 번만 보낸다. 반복해서 보내면 새 그림과 구별되지 않는다.
- `--max-age`(1 s)보다 오래전에 렌더한 프레임은 보내지 않는다. sim과 같은 PC라서 `wall`은 `time.time()`과 비교할 수 있다.

## 확인한 것, 못 한 것

확인(2026-10-31, 이 PC와 Spark, 가짜 sim): 골든 바이트 일치, pymycobot 4.0.7이 `get_angles`/`get_gripper_value`/`is_power_on`/`get_error_information` 응답을 제대로 읽음, 한 PC e2e와 네트워크 너머 G2(결과는 `BOARD.md`와 커밋 메시지).

TODO(unverified):

- 진짜 Atom 펌웨어가 명령에 응답 프레임을 보내는지, `get_gripper_value`/`is_power_on`/`get_error_information` 응답의 정확한 모양. pymycobot이 읽는 모양으로 답하게 했을 뿐이다.
- 위의 speed → °/s와 그리퍼 방향.
- pty에는 1 Mbaud의 시간 특성이 없다. 진짜 UART에서의 `get_angles` 지연(팔 서버 기본 50 ms)은 이 시험으로 알 수 없다.
- 이 시험은 Atom의 오류 상태(충돌 보호, 한계 초과)를 흉내 내지 않는다.
