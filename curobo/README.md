# mycobot_curobo

myCobot 280 JetsonNano(adaptive gripper는 열린 상태로 고정)용 cuRobo v2 설정입니다. 로봇 없이 trajectory를 계획하고 확인할 수 있습니다.
`nero_curobo`와 같은 구조이며 `~/manipulation/curobo/.venv`(torch + cuRobo)에서 실행합니다.

## 위치와 환경 (2026-11-26)

- 이 폴더는 원래 저장소 밖의 `~/manipulation/mycobot_curobo`였습니다. 2026-11-26에 `mycobot-link/curobo/`로 옮겼고, 예전 경로는 이 폴더를 가리키는 symlink입니다.
- 웹 콘솔과 `planner/`가 쓰는 계획 서버는 `scripts/run.sh server`(API 8096, viser 8080)입니다. `tools/console_up.sh`가 이 폴더에서 띄웁니다.
- `configs/*.yml`과 `urdf/*.urdf`에는 절대 경로가 들어 있습니다. cuRobo가 상대 경로를 자기 asset 폴더 기준으로 붙이기 때문입니다. 경로는 `/home/hkim/manipulation/mycobot-link/curobo` 기준이고, 다른 위치에 clone하면 `run.sh`가 경로를 바꾼 사본을 `.gen/`에 만들어 씁니다.
- 실행 환경(이 PC, 2026-11-26): `~/manipulation/curobo`는 NVlabs/curobo `main` 78fd485(`v0.8.0-43`), venv는 Python 3.11.16, torch 2.14.0+cu130입니다. cuRobo는 non-editable로 설치돼 있습니다. 새 PC에서는 같은 commit을 clone하고 그 안에 `.venv`를 만든 뒤 `uv pip install . --no-build-isolation`으로 설치합니다. 위치가 다르면 `CUROBO_DIR`로 알려 줍니다.
- `plans/`(2026-09-23 데모 GIF·MP4·JSON)는 생성물이라 git에 넣지 않았습니다.
- `scripts/teleop_sim.py`는 `~/manipulation/mycobot-control/src`의 키 배치를 import합니다. `rebuild`는 `~/manipulation/arc-mycobot`(315b731)과 vendor `~/manipulation/mycobot_ros2`(d42ff61)가 있어야 합니다. 다시 맞춘 collision sphere는 지금 것과 달라질 수 있으니, 꼭 필요할 때만 rebuild합니다.
`mycobot-control`은 torch나 cuRobo를 import하지 않습니다. 두 프로젝트 사이에는 `mycobot-joint-trajectory/1` JSON 파일만 오갑니다.

```
urdf/mycobot_280_jn.urdf       arc-mycobot 수리본(welded) + 미터 단위 STL + convex-hull collision + tcp
urdf/mycobot_280_jn_camera_gripper.urdf   위 URDF에 camera flange 추가 (J6 flange와 gripper 사이) + camera_link
assets/meshes/                 변환된 STL과 *_hull.stl (생성물)
configs/mycobot_280_jn.yml     cuRobo robot config (sphere 153개, self-collision ignore)
configs/mycobot_280_jn_camera_gripper.yml   camera flange + gripper 버전 (camera +y relative to tool mount)
mycobot_planner.py             planner 인터페이스 (plan_to_joints / plan_to_pose / to_json)
scripts/build_urdf.py          URDF 생성
scripts/convert_meshes.py      DAE -> STL (단위와 깨진 material 처리)
scripts/build_config.py        URDF -> cuRobo config
scripts/plan_demo.py           계획 결과를 plans/ 아래 JSON + GIF + 관절 그래프로 저장
scripts/viewer.py              브라우저 인터랙티브 viewer
scripts/teleop_sim.py          키보드 EE teleop (터미널 키 입력 + viser 화면)
scripts/diagnose_collision.py  특정 자세에서 어떤 link가 충돌하는지 표시
scripts/run.sh                 실행 스크립트
```

## 실행

```bash
scripts/run.sh viewer                       # http://localhost:8080
scripts/run.sh teleop                       # 키보드 EE 조작 (http://localhost:8080, 키는 터미널에서)
scripts/run.sh demo                         # 예제 4개: zero→home, 좌 10cm, 아래 5cm, home 복귀
scripts/run.sh demo --joints 45 -30 -60 0 0 0          # 관절 목표 (도, pymycobot 단위)
scripts/run.sh demo --pose 0.15 0.05 0.15 --name reach # TCP 목표 (m, 현재 방향 유지)
scripts/run.sh check                        # self-check
scripts/run.sh spheres                      # collision sphere 확인 (브라우저)
```

앞에 `VARIANT=camera`를 붙이면 camera flange + gripper 모델을 사용합니다. 예: `VARIANT=camera scripts/run.sh viewer`.
두 모델은 JSON의 `"robot"` 필드로 구분됩니다.

viewer 사용법:
- gizmo를 끌고 **Move**를 누르면 TCP 목표로 이동합니다(방향은 고정).
- 관절 슬라이더를 맞추고 **Plan to joints**를 누르면 관절 목표로 이동합니다.
- **Save plan**을 누르면 마지막 계획이 `plans/viewer-*.json`으로 저장됩니다.
- 계획이 실패하면 IK 단계에서 실패했는지 trajectory 단계에서 실패했는지 상태 패널에 표시됩니다.

## 키보드 EE teleop (시뮬레이션)

```bash
VARIANT=camera scripts/run.sh teleop        # 브라우저로 http://localhost:8080 을 열고, 키는 이 터미널에서 누릅니다
```

viser에는 키보드 이벤트 API가 없습니다. 그래서 키는 **터미널**에서 받고, 화면은 **브라우저**에 그립니다. 브라우저 패널에도 같은 동작을 하는 jog 버튼이 있습니다.
키 배치는 mycobot-control의 `teleop/keymap.py`를 그대로 import하므로, 실제 로봇 teleop(`mycobot-teleop-keyboard`)과 항상 같습니다.

```
cartesian  w/s ±X  a/d ±Y  r/f ±Z   u/j ±rx  i/k ±ry  o/l ±rz     (기본 5 mm / 2°)
joint      w e r t y u = +J1..+J6   s d f g h j = -J1..-J6          (기본 2°)
[ ] 스텝 ½/2배   m 모드 전환   n 회전 기준 base↔tool   z home(cuRobo 계획)   space 정지   ? 도움말   q 종료
```

- **cartesian 키:** 현재 TCP 자세에 한 스텝을 더한 뒤 IK를 풉니다. 충돌이 없고, 현재 관절값에 가장 가까운 해를 고릅니다.
  - 어떤 관절이든 한 스텝에 20° 넘게 움직이는 해는 거부합니다. mycobot-control이 실제 명령에 적용하는 `MAX_COMMAND_DELTA_DEG`와 같은 값입니다. 20°가 넘는다면 IK branch가 바뀌었거나 특이점 근처라는 뜻입니다.
  - IK 한 스텝은 3 ms 정도 걸립니다.
- **joint 키:** 관절 하나를 한 스텝 움직입니다. 관절 한계(margin 3°)를 넘거나 충돌하면 거부합니다.
- **기준 자세:** 매 키는 **현재 화면에 그려진 자세**에서 목표를 계산합니다. 목표를 누적하지 않는 실제 teleop과 같은 방식입니다.
- **화면 움직임:** 그려지는 팔은 해를 향해 90°/s(`--display-speed`)로 따라갑니다.
- **축 방향 주의:** cartesian 축은 URDF(`joint1`) 기준입니다. 펌웨어 coords는 z축 기준으로 90° 돌아가 있다고 vendor가 말하므로(**검증하지 않음**), 실제 로봇 cartesian 모드의 +X와 다를 수 있습니다. joint 모드는 양쪽이 같습니다.

구현 메모 (2026-09-23에 재현하고 고친 것):
- **렌더 스레드는 GPU를 쓰지 않습니다.** 렌더 스레드가 cuRobo FK를 부르는 동안 키 스레드가 IK를 풀면 `CUDA_ERROR_ILLEGAL_INSTRUCTION`으로 죽었습니다. 그래서 화면용 좌표계는 numpy URDF FK로 계산합니다. numpy FK는 cuRobo FK와 0.0000 mm로 일치합니다.
- **충돌 검사는 planner 자신의 kinematics와 numpy로 합니다**(`collision_report`).
  - 같은 프로세스에 cuRobo `RobotCollisionChecker`를 만들면, 그 뒤의 `solve_pose`가 모두 CUDA 오류로 죽었습니다.
  - 그 전에 쓰던 `plan_cspace(q, q)` 방식은 충돌이 아닌 이유로도 실패해서, 방금 지나온 자세를 거부했습니다.

## camera flange + gripper 모델

vendor는 camera flange와 gripper를 **함께** 붙인 URDF를 제공하지 않습니다. JN용 camera flange URDF도 없습니다.
vendor가 제공하는 것은 280 Pi/M5/RISCV용 camera flange 단독 버전과 camera flange + pump 버전입니다. 세 모델의 `camera_flange.dae`는 md5가 같은 동일한 파일입니다.
이 파일들을 `joint6_flange` 좌표계에서 측정한 결과는 다음과 같습니다.

| | flange 축 방향(z) 범위 |
| --- | --- |
| J6 flange 면 | z = 1.7 mm |
| pump / adaptive gripper (camera 없음) | 1.6–1.9 mm부터 시작 |
| camera flange 판 | 1.9–18.1 mm (두께 16.2 mm). vendor 배치에서 lens 중심 (41.5, 0, 18.1) mm, +z 방향을 봄 |
| pump (camera 있음) | 18.9 mm부터 시작, 즉 **+17.0 mm** |

vendor는 도구를 17.0 mm 더 바깥으로 옮기는 것 외에는 아무것도 바꾸지 않습니다. `build_urdf.py --camera-flange`는 이 규칙을 adaptive gripper에 그대로 적용합니다.
- camera flange는 vendor Pi 파일의 상대 위치에 둡니다.
- gripper와 `tcp`를 17 mm 바깥으로 옮깁니다(tcp는 flange +z 96 mm).
- lens 위치에 `camera_link`를 추가합니다(z축이 보는 방향). 카메라의 상대 yaw 90°에서는 `tool_mount` 기준 (0, 41.5, 18.1) mm입니다.

실물에서 `pymycobot.send_angles([0, 0, 0, 0, 0, 0], 20)`을 실행했을 때, 정면에서 본 카메라와 그리퍼가 이전 모델보다 함께 반시계 방향으로 약 45° 돌아가 있다는 사용자 관찰(2026-09-28)을 반영했습니다. 결합 URDF의 `joint6_flange` 아래에 `tool_mount`를 추가해 flange +z축으로 +45° 회전시키고, 카메라 플랜지·그리퍼·`tcp`를 모두 그 아래에 연결합니다. `--assembly-yaw` 기본값이 45°이며, `--camera-yaw` 기본값 90°는 **그리퍼에 대한 카메라의 상대 방향**으로 유지됩니다. 실측 각도와 hand-eye 보정은 아직 하지 않았습니다.

실제 조립된 로봇에서 확인해야 할 것:
1. **카메라 상대 방향(`--camera-yaw`, 기본 90°).** vendor 파일(0°)에서는 카메라가 장착 프레임 +x 쪽으로 나옵니다. 이 방향은 jaw가 벌어지는 축과 같아서, 오른쪽 finger가 렌즈 바로 위에 옵니다. 그래서 2026-09-23에 사용자 요청으로 **+y 쪽(yaw 90°)**을 기본값으로 바꿨습니다. 렌즈 (0, 41.5, 18.1) mm는 finger(y ≤ 18 mm)와 gripper base(y ≤ 13 mm) 바깥에 있습니다. 실물의 공통 장착각이 다르면 `--assembly-yaw <deg>`, 카메라의 상대 방향이 다르면 `--camera-yaw <deg>`로 URDF를 재생성하고 config를 rebuild하세요.
2. **17.0 mm 스택 높이.** camera flange 두께를 캘리퍼로 재면 확정됩니다.
3. **`camera_link`.** 메시에서 구한 렌즈 중심일 뿐입니다. 실제 광학 중심과 축은 hand-eye calibration이 필요합니다.

## 모든 계획에 적용되는 제한

| 항목 | 값 | 근거 |
| --- | --- | --- |
| 관절 위치 | URDF 한계 − 3° | mycobot-control의 `check_joint_target` margin과 같습니다 |
| 관절 속도 | 120°/s × `speed_fraction`(기본 0.25 = 30°/s) | spec sheet 값입니다. **측정하지 않았습니다** |
| 가속도 | 4 rad/s² | 임의로 정한 보수적인 값입니다. **측정하지 않았습니다** |
| 바닥 | base 아래 slab (윗면 z = −2 mm) | 테이블을 관통하는 경로를 막습니다 |

## 검증한 것 (2026-09-23, 하드웨어 없이)

- cuRobo FK와 arc-mycobot의 numpy FK(`urdf_fk.py`)가 home 자세와 무작위 50개 자세에서 0.0001 mm 이내로 일치합니다.
- self-check 4개 계획이 모두 성공했습니다. goal 오차는 0.00 mm였고, peak 속도 26–28°/s로 30°/s cap 안에 들었습니다.
- home 자세에서 self-collision과 바닥 충돌이 없습니다.

## 검증하지 않은 것

- **TCP 위치:** flange +z 79 mm는 URDF에서 추정한 값입니다.
- **timing:** pymycobot `speed`(1–100)가 실제로 몇 °/s인지 모르기 때문에, 계획의 시간 축은 모양만 참고하세요.
- **pymycobot 좌표계와의 관계:** 펌웨어 base frame은 URDF `joint1`과 z축 기준 90° 돌아가 있다고 vendor가 말합니다. 관절 공간 계획에는 영향이 없습니다.

## 빌드 과정에서 고친 문제

1. **vendor DAE의 단위가 파일마다 다릅니다.** mm와 m가 섞여 있는데 trimesh는 `<unit>` 태그를 무시합니다. 그래서 base와 gripper가 1000배 크게 읽혔습니다.
2. **`joint5.dae`는 존재하지 않는 material을 참조합니다.** 이 때문에 pycollada가 scene node를 버리고 회전 정보가 사라졌습니다. `<bind_material>`을 제거한 뒤 읽습니다.
3. **mesh 대부분이 열린 shell입니다.** 그래서 sphere fitting이 실패했습니다(`gripper_base` coverage 0.6%). 모든 link에 convex hull을 collision mesh로 주었습니다.
4. **base sphere가 바닥 아래로 11 mm 튀어나왔습니다.** 이 때문에 모든 자세가 충돌로 판정됐습니다. z ≥ 0 위로 clamp했습니다(nero_curobo와 같은 방법).
5. **vendor camera flange joint에는 주석 처리된 `<origin>`이 먼저 나옵니다.** 텍스트 검색으로는 그 값(rpy −1.5708)이 잡히지만, 실제로 쓰이는 값은 `xyz 0 0 0.01, rpy 1.579 0 0`입니다.
