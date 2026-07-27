# Apollo Planning + Control — Launch부터 Output까지 전 과정 분석

> Apollo 7.0 기반 코드베이스. Planning 모듈과 Control 모듈의 launch → input → 처리 → output 전 과정을 코드 레벨로 정리.

---

## 0. 전체 파이프라인 한눈에

```
[prediction] [chassis] [localization]
        │
        ▼
  ┌─────────────────┐   /apollo/planning
  │ Planning Module │ ──(ADCTrajectory)──┐
  └─────────────────┘                    │
        ▲                                ▼
        │ (ControlCommand.if_full_stop) ┌─────────────────┐  /apollo/control
        └───────────────────────────────│ Control Module  │──(ControlCommand)──▶ [canbus]
                                         └─────────────────┘   steering/throttle/brake
```

- 두 모듈은 CyberRT의 **독립 프로세스**로 뜨며 **채널(topic)로만 통신**한다.
- **Planning** = "어디로·어떤 속도로 갈지"(궤적)를 결정 → 비주기(예측 입력마다).
- **Control** = 그 궤적을 추종하도록 "핸들/가속/브레이크"를 매 10ms 계산 → 고정 100Hz.

---

## 1. Planning 모듈

### 1-1. Launch → DAG → Component

- **launch**: `modules/planning/launch/planning.launch` → `planning.dag`를 로드
- **DAG**: `modules/planning/dag/planning.dag`
  - 라이브러리: `libplanning_component.so`, 클래스 `PlanningComponent`
  - config: `planning_config.pb.txt`, flags: `planning.conf`
  - **트리거 방식**: `Component`(메시지 구동). readers 3개 중 **첫 채널 `/apollo/prediction`이 메인 트리거**. `chassis`, `localization`은 함께 fused input으로 들어옴.

### 1-2. Init() — `planning_component.cc:52`

- `FLAGS_use_navigation_mode`에 따라 `NaviPlanning`(상대지도) 또는 **`OnLanePlanning`(HD맵 기반, 기본값)** 선택 (`:55-59`)
- 다수의 보조 Reader 등록: routing, traffic_light(v2x/mtl/ptl), pad_msg, storytelling + 커스텀 채널(`ManualEmergencyAvoidance`, `AudioDetection`, `PerceptionLanes`, `GnssInsPvax`, crosswalk_monitor — 한국 도로/커스텀 기능)
- **주목**: `:180` planning이 **control로부터 `ControlCommand`를 역으로 구독** — control의 `if_full_stop` 상태를 참조하는 피드백 루프 존재
- **Writer 등록**: `:207` `planning_writer_` → **`ADCTrajectory`를 planning_trajectory_topic으로 발행** (control의 입력)

### 1-3. Proc() — `planning_component.cc:256` (사이클당 1회)

1. `CheckRerouting()` — 리라우팅 필요시 RoutingRequest 발행 (`:493`)
2. 입력 3종을 `local_view_`에 적재 (`:268-270`)
3. **localization 지연 보정**: 지연이 임계값 초과 시 속도×Δt로 위치 외삽 (`:278-299`)
4. routing 신규성 판단, traffic_light/보조 입력을 local_view에 복사 (`:301-359`)
5. `CheckInput()` — localization/chassis/map/routing 준비 확인. 미준비 시 `not_ready` decision 담은 빈 trajectory 발행 후 종료 (`:513`)
6. **핵심 호출**: `planning_base_->RunOnce(local_view_, &adc_trajectory_pb)` (`:413`)
7. FillHeader 후 timestamp 변경분(dt)만큼 각 point의 relative_time 보정 (`:420-427`)
8. **출력**: `planning_writer_->Write(adc_trajectory_pb)` (`:443`) + 커스텀 `CanEdgeMonitor`(주행구역 접근상태) 발행
9. history에 기록 (`:488`)

> 특이점: `FLAGS_trajectory_write_by_timer`가 켜지면 Proc에서 직접 쓰지 않고 100ms 타이머가 최신 trajectory를 반복 발행 (`:244-252`)

**Planning 출력물 = `ADCTrajectory`**: trajectory_point 배열(x,y,θ,κ,v,a,relative_time), decision(main_decision, vehicle_signal), estop, gear, is_replan 등.

---

## 2. Planning 내부 알고리즘 (RunOnce → Scenario → Task)

### 2-1. 계층 구조

```
PlanningComponent → OnLanePlanning(기본) → PublicRoadPlanner(기본)
  → ScenarioManager → Scenario → Stage → Task
```

- **Planner 선택**: `on_lane_planner_dispatcher.cc:26` → config `planner_type = PUBLIC_ROAD`. 대안: LATTICE, RTK, NAVI

### 2-2. RunOnce — 계획 1사이클 준비 (`on_lane_planning.cc:368`)

1. `vehicle_state()->Update()` — localization+chassis로 차량 상태 갱신
2. `reference_line_provider_->UpdateVehicleState()` — 백그라운드 스레드에 상태 전달
3. **`TrajectoryStitcher::ComputeStitchingTrajectory()`** (`:581`) — 이전 궤적 이어붙이기(stitching), replan 판정. 제어 안정성을 위해 매 사이클 처음부터 다시 계획하지 않고 직전 궤적의 현재 시점 지점부터 연결
4. **`InitFrame()`** (`:591`) → `GetReferenceLines()`로 기준선 확보 + `Frame::Init()`으로 장애물·reference_line_info 구성
5. **`TrafficDecider::Execute()`** (`:642`) — 각 기준선에 교통 규칙 적용(신호등/정지선/횡단보도/양보), 위반 시 non-drivable 마킹
6. **`Plan()`** (`:653`) — 실제 계획

### 2-3. Scenario 선택 — 상태기계 (`scenario_manager.cc:1071`)

`PublicRoadPlanner::Plan`이 `ScenarioManager::Update`로 시나리오 선택. **우선순위 순 디스패치**:

1. 기본 = **`LANE_FOLLOW`**
2. **sticky 유지**: 교차로/정지/신호/주차/데드엔드 시나리오 진행 중(STATUS_DONE 아님)이면 계속 유지
3. `SelectParkAndGo` → `SelectInterception`(교차로) → `SelectPullOver` → `SelectValetParking` → `SelectDeadEnd` → `SelectDetour`

지원 시나리오: LANE_FOLLOW, BARE_INTERSECTION, STOP_SIGN, TRAFFIC_LIGHT(protected/좌/우), YIELD_SIGN, PULL_OVER, EMERGENCY(pull_over/stop), PARK_AND_GO, VALET_PARKING, DEADEND, DETOUR 등

### 2-4. Stage → Task 실행 — 실제 궤적 생성

`Scenario::Process` → `Stage::Process` → task_list_ 순차 실행. **LANE_FOLLOW task 순서**:

```
── 경로(Path) 단계 ──
LANE_CHANGE_DECIDER → LFA_DECIDER → PATH_REUSE_DECIDER → PATH_LANE_BORROW_DECIDER
→ PATH_BOUNDS_DECIDER → PIECEWISE_JERK_PATH_OPTIMIZER → PATH_ASSESSMENT_DECIDER → PATH_DECIDER
── 속도(Speed) 단계 ──
→ RULE_BASED_STOP_DECIDER → ST_BOUNDS_DECIDER → SPEED_BOUNDS_PRIORI_DECIDER
→ SPEED_HEURISTIC_OPTIMIZER → SPEED_DECIDER → SPEED_BOUNDS_FINAL_DECIDER
→ PIECEWISE_JERK_SPEED_OPTIMIZER → RSS_DECIDER
```

핵심 설계: **경로(횡방향)와 속도(종방향)를 분리(path-speed decoupling)** 해서 계획.

- **Deciders**(결정): 경계 설정, 차선 빌림, 정지 결정 등 규칙/판단
- **Optimizers**(최적화):
  - `PIECEWISE_JERK_PATH_OPTIMIZER` — 횡방향 경로 QP
  - `SPEED_HEURISTIC_OPTIMIZER` — ST 그래프 DP 탐색
  - `PIECEWISE_JERK_SPEED_OPTIMIZER` — 종방향 속도 QP
  - jerk 최소화로 부드러운 프로파일 생성
- `CombinePathAndSpeedProfile()`이 path+speed 결합 → `DiscretizedTrajectory` → `ConstraintChecker::ValidTrajectory()` 검증

### 2-5. ReferenceLineProvider — 기준선 공급 (별도 스레드)

- routing 응답을 부드러운 중심선 궤적으로 변환, **50ms 주기 백그라운드 스레드**로 지속 생성 (`reference_line_provider.cc:260`)
- 스무더: discrete_points / qp_spline / spiral 방식 선택 가능
- planning 메인 루프와 **비동기 분리**되어 계획 사이클 지연 감소

### 2-6. Planning 출력 = `ADCTrajectory` 조립

- `frame_->FindDriveReferenceLineInfo()` — **최소 cost의 주행가능 기준선 선택** (`:991`)
- `PopulateTrajectoryProtobuf()` — 최종 trajectory_point 기록 (`:1267`)
- 주요 필드: **`trajectory_point`(x,y,θ,κ,v,a,relative_time)**, `decision`(main/object + vehicle_signal), `estop`, `gear`, `is_replan`, `trajectory_type`, **`control_trajectory`(control 전용)**, `engage_advice`
- **채널**: `/apollo/planning`

---

## 3. Control 모듈

### 3-1. Launch → DAG → Component

- **launch**: `modules/control/launch/control.launch` → `control.dag`
- **DAG**: `modules/control/dag/control.dag`
  - 라이브러리 `libcontrol_component.so`, 클래스 `ControlComponent`
  - **트리거 방식이 planning과 다름**: `timer_components`, **`interval: 10` (10ms = 100Hz 고정 주기)**. control은 메시지가 아니라 **타이머로 주기 구동**되고, 내부에서 최신 메시지를 Observe.

### 3-2. Init() — `control_component.cc:41`

- `control_conf.pb.txt` 로드 (`:47`)
- `FLAGS_use_control_submodules`가 **false(기본)**면 `controller_agent_.Init()` 호출 (`:54`)
- Reader 등록(콜백 없이 nullptr, Proc에서 직접 Observe): **chassis, trajectory(=planning 출력), localization, pad_msg** (`:61-87`)
- **Writer**: submodule 미사용 시 `control_cmd_writer_` → **`ControlCommand`를 control_command_topic으로 발행** (`:90`). submodule 모드면 `LocalView` 발행.

### 3-3. Proc() — `control_component.cc:257` (10ms마다)

1. chassis/trajectory/localization/pad를 각각 `Observe()` → 최신값, 없으면 조기 return (`:260-289`)
2. 세 입력을 `local_view_`에 복사 (`:291-300`)
3. submodule 모드면 local_view만 발행하고 종료 (`:303-317`)
4. pad_msg가 RESET이면 estop 해제 (`:321`)
5. **핵심 호출**: `ProduceControlCommand(&control_command)` (`:338`)
6. header/latency 채우고 **출력**: `control_cmd_writer_->Write(control_command)` (`:379`)

### 3-4. ProduceControlCommand() — `control_component.cc:147` (제어 계산 관문)

1. `CheckInput()` — trajectory point 유무, 저속시 v/a를 0으로 스냅, **VehicleState 업데이트** (`:383-411`)
2. `CheckTimestamp()` — localization/chassis/trajectory 타임아웃 검사 (`:413`)
3. **estop 판정** — planning estop, 빈 trajectory, GEAR_DRIVE인데 음속도 (`:172-196`)
4. estop 아니면 → **`controller_agent_.ComputeControlCommand(localization, chassis, trajectory, control_command)`** (`:217`) ← 실제 LQR/PID/MPC 계산
5. estop이면 → 안전 정지: speed=0, throttle=0, brake=`soft_estop_brake`, accel=`soft_estop_deceleration` (`:236-244`)
6. planning decision의 vehicle_signal(등화)과 gear를 명령에 반영 (`:246-252`)

---

## 4. Control 내부 알고리즘 (LQR / PID / MPC)

### 4-1. 두 가지 실행 아키텍처

`FLAGS_use_control_submodules`로 결정 (기본 false):

- **모놀리식(기본)**: `ControlComponent` 하나가 입력→계산→출력 전부 수행 (`control.dag`)
- **서브모듈 파이프라인**: `Preprocessor → (LatLon 또는 MPC) → Postprocessor` 3단 컴포넌트 체인

> **현재 활성 설정**: `control_conf.pb.txt:27` → `active_controllers: MPC_CONTROLLER` **하나만** 활성. 지금은 MPC가 횡·종방향 통합 제어.

### 4-2. ControllerAgent — 컨트롤러 오케스트레이터 (`controller_agent.cc`)

- `Init()` → `RegisterControllers()` + `InitializeConf()`: `active_controllers()`를 순회하며 factory로 인스턴스 생성 → `controller_list_`에 push
- `ComputeControlCommand()` (`:143`): **`controller_list_`를 순서대로 순회**하며 각 컨트롤러가 **같은 `cmd` 객체를 나눠 채움** (예: LQR-Lat이 steering, PID-Lon이 throttle/brake)

### 4-3. 세 컨트롤러의 이론적 역할

**(1) MPCController — 현재 활성, 통합 QP 제어** (`mpc_controller.cc:338`)

- **상태 6개**: lateral_error, lateral_error_rate, heading_error, heading_error_rate, station_error, station_error_rate
- **제어 2개**: steer, acceleration (조향+가감속 **동시 최적화**)
- 흐름: 속도 저역필터 → handover 리셋 → `ComputeLongitudinalErrors()` → `UpdateState/UpdateMatrix/FeedforwardUpdate()`로 이산 시스템 행렬 갱신 → **Gain Scheduler**(속도별 Q/R/feedforward 보간) → 제어 상·하한 설정 → **`MpcOsqp::Solve()`** (`:508`, OSQP 기반 QP 풀이)
- **출력**: `set_steering_target()` + `set_steering_target_torque()`(backstepping) → accel은 폐루프+slope보정+standstill 처리 후 `Clamp` → **calibration table 룩업으로 accel→throttle/brake 변환** (`:788`)
- **설정**: `mass:1785, izz:3392, steer_ratio:15.3`, Q가중 `matrix_q1=[18000,1000,10,1]`, R가중 `matrix_r1=3000000`, `standstill_acceleration:-3.0`

**(2) LatController — LQR 횡방향** (`lat_controller.cc`)

- **자전거 동역학 모델**: 시변 상태행렬 `matrix_a_`(속도 종속, cf/cr·mass·lf/lr·iz)
- `SolveLQRProblem()`로 매 주기 이득 `matrix_k_` 재계산
- **제어법칙**: `steer = -(K·state)`(LQR 피드백) + `feedforward`(곡률 기반) + `feedback_augment`(leadlag) → limit/MRAC/filter/`Clamp(-100,100)`
- **출력**: `set_steering_target()`, `set_steering_rate()`만. throttle/brake는 안 건드림 → LonController와 짝

**(3) LonController — PID 종방향** (`lon_controller.cc`)

- **이중 PID 캐스케이드**: `station_pid_controller_`(위치 오차 → speed_offset) → `speed_pid_controller_`(속도 오차 → accel_cmd)
- 속도대역별 PID 세트(low/high/reverse) 전환
- **calibration table 보간**으로 (speed, accel) → command 룩업, 양수=throttle / 음수=brake
- **출력**: `set_throttle/set_brake/set_acceleration`만. steering은 안 건드림

> **핵심 대비**: MPC = 단일 QP로 steer+accel 동시 / LQR+PID 조합 = 두 컨트롤러가 cmd 분담. 지금은 MPC 단독 활성.

### 4-4. Control 출력 = `ControlCommand`

`steering_target`[-100,100%], `steering_rate`, `throttle`[0,100%], `brake`[0,100%], `acceleration`, `speed`, `gear_location`, `signal`(등화), `engage_advice`, `latency_stats`, `debug`(SimpleMPCDebug 등) → **채널 `/apollo/control`로 발행 → canbus 수신**

---

## 5. 전체 통합 — End-to-End

```
                          ┌──────────── PLANNING (메시지 구동, prediction마다) ─────────────┐
[prediction]              planning.launch → planning.dag (Component<Prediction,Chassis,Localization>)
[chassis]      ──fused──▶ PlanningComponent::Proc()  [planning_component.cc:256]
[localization]              └ OnLanePlanning::RunOnce()  [on_lane_planning.cc:368]
                               ├ VehicleState Update
                               ├ TrajectoryStitcher (이전궤적 연결·replan판정)
                               ├ InitFrame ← ReferenceLineProvider (50ms 별도스레드)
                               ├ TrafficDecider (신호/정지/양보 규칙)
                               └ PublicRoadPlanner::Plan()
                                   ├ ScenarioManager::Update (LANE_FOLLOW 등 시나리오 선택)
                                   └ Scenario→Stage→Task
                                       [Path]  bounds→PiecewiseJerkPath(QP)→decider
                                       [Speed] STbounds→heuristic(DP)→PiecewiseJerkSpeed(QP)
                               └ ADCTrajectory 조립 → publish
                          └─────────────────────────────┬────────────────────────────────┘
                                                        │ /apollo/planning (ADCTrajectory)
                                                        ▼
                          ┌──────────── CONTROL (타이머 구동, 10ms=100Hz) ─────────────────┐
[chassis]                 control.launch → control.dag (TimerComponent interval:10)
[localization] ─Observe─▶ ControlComponent::Proc()  [control_component.cc:257]
[/apollo/planning]          └ ProduceControlCommand()  [:147]
[pad]                          ├ CheckInput → VehicleState Update
                               ├ CheckTimestamp / estop 판정
                               └ ControllerAgent::ComputeControlCommand()  [:217]
                                   └ MPCController (현재 활성): 상태6·제어2 QP(OSQP)
                                      · set_steering_target (조향)
                                      · calibration table → set_throttle/brake (가감속)
                                      (대안: LQR-Lat[조향] + PID-Lon[가감속] 조합)
                          └─────────────────────────────┬────────────────────────────────┘
                                                        │ /apollo/control (ControlCommand)
                                                        ▼
                                                    [canbus] → 차량 액추에이터
```

---

## 6. 핵심 대비표

| 구분 | Planning | Control |
|---|---|---|
| **트리거** | 메시지 구동 (`/apollo/prediction` fusion) | 타이머 100Hz (10ms 고정) |
| **입력** | prediction, chassis, localization (+routing, traffic_light…) | planning의 ADCTrajectory, chassis, localization, pad |
| **핵심 알고리즘** | Scenario FSM + Path/Speed 분리 최적화(QP) | MPC(QP/OSQP) 또는 LQR+PID |
| **핵심 질문** | "어디로·어떤 속도로 갈까" (궤적 생성) | "궤적 추종하려면 핸들·가속·브레이크 얼마" |
| **출력** | `ADCTrajectory` (경로점+속도 프로파일) | `ControlCommand` (steering/throttle/brake/accel) |
| **주기 특성** | 비주기, 계산량 큼 | 고정 고주파, 실시간성 중요 |

---

## 7. 이 코드베이스의 특이사항 (표준 Apollo 7.0 대비)

- **Planning**: control로부터 `ControlCommand`를 **역방향 구독**(`planning_component.cc:180`) — full_stop 피드백. 한국 도로 대응 커스텀 채널 다수(mtl/ptl 신호등, 수동긴급회피 mea, 긴급차량대응 evr, LFA 차선인식, crosswalk_monitor). localization 지연 외삽 보정.
- **Control**: 표준 Apollo는 보통 LAT+LON 조합이 기본이지만, **이 설정은 MPC_CONTROLLER 단독 활성**(`control_conf.pb.txt:27`).

---

## 부록. 주요 파일 인덱스

**Planning**
- 진입점: `planning_component.cc:52`(Init), `:256`(Proc), `:413`(RunOnce 호출), `:443`(publish)
- 메인 로직: `on_lane_planning.cc:368`(RunOnce), `:591`(InitFrame), `:653`(Plan)
- Planner: `public_road_planner.cc:33`(Plan), `on_lane_planner_dispatcher.cc:26`(선택)
- Scenario: `scenario_manager.cc:1071`(선택), `scenario.cc:66`(Process), `:69`(생성)
- Stage/Task: `stage.cc:91`(task 실행), `lane_follow_stage.cc:494`, `task_factory.cc:70`(등록)
- Reference line: `reference_line_provider.cc:260`(GenerateThread), `:295`(GetReferenceLines)
- Output proto: `proto/planning.proto`(ADCTrajectory)
- Config: `conf/planning_config.pb.txt`, `conf/scenario/lane_follow_config.pb.txt`

**Control**
- 진입점/오케스트레이션: `control_component.cc`, `control_component.h`
- Controller 관리: `controller/controller_agent.cc`, `controller.h`
- 활성 controller: `controller/mpc_controller.cc`(활성), `lat_controller.cc`(LQR), `lon_controller.cc`(PID)
- 서브모듈: `submodules/{preprocessor,lat_lon_controller,mpc_controller,postprocessor}_submodule.cc`
- 설정: `conf/control_conf.pb.txt`(`active_controllers`, `:27`), `conf/control.conf`
- 메시지: `proto/{control_cmd,control_conf,local_view,pad_msg}.proto`
- DAG/launch: `dag/control.dag`, `launch/control.launch`
