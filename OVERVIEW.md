# PLUTO × Apollo 시뮬레이션 파이프라인 — 전체 개요

> 운영 관점 개요(무엇이 도는가 / 무엇이 필요한가 / 출처). 각 단계의 내부 원리·의미는 [PIPELINE.md](PIPELINE.md) 참고.

## 1. 이게 뭐 하는 거냐

**Apollo 7.0 자율주행 차량 로그(`.record`) + HD맵** 을 입력으로, PLUTO 플래너를 돌려 **closed-loop 시뮬레이션 영상**을 만든다. 팀 공유·평가용.

큰 그림은 **두 개의 파이프라인**이 하나의 아티팩트(`model.onnx`)로 연결된 구조:

```
[학습 파이프라인 · GPU 이미지]                 [시뮬 파이프라인 · CPU 이미지 = 이 문서]
 train(.pth) → export(onnx) → parity check  ──▶  model.onnx  ──▶  parse → feature → 추론 → 렌더 → mp4/gif
   (모델당 1번, 무거움)                          (버전 아티팩트)     (모델당 N번, 시나리오·설정마다)
```
- 학습·export는 이 이미지가 **아니다**(별도 GPU 환경). 여기는 **sim 전용**.
- sim은 **CPU로 충분**(추론=onnxruntime CPU). 신경망 forward는 전체의 ~4%, 병목은 CPU 지오메트리(feature 빌드)+렌더라 GPU 이득 없음. → 2060·A6000·5880·노트북 어디서든 동일 동작.

## 2. 전체 데이터 흐름

```
 (1회) base_map.bin ──build_map_graph.py──▶ map_graph.pkl          (레인그래프+신호매핑)
                                                   │
 record(.record) ──dump_raw.py──▶ raw.pkl ─────────┤   (프레임별 ego·agent·route·TL)
                                                   │
             model.onnx ─────────────────┐         │
                                         ▼         ▼
                        render_closed_loop_post.py (NRCLS)   ─▶ nrcls.mp4
                        render_ols_drive.py (gt/rollout/…)   ─▶ gt.mp4 …
                                                   │
                                          ffmpeg   ▼
                                                  x.gif
```

## 3. 필요한 입력과 출처

| 입력 | 무엇 | 출처 / 만드는 법 |
|---|---|---|
| **`.record`** | Apollo 7.0 차량 주행 로그 (perception 10Hz / localization 125Hz / chassis / planning / routing / traffic_light) | 차량에서 수집된 bag. 현재 `bag/`, `bag_2/`에 6개. |
| **`base_map.bin`** | Apollo HD맵 (lane/roadblock/crosswalk/signal) | 맵 zip. **SEL_KNM2** → `map/base_map.bin`, **AYG** → `map_ayg/base_map.bin`. record별로 맞는 맵 확인 필수. |
| **`map_graph.pkl`** | 위 base_map을 PLUTO용 레인그래프로 변환한 캐시(신호매핑 포함) | `build_map_graph.py` 로 1회 생성. SEL→`out/map_graph.pkl`, AYG→`out/map_graph_ayg.pkl`. |
| **`model.onnx`** | PLUTO 추론 모델 (17MB) | 학습 파이프라인의 export 산출물. `pluto_onnx/onnx_export/model.onnx` (심링크 → `pluto_3rd_model.onnx`, 버전관리됨: 2nd/3rd/4th). |
| **차량 제원** | E100 / U100 치수·조향비·IMU오프셋 | 코드 상수 (`apollo_scenario.py`의 `VEH`). `PLUTO_VEHICLE`로 선택. |

## 4. 스테이지별 스크립트

### 실행하는 것 (파이프라인 본체)
| 스크립트 | 역할 | 핵심 인자 |
|---|---|---|
| `build_map_graph.py` | base_map.bin → map_graph.pkl (레인그래프+roadblock+신호). **맵당 1회.** | `--map <base_map.bin>` |
| `dump_raw.py` | record → raw.pkl. 프레임별 ego 21스텝·agent·미래경로·**route(routing 기반)**·**신호등(TL)**. | `--map-graph`, `--route-from routing`(기본), `--step 2`, `--out` |
| `render_ols_drive.py` | **GT / rollout / combo / ols** 모드 주행 영상. | `--mode`, `--steps`, `--start`, `--switch-frame`(combo) |
| `render_closed_loop_post.py` | **NRCLS** (진짜 closed-loop: 매 스텝 PLUTO 재계획 + LQR + NR agent). | `--steps`, `--ego-pose`, `NRCLS_TRACE` |
| (ffmpeg) | 영상 구간 잘라 gif. | `-ss <시작s> -to <끝s>` + palettegen |

### 어댑터 / 라이브러리 (직접 실행 X, import됨)
| 스크립트 | 역할 |
|---|---|
| `apollo_env.py` | apollo_pb2 경로 세팅 + 메시지 타입 레지스트리 |
| `apollo_nuplan_map.py` | `ApolloMap` = nuPlan `AbstractMap` 덕타이핑(레인/roadblock/폴리라인, 공간인덱스, 0.5m densify) |
| `apollo_scenario.py` | raw+맵 → nuPlan `EgoState`/`Agent`/`PlannerInput` 변환. `build_observations`(agent·margin), `build_tl`(신호), `check_frame`(불변식 가드), `VEH`(차량제원) |
| `render_closed_loop.py` | `propagate_lqr` = 레포 `ForwardSimulator`(LQR+bicycle) 래퍼, `DT=0.1` |
| `render_real.py` | OLS 오버레이 + `_annotate`/`_font`/`_GtState`(다른 렌더가 재사용) |
| `record_to_npz.py` | `load_channels`·`nearest_idx`·`HIST_STEPS` 등 record 읽기 유틸 |

### 진단 / 실험용 (파이프라인 아님)
`overview_bt25.py`(route/ref 시각화), `check_collision.py`, `idm_agents.py`(반응형 agent 실험), `render_reactive.py`

### PLUTO 쪽 (pluto_onnx, **허락 없이 수정 금지 — 네트워크/post**)
`src/planners/pluto_planner.py`(PlutoPlanner: 12후보→trajectory_evaluator→best), `src/planners/ort_planning_model.py`(ONNX 러너), `src/feature_builders/pluto_feature_builder.py`(feature), `src/feature_builders/nuplan_scenario_render.py`(렌더러), `src/post_processing/forward_simulation/*`(LQR), `.../trajectory_evaluator.py`(충돌평가; `PLUTO_EGO_WIDTH/LENGTH/COLLISION_SCALE` 여기서 읽음)

## 5. 모드 (영상 종류)

| 모드 | 스크립트 | ego 위치 | 용도 |
|---|---|---|---|
| **GT** | render_ols_drive `--mode gt` | **실제 로그 그대로**(bag 움직임). LQR X. 파란선=지나온 경로, magenta=미래 로그경로(속도색) | 실측 주행 재현 |
| **OLS** | render_ols_drive `--mode ols` | 로그 위치 + PLUTO 예측 오버레이 | 예측 품질 확인 |
| **NRCLS** | render_closed_loop_post | **매 스텝 추론 결과**(closed-loop). NR agent=시간기준 로그재생 | 실제 주행 시뮬 |
| **rollout** | render_ols_drive `--mode rollout` | t0 계획 1개를 LQR로 끝까지 실행(open-loop). 그리는 선=LQR 실제 주행경로 | 단일 계획 실행 |
| **combo** | render_ols_drive `--mode combo` | GT 주행하다 `--switch-frame`에서 rollout 전환(ego 연속) | 특정 지점부터 계획 실행 |

**불변식(모든 모드 자동검사, `check_frame`)**: ① agent는 항상 위치대로 주입 ② GT 제외 ego는 궤적 위(sep≈0). 위반 시 렌더가 터짐.

## 6. 노브 (환경변수 · 인자)

**환경변수**
| 변수 | 읽는 곳 | 의미 | 예 |
|---|---|---|---|
| `PLUTO_VEHICLE` | apollo_scenario | 차량 제원 선택 | `E100` / `U100` |
| `PLUTO_AGENT_MARGIN` | apollo_scenario | agent box 안전마진(m). **planner엔 부풀림, 화면엔 원본**. 0.15 권장(0.3은 과함) | `0.15` |
| `PLUTO_COLLISION_SCALE` | trajectory_evaluator(post) | ego 충돌박스 폭 배율. 팬텀장애물·교차로 freeze 완화 | `1.0` (필요시 `0.6`) |
| `PLUTO_EGO_WIDTH` / `PLUTO_EGO_LENGTH` | trajectory_evaluator(post) | 실제 차폭/길이 | E100 `1.87`/`4.46`, U100 `1.89`/`4.72` |
| `NRCLS_TRACE` | render_closed_loop_post | 렌더 스킵, 진단 CSV(속도·prog_j·충돌gap·red·kick)만 기록 | `1` |

**인자 (주요)**: `--mode` `--steps`(넉넉히; frame dt≠0.1인 record는 실제시간/0.1만큼 필요, 예 92=520) `--start` `--route-from`(routing 기본) `--ego-pose "x,y,heading[,speed]"`(임의 초기위치) `--switch-frame`(combo) `--map-graph` `--out` `--fps`

## 7. 시나리오 레지스트리

| 시나리오 | 차량 | 맵 | record | 비고 |
|---|---|---|---|---|
| E100BT-25 | E100 | AYG | 00006 | 우회전+신호. routing에 reroute 있음 |
| E100BT-26 | E100 | AYG | 00044 | 끝에서 정체 큐 정지(로그도 정지). TL 검출 0 |
| E100BT-22 | E100 | SEL | 00014 | **미처리** |
| U100BT-680 | U100 | SEL | 00092 | 대형 교차로 우회전. **frame dt=0.13** (steps 넉넉히) |
| U100BT-716 | U100 | SEL | 00047(bag_2) | |
| E100BT-29 | E100 | SEL | 00081(bag_2) | 후반 실제 빨간불 정지 |

## 8. 출력
- `*_gt.mp4`, `*_nrcls.mp4` (1000×1000, 10fps)
- `*.gif` (ffmpeg palettegen, 예 600px 13초=14MB / 8초=1.3MB)
- `*.trace.csv` (NRCLS_TRACE 진단용)

## 9. 알아둘 것 (핵심 결정·한계)

- **route = routing_response(timestamped)**. planning.lane_id는 순간 glitch로 목적지 튀어서 폐기. reroute는 발생시각부터 반영.
- **NRCLS agent = 시간(wall-clock) 기준 재생**. 예전 진행매칭(prog_j)은 freeze·순간이동 유발해 폐기.
- **agent margin은 planner/render 분리** (planner만 부풀림). NR agent가 비반응이라 margin 과하면 ego 멈추고 로그agent가 받음 → 0.15 이내.
- **frozen-robot**: PLUTO는 완전정지→재출발 못함 → kickstart(로그가 움직이면=green, reference 따라 1.5m/s² 가속). 빨간불이면 로그도 정지→kick 안 함(신호 존중).
- **NR 정체 큐 충돌**: 끝에서 ego가 큐에 정지하면 비반응 로그agent가 겹쳐 들어옴 → 충돌로 잡히나 이는 계획 실패가 아니라 NR 재생의 근본 한계.
- **frame dt ≠ 0.1**: record별 perception rate 다름(예 92=0.13). `--steps`를 로그 전체시간/0.1로 넉넉히(안 그러면 뒤가 잘림).
- **하드코딩 경로**(재구축 TODO): 스크립트가 `/home/culee/pluto_onnx`·ONNX 경로 하드코딩 → docker는 동일경로 미러링. `PLUTO_ONNX_ROOT`/`ONNX_MODEL` env로 파라미터화 권장.

## 10. 환경
- **단일 venv**(numpy 1.23.4 / protobuf 3.20.3): 파싱(cyber-record)·추론(onnxruntime)·렌더 전부 한 env. 예전 numpy2 분리·cross-env pickle 꼼수 제거됨.
- **Docker**: `docker/` 참고. CPU 이미지, `bash docker/build.sh`. nuplan-devkit + pluto_onnx 추론 서브셋 + apollo_pb2 + scripts 포함.
