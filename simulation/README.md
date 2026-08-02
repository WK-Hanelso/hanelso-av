# simulation/ — sim 전용 렌더/주행 모듈 (C-SWM-018 → C-SWM-022 추상화)

배포 planner(`planning/`)와 분리된 **시뮬레이션 전용** 계층. planner 출력(모델 forward + `PlutoPostProcessor` best)을 데이터로만 소비한다.

sim은 **오프라인 검증 경로**라 CPU(`swm-base` 이미지)로 돈다 — GPU 실추론 경로는
`planning/run_inference.py --device cuda`(모델 소유 env `pluto-inf`)로 분리.

## 구성 (ABC + registry)

| 경로 | 역할 |
|---|---|
| `renderers/` | `Renderer` ABC + registry — `"matplotlib"`(자체 BEV, ego-following crop) / `"nuplan"`(원본 pluto `NuplanScenarioRender`, 공식 스타일). 드라이버는 frame dict 계약(`renderers/base.py`)만 채운다. |
| `drivers/` | `EgoDriver` ABC + registry — `"log_replay"`(open_loop: ego=로그 GT) / `"model_driven"`(closed_loop: ego=모델 운전, 원본 `ForwardSimulator` 전파). |
| `configs/` | 모듈 config: `closed_loop_nuplan.py`, `open_loop_matplotlib.py`(`_base_` 상속) — mode/renderer/steps/stride/fps 등. |
| `sim_utils.py` | 지오메트리·로그 agent fetch·ego dynamics·mp4 합성 공용 헬퍼. |
| `render_sim.py` | **조립 전용** 드라이버 — concrete 모델 import 0 (`planning.interface.load_model` 동적 로딩, C-SWM-023). root config 하나 + CLI override. |

## 실행

```bash
# root config의 simulation 이름(예: closed_loop_nuplan)이 기본값, CLI로 override
python3 simulation/render_sim.py configs/e100bt25.py
python3 simulation/render_sim.py configs/e100bt25.py --mode open_loop --renderer matplotlib --steps 100
python3 simulation/render_sim.py configs/e100bt25.py --mode closed_loop --renderer nuplan --steps 3
```

실행은 시스템 `python3`(torch1.12 + natten + nuplan + shapely). 산출물(클립-우선):

- `work/<clip>/sim/<mode>[_nuplan]/frame_%05d.png` — BEV 프레임
- `work/<clip>/sim/<mode>[_nuplan].mp4` — ffmpeg(libx264) 합성
- `work/<clip>/sim/<mode>[_nuplan]_metrics.json` — 프레임/스텝별 지표 + closed-loop 요약(발산·이탈·충돌)

## closed-loop 스텝 구조 (`drivers/model_driven.py`)

1. **feature**: `ApolloPlutoDataloader.build_frame(clip, log_idx, sim_ego=…)` — ego row는
   주입된 sim ego-history(21스텝)만 사용, agent/static은 로그 프레임 `log_idx`에서 fetch.
   ego 로그 미래는 어떤 경로로도 참조하지 않는다(leakage 없음).
2. **plan**: `policy.infer` → `PlutoPostProcessor.run` → best 궤적(global).
3. **전파**: `[현재 포즈 + best 80스텝]`(81,3)을 원본 pluto `ForwardSimulator(dt=0.1,
   num_frames=1)`(BatchLQR + kinematic bicycle)에 넣어 1스텝 전진, state array에서
   `EgoState` 복원.
4. **agent 재생(시간축, issue #1)**: 스텝의 로그 프레임은 경과 sim 시간으로 고정 —
   `log_idx = min(start + step, n_samples−1)`. sim ego가 어떻게 움직이든 주변 agent는
   실제 로그의 시간 흐름 그대로 등장/이동한다. feature(build_frame t0)·충돌 체크·렌더가
   전부 같은 `log_idx`를 사용(한 프레임 = 한 로그 시각). ego-GT 발산 비교만
   `time_idx = start + step + 1`(전파 후 ego와 같은 시각의 로그 ego).
5. 렌더 + 지표 기록(스텝 변위, 시간/progress 기준 로그 대비 발산, drivable 여부,
   shapely oriented-box 충돌, emergency brake). progress-정렬 발산 지표
   (`divergence_progress_aligned_m`)는 ego 궤적 비교용 metric으로만 유지 —
   agent 선택에는 관여하지 않는다.

## 어댑터 sim 주입 계약 (`planning/models/pluto/dataloader.py`)

- `prepare_clip(parsed_dir, map_graph, config)` — 프레임 불변 요소(파싱 테이블, route.json
  이벤트 이력, ApolloMap) 1회 로드/캐시. route t0 검사는 sim 경로에서 생략 —
  대신 각 프레임이 자기 로그 시각으로 라우팅 이벤트를 재선택한다(issue #2).
- route 이벤트 선택(issue #2): 각 프레임은 `route.json all_sequences` 중
  `timestamp_ns ≤ 프레임 로그 시각`인 최신 이벤트를 사용(이전엔 t0 시점 선택본을 전
  프레임에 재사용). reroute 시각을 지나면 reference line·route_lane_dict·on_route가
  새 경로로 전환되며, 이벤트 단위 lane→roadblock 해석은 `route_event_cache`
  (이벤트 timestamp 키)로 캐시 — 같은 이벤트 구간은 캐시 히트, 전환 시 1회만 재계산.
- `build_frame(clip, t0_index, sim_ego=None)` — `sim_ego=None`이면 open-loop(로그 ego),
  dict면 closed-loop. 정규화/pack은 기존 `build()`와 동일 코드 경로.
- `sim_ego` 키: `position(21,2)` `heading(21,)` `velocity_global(21,2)` `valid_mask(21,)`
  `current_state(7,)=[x,y,heading,speed,accel,steering_angle,yaw_rate]` `ego_state(EgoState)`.
  전부 global/UTM, rear-axle 포즈, oldest-first(index -1 = 현재).

## 한계

- traffic light 상태 없음(전부 UNKNOWN), agent는 로그 재생(반응 없음, non-reactive).
- 렌더러 mission_goal(`destination_xy`)은 클립 단위 고정 — 프레임별 이벤트의
  destination이 다른 클립이면 미반영(현재 검증 클립들은 전 이벤트 destination 동일).
