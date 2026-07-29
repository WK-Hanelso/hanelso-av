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

1. **feature**: `ApolloPlutoFeatureAdapter.build_frame(clip, prog_j, sim_ego=…)` — ego row는
   주입된 sim ego-history(21스텝)만 사용, agent/static은 로그 프레임 `prog_j`에서 fetch.
   ego 로그 미래는 어떤 경로로도 참조하지 않는다(leakage 없음).
2. **plan**: `policy.infer` → `PlutoPostProcessor.run` → best 궤적(global).
3. **전파**: `[현재 포즈 + best 80스텝]`(81,3)을 원본 pluto `ForwardSimulator(dt=0.1,
   num_frames=1)`(BatchLQR + kinematic bicycle)에 넣어 1스텝 전진, state array에서
   `EgoState` 복원.
4. **agent 재fetch(progress-정렬 하이브리드, PIPELINE.md §4.4)**:
   `matched = prog_j + argmin ||log_ego[prog_j:prog_j+80] − sim_ego||`,
   `prog_j = max(matched, prog_j+1)` — ego가 앞서면 fast-forward, 멈춰도 최소 1프레임 전진.
5. 렌더 + 지표 기록(스텝 변위, 시간/progress 기준 로그 대비 발산, drivable 여부,
   shapely oriented-box 충돌, emergency brake).

## 어댑터 sim 주입 계약 (`planning/models/pluto/dataloader.py`)

- `prepare_clip(parsed_dir, map_graph, config)` — 프레임 불변 요소(파싱 테이블, route.json,
  ApolloMap) 1회 로드/캐시. route t0 검사는 클립 단위 재사용을 위해 sim 경로에서 생략.
- `build_frame(clip, t0_index, sim_ego=None)` — `sim_ego=None`이면 open-loop(로그 ego),
  dict면 closed-loop. 정규화/pack은 기존 `build()`와 동일 코드 경로.
- `sim_ego` 키: `position(21,2)` `heading(21,)` `velocity_global(21,2)` `valid_mask(21,)`
  `current_state(7,)=[x,y,heading,speed,accel,steering_angle,yaw_rate]` `ego_state(EgoState)`.
  전부 global/UTM, rear-axle 포즈, oldest-first(index -1 = 현재).

## 한계

- route는 클립 단위 route.json 하나를 전 프레임에 재사용 — 중간 reroute 이벤트는 미반영.
- traffic light 상태 없음(전부 UNKNOWN), agent는 로그 재생(반응 없음, non-reactive).
