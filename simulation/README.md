# simulation/ — sim 전용 렌더/주행 모듈 (C-SWM-018)

배포 planner(`common/`)와 분리된 **시뮬레이션 전용** 계층. planner 출력(모델 forward + `PlutoPostProcessor` best)을 데이터로만 소비한다.

## render_sim.py

```bash
# open-loop: ego = bag(GT), 매 stride 프레임을 t0로 forward + 예측 오버레이
python3 simulation/render_sim.py --mode open_loop \
  --parsed-dir work/<clip>/parsed --map-path work/maps/<MAP>/map_graph.json --map-name <MAP> \
  --start-index 20 --steps 278 --stride 2

# closed-loop: ego = 모델 운전 (feature→model→postprocess best→ForwardSimulator 1스텝)
python3 simulation/render_sim.py --mode closed_loop \
  --parsed-dir work/<clip>/parsed --map-path work/maps/<MAP>/map_graph.json --map-name <MAP> \
  --start-index 140 --steps 200
```

실행은 시스템 `python3`(torch1.12 + natten + nuplan + shapely). 산출물:

- `work/sim/<clip>/<mode>/frame_%05d.png` — BEV 프레임 (global UTM, ego-following crop)
- `work/sim/<clip>/<mode>.mp4` — ffmpeg(libx264) 합성
- `work/sim/<clip>/<mode>_metrics.json` — 프레임/스텝별 지표 + closed-loop 요약(발산·이탈·충돌)

## closed-loop 스텝 구조

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

## 어댑터 sim 주입 계약 (`common/input/pluto_feature_adapter.py`)

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
