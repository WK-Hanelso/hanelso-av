# planning/ — 궤적 생성 도메인

C-SWM-022에서 `common/`의 planning 자산을 도메인으로 이동한 구조. 모든 원본 PLUTO
import(`src.*`)는 `pluto_paths.ensure_pluto_on_path()` 한 곳으로만 vendored
`third_party/pluto`를 참조한다 (외부 절대경로 금지).

## 구성

| 경로 | 역할 |
|---|---|
| `policy/` | `Policy` ABC + registry. `pluto_torch`(PlanningModel strict 로드, `device` cpu/cuda 지원), `v3_planning_decoder`, `pluto_postprocess`(원본 TrajectoryEvaluator+EmergencyBrake). |
| `input/` | `InputBuilder` ABC + registry(`"pluto"` = plain numpy feed) 및 feature adapter registry(`"pluto_feature"` = `ApolloPlutoFeatureAdapter`, 원본 ScenarioManager reference line). |
| `map_adapter/` | `apollo_map.py` — `map_graph.json` → nuPlan `AbstractMap` duck-type. |
| `configs/pluto.py` | 모듈 config: policy/bundle/native config 경로 참조 + env(`pluto-inf`). |
| `pluto_paths.py` | vendored pluto sys.path 주입 일원화. |
| `run_inference.py` | 드라이버. `python planning/run_inference.py configs/e100bt25.py [--device cuda]`. |

## 실행 경로와 env

- **실추론(GPU)**: `planning/run_inference.py --device cuda` — 모델 소유 이미지 `pluto-inf`
  (`docker/pluto-inf/`). 출력 텐서는 cpu로 회수되므로 후처리(numpy)는 device 무관.
- **오프라인 검증(sim/CPU)**: `simulation/render_sim.py` — 공용 이미지 `swm-base`.

모델 아티팩트는 bundle(`data/model/pluto_v3/{v3_pluto.ckpt, config.yaml}`) 쌍으로
관리한다: ckpt와 그 학습 native config는 항상 같은 디렉토리에 있고, 우리 config는
경로만 참조한다(전사 금지).
