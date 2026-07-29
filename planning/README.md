# planning/ — 궤적 생성 도메인

C-SWM-023에서 **모델-우선 배치**로 재편: 모델의 모든 코드는 `models/<이름>/` 한 디렉토리
소유, 드라이버는 동적 로딩(`interface.load_model`)으로만 접근. 원본 PLUTO import(`src.*`)는
`models/pluto/paths.py::ensure_pluto_on_path()` 한 곳으로만 vendored `third_party/pluto`를
참조한다 (외부 절대경로 금지).

## 구성

| 경로 | 역할 |
|---|---|
| `interface.py` | Policy/InputBuilder ABC + registry 4종(policy·input_builder·feature adapter·postprocessor) + `load_model(이름)`/`available_models()` 동적 로딩. 드라이버의 유일한 planning 진입점. |
| `models/` | 모델-우선 배치. 규약은 `models/README.md`. 현재 `pluto/`. |
| `map_adapter/` | `apollo_map.py` — `map_graph.json` → nuPlan `AbstractMap` duck-type (nuplan 계열 모델 공유). |
| `configs/pluto.py` | 모듈 config: registry 키 + bundle/native config 경로 참조 + env(`pluto-inf`). |
| `run_inference.py` | 드라이버 (concrete import 0). `python3 planning/run_inference.py configs/e100bt25.py [--device cuda]`. |

## 실행 경로와 env

- **실추론(GPU)**: `run_inference.py --device cuda` — 모델 소유 이미지 `pluto-inf`(`docker/pluto-inf/`).
- **오프라인 검증(sim/CPU)**: `simulation/render_sim.py` — 공용 이미지 `swm-base`.

모델 아티팩트는 bundle(`data/model/pluto_v3/{v3_pluto.ckpt, config.yaml}`) 쌍으로 관리 —
ckpt와 학습 native config는 같은 디렉토리, 우리 config는 경로만 참조(전사 금지).
데이터 입력은 dataloader `REQUIRES` 선언 + `data_devkit.contract.check()` fail-fast
(누락 시 무엇을 돌려야 하는지 안내).
