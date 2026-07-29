# third_party — vendored 외부 소스

외부 repo의 소스 코드를 프로젝트 내부로 반입(vendor)하는 곳. **"git clone 하나로 재현"** 원칙을 위해, 외부 절대경로 `sys.path` 주입이나 docker 외부 마운트 대신 여기에 포함한다.

## 구조

| 디렉토리 | 내용 |
|---|---|
| `pluto/` | PLUTO 소스(`src/`) — planning `pluto` 모델과 postprocess·sim이 재사용하는 원본 컴포넌트(PlanningModel, PlutoFeature, ScenarioManager, TrajectoryEvaluator, ForwardSimulator, NuplanScenarioRender). NATTEN 순수 torch 대체(`native_nat.py`) 포함(원본 natten과 1e-6 수치 동등 검증). 상세는 `pluto/NOTICE.md` |

## 규약

- **수정 금지** — 반입은 충실 복사. 우리 측 수정본은 해당 도메인 모듈에 별도 파일로 (예: `planning/models/pluto/v3_planning_decoder.py`).
- 반입 시 출처·라이선스를 `<이름>/NOTICE.md`에 고지. 배포/공개 전 원저작 라이선스 포함 확인 필수.
- 라이브러리(자체 remote·버전 존재, 예: nuplan-devkit)는 vendor 대상이 아니라 docker/requirements가 설치한다 — vendor는 "remote 없는 소스 트리"만.
- 접근은 각 모델의 `paths.py`(sys.path 유틸) 경유 (예: `planning/models/pluto/paths.py`).
