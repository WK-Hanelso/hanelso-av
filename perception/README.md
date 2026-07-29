# perception/ — 세계 상태 추정 도메인 (뼈대)

센서로부터 3D 객체·track을 추정하는 도메인 — 현재 뼈대만 (BEVFusion 예정).

## 구조 (모델-우선 패턴 예비, C-SWM-023)

| 경로 | 역할 |
|---|---|
| `interface.py` | placeholder — 모델 진입 시 `planning/interface.py` 패턴(ABC+registry+`load_model` 동적 로딩)으로 채운다. |
| `models/` | 모델-우선 배치 예비. 모델 추가 = `models/<이름>/` 1개 + `configs/<이름>.py` 1개. |
| `configs/` | 모듈 config 자리. |

산출물은 data_devkit 계약 아티팩트로 나간다 (예: `agent_tracks` provenance
`bevfusion` — root config `data=dict(agents=...)`로 선택, 현재 예약만).
