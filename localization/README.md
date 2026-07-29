# localization/ — ego pose·시간 기준 도메인 (뼈대)

ego pose와 시간 기준을 확립하는 도메인 — 현재 뼈대만
(Apollo MSF pose 소비는 `data_devkit/parsers/pose/`).

## 구조 (모델-우선 패턴 예비, C-SWM-023)

| 경로 | 역할 |
|---|---|
| `interface.py` | placeholder — 모델 진입 시 `planning/interface.py` 패턴(ABC+registry+`load_model` 동적 로딩)으로 채운다. |
| `models/` | 모델-우선 배치 예비. 모델 추가 = `models/<이름>/` 1개 + `configs/<이름>.py` 1개. |
| `configs/` | 모듈 config 자리. |
