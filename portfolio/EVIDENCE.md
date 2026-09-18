# Evidence Registry

| Claim | Value | Source file/log | Reproducible | Public safe | Note |
|---|---|---|---|---|---|
| C01 — 데이터/모델 책임 경계 | Raw fact와 validation은 공통 계층, sampling·좌표 변환·window·tensorization은 model consumer가 소유 | `data_devkit/README.md:5-15`; `data_devkit/docs/ml_data_architecture_case_study.md:5-25`; `planning/models/pluto/dataloader.py:418-468,1003-1103` | yes (code inspection) | yes | Data side가 model-specific tensorization을 소유하지 않는다는 설계·구현 범위의 주장이다. |
| C02 — 등록된 artifact contract | 9 | `data_devkit/contract.py:65-172` | yes (registry inspection) | yes | `ArtifactSpec(...)` 등록 항목을 직접 셌다. |
| C03 — 구현된 non-empty contract | 8 | `data_devkit/contract.py:68-161` | yes (registry inspection) | yes | `required_files`가 비어 있지 않은 계약 수다. “9개 구현”이 아니다. |
| C04 — reserved contract | 1 (`prediction`) | `data_devkit/contract.py:162-170` | yes (registry inspection) | yes | required file과 producer가 없는 예약 슬롯이다. |
| C05 — artifact contract fail-fast tests | 3 | `tests/test_contract.py:12-46` | yes (test source inspection) | yes | Missing file+producer hint, missing key, unsupported provenance의 세 테스트 함수다. 테스트 실행 결과 수가 아니다. |
| C06 — PLUTO 앞 contract preflight | `REQUIRES` 선언 후 consumer 준비 전에 `contract.check()` 실행 | `planning/interface.py:66-109`; `planning/models/pluto/dataloader.py:785-825,899-911` | yes (code inspection) | yes | 실제 consumer 앞에 연결된 runtime code path이며, 외부 bundle을 포함한 E2E 실행을 이번 작업에서 재현했다는 주장은 아니다. |
| C07 — validation logs | 64 logs | `data_devkit/counterfactual/README.md:26-29` | no (raw report not tracked) | yes | 문서화된 OpenScene mini validation run. |
| C08 — validation anchors | 2,848 anchors | `data_devkit/counterfactual/README.md:26-29` | no (raw report not tracked) | yes | 문서화된 실행 기록. |
| C09 — counterfactual samples | 7,608 samples | `data_devkit/counterfactual/README.md:26-29` | no (raw report not tracked) | yes | 문서화된 실행 기록. |
| C10 — physical quality gate pass | 88.5% | `data_devkit/counterfactual/README.md:26-29`; gate: `data_devkit/counterfactual/generator.py:44-58,89-119` | no (raw report not tracked) | yes | 곡률과 횡가속 gate 통과율만 의미한다. Accuracy, dataset quality score, training success rate가 아니다. |
| C11 — vocabulary nearest distance p50 | 0.30 m | `data_devkit/counterfactual/README.md:26-29`; metric/report: `data_devkit/counterfactual/generator.py:61-86,178-193` | no (raw report not tracked) | yes | 분포 측정값이며 pass/fail threshold가 없다. |
| C12 — vocabulary nearest distance p90 | 0.47 m | `data_devkit/counterfactual/README.md:26-29`; metric/report: `data_devkit/counterfactual/generator.py:61-86,178-193` | no (raw report not tracked) | yes | 분포 측정값이며 pass/fail threshold가 없다. |
| C13 — counterfactual contract handoff | Partial | contract: `data_devkit/contract.py:151-161`; CLI output: `data_devkit/counterfactual/generate_openscene.py:101-105` | yes (code inspection) | yes | Standalone CLI output과 clip-scoped contract 경로를 잇는 직접 handoff가 없다. |

## Counting method

- Registered contracts: `data_devkit/contract.py`의 `ARTIFACTS` comprehension 안 `ArtifactSpec(...)` 9개를 직접 확인했다.
- Implemented non-empty contracts: 위 9개 중 `required_files=()`인 `prediction`을 제외해 8개로 확인했다.
- Fail-fast tests: `tests/test_contract.py`의 `def test_...` 함수 3개를 직접 확인했다.
- 저장소 문서의 “9 registered = 8 implemented + 1 reserved”, “3 fail-fast tests”와 코드 확인값 사이 차이는 없었다.

## Missing requested sources

지시에서 먼저 읽도록 지정한 루트 `OVERVIEW.md`, `PIPELINE.md`, `CHARTER.md`, `docs/`는 이 checkout에 없다. 관련 설계 근거는 추적 중인 `data_devkit/docs/` 두 문서에서 확인했다. 없는 파일의 내용을 추정하지 않았다.
