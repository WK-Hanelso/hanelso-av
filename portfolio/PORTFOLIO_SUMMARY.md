# 모델 비종속 데이터 구조와 계약 기반 ML 파이프라인

## One-line

원천 데이터의 의미와 검증은 공통 계층에 두고, sampling·좌표 변환·tensorization은 모델별 consumer가 담당하도록 책임을 분리했다.

## Problem

모델별 데이터 준비 로직이 raw-data layer까지 침투해 모델 변경마다 같은 분석을 반복하는 문제였다. Temporal window, 좌표계, object selection, tensor layout 같은 model knowledge가 producer로 역류하면 data engineer와 model engineer의 변경 범위가 함께 커진다. 또한 소비 직전까지 파일·필수 키·producer 선택 오류가 드러나지 않으면 디버깅이 늦어진다.

## Decision

- Raw fact와 model-specific interpretation을 분리했다.
- `Raw source → Parser → Canonical schema → Artifact contract / Validation`은 data side가, sampling·좌표 변환·window·tensorization은 model consumer가 소유하도록 경계를 정했다.
- Consumer가 `REQUIRES`를 선언하고 준비 단계에서 공통 contract를 fail-fast로 실행하도록 했다.

## What I Built

- Source parser interface와 canonical dataclass/JSON representation
- 경로·필수 파일·필수 키·producer selection을 정의하는 artifact registry와 validator
- Contract preflight 뒤에서 sampling·좌표 변환·feature assembly를 수행하는 PLUTO consumer
- 두 counterfactual intervention, physical gate, vocabulary nearest-distance report를 갖춘 dataset enrichment producer

## Evidence

- 등록 계약 9개 = 구현된 non-empty 계약 8개 + 예약 `prediction` 1개 (`data_devkit/contract.py:65-172`)
- Artifact contract fail-fast 진단 테스트 3개 (`tests/test_contract.py:12-46`)
- 문서화된 OpenScene mini run: 64 logs, 2,848 anchors, counterfactual samples 7,608개 (`data_devkit/counterfactual/README.md:26-29`)
- Physical quality gate pass 88.5% — curvature/lateral-acceleration gate 통과율이며 accuracy나 dataset quality score가 아님 (`data_devkit/counterfactual/generator.py:44-58,89-119`)
- Vocabulary nearest distance p50 0.30 m, p90 0.47 m — threshold가 아닌 분포 측정값 (`data_devkit/counterfactual/generator.py:61-86,178-193`)

## Failure / Debugging

- `prediction`은 registry에 이름은 있지만 `required_files=()`이고 producer가 없다. 등록 수를 구현 수로 오해하지 않도록 결과를 **8 implemented + 1 reserved**로 분리했다.
- Counterfactual standalone CLI는 `<out>/counterfactual.json`을 쓰고 clip contract는 `labels/counterfactual.json`을 요구한다. 두 구현을 직접 잇는 handoff가 없으므로 integration 상태는 partial이다.

## Result

Contract가 실제 PLUTO consumer 앞에서 실행되도록 구현했다. Missing file, required key, unsupported provenance를 consumption 전에 진단하고, model-specific sampling과 tensorization은 consumer 코드에 남겼다.

## Scope / Limitation

- Counterfactual 수치는 저장소 README의 documented validation run이며 원시 `report.json`이 추적되지 않아 독립 재현 가능하다고 주장하지 않는다.
- Contract validation은 파일 존재와 required key 수준이다. 물리적 타당성 전체를 검증하는 범용 validator가 아니다.
- Producer selection은 runtime config validation이며 persisted lineage DAG가 아니다.
- Counterfactual generator와 clip-scoped contract의 직접 handoff, prediction producer, raw sensor E2E contract, SparseDriveV2 consumer는 완료로 표현하지 않는다.
- 외부 데이터·map·model bundle이 필요한 PLUTO closed-loop E2E 실행은 이번 evidence 작성 범위에 포함하지 않았다.

## Portfolio Assets

- `assets/01_data_model_boundary.svg` / `.png` — data side와 model side의 책임 경계, 특히 tensorization 소유권을 한눈에 보여준다.
- `assets/02_contract_evidence.svg` / `.png` — 등록 9개를 구현 8개와 예약 1개로 분리하고 fail-fast 테스트 3개를 연결한다.
- `assets/03_counterfactual_quality.svg` / `.png` — 문서화된 run 규모, physical gate 통과율, vocabulary distance 분포를 주장 경계와 함께 보여준다.
