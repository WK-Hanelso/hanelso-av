# Notion Asset Captions

## 01_data_model_boundary.svg

Notion 위치: Featured Work 2 첫 화면, One-line 바로 아래

Caption: 원천 사실과 검증은 data side가, sampling·좌표 변환·window·tensorization은 model consumer가 소유한다. 공통 contract는 PLUTO consumer가 입력을 만들기 전에 실행된다.

Claim boundary: 저장소가 증명하는 범위는 parser·canonical schema·artifact contract와 PLUTO adapter의 구현 경계다. 조직 전체 표준 채택이나 다른 모델까지의 적용 완료를 뜻하지 않는다.

## 01_data_model_boundary.png

Notion 위치: SVG 업로드가 제한될 때 첫 화면 대체 이미지

Caption: 원천 사실과 model-specific interpretation을 분리한 ownership architecture의 PNG 버전.

Claim boundary: Data side는 model-specific tensorization을 소유하지 않는다는 코드 구조의 주장이다.

## 02_contract_evidence.svg

Notion 위치: Decision 다음, contract 구현 근거 섹션

Caption: Registry에는 9개 contract가 있지만 구현된 non-empty contract는 8개다. `prediction` 1개는 reserved이며, 세 data-free 테스트가 missing file, schema mismatch, unsupported provenance를 fail-fast로 확인한다.

Claim boundary: “9개 구현”이 아니다. 테스트 수 3은 `tests/test_contract.py`의 함수 수이며 이번 작업에서 새로 얻은 pass 결과가 아니다.

## 02_contract_evidence.png

Notion 위치: SVG 업로드가 제한될 때 contract 근거 대체 이미지

Caption: 9 registered = 8 implemented + 1 reserved, 그리고 fail-fast tests 3개를 분리한 PNG 버전.

Claim boundary: Reserved contract에는 required file이나 producer가 없다.

## 03_counterfactual_quality.svg

Notion 위치: Evidence 섹션의 counterfactual dataset 설명 옆

Caption: 문서화된 OpenScene mini validation run은 64 logs·2,848 anchors에서 7,608 counterfactual samples를 만들었고, curvature/lateral-acceleration physical gate 통과율은 88.5%였다. Vocabulary nearest distance는 p50 0.30 m, p90 0.47 m였다.

Claim boundary: 88.5%는 model accuracy나 dataset quality score가 아니다. 원시 `report.json`이 저장소에 없어 수치는 documented run record이며 독립 재현 결과로 표현하지 않는다.

## 03_counterfactual_quality.png

Notion 위치: SVG 업로드가 제한될 때 counterfactual 근거 대체 이미지

Caption: Counterfactual run 규모와 physical gate, vocabulary distance를 함께 보여주는 PNG 버전.

Claim boundary: Vocabulary distance는 분포 측정값이며 구현에 pass/fail threshold가 없다.
