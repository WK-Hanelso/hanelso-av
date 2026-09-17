# ML Data Architecture Case Study

*Separating Data Producers from Model Consumers with Contracts*

## Problem

Model마다 raw source를 다시 해석하고 전용 dataset을 만들면 temporal window, 좌표 변환, object selection, tensor layout 같은 model knowledge가 data producer로 역류한다. 이 프로젝트는 source parsing과 canonical rows를 `data_devkit`에 두고, PLUTO-specific sampling과 feature assembly를 model dataloader에 둠으로써 이 coupling을 분리한다. ([parser contract](../data_devkit/parsers/base.py#L7-L33), [canonical rows](../data_devkit/parsers/schema.py#L7-L130), [PLUTO sampling](../planning/models/pluto/dataloader.py#L418-L468), [PLUTO assembly](../planning/models/pluto/dataloader.py#L1003-L1103))

핵심 질문은 “data side가 downstream model 내부를 몰라도 factual representation과 validation을 제공하고, model side가 자신의 해석을 소유할 수 있는가?”였다. 실제 경계는 dataloader class의 `REQUIRES` 선언과 consumption 전 contract preflight로 구현되어 있다. ([generic contract](../planning/interface.py#L66-L109), [PLUTO declaration/preflight](../planning/models/pluto/dataloader.py#L785-L825))

## Design Goal

1. Raw source의 의미는 source-specific parser가 canonical dataclass rows로 옮기고 JSON writer가 통합 table로 직렬화한다. ([schema](../data_devkit/parsers/schema.py#L7-L159))
2. Data layer는 model import 없이 parser registry와 artifact contract를 제공한다. ([parser registry](../data_devkit/parsers/registry.py#L5-L40), [contract](../data_devkit/contract.py#L41-L172))
3. Model consumer는 필요한 artifact와 temporal/model-native transformation을 직접 소유한다. ([`REQUIRES`](../planning/models/pluto/dataloader.py#L785-L798), [model transform](../planning/models/pluto/dataloader.py#L1003-L1103))

## Ownership Boundary

| Data side | Model side |
|---|---|
| Raw source parsing과 parser selection ([ABC](../data_devkit/parsers/base.py#L7-L33), [registry](../data_devkit/parsers/registry.py#L5-L40)) | Model input artifact selection ([`REQUIRES`](../planning/models/pluto/dataloader.py#L785-L798)) |
| Canonical dataclass schema와 serialization ([schema/writers](../data_devkit/parsers/schema.py#L7-L159)) | History sampling과 window construction ([sampling](../planning/models/pluto/dataloader.py#L418-L468)) |
| Artifact path와 required keys ([`ArtifactSpec`](../data_devkit/contract.py#L41-L53), [registry](../data_devkit/contract.py#L65-L172)) | Coordinate/model feature transformation ([feature assembly](../planning/models/pluto/dataloader.py#L1003-L1103)) |
| Config-level producer selection과 validation ([axes](../data_devkit/contract.py#L29-L34), [validation](../data_devkit/contract.py#L175-L191)) | Policy inference와 model-native postprocess ([simulation assembly](../simulation/render_sim.py#L85-L137)) |
| File existence·required-key diagnostics ([key check](../data_devkit/contract.py#L194-L208), [contract check](../data_devkit/contract.py#L211-L303)) | Closed-loop state propagation과 evaluation ([driver](../simulation/drivers/model_driven.py#L95-L200)) |
| Log/scene identifiers를 통한 제한적 source traceability ([schema](../data_devkit/parsers/schema.py#L7-L23), [record metadata](../data_devkit/parsers/apollo/record_parser.py#L429-L448)) | Training/inference-specific configuration ([PLUTO config](../planning/configs/pluto.py#L5-L15)) |

이 표의 provenance는 artifact에 lineage DAG를 기록한다는 뜻이 아니다. 현재 구현은 root config의 producer value를 runtime에 선택·검증하고 결과 report에 반환하는 수준이다. ([axis validation](../data_devkit/contract.py#L175-L191), [report](../data_devkit/contract.py#L233-L303))

## Data Contract: declaration before consumption

`ArtifactSpec`은 `name`, `scope`, `required_files`, `provenances`, `axis`, `required_keys`, `produce_hint`로 logical artifact를 정의한다. Registry에는 9개 contract가 있으며 8개는 required file을 가진 non-empty spec이고 `prediction`은 producer가 없는 예약 contract다. ([spec](../data_devkit/contract.py#L41-L53), [registry](../data_devkit/contract.py#L65-L172))

Model-specific dataloader는 전역 분기 대신 class-level `REQUIRES`로 소비 목록을 선언한다. PLUTO adapter는 clip preparation 전에 같은 public `check()`를 호출한다. ([interface](../planning/interface.py#L66-L109), [PLUTO requirements](../planning/models/pluto/dataloader.py#L785-L825))

`check()`는 모든 요구 artifact를 순회해 missing file과 missing required key를 모아서 `ContractError`로 실패한다. Missing/schema 오류에는 producer command hint가 붙고, invalid provenance에는 현재 지원값이 표시된다. ([key validation](../data_devkit/contract.py#L194-L208), [aggregation/diagnostics](../data_devkit/contract.py#L239-L303), [contract tests](../tests/test_contract.py#L12-L46))

현재 유효한 producer selection은 `agents="apollo_gt"`와 `prediction=None`이다. `bevfusion`과 non-null prediction producer는 주석과 slot만 예약되었으며 구현된 producer로 취급하지 않는다. ([axes](../data_devkit/contract.py#L29-L34), [agent spec](../data_devkit/contract.py#L100-L123), [prediction reservation](../data_devkit/contract.py#L162-L170))

## From contract to an actual model consumer

```text
Apollo driving log + HD Map
→ Pluggable Parser
→ Canonical Artifacts
→ Contract Validation
→ PLUTO-specific Dataloader
→ PLUTO Policy
→ NR Closed-loop Simulation
→ Metrics JSON + Video
```

이 경로는 parser 구현, adapter preflight, simulation assembly와 model-driven driver로 연결된다. Contract가 독립 library로 끝나지 않고 실제 model consumer 앞의 executable boundary로 사용된다. ([record parser](../data_devkit/parsers/apollo/record_parser.py#L230-L484), [map parser](../data_devkit/parsers/apollo/map_parser.py#L84-L247), [preflight](../planning/models/pluto/dataloader.py#L803-L825), [assembly](../simulation/render_sim.py#L85-L137))

### Evaluation data integrity

Deployed closed-loop inference adapter는 log future ego 대신 injected simulated ego history를 input으로 만들고 future ego tensor를 채우지 않는다. 이는 closed-loop input path에 한정한 evaluation leakage control이며, future target을 쓰는 vendored training builder 전체에 대한 주장이 아니다. ([closed-loop frame path](../planning/models/pluto/dataloader.py#L939-L975), [future tensor handling](../planning/models/pluto/dataloader.py#L1034-L1048), [reference lines](../planning/models/pluto/dataloader.py#L1380-L1425), [training builder](../planning/models/pluto/src/feature_builders/pluto_feature_builder.py#L99-L154))

### Temporal data consistency and outputs

Simulation decision, agent selection, collision check와 renderer는 같은 decision-time `log_idx`를 사용하고, post-step ego divergence는 다음 original-log `time_idx`에 맞춘다. 이로써 feature와 evaluation이 서로 다른 log time을 암묵적으로 소비하지 않게 한다. ([decision index](../simulation/drivers/model_driven.py#L95-L127), [metric/render indices](../simulation/drivers/model_driven.py#L168-L235))

Driver는 divergence, drivable-area 여부, collision과 emergency-brake를 step/summary metrics에 기록하고 MP4를 생성하며, renderer entry point가 metrics JSON을 저장한다. ([step metrics](../simulation/drivers/model_driven.py#L168-L200), [summary/video](../simulation/drivers/model_driven.py#L245-L268), [JSON output](../simulation/render_sim.py#L135-L145))

명시적 consumer 실행 경로는 `scripts/sim.sh <record> --model pluto`다. 현재 model directory가 여러 개이므로 `--model pluto`를 생략한 자동 선택을 보장하지 않는다. ([launcher](../scripts/sim.sh#L24-L50), [handoff](../scripts/sim.sh#L101-L107), [model detection](../run_sim.py#L153-L182))

## Dataset Enrichment: Counterfactual Supervision

자연 궤적의 condition/goal 신호가 학습에서 무시될 수 있다는 문제에 대해, generator는 controlled intervention으로 alternative goal과 corresponding target trajectory를 함께 만든다. 구현된 intervention은 quintic lateral offset 기반 `lane_transplant`와 같은 log의 donor trajectory를 쓰는 `time_transplant`다. ([lane intervention](../data_devkit/counterfactual/generator.py#L129-L145), [time intervention](../data_devkit/counterfactual/generator.py#L148-L163))

```text
Natural trajectory
→ Controlled intervention
→ Alternative goal + corresponding target trajectory
→ Physical quality gates + vocabulary distance measurement
→ Training supervision record
```

`max_curvature`와 `max_lat_accel` 두 physical threshold가 `valid`를 결정한다. Invalid entry도 삭제되지 않고 `valid=false`와 measured quality values를 보존하므로 downstream에서 거절 규모와 원인을 다시 분석할 수 있다. 자동 reason string은 생성하지 않고 `flags`는 caller-provided다. ([quality calculation](../data_devkit/counterfactual/generator.py#L44-L58), [entry decision/schema](../data_devkit/counterfactual/generator.py#L89-L119), [preservation test](../tests/test_counterfactual_schema.py#L59-L65))

Vocabulary coverage는 pass/fail gate가 아니라 optional nearest-distance metric과 p50/p90 distribution이다. 구현에는 vocabulary threshold가 없다. ([metric](../data_devkit/counterfactual/generator.py#L61-L86), [report](../data_devkit/counterfactual/generator.py#L178-L193))

### Documented validation record

저장소에는 OpenScene mini **64 logs / 2,848 anchors / 7,608 counterfactual samples / 88.5% valid / nearest distance p50 0.30 m·p90 0.47 m** 실행 기록이 있다. 88.5%는 generated samples 중 두 physical gate를 통과한 비율이며 model performance가 아니다. 원시 `report.json`은 추적되지 않아 이 문서는 해당 기록을 독립 재검증 결과로 표현하지 않는다. ([record](../data_devkit/counterfactual/README.md#L26-L29), [valid logic](../data_devkit/counterfactual/generator.py#L89-L119), [report writer](../data_devkit/counterfactual/generate_openscene.py#L101-L105))

Token identity는 입력으로부터 결정되고 donor selection은 CLI seed를 사용하지만 payload `meta.created`는 runtime timestamp다. 따라서 deterministic sample identity와 seeded selection까지만 주장하며 whole-payload byte determinism은 주장하지 않는다. ([token](../data_devkit/counterfactual/generator.py#L34-L36), [timestamp](../data_devkit/counterfactual/generator.py#L166-L175), [seed](../data_devkit/counterfactual/generate_openscene.py#L49-L61), [token test](../tests/test_counterfactual_schema.py#L41-L46))

Standalone CLI는 `<out>/counterfactual.json`을 쓰고 registered contract는 clip root의 `labels/counterfactual.json`을 요구한다. 양쪽 구현은 존재하지만 producer-to-contract handoff는 아직 partial이다. ([CLI output](../data_devkit/counterfactual/generate_openscene.py#L101-L105), [contract path](../data_devkit/contract.py#L151-L161))

## Architecture as a collaboration boundary

서로 다른 역할의 전문성 차이를 모든 사람에게 강제로 복제하는 대신 합의 지점을 artifact name, schema semantics, producer selection과 validation rule로 제한했다. Data engineer는 parser/canonical facts를, model engineer는 `REQUIRES`와 native transform을 소유하도록 코드 경계가 나뉜다. ([artifact definition](../data_devkit/contract.py#L41-L53), [consumer declaration](../planning/models/pluto/dataloader.py#L785-L825), [model transformation](../planning/models/pluto/dataloader.py#L1003-L1103))

이는 실제 model/data 협업에서 반복되는 coupling을 기준으로 만든 architecture proposal과 public implementation이지, 특정 조직 전체가 채택한 운영 표준이라는 주장이 아니다. 구현 근거는 parser → contract → PLUTO consumer 경로이고 조직 채택 여부는 repository가 증명하지 않는다. ([parser boundary](../data_devkit/parsers/base.py#L7-L33), [contract boundary](../data_devkit/contract.py#L211-L303), [consumer boundary](../planning/models/pluto/dataloader.py#L785-L825))

## Transferable Pattern

같은 producer-consumer boundary는 여러 model이 공유하는 domain data를 reusable training artifacts로 바꿔야 하는 ML system에 적용할 수 있다. 이 저장소가 직접 증명하는 범위는 public autonomous-driving parser, contract, PLUTO consumer와 counterfactual generator이며 다른 domain 적용 자체는 일반화 가능한 설계 해석이다. ([parser](../data_devkit/parsers/base.py#L7-L33), [contract](../data_devkit/contract.py#L41-L53), [consumer](../planning/models/pluto/dataloader.py#L785-L825), [enrichment](../data_devkit/counterfactual/generator.py#L89-L163))

## Limits and next work

| Boundary | Current status |
|---|---|
| Producer provenance | Runtime config selection/validation까지 구현; persisted artifact lineage는 없음. ([contract](../data_devkit/contract.py#L175-L191), [return report](../data_devkit/contract.py#L297-L303)) |
| Counterfactual integration | Generator와 contract가 각각 구현; clip-scoped handoff는 partial. ([generator output](../data_devkit/counterfactual/generate_openscene.py#L101-L105), [contract](../data_devkit/contract.py#L151-L161)) |
| Counterfactual reproducibility | Token과 seeded donor selection은 deterministic; timestamp 때문에 full payload는 byte-deterministic하지 않음. ([token](../data_devkit/counterfactual/generator.py#L34-L36), [payload](../data_devkit/counterfactual/generator.py#L166-L175)) |
| Closed-loop verification | Runtime code path는 구현; data-free automated E2E test는 없음. ([driver dependencies](../simulation/drivers/model_driven.py#L38-L88), [simulation assembly](../simulation/render_sim.py#L85-L137)) |
| Future producers/models | `bevfusion`, prediction, raw sensor tier와 SparseDriveV2는 reserved/planned이며 implemented로 분류하지 않음. ([axes](../data_devkit/contract.py#L29-L34), [prediction](../data_devkit/contract.py#L162-L170), [data tiers](../data_devkit/README.md#L26-L34), [model slot](../planning/models/sparsedrive_v2/__init__.py#L1-L10)) |
