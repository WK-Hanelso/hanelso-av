# Portfolio Summary

## Problem
모델마다 raw data 형식과 의미를 다시 해석하면 data producer가 downstream model의 window·좌표 변환·tensor layout까지 알아야 한다. ([boundary evidence](../data_devkit/parsers/base.py#L7-L33), [model transform](../planning/models/pluto/dataloader.py#L1003-L1103))

## Decision
Raw facts와 model interpretation을 분리하고, canonical artifact의 경로·필수 키·producer selection을 contract로 합의했다. ([schema](../data_devkit/parsers/schema.py#L7-L159), [contract](../data_devkit/contract.py#L41-L53))

## Implementation
Pluggable parser → canonical tables → 9 registered artifact contracts → fail-fast validation → dataloader-owned transform을 구현했다. 이 중 8개 contract만 required file을 가지며 `prediction`은 예약 상태다. ([registry](../data_devkit/contract.py#L65-L172), [PLUTO preflight](../planning/models/pluto/dataloader.py#L785-L825))

## Evidence
실제 PLUTO closed loop가 같은 경계를 소비해 metrics/video를 만들고, counterfactual engine은 두 intervention과 curvature·lateral-acceleration gate를 제공한다. ([simulation](../simulation/drivers/model_driven.py#L95-L268), [enrichment](../data_devkit/counterfactual/generator.py#L89-L163))

## Result
Data side는 source facts와 validation에, model side는 model-native interpretation에 집중할 수 있는 재사용 경계를 public code로 제시했다. ([data boundary](../data_devkit/contract.py#L41-L53), [model boundary](../planning/models/pluto/dataloader.py#L785-L825))

## Limitations
Provenance는 runtime selection이며 persisted lineage가 아니고, counterfactual CLI와 clip-scoped contract handoff는 partial이다. ([provenance](../data_devkit/contract.py#L175-L191), [handoff](../data_devkit/contract.py#L151-L161))
