# data_devkit/ — 데이터 생산·계약·검증 최상위 모듈

이종 원본(record/map)을 우리 통합 포맷(`work/`)으로 만드는 **파서**와, 그 산출물(아티팩트)의
**계약(존재+스키마) 검증**을 담는다. 구 `common/io`의 승격.

**핵심 원칙**: devkit은 모델을 모른다(모델별 분기 금지). 각 모델의 dataloader가
`REQUIRES`(아티팩트 이름 목록)를 선언하고 `contract.check()`로 fail-fast 검증한다.
`torch` import 금지 — 파싱 경로는 torch 없이 import 가능해야 한다.

## 구성

| 경로 | 역할 |
|---|---|
| `parsers/` | pluggable 파싱 프레임워크 (구 `common/io` 그대로). `base.py`(SourceParser/EgoPoseProvider/MapParser ABC), `registry.py`(문자열 키→클래스), `schema.py`(통합 포맷 dataclass+writer), `config.py`(ParseConfig), `apollo/`(record·map 파서, 키 `"apollo_record"`/`"apollo"`), `pose/`(`"apollo_record"`/`"identity"`). |
| `contract.py` | **아티팩트 registry** — 이름 → `work/<clip>/`(clip-scope) 또는 `work/maps/<map>/`(map-scope) 경로규약 + 필수 파일/필수 키 + provenance 축. `check(clip_id, requires, data_cfg, map_name)` 실패 시 "무엇을 돌려야 하는지" 안내를 담은 `ContractError` (자동 생성 없음). |

## 아티팩트 (1차)

`sample`, `ego_pose`, `ego_dynamics`, `agent_tracks`(=sample_annotation+instance+category),
`route`, `scene_log`(=scene+log), `map_graph`(클립무관), `prediction`(예약, 생산자 없음).
provenance 축은 root config `data=dict(agents="apollo_gt", prediction=None)`이 고른다 —
현재 유효 값은 이 조합뿐(bevfusion/apollo는 예약). 스키마 체크는 "파일 존재+필수 키"
수준이며 물리 검증은 `tools/parser_validation` 소관.

## 데이터 티어 계약

- `raw`: 센서 원본 tier. record/bag, camera images, lidar, calib처럼 재파싱 가능한 원재료를 뜻한다.
- `parsed`: 현재 `work/<clip>/parsed`에 놓이는 통합 JSON 테이블 tier다.
- `derived`: 모델/평가용 파생 산출물 tier다. 예: `map_graph`, prediction, future sensor cache.

E2E·VLM 계열 모델은 `raw` tier를 직접 요구할 수 있다. 그 경우에도 같은 자리에 꽂히며,
camera/lidar/calib를 `REQUIRES`로 선언하는 방식으로 계약을 확장한다. `raw` tier의 실제 구현은
첫 소비자 모델이 들어올 때 추가한다.

센서 기반 E2E 모델의 closed-loop simulation은 센서 재시뮬레이션이 없으면 완전한 의미의 회귀가 아니다.

## 사용 예

```python
from data_devkit import contract
contract.check(
    clip_id="E100BT-25_20260716151711_00006",
    requires=["sample", "ego_pose", "route", "map_graph"],
    data_cfg=dict(agents="apollo_gt", prediction=None),
    map_name="AYG",
)  # 누락 시 ContractError: "route 없음 -> parse_clip.py ... 실행 필요"
```

파싱 드라이버 — 실행은 항상 `docker run`:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace av-base:latest \
  python parse_clip.py configs/e100bt25.py
docker run --rm -v "$PWD":/workspace -w /workspace av-base:latest \
  python parse_map.py --map <base_map.bin> --name AYG
```
