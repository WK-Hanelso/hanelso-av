# configs/ — ROOT config (파이프라인 run 단위 최고봉)

"무엇을 쓸지"(이름)만 적는다. 이름 해석·병합은 `common/config.py::load_config`가 한다:
`modules.planning="pluto"` → `planning/configs/pluto.py`, `calibration="e100"` →
`calibration/configs/e100.py`, `simulation="closed_loop_nuplan"` →
`simulation/configs/closed_loop_nuplan.py`. `_base_` 상속(dict 재귀 병합) 지원.

## 형식 (configs/e100bt25.py)

```python
config = dict(
    clip_id="E100BT-25_20260716151711_00006",
    record="data/bag/E100BT-25/20260716151711.record.00006",
    source="apollo_record",   # data_devkit/parsers parser registry key
    pose="apollo_record",     # data_devkit/parsers pose provider registry key
    map_name="AYG",
    map_path="work/maps/AYG/map_graph.json",
    modules=dict(planning="pluto", perception=None, localization=None),
    calibration="e100",
    simulation="closed_loop_nuplan",
    data=dict(agents="apollo_gt", prediction=None),
)
```

## data 섹션 (provenance 축, C-SWM-023)

같은 아티팩트에 복수 생산자가 올 수 있어 출처를 root에서 고른다
(`data_devkit/contract.py::DATA_AXES`). 현재 유효 값: `agents="apollo_gt"`(perception GT),
`prediction=None`. `bevfusion`/`apollo`는 예약 — 지정하면 contract가 명확한 에러로 거부.
모델 dataloader는 `REQUIRES` 아티팩트를 이 축으로 해석해 `contract.check()` fail-fast.

## calibration / feature vehicle

- `calibration="e100"` 같은 root 항목은 실차 물리 제원을 고른다.
- feature ego-shape pin은 root가 아니라 모델 config
  (`planning/configs/pluto.py::feature_vehicle`)가 소유한다.

## 사용 예

실행은 항상 `docker run`(환경 = 이미지). 원샷은 `scripts/sim.sh`가 docker를 자동 처리한다.

```bash
docker run --rm -v "$PWD":/workspace -w /workspace swm-base:latest \
  python parse_clip.py configs/e100bt25.py                  # 파싱
docker run --rm -v "$PWD":/workspace -w /workspace swm-base:latest \
  python planning/run_inference.py configs/e100bt25.py      # 추론
docker run --rm -v "$PWD":/workspace -w /workspace swm-base:latest \
  python simulation/render_sim.py configs/e100bt25.py --steps 3
```
(bag 등 원본 접근 시 데이터 디스크 마운트 추가: `-v /mnt/hdd_storage:/mnt/hdd_storage` — [../docker/README.md](../docker/README.md))

`modules.planning` 이름은 `planning/models/<이름>/` 동적 로딩에도 쓰인다 — 없는 이름이면
Available 목록을 포함한 에러.
