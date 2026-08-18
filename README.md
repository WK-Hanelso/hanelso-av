# dopamine-av — 계약으로 조립하는 자율주행 SW 스택

자율주행 SW 스택 — 센서 원본 → 데이터 → 인지 → 예측 → 판단(planning) → 시뮬레이션/평가 — 을 하나의 모노레포에서 개발한다. **이 프로젝트의 본체는 특정 모델이나 시뮬레이터가 아니라, 전 스택을 모듈로 나누고 경계마다 계약(ABC + registry + 데이터 계약)을 두어 갈아끼울 수 있게 조립하는 구조 그 자체다.** 모델·파서·렌더러는 그 구조에 꽂히는 교체 가능한 구성요소다.

**핵심 원칙 — 원본 기준(source-of-truth).** 모든 것은 불변의 원본 파일(주행 raw, HD맵, nuPlan/nuScenes, 이미지, 차량 제원)에서 출발한다. 원본은 절대 수정하지 않고, 파생물은 전부 원본에서 계산으로 뽑는다. *파싱 = 사실 기록, 정규화·가공 = 소비 시점 결정.*

---

## 계약 지도 (이 repo를 읽는 법)

스택의 각 경계는 계약으로 고정되고, 구현은 이름으로 선택된다. 어떤 슬롯은 이미 구현이 꽂혀 있고(활성), 어떤 슬롯은 계약만 먼저 존재한다(예약/빈 슬롯) — **계약이 구현보다 먼저**다.

| 경계 | 계약 | 현재 꽂힌 것 | 상태 |
|---|---|---|---|
| 원본 파싱 | `data_devkit/parsers` — `SourceParser` ABC + registry | `"apollo_record"` (record), `"apollo"` (HD맵) | 활성 |
| 측위(ego pose) | `EgoPoseProvider` ABC + registry | `"apollo_record"`, `"identity"` | 활성 (SLAM/odometry 슬롯 예약) |
| 데이터 아티팩트 | `data_devkit/contract.py` — 이름→경로·필수키 + provenance 축 | `sample`·`ego_pose`·`ego_dynamics`·`agent_tracks`·`scene_log`·`route`·`map_graph` | 활성 (`prediction` 예약) |
| 인지 | `agent_tracks` provenance 축 (`agents=`) | `"apollo_gt"` (로그 GT) | 활성 (`"bevfusion"` 예약) |
| 판단(planning) | `planning/interface.py` — policy/dataloader/postprocessor registry + `load_model(이름)` | `"pluto"` | 활성 (모델 추가 = 디렉토리 1개 + config 1개, 기존 파일 수정 0) |
| 지도 | `planning/map_adapter` — nuPlan `AbstractMap` duck-type | `ApolloMap` (`map_graph.json`) | 활성 |
| 차량 제원 | `calibration/` — 실차 물리 계약 | `e100`, `u100` | 활성 |
| 시뮬 주행 | `simulation/drivers` — `EgoDriver` ABC + registry | `"log_replay"`(open), `"model_driven"`(closed) | 활성 |
| 렌더 | `simulation/renderers` — `Renderer` ABC + registry | `"matplotlib"`, `"nuplan"` | 활성 |
| 실행 환경 | 모듈 config `env` 필드 = docker 이미지 (이미지는 모델/기능이 소유) | `av-base`(CPU 공용), `pluto-inf`(GPU 추론) | 활성 |
| localization / perception 도메인 | `interface.py` 자리 (planning 패턴 예비) | — | 빈 슬롯 |

**조립은 ROOT config가 한다.** `configs/*.py`는 "무엇을 쓸지" 이름만 적는 조립 명세서이고, `common/config.py::load_config`가 이름을 `<domain>/configs/<이름>.py`로 해석·병합한다:

```python
# configs/e100bt25.py — 이름만 고르면 전 스택이 조립된다
config = dict(
    record="data/bag/E100BT-25/20260716151711.record.00006",
    source="apollo_record", pose="apollo_record",        # 파싱·측위 선택
    map_name="AYG", map_path="work/maps/AYG/map_graph.json",
    data=dict(agents="apollo_gt", prediction=None),      # 아티팩트 출처(provenance) 선택
    modules=dict(planning="pluto", perception=None, localization=None),  # 도메인 슬롯
    calibration="e100",                                  # 실차 제원 선택
    simulation="closed_loop_nuplan",                     # sim 모드/렌더러 선택
)
```

의존 규칙: 드라이버는 concrete를 import하지 않는다(registry 문자열로만 획득). `simulation`은 `planning.interface`·`planning.nuplan_common`만 의존하고, 도메인 코드는 `planning/models/<이름>/src` 내부를 직접 import하지 않는다.

---

## 도메인 지도

| 도메인 | 한 줄 목표 | 현재 슬롯 상태 |
|---|---|---|
| **localization** | ego pose·시간 기준 확립 (좌표계의 토대) | Apollo MSF pose 소비(파서 provider). 자체 모델 슬롯은 빈 슬롯 |
| **perception** | 센서로부터 세계 상태(3D 객체·track) 추정 | 로그 GT(`apollo_gt`) 소비. `bevfusion` provenance 예약 |
| **data labeling** | perception 학습용 true-value 라벨 오프라인 생성 | 설계 단계 |
| **planning** | 궤적 생성 | **PLUTO 활성** (registry 온보딩 완료) |
| **simulation** | planning을 실주행 로그로 재현·검증 | **Apollo bag → PLUTO NR closed-loop 활성** |
| **tools** | 시각화·큐레이션·검수 등 횡단 지원 | parser_validation(BEV viz) 활성 |
| **common / data_devkit** | 도메인 관통 공유물 — config 조합, 파싱·계약·스키마 | 활성 |

```
원본(raw) ─▶ [data_devkit 파싱] ─▶ 통합 포맷(clip 단위, work/<clip>/parsed)
                                 ├─▶ localization ─┐
                                 ├─▶ perception ───┤
                                 ├─▶ data labeling ┤ (라벨)
                                 └─────────────────┴─▶ planning ─▶ simulation
                            tools = 전 단계 횡단 지원
```

---

## 리포지토리 구조

```
dopamine-av/
├── run_sim.py          # bag → sim mp4 원샷 오케스트레이터 (진입점)
├── parse_clip.py       # record → 통합 clip 아티팩트 파싱
├── parse_map.py        # base_map.bin → map_graph
├── pyproject.toml      # 공용 base 패키지 정의
├── common/             # config 조합 로더 등 도메인 관통 공유
├── data_devkit/        # 파싱 프레임워크 + 아티팩트 계약 — data_devkit/README.md
├── calibration/        # 차량 제원 config (E100 / U100 …)
├── planning/           # 판단 도메인 (PLUTO 활성) — planning/README.md
├── simulation/         # closed-loop 재현·렌더 — simulation/README.md
├── localization/       # 도메인 슬롯 (빈)
├── perception/         # 도메인 슬롯 (빈)
├── tools/              # 시각화·검수 (parser_validation 등)
├── configs/            # ROOT config = 조립 명세서 — configs/README.md
├── third_party/        # vendored 외부 소스 자리 (현재 비어 있음)
└── docker/             # 실행 환경 이미지 (av-base, pluto-inf) — docker/README.md
```

> `data/`(읽기전용 입력 — 물리 디스크 심볼릭)와 `work/`(파생물·캐시), `.venv-*`, 운영용 로컬 문서는 `.gitignore`로 git에서 제외된다. 리포에는 **코드 + 각 모듈 README**만 담긴다.

---

## 첫 관통 사례: Apollo bag → PLUTO closed-loop 시뮬레이션

위 조립 구조가 실물로 동작함을 증명한 첫 end-to-end 스레드. Apollo 7.0 주행로그(`.record`)와 HD맵을 입력으로, nuPlan 학습된 vectorized planner **PLUTO**를 태워 **non-reactive(NR) closed-loop** 주행 영상을 재현한다. PLUTO는 raw 센서가 아니라 벡터화된 상태(agent 궤적·ego·맵 polygon·reference line)를 먹으므로, 이 스레드의 실무는 **Apollo 산출물을 nuPlan 인터페이스 규격에 정합**시키는 어댑터다.

### `run_sim.py` 단계
```
record(.record) ─┬─▶ ① inspect_record   토픽/맵/차량 판정 (tools.parser_validation)
                 ├─▶ ② parse_clip.py    record → parsed 아티팩트
                 │        (sample · ego_pose · ego_dynamics · agent_tracks ·
                 │         scene_log · route · map_graph)
                 └─▶ ③ render_sim.py     closed_loop_nuplan 시뮬 → mp4 + metrics
```
- docker 이미지(av-base/pluto-inf) 하나에 파싱·추론·렌더 의존성이 모두 있어, 세 스테이지가 컨테이너 **단일 환경**에서 돈다 (host 파이썬 env 불필요 — 실행은 항상 `docker run`).

### 빠른 시작 — `scripts/sim.sh` (docker 한 줄, 권장)

**docker만 있으면 된다.** 파이썬 env 설치·`docker run` 타이핑이 필요 없다 — 런처가 이미지 확인(없으면 자동 빌드)·마운트·실행을 다 처리한다.

```bash
# CPU (기본)
scripts/sim.sh data/bag/E100BT-25/20260716151711.record.00006

# GPU
scripts/sim.sh data/bag/E100BT-25/20260716151711.record.00006 --gpu --steps 100

# 옵션: --gpu  --steps N  --mode closed_loop|open_loop  --verbose  -h
```
- **전제**: `docker` 설치 + bag이 있는 데이터 디스크(`/mnt/hdd_storage`) 마운트. 이미지는 없으면 런처가 **자동 빌드**(처음 한 번만, 수 분~십수 분).
- **산출물**: `work/<clip>/sim/<mode>[_nuplan].mp4` (+ metrics json). 렌더러가 `nuplan`(closed_loop 기본값)이면 `_nuplan` suffix가 붙는다 — 예: `closed_loop_nuplan.mp4`.
- `--steps`는 넉넉히 준다. record별 perception rate가 달라(frame dt ≠ 0.1) 부족하면 뒤가 잘린다.

> 런처 없이 직접(`docker run …` 또는 host env)은 [docker/README.md](docker/README.md) 참고.

### 모델 직접 추론 (planning 도메인)
실행은 항상 `docker run` — 실추론(GPU)은 모델 소유 이미지 `pluto-inf`, 오프라인 sim(CPU)은 공용 이미지 `av-base`.

```bash
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace pluto-inf:latest \
  python planning/run_inference.py configs/<scenario>.py --device cuda
docker run --rm -v "$PWD":/workspace -w /workspace av-base:latest \
  python simulation/render_sim.py configs/<scenario>.py
```
자세한 배치·registry·env 규약은 [planning/README.md](planning/README.md), 이미지 빌드는 [docker/README.md](docker/README.md) 참고.

---

## 환경

- **파이썬 base 패키지**: `pyproject.toml` (`dopamine-av`, requires-python ≥ 3.9, numpy / protobuf 3.20.3 / cyber_record).
- **단일 docker 환경이 기본 경로.** 이미지(av-base/pluto-inf) 하나에 파싱(cyber_record, Apollo `pb2`)·추론(torch, nuplan-devkit)·렌더 의존성이 모두 들어 있어 전 스테이지가 한 인터프리터로 돈다. 스테이지별로 다른 인터프리터가 필요한 특수한 경우만 `RUN_SIM_PY_*` env로 오버라이드한다(`run_sim.py` 상단 참고).
- **Docker**: `docker/base`(= `av-base`, CPU 시뮬 공용) / `docker/pluto-inf`(모델 GPU 추론). `docker/README.md` 참고.

---

## 데이터 규약

- **`data/` = 입력 한곳(읽기전용)**: 코드는 `data/raw/…`, `data/nuplan/…` 짧은 경로로 접근. 물리 디스크는 심볼릭으로 흡수 — 디스크가 옮겨져도 심볼릭만 교체하면 코드 무변경. ( data는 별도 필요 )
- **`work/` = 파생물 한곳, clip 단위**: `work/<clip>/`(parsed / sim / …). 스테이지별 캐싱으로 뒷단 수정 시 앞단 재실행을 막는다. `work/maps/`에 맵 그래프 캐시.
- **원본 직접수정 금지**, 실험은 복사본에서.

---

## 설계 원칙

1. **계약이 구현보다 먼저** — 경계는 ABC + registry + 데이터 계약으로 고정하고, 구현은 이름으로 선택한다. 새 구성요소는 계약에 와서 꽂힌다(기존 파일 수정 0). 새 포맷·필드는 구조·타입·의미를 먼저 문서화한 뒤 구현한다(명세 선행).
2. **재구현 금지** — 컨트롤러(LQR)·planner·렌더러는 레포의 진짜 nuPlan·PLUTO 컴포넌트를 duck-typing 어댑터로 재사용. 우리가 새로 짜는 건 경계의 어댑터(Apollo→nuPlan 규격 변환)뿐.
3. **상수 일치** — feature 상수(HIST_STEPS / RADIUS / MAX_AGENTS / ego shape)는 학습 분포와 정확히 맞춘다. 근거는 모델 번들의 native config(SoT)이며 전사하지 않는다.
4. **재현 우선(NR)** — 자연스러움보다 로그 재현. agent는 로그 시간축 그대로 재생하고, NR의 구조적 한계(유령트랙 등)는 입력 정제로 완화한다.
5. **버전 독립** — env 경계를 넘는 데이터는 python list로 직렬화.
