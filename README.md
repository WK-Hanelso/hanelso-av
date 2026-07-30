# dopamine-av — Autonomous Driving SW Monorepo

자율주행에 필요한 SW 스택(**localization · perception · data labeling · planning · simulation · tools**)을 하나의 모노레포에서 개발한다. 각 도메인이 고유 목표를 갖고, 도메인 간 데이터가 흐르며 서로의 입력·검증이 되는 구조다.

**핵심 원칙 — 원본 기준(source-of-truth).** 모든 것은 불변의 원본 파일(주행 raw, HD맵, nuPlan/nuScenes, 이미지, 차량 제원)에서 출발한다. 원본은 절대 수정하지 않고, 파생물은 전부 원본에서 계산으로 뽑는다. *파싱 = 사실 기록, 정규화·가공 = 소비 시점 결정.*

현재 동작하는 플래그십은 **Apollo 7.0 주행로그(`.record`) + HD맵 → PLUTO 플래너 → closed-loop 시뮬레이션 영상** 파이프라인이다.

---

## 도메인 지도

| 도메인 | 한 줄 목표 | 현재 자산 |
|---|---|---|
| **localization** | ego pose·시간 기준 확립 (좌표계의 토대) | Apollo MSF pose 소비 → 향후 GNSS+IMU/LiDAR 융합 |
| **perception** | 센서로부터 세계 상태(3D 객체·track) 추정 | BEVFusion 3D detection |
| **data labeling** | perception 학습용 고정밀 true-value 라벨 오프라인 생성 | offboard auto-labeling (설계) |
| **planning** | 궤적 생성 | PLUTO, planTF (ONNX/TRT 이식) |
| **simulation** | planning을 실주행 로그로 재현·검증 | Apollo bag → PLUTO NR closed-loop |
| **tools** | 시각화·큐레이션·검수 등 횡단 지원 | parser_validation(BEV viz), FiftyOne |
| **common** | 도메인 관통 공유물 | 파싱·좌표계(UTM52 / ego 정규화)·스키마 |

```
원본(raw) ─▶ [common 파싱] ─▶ 통합 포맷(clip 단위)
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
├── common/             # 도메인 관통 공유(파싱·좌표·스키마)
├── data_devkit/        # 원본→통합 포맷 데이터 계약
├── calibration/        # 차량 제원 config (E100 / U100 …)
├── planning/           # 궤적 생성 도메인 (PLUTO) — planning/README.md
├── simulation/         # closed-loop 재현·렌더 — simulation/README.md
├── localization/       # 도메인 스켈레톤
├── perception/         # 도메인 스켈레톤
├── tools/              # 시각화·검수 (parser_validation 등)
├── configs/            # 공용 설정
├── third_party/        # 외부 의존
└── docker/             # 이미지 (base = swm-base, pluto-inf) — docker/README.md
```

> `data/`(읽기전용 입력 — 물리 디스크 심볼릭)와 `work/`(파생물·캐시), `.venv-*`(파이썬 환경), 그리고 운영용 로컬 문서는 `.gitignore`로 git에서 제외된다. 리포에는 **코드 + 각 도메인 README**만 담긴다.

---

## 시뮬레이션 파이프라인 (Apollo → PLUTO)

### 무엇을 하나
Apollo 차량 로그와 HD맵을 입력으로, nuPlan 학습된 vectorized planner **PLUTO**를 태워 **non-reactive(NR) closed-loop** 주행 영상을 재현한다. PLUTO는 raw 센서가 아니라 벡터화된 상태(agent 궤적·ego·맵 polygon·reference line)를 먹으므로, 이 프로젝트의 본질은 **Apollo 산출물을 nuPlan 인터페이스 규격에 정합**시키는 어댑터다.

### `run_sim.py` 단계
```
record(.record) ─┬─▶ ① inspect_record   토픽/맵/차량 판정 (tools.parser_validation)
                 ├─▶ ② parse_clip.py    record → parsed 아티팩트
                 │        (sample · ego_pose · ego_dynamics · agent_tracks ·
                 │         scene_log · route · map_graph)
                 └─▶ ③ render_sim.py     closed_loop_nuplan 시뮬 → mp4 + metrics
```
- ①② 파싱은 `.venv-apollo`(cyber_record + Apollo `pb2`), ③ 렌더/추론은 별도 nuPlan/torch env에서 돈다 (라이브러리·numpy 버전 충돌 회피).
- env 경계를 넘는 지오메트리는 python list로 직렬화(버전 독립).

### 빠른 시작
```bash
# 0) 맵 그래프 1회 생성 (차량/맵 대응: E100→AYG, U100→SEL)
.venv-apollo/bin/python parse_map.py --map <base_map.bin> --name <map_name>

# 1) 원샷 실행 — 파싱부터 렌더까지
python3 run_sim.py <record 경로> --mode closed_loop --renderer nuplan --steps N
#   --mode      closed_loop | open_loop
#   --renderer  nuplan | matplotlib
#   --device    cpu | cuda      --force  (parsed 계약 통과해도 재파싱)
```
> `--steps`는 넉넉히 준다. record별 perception rate가 달라(frame dt ≠ 0.1) 부족하면 뒤가 잘린다 (실제시간 / 0.1 만큼).

### 모델 직접 추론 (planning 도메인)
```bash
# 실추론(GPU): 모델 소유 이미지 pluto-inf
python3 planning/run_inference.py configs/<scenario>.py --device cuda
# 오프라인 sim(CPU): 공용 이미지 swm-base → simulation/render_sim.py
```
자세한 배치·registry·env 규약은 [planning/README.md](planning/README.md), 이미지 빌드는 [docker/README.md](docker/README.md) 참고.

---

## 환경

- **파이썬 base 패키지**: `pyproject.toml` (`dopamine-av`, requires-python ≥ 3.9, numpy / protobuf 3.20.3 / cyber_record).
- **2개 env**
  - `.venv-apollo` — `.record`/맵 **파싱** (cyber_record, Apollo `pb2`, protobuf 3.20.3).
  - 렌더/추론 env — nuplan-devkit, torch, onnxruntime, numpy 1.23.4, shapely.
- **Docker**: `docker/base`(= `swm-base`, CPU 시뮬 공용) / `docker/pluto-inf`(모델 GPU 추론). `docker/README.md` 참고.

---

## 데이터 규약

- **`data/` = 입력 한곳(읽기전용)**: 코드는 `data/raw/…`, `data/nuplan/…` 짧은 경로로 접근. 물리 디스크는 심볼릭으로 흡수 — 디스크가 옮겨져도 심볼릭만 교체하면 코드 무변경. ( data는 별도 필요 )
- **`work/` = 파생물 한곳, clip 단위**: `work/<clip>/`(parsed / sim / …). 스테이지별 캐싱으로 뒷단 수정 시 앞단 재실행을 막는다. `work/maps/`에 맵 그래프 캐시.
- **원본 직접수정 금지**, 실험은 복사본에서.

---

## 설계 원칙

1. **재구현 금지** — 컨트롤러(LQR)·planner·렌더러는 레포의 진짜 nuPlan·PLUTO 컴포넌트를 duck-typing 어댑터로 재사용. 우리가 새로 짜는 건 경계의 어댑터(Apollo→nuPlan 규격 변환)뿐.
2. **버전 독립** — env 경계를 넘는 데이터는 python list로 직렬화.
3. **상수 일치** — feature 상수(HIST_STEPS / RADIUS / MAX_AGENTS / ego shape)는 학습 분포와 정확히 맞춘다.
4. **재현 우선(NR)** — 자연스러움보다 로그 재현. NR의 구조적 한계(발산·유령트랙)는 입력 정제(progress-정렬 fetch, 트랙 필터)로 완화.
5. **명세 선행** — 새 포맷·필드는 구조·타입·의미를 먼저 문서화한 뒤 구현.
