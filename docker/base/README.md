# hanelso_swm Environment Image

## 개요

이 이미지는 `hanelso_swm`의 실행 환경만 제공한다. 프로젝트 소스는 이미지를 빌드할 때 넣지 않고 `git clone`으로 확보한 뒤 `docker run`에서 마운트한다. PLUTO 소스는 프로젝트 내부(`planning/models/pluto/src/`)에 vendored 되어 있어 **git clone에 포함**된다(외부 마운트 불요).

목표 파이프라인은 다음과 같다.

1. bag + HD map 준비
2. `parse_map.py`로 `map_graph.json` 생성
3. `parse_clip.py`로 record를 통합 포맷으로 변환
4. `build_input.py`로 PLUTO 입력 텐서 생성
5. `run_inference.py`로 추론 실행
6. 이후 cls 시뮬레이션, 렌더, mp4 인코딩

이 이미지가 책임지는 범위는 2-6단계를 실행하기 위한 시스템 패키지와 파이썬 의존성이다. 코드는 컨테이너에 굽지 않는다.

## 설계 원리

| 원리 | 이유 |
| --- | --- |
| 코드를 이미지에 굽지 않음 | 깨끗한 머신에서 `git clone -> docker build -> docker run`만으로 같은 환경을 재현하기 위해서다. |
| CPU 전용 | 현재 요구사항은 `torch==1.12.0+cpu`와 CPU `onnxruntime`이다. CUDA, `onnxruntime-gpu`, `natten`에 의존하지 않는다. |
| `build.sh` 없음 | 로컬 파일을 임시 컨텍스트에 모으는 방식은 재현성을 깨뜨린다. |
| `nuplan-devkit`는 Dockerfile에서 설치 | third-party 라이브러리이므로 Dockerfile이 버전 고정으로 설치해야 한다. |
| PLUTO는 프로젝트 내부 vendored | `planning/models/pluto/src/`에 반입돼 git clone에 포함된다. 외부 repo 의존 없이 repo 내부 소스를 직접 사용한다. |

## 이미지 구성

| 항목 | 포함 여부 | 비고 |
| --- | --- | --- |
| `torch==1.12.0+cpu` | 포함 | Dockerfile에서 별도 설치 |
| `onnxruntime` CPU | 포함 | `docker/base/requirements.txt` |
| `hydra-core`, `omegaconf`, `numpy`, `scipy`, `numba` | 포함 | 추론/시뮬레이션 기반 |
| `matplotlib`, `opencv-python`, `pillow`, `shapely`, `pandas` | 포함 | 렌더/시각화 런타임 |
| `geopandas`, `fiona`, `rasterio`, `pyproj`, `rtree` | 포함 | nuPlan 지오 런타임 |
| `nuplan-devkit@e924167` | 포함 | Dockerfile에서 `pip install --no-deps` |
| 프로젝트 소스 | 제외 | `docker run`에서 마운트 |
| PLUTO 소스 | 제외(이미지엔 안 구움) | `planning/models/pluto/src/`에 vendored → git clone에 포함 |
| `data/`, `work/`, 모델 파일 | 제외 | `docker run`에서 마운트 |

## 빌드

빌드 컨텍스트는 `docker/base/`만 사용한다.

```bash
docker build -t swm-base:latest -f docker/base/Dockerfile docker/
```

예상 사항:

- 첫 빌드는 네트워크 속도에 따라 수 분 이상 걸릴 수 있다.
- `torch==1.12.0+cpu`와 `nuplan-devkit`를 원격에서 가져온다.
- `geopandas` 계열이 휠 대신 시스템 GDAL을 요구하는 환경이면 Dockerfile의 `libgdal-dev gdal-bin` 주석을 해제해 다시 빌드한다.
- Dockerfile이 `COPY`하는 파일은 `requirements.txt` 하나뿐이다.

## 실행

### 마운트 규약

- 리포지토리 루트는 `/workspace`로 마운트한다.
- 실행 중 생성되는 결과는 `/workspace/work` 아래에 쌓인다.
- 입력 데이터와 모델은 `/workspace/data`에서 읽는다.
- PLUTO 소스는 `planning/models/pluto/src/`에 vendored 되어 있어 별도 마운트가 필요 없다.

예시:

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  swm-base:latest \
  bash
```

`data/`와 `work/`를 외부 볼륨으로 분리하고 싶다면 다음처럼 추가 마운트한다.

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/work:/workspace/work" \
  swm-base:latest \
  bash
```

### 1. 맵 파싱

맵 원본 바이너리는 리포지토리에 포함되지 않으므로 실제 `base_map.bin` 경로를 넘겨야 한다.

```bash
MAP_BIN=/absolute/path/to/base_map.bin

docker run --rm \
  -v "$PWD:/workspace" \
  swm-base:latest \
  python parse_map.py \
    --map "$MAP_BIN" \
    --name AYG \
    --source apollo \
    --out-root work/maps
```

출력은 `work/maps/AYG/map_graph.json`에 생성된다.

### 2. 클립 파싱

리포지토리에 포함된 config 예시는 `configs/e100bt25.py`, `configs/e100bt22.py`다.

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  swm-base:latest \
  python parse_clip.py configs/e100bt25.py
```

출력은 `work/E100BT-25_20260716151711_00006/parsed/` 아래에 생성된다.

### 3. 입력 텐서 생성

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  swm-base:latest \
  python build_input.py configs/e100bt25.py
```

출력은 `work/E100BT-25_20260716151711_00006/input/` 아래에 생성된다.

### 4. 추론 실행 (CPU)

드라이버는 root config 하나만 받는다 (C-SWM-022 계층 config).

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  swm-base:latest \
  python planning/run_inference.py configs/e100bt25.py
```

출력은 `work/E100BT-25_20260716151711_00006/inference/` 아래에 생성된다.
GPU 실추론은 모델 소유 이미지 `pluto-inf`(`docker/pluto-inf/`)에서
`--device cuda`로 돌린다 — swm-base는 CPU 공용 파이프라인 전용.

비교 기준으로 저장된 리포트는 다음 경로에 있다.

- `work/E100BT-25_20260716151711_00006/inference/infer_report.txt`
- `work/E100BT-22_20260716031426_00014/inference/infer_report.txt`

### 5. 시뮬레이션 + 렌더 (mp4)

`simulation/render_sim.py`가 시뮬레이션 루프와 렌더링을 함께 수행한다.

- `--mode open_loop` : ego=로그(GT), 매 프레임 모델 예측 오버레이
- `--mode closed_loop` : ego=모델이 운전(원본 ForwardSimulator로 전파), agent progress-정렬 재fetch
- `--renderer matplotlib` : 자체 BEV / `--renderer nuplan` : 원본 PLUTO 공식 렌더(NuplanScenarioRender)

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  swm-base:latest \
  python simulation/render_sim.py configs/e100bt25.py \
    --mode closed_loop --renderer nuplan --start-index 160 --steps 120
```

출력: `work/<clip>/sim/<mode>[_nuplan]/frame_%05d.png` + `<mode>[_nuplan].mp4` (컨테이너 내 ffmpeg 인코딩). matplotlib/nuplan × open/closed 네 조합 모두 컨테이너에서 동작 확인됨.

## 트러블슈팅

### GDAL / geopandas 설치 실패

`fiona`, `rasterio`, `geopandas` 설치 중 GDAL 관련 에러가 나면 Dockerfile의 다음 주석을 활성화한 뒤 다시 빌드한다.

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ... \
    libgdal-dev \
    gdal-bin \
    ...
```

### 헬스체크 단계 실패

Dockerfile의 마지막 `RUN python -c ...`는 설치형 의존성만 검사한다. 여기서 실패하면 다음을 확인한다.

- 해당 패키지가 `docker/base/requirements.txt`에 있는지
- 원격 설치가 네트워크 문제 없이 완료됐는지
- `nuplan-devkit` 설치 단계가 성공했는지

`common`(프로젝트 소스)나 `planning/models/pluto/src`(vendored) import 실패는 여기서 검사하지 않는다. 그 코드는 이미지에 없고 git clone(마운트)으로 확보되기 때문이다.

### CPU 전용 여부 확인

컨테이너 안에서 다음 명령으로 CPU 빌드 여부를 확인할 수 있다.

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda_available=", torch.cuda.is_available())
PY
```

기대값은 `torch==1.12.0+cpu`, `cuda_available=False`다.

## 상태 (컨테이너에서 검증됨)

- **전 파이프라인 컨테이너 실행 확인**: `parse_map → parse_clip → build_input → run_inference(--postprocess) → simulation/render_sim(open/closed-loop, matplotlib/nuplan)` 이 이미지 안에서 동작하며, run_inference/postprocess 출력이 호스트와 일치하고 closed-loop `nuplan` mp4까지 생성된다.
- **native_nat ≡ natten 검증**: vendored `native_nat`(planning/models/pluto/src, 순수 torch NAT, natten 미설치)와 원본 `natten` forward 출력이 `max_abs_diff ~1e-6`(allclose) — CPU/native_nat 경로가 수치 동등하므로 GPU/natten 없이도 결과가 신뢰 가능.
- ONNX 추론 백엔드는 환경(onnxruntime)만 준비돼 있다. 현재 파이프라인은 pth backend를 쓰며, onnx CLI 연결은 필요 시 별도.
- 이 이미지는 **환경 레이어**다. 소스(git clone)와 데이터(마운트) 없이 단독으로는 파이프라인을 수행하지 않는다.
