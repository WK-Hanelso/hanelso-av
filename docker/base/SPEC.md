# hanelso-av Docker 환경 명세 (SPEC)

> **핵심 순서**: `git clone`(=src 전부 확보) → `docker build`(=의존성·환경만 구성) → `docker run`(=clone된 src 마운트해 실행).
> **Dockerfile의 역할 = 환경(패키지·라이브러리) 구성 그 하나.** src를 fetch/COPY하지 않는다. build.sh 없음.

---

## 0. 역할 분담 (반드시 이 경계를 지킴)

| 단계 | 주체 | 책임 | 산출 |
|---|---|---|---|
| **git clone** | git repo | **모든 src 확보** (hanelso-av 코드 + `planning/models/pluto/src` vendored 포함) | 워킹트리 |
| **docker build** | **Dockerfile** | **의존성 package·library 설치 + 환경 변수/시스템 라이브러리 구성** | 이미지 |
| **docker run** | 실행자 | clone된 src를 **마운트**해서 환경 위에서 실행 | parsed·infer·mp4 |

- **Dockerfile은 src를 모른다.** hanelso-av 코드(`planning/models/pluto/src` vendored 포함)는 clone이 가져오고 run 시 마운트된다. Dockerfile은 그 코드가 **의존하는 외부 패키지·라이브러리·시스템 lib**만 깐다.
- **build.sh 폐기.** 로컬 파일을 컨텍스트에 모으는 staging 방식은 "clone+build로 재현" 원칙에 위배 → 제거.
- **재현성**: 이 호스트에 뭐가 있든 무관. 깨끗한 머신에서 `git clone` + `docker build` 하면 동일 환경이 생겨야 한다.

---

## 1. Dockerfile이 구성하는 환경 (설치 대상)

### 1-A. 시스템 패키지 (apt)
`ffmpeg`, `libgl1-mesa-glx`, `libglib2.0-0`, `libgomp1`, `libsm6`, `libxext6`, `libxrender1`, `fonts-dejavu-core`, `git`, `build-essential`
- ffmpeg: render→mp4/gif 인코딩. libgl/glib/sm6/xext6/xrender1: opencv+matplotlib 런타임. fonts-dejavu-core: render 텍스트. libgomp1: scipy/numba/nuplan OpenMP. git: nuplan-devkit를 git에서 설치하는 데 필요. build-essential: 일부 소스빌드.
- (GDAL 폴백) geopandas/fiona/rasterio가 시스템 GDAL 요구 시 `libgdal-dev gdal-bin` 추가. 대개 manylinux 휠 번들 → 불요.

### 1-B. 파이썬 의존성 (pip)
목표 전구간(parse·input·추론[pth/onnx]·cls 시뮬·render→mp4)이 **코드만 얹으면 돌 수 있는 슈퍼셋**. 근거 = 구 검증 render freeze(`simulation/docker/base/requirements.txt`, 241개) + 코드가 import하는 라이브러리.

| 그룹 | 패키지 |
|---|---|
| backend (pth) | `torch==1.12.0+cpu` (별도 인덱스, requirements 밖) |
| backend (onnx) | `onnxruntime` (CPU, 구 freeze 1.13.1) |
| config 로딩 (지금도 필수) | `hydra-core`(1.1.0rc1), `omegaconf`(2.1.0rc1) — 모델 번들 native config(`data/model/pluto_v3/config.yaml`) 로딩 |
| cls 시뮬 | `scipy`, `numba` |
| render | `matplotlib`, `shapely`, `opencv-python`, `pillow`, `pyquaternion`, `pandas` |
| 맵/지오 (nuplan) | `geopandas`, `fiona`, `rasterio` |
| 파싱/기타 | `numpy`, `protobuf==3.20.3`, `cyber_record`, `pyyaml` |

### 1-C. 외부 라이브러리 = nuplan-devkit (Dockerfile이 버전 고정 fetch)
- **third-party 라이브러리**라 Dockerfile이 **pip로 버전 고정 설치**한다(소스가 아님 → clone 대상 아님).
- 소스: `github.com/motional/nuplan-devkit`, 커밋 **`e924167`**(v1.2+13, setup.py 1.2.2), 로컬 수정 없음(upstream clean) → 재현 가능.
- 설치: `pip install --no-deps "git+https://github.com/motional/nuplan-devkit.git@e924167"` (deps는 1-B에서 이미 설치).

### 1-D. 이미지에 넣지 않는 것
- **src 전체**(hanelso-av 코드; PLUTO는 `planning/models/pluto/src`에 vendored): clone이 가져와 **런타임 마운트**. Dockerfile 무관.
- **data/·work/**: 마운트. **model.onnx**: 아티팩트라 마운트(생기면).
- **onnxruntime-gpu, natten, torchvision/lightning/torchmetrics**: 불요(§ 학습전용·CUDA).

---

## 2. 이미지 환경변수 · WORKDIR

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace
WORKDIR /workspace
```
- `MPLBACKEND=Agg`: headless matplotlib(BEV·render).
- `PYTHONPATH=/workspace`: 마운트될 repo 코드(`common` 등) import.
- PLUTO는 `planning/models/pluto/src`에 vendored 되어 모델 소스는 `planning/models/pluto/src/`와 도메인 승격 코드로 repo 안에 직접 포함된다. 외부 마운트/고정 경로 불요.

---

## 3. 레퍼런스 Dockerfile (환경 전용 — src 없음)

```dockerfile
FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ffmpeg \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
        libsm6 libxext6 libxrender1 fonts-dejavu-core \
    # libgdal-dev gdal-bin   # geopandas/rasterio가 시스템 GDAL 요구시 주석해제
    && rm -rf /var/lib/apt/lists/*

# nuplan이 요구하는 hydra-core==1.1.0rc1 / omegaconf==2.1.0rc1 의 낡은 메타데이터를
# pip 24.1+ 가 거부하므로 pip<24.1 고정
RUN pip install "pip<24.1"

# 1) torch/torchvision CPU 쌍 고정 (timm import 체인이 torchvision 요구 — CUDA판 끌림 방지)
RUN pip install torch==1.12.0+cpu torchvision==0.13.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html

# 2) 파이썬 의존성 슈퍼셋 — 검증 freeze 를 --no-deps 로 그대로 재현 (resolver 미개입)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-deps -r /tmp/requirements.txt

# 3) nuplan-devkit (버전 고정, deps는 위에서 설치 → --no-deps)
RUN pip install --no-deps \
        "git+https://github.com/motional/nuplan-devkit.git@e924167"

WORKDIR /workspace

# 헬스체크: "환경"만 검증(설치형 의존성). src(common, planning/models/pluto/src)는 이미지에 없으므로
#   여기서 import하지 않는다 — 그건 run 시 마운트 후 스모크(§6)에서 검증.
RUN python -c "import numpy, torch, onnxruntime, cyber_record, nuplan, shapely, \
matplotlib, scipy, cv2, geopandas, hydra, omegaconf, pandas; \
print('ENV OK | torch', torch.__version__, '| onnxruntime', onnxruntime.__version__)"

CMD ["bash"]
```

> **주의(설계 불변식)**: 이 Dockerfile에는 `COPY <소스>`(hanelso-av/planning/models/pluto/src/nuplan-devkit)가 **없다**. requirements.txt 외에 호스트 로컬 파일에 의존하지 않는다(소스는 clone, nuplan은 pip).

---

## 4. requirements.txt 도출 방침 (codex 실작업)

- **1차 소스 = 구 검증 render freeze**(`simulation/docker/base/requirements.txt`, 241개). 렌더·시뮬까지 실동작하던 슈퍼셋.
- 절차:
  1. 구 freeze에서 **제거**: torch 계열, onnxruntime-gpu, natten/pytorch-lightning/torchmetrics/torchvision(학습전용·§1-D), nuplan-devkit(§1-C에서 git로 별도).
  2. §1-B 그룹(backend-onnx/config/sim/render/geo/파싱) 전부 포함 확인.
  3. `docker build` → §3 헬스체크 통과 → `ModuleNotFoundError` 시 구 freeze 대조해 추가.
  4. §6 스모크까지 통과할 때까지 반복, 최종 pin.
- torch는 requirements 밖(Dockerfile 별도). onnxruntime(CPU) 포함. onnxruntime-gpu·natten·model.onnx 제외.

---

## 5. 실행 규약 (docker run — 빌드 밖)

```bash
# clone된 워킹트리(= hanelso-av 코드 + planning/models/pluto/src vendored 포함)를 마운트
RUN="docker run --rm \
  -v $PWD:/workspace \
  av-base:latest"

$RUN python parse_map.py    ...                     # base_map.bin -> MapGraph
$RUN python parse_clip.py   configs/e100bt25.py     # bag -> 통합포맷
$RUN python planning/run_inference.py \
      --checkpoint-path data/model/v3_pluto.ckpt ... # 추론(pth). onnx backend는 코드 추가 후 동일 환경에서
# render -> mp4 : 스크립트 추가 후 이 환경에서 동작(ffmpeg 포함)
```
> 정확한 CLI 인자는 실소스 기준. PLUTO는 vendored라 .

---

## 6. 검증 기준 (DoD — codex)

1. **깨끗한 컨텍스트**(호스트 로컬 소스 없이)에서 `docker build` 성공 + **§3 환경 헬스체크 통과**.
2. **재현성 확인**: Dockerfile이 로컬 파일(requirements.txt 제외)에 의존하지 않음 — `COPY <소스>`·build.sh 부재 확인.
3. **마운트 스모크**: clone된 트리를 마운트해 parse_clip→run_inference 실행 →
   `work/inference/<scene>/infer_report.txt` 가 기존 결과와 수치 정합.
   (기준: `work/inference/{E100BT-25_...00006, E100BT-22_...00014}/`)
4. **CPU 전용**: `--gpus` 없이 동작.
5. 운영 EXEC 문서(로컬 `agent/`)에 빌드 로그·헬스체크·스모크 수치 기록.
6. 결과물: `docker/base/{Dockerfile, requirements.txt, README.md}`(build.sh 없음) + 구 `simulation/docker` deprecated 배너.
7. git commit + push.

---

## 7. 미결·리스크

- **cls 시뮬·render→mp4·onnx backend 코드 부재**: 본 이미지는 "환경"만 갖춤(pth+onnx CPU, sim/render 라이브러리 완비). 실제 동작은 해당 src가 clone 트리에 존재해야 함 — **코드는 별도 태스크**(docker 범위 밖).
- **PLUTO 소스 관리**: `planning/models/pluto/src`에 vendored 되어 repo(clone)에 포함 — docker 무관.
- **nuplan 커밋 e924167**: 원격 접근성 전제. 접근 불가 시 대체 pin 필요.
- **GDAL**: geopandas/fiona/rasterio 빌드 실패 시 §1-A GDAL 폴백.

---

## 8. codex 위임 노트
- **workdir**: `/home/hanelso/hanelso/hanelso-av`. 산출물 = `docker/base/{Dockerfile, requirements.txt, README.md}`.
- **금지**: Dockerfile에 `COPY <소스>` / build.sh 재도입 / 호스트 로컬 경로 의존.
- **EXEC.md**: workdir 내 `agent/` EXEC 문서 작성 후 오케스트레이터가 별도 보관.
- `run_in_background=true`.
