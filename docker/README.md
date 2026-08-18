# docker/ — env 구성 (이미지 = 모델 소유)

## 규약 (config `env` 필드)

- **공용 파이프라인**(파싱·input·sim·render, 전부 CPU) → `av-base` 이미지.
  simulation 모듈 config의 `env="av-base"`.
- **모델 실추론**(GPU) → 이미지는 **모델이 소유**하고 `<모델>-inf`로 명명한다.
  planning 모듈 config의 `env="pluto-inf"` — 그 모델의 추론 환경(torch/CUDA 버전)은
  모델 config가 결정하고, 파이프라인은 이름만 참조한다.
- GPU(sm 대역) 변형은 **하나의 pluto-inf Dockerfile + build-arg**로 만들고 **태그**로 구분한다:
  `pluto-inf:cu116`(Turing~Ampere, sm_75/86 — 2060·A6000), `pluto-inf:cu121`(Ada, sm_89 — 5880).
  새 GPU = build-arg 세트 하나 추가. `latest`는 현 서버 기준.

## 구성

| 경로 | 이미지 | 내용 |
|---|---|---|
| `base/` | `av-base:latest` | python3.9 + torch 1.12 **CPU** + 검증 freeze + nuplan-devkit. 전 파이프라인(parse→input→추론(CPU)→sim→mp4) 컨테이너 검증됨. 빌드/사용법은 `base/README.md`. |
| `pluto-inf/` | `pluto-inf:cu116` (외 arch 태그) | nvidia/cuda + python3.9 + torch(+cuXXX) + **자기 requirements.txt**(base 독립) + nuplan-devkit. 하나의 Dockerfile이 build-arg로 GPU 플랫폼별 빌드. GPU 전 파이프라인(추론·sim·mp4). |

## 빌드

```bash
# CPU 공용
docker build -t av-base:latest -f docker/base/Dockerfile docker/base/

# GPU (pluto-inf) — 하나의 Dockerfile이 GPU 플랫폼별 변형을 build-arg로 빌드.
# 컨텍스트 = docker/pluto-inf/ (자기 requirements.txt, base 참조 없음)
#   2060·A6000 (Turing~Ampere, sm_75/86) — 기본값
docker build -t pluto-inf:cu116 -t pluto-inf:latest -f docker/pluto-inf/Dockerfile docker/pluto-inf/
#   5880 (Ada, sm_89) — Ada 서버에서 빌드·검증
docker build -t pluto-inf:cu121 \
  --build-arg CUDA_TAG=12.1.1 --build-arg TORCH_VERSION=2.1.2 \
  --build-arg TORCH_CU=cu121 --build-arg TORCHVISION_VERSION=0.16.2 \
  -f docker/pluto-inf/Dockerfile docker/pluto-inf/
```

## GPU 실추론 실행

```bash
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace pluto-inf:latest \
  python planning/run_inference.py configs/e100bt25.py --device cuda
```

## bag → mp4 시뮬 실행 (docker 단독 — host 파이썬 env 불필요)

`run_sim.py` 하나로 bag→parse→추론→render→mp4 전 구간을 컨테이너에서 돈다. host 에 별도
host 파이썬 env가 없어도 된다 — 파싱·추론·렌더 의존성이 이미지 단일 환경에 모두
있고, 전 스테이지가 컨테이너의 인터프리터 하나로 실행된다. CPU·GPU 둘 다 검증됨.

> **데이터 마운트**: bag 등 `data/` 는 `/mnt/hdd_storage` 심볼릭이라, 그 디스크를 같이
> 마운트해야 컨테이너에서 원본 bag 이 보인다. 맵 그래프가 `work/maps/` 에 이미 있으면
> bag 접근용 `/mnt/hdd_storage` 하나면 충분하다.

```bash
# CPU (av-base)
docker run --rm \
  -v "$PWD":/workspace \
  -v /mnt/hdd_storage:/mnt/hdd_storage \
  -w /workspace av-base:latest \
  python run_sim.py <record> --mode closed_loop

# GPU (pluto-inf)
docker run --rm --gpus all \
  -v "$PWD":/workspace \
  -v /mnt/hdd_storage:/mnt/hdd_storage \
  -w /workspace pluto-inf:latest \
  python run_sim.py <record> --mode closed_loop --device cuda
```

산출물: `work/<clip>/sim/<mode>[_nuplan].mp4` (+ metrics json — 기본 렌더러 `nuplan`은 `_nuplan` suffix). 개별 스테이지로 나눠 돌리려면
`python parse_clip.py <config>` → `python simulation/render_sim.py <config> [--device cuda]`.
