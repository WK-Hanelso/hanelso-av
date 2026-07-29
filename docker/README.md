# docker/ — env 구성 (이미지 = 모델 소유)

## 규약 (config `env` 필드)

- **공용 파이프라인**(파싱·input·sim·render, 전부 CPU) → `swm-base` 이미지.
  simulation 모듈 config의 `env="swm-base"`.
- **모델 실추론**(GPU) → 이미지는 **모델이 소유**하고 `<모델>-inf`로 명명한다.
  planning 모듈 config의 `env="pluto-inf"` — 그 모델의 추론 환경(torch/CUDA 버전)은
  모델 config가 결정하고, 파이프라인은 이름만 참조한다.
- 하드웨어 변형은 **태그**로 구분한다: `pluto-inf:cu116`(현 서버 RTX 2060),
  향후 `pluto-inf:cu121` 등. `latest`는 현 서버 기준.

## 구성

| 경로 | 이미지 | 내용 |
|---|---|---|
| `base/` | `swm-base:latest` | python3.9 + torch 1.12 **CPU** + 검증 freeze + nuplan-devkit. 전 파이프라인(parse→input→추론(CPU)→sim→mp4) 컨테이너 검증됨. 빌드/사용법은 `base/README.md`. |
| `pluto-inf/` | `pluto-inf:cu116` | nvidia/cuda 11.6.2 + python3.9 + torch 1.12.0+cu116 + base freeze 재사용(--no-deps) + nuplan-devkit. GPU 실추론 전용. |

## 빌드

```bash
docker build -t swm-base:latest -f docker/base/Dockerfile docker/base/
docker build -t pluto-inf:cu116 -t pluto-inf:latest -f docker/pluto-inf/Dockerfile docker/
```

## GPU 실추론 실행

```bash
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace pluto-inf:latest \
  python planning/run_inference.py configs/e100bt25.py --device cuda
```
