# pluto GPU 실추론 env — 이미지는 모델이 소유한다 (planning/configs/pluto.py env="pluto-inf").
# 공용 파이프라인(파싱·sim·render)은 swm-base(CPU)로 돌고, GPU는 이 이미지를 쓴다.
# CPU 이미지(base)와 완전 독립: 자기 requirements.txt 를 가진다 (base 참조 없음).
#
# 하나의 Dockerfile 이 GPU 플랫폼(sm 대역)별 변형을 build-arg 로 빌드한다.
# 플랫폼 모듈 = (CUDA_TAG, TORCH_VERSION, TORCH_CU, TORCHVISION_VERSION) 한 세트.
#
# 빌드 (컨텍스트 = docker/pluto-inf/ — 자기 requirements.txt 사용):
#   # 2060·A6000 (Turing~Ampere, sm_75/86) — 기본값
#   docker build -t pluto-inf:cu116 -t pluto-inf:latest \
#     -f docker/pluto-inf/Dockerfile docker/pluto-inf/
#
#   # 5880 (Ada, sm_89) — Ada 서버에서 빌드·검증
#   docker build -t pluto-inf:cu121 \
#     --build-arg CUDA_TAG=12.1.1 \
#     --build-arg TORCH_VERSION=2.1.2 \
#     --build-arg TORCH_CU=cu121 \
#     --build-arg TORCHVISION_VERSION=0.16.2 \
#     -f docker/pluto-inf/Dockerfile docker/pluto-inf/
#
# 실행 (repo 마운트, GPU):
#   docker run --rm --gpus all -v "$PWD":/workspace pluto-inf:latest \
#     python run_sim.py <record> --mode closed_loop --device cuda

ARG CUDA_TAG=11.6.2
FROM nvidia/cuda:${CUDA_TAG}-cudnn8-runtime-ubuntu20.04

# GPU 플랫폼별 torch 스택 (build-arg). 기본 = 2060/A6000(sm_75/86) 검증본.
ARG TORCH_VERSION=1.12.0
ARG TORCH_CU=cu116
ARG TORCHVISION_VERSION=0.13.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
        curl \
        git \
        build-essential \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python3

# pip<24.1 고정: nuplan이 요구하는 hydra-core==1.1.0rc1 / omegaconf==2.1.0rc1 이
# 낡은(invalid) 메타데이터라 pip 24.1+ 에서 설치 거부됨 (swm-base와 동일 사유)
RUN curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && pip install "pip<24.1"

# GPU torch: 플랫폼별 build-arg (sm 대역에 맞는 CUDA 빌드)
RUN pip install torch==${TORCH_VERSION}+${TORCH_CU} torchvision==${TORCHVISION_VERSION}+${TORCH_CU} \
        -f https://download.pytorch.org/whl/torch_stable.html

# 자기 requirements (base 참조 제거 — CPU/GPU 독립). --no-deps 로 freeze 그대로 재현
# (resolver 미개입 → torch 를 CPU판으로 덮어쓰는 사고 방지; freeze 에 torch 계열 없음)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-deps -r /tmp/requirements.txt

RUN pip install --no-deps \
        "git+https://github.com/motional/nuplan-devkit.git@e924167"

WORKDIR /workspace

# 빌드 머신에 GPU 가 없을 수 있으므로 import + cuda 빌드 문자열까지만 확인
RUN python -c "import cv2, cyber_record, hydra, matplotlib, numpy, omegaconf, pandas, scipy, shapely, timm, torch, yaml; import nuplan; print('ENV OK | torch', torch.__version__, '| cuda build', torch.version.cuda)"

CMD ["bash"]
