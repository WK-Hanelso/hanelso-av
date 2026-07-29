# pluto GPU 실추론 env — 이미지는 모델이 소유한다 (planning/configs/pluto.py env="pluto-inf").
# 공용 파이프라인(파싱·sim·render)은 swm-base(CPU)로 돌고, 실추론만 <모델>-inf 이미지를 쓴다.
# 태그 규약: 하드웨어 변형은 태그로 구분한다 (예: pluto-inf:cu116, pluto-inf:cu121);
# latest = 현재 서버(RTX 2060, driver cu116)에 맞는 cu116.
#
# 빌드 (컨텍스트 = docker/ — base/requirements.txt 재사용):
#   docker build -t pluto-inf:cu116 -t pluto-inf:latest -f docker/pluto-inf/Dockerfile docker/
# 실행 (repo만 마운트, GPU):
#   docker run --rm --gpus all -v "$PWD":/workspace pluto-inf:latest \
#     python planning/run_inference.py configs/e100bt25.py --device cuda

FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

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

# GPU torch: 호스트 검증 환경과 동일한 1.12.0+cu116 고정 (재현성)
RUN pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 \
        -f https://download.pytorch.org/whl/torch_stable.html

# swm-base의 검증된 freeze를 --no-deps로 그대로 재사용 (resolver 미개입 —
# torch cu116을 CPU판으로 덮어쓰는 사고 방지; freeze에 torch 계열 없음 확인됨)
COPY base/requirements.txt /tmp/requirements.txt
RUN pip install --no-deps -r /tmp/requirements.txt

RUN pip install --no-deps \
        "git+https://github.com/motional/nuplan-devkit.git@e924167"

WORKDIR /workspace

# 빌드 머신에는 GPU가 없을 수 있으므로 cuda 빌드 문자열까지만 확인
RUN python -c "import cv2, cyber_record, hydra, matplotlib, numpy, omegaconf, pandas, scipy, shapely, timm, torch, yaml; import nuplan; print('ENV OK | torch', torch.__version__, '| cuda build', torch.version.cuda)"

CMD ["bash"]
