FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/workspace

# Enable libgdal-dev and gdal-bin here only if the geospatial wheels fall back to source builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# pip<24.1 고정: nuplan이 요구하는 hydra-core==1.1.0rc1 / omegaconf==2.1.0rc1 이
# 낡은(invalid) 메타데이터라 pip 24.1+ 에서 설치 거부됨 → 24.1 미만 사용
RUN pip install "pip<24.1"

# torchvision(CPU)은 코드가 직접 쓰진 않지만 timm import 체인이 요구 → CPU판 고정으로 CUDA 끌림 방지
RUN pip install torch==1.12.0+cpu torchvision==0.13.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html

# requirements.txt = 검증된 render+sim freeze(OS 패키지 제거본). 완전한 클로저라
# --no-deps로 그대로 재현(resolver 미개입 → cyber-record protobuf 상한 등 충돌 회피).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-deps -r /tmp/requirements.txt

RUN pip install --no-deps \
        "git+https://github.com/motional/nuplan-devkit.git@e924167"

WORKDIR /workspace

RUN python -c "import cv2, cyber_record, fiona, geopandas, hydra, matplotlib, numba, numpy, omegaconf, onnxruntime, pandas, pyquaternion, rasterio, scipy, shapely, timm, torch, yaml; import nuplan; print('ENV OK | torch', torch.__version__, '| onnxruntime', onnxruntime.__version__)"

CMD ["bash"]
