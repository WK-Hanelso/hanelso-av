# =============================================================================
# PLUTO x Apollo closed-loop 시뮬레이션 파이프라인 (파싱 + 추론 + 렌더 단일 env)
# -----------------------------------------------------------------------------
# 설계 근거(검증됨 2026-07-24):
#  - 파싱(cyber-record)+추론(onnxruntime)+렌더가 numpy 1.23.4 한 venv에서 다 돎.
#    (cyber-record는 numpy 미요구 → 예전 numpy2 분리/피클꼼수 불필요)
#  - 추론은 CPU(onnxruntime, PlutoPlanner use_gpu=False). torch도 feature builder용으로만
#    쓰이고 CPU로 충분 → CPU 빌드. 그래서 2060/5880 Ada/노트북 어디서든 동작(CUDA 불필요).
#  - 이건 'sim 전용' 이미지. 학습/pth→onnx export는 별도 GPU 이미지(아티팩트 onnx로 연결).
#  - 제외한 학습전용 deps: natten(cu116 컴파일), pytorch-lightning, torchmetrics,
#    torchvision (렌더 import 체인에 없음 확인됨).
#
# GPU 추론이 꼭 필요해지면: base를 nvidia/cuda:11.6-cudnn8-runtime 로 바꾸고
#   torch는 cu116, onnxruntime→onnxruntime-gpu 로 교체(단 5880 Ada sm_89 호환 안 됨, 2060 전용).
# =============================================================================
FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

# --- 시스템 의존성 ---
#  ffmpeg          : 렌더 스크립트가 프레임→mp4 인코딩에 사용(subprocess)
#  libgl1/libglib  : opencv-python 런타임
#  libgomp1        : scipy/onnxruntime OpenMP
#  git/build-essential : 일부 소스빌드 및 nuplan-devkit 설치
#  (geopandas/rasterio는 manylinux 휠이 GDAL 번들 → 보통 시스템 GDAL 불필요.
#   만약 그 계열 빌드 실패하면 아래 주석 해제)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ffmpeg \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
    # libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# --- 1) torch CPU 먼저 (다른 패키지가 torch 요구시 CPU판으로 고정) ---
RUN pip install torch==1.12.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html

# --- 2) 나머지 파이썬 의존성 (렌더 freeze - 학습전용 - torch계열 + cyber-record) ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# --- 3) nuplan-devkit (로컬 수정본; deps는 위에서 설치했으니 --no-deps) ---
COPY nuplan-devkit /opt/nuplan-devkit
RUN pip install --no-deps /opt/nuplan-devkit

# --- 4) 코드 배치 ---
# ※ 스크립트가 아직 "/home/culee/pluto_onnx" 및 ONNX 모델 경로를 하드코딩 중이라
#    무수정 동작을 위해 동일 경로로 미러링함.
#    [재구축 TODO] PLUTO_ONNX_ROOT / ONNX_MODEL 환경변수로 파라미터화하면 이 미러링 제거 가능.
#    (pluto_onnx는 추론 서브셋만: src/ + onnx_export/model.onnx. 학습 스캐폴딩 제외)
COPY pluto_onnx   /home/culee/pluto_onnx
COPY apollo_pb2   /home/culee/pluto_apollo/apollo_pb2
COPY scripts      /home/culee/pluto_apollo/scripts

ENV PYTHONPATH=/home/culee/pluto_apollo/scripts:/home/culee/pluto_apollo/apollo_pb2:/home/culee/pluto_onnx \
    ONNX_MODEL=/home/culee/pluto_onnx/onnx_export/model.onnx

WORKDIR /home/culee/pluto_apollo

# 입력(record/map/raw)·출력(mp4) 마운트 지점
VOLUME ["/data", "/out"]

# 헬스체크: 핵심 스택이 한 env에서 다 import되는지
RUN python -c "import numpy, torch, onnxruntime, nuplan, cyber_record, shapely, matplotlib; \
print('OK numpy', numpy.__version__, '| torch', torch.__version__, '| onnxruntime', onnxruntime.__version__)"

# 기본은 셸. 실제 사용은 예:
#   docker run --rm -v $PWD/out:/out pluto-apollo-sim \
#     python scripts/render_closed_loop_post.py /data/raw.pkl /data/map.pkl --out /out/x.mp4
CMD ["bash"]
