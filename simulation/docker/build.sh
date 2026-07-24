#!/usr/bin/env bash
# PLUTO x Apollo sim 이미지 빌드.
# 필요한 소스가 세 곳에 흩어져 있어(/opt/nuplan-devkit, ~/pluto_onnx, ~/pluto_apollo)
# 임시 빌드 컨텍스트에 모아서 docker build 한다.
set -euo pipefail

IMAGE="${IMAGE:-pluto-apollo-sim:latest}"
HERE="$(cd "$(dirname "$0")" && pwd)"          # .../pluto_apollo/docker
APOLLO="$(dirname "$HERE")"                     # .../pluto_apollo
NUPLAN="${NUPLAN_DEVKIT:-/opt/nuplan-devkit}"
PLUTO_ONNX="${PLUTO_ONNX_ROOT:-/home/culee/pluto_onnx}"

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
echo "[build] staging context -> $CTX"

# nuplan-devkit (로컬 수정본, .git/데이터 제외)
rsync -a --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
      --exclude 'nuplan/database/**/*.db' "$NUPLAN/" "$CTX/nuplan-devkit/"

# PLUTO 추론 서브셋만: src/ + onnx_export/model.onnx  (학습 스캐폴딩 제외)
mkdir -p "$CTX/pluto_onnx/onnx_export"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$PLUTO_ONNX/src/" "$CTX/pluto_onnx/src/"
cp "$PLUTO_ONNX/onnx_export/model.onnx" "$CTX/pluto_onnx/onnx_export/model.onnx"

# Apollo 어댑터/스크립트 + 컴파일된 pb2
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$APOLLO/scripts/"    "$CTX/scripts/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$APOLLO/apollo_pb2/" "$CTX/apollo_pb2/"

# Dockerfile + requirements
cp "$HERE/Dockerfile"       "$CTX/Dockerfile"
cp "$HERE/requirements.txt" "$CTX/requirements.txt"

echo "[build] docker build -t $IMAGE"
docker build -t "$IMAGE" "$CTX"
echo "[build] done: $IMAGE"
echo
echo "실행 예시:"
echo "  # 렌더(추론): raw+map -> mp4"
echo "  docker run --rm -v \$PWD/pluto_apollo/out:/out -v \$PWD/pluto_apollo/out:/data \\"
echo "    $IMAGE python scripts/render_closed_loop_post.py /data/raw.pkl /data/map.pkl --out /out/x.mp4"
echo "  # 파싱(record -> raw): record와 map을 /data에 마운트"
echo "  docker run --rm -v /path/bag:/data $IMAGE \\"
echo "    python scripts/dump_raw.py /data/xxx.record.00006 --map-graph /data/map_graph.pkl --out /data/raw.pkl"
