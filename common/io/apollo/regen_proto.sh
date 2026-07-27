#!/usr/bin/env bash
# Apollo 커스텀 HD맵 proto → *_pb2.py 재생성.
# 소스: proto_src/ (stock Apollo 7.0 + 커스텀 zone 5종 + Id.id=bytes 패치, MAP_PROTO_CUSTOM_EXTENSIONS.md 참고)
# 도구: grpcio-tools(pyproject [dev]). 생성물 proto/ 는 커밋됨 → 런타임 protoc 불요.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; PY="${PYTHON:-python}"
SRC="$HERE/proto_src"; OUT="$HERE/proto"; rm -rf "$OUT"; mkdir -p "$OUT"
$PY -m grpc_tools.protoc -I="$SRC" --python_out="$OUT" \
  "$SRC"/modules/map/proto/*.proto "$SRC"/modules/common/proto/geometry.proto
find "$OUT" -type d -exec touch {}/__init__.py \;
echo "regenerated -> $OUT"
