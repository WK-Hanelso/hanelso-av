#!/usr/bin/env bash
#
# sim.sh — bag → 시뮬레이션 mp4 원샷 런처
#
# docker 명령을 몰라도 시뮬을 돌릴 수 있게 하는 래퍼. 내부에서 알아서:
#   이미지 확인(없으면 자동 빌드) → docker run(코드·데이터 마운트) → run_sim.py 실행.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 기본값 (CPU) ---
DEVICE=cpu
IMAGE=av-base:latest
DOCKERFILE=docker/base/Dockerfile
BUILD_CONTEXT=docker/base
MODE=closed_loop
STEPS=""
VERBOSE=0
USE_GPU=0
RECORD=""

usage() {
  cat <<'USAGE'
sim.sh — bag → 시뮬레이션 mp4 원샷 런처 (docker 자동 처리)

사용법:
  scripts/sim.sh <record> [옵션]

인자:
  <record>       Apollo .record 경로
                 예: data/bag/E100BT-25/20260716151711.record.00006

옵션:
  --gpu          GPU로 실행 (pluto-inf 이미지 + CUDA). 기본은 CPU(av-base).
  --steps N      시뮬 스텝 수
  --mode M       closed_loop | open_loop  (기본: closed_loop)
  --verbose      빌드/실행 상세 로그 출력
  -h, --help     이 도움말

동작:
  이미지가 없으면 자동 빌드 → docker run(코드·데이터 마운트) → run_sim.py 실행.
  산출물: work/<clip>/sim/<mode>[_nuplan].mp4 (기본 렌더러 nuplan 은 _nuplan suffix)

예시:
  scripts/sim.sh data/bag/E100BT-25/20260716151711.record.00006
  scripts/sim.sh data/bag/E100BT-25/20260716151711.record.00006 --gpu --steps 100
USAGE
}

# --- 인자 파싱 ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)     USE_GPU=1; shift ;;
    --steps)   STEPS="${2:?--steps 값이 필요합니다}"; shift 2 ;;
    --mode)    MODE="${2:?--mode 값이 필요합니다}"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*)        echo "[sim] 알 수 없는 옵션: $1" >&2; usage; exit 1 ;;
    *)         if [[ -z "$RECORD" ]]; then RECORD="$1"; shift; else echo "[sim] 인자 과다: $1" >&2; exit 1; fi ;;
  esac
done

[[ -n "$RECORD" ]] || { echo "[sim] 오류: <record> 경로가 필요합니다." >&2; echo; usage; exit 1; }

if [[ "$USE_GPU" == 1 ]]; then
  DEVICE=cuda
  IMAGE=pluto-inf:latest
  DOCKERFILE=docker/pluto-inf/Dockerfile
  BUILD_CONTEXT=docker/pluto-inf
fi

command -v docker >/dev/null 2>&1 || { echo "[sim] 오류: docker가 설치되어 있지 않습니다." >&2; exit 1; }

# --- 이미지 없으면 자동 빌드 ---
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[sim] 이미지 '$IMAGE' 없음 → 빌드합니다 (처음 한 번은 수 분~십수 분 걸립니다)..."
  if [[ "$VERBOSE" == 1 ]]; then
    docker build -t "$IMAGE" -f "$REPO_ROOT/$DOCKERFILE" "$REPO_ROOT/$BUILD_CONTEXT"
  else
    docker build -q -t "$IMAGE" -f "$REPO_ROOT/$DOCKERFILE" "$REPO_ROOT/$BUILD_CONTEXT" >/dev/null
  fi
  echo "[sim] 이미지 빌드 완료: $IMAGE"
else
  echo "[sim] 이미지 확인됨: $IMAGE"
fi

# --- 마운트: 코드(repo) + 데이터 디스크(있으면) ---
# data/ 는 물리 디스크(/mnt/hdd_storage 등) 심볼릭이라, 그 디스크도 함께 마운트해야
# 컨테이너에서 원본 bag 이 보인다.
MOUNTS=(-v "$REPO_ROOT:/workspace")
[[ -d /mnt/hdd_storage ]] && MOUNTS+=(-v /mnt/hdd_storage:/mnt/hdd_storage)
[[ -d /mnt/hdd ]] && MOUNTS+=(-v /mnt/hdd:/mnt/hdd)

GPU_RUN=()
[[ "$USE_GPU" == 1 ]] && GPU_RUN=(--gpus all)

RUN_ARGS=(python run_sim.py "$RECORD" --mode "$MODE" --device "$DEVICE")
[[ -n "$STEPS" ]] && RUN_ARGS+=(--steps "$STEPS")

echo "[sim] 실행: image=$IMAGE gpu=$USE_GPU mode=$MODE device=$DEVICE record=$RECORD${STEPS:+ steps=$STEPS}"
[[ "$VERBOSE" == 1 ]] && set -x
docker run --rm "${GPU_RUN[@]}" "${MOUNTS[@]}" -w /workspace "$IMAGE" "${RUN_ARGS[@]}"
