"""반사실 goal 생성기 — FORMAT_SPEC v0.2 §11 생산자.

"실제와 다른 goal + 그리로 가는 것이 정답인 궤적" 쌍을 생성한다 (goal 채널 사망 방지 —
설계 원장 §7.2; 근거: stock 모델 명령 무시 실측). 개입 2종:
  lane_transplant: GT 미래에 횡방향 quintic 프로파일(±offset)을 이식 — 차선 변경형 반사실
  time_transplant: 같은 로그의 속도 유사·모양 상이 구간 궤적을 이식

torch import 금지 (data_devkit 규약 — 경량 파싱 환경 호환). numpy만 사용.
스키마·품질 게이트·vocab 커버리지의 정의는 common/FORMAT_SPEC.md §11이 SoT다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

GENERATOR_NAME = "counterfactual_rule_v1"
GENERATOR_VERSION = "1.0"

DT = 0.5                 # reference 궤적 간격 (초) — §11
N_STEPS = 8              # 4초 horizon
GOAL_HORIZON_S = 2.0     # §10 goal 규약 (index 3 = 2.0s)
LANE_OFFSET_M = 3.5

# §11.1 기하 타당성 한계 (생성기 공통 게이트)
MAX_LAT_ACCEL = 4.0      # m/s^2
MAX_CURVATURE = 0.2      # 1/m


def _token(*parts: object) -> str:
    """결정론적 32-hex 토큰 (동일 입력 → 동일 토큰: 재실행 안정성)."""
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()


def _quintic(t: np.ndarray) -> np.ndarray:
    """C1 연속 횡 프로파일 (시작 변위·변화율 0)."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def trajectory_quality(ref: np.ndarray) -> Dict[str, float]:
    """§11.1 지표: 최대 곡률·횡가속 (원점에서 시작하는 (N,3) ego 궤적)."""
    xy = np.vstack([[0.0, 0.0], ref[:, :2]])
    seg = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    speed = seg_len / DT
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    dh = np.diff(heading)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    curvature = np.abs(dh) / np.maximum(seg_len[1:], 0.3)
    lat_accel = (speed[1:] ** 2) * curvature
    return {
        "max_curvature": float(curvature.max()) if len(curvature) else 0.0,
        "max_lat_accel": float(lat_accel.max()) if len(lat_accel) else 0.0,
    }


class VocabCoverage:
    """§11.2 — scoring vocabulary 최근접 거리 (조합 vocab (K, N, >=2))."""

    def __init__(self, trajectory: np.ndarray, mask: Optional[np.ndarray] = None):
        self.vocab = trajectory.reshape(-1, trajectory.shape[-2], trajectory.shape[-1])[
            :, :, :2
        ].astype(np.float32)
        if mask is None:
            self.mask = np.ones(self.vocab.shape[:2], dtype=bool)
        else:
            self.mask = mask.reshape(-1, mask.shape[-1]).astype(bool)

    @classmethod
    def from_npz(cls, path: str) -> "VocabCoverage":
        data = np.load(path)
        return cls(data["trajectory"], data.get("trajectory_mask"))

    def min_dist(self, ref_xy: np.ndarray, chunk: int = 65536) -> float:
        best = np.inf
        target = ref_xy[None].astype(np.float32)
        for i in range(0, len(self.vocab), chunk):
            v, m = self.vocab[i : i + chunk], self.mask[i : i + chunk]
            d = np.linalg.norm(v - target, axis=-1)
            mean_d = np.where(m, d, 0.0).sum(1) / np.maximum(m.sum(1), 1)
            best = min(best, float(mean_d.min()))
        return best


def make_entry(
    source_sample_token: str,
    intervention_type: str,
    params: Dict[str, Any],
    reference: np.ndarray,
    intent: str,
    flags: Sequence[str],
    vocab: Optional[VocabCoverage] = None,
) -> Dict[str, Any]:
    """§11 스키마의 entry 1개를 만든다 (스키마 필드 추가·의미 변경 금지 — 설계 고정)."""
    q = trajectory_quality(reference)
    valid = q["max_curvature"] <= MAX_CURVATURE and q["max_lat_accel"] <= MAX_LAT_ACCEL
    goal_idx = int(round(GOAL_HORIZON_S / DT)) - 1
    return {
        "token": _token(source_sample_token, intervention_type, json.dumps(params, sort_keys=True)),
        "source_sample_token": source_sample_token,
        "intervention": {"type": intervention_type, "params": params},
        "goal": {
            "point_ego": [round(float(reference[goal_idx, 0]), 3), round(float(reference[goal_idx, 1]), 3)],
            "horizon_s": GOAL_HORIZON_S,
            "intent": intent,
        },
        "reference_trajectory_ego": [[round(float(v), 3) for v in row] for row in reference],
        "quality": {
            "max_curvature": round(q["max_curvature"], 4),
            "max_lat_accel": round(q["max_lat_accel"], 3),
            "vocab_min_dist_m": (round(vocab.min_dist(reference[:, :2]), 3) if vocab else None),
            "flags": list(flags),
        },
        "valid": bool(valid),
    }


def _recompute_heading(ref: np.ndarray) -> np.ndarray:
    seg = np.diff(np.vstack([[0.0, 0.0, 0.0], ref]), axis=0)
    ref = ref.copy()
    ref[:, 2] = np.arctan2(seg[:, 1], seg[:, 0])
    return ref


def lane_transplant(
    source_sample_token: str,
    gt_future: np.ndarray,
    offset_m: float,
    flags: Sequence[str] = (),
    vocab: Optional[VocabCoverage] = None,
) -> Dict[str, Any]:
    """GT 미래에 횡 quintic(offset_m)을 이식 — 차선 변경형 반사실."""
    ts = np.arange(1, N_STEPS + 1) / N_STEPS
    ref = gt_future.copy()
    ref[:, 1] += offset_m * _quintic(ts)
    ref = _recompute_heading(ref)
    intent = "lane_change_left" if offset_m > 0 else "lane_change_right"
    return make_entry(
        source_sample_token, "lane_transplant", {"lateral_offset_m": offset_m},
        ref, intent, flags, vocab,
    )


def time_transplant(
    source_sample_token: str,
    donor_future: np.ndarray,
    donor_sample_token: str,
    donor_log: str,
    flags: Sequence[str] = (),
    vocab: Optional[VocabCoverage] = None,
) -> Dict[str, Any]:
    """같은 로그 타 구간의 ego-frame 궤적을 이식."""
    lat_end = float(donor_future[-1, 1])
    intent = "turn_left" if lat_end > 3 else ("turn_right" if lat_end < -3 else "follow_lane")
    return make_entry(
        source_sample_token, "time_transplant",
        {"donor_sample_token": donor_sample_token, "donor_log": donor_log},
        donor_future, intent, flags, vocab,
    )


def build_payload(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """§11 counterfactual.json 최상위 구조."""
    return {
        "meta": {
            "generator": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "created": datetime.now(timezone.utc).isoformat(),
        },
        "entries": entries,
    }


def coverage_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """§11.2 생성 리포트 — 커버리지·유효율 분포 (침묵 절단 방지)."""
    valid = np.array([e["valid"] for e in entries], dtype=bool)
    dists = np.array(
        [e["quality"]["vocab_min_dist_m"] for e in entries if e["quality"]["vocab_min_dist_m"] is not None],
        dtype=float,
    )
    report: Dict[str, Any] = {
        "entries": len(entries),
        "valid": int(valid.sum()),
        "valid_rate": float(valid.mean()) if len(valid) else 0.0,
    }
    if len(dists):
        report["vocab_dist_p50"] = float(np.median(dists))
        report["vocab_dist_p90"] = float(np.percentile(dists, 90))
        report["vocab_dist_max"] = float(dists.max())
    by_type: Dict[str, Any] = {}
    for e in entries:
        t = e["intervention"]["type"]
        by_type.setdefault(t, {"n": 0, "valid": 0})
        by_type[t]["n"] += 1
        by_type[t]["valid"] += int(e["valid"])
    report["by_type"] = by_type
    return report
