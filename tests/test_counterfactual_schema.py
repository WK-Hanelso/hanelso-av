"""반사실 생성기 스키마·게이트 테스트 (FORMAT_SPEC §11) — 외부 데이터 불필요(합성 입력).

실행: docker run --rm -v "$PWD":/workspace -w /workspace av-base:latest \
        python -m pytest tests/test_counterfactual_schema.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from data_devkit.counterfactual.generator import (
    GOAL_HORIZON_S,
    MAX_CURVATURE,
    MAX_LAT_ACCEL,
    N_STEPS,
    VocabCoverage,
    build_payload,
    lane_transplant,
    time_transplant,
)


def straight_future(speed: float = 8.0) -> np.ndarray:
    x = speed * 0.5 * np.arange(1, N_STEPS + 1)
    return np.stack([x, np.zeros(N_STEPS), np.zeros(N_STEPS)], axis=1)


def test_entry_schema_conforms():
    """§11 필수 필드·타입이 정확히 생성된다 (추가·누락 없음)."""
    e = lane_transplant("a" * 32, straight_future(), +3.5)
    assert set(e) == {"token", "source_sample_token", "intervention", "goal",
                      "reference_trajectory_ego", "quality", "valid"}
    assert set(e["goal"]) == {"point_ego", "horizon_s", "intent"}
    assert set(e["quality"]) == {"max_curvature", "max_lat_accel", "vocab_min_dist_m", "flags"}
    assert e["goal"]["horizon_s"] == GOAL_HORIZON_S
    assert len(e["reference_trajectory_ego"]) == N_STEPS
    assert len(e["token"]) == 32


def test_tokens_deterministic():
    """같은 입력 → 같은 토큰 (재실행 안정성)."""
    a = lane_transplant("b" * 32, straight_future(), +3.5)
    b = lane_transplant("b" * 32, straight_future(), +3.5)
    c = lane_transplant("b" * 32, straight_future(), -3.5)
    assert a["token"] == b["token"] and a["token"] != c["token"]


def test_lane_transplant_geometry_and_intent():
    """횡 이식: 끝점이 ±offset에 도달, C1 시작(초기 횡변위 미미), intent 부호 정합."""
    left = lane_transplant("c" * 32, straight_future(), +3.5)
    ref = np.array(left["reference_trajectory_ego"])
    assert abs(ref[-1, 1] - 3.5) < 0.1
    assert abs(ref[0, 1]) < 0.2          # quintic 시작 완만
    assert left["goal"]["intent"] == "lane_change_left"
    assert lane_transplant("c" * 32, straight_future(), -3.5)["goal"]["intent"] == "lane_change_right"


def test_quality_gate_rejects_infeasible():
    """§11.1: 물리 한계 위반은 valid=false로 저장된다 (버리지 않음)."""
    jerky = straight_future(speed=15.0)
    jerky[::2, 1] += 4.0                  # 지그재그 — 곡률·횡가속 폭발
    e = lane_transplant("d" * 32, jerky, +3.5)
    assert e["valid"] is False
    assert e["quality"]["max_curvature"] > MAX_CURVATURE or e["quality"]["max_lat_accel"] > MAX_LAT_ACCEL


def test_time_transplant_intent_from_shape():
    donor = straight_future()
    donor[:, 1] = np.linspace(0.5, 6.0, N_STEPS)   # 좌로 크게 휨
    e = time_transplant("e" * 32, donor, "f" * 32, "log_x")
    assert e["intervention"]["type"] == "time_transplant"
    assert e["goal"]["intent"] == "turn_left"


def test_vocab_coverage_min_dist():
    """§11.2: 자기 자신이 vocab에 있으면 거리 0, 없으면 양수."""
    ref = straight_future()
    vocab = np.stack([ref, ref + [0, 5.0, 0]])[:, None]   # (2,1,8,3) 조합축 흉내
    cov = VocabCoverage(vocab)
    assert cov.min_dist(ref[:, :2]) < 1e-6
    assert cov.min_dist(ref[:, :2] + [0, 2.0]) > 1.0


def test_payload_meta():
    p = build_payload([lane_transplant("a" * 32, straight_future(), +3.5)])
    assert set(p) == {"meta", "entries"}
    assert set(p["meta"]) == {"generator", "generator_version", "created"}
