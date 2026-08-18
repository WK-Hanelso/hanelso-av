"""feature 규격 골든 패리티 테스트 (issue #16).

모델 입력 규격의 사실 근거(SoT)는 모델 번들 native config(`feature_builder` 절)와
원본 builder 기본값이다. 이 테스트는 dataloader가 실제로 그 규격대로 feature를
만드는지 — 상한(agent/static)·히스토리 길이·반경·ego shape — 를 검증한다.
규격 관련 코드를 수정했을 때 이 테스트가 어긋남을 즉시 드러낸다.

실행 (실행은 항상 docker run — av-base 이미지):

    docker run --rm -v "$PWD":/workspace -w /workspace av-base:latest \
        python -m pytest tests/test_feature_parity.py -v

번들(`data/model/pluto_v3`)이나 parsed 클립(`work/...`)이 없는 환경에서는
해당 테스트가 skip된다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "data" / "model" / "pluto_v3"
NATIVE_CONFIG = BUNDLE / "config.yaml"
PARSED_CLIP = REPO_ROOT / "work" / "E100BT-25_20260716151711_00006" / "parsed"
MAP_GRAPH = REPO_ROOT / "work" / "maps" / "AYG" / "map_graph.json"

needs_bundle = pytest.mark.skipif(
    not NATIVE_CONFIG.exists(), reason="model bundle (data/model/pluto_v3) not mounted"
)
needs_clip = pytest.mark.skipif(
    not (PARSED_CLIP.exists() and MAP_GRAPH.exists()),
    reason="parsed clip / map_graph (work/...) not available",
)


def _native_feature_builder_text() -> str:
    """번들 yaml에서 feature_builder 블록 원문을 dataloader와 독립적으로 추출."""
    text = NATIVE_CONFIG.read_text()
    match = re.search(r"^(\s*)feature_builder:\n((?:\1\s+.*\n)+)", text, re.M)
    assert match, "native config에 feature_builder 절이 없음"
    return match.group(2)


def _native_int(block: str, key: str, default: int | None = None) -> int:
    match = re.search(rf"^\s*{key}:\s*([\d.]+)\s*$", block, re.M)
    if match is None:
        assert default is not None, f"feature_builder에 {key} 없음 (기본값 미지정)"
        return default
    return int(float(match.group(1)))


def _native_float(block: str, key: str) -> float:
    match = re.search(rf"^\s*{key}:\s*([\d.]+)\s*$", block, re.M)
    assert match, f"feature_builder에 {key} 없음"
    return float(match.group(1))


@needs_bundle
def test_agent_static_limits_resolve_from_bundle():
    """issue #22: dataloader의 agent/static 상한 = 번들 native config 값 (전사 금지)."""
    from planning.models.pluto.dataloader import (
        NATIVE_DEFAULT_MAX_STATIC,
        native_feature_limits,
    )

    block = _native_feature_builder_text()
    expected_agents = _native_int(block, "max_agents")
    expected_static = _native_int(
        block, "max_static_obstacles", default=NATIVE_DEFAULT_MAX_STATIC
    )

    max_agents, max_static, source = native_feature_limits(
        {"bundle": str(BUNDLE), "model_config": "config.yaml"}
    )
    assert max_agents == expected_agents
    assert max_static == expected_static
    assert "config.yaml" in source  # 값의 출처가 번들임이 기록된다


def test_limits_require_bundle():
    """번들 정보 없이 상한을 침묵 기본값으로 채우지 않는다 — fail-fast."""
    from planning.models.pluto.dataloader import native_feature_limits

    with pytest.raises(ValueError):
        native_feature_limits({})


@needs_bundle
def test_history_and_radius_match_native():
    """HIST_STEPS/RADIUS가 학습 규격(history_horizon/sample_interval, radius)과 일치."""
    from planning.models.pluto.dataloader import DT, HIST_STEPS, RADIUS

    block = _native_feature_builder_text()
    history_horizon = _native_float(block, "history_horizon")
    sample_interval = _native_float(block, "sample_interval")
    assert sample_interval == DT
    assert HIST_STEPS == int(round(history_horizon / sample_interval)) + 1
    assert RADIUS == _native_float(block, "radius")


def test_feature_ego_shape_is_training_vehicle():
    """feature ego shape = 학습 분포 차량(pacifica 2.297×5.176) — 실차 제원과 분리."""
    from planning.models.pluto.dataloader import FEATURE_VEHICLE_DIMENSIONS

    assert FEATURE_VEHICLE_DIMENSIONS["pacifica"] == (2.297, 5.176)


def test_dynamic_categories_match_training_interest_types():
    """issue #23: 동적 분류 기준 = 학습 interested_objects_types (종류 기반, 속도 무관)."""
    from planning.models.pluto.dataloader import DYNAMIC_CATEGORIES

    # 원본 builder: TrackedObjectType.{VEHICLE, PEDESTRIAN, BICYCLE} (+EGO 별도)
    assert set(DYNAMIC_CATEGORIES) == {"vehicle", "pedestrian", "bicycle"}


@needs_bundle
@needs_clip
def test_built_feature_matches_native_spec():
    """실제 클립으로 feature를 빌드해 텐서 규격이 native 규격과 일치하는지 검증."""
    import json

    from planning.interface import get_dataloader, load_model
    from planning.models.pluto.dataloader import CATEGORY_CODES, HIST_STEPS

    load_model("pluto")
    block = _native_feature_builder_text()
    expected_agents = _native_int(block, "max_agents")
    agent_rows = expected_agents + 1  # slot 0 = ego (학습 builder의 ego prepend)

    # 실제 조립 경로와 동일하게 root config를 해석해 config를 구성한다
    from common.config import load_config

    cfg = load_config(str(REPO_ROOT / "configs" / "e100bt25.py"))
    plan_cfg = cfg["planning"]
    config = {
        "clip_id": cfg["clip_id"],
        "map_name": cfg["map_name"],
        "feature_vehicle": plan_cfg.get("feature_vehicle", "pacifica"),
        "calibration": cfg["calibration"],
        "data": cfg.get("data"),
        "bundle": str(BUNDLE),
        "model_config": plan_cfg["model_config"],
    }
    map_graph = json.loads(MAP_GRAPH.read_text())
    adapter = get_dataloader("pluto_feature")()
    clip = adapter.prepare_clip(str(PARSED_CLIP), map_graph, config)

    dataset = clip["dataset"]
    assert dataset["max_agents_native"] == expected_agents
    assert "config.yaml" in dataset["feature_limits_source"]

    t0_index = HIST_STEPS  # 첫 full-history 프레임 이후
    result = adapter.build_frame(clip, t0_index)

    agent = result.normalized_numpy_data["agent"]
    assert agent["position"].shape[0] == agent_rows
    assert agent["position"].shape[1] >= HIST_STEPS
    # slot 0 = ego, 나머지 유효 슬롯은 자주행 가능 종류 코드(1~3)만
    assert int(agent["category"][0]) == CATEGORY_CODES["ego"]
    valid_slots = agent["valid_mask"].any(axis=-1)
    non_ego = [
        int(code)
        for slot, code in enumerate(agent["category"])
        if slot > 0 and valid_slots[slot]
    ]
    assert non_ego, "선택된 주변 agent가 없음"
    assert set(non_ego) <= {
        CATEGORY_CODES["vehicle"],
        CATEGORY_CODES["pedestrian"],
        CATEGORY_CODES["bicycle"],
    }
