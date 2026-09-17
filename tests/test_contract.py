"""Data-free tests for artifact contract fail-fast diagnostics."""

from __future__ import annotations

import json

import pytest

from data_devkit.contract import ContractError, check


def test_missing_artifact_includes_producer_hint(tmp_path):
    with pytest.raises(ContractError) as caught:
        check(clip_id="clip", requires=["sample"], clip_dir=tmp_path)

    message = str(caught.value)
    assert "missing file" in message
    assert "python parse_clip.py" in message


def test_missing_required_key_reports_schema_mismatch(tmp_path):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "sample.json").write_text(json.dumps([{"token": "sample"}]))

    with pytest.raises(ContractError) as caught:
        check(clip_id="clip", requires=["sample"], clip_dir=tmp_path)

    message = str(caught.value)
    assert "schema mismatch" in message
    assert "timestamp" in message
    assert "scene_token" in message


def test_unsupported_provenance_reports_supported_values(tmp_path):
    with pytest.raises(ContractError) as caught:
        check(
            clip_id="clip",
            requires=[],
            data_cfg={"agents": "bevfusion"},
            clip_dir=tmp_path,
        )

    message = str(caught.value)
    assert "Unsupported provenance data.agents='bevfusion'" in message
    assert "apollo_gt" in message
