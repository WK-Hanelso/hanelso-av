"""데이터 계약 — 아티팩트 registry + 존재/스키마 check (C-SWM-023).

계약
----
- 아티팩트 이름 → work/ 내 경로규약(required_files) + 간단 스키마(필수 키).
- scope="clip"  : ``work/<clip_id>/`` 상대 경로.
- scope="map"   : ``work/maps/<map_name>/`` 상대 경로 (클립 무관).
- provenance 축 : 같은 아티팩트에 복수 생산자가 올 수 있다
  (예: agent_tracks = apollo_gt | bevfusion).  root config의
  ``data=dict(agents="apollo_gt", prediction=None)`` 섹션이 출처를 고른다.
  현재 유효 값: agents="apollo_gt", prediction=None 뿐 — 축만 세워둔다.
- check 실패 = fail-fast + "무엇을 돌려야 하는지" 안내.  자동 생성 금지
  (그건 오케스트레이터 몫).
- 스키마 체크는 "필수 파일 존재 + 필수 키 존재" 수준.  물리 검증은
  tools/parser_validation 소관 — 여기서 중복 구현하지 않는다.
- torch import 금지 (.venv-apollo / python3 양쪽에서 동작해야 함).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_ROOT = REPO_ROOT / "work"

# root config data 섹션의 축과 현재 유효 값.
DATA_AXES: Dict[str, Tuple[Optional[str], ...]] = {
    "agents": ("apollo_gt",),   # 예약: "bevfusion" (perception 모델 산출)
    "prediction": (None,),      # 예약: "apollo" (prediction 모듈 산출)
}
DEFAULT_DATA_CFG: Dict[str, Optional[str]] = {"agents": "apollo_gt", "prediction": None}


class ContractError(RuntimeError):
    """계약 위반 (누락 파일 / 스키마 불일치 / 잘못된 provenance)."""


@dataclass(frozen=True)
class ArtifactSpec:
    """아티팩트 하나의 계약: 경로규약 + provenance + 필수 키."""

    name: str
    scope: str                              # "clip" | "map"
    required_files: Tuple[str, ...]         # scope 루트 상대 경로
    provenances: Tuple[str, ...] = ("apollo_gt",)
    axis: Optional[str] = None              # data 섹션에서 이 아티팩트를 고르는 축
    # 파일(상대경로) -> 필수 키.  list-of-rows json이면 첫 row에서,
    # dict json이면 최상위에서 키 존재를 본다.
    required_keys: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    produce_hint: str = ""


_PARSE_HINT = (
    "run: .venv-apollo/bin/python parse_clip.py configs/<root>.py "
    "(root config with clip_id={clip_id})"
)
_MAP_HINT = (
    "run: .venv-apollo/bin/python parse_map.py --map <base_map.bin> "
    "--name {map_name}"
)

ARTIFACTS: Dict[str, ArtifactSpec] = {
    spec.name: spec
    for spec in [
        ArtifactSpec(
            name="sample",
            scope="clip",
            required_files=("parsed/sample.json",),
            required_keys={"parsed/sample.json": ("token", "timestamp", "scene_token")},
            produce_hint=_PARSE_HINT,
        ),
        ArtifactSpec(
            name="ego_pose",
            scope="clip",
            required_files=("parsed/ego_pose.json",),
            required_keys={
                "parsed/ego_pose.json": ("token", "timestamp", "translation", "rotation")
            },
            produce_hint=_PARSE_HINT,
        ),
        ArtifactSpec(
            name="ego_dynamics",
            scope="clip",
            required_files=("parsed/ego_dynamics.json",),
            required_keys={
                "parsed/ego_dynamics.json": (
                    "sample_token",
                    "ego_pose_token",
                    "speed_mps",
                    "linear_acceleration",
                    "angular_velocity",
                    "steering_percentage",
                )
            },
            produce_hint=_PARSE_HINT,
        ),
        ArtifactSpec(
            # agent_tracks = sample_annotation + instance + category 3종 세트.
            name="agent_tracks",
            scope="clip",
            required_files=(
                "parsed/sample_annotation.json",
                "parsed/instance.json",
                "parsed/category.json",
            ),
            provenances=("apollo_gt",),  # 예약: "bevfusion"
            axis="agents",
            required_keys={
                "parsed/sample_annotation.json": (
                    "sample_token",
                    "instance_token",
                    "translation",
                    "rotation",
                    "size",
                ),
                "parsed/instance.json": ("token", "category_token"),
                "parsed/category.json": ("token", "name"),
            },
            produce_hint=_PARSE_HINT,
        ),
        ArtifactSpec(
            name="route",
            scope="clip",
            required_files=("parsed/route.json",),
            required_keys={"parsed/route.json": ("route_lane_ids", "t0_index")},
            produce_hint=_PARSE_HINT,
        ),
        ArtifactSpec(
            # log/scene 메타 (클립 식별용 — dataloader가 clip_id/logfile을 읽는다).
            name="scene_log",
            scope="clip",
            required_files=("parsed/scene.json", "parsed/log.json"),
            required_keys={
                "parsed/scene.json": ("token", "name", "log_token"),
                "parsed/log.json": ("token", "logfile"),
            },
            produce_hint=_PARSE_HINT,
        ),
        ArtifactSpec(
            name="map_graph",
            scope="map",
            required_files=("map_graph.json",),
            required_keys={
                "map_graph.json": ("lanes", "crosswalks", "lane_to_roadblock", "proj")
            },
            produce_hint=_MAP_HINT,
        ),
        ArtifactSpec(
            # 예약: prediction 산출물.  생산자 미구현 (data.prediction=None만 유효).
            name="prediction",
            scope="clip",
            required_files=(),
            provenances=(),
            axis="prediction",
            produce_hint="no producer yet — data.prediction must stay None",
        ),
    ]
}


def _validate_data_cfg(data_cfg: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    merged = dict(DEFAULT_DATA_CFG)
    if data_cfg:
        unknown = set(data_cfg) - set(DATA_AXES)
        if unknown:
            raise ContractError(
                f"Unknown data axes {sorted(unknown)} in data config. "
                f"Known axes: {sorted(DATA_AXES)}"
            )
        merged.update(data_cfg)
    for axis, value in merged.items():
        if value not in DATA_AXES[axis]:
            raise ContractError(
                f"Unsupported provenance data.{axis}={value!r}. "
                f"Currently supported: {list(DATA_AXES[axis])}"
            )
    return merged


def _check_keys(path: Path, required: Sequence[str]) -> List[str]:
    """필수 키 누락 목록을 반환 (list json은 첫 row, dict json은 최상위)."""
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return [f"unreadable json ({error})"]
    if isinstance(payload, list):
        if not payload:
            return [f"empty table (expected keys {list(required)})"]
        row = payload[0]
    elif isinstance(payload, dict):
        row = payload
    else:
        return [f"unexpected json type {type(payload).__name__}"]
    return [key for key in required if key not in row]


def check(
    clip_id: str,
    requires: Sequence[str],
    data_cfg: Optional[Dict[str, Any]] = None,
    map_name: Optional[str] = None,
    clip_dir: Optional[Path] = None,
    maps_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """requires의 각 아티팩트가 계약대로 존재하는지 검증한다.

    실패 시 :class:`ContractError` 에 "무엇이 없고 무엇을 돌려야 하는지"를
    전부 모아 담는다 (fail-fast, 자동 생성 없음).

    Parameters
    ----------
    clip_id:   work/<clip_id>/ 규약의 클립 id.
    requires:  모델 dataloader가 선언한 아티팩트 이름 목록.
    data_cfg:  root config의 ``data`` 섹션 (None이면 기본 provenance).
    map_name:  scope="map" 아티팩트 해석에 필요.
    clip_dir:  기본 ``work/<clip_id>`` 대신 쓸 클립 루트 (검증/데모용).
    maps_root: 기본 ``work/maps`` 대신 쓸 맵 루트.
    """
    provenance = _validate_data_cfg(data_cfg)
    clip_root = Path(clip_dir) if clip_dir is not None else DEFAULT_WORK_ROOT / clip_id
    maps_root = Path(maps_root) if maps_root is not None else DEFAULT_WORK_ROOT / "maps"

    problems: List[str] = []
    checked: List[str] = []
    for name in requires:
        spec = ARTIFACTS.get(name)
        if spec is None:
            problems.append(
                f"unknown artifact '{name}' (registry: {sorted(ARTIFACTS)})"
            )
            continue
        if spec.axis is not None:
            selected = provenance.get(spec.axis)
            if selected is None:
                problems.append(
                    f"artifact '{name}' requires data.{spec.axis} to select a "
                    f"producer, but it is None. {spec.produce_hint}"
                )
                continue
            if selected not in spec.provenances:
                problems.append(
                    f"artifact '{name}': provenance '{selected}' not available. "
                    f"Available: {list(spec.provenances)}"
                )
                continue
        if spec.scope == "map":
            if not map_name:
                problems.append(
                    f"artifact '{name}' is map-scoped but no map_name was given."
                )
                continue
            root = maps_root / map_name
        else:
            root = clip_root

        artifact_problems: List[str] = []
        for rel in spec.required_files:
            path = root / rel
            if not path.exists():
                artifact_problems.append(f"missing file: {path}")
                continue
            missing_keys = _check_keys(path, spec.required_keys.get(rel, ()))
            if missing_keys:
                artifact_problems.append(
                    f"schema mismatch in {path}: missing keys {missing_keys}"
                )
        if artifact_problems:
            hint = spec.produce_hint.format(clip_id=clip_id, map_name=map_name or "<map>")
            problems.append(
                f"artifact '{name}' failed:\n    "
                + "\n    ".join(artifact_problems)
                + f"\n    -> {hint}"
            )
        else:
            checked.append(name)

    if problems:
        raise ContractError(
            f"data contract check failed for clip '{clip_id}' "
            f"({len(problems)} problem(s)):\n  "
            + "\n  ".join(problems)
        )
    return {
        "clip_id": clip_id,
        "checked": checked,
        "provenance": provenance,
        "clip_dir": str(clip_root),
        "maps_root": str(maps_root),
    }
