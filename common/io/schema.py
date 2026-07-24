import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass
class LogRow:
    token: str
    logfile: str
    vehicle: str
    date_captured: str
    location: str


@dataclass
class SceneRow:
    token: str
    name: str
    log_token: str
    nbr_samples: int
    first_sample_token: str
    last_sample_token: str


@dataclass
class SampleRow:
    token: str
    timestamp: int
    scene_token: str
    prev: str
    next: str


@dataclass
class EgoPoseRow:
    token: str
    timestamp: int
    translation: List[float]
    rotation: List[float]


@dataclass
class EgoDynamicsRow:
    token: str
    sample_token: str
    ego_pose_token: str
    timestamp: int
    speed_mps: float
    steering_percentage: float
    linear_acceleration: List[float]
    angular_velocity: List[float]


@dataclass
class CategoryRow:
    token: str
    name: str
    description: str


@dataclass
class InstanceRow:
    token: str
    category_token: str
    nbr_annotations: int
    first_annotation_token: str
    last_annotation_token: str


@dataclass
class SampleAnnotationRow:
    token: str
    sample_token: str
    instance_token: str
    visibility_token: str
    attribute_tokens: List[str]
    translation: List[float]
    size: List[float]
    rotation: List[float]
    prev: str
    next: str
    num_lidar_pts: int
    num_radar_pts: int


SerializableValue = Union[Dict[str, Any], List[Any]]


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def write_tables(out_dir: str, tables: Dict[str, SerializableValue]) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    for table_name, rows in tables.items():
        file_path = path / f"{table_name}.json"
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(_serialize_value(rows), handle, ensure_ascii=True, indent=2)
