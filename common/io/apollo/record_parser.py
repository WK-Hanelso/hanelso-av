import uuid
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from cyber_record.record import Record

from common.io.base import EgoPoseProvider, SourceParser
from common.io.registry import register_parser
from common.io.schema import (
    CategoryRow,
    EgoDynamicsRow,
    EgoPoseRow,
    InstanceRow,
    LogRow,
    SampleAnnotationRow,
    SampleRow,
    SceneRow,
    write_tables,
)

CHASSIS_TOPIC = "/apollo/canbus/chassis"
OBSTACLES_TOPIC = "/apollo/perception/obstacles"

APOLLO_CATEGORY_MAP = {
    3: "pedestrian",
    4: "bicycle",
    5: "vehicle",
}


def _token() -> str:
    return uuid.uuid4().hex


def _yaw_to_quaternion(yaw: float) -> List[float]:
    from math import cos, sin

    half = yaw * 0.5
    return [cos(half), 0.0, 0.0, sin(half)]


def _nearest_index(timestamps: List[int], target: int) -> int:
    if not timestamps:
        raise ValueError("Nearest lookup requested on empty timestamp list.")
    index = bisect_left(timestamps, target)
    if index == 0:
        return 0
    if index == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[index - 1]
    after = timestamps[index]
    if target - before <= after - target:
        return index - 1
    return index


def _extract_obstacle(obstacle) -> Dict[str, object]:
    return {
        "id": int(obstacle.id),
        "type": int(obstacle.type),
        "translation": [
            float(obstacle.position.x),
            float(obstacle.position.y),
            float(obstacle.position.z),
        ],
        "rotation": _yaw_to_quaternion(float(obstacle.theta)),
        "size": [
            float(obstacle.width),
            float(obstacle.length),
            float(obstacle.height),
        ],
    }


def _derive_vehicle_and_date(record_path: str) -> Tuple[str, str]:
    path = Path(record_path)
    vehicle = path.parent.name
    stem = path.name.split(".record.")[0]
    if len(stem) == 14:
        date_captured = f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}"
    else:
        date_captured = ""
    return vehicle, date_captured


class ApolloRecordParser(SourceParser):
    def parse(
        self,
        record_path: str,
        out_dir: str,
        clip_id: str,
        pose_provider: EgoPoseProvider,
    ) -> Dict[str, object]:
        pose_provider.prepare(record_path)

        chassis_rows: List[Tuple[int, float, float]] = []
        obstacle_frames: List[Tuple[int, List[Dict[str, object]]]] = []

        for topic, msg, timestamp_ns in Record(record_path).read_messages():
            if topic == CHASSIS_TOPIC:
                chassis_rows.append(
                    (
                        int(timestamp_ns),
                        float(msg.speed_mps),
                        float(msg.steering_percentage),
                    )
                )
            elif topic == OBSTACLES_TOPIC:
                obstacle_frames.append(
                    (
                        int(timestamp_ns),
                        [_extract_obstacle(obstacle) for obstacle in msg.perception_obstacle],
                    )
                )

        if not obstacle_frames:
            raise ValueError("No obstacle frames found; cannot build keyframe timeline.")

        chassis_timestamps = [row[0] for row in chassis_rows]
        category_tokens = {
            "pedestrian": _token(),
            "bicycle": _token(),
            "vehicle": _token(),
            "unknown": _token(),
        }
        categories = [
            CategoryRow(token=token, name=name, description=f"Apollo obstacle class: {name}")
            for name, token in category_tokens.items()
        ]

        log_token = _token()
        scene_token = _token()
        samples: List[SampleRow] = []
        ego_poses: List[EgoPoseRow] = []
        ego_dynamics: List[EgoDynamicsRow] = []
        annotations: List[SampleAnnotationRow] = []
        instance_token_by_obstacle_id: Dict[int, str] = {}
        instance_category_by_obstacle_id: Dict[int, str] = {}
        annotation_tokens_by_obstacle_id: Dict[int, List[str]] = defaultdict(list)

        for frame_index, (timestamp_ns, obstacles) in enumerate(obstacle_frames):
            sample_token = _token()
            prev_token = samples[-1].token if samples else ""
            sample = SampleRow(
                token=sample_token,
                timestamp=timestamp_ns,
                scene_token=scene_token,
                prev=prev_token,
                next="",
            )
            if samples:
                samples[-1].next = sample_token
            samples.append(sample)

            pose = pose_provider.pose_at(timestamp_ns)
            ego_pose_token = _token()
            ego_poses.append(
                EgoPoseRow(
                    token=ego_pose_token,
                    timestamp=timestamp_ns,
                    translation=[float(value) for value in pose["translation"]],
                    rotation=[float(value) for value in pose["rotation"]],
                )
            )

            speed_mps = 0.0
            steering_percentage = 0.0
            if chassis_timestamps:
                chassis_index = _nearest_index(chassis_timestamps, timestamp_ns)
                _, speed_mps, steering_percentage = chassis_rows[chassis_index]
            ego_dynamics.append(
                EgoDynamicsRow(
                    token=_token(),
                    sample_token=sample_token,
                    ego_pose_token=ego_pose_token,
                    timestamp=timestamp_ns,
                    speed_mps=speed_mps,
                    steering_percentage=steering_percentage,
                )
            )

            for obstacle in obstacles:
                obstacle_id = int(obstacle["id"])
                category_name = APOLLO_CATEGORY_MAP.get(int(obstacle["type"]), "unknown")
                instance_token = instance_token_by_obstacle_id.setdefault(obstacle_id, _token())
                instance_category_by_obstacle_id.setdefault(obstacle_id, category_name)
                annotation_token = _token()
                annotation_tokens_by_obstacle_id[obstacle_id].append(annotation_token)
                annotations.append(
                    SampleAnnotationRow(
                        token=annotation_token,
                        sample_token=sample_token,
                        instance_token=instance_token,
                        visibility_token="",
                        attribute_tokens=[],
                        translation=list(obstacle["translation"]),
                        size=list(obstacle["size"]),
                        rotation=list(obstacle["rotation"]),
                        prev="",
                        next="",
                        num_lidar_pts=0,
                        num_radar_pts=0,
                    )
                )

        annotation_lookup = {annotation.token: annotation for annotation in annotations}
        for token_list in annotation_tokens_by_obstacle_id.values():
            for index, token_value in enumerate(token_list):
                annotation = annotation_lookup[token_value]
                annotation.prev = token_list[index - 1] if index > 0 else ""
                annotation.next = token_list[index + 1] if index + 1 < len(token_list) else ""

        instances = [
            InstanceRow(
                token=instance_token,
                category_token=category_tokens[instance_category_by_obstacle_id[obstacle_id]],
                nbr_annotations=len(annotation_tokens_by_obstacle_id[obstacle_id]),
                first_annotation_token=annotation_tokens_by_obstacle_id[obstacle_id][0],
                last_annotation_token=annotation_tokens_by_obstacle_id[obstacle_id][-1],
            )
            for obstacle_id, instance_token in sorted(instance_token_by_obstacle_id.items())
        ]

        vehicle, date_captured = _derive_vehicle_and_date(record_path)
        log = [
            LogRow(
                token=log_token,
                logfile=record_path,
                vehicle=vehicle,
                date_captured=date_captured,
                location="",
            )
        ]
        scene = [
            SceneRow(
                token=scene_token,
                name=clip_id,
                log_token=log_token,
                nbr_samples=len(samples),
                first_sample_token=samples[0].token,
                last_sample_token=samples[-1].token,
            )
        ]

        tables = {
            "log": log,
            "scene": scene,
            "sample": samples,
            "ego_pose": ego_poses,
            "sample_annotation": annotations,
            "instance": instances,
            "category": categories,
            "sensor": [],
            "calibrated_sensor": [],
            "sample_data": [],
            "ego_dynamics": ego_dynamics,
            "language": {"frame_text": [], "object_text": []},
        }
        write_tables(out_dir, tables)
        return {
            "clip_id": clip_id,
            "n_samples": len(samples),
            "tables": {
                name: len(rows) if isinstance(rows, list) else sum(len(v) for v in rows.values())
                for name, rows in tables.items()
            },
        }


register_parser("apollo_record", ApolloRecordParser)
