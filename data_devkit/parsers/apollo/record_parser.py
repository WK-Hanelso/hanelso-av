import uuid
from bisect import bisect_left
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cyber_record.record import Record

from data_devkit.parsers.base import EgoPoseProvider, SourceParser
from data_devkit.parsers.config import ParseConfig
from data_devkit.parsers.registry import register_parser
from data_devkit.parsers.schema import (
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

POSE_TOPIC = "/apollo/localization/pose"
CHASSIS_TOPIC = "/apollo/canbus/chassis"
IMU_TOPIC = "/apollo/sensor/gnss/imu"
OBSTACLES_TOPIC = "/apollo/perception/obstacles"
ROUTING_TOPICS = {"/apollo/routing_response", "/apollo/routing_response_history"}

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


def _downsample_pose_timestamps(timestamps: List[int], keyframe_hz: float) -> List[int]:
    if keyframe_hz <= 0.0:
        raise ValueError(f"keyframe_hz must be > 0, got {keyframe_hz}")
    if not timestamps:
        return []
    interval_ns = int(round(1_000_000_000.0 / keyframe_hz))
    selected = [timestamps[0]]
    next_target = timestamps[0] + interval_ns
    while True:
        next_index = bisect_left(timestamps, next_target)
        if next_index >= len(timestamps):
            break
        selected.append(timestamps[next_index])
        next_target = timestamps[next_index] + interval_ns
    return selected


def _derive_vehicle_and_date(record_path: str) -> Tuple[str, str]:
    path = Path(record_path)
    vehicle = path.parent.name
    stem = path.name.split(".record.")[0]
    if len(stem) == 14:
        date_captured = f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}"
    else:
        date_captured = ""
    return vehicle, date_captured


def _extract_route_sequence(routing_msg) -> List[str]:
    sequence: List[str] = []
    for road in getattr(routing_msg, "road", []):
        for passage in getattr(road, "passage", []):
            for segment in getattr(passage, "segment", []):
                lane_id = getattr(segment, "id", "")
                if lane_id:
                    sequence.append(str(lane_id))
    return sequence


def _extract_destination_xy(routing_msg) -> Optional[List[float]]:
    routing_request = getattr(routing_msg, "routing_request", None)
    if routing_request is None:
        return None
    waypoints = getattr(routing_request, "waypoint", [])
    if not waypoints:
        return None
    pose = getattr(waypoints[-1], "pose", None)
    if pose is None:
        return None
    return [float(pose.x), float(pose.y)]


def _polyline_length(points_xy: List[List[float]]) -> float:
    if len(points_xy) < 2:
        return 0.0
    total = 0.0
    for p0, p1 in zip(points_xy, points_xy[1:]):
        total += math.dist(p0, p1)
    return total


def _select_default_t0_index(
    sample_timestamps: List[int],
    ego_positions_xy: List[List[float]],
    ref_steps: int = 120,
    ref_spacing: float = 1.0,
    hist_steps: int = 21,
) -> Tuple[int, str]:
    if len(sample_timestamps) < hist_steps:
        raise ValueError(
            f"Need at least {hist_steps} parsed samples to select default t0, "
            f"got {len(sample_timestamps)}."
        )
    best_index = hist_steps - 1
    for index in range(hist_steps - 1, len(sample_timestamps)):
        remaining = _polyline_length(ego_positions_xy[index:])
        if remaining >= ref_steps * ref_spacing - 1e-3:
            return index, "default_full_reference"
        best_index = index
    return best_index, "default_partial_reference"


def _write_route_payload(
    out_dir: str,
    routing_events: List[Dict[str, object]],
    t0_timestamp_ns: int,
    t0_index: int,
    t0_reason: str,
) -> None:
    route_path = Path(out_dir) / "route.json"
    if not routing_events:
        payload = {
            "route_lane_ids": [],
            "destination_xy": None,
            "t0_timestamp_ns": int(t0_timestamp_ns),
            "t0_index": int(t0_index),
            "t0_reason": t0_reason,
            "selected_event_timestamp_ns": None,
            "selected_event_topic": None,
            "reroute_events": [],
            "all_sequences": [],
        }
        route_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return

    ordered = sorted(routing_events, key=lambda item: int(item["timestamp_ns"]))
    unique_sequences: List[Dict[str, object]] = []
    reroute_events: List[Dict[str, object]] = []
    last_sequence: Optional[Tuple[str, ...]] = None
    for event in ordered:
        sequence_tuple = tuple(event["lane_ids"])
        current = {
            "topic": str(event["topic"]),
            "timestamp_ns": int(event["timestamp_ns"]),
            "lane_ids": list(sequence_tuple),
            "destination_xy": event["destination_xy"],
        }
        if sequence_tuple != last_sequence:
            unique_sequences.append(current)
            if last_sequence is not None:
                reroute_events.append(current)
            last_sequence = sequence_tuple

    eligible = [
        event for event in ordered if int(event["timestamp_ns"]) <= int(t0_timestamp_ns)
    ]
    preferred = [
        event for event in eligible if str(event["topic"]) == "/apollo/routing_response"
    ]
    selected = preferred[-1] if preferred else (eligible[-1] if eligible else None)
    if selected is None:
        selected = ordered[0]

    payload = {
        "route_lane_ids": list(selected["lane_ids"]),
        "destination_xy": selected["destination_xy"],
        "t0_timestamp_ns": int(t0_timestamp_ns),
        "t0_index": int(t0_index),
        "t0_reason": t0_reason,
        "selected_event_timestamp_ns": int(selected["timestamp_ns"]),
        "selected_event_topic": str(selected["topic"]),
        "reroute_events": reroute_events,
        "all_sequences": unique_sequences,
    }
    route_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


class ApolloRecordParser(SourceParser):
    def parse(
        self,
        record_path: str,
        out_dir: str,
        clip_id: str,
        pose_provider: EgoPoseProvider,
        config: ParseConfig,
    ) -> Dict[str, object]:
        pose_timestamps: List[int] = []
        chassis_rows: List[Tuple[int, float, float]] = []
        imu_rows: List[Tuple[int, List[float], List[float]]] = []
        obstacle_frames: List[Tuple[int, List[Dict[str, object]]]] = []
        routing_events: List[Dict[str, object]] = []

        for topic, msg, timestamp_ns in Record(record_path).read_messages():
            if topic == POSE_TOPIC:
                pose_timestamps.append(int(timestamp_ns))
            elif topic == CHASSIS_TOPIC:
                chassis_rows.append(
                    (
                        int(timestamp_ns),
                        float(msg.speed_mps),
                        float(msg.steering_percentage),
                    )
                )
            elif topic == IMU_TOPIC:
                imu_rows.append(
                    (
                        int(timestamp_ns),
                        [
                            float(msg.linear_acceleration.x),
                            float(msg.linear_acceleration.y),
                            float(msg.linear_acceleration.z),
                        ],
                        [
                            float(msg.angular_velocity.x),
                            float(msg.angular_velocity.y),
                            float(msg.angular_velocity.z),
                        ],
                    )
                )
            elif topic == OBSTACLES_TOPIC:
                obstacle_frames.append(
                    (
                        int(timestamp_ns),
                        [_extract_obstacle(obstacle) for obstacle in msg.perception_obstacle],
                    )
                )
            elif topic in ROUTING_TOPICS:
                lane_ids = _extract_route_sequence(msg)
                if lane_ids:
                    routing_events.append(
                        {
                            "topic": topic,
                            "timestamp_ns": int(timestamp_ns),
                            "lane_ids": lane_ids,
                            "destination_xy": _extract_destination_xy(msg),
                        }
                    )

        missing_topics: List[str] = []
        if not pose_timestamps:
            missing_topics.append(POSE_TOPIC)
        if not chassis_rows:
            missing_topics.append(CHASSIS_TOPIC)
        if not imu_rows:
            missing_topics.append(IMU_TOPIC)
        if missing_topics:
            raise ValueError(f"missing required topic(s): {', '.join(missing_topics)}")

        if config.keyframe == "pose":
            keyframe_timestamps = _downsample_pose_timestamps(pose_timestamps, config.keyframe_hz)
        elif config.keyframe == "obstacles":
            if not obstacle_frames:
                raise ValueError(
                    f"missing required topic(s): {OBSTACLES_TOPIC} (required when keyframe='obstacles')"
                )
            keyframe_timestamps = [timestamp_ns for timestamp_ns, _ in obstacle_frames]
        else:
            raise ValueError(
                f"unsupported keyframe source: {config.keyframe!r} "
                "(expected 'pose' or 'obstacles')"
            )

        pose_provider.prepare(record_path)

        chassis_timestamps = [row[0] for row in chassis_rows]
        imu_timestamps = [row[0] for row in imu_rows]
        obstacle_timestamps = [row[0] for row in obstacle_frames]
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

        for timestamp_ns in keyframe_timestamps:
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

            chassis_index = _nearest_index(chassis_timestamps, timestamp_ns)
            _, speed_mps, steering_percentage = chassis_rows[chassis_index]
            imu_index = _nearest_index(imu_timestamps, timestamp_ns)
            _, linear_acceleration, angular_velocity = imu_rows[imu_index]
            ego_dynamics.append(
                EgoDynamicsRow(
                    token=_token(),
                    sample_token=sample_token,
                    ego_pose_token=ego_pose_token,
                    timestamp=timestamp_ns,
                    speed_mps=speed_mps,
                    steering_percentage=steering_percentage,
                    linear_acceleration=list(linear_acceleration),
                    angular_velocity=list(angular_velocity),
                )
            )

            obstacles: List[Dict[str, object]] = []
            if obstacle_timestamps:
                obstacle_index = _nearest_index(obstacle_timestamps, timestamp_ns)
                _, obstacles = obstacle_frames[obstacle_index]
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
        ego_positions_xy = [row.translation[:2] for row in ego_poses]
        t0_index, t0_reason = _select_default_t0_index(keyframe_timestamps, ego_positions_xy)
        _write_route_payload(
            out_dir=out_dir,
            routing_events=routing_events,
            t0_timestamp_ns=keyframe_timestamps[t0_index],
            t0_index=t0_index,
            t0_reason=t0_reason,
        )
        return {
            "clip_id": clip_id,
            "n_samples": len(samples),
            "tables": {
                name: len(rows) if isinstance(rows, list) else sum(len(v) for v in rows.values())
                for name, rows in tables.items()
            },
        }


register_parser("apollo_record", ApolloRecordParser)
