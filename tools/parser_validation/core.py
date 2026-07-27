import json
import math
import os
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cyber_record.record import Record

POSE_TOPIC = "/apollo/localization/pose"
CHASSIS_TOPIC = "/apollo/canbus/chassis"
IMU_TOPIC = "/apollo/sensor/gnss/imu"
OBSTACLES_TOPIC = "/apollo/perception/obstacles"
PLANNING_TOPIC = "/apollo/planning"

OBSTACLE_TYPE_NAMES = {
    3: "pedestrian",
    4: "bicycle",
    5: "vehicle",
}

OBSTACLE_TYPE_COLORS = {
    "pedestrian": "#d1495b",
    "bicycle": "#edae49",
    "vehicle": "#00798c",
    "other": "#7f8c8d",
}


def default_clip_id(record_path: str) -> str:
    path = Path(record_path)
    stem, suffix = path.name.split(".record.")
    return f"{path.parent.name}_{stem}_{suffix}"


def _ns_to_sec(timestamp_ns: int) -> float:
    return float(timestamp_ns) / 1_000_000_000.0


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _vector_bbox(points: Sequence[Sequence[float]]) -> Optional[Dict[str, float]]:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "center_x": (min(xs) + max(xs)) * 0.5,
        "center_y": (min(ys) + max(ys)) * 0.5,
    }


def _merge_bbox(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if a is None:
        return b
    if b is None:
        return a
    return {
        "min_x": min(a["min_x"], b["min_x"]),
        "max_x": max(a["max_x"], b["max_x"]),
        "min_y": min(a["min_y"], b["min_y"]),
        "max_y": max(a["max_y"], b["max_y"]),
        "center_x": 0.0,
        "center_y": 0.0,
    }


def _finalize_bbox(bbox: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if bbox is None:
        return None
    bbox["center_x"] = (bbox["min_x"] + bbox["max_x"]) * 0.5
    bbox["center_y"] = (bbox["min_y"] + bbox["max_y"]) * 0.5
    for key in ("min_x", "max_x", "min_y", "max_y", "center_x", "center_y"):
        bbox[key] = _round(bbox[key], 3)
    return bbox


def _point_in_bbox(x: float, y: float, bbox: Dict[str, float]) -> bool:
    return bbox["min_x"] <= x <= bbox["max_x"] and bbox["min_y"] <= y <= bbox["max_y"]


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(mean(values))


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _enum_name(message, field_name: str, value: int) -> str:
    try:
        field = message.DESCRIPTOR.fields_by_name[field_name]
        return field.enum_type.values_by_number[int(value)].name
    except Exception:
        return str(value)


def _extract_main_decision_type(planning_msg) -> str:
    decision = getattr(planning_msg, "decision", None)
    if decision is None:
        return "unknown"
    main_decision = getattr(decision, "main_decision", None)
    if main_decision is None:
        return "unknown"
    fields = [field.name for field, _ in main_decision.ListFields()]
    return fields[0] if fields else "none"


def _extract_route_sequence(routing_msg) -> List[str]:
    sequence: List[str] = []
    roads = getattr(routing_msg, "road", [])
    for road in roads:
        passages = getattr(road, "passage", [])
        for passage in passages:
            for segment in getattr(passage, "segment", []):
                lane_id = getattr(segment, "id", "")
                if lane_id:
                    sequence.append(str(lane_id))
    return sequence


def _extract_traffic_light_states(msg) -> Counter:
    states: Counter = Counter()
    lights = getattr(msg, "traffic_light", [])
    for light in lights:
        color_value = getattr(light, "color", None)
        if color_value is None:
            states["UNKNOWN"] += 1
            continue
        states[_enum_name(light, "color", int(color_value))] += 1
    return states


def _compute_stop_segments(chassis_rows: Sequence[Tuple[int, float]]) -> List[Dict[str, float]]:
    if not chassis_rows:
        return []
    segments: List[Dict[str, float]] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    for timestamp_ns, speed_mps in chassis_rows:
        is_stop = speed_mps < 0.1
        if is_stop:
            if current_start is None:
                current_start = timestamp_ns
            current_end = timestamp_ns
        elif current_start is not None and current_end is not None:
            segments.append(
                {
                    "start_sec": _round(_ns_to_sec(current_start), 3),
                    "end_sec": _round(_ns_to_sec(current_end), 3),
                    "duration_sec": _round(_ns_to_sec(current_end - current_start), 3),
                }
            )
            current_start = None
            current_end = None
    if current_start is not None and current_end is not None:
        segments.append(
            {
                "start_sec": _round(_ns_to_sec(current_start), 3),
                "end_sec": _round(_ns_to_sec(current_end), 3),
                "duration_sec": _round(_ns_to_sec(current_end - current_start), 3),
            }
        )
    return segments


def _compute_distance(points: Sequence[Tuple[float, float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    distance = 0.0
    for (x0, y0, z0), (x1, y1, z1) in zip(points, points[1:]):
        distance += math.dist((x0, y0, z0), (x1, y1, z1))
    return distance


def _relative_sec(timestamp_ns: int, base_ns: int) -> float:
    return _round(_ns_to_sec(timestamp_ns - base_ns), 3) or 0.0


def _inventory_entry(count: int, duration_sec: float) -> Dict[str, Optional[float]]:
    rate_hz = float(count) / duration_sec if duration_sec > 0 else None
    return {"count": count, "rate_hz": _round(rate_hz, 3)}


def _nearest_index(timestamps: Sequence[int], target: int) -> int:
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


def _format_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _sample_examples(rows: Sequence[Dict[str, object]], limit: int = 3) -> List[Dict[str, object]]:
    return [dict(row) for row in rows[:limit]]


def _make_check(name: str, passed: bool, detail: Dict[str, object]) -> Dict[str, object]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def _check_timestamp_monotonic(
    name: str,
    timestamps: Sequence[int],
    base_ns: int,
    label: str,
) -> Dict[str, object]:
    violations: List[Dict[str, object]] = []
    for previous, current in zip(timestamps, timestamps[1:]):
        if current <= previous:
            violations.append(
                {
                    "stream": label,
                    "previous_sec": _round(_ns_to_sec(previous), 6),
                    "current_sec": _round(_ns_to_sec(current), 6),
                    "relative_previous_sec": _relative_sec(previous, base_ns),
                    "relative_current_sec": _relative_sec(current, base_ns),
                }
            )
    detail = {
        "stream": label,
        "count": len(timestamps),
        "violation_count": len(violations),
        "violation_ratio": _round(_format_ratio(len(violations), max(len(timestamps) - 1, 1)), 6),
        "examples": _sample_examples(violations),
    }
    return _make_check(name, not violations, detail)


def _compute_consistency_checks(
    pose_rows: Sequence[Tuple[int, float, float, float]],
    chassis_rows: Sequence[Tuple[int, float, float]],
    imu_rows: Sequence[Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]],
    obstacle_frames: Sequence[Tuple[int, List[Dict[str, object]]]],
    first_timestamp_ns: int,
) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []

    pose_timestamps = [timestamp_ns for timestamp_ns, *_ in pose_rows]
    obstacle_timestamps = [timestamp_ns for timestamp_ns, _ in obstacle_frames]
    pose_check = _check_timestamp_monotonic(
        name="timestamp_monotonic",
        timestamps=pose_timestamps,
        base_ns=first_timestamp_ns,
        label=POSE_TOPIC,
    )
    obstacle_check = _check_timestamp_monotonic(
        name="timestamp_monotonic",
        timestamps=obstacle_timestamps,
        base_ns=first_timestamp_ns,
        label=OBSTACLES_TOPIC,
    )
    checks.append(
        _make_check(
            "timestamp_monotonic",
            bool(pose_check["pass"]) and bool(obstacle_check["pass"]),
            {
                "streams": [
                    pose_check["detail"],
                    obstacle_check["detail"],
                ]
            },
        )
    )

    displacement_errors: List[float] = []
    displacement_violations: List[Dict[str, object]] = []
    if pose_rows and chassis_rows:
        chassis_timestamps = [timestamp_ns for timestamp_ns, _, _ in chassis_rows]
        for (ts0, x0, y0, z0), (ts1, x1, y1, z1) in zip(pose_rows, pose_rows[1:]):
            dt = _ns_to_sec(ts1 - ts0)
            if dt <= 0.0:
                continue
            chassis_index0 = _nearest_index(chassis_timestamps, ts0)
            chassis_index1 = _nearest_index(chassis_timestamps, ts1)
            speed0 = chassis_rows[chassis_index0][1]
            speed1 = chassis_rows[chassis_index1][1]
            observed_distance = math.dist((x0, y0, z0), (x1, y1, z1))
            expected_distance = max(0.0, 0.5 * (speed0 + speed1) * dt)
            error = abs(observed_distance - expected_distance)
            tolerance = max(3.0, 0.35 * expected_distance + 0.75)
            displacement_errors.append(error)
            if error > tolerance:
                displacement_violations.append(
                    {
                        "relative_start_sec": _relative_sec(ts0, first_timestamp_ns),
                        "relative_end_sec": _relative_sec(ts1, first_timestamp_ns),
                        "dt_sec": _round(dt, 4),
                        "observed_distance_m": _round(observed_distance, 3),
                        "expected_distance_m": _round(expected_distance, 3),
                        "error_m": _round(error, 3),
                        "tolerance_m": _round(tolerance, 3),
                    }
                )
    displacement_p95 = _percentile(displacement_errors, 0.95)
    displacement_ratio = _format_ratio(len(displacement_violations), max(len(displacement_errors), 1))
    checks.append(
        _make_check(
            "displacement_vs_speed",
            bool(displacement_errors)
            and (displacement_p95 is not None)
            and displacement_p95 <= 3.0
            and displacement_ratio <= 0.05,
            {
                "sample_count": len(displacement_errors),
                "violation_count": len(displacement_violations),
                "violation_ratio": _round(displacement_ratio, 6),
                "abs_error_p95_m": _round(displacement_p95, 3),
                "abs_error_max_m": _round(max(displacement_errors), 3) if displacement_errors else None,
                "examples": _sample_examples(displacement_violations),
            },
        )
    )

    agent_velocity_samples: List[float] = []
    agent_violations: List[Dict[str, object]] = []
    track_history: Dict[int, Tuple[int, float, float]] = {}
    for timestamp_ns, obstacles in obstacle_frames:
        for obstacle in obstacles:
            track_id = int(obstacle["id"])
            x = float(obstacle["x"])
            y = float(obstacle["y"])
            if track_id not in track_history:
                track_history[track_id] = (timestamp_ns, x, y)
                continue
            previous_ts, previous_x, previous_y = track_history[track_id]
            dt = _ns_to_sec(timestamp_ns - previous_ts)
            if dt <= 0.0:
                if math.dist((x, y), (previous_x, previous_y)) > 2.0:
                    agent_violations.append(
                        {
                            "track_id": track_id,
                            "relative_previous_sec": _relative_sec(previous_ts, first_timestamp_ns),
                            "relative_current_sec": _relative_sec(timestamp_ns, first_timestamp_ns),
                            "dt_sec": _round(dt, 4),
                            "jump_m": _round(math.dist((x, y), (previous_x, previous_y)), 3),
                            "limit_m": 2.0,
                        }
                    )
                track_history[track_id] = (timestamp_ns, x, y)
                continue
            jump = math.dist((x, y), (previous_x, previous_y))
            velocity = jump / dt
            agent_velocity_samples.append(velocity)
            limit = 50.0 * dt + 2.0
            if jump > limit:
                agent_violations.append(
                    {
                        "track_id": track_id,
                        "relative_previous_sec": _relative_sec(previous_ts, first_timestamp_ns),
                        "relative_current_sec": _relative_sec(timestamp_ns, first_timestamp_ns),
                        "dt_sec": _round(dt, 4),
                        "jump_m": _round(jump, 3),
                        "speed_mps": _round(velocity, 3),
                        "limit_m": _round(limit, 3),
                    }
                )
            track_history[track_id] = (timestamp_ns, x, y)
    agent_violation_ratio = _format_ratio(len(agent_violations), max(len(agent_velocity_samples), 1))
    checks.append(
        _make_check(
            "agent_no_teleport",
            len(agent_velocity_samples) > 0 and agent_violation_ratio <= 0.01,
            {
                "sample_count": len(agent_velocity_samples),
                "violation_count": len(agent_violations),
                "violation_ratio": _round(agent_violation_ratio, 6),
                "speed_p95_mps": _round(_percentile(agent_velocity_samples, 0.95), 3),
                "speed_max_mps": _round(max(agent_velocity_samples), 3) if agent_velocity_samples else None,
                "examples": _sample_examples(agent_violations),
            },
        )
    )

    bound_violations: List[Dict[str, object]] = []
    for timestamp_ns, speed_mps, _ in chassis_rows:
        if speed_mps >= 40.0:
            bound_violations.append(
                {
                    "stream": CHASSIS_TOPIC,
                    "relative_sec": _relative_sec(timestamp_ns, first_timestamp_ns),
                    "value": _round(speed_mps, 3),
                    "bound": "< 40.0",
                }
            )
    for timestamp_ns, linear_acceleration, _ in imu_rows:
        ax, ay, az = linear_acceleration
        if abs(ax) >= 20.0:
            bound_violations.append(
                {
                    "stream": IMU_TOPIC,
                    "axis": "x",
                    "relative_sec": _relative_sec(timestamp_ns, first_timestamp_ns),
                    "value": _round(ax, 3),
                    "bound": "|x| < 20.0",
                }
            )
        if abs(ay) >= 20.0:
            bound_violations.append(
                {
                    "stream": IMU_TOPIC,
                    "axis": "y",
                    "relative_sec": _relative_sec(timestamp_ns, first_timestamp_ns),
                    "value": _round(ay, 3),
                    "bound": "|y| < 20.0",
                }
            )
        if not (0.0 <= az <= 20.0):
            bound_violations.append(
                {
                    "stream": IMU_TOPIC,
                    "axis": "z",
                    "relative_sec": _relative_sec(timestamp_ns, first_timestamp_ns),
                    "value": _round(az, 3),
                    "bound": "0.0 <= z <= 20.0",
                }
            )
    checks.append(
        _make_check(
            "speed_accel_bounds",
            not bound_violations,
            {
                "chassis_sample_count": len(chassis_rows),
                "imu_sample_count": len(imu_rows),
                "violation_count": len(bound_violations),
                "examples": _sample_examples(bound_violations),
            },
        )
    )
    return checks


def _extract_obstacle_frame(msg) -> List[Dict[str, object]]:
    obstacles: List[Dict[str, object]] = []
    for obstacle in getattr(msg, "perception_obstacle", []):
        obstacle_type = OBSTACLE_TYPE_NAMES.get(int(obstacle.type), "other")
        obstacles.append(
            {
                "id": int(obstacle.id),
                "type": obstacle_type,
                "type_raw": int(obstacle.type),
                "x": float(obstacle.position.x),
                "y": float(obstacle.position.y),
                "z": float(obstacle.position.z),
                "theta": float(obstacle.theta),
                "length": float(obstacle.length),
                "width": float(obstacle.width),
                "height": float(obstacle.height),
            }
        )
    return obstacles


def _select_keyframe(obstacle_frames: Sequence[Tuple[int, List[Dict[str, object]]]]) -> Optional[Dict[str, object]]:
    if not obstacle_frames:
        return None
    keyframe_timestamp, keyframe_obstacles = max(
        obstacle_frames,
        key=lambda item: (len(item[1]), -item[0]),
    )
    return {
        "timestamp_sec": _round(_ns_to_sec(keyframe_timestamp), 3),
        "obstacle_count": len(keyframe_obstacles),
        "obstacles": keyframe_obstacles,
    }


def _route_variants(route_sequences: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not route_sequences:
        return {"current": [], "before": [], "after": []}
    current = route_sequences[-1]["lane_sequence"]
    before = route_sequences[0]["lane_sequence"]
    after = route_sequences[-1]["lane_sequence"]
    return {"current": current, "before": before, "after": after}


def load_map_candidates(maps_root: Path) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for map_graph_path in sorted(maps_root.glob("*/map_graph.json")):
        payload = json.loads(map_graph_path.read_text())
        lane_bboxes: List[Dict[str, object]] = []
        map_bbox: Optional[Dict[str, float]] = None
        for lane in payload.get("lanes", []):
            lane_points: List[Sequence[float]] = []
            for key in ("central", "left", "right"):
                lane_points.extend(lane.get(key, []))
            bbox = _vector_bbox(lane_points)
            if bbox is None:
                continue
            lane_bboxes.append({"lane_id": lane["id"], "bbox": _finalize_bbox(bbox)})
            map_bbox = _merge_bbox(map_bbox, bbox)
        candidates.append(
            {
                "map_name": map_graph_path.parent.name,
                "map_path": str(map_graph_path),
                "proj": payload.get("proj", ""),
                "map_bbox": _finalize_bbox(map_bbox),
                "lane_bboxes": lane_bboxes,
            }
        )
    if not candidates:
        raise ValueError(f"No map candidates found under {maps_root}")
    return candidates


def summarize_record(record_path: str, maps_root: Path) -> Dict[str, object]:
    channel_counts: Dict[str, int] = defaultdict(int)
    first_timestamp_ns: Optional[int] = None
    last_timestamp_ns: Optional[int] = None

    pose_rows: List[Tuple[int, float, float, float]] = []
    chassis_rows: List[Tuple[int, float, float]] = []
    imu_rows: List[Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]] = []
    obstacle_frames: List[Tuple[int, List[Dict[str, object]]]] = []
    planning_count = 0
    planning_sample: Optional[Dict[str, object]] = None
    traffic_light_counts: Dict[str, int] = defaultdict(int)
    traffic_light_states: Counter = Counter()
    traffic_light_detected = False
    routing_events: List[Dict[str, object]] = []

    for topic, msg, timestamp_ns in Record(record_path).read_messages():
        timestamp_ns = int(timestamp_ns)
        channel_counts[topic] += 1
        if first_timestamp_ns is None or timestamp_ns < first_timestamp_ns:
            first_timestamp_ns = timestamp_ns
        if last_timestamp_ns is None or timestamp_ns > last_timestamp_ns:
            last_timestamp_ns = timestamp_ns

        if topic == POSE_TOPIC:
            pose_rows.append(
                (
                    timestamp_ns,
                    float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    float(msg.pose.position.z),
                )
            )
        elif topic == CHASSIS_TOPIC:
            chassis_rows.append(
                (
                    timestamp_ns,
                    float(msg.speed_mps),
                    float(msg.steering_percentage),
                )
            )
        elif topic == IMU_TOPIC:
            imu_rows.append(
                (
                    timestamp_ns,
                    (
                        float(msg.linear_acceleration.x),
                        float(msg.linear_acceleration.y),
                        float(msg.linear_acceleration.z),
                    ),
                    (
                        float(msg.angular_velocity.x),
                        float(msg.angular_velocity.y),
                        float(msg.angular_velocity.z),
                    ),
                )
            )
        elif topic == OBSTACLES_TOPIC:
            obstacle_frames.append((timestamp_ns, _extract_obstacle_frame(msg)))
        elif topic == PLANNING_TOPIC:
            planning_count += 1
            if planning_sample is None:
                planning_sample = {
                    "trajectory_point_count": len(getattr(msg, "trajectory_point", [])),
                    "main_decision": _extract_main_decision_type(msg),
                }

        if "traffic_light" in topic:
            traffic_light_counts[topic] += 1
            states = _extract_traffic_light_states(msg)
            if states:
                traffic_light_detected = True
                traffic_light_states.update(states)

        if "routing" in topic:
            sequence = _extract_route_sequence(msg)
            if sequence:
                routing_events.append(
                    {
                        "topic": topic,
                        "timestamp_ns": timestamp_ns,
                        "sequence": sequence,
                    }
                )

    if first_timestamp_ns is None or last_timestamp_ns is None:
        raise ValueError(f"No messages found in record: {record_path}")

    duration_sec = max(0.0, _ns_to_sec(last_timestamp_ns - first_timestamp_ns))
    pose_points_xyz = [(x, y, z) for _, x, y, z in pose_rows]
    pose_points_xy = [(x, y) for _, x, y, _ in pose_rows]
    ego_bbox = _finalize_bbox(_vector_bbox(pose_points_xy))

    map_candidates = load_map_candidates(maps_root)
    matched_candidates: List[Dict[str, object]] = []
    if ego_bbox is not None:
        center_x = ego_bbox["center_x"]
        center_y = ego_bbox["center_y"]
        for candidate in map_candidates:
            lane_hits = [
                lane["lane_id"]
                for lane in candidate["lane_bboxes"]
                if _point_in_bbox(center_x, center_y, lane["bbox"])
            ]
            if lane_hits:
                matched_candidates.append(
                    {
                        "map_name": candidate["map_name"],
                        "map_path": candidate["map_path"],
                        "proj": candidate["proj"],
                        "map_bbox": candidate["map_bbox"],
                        "lane_match_count": len(lane_hits),
                        "lane_match_examples": lane_hits[:5],
                    }
                )
    matched_candidates.sort(key=lambda item: (-item["lane_match_count"], item["map_name"]))
    matched_map = matched_candidates[0] if matched_candidates else None

    speeds = [speed for _, speed, _ in chassis_rows]
    steerings = [steer for _, _, steer in chassis_rows]
    stop_segments = _compute_stop_segments([(ts, speed) for ts, speed, _ in chassis_rows])

    linear_axes = list(zip(*[linear for _, linear, _ in imu_rows])) if imu_rows else []
    angular_axes = list(zip(*[angular for _, _, angular in imu_rows])) if imu_rows else []

    obstacle_counts = [len(frame) for _, frame in obstacle_frames]
    obstacle_track_ids = {int(obstacle["id"]) for _, frame in obstacle_frames for obstacle in frame}
    obstacle_type_histogram: Counter = Counter()
    for _, frame in obstacle_frames:
        for obstacle in frame:
            obstacle_type_histogram[str(obstacle["type"])] += 1

    routing_sequences: List[Dict[str, object]] = []
    last_sequence: Optional[Tuple[str, ...]] = None
    reroute = False
    reroute_at_ns: Optional[int] = None
    for event in sorted(routing_events, key=lambda item: item["timestamp_ns"]):
        sequence_tuple = tuple(event["sequence"])
        if sequence_tuple == last_sequence:
            continue
        routing_sequences.append(
            {
                "topic": event["topic"],
                "timestamp_sec": _round(_ns_to_sec(event["timestamp_ns"]), 3),
                "relative_sec": _relative_sec(event["timestamp_ns"], first_timestamp_ns),
                "lane_sequence": list(sequence_tuple),
                "lane_sequence_length": len(sequence_tuple),
            }
        )
        if last_sequence is not None and sequence_tuple != last_sequence and reroute_at_ns is None:
            reroute = True
            reroute_at_ns = event["timestamp_ns"]
        last_sequence = sequence_tuple

    checks = _compute_consistency_checks(
        pose_rows=pose_rows,
        chassis_rows=chassis_rows,
        imu_rows=imu_rows,
        obstacle_frames=obstacle_frames,
        first_timestamp_ns=first_timestamp_ns,
    )
    validation_pass = all(check["pass"] for check in checks)

    pose_speed_rows: List[Dict[str, object]] = []
    if pose_rows and chassis_rows:
        chassis_timestamps = [timestamp_ns for timestamp_ns, _, _ in chassis_rows]
        for timestamp_ns, x, y, z in pose_rows:
            chassis_index = _nearest_index(chassis_timestamps, timestamp_ns)
            pose_speed_rows.append(
                {
                    "timestamp_sec": _round(_ns_to_sec(timestamp_ns), 3),
                    "relative_sec": _relative_sec(timestamp_ns, first_timestamp_ns),
                    "x": _round(x, 3),
                    "y": _round(y, 3),
                    "z": _round(z, 3),
                    "speed_mps": _round(chassis_rows[chassis_index][1], 3),
                }
            )

    channel_inventory = {
        topic: _inventory_entry(count, duration_sec) for topic, count in sorted(channel_counts.items())
    }

    summary = {
        "record_path": record_path,
        "clip_id": default_clip_id(record_path),
        "record_window": {
            "start_sec": _round(_ns_to_sec(first_timestamp_ns), 3),
            "end_sec": _round(_ns_to_sec(last_timestamp_ns), 3),
            "duration_sec": _round(duration_sec, 3),
        },
        "validation": {
            "pass": validation_pass,
            "status": "PASS" if validation_pass else "FAIL",
            "failed_checks": [check["name"] for check in checks if not check["pass"]],
        },
        "map_match": {
            "matched_map": matched_map["map_name"] if matched_map else None,
            "matched_map_path": matched_map["map_path"] if matched_map else None,
            "matched_map_proj": matched_map["proj"] if matched_map else None,
            "ego_bbox": ego_bbox,
            "map_bbox": matched_map["map_bbox"] if matched_map else None,
            "lane_match_count": matched_map["lane_match_count"] if matched_map else 0,
            "lane_match_examples": matched_map["lane_match_examples"] if matched_map else [],
            "candidate_results": matched_candidates,
        },
        "channel_inventory": channel_inventory,
        "channel_summaries": {
            "localization_pose": {
                "status": "present" if pose_rows else "absent",
                "count": len(pose_rows),
                "rate_hz": _round(len(pose_rows) / duration_sec, 3) if pose_rows and duration_sec > 0 else None,
                "utm_bbox": ego_bbox,
                "distance_m": _round(_compute_distance(pose_points_xyz), 3),
                "duration_sec": _round(
                    _ns_to_sec(pose_rows[-1][0] - pose_rows[0][0]), 3
                )
                if len(pose_rows) >= 2
                else 0.0,
            },
            "canbus_chassis": {
                "status": "present" if chassis_rows else "absent",
                "count": len(chassis_rows),
                "rate_hz": _round(len(chassis_rows) / duration_sec, 3)
                if chassis_rows and duration_sec > 0
                else None,
                "speed_mps": {
                    "min": _round(min(speeds)) if speeds else None,
                    "max": _round(max(speeds)) if speeds else None,
                    "mean": _round(_safe_mean(speeds)) if speeds else None,
                },
                "steering_percentage": {
                    "min": _round(min(steerings)) if steerings else None,
                    "max": _round(max(steerings)) if steerings else None,
                },
                "stop_segments": stop_segments,
            },
            "gnss_imu": {
                "status": "present" if imu_rows else "absent",
                "count": len(imu_rows),
                "rate_hz": _round(len(imu_rows) / duration_sec, 3) if imu_rows and duration_sec > 0 else None,
                "linear_acceleration": {
                    axis: {
                        "min": _round(min(values)) if values else None,
                        "max": _round(max(values)) if values else None,
                    }
                    for axis, values in zip(("x", "y", "z"), linear_axes)
                },
                "angular_velocity": {
                    axis: {
                        "min": _round(min(values)) if values else None,
                        "max": _round(max(values)) if values else None,
                    }
                    for axis, values in zip(("x", "y", "z"), angular_axes)
                },
            },
            "perception_obstacles": {
                "status": "present" if obstacle_frames else "absent",
                "frame_count": len(obstacle_frames),
                "unique_track_ids": len(obstacle_track_ids),
                "type_histogram": dict(sorted(obstacle_type_histogram.items())),
                "objects_per_frame": {
                    "min": min(obstacle_counts) if obstacle_counts else None,
                    "max": max(obstacle_counts) if obstacle_counts else None,
                    "mean": _round(_safe_mean(obstacle_counts)) if obstacle_counts else None,
                },
            },
            "planning": {
                "status": "present" if planning_count else "absent",
                "count": planning_count,
                "sample": planning_sample,
            },
            "traffic_light": {
                "status": "present" if traffic_light_counts else "absent",
                "channels": dict(sorted(traffic_light_counts.items())),
                "count": sum(traffic_light_counts.values()),
                "detected": traffic_light_detected,
                "state_histogram": dict(sorted(traffic_light_states.items())),
            },
            "routing": {
                "status": "present" if routing_sequences else "absent",
                "channels": sorted({event["topic"] for event in routing_events}),
                "count": len(routing_events),
                "unique_route_sequence_count": len(routing_sequences),
                "route_sequences": routing_sequences,
                "reroute": reroute,
                "reroute_at_sec": _round(_ns_to_sec(reroute_at_ns), 3) if reroute_at_ns else None,
                "reroute_relative_sec": _relative_sec(reroute_at_ns, first_timestamp_ns)
                if reroute_at_ns
                else None,
            },
        },
        "checks": checks,
        "visualization": {
            "coord_system": "UTM52",
            "pose_samples": pose_speed_rows,
            "representative_obstacles": _select_keyframe(obstacle_frames),
            "route_variants": _route_variants(routing_sequences),
            "bev_png": None,
            "bev_route_png": None,
        },
    }

    summary["notable"] = {
        "reroute": summary["channel_summaries"]["routing"]["reroute"],
        "reroute_at_sec": summary["channel_summaries"]["routing"]["reroute_at_sec"],
        "reroute_relative_sec": summary["channel_summaries"]["routing"]["reroute_relative_sec"],
        "stop_segment_count": len(stop_segments),
        "max_simultaneous_obstacles": max(obstacle_counts) if obstacle_counts else 0,
        "traffic_light_detected": traffic_light_detected,
        "matched_map": summary["map_match"]["matched_map"],
        "validation_status": summary["validation"]["status"],
    }
    return summary


def _load_map_payload(map_path: Optional[str]) -> Dict[str, object]:
    if not map_path:
        return {}
    return json.loads(Path(map_path).read_text())


def _map_path_for_name(map_name: str, maps_root: Path) -> Path:
    map_path = maps_root / map_name / "map_graph.json"
    if not map_path.exists():
        raise FileNotFoundError(f"Map graph not found: {map_path}")
    return map_path


def _xy_points(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        result.append((float(point[0]), float(point[1])))
    return result


def _bbox_intersects(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return not (
        a["max_x"] < b["min_x"]
        or a["min_x"] > b["max_x"]
        or a["max_y"] < b["min_y"]
        or a["min_y"] > b["max_y"]
    )


def _expand_bbox(bbox: Optional[Dict[str, float]], margin_m: float) -> Optional[Dict[str, float]]:
    if bbox is None:
        return None
    return {
        "min_x": bbox["min_x"] - margin_m,
        "max_x": bbox["max_x"] + margin_m,
        "min_y": bbox["min_y"] - margin_m,
        "max_y": bbox["max_y"] + margin_m,
        "center_x": bbox["center_x"],
        "center_y": bbox["center_y"],
    }


def _bbox_from_point_groups(point_groups: Sequence[Sequence[Tuple[float, float]]]) -> Optional[Dict[str, float]]:
    merged: Optional[Dict[str, float]] = None
    for points in point_groups:
        bbox = _vector_bbox(points)
        merged = _merge_bbox(merged, bbox)
    return _finalize_bbox(merged)


def _apply_crop(ax, crop_bbox: Optional[Dict[str, float]]) -> None:
    if crop_bbox is None:
        return
    ax.set_xlim(crop_bbox["min_x"], crop_bbox["max_x"])
    ax.set_ylim(crop_bbox["min_y"], crop_bbox["max_y"])


def _lane_geometry(map_payload: Dict[str, object]) -> List[Dict[str, object]]:
    lanes: List[Dict[str, object]] = []
    for lane in map_payload.get("lanes", []):
        left_points = _xy_points(lane.get("left", []))
        right_points = _xy_points(lane.get("right", []))
        center_points = _xy_points(lane.get("central", []))
        polygon_points = left_points + list(reversed(right_points)) if len(left_points) >= 2 and len(left_points) == len(right_points) else []
        bbox = _bbox_from_point_groups([polygon_points, center_points, left_points, right_points])
        if bbox is None:
            continue
        lanes.append(
            {
                "id": str(lane.get("id", "")),
                "center": center_points,
                "polygon": polygon_points,
                "bbox": bbox,
            }
        )
    return lanes


def _crosswalk_geometry(map_payload: Dict[str, object]) -> List[Dict[str, object]]:
    crosswalks: List[Dict[str, object]] = []
    for crosswalk_id, polygon_points in map_payload.get("crosswalks", {}).items():
        polygon = _xy_points(polygon_points)
        bbox = _vector_bbox(polygon)
        if len(polygon) < 3 or bbox is None:
            continue
        crosswalks.append({"id": str(crosswalk_id), "polygon": polygon, "bbox": _finalize_bbox(bbox)})
    return crosswalks


def _signal_geometry(map_payload: Dict[str, object]) -> List[Dict[str, object]]:
    signals: List[Dict[str, object]] = []
    for signal_id, signal in map_payload.get("signals", {}).items():
        stop_line = _xy_points(signal.get("stop_line", []))
        bbox = _vector_bbox(stop_line)
        if len(stop_line) != 2 or bbox is None:
            continue
        midpoint = (
            (stop_line[0][0] + stop_line[1][0]) * 0.5,
            (stop_line[0][1] + stop_line[1][1]) * 0.5,
        )
        signals.append(
            {
                "id": str(signal_id),
                "stop_line": stop_line,
                "midpoint": midpoint,
                "bbox": _finalize_bbox(bbox),
            }
        )
    return signals


def _zone_geometry(map_payload: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[str]]:
    zone_colors = {
        "tunnel": "#6c757d",
        "underpass": "#457b9d",
        "no_auto_driving": "#e76f51",
        "construction": "#f4a261",
        "alleyway": "#2a9d8f",
    }
    zones: List[Dict[str, object]] = []
    warnings: List[str] = []
    raw_zones = map_payload.get("custom_zones", {})
    for zone_type in ("tunnel", "underpass", "no_auto_driving", "construction", "alleyway"):
        raw_polygons = raw_zones.get(zone_type, [])
        if not raw_polygons:
            continue
        if not isinstance(raw_polygons, list):
            warnings.append(f"custom_zones.{zone_type}: expected list, got {type(raw_polygons).__name__}; skipped")
            continue
        for index, raw_polygon in enumerate(raw_polygons):
            if not isinstance(raw_polygon, list):
                warnings.append(
                    f"custom_zones.{zone_type}[{index}]: expected polygon point list, got {type(raw_polygon).__name__}; skipped"
                )
                continue
            polygon = _xy_points(raw_polygon)
            bbox = _vector_bbox(polygon)
            if len(polygon) < 3 or bbox is None:
                warnings.append(f"custom_zones.{zone_type}[{index}]: invalid polygon; skipped")
                continue
            zones.append(
                {
                    "type": zone_type,
                    "color": zone_colors[zone_type],
                    "polygon": polygon,
                    "bbox": _finalize_bbox(bbox),
                }
            )
    return zones, warnings


def _visible_entries(entries: Sequence[Dict[str, object]], crop_bbox: Optional[Dict[str, float]]) -> List[Dict[str, object]]:
    if crop_bbox is None:
        return list(entries)
    visible: List[Dict[str, object]] = []
    for entry in entries:
        bbox = entry.get("bbox")
        if bbox is not None and _bbox_intersects(bbox, crop_bbox):
            visible.append(entry)
    return visible


def _lane_points_by_id(map_payload: Dict[str, object]) -> Dict[str, List[Tuple[float, float]]]:
    return {str(lane["id"]): list(lane["center"]) for lane in _lane_geometry(map_payload)}


def _obstacle_patch_points(obstacle: Dict[str, object], min_length: float = 1.5, min_width: float = 0.8) -> List[Tuple[float, float]]:
    half_length = max(float(obstacle["length"]), min_length) * 0.5
    half_width = max(float(obstacle["width"]), min_width) * 0.5
    theta = float(obstacle["theta"])
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    corners_local = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    corners_world: List[Tuple[float, float]] = []
    for dx, dy in corners_local:
        x = float(obstacle["x"]) + dx * cos_theta - dy * sin_theta
        y = float(obstacle["y"]) + dx * sin_theta + dy * cos_theta
        corners_world.append((x, y))
    return corners_world


def _configure_bev_axes(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#ececec", linewidth=0.5)


def _draw_lane_direction_arrows(ax, lanes: Sequence[Dict[str, object]], crop_bbox: Optional[Dict[str, float]]) -> None:
    for lane in lanes:
        center_points = lane["center"]
        if len(center_points) < 2:
            continue
        segment_index = max(0, (len(center_points) - 1) // 2)
        start = center_points[segment_index]
        end = center_points[min(segment_index + 1, len(center_points) - 1)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 0.5:
            continue
        midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        if crop_bbox is not None and not _point_in_bbox(midpoint[0], midpoint[1], crop_bbox):
            continue
        scale = min(5.0, max(2.2, length * 0.35))
        ax.arrow(
            midpoint[0],
            midpoint[1],
            dx / length * scale,
            dy / length * scale,
            width=0.14,
            head_width=1.1,
            head_length=1.5,
            length_includes_head=True,
            color="#7f8c8d",
            alpha=0.55,
            zorder=2.8,
        )


def _draw_map_background(ax, map_payload: Dict[str, object], crop_bbox: Optional[Dict[str, float]], draw_lane_arrows: bool) -> Dict[str, object]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon

    lanes = _visible_entries(_lane_geometry(map_payload), crop_bbox)
    crosswalks = _visible_entries(_crosswalk_geometry(map_payload), crop_bbox)
    signals = _visible_entries(_signal_geometry(map_payload), crop_bbox)
    zones, warnings = _zone_geometry(map_payload)
    visible_zones = _visible_entries(zones, crop_bbox)

    for lane in lanes:
        polygon_points = lane["polygon"]
        if len(polygon_points) >= 3:
            ax.add_patch(
                Polygon(
                    polygon_points,
                    closed=True,
                    facecolor="#d9d9d9",
                    edgecolor="none",
                    alpha=0.6,
                    zorder=1,
                )
            )
        center_points = lane["center"]
        if len(center_points) >= 2:
            ax.plot(
                [point[0] for point in center_points],
                [point[1] for point in center_points],
                color="#b8b8b8",
                linewidth=0.45,
                alpha=0.7,
                zorder=2,
            )

    for crosswalk in crosswalks:
        ax.add_patch(
            Polygon(
                crosswalk["polygon"],
                closed=True,
                facecolor="#f4a261",
                edgecolor="none",
                alpha=0.5,
                zorder=2.1,
            )
        )

    for signal in signals:
        stop_line = signal["stop_line"]
        midpoint = signal["midpoint"]
        ax.plot(
            [stop_line[0][0], stop_line[1][0]],
            [stop_line[0][1], stop_line[1][1]],
            color="#d62828",
            linewidth=2.3,
            alpha=0.95,
            zorder=2.4,
        )
        ax.scatter([midpoint[0]], [midpoint[1]], color="#d62828", s=12.0, zorder=2.5)

    for zone in visible_zones:
        ax.add_patch(
            Polygon(
                zone["polygon"],
                closed=True,
                facecolor=zone["color"],
                edgecolor="none",
                alpha=0.22,
                zorder=1.8,
            )
        )

    if draw_lane_arrows:
        _draw_lane_direction_arrows(ax, lanes, crop_bbox)

    legend_handles = [
        Patch(facecolor="#d9d9d9", edgecolor="none", alpha=0.6, label="road"),
        Patch(facecolor="#f4a261", edgecolor="none", alpha=0.5, label="crosswalk"),
        Line2D([0], [0], color="#d62828", linewidth=2.3, marker="o", markersize=4, label="signal"),
    ]
    seen_zone_types = set()
    for zone in visible_zones:
        zone_type = str(zone["type"])
        if zone_type in seen_zone_types:
            continue
        seen_zone_types.add(zone_type)
        legend_handles.append(
            Patch(facecolor=zone["color"], edgecolor="none", alpha=0.22, label=f"zone:{zone_type}")
        )
    return {
        "lanes": lanes,
        "crosswalks": crosswalks,
        "signals": signals,
        "zones": visible_zones,
        "warnings": warnings,
        "legend_handles": legend_handles,
    }


def render_map_bev(map_name: str, maps_root: Path) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    map_path = _map_path_for_name(map_name, maps_root)
    map_payload = _load_map_payload(str(map_path))
    fig, ax = plt.subplots(figsize=(24, 24))
    background = _draw_map_background(ax, map_payload, crop_bbox=None, draw_lane_arrows=False)
    zone_counts = Counter(zone["type"] for zone in background["zones"])
    zone_summary = ",".join(f"{zone_type}:{count}" for zone_type, count in sorted(zone_counts.items())) or "none"
    _configure_bev_axes(
        ax,
        (
            f"{map_name} map | lanes={len(background['lanes'])} "
            f"crosswalks={len(background['crosswalks'])} signals={len(background['signals'])} "
            f"zones={zone_summary} | UTM52"
        ),
    )
    ax.legend(handles=background["legend_handles"], loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    out_path = maps_root / map_name / "map_bev.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def render_bev(summary: Dict[str, object], out_dir: Path) -> Dict[str, Optional[str]]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon

    visualization = summary.get("visualization", {})
    pose_samples = visualization.get("pose_samples", [])
    ego_bbox = summary.get("map_match", {}).get("ego_bbox")
    matched_map_path = summary.get("map_match", {}).get("matched_map_path")
    matched_map = summary.get("map_match", {}).get("matched_map")
    route_variants = visualization.get("route_variants", {})
    reroute = summary.get("channel_summaries", {}).get("routing", {}).get("reroute", False)
    keyframe = visualization.get("representative_obstacles")

    map_payload = _load_map_payload(matched_map_path)
    lane_geometries = _lane_geometry(map_payload)
    lane_lookup = {str(lane["id"]): lane for lane in lane_geometries}
    crop_bbox = _expand_bbox(ego_bbox, margin_m=75.0)

    xs = [float(sample["x"]) for sample in pose_samples]
    ys = [float(sample["y"]) for sample in pose_samples]
    speeds = [float(sample["speed_mps"]) for sample in pose_samples]

    def _draw_route(ax) -> Tuple[List[Line2D], Optional[Dict[str, float]]]:
        legend_items: List[Line2D] = []
        route_bbox: Optional[Dict[str, float]] = None
        if reroute:
            route_pairs = [
                ("route before", route_variants.get("before", []), "#ef476f"),
                ("route after", route_variants.get("after", []), "#118ab2"),
            ]
        else:
            route_pairs = [("route", route_variants.get("current", []), "#ef476f")]
        for label, lane_ids, color in route_pairs:
            for lane_id in lane_ids:
                lane = lane_lookup.get(str(lane_id))
                if lane is None:
                    continue
                route_bbox = _merge_bbox(route_bbox, lane["bbox"])
                polygon_points = lane["polygon"]
                if len(polygon_points) >= 3:
                    ax.add_patch(
                        Polygon(
                            polygon_points,
                            closed=True,
                            facecolor=color,
                            edgecolor="none",
                            alpha=0.24,
                            zorder=3.0 if "before" in label else 3.2,
                        )
                    )
                center_points = lane["center"]
                if len(center_points) >= 2:
                    ax.plot(
                        [point[0] for point in center_points],
                        [point[1] for point in center_points],
                        color=color,
                        linewidth=4.0 if "before" in label else 3.2,
                        alpha=0.7,
                        zorder=3.4,
                    )
            legend_items.append(Line2D([0], [0], color=color, linewidth=3.2, label=label))
        return legend_items, _finalize_bbox(route_bbox)

    def _draw_ego(ax):
        if len(xs) < 2:
            return None
        segments = [
            [[xs[index], ys[index]], [xs[index + 1], ys[index + 1]]]
            for index in range(len(xs) - 1)
        ]
        segment_speeds = [0.5 * (speeds[index] + speeds[index + 1]) for index in range(len(speeds) - 1)]
        collection = LineCollection(
            segments,
            cmap="viridis",
            linewidths=2.2,
            zorder=4,
        )
        collection.set_array(segment_speeds)
        ax.add_collection(collection)
        arrow_step = max(1, len(xs) // 18)
        for index in range(0, len(xs) - 1, arrow_step):
            dx = xs[index + 1] - xs[index]
            dy = ys[index + 1] - ys[index]
            length = math.hypot(dx, dy)
            if length < 0.3:
                continue
            scale = min(5.0, max(1.8, length * 0.3))
            ax.arrow(
                xs[index],
                ys[index],
                dx / length * scale,
                dy / length * scale,
                width=0.16,
                head_width=1.2,
                head_length=1.5,
                length_includes_head=True,
                color="#264653",
                alpha=0.75,
                zorder=4.2,
            )
        ax.scatter([xs[0]], [ys[0]], color="#1d3557", s=30.0, marker="o", zorder=4.4)
        ax.scatter([xs[-1]], [ys[-1]], color="#e63946", s=34.0, marker="X", zorder=4.4)
        ax.text(xs[0], ys[0], " S", color="#1d3557", fontsize=9, weight="bold", zorder=4.5)
        ax.text(xs[-1], ys[-1], " E", color="#e63946", fontsize=9, weight="bold", zorder=4.5)
        return collection

    def _draw_obstacles(ax) -> List[Line2D]:
        type_counts: Counter = Counter()
        if not keyframe:
            return []
        for obstacle in keyframe.get("obstacles", []):
            obstacle_type = str(obstacle["type"])
            color = OBSTACLE_TYPE_COLORS.get(obstacle_type, OBSTACLE_TYPE_COLORS["other"])
            type_counts[obstacle_type] += 1
            ax.add_patch(
                Polygon(
                    _obstacle_patch_points(obstacle),
                    closed=True,
                    facecolor=color,
                    edgecolor="#222222",
                    linewidth=0.95,
                    alpha=0.62,
                    zorder=5,
                )
            )
            heading_scale = max(float(obstacle["length"]) * 0.6, 1.2)
            heading_end_x = float(obstacle["x"]) + heading_scale * math.cos(float(obstacle["theta"]))
            heading_end_y = float(obstacle["y"]) + heading_scale * math.sin(float(obstacle["theta"]))
            ax.plot(
                [float(obstacle["x"]), heading_end_x],
                [float(obstacle["y"]), heading_end_y],
                color="#222222",
                linewidth=1.1,
                zorder=6,
            )
        legend_items: List[Line2D] = []
        for obstacle_type, count in sorted(type_counts.items()):
            color = OBSTACLE_TYPE_COLORS.get(obstacle_type, OBSTACLE_TYPE_COLORS["other"])
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="s",
                    linestyle="None",
                    markersize=8,
                    label=f"{obstacle_type} ({count})",
                )
            )
        return legend_items

    def _finalize(ax, title: str, crop: Optional[Dict[str, float]], legend_items: List[object], colorbar_source) -> None:
        _configure_bev_axes(ax, title)
        _apply_crop(ax, crop)
        if legend_items:
            ax.legend(handles=legend_items, loc="upper right", fontsize=8, frameon=True)
        if colorbar_source is not None:
            colorbar = plt.colorbar(colorbar_source, ax=ax, shrink=0.82, pad=0.02)
            colorbar.set_label("ego speed [m/s]")

    bev_path = out_dir / "scene_bev.png"
    fig, ax = plt.subplots(figsize=(14, 14))
    background = _draw_map_background(ax, map_payload, crop_bbox, draw_lane_arrows=True)
    ego_collection = _draw_ego(ax)
    obstacle_legend = _draw_obstacles(ax)
    route_legend, route_bbox = _draw_route(ax)
    legend_items: List[object] = [
        *background["legend_handles"],
        Line2D([0], [0], color="#264653", linewidth=2.2, label="ego trajectory"),
        Patch(facecolor="#00798c", edgecolor="#222222", alpha=0.62, label="agents"),
        *route_legend,
        *obstacle_legend,
    ]
    _finalize(
        ax,
        f"{summary['clip_id']} | map={matched_map} | reroute={reroute} | UTM52",
        crop_bbox,
        legend_items,
        ego_collection,
    )
    fig.tight_layout()
    fig.savefig(bev_path, dpi=180)
    plt.close(fig)

    bev_route_path: Optional[Path] = None
    if reroute:
        bev_route_path = out_dir / "bev_route.png"
        fig, ax = plt.subplots(figsize=(14, 14))
        route_crop = _expand_bbox(route_bbox, margin_m=35.0) if route_bbox is not None else crop_bbox
        route_background = _draw_map_background(ax, map_payload, route_crop, draw_lane_arrows=False)
        route_legend, _ = _draw_route(ax)
        _finalize(
            ax,
            f"{summary['clip_id']} | map={matched_map} | reroute={reroute} | route focus | UTM52",
            route_crop,
            [*route_background["legend_handles"], *route_legend],
            None,
        )
        fig.tight_layout()
        fig.savefig(bev_route_path, dpi=180)
        plt.close(fig)

    return {
        "bev_png": str(bev_path),
        "scene_bev_png": str(bev_path),
        "bev_route_png": str(bev_route_path) if bev_route_path is not None else None,
        "bev_warnings": background["warnings"],
    }


def _format_bbox(bbox: Optional[Dict[str, float]]) -> str:
    if bbox is None:
        return "absent"
    return (
        f"x[{bbox['min_x']}, {bbox['max_x']}], "
        f"y[{bbox['min_y']}, {bbox['max_y']}], "
        f"center=({bbox['center_x']}, {bbox['center_y']})"
    )


def _format_axis_stats(stats: Dict[str, Dict[str, Optional[float]]]) -> str:
    parts = []
    for axis in ("x", "y", "z"):
        axis_stats = stats.get(axis, {})
        parts.append(f"{axis}[{axis_stats.get('min')}, {axis_stats.get('max')}]")
    return ", ".join(parts)


def render_text_report(summary: Dict[str, object]) -> str:
    map_match = summary["map_match"]
    channel_summaries = summary["channel_summaries"]
    notable = summary["notable"]
    lines: List[str] = []
    lines.append(f"Clip: {summary['clip_id']}")
    lines.append(f"Record: {summary['record_path']}")
    lines.append(f"Validation: {summary['validation']['status']}")
    window = summary["record_window"]
    lines.append(
        "Window: "
        f"{window['start_sec']}s -> {window['end_sec']}s "
        f"(duration {window['duration_sec']}s)"
    )
    lines.append("")
    lines.append("== Overview ==")
    lines.append(f"Matched map: {map_match['matched_map']}")
    lines.append(f"Ego bbox: {_format_bbox(map_match['ego_bbox'])}")
    lines.append(f"Matched map bbox: {_format_bbox(map_match['map_bbox'])}")
    if notable["reroute"]:
        lines.append(
            f"Reroute: True (relative {notable['reroute_relative_sec']}s, "
            f"absolute {notable['reroute_at_sec']}s)"
        )
    else:
        lines.append("Reroute: False")
    lines.append(f"Stop segments: {notable['stop_segment_count']}")
    lines.append(f"Max simultaneous obstacles: {notable['max_simultaneous_obstacles']}")
    lines.append(f"Traffic light detected: {notable['traffic_light_detected']}")
    lines.append("")
    lines.append("== Channel Inventory ==")
    for topic, item in summary["channel_inventory"].items():
        lines.append(f"{topic}: {item['count']} msgs, {item['rate_hz']} Hz")
    lines.append("")
    lines.append("== Channel Summaries ==")

    loc = channel_summaries["localization_pose"]
    lines.append("[/apollo/localization/pose]")
    lines.append(
        f"status={loc['status']}, count={loc['count']}, rate={loc['rate_hz']} Hz, "
        f"bbox={_format_bbox(loc['utm_bbox'])}, distance={loc['distance_m']} m, "
        f"duration={loc['duration_sec']} s"
    )

    chassis = channel_summaries["canbus_chassis"]
    lines.append("[/apollo/canbus/chassis]")
    lines.append(
        f"status={chassis['status']}, count={chassis['count']}, rate={chassis['rate_hz']} Hz, "
        f"speed_mps[min={chassis['speed_mps']['min']}, max={chassis['speed_mps']['max']}, "
        f"mean={chassis['speed_mps']['mean']}], steering_percentage[min={chassis['steering_percentage']['min']}, "
        f"max={chassis['steering_percentage']['max']}]"
    )
    if chassis["stop_segments"]:
        lines.append("stop_segments:")
        for segment in chassis["stop_segments"]:
            lines.append(
                f"  - abs {segment['start_sec']}s -> {segment['end_sec']}s "
                f"(duration {segment['duration_sec']}s)"
            )
    else:
        lines.append("stop_segments: none")

    imu = channel_summaries["gnss_imu"]
    lines.append("[/apollo/sensor/gnss/imu]")
    lines.append(
        f"status={imu['status']}, count={imu['count']}, rate={imu['rate_hz']} Hz, "
        f"linear_acceleration={_format_axis_stats(imu['linear_acceleration'])}, "
        f"angular_velocity={_format_axis_stats(imu['angular_velocity'])}"
    )

    obstacles = channel_summaries["perception_obstacles"]
    lines.append("[/apollo/perception/obstacles]")
    lines.append(
        f"status={obstacles['status']}, frames={obstacles['frame_count']}, "
        f"unique_track_ids={obstacles['unique_track_ids']}, type_histogram={obstacles['type_histogram']}, "
        f"objects_per_frame[min={obstacles['objects_per_frame']['min']}, "
        f"max={obstacles['objects_per_frame']['max']}, mean={obstacles['objects_per_frame']['mean']}]"
    )

    planning = channel_summaries["planning"]
    lines.append("[/apollo/planning]")
    lines.append(
        f"status={planning['status']}, count={planning['count']}, sample={planning['sample']}"
    )

    traffic_light = channel_summaries["traffic_light"]
    lines.append("[traffic_light*]")
    lines.append(
        f"status={traffic_light['status']}, count={traffic_light['count']}, "
        f"channels={traffic_light['channels']}, detected={traffic_light['detected']}, "
        f"state_histogram={traffic_light['state_histogram']}"
    )

    routing = channel_summaries["routing"]
    lines.append("[routing*]")
    lines.append(
        f"status={routing['status']}, count={routing['count']}, channels={routing['channels']}, "
        f"reroute={routing['reroute']}, reroute_relative_sec={routing['reroute_relative_sec']}, "
        f"unique_route_sequence_count={routing['unique_route_sequence_count']}"
    )
    if routing["route_sequences"]:
        for sequence in routing["route_sequences"]:
            preview = sequence["lane_sequence"][:8]
            suffix = "..." if sequence["lane_sequence_length"] > len(preview) else ""
            lines.append(
                f"  - {sequence['topic']} @ {sequence['relative_sec']}s "
                f"({sequence['lane_sequence_length']} lanes): {preview}{suffix}"
            )
    else:
        lines.append("  - routing absent")

    lines.append("")
    lines.append("== Consistency Checks ==")
    for check in summary["checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        lines.append(f"[{status}] {check['name']}")
        lines.append(json.dumps(check["detail"], ensure_ascii=False, sort_keys=True))

    lines.append("")
    lines.append("== Notable ==")
    lines.append(f"matched_map={notable['matched_map']}")
    lines.append(f"validation_status={notable['validation_status']}")
    if notable["reroute"]:
        lines.append(
            f"reroute=True at relative {notable['reroute_relative_sec']}s "
            f"(absolute {notable['reroute_at_sec']}s)"
        )
    else:
        lines.append("reroute=False")
    lines.append(f"stop_segment_count={notable['stop_segment_count']}")
    lines.append(f"max_simultaneous_obstacles={notable['max_simultaneous_obstacles']}")
    lines.append(f"traffic_light_detected={notable['traffic_light_detected']}")
    return "\n".join(lines) + "\n"
