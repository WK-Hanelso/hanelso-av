import json
import math
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
    obstacle_frames: List[Tuple[int, List[Tuple[int, int]]]] = []
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
            obstacle_frames.append(
                (
                    timestamp_ns,
                    [(int(ob.id), int(ob.type)) for ob in msg.perception_obstacle],
                )
            )
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
    obstacle_track_ids = {track_id for _, frame in obstacle_frames for track_id, _ in frame}
    obstacle_type_histogram: Counter = Counter()
    for _, frame in obstacle_frames:
        for _, obstacle_type in frame:
            obstacle_type_histogram[OBSTACLE_TYPE_NAMES.get(obstacle_type, "other")] += 1

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
        "map_match": {
            "matched_map": matched_map["map_name"] if matched_map else None,
            "matched_map_path": matched_map["map_path"] if matched_map else None,
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
    }

    summary["notable"] = {
        "reroute": summary["channel_summaries"]["routing"]["reroute"],
        "reroute_at_sec": summary["channel_summaries"]["routing"]["reroute_at_sec"],
        "reroute_relative_sec": summary["channel_summaries"]["routing"]["reroute_relative_sec"],
        "stop_segment_count": len(stop_segments),
        "max_simultaneous_obstacles": max(obstacle_counts) if obstacle_counts else 0,
        "traffic_light_detected": traffic_light_detected,
        "matched_map": summary["map_match"]["matched_map"],
    }
    return summary


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


def _format_rate(count: Optional[int], rate_hz: Optional[float]) -> str:
    if count is None:
        return "absent"
    return f"{count} msgs, {rate_hz} Hz"


def render_text_report(summary: Dict[str, object]) -> str:
    map_match = summary["map_match"]
    channel_summaries = summary["channel_summaries"]
    notable = summary["notable"]
    lines: List[str] = []
    lines.append(f"Clip: {summary['clip_id']}")
    lines.append(f"Record: {summary['record_path']}")
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
    lines.append("== Notable ==")
    lines.append(f"matched_map={notable['matched_map']}")
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
