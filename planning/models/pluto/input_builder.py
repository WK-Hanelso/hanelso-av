import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from planning.interface import InputBuilder, register_input_builder

HIST_STEPS = 21
DT = 0.1
RADIUS = 120.0
MAX_AGENTS = 32
REF_STEPS = 120
REF_SPACING = 1.0
MAX_STATIC = 32
MAX_LANES = 80
MAX_CROSSWALKS = 32
LANE_POINTS = 20
CROSSWALK_POINTS = 20
STATIC_SPEED_THRESHOLD_MPS = 0.5
ON_ROUTE_THRESHOLD_M = 5.0
MAP_SELECTION_MARGIN_M = 15.0

PACIFICA_DIMS = (2.297, 5.176)
VEHICLE_DIMENSIONS = {
    "pacifica": PACIFICA_DIMS,
    "e100": PACIFICA_DIMS,
    "e100bt-25": PACIFICA_DIMS,
    "e100bt-22": PACIFICA_DIMS,
    "u100": PACIFICA_DIMS,
}

CATEGORY_CODES = {
    "ego": 0,
    "vehicle": 1,
    "pedestrian": 2,
    "bicycle": 3,
    "unknown": 4,
}


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _quat_to_yaw(quat: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quat)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi:
        return wrapped + 2.0 * math.pi
    return wrapped


def _rotation_matrix(angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)


def _nearest_index(sorted_values: Sequence[float], target: float) -> int:
    index = bisect_left(sorted_values, target)
    if index == 0:
        return 0
    if index >= len(sorted_values):
        return len(sorted_values) - 1
    before = sorted_values[index - 1]
    after = sorted_values[index]
    if abs(target - before) <= abs(after - target):
        return index - 1
    return index


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(points) == 1:
        return np.zeros((1,), dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg)])


def _resample_polyline(points: np.ndarray, sample_count: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((sample_count, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points[:, :2], sample_count, axis=0).astype(np.float32)
    xy = points[:, :2].astype(np.float64)
    cum = _cumulative_lengths(xy)
    total = cum[-1]
    if total <= 1e-6:
        return np.repeat(xy[:1], sample_count, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, sample_count, dtype=np.float64)
    out = np.zeros((sample_count, 2), dtype=np.float64)
    seg_idx = 0
    for i, target in enumerate(targets):
        while seg_idx + 1 < len(cum) and cum[seg_idx + 1] < target:
            seg_idx += 1
        if seg_idx + 1 >= len(cum):
            out[i] = xy[-1]
            continue
        span = cum[seg_idx + 1] - cum[seg_idx]
        if span <= 1e-6:
            out[i] = xy[seg_idx]
            continue
        ratio = (target - cum[seg_idx]) / span
        out[i] = xy[seg_idx] * (1.0 - ratio) + xy[seg_idx + 1] * ratio
    return out.astype(np.float32)


def _resample_by_spacing(points: np.ndarray, max_steps: int, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros((max_steps, 2), dtype=np.float32)
    valid = np.zeros((max_steps,), dtype=bool)
    if len(points) == 0:
        return out, valid
    if len(points) == 1:
        out[0] = points[0, :2]
        valid[0] = True
        return out, valid
    xy = points[:, :2].astype(np.float64)
    cum = _cumulative_lengths(xy)
    total = cum[-1]
    for i in range(max_steps):
        target = float(i) * spacing
        if target > total + 1e-6:
            break
        valid[i] = True
        seg_idx = np.searchsorted(cum, target, side="right") - 1
        seg_idx = max(0, min(seg_idx, len(cum) - 2))
        span = cum[seg_idx + 1] - cum[seg_idx]
        if span <= 1e-6:
            out[i] = xy[seg_idx]
            continue
        ratio = (target - cum[seg_idx]) / span
        out[i] = (xy[seg_idx] * (1.0 - ratio) + xy[seg_idx + 1] * ratio).astype(np.float32)
    return out, valid


def _transform_points(points: np.ndarray, origin_xy: np.ndarray, rot_inv: np.ndarray) -> np.ndarray:
    return ((points[:, :2] - origin_xy) @ rot_inv.T).astype(np.float32)


def _transform_vectors(vectors: np.ndarray, rot_inv: np.ndarray) -> np.ndarray:
    return (vectors @ rot_inv.T).astype(np.float32)


def _min_distance_to_polyline(point_xy: np.ndarray, polyline_xy: np.ndarray) -> float:
    if len(polyline_xy) == 0:
        return float("inf")
    return float(np.linalg.norm(polyline_xy - point_xy[None, :], axis=1).min())


class PlutoInputBuilder(InputBuilder):
    def build(
        self,
        parsed_dir: str,
        map_graph: dict,
        t0_time: float,
        config: dict,
    ) -> dict[str, np.ndarray]:
        parsed_path = Path(parsed_dir)
        tables = self._load_tables(parsed_path)
        dataset = self._prepare_dataset(tables, config)
        t0_index, t0_reason = self._select_t0_index(dataset, t0_time)
        feed, context = self._build_feed(dataset, map_graph, t0_index, t0_reason, config)
        self.last_context = context
        return feed

    def _load_tables(self, parsed_path: Path) -> Dict[str, object]:
        names = [
            "sample",
            "ego_pose",
            "ego_dynamics",
            "sample_annotation",
            "instance",
            "category",
            "scene",
            "log",
        ]
        return {name: _load_json(parsed_path / f"{name}.json") for name in names}

    def _prepare_dataset(self, tables: Dict[str, object], config: dict) -> Dict[str, object]:
        samples = sorted(tables["sample"], key=lambda row: row["timestamp"])
        sample_tokens = [row["token"] for row in samples]
        sample_ts_ns = [int(row["timestamp"]) for row in samples]
        sample_ts_sec = [float(ts) / 1e9 for ts in sample_ts_ns]

        pose_by_token = {row["token"]: row for row in tables["ego_pose"]}
        dynamics_by_sample = {row["sample_token"]: row for row in tables["ego_dynamics"]}
        category_name_by_token = {
            row["token"]: str(row["name"]).lower() for row in tables["category"]
        }
        instance_category = {
            row["token"]: category_name_by_token.get(row["category_token"], "unknown")
            for row in tables["instance"]
        }

        ego_positions = np.zeros((len(samples), 2), dtype=np.float64)
        ego_headings = np.zeros((len(samples),), dtype=np.float64)
        ego_speed = np.zeros((len(samples),), dtype=np.float64)
        ego_accel = np.zeros((len(samples),), dtype=np.float64)
        ego_steering = np.zeros((len(samples),), dtype=np.float64)
        ego_ang_vel = np.zeros((len(samples),), dtype=np.float64)
        sample_pose_tokens: List[str] = []

        for idx, sample in enumerate(samples):
            dyn = dynamics_by_sample[sample["token"]]
            pose = pose_by_token[dyn["ego_pose_token"]]
            ego_positions[idx] = np.array(pose["translation"][:2], dtype=np.float64)
            ego_headings[idx] = _quat_to_yaw(pose["rotation"])
            ego_speed[idx] = float(dyn["speed_mps"])
            acc = np.array(dyn["linear_acceleration"][:2], dtype=np.float64)
            ego_accel[idx] = float(np.linalg.norm(acc))
            ego_steering[idx] = float(dyn["steering_percentage"])
            ego_ang_vel[idx] = float(dyn["angular_velocity"][2])
            sample_pose_tokens.append(pose["token"])

        ann_by_sample: Dict[str, List[dict]] = {}
        track_annotations: Dict[str, List[dict]] = {}
        for ann in tables["sample_annotation"]:
            ann_by_sample.setdefault(ann["sample_token"], []).append(ann)
            track_annotations.setdefault(ann["instance_token"], []).append(ann)

        track_data: Dict[str, dict] = {}
        for instance_token, ann_list in track_annotations.items():
            ordered = sorted(
                ann_list, key=lambda row: sample_ts_ns[sample_tokens.index(row["sample_token"])]
            )
            positions = np.array([row["translation"][:2] for row in ordered], dtype=np.float64)
            timestamps = np.array(
                [sample_ts_sec[sample_tokens.index(row["sample_token"])] for row in ordered],
                dtype=np.float64,
            )
            headings = np.array([_quat_to_yaw(row["rotation"]) for row in ordered], dtype=np.float64)
            widths = np.array([float(row["size"][0]) for row in ordered], dtype=np.float64)
            lengths = np.array([float(row["size"][1]) for row in ordered], dtype=np.float64)
            heights = np.array(
                [
                    float(row["size"][2]) if len(row["size"]) > 2 else 1.8
                    for row in ordered
                ],
                dtype=np.float64,
            )
            velocities = np.zeros((len(ordered), 2), dtype=np.float64)
            if len(ordered) >= 2:
                for idx in range(len(ordered)):
                    if idx == 0:
                        delta = positions[1] - positions[0]
                        dt_sec = max(timestamps[1] - timestamps[0], 1e-6)
                    else:
                        delta = positions[idx] - positions[idx - 1]
                        dt_sec = max(timestamps[idx] - timestamps[idx - 1], 1e-6)
                    velocities[idx] = delta / dt_sec
            sample_to_index = {row["sample_token"]: idx for idx, row in enumerate(ordered)}
            track_data[instance_token] = {
                "sample_to_index": sample_to_index,
                "positions": positions,
                "headings": headings,
                "widths": widths,
                "lengths": lengths,
                "heights": heights,
                "velocities": velocities,
                "category": instance_category.get(instance_token, "unknown"),
            }

        clip_id = str(config.get("clip_id") or tables["scene"][0]["name"])
        vehicle_key = str(config.get("vehicle", "pacifica")).lower()
        ego_dims = VEHICLE_DIMENSIONS.get(vehicle_key, PACIFICA_DIMS)

        return {
            "samples": samples,
            "sample_tokens": sample_tokens,
            "sample_ts_sec": sample_ts_sec,
            "ego_positions": ego_positions,
            "ego_headings": ego_headings,
            "ego_speed": ego_speed,
            "ego_accel": ego_accel,
            "ego_steering": ego_steering,
            "ego_ang_vel": ego_ang_vel,
            "sample_pose_tokens": sample_pose_tokens,
            "ann_by_sample": ann_by_sample,
            "track_data": track_data,
            "clip_id": clip_id,
            "ego_dims": ego_dims,
            "vehicle_key": vehicle_key,
            "logfile": tables["log"][0]["logfile"],
        }

    def _select_t0_index(self, dataset: Dict[str, object], t0_time: float) -> Tuple[int, str]:
        sample_ts_sec = dataset["sample_ts_sec"]
        if len(sample_ts_sec) < HIST_STEPS:
            raise ValueError(
                f"Need at least {HIST_STEPS} samples, got {len(sample_ts_sec)}."
            )
        if t0_time is not None:
            index = _nearest_index(sample_ts_sec, float(t0_time))
            if index < HIST_STEPS - 1:
                raise ValueError(
                    f"Requested t0_time={t0_time:.3f}s maps to sample index {index}, "
                    f"but at least {HIST_STEPS - 1} history samples are required."
                )
            return index, "config_t0_time"

        ego_positions = dataset["ego_positions"]
        best_index = HIST_STEPS - 1
        best_with_full_ref = None
        for index in range(HIST_STEPS - 1, len(sample_ts_sec)):
            future = ego_positions[index:]
            if len(future) < 2:
                remaining = 0.0
            else:
                remaining = _polyline_length(future)
            if remaining >= REF_STEPS * REF_SPACING - 1e-3:
                best_with_full_ref = index
                break
            best_index = index
        if best_with_full_ref is not None:
            return best_with_full_ref, "default_full_reference"
        return best_index, "default_partial_reference"

    def _history_target_indices(self, sample_ts_sec: Sequence[float], t0_index: int) -> Tuple[np.ndarray, np.ndarray]:
        t0_sec = sample_ts_sec[t0_index]
        targets = np.array(
            [t0_sec - (HIST_STEPS - 1 - step) * DT for step in range(HIST_STEPS)],
            dtype=np.float64,
        )
        indices = np.array([_nearest_index(sample_ts_sec, target) for target in targets], dtype=np.int32)
        deltas = np.array([abs(sample_ts_sec[idx] - target) for idx, target in zip(indices, targets)], dtype=np.float64)
        return indices, deltas

    def _build_feed(
        self,
        dataset: Dict[str, object],
        map_graph: dict,
        t0_index: int,
        t0_reason: str,
        config: dict,
    ) -> Tuple[dict[str, np.ndarray], Dict[str, object]]:
        sample_tokens = dataset["sample_tokens"]
        sample_ts_sec = dataset["sample_ts_sec"]
        history_indices, history_deltas = self._history_target_indices(sample_ts_sec, t0_index)
        origin_xy = dataset["ego_positions"][t0_index].copy()
        origin_z = 0.0
        angle = float(dataset["ego_headings"][t0_index])
        rot_inv = _rotation_matrix(-angle)

        agent_position = np.zeros((MAX_AGENTS, HIST_STEPS, 2), dtype=np.float32)
        agent_heading = np.zeros((MAX_AGENTS, HIST_STEPS), dtype=np.float32)
        agent_velocity = np.zeros((MAX_AGENTS, HIST_STEPS, 2), dtype=np.float32)
        agent_shape = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
        agent_category = np.full((MAX_AGENTS,), CATEGORY_CODES["unknown"], dtype=np.int32)
        agent_valid_mask = np.zeros((MAX_AGENTS, HIST_STEPS), dtype=bool)

        ego_positions_hist = dataset["ego_positions"][history_indices]
        ego_local = _transform_points(ego_positions_hist, origin_xy, rot_inv)
        agent_position[0] = ego_local
        ego_headings_hist = dataset["ego_headings"][history_indices]
        agent_heading[0] = np.array([_wrap_angle(value - angle) for value in ego_headings_hist], dtype=np.float32)
        ego_speed_hist = dataset["ego_speed"][history_indices]
        ego_vectors_global = np.stack(
            [ego_speed_hist * np.cos(ego_headings_hist), ego_speed_hist * np.sin(ego_headings_hist)],
            axis=1,
        )
        agent_velocity[0] = _transform_vectors(ego_vectors_global, rot_inv)
        agent_shape[0] = np.array(dataset["ego_dims"], dtype=np.float32)
        agent_category[0] = CATEGORY_CODES["ego"]
        agent_valid_mask[0] = history_deltas <= (DT * 0.6)

        t0_sample_token = sample_tokens[t0_index]
        candidates: List[Tuple[float, str]] = []
        static_candidates: List[Tuple[float, str]] = []
        for instance_token, track in dataset["track_data"].items():
            sample_to_index = track["sample_to_index"]
            if t0_sample_token not in sample_to_index:
                continue
            ann_idx = sample_to_index[t0_sample_token]
            position = track["positions"][ann_idx]
            distance = float(np.linalg.norm(position - origin_xy))
            if distance > RADIUS + MAP_SELECTION_MARGIN_M:
                continue
            speed = float(np.linalg.norm(track["velocities"][ann_idx]))
            if speed <= STATIC_SPEED_THRESHOLD_MPS:
                static_candidates.append((distance, instance_token))
            else:
                candidates.append((distance, instance_token))
        candidates.sort(key=lambda item: item[0])
        static_candidates.sort(key=lambda item: item[0])

        selected_agents = [instance_token for _, instance_token in candidates[: MAX_AGENTS - 1]]
        selected_static = [
            instance_token
            for _, instance_token in static_candidates
            if instance_token not in selected_agents
        ][:MAX_STATIC]

        for agent_slot, instance_token in enumerate(selected_agents, start=1):
            track = dataset["track_data"][instance_token]
            sample_to_index = track["sample_to_index"]
            valid_steps: List[int] = []
            for hist_slot, sample_index in enumerate(history_indices):
                sample_token = sample_tokens[sample_index]
                ann_idx = sample_to_index.get(sample_token)
                if ann_idx is None:
                    continue
                position = track["positions"][ann_idx : ann_idx + 1]
                agent_position[agent_slot, hist_slot] = _transform_points(position, origin_xy, rot_inv)[0]
                agent_heading[agent_slot, hist_slot] = _wrap_angle(track["headings"][ann_idx] - angle)
                velocity = _transform_vectors(track["velocities"][ann_idx : ann_idx + 1], rot_inv)[0]
                agent_velocity[agent_slot, hist_slot] = velocity
                agent_valid_mask[agent_slot, hist_slot] = True
                valid_steps.append(ann_idx)
            if valid_steps:
                last_idx = valid_steps[-1]
                agent_shape[agent_slot] = np.array(
                    [track["widths"][last_idx], track["lengths"][last_idx]], dtype=np.float32
                )
            agent_category[agent_slot] = CATEGORY_CODES.get(track["category"], CATEGORY_CODES["unknown"])

        static_position = np.zeros((MAX_STATIC, 2), dtype=np.float32)
        static_heading = np.zeros((MAX_STATIC,), dtype=np.float32)
        static_shape = np.zeros((MAX_STATIC, 2), dtype=np.float32)
        static_valid_mask = np.zeros((MAX_STATIC,), dtype=bool)
        for static_slot, instance_token in enumerate(selected_static):
            track = dataset["track_data"][instance_token]
            ann_idx = track["sample_to_index"][t0_sample_token]
            static_position[static_slot] = _transform_points(
                track["positions"][ann_idx : ann_idx + 1], origin_xy, rot_inv
            )[0]
            static_heading[static_slot] = _wrap_angle(track["headings"][ann_idx] - angle)
            static_shape[static_slot] = np.array(
                [track["widths"][ann_idx], track["lengths"][ann_idx]], dtype=np.float32
            )
            static_valid_mask[static_slot] = True

        ref_position, ref_vector, ref_orientation, ref_valid_mask = self._build_reference_line(
            dataset["ego_positions"][t0_index:],
            origin_xy,
            rot_inv,
        )
        polygon_points, on_route, map_valid_mask, map_notes = self._build_lane_tensor(
            map_graph,
            origin_xy,
            rot_inv,
            ref_position,
            ref_valid_mask,
        )
        point_position, point_vector, point_valid_mask, crosswalk_notes = self._build_crosswalk_tensor(
            map_graph,
            origin_xy,
            rot_inv,
        )

        current_state = np.array(
            [
                0.0,
                0.0,
                0.0,
                float(dataset["ego_speed"][t0_index]),
                float(dataset["ego_accel"][t0_index]),
                float(dataset["ego_steering"][t0_index]),
                float(dataset["ego_ang_vel"][t0_index]),
            ],
            dtype=np.float32,
        )

        feed = {
            "agent_position": agent_position,
            "agent_heading": agent_heading.astype(np.float32),
            "agent_velocity": agent_velocity,
            "agent_shape": agent_shape,
            "agent_category": agent_category,
            "agent_valid_mask": agent_valid_mask,
            "current_state": current_state,
            "static_position": static_position,
            "static_heading": static_heading,
            "static_shape": static_shape,
            "static_valid_mask": static_valid_mask,
            "ref_position": ref_position,
            "ref_vector": ref_vector,
            "ref_orientation": ref_orientation,
            "ref_valid_mask": ref_valid_mask,
            "polygon_points": polygon_points,
            "map_valid_mask": map_valid_mask,
            "on_route": on_route,
            "point_position": point_position,
            "point_vector": point_vector,
            "point_valid_mask": point_valid_mask,
            "origin": origin_xy.astype(np.float64),
            "angle": np.array([angle], dtype=np.float64),
        }

        context = {
            "clip_id": dataset["clip_id"],
            "logfile": dataset["logfile"],
            "vehicle_key": dataset["vehicle_key"],
            "ego_dims": list(dataset["ego_dims"]),
            "map_name": str(config.get("map_name") or ""),
            "t0_index": t0_index,
            "t0_time_sec": sample_ts_sec[t0_index],
            "t0_reason": t0_reason,
            "origin_xy": origin_xy.tolist(),
            "origin_z": origin_z,
            "angle_rad": angle,
            "history_sample_indices": history_indices.tolist(),
            "history_sample_tokens": [sample_tokens[index] for index in history_indices],
            "history_timestamps_sec": [sample_ts_sec[index] for index in history_indices],
            "history_pose_tokens": [dataset["sample_pose_tokens"][index] for index in history_indices],
            "ego_history_global_xy": ego_positions_hist.tolist(),
            "selected_agent_instance_tokens": selected_agents,
            "selected_static_instance_tokens": selected_static,
            "category_codes": CATEGORY_CODES,
            "current_state_definition": [
                "x_local_m",
                "y_local_m",
                "heading_local_rad",
                "speed_mps",
                "linear_accel_norm_mps2",
                "steering_percentage",
                "yaw_rate_rps",
            ],
            "shape_constants": {
                "MAX_AGENTS": MAX_AGENTS,
                "HIST_STEPS": HIST_STEPS,
                "REF_STEPS": REF_STEPS,
                "MAX_STATIC": MAX_STATIC,
                "MAX_LANES": MAX_LANES,
                "LANE_POINTS": LANE_POINTS,
                "MAX_CROSSWALKS": MAX_CROSSWALKS,
                "CROSSWALK_POINTS": CROSSWALK_POINTS,
            },
            "notes": [
                "Map tensor shape caps are provisional placeholders pending final ONNX collation reconciliation.",
                "on_route is a best-effort proximity test against the reference line rather than a route-lane ground truth flag.",
                "Static objects are derived from tracked agents with t0 speed <= 0.5 m/s.",
                "Ego box falls back to Chrysler Pacifica dimensions for all vehicles until exact per-vehicle box specs are reconciled.",
            ],
            "map_notes": map_notes,
            "crosswalk_notes": crosswalk_notes,
        }
        return feed, context

    def _build_reference_line(
        self,
        ego_future_xy: np.ndarray,
        origin_xy: np.ndarray,
        rot_inv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        local_future = _transform_points(ego_future_xy, origin_xy, rot_inv)
        ref_position, ref_valid_mask = _resample_by_spacing(local_future, REF_STEPS, REF_SPACING)
        ref_vector = np.zeros((REF_STEPS, 2), dtype=np.float32)
        ref_orientation = np.zeros((REF_STEPS,), dtype=np.float32)
        valid_indices = np.flatnonzero(ref_valid_mask)
        for idx in valid_indices:
            if idx + 1 in valid_indices:
                delta = ref_position[idx + 1] - ref_position[idx]
            elif idx - 1 in valid_indices:
                delta = ref_position[idx] - ref_position[idx - 1]
            else:
                delta = np.array([1.0, 0.0], dtype=np.float32)
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-6:
                tangent = np.array([1.0, 0.0], dtype=np.float32)
            else:
                tangent = delta / norm
            ref_vector[idx] = tangent.astype(np.float32)
            ref_orientation[idx] = math.atan2(float(tangent[1]), float(tangent[0]))
        return ref_position, ref_vector, ref_orientation, ref_valid_mask

    def _build_lane_tensor(
        self,
        map_graph: dict,
        origin_xy: np.ndarray,
        rot_inv: np.ndarray,
        ref_position: np.ndarray,
        ref_valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
        lane_candidates: List[Tuple[float, dict]] = []
        for lane in map_graph.get("lanes", []):
            center_xy = np.array([point[:2] for point in lane["central"]], dtype=np.float64)
            distance = _min_distance_to_polyline(origin_xy, center_xy)
            if distance <= RADIUS + MAP_SELECTION_MARGIN_M:
                lane_candidates.append((distance, lane))
        lane_candidates.sort(key=lambda item: item[0])

        polygon_points = np.zeros((MAX_LANES, 3, LANE_POINTS, 2), dtype=np.float32)
        map_valid_mask = np.zeros((MAX_LANES,), dtype=bool)
        on_route = np.zeros((MAX_LANES,), dtype=bool)
        ref_valid_points = ref_position[ref_valid_mask]

        for lane_slot, (_, lane) in enumerate(lane_candidates[:MAX_LANES]):
            left_local = _transform_points(np.array(lane["left"], dtype=np.float64), origin_xy, rot_inv)
            center_local = _transform_points(np.array(lane["central"], dtype=np.float64), origin_xy, rot_inv)
            right_local = _transform_points(np.array(lane["right"], dtype=np.float64), origin_xy, rot_inv)
            polygon_points[lane_slot, 0] = _resample_polyline(center_local, LANE_POINTS)
            polygon_points[lane_slot, 1] = _resample_polyline(left_local, LANE_POINTS)
            polygon_points[lane_slot, 2] = _resample_polyline(right_local, LANE_POINTS)
            map_valid_mask[lane_slot] = True
            if len(ref_valid_points) > 0:
                center_points = polygon_points[lane_slot, 0]
                distances = np.linalg.norm(
                    center_points[:, None, :] - ref_valid_points[None, :, :],
                    axis=2,
                )
                on_route[lane_slot] = float(distances.min()) <= ON_ROUTE_THRESHOLD_M

        notes = {
            "candidate_count": len(lane_candidates),
            "kept_count": int(map_valid_mask.sum()),
            "cut_count": max(0, len(lane_candidates) - MAX_LANES),
        }
        return polygon_points, on_route, map_valid_mask, notes

    def _build_crosswalk_tensor(
        self,
        map_graph: dict,
        origin_xy: np.ndarray,
        rot_inv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
        candidates: List[Tuple[float, np.ndarray]] = []
        for polygon in map_graph.get("crosswalks", {}).values():
            points_xy = np.array([point[:2] for point in polygon], dtype=np.float64)
            distance = _min_distance_to_polyline(origin_xy, points_xy)
            if distance <= RADIUS + MAP_SELECTION_MARGIN_M:
                candidates.append((distance, points_xy))
        candidates.sort(key=lambda item: item[0])

        point_position = np.zeros((MAX_CROSSWALKS, CROSSWALK_POINTS, 2), dtype=np.float32)
        point_vector = np.zeros((MAX_CROSSWALKS, CROSSWALK_POINTS, 2), dtype=np.float32)
        point_valid_mask = np.zeros((MAX_CROSSWALKS, CROSSWALK_POINTS), dtype=bool)

        for slot, (_, polygon_xy) in enumerate(candidates[:MAX_CROSSWALKS]):
            polygon_loop = polygon_xy
            if len(polygon_loop) >= 2 and not np.allclose(polygon_loop[0], polygon_loop[-1]):
                polygon_loop = np.vstack([polygon_loop, polygon_loop[0]])
            local = _transform_points(polygon_loop, origin_xy, rot_inv)
            resampled = _resample_polyline(local, CROSSWALK_POINTS)
            point_position[slot] = resampled
            point_valid_mask[slot] = True
            deltas = np.roll(resampled, -1, axis=0) - resampled
            norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            norms[norms <= 1e-6] = 1.0
            point_vector[slot] = (deltas / norms).astype(np.float32)

        notes = {
            "candidate_count": len(candidates),
            "kept_count": int(point_valid_mask.any(axis=1).sum()),
            "cut_count": max(0, len(candidates) - MAX_CROSSWALKS),
        }
        return point_position, point_vector, point_valid_mask, notes


register_input_builder("pluto", PlutoInputBuilder)
