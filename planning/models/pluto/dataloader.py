from __future__ import annotations

import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from calibration.vehicle import (
    apply_imu_lateral_offset,
    ego_dims as calibration_ego_dims,
    max_tire_angle,
    steering_pct_to_tire_angle,
    to_vehicle_parameters,
)
from planning.interface import Dataloader, register_dataloader


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
FEATURE_VEHICLE_DIMENSIONS = {"pacifica": PACIFICA_DIMS}
# C-SWM-026 sign experiment on BT-25/BT-22 kept +1 as the default:
# BT-22 improved both reference and lane-center distances; BT-25 split.
DEFAULT_IMU_LAT_SIGN = 1.0

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


class PlutoFeedBuilder:
    """PLUTO feed-building body kept local to the dataloader module."""
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
        calib = dict(config.get("calibration") or {})
        if not calib:
            raise ValueError("config.calibration must be resolved before Pluto dataloader use")
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

        ego_positions_imu = np.zeros((len(samples), 2), dtype=np.float64)
        ego_headings = np.zeros((len(samples),), dtype=np.float64)
        ego_speed = np.zeros((len(samples),), dtype=np.float64)
        ego_accel = np.zeros((len(samples),), dtype=np.float64)
        ego_steering = np.zeros((len(samples),), dtype=np.float64)
        ego_ang_vel = np.zeros((len(samples),), dtype=np.float64)
        sample_pose_tokens: List[str] = []

        for idx, sample in enumerate(samples):
            dyn = dynamics_by_sample[sample["token"]]
            pose = pose_by_token[dyn["ego_pose_token"]]
            ego_positions_imu[idx] = np.array(pose["translation"][:2], dtype=np.float64)
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

        imu_sign = float(config.get("imu_lat_sign", DEFAULT_IMU_LAT_SIGN))
        ego_positions = apply_imu_lateral_offset(
            ego_positions_imu,
            ego_headings,
            float(calib.get("imu_lat_offset_m", 0.0)),
            imu_sign,
        )
        clip_id = str(config.get("clip_id") or tables["scene"][0]["name"])
        feature_vehicle = str(config.get("feature_vehicle", "pacifica")).lower()
        feature_ego_dims = FEATURE_VEHICLE_DIMENSIONS.get(feature_vehicle, PACIFICA_DIMS)
        physical_ego_dims = calibration_ego_dims(calib)
        vehicle_parameters = to_vehicle_parameters(calib)

        return {
            "samples": samples,
            "sample_tokens": sample_tokens,
            "sample_ts_sec": sample_ts_sec,
            "ego_positions": ego_positions,
            "ego_positions_imu": ego_positions_imu,
            "ego_headings": ego_headings,
            "ego_speed": ego_speed,
            "ego_accel": ego_accel,
            "ego_steering": ego_steering,
            "ego_ang_vel": ego_ang_vel,
            "sample_pose_tokens": sample_pose_tokens,
            "ann_by_sample": ann_by_sample,
            "track_data": track_data,
            "clip_id": clip_id,
            "feature_ego_dims": feature_ego_dims,
            "physical_ego_dims": physical_ego_dims,
            "feature_vehicle": feature_vehicle,
            "vehicle_parameters": vehicle_parameters,
            "max_tire_angle_rad": max_tire_angle(calib),
            "calibration": calib,
            "imu_lat_sign": imu_sign,
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
        agent_shape[0] = np.array(dataset["feature_ego_dims"], dtype=np.float32)
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

        steering_angle = steering_pct_to_tire_angle(
            float(dataset["ego_steering"][t0_index]),
            dataset["calibration"],
        )
        current_state = np.array(
            [
                0.0,
                0.0,
                0.0,
                float(dataset["ego_speed"][t0_index]),
                float(dataset["ego_accel"][t0_index]),
                steering_angle,
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
            "feature_vehicle": dataset["feature_vehicle"],
            "feature_ego_dims": list(dataset["feature_ego_dims"]),
            "physical_ego_dims": list(dataset["physical_ego_dims"]),
            "calibration_vehicle": str(dataset["calibration"]["vehicle"]),
            "imu_lat_sign": float(dataset["imu_lat_sign"]),
            "max_tire_angle_rad": float(dataset["max_tire_angle_rad"]),
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
                "tire_steering_angle_rad",
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
                "Feature ego box stays pinned to the model feature_vehicle; physical ego box follows calibration.",
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

POLYGON_TYPE_LANE = 0
POLYGON_TYPE_LANE_CONNECTOR = 1
POLYGON_TYPE_CROSSWALK = 2

TL_GREEN = 0
TL_YELLOW = 1
TL_RED = 2
TL_UNKNOWN = 3

STATIC_GENERIC = 3
FUTURE_STEPS = 80
TOTAL_STEPS = HIST_STEPS + FUTURE_STEPS


@dataclass
class AdapterBuildResult:
    feature: Any
    context: Dict[str, Any]
    normalized_numpy_data: Dict[str, Any]
    scene_context: Optional[Dict[str, Any]] = None


class ApolloPlutoDataloader(Dataloader):
    # 데이터 계약 (C-SWM-023): 이 dataloader가 소비하는 data_devkit 아티팩트.
    # prepare_clip()/build() 시작 시 data_devkit.contract.check로 fail-fast
    # 검증한다 (누락 시 "무엇을 돌려야 하는지" 안내 포함 예외).
    # 스펙 §3의 6종 + scene_log(_load_tables가 scene/log.json도 읽으므로 명시).
    REQUIRES = [
        "sample",
        "ego_pose",
        "ego_dynamics",
        "agent_tracks",
        "scene_log",
        "route",
        "map_graph",
    ]

    def __init__(self) -> None:
        self._builder = PlutoFeedBuilder()

    def _check_contract(self, parsed_dir: str, config: dict) -> None:
        """data_devkit 계약 check — 경로는 parsed_dir 기준으로 해석한다.

        clip 루트 = parsed_dir 부모, maps 루트 = <work 루트>/maps.  검증/데모용
        임시 클립 사본(work/ 안 임시 디렉토리)도 같은 규약으로 검사된다.
        """
        from data_devkit import contract

        parsed_path = Path(parsed_dir).resolve()
        clip_root = parsed_path.parent
        clip_id = str(config.get("clip_id") or clip_root.name)
        report = contract.check(
            clip_id=clip_id,
            requires=self.REQUIRES,
            data_cfg=config.get("data"),
            map_name=str(config.get("map_name") or "") or None,
            clip_dir=clip_root,
            maps_root=clip_root.parent / "maps",
        )
        print(
            f"[data_devkit.contract] check ok: clip={clip_id} "
            f"artifacts={report['checked']} provenance={report['provenance']}"
        )

    def _import_pluto_feature(self):
        from planning.models.pluto.src.features.pluto_feature import PlutoFeature

        return PlutoFeature

    def _import_scenario_manager(self):
        from planning.nuplan_common.scenario_manager.scenario_manager import ScenarioManager

        return ScenarioManager

    def build(
        self,
        parsed_dir: str,
        map_graph: dict,
        config: dict,
        t0_time: float | None = None,
    ) -> AdapterBuildResult:
        self._check_contract(parsed_dir, config)
        PlutoFeature = self._import_pluto_feature()
        parsed_path = Path(parsed_dir)
        tables = self._builder._load_tables(parsed_path)
        dataset = self._builder._prepare_dataset(tables, config)
        t0_index, t0_reason = self._builder._select_t0_index(dataset, t0_time)

        raw_data, context, scene_context = self._build_global_data(
            parsed_path,
            dataset,
            map_graph,
            t0_index,
            t0_reason,
            map_name=str(config.get("map_name") or ""),
        )
        normalized = PlutoFeature.normalize(
            raw_data,
            first_time=True,
            radius=RADIUS,
            hist_steps=HIST_STEPS,
        )
        tensor_feature = PlutoFeature.collate([normalized.to_feature_tensor()])
        return AdapterBuildResult(
            feature=tensor_feature,
            context=context,
            normalized_numpy_data=normalized.data,
            scene_context=scene_context,
        )

    # ------------------------------------------------------------- sim API
    #
    # C-SWM-018: repeated per-frame builds for the simulation driver.
    #   prepare_clip()  loads/caches everything that is frame-independent
    #                   (parsed tables, route.json event history, ApolloMap).
    #   build_frame()   builds a PlutoFeature for an arbitrary log frame
    #                   (open-loop) or with an injected sim ego history
    #                   (closed-loop).  Normalization/packing reuses the
    #                   exact same code path as build().
    #
    # sim_ego contract (all history arrays cover the HIST_STEPS window,
    # oldest first, index -1 = current sim state; global/UTM coords,
    # rear-axle pose convention identical to the parsed log):
    #   {
    #     "position":        (21, 2) float64  global xy
    #     "heading":         (21,)   float64
    #     "velocity_global": (21, 2) float64  global velocity vector
    #     "valid_mask":      (21,)   bool
    #     "current_state":   (7,)    [x, y, heading, speed, accel,
    #                                 steering_angle, yaw_rate]
    #     "ego_state":       nuplan EgoState at the current sim time
    #   }
    # Ego log data is NOT read when sim_ego is given (no ego-future
    # leakage); agents/statics come from the log frame t0_index chosen by
    # the caller (time-axis replay in closed-loop, issue #1).

    def prepare_clip(
        self,
        parsed_dir: str,
        map_graph: dict,
        config: dict,
    ) -> Dict[str, Any]:
        """Loads per-clip inputs once so build_frame() can run per frame."""
        from planning.map_adapter.apollo_map import ApolloMap

        self._check_contract(parsed_dir, config)
        parsed_path = Path(parsed_dir)
        tables = self._builder._load_tables(parsed_path)
        dataset = self._builder._prepare_dataset(tables, config)
        route_path = parsed_path / "route.json"
        if not route_path.exists():
            raise FileNotFoundError(
                f"Missing {route_path}; parse route extraction before simulation."
            )
        route_payload = json.loads(route_path.read_text())
        map_name = str(config.get("map_name") or "")
        apollo_map = ApolloMap(map_name=map_name or None, payload=map_graph)
        return {
            "parsed_path": parsed_path,
            "dataset": dataset,
            "map_graph": map_graph,
            "route_payload": route_payload,
            # issue #2: per-event route derivations (lane_ids -> public
            # roadblock ids), keyed by event timestamp_ns.  Frames that keep
            # the same routing event hit this cache; an event switch
            # recomputes exactly once.
            "route_event_cache": {},
            "map_api": apollo_map,
            "map_name": map_name,
            "dt": DT,
            "hist_steps": HIST_STEPS,
            "config": dict(config),
            "vehicle_parameters": dataset["vehicle_parameters"],
            "max_tire_angle_rad": dataset["max_tire_angle_rad"],
        }

    def build_frame(
        self,
        clip: Dict[str, Any],
        t0_index: int,
        sim_ego: Optional[Dict[str, Any]] = None,
    ) -> AdapterBuildResult:
        """Builds one PlutoFeature frame from a prepare_clip() bundle.

        sim_ego=None  -> open-loop: ego history/state from the log at
                         t0_index (route t0 check skipped: the route event
                         is re-selected per frame, see below).
        sim_ego=dict  -> closed-loop: ego row comes exclusively from the
                         injected sim history; agents/statics are fetched
                         from log frame t0_index.

        Route (issue #2): every frame selects the routing event that is the
        latest one at the frame's own log timestamp (sample timestamp at
        t0_index), so a reroute published mid-clip switches the reference
        lines / route_lane_dict / on_route flags once the sim passes its
        time.  All route consumers of one frame (ScenarioManager, map
        features, scene_context handed to postprocess/render) derive from
        that single event — no mixing within a frame.
        """
        PlutoFeature = self._import_pluto_feature()
        t0_reason = "sim_open_loop_frame" if sim_ego is None else "sim_closed_loop_frame"
        raw_data, context, scene_context = self._build_global_data(
            clip["parsed_path"],
            clip["dataset"],
            clip["map_graph"],
            t0_index,
            t0_reason,
            map_name=clip["map_name"],
            ego_override=sim_ego,
            route_payload=clip["route_payload"],
            map_api=clip["map_api"],
            route_event_cache=clip.setdefault("route_event_cache", {}),
        )
        normalized = PlutoFeature.normalize(
            raw_data,
            first_time=True,
            radius=RADIUS,
            hist_steps=HIST_STEPS,
        )
        tensor_feature = PlutoFeature.collate([normalized.to_feature_tensor()])
        return AdapterBuildResult(
            feature=tensor_feature,
            context=context,
            normalized_numpy_data=normalized.data,
            scene_context=scene_context,
        )

    def _build_global_data(
        self,
        parsed_path: Path,
        dataset: Dict[str, Any],
        map_graph: dict,
        t0_index: int,
        t0_reason: str,
        map_name: str = "",
        ego_override: Optional[Dict[str, Any]] = None,
        route_payload: Optional[Dict[str, Any]] = None,
        map_api: Optional[Any] = None,
        route_event_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        sample_tokens = dataset["sample_tokens"]
        sample_ts_sec = dataset["sample_ts_sec"]
        history_indices, history_deltas = self._builder._history_target_indices(sample_ts_sec, t0_index)
        if ego_override is None:
            origin_xy = dataset["ego_positions"][t0_index].copy()
            angle = float(dataset["ego_headings"][t0_index])
        else:
            origin_xy = np.asarray(
                ego_override["position"][-1], dtype=np.float64
            ).copy()
            angle = float(ego_override["heading"][-1])

        agent_position = np.zeros((MAX_AGENTS, TOTAL_STEPS, 2), dtype=np.float64)
        agent_heading = np.zeros((MAX_AGENTS, TOTAL_STEPS), dtype=np.float64)
        agent_velocity = np.zeros((MAX_AGENTS, TOTAL_STEPS, 2), dtype=np.float64)
        agent_shape = np.zeros((MAX_AGENTS, TOTAL_STEPS, 2), dtype=np.float64)
        agent_category = np.zeros((MAX_AGENTS,), dtype=np.int8)
        agent_valid_mask = np.zeros((MAX_AGENTS, TOTAL_STEPS), dtype=bool)

        if ego_override is None:
            ego_positions_hist = dataset["ego_positions"][history_indices]
            ego_headings_hist = dataset["ego_headings"][history_indices]
            ego_speed_hist = dataset["ego_speed"][history_indices]
            ego_vectors_global = np.stack(
                [ego_speed_hist * np.cos(ego_headings_hist), ego_speed_hist * np.sin(ego_headings_hist)],
                axis=1,
            )
            ego_hist_valid = history_deltas <= (DT * 0.6)
        else:
            ego_positions_hist = np.asarray(ego_override["position"], dtype=np.float64)
            ego_headings_hist = np.asarray(ego_override["heading"], dtype=np.float64)
            ego_vectors_global = np.asarray(
                ego_override["velocity_global"], dtype=np.float64
            )
            ego_hist_valid = np.asarray(ego_override["valid_mask"], dtype=bool)

        agent_position[0, :HIST_STEPS] = ego_positions_hist
        agent_heading[0, :HIST_STEPS] = ego_headings_hist
        agent_velocity[0, :HIST_STEPS] = ego_vectors_global
        agent_shape[0] = np.array(dataset["feature_ego_dims"], dtype=np.float64)
        agent_category[0] = CATEGORY_CODES["ego"]
        agent_valid_mask[0, :HIST_STEPS] = ego_hist_valid

        t0_sample_token = sample_tokens[t0_index]
        dynamic_candidates: List[Tuple[float, str]] = []
        static_candidates: List[Tuple[float, str]] = []
        for instance_token, track in dataset["track_data"].items():
            ann_idx = track["sample_to_index"].get(t0_sample_token)
            if ann_idx is None:
                continue
            position = track["positions"][ann_idx]
            distance = float(np.linalg.norm(position - origin_xy))
            if distance > RADIUS + MAP_SELECTION_MARGIN_M:
                continue
            speed = float(np.linalg.norm(track["velocities"][ann_idx]))
            if speed <= STATIC_SPEED_THRESHOLD_MPS:
                static_candidates.append((distance, instance_token))
            else:
                dynamic_candidates.append((distance, instance_token))
        dynamic_candidates.sort(key=lambda item: item[0])
        static_candidates.sort(key=lambda item: item[0])

        selected_agents = [token for _, token in dynamic_candidates[: MAX_AGENTS - 1]]
        selected_static = [
            token
            for _, token in static_candidates
            if token not in selected_agents
        ][:MAX_STATIC]

        for agent_slot, instance_token in enumerate(selected_agents, start=1):
            track = dataset["track_data"][instance_token]
            sample_to_index = track["sample_to_index"]
            last_idx = None
            for hist_slot, sample_index in enumerate(history_indices):
                ann_idx = sample_to_index.get(sample_tokens[sample_index])
                if ann_idx is None:
                    continue
                agent_position[agent_slot, hist_slot] = track["positions"][ann_idx]
                agent_heading[agent_slot, hist_slot] = track["headings"][ann_idx]
                agent_velocity[agent_slot, hist_slot] = track["velocities"][ann_idx]
                agent_valid_mask[agent_slot, hist_slot] = True
                last_idx = ann_idx
            if last_idx is not None:
                shape = np.array(
                    [track["widths"][last_idx], track["lengths"][last_idx]],
                    dtype=np.float64,
                )
                agent_shape[agent_slot] = shape
            category = CATEGORY_CODES.get(track["category"], CATEGORY_CODES["vehicle"])
            agent_category[agent_slot] = np.int8(max(0, min(3, category)))

        static_position = np.zeros((MAX_STATIC, 2), dtype=np.float64)
        static_heading = np.zeros((MAX_STATIC,), dtype=np.float64)
        static_shape = np.zeros((MAX_STATIC, 2), dtype=np.float64)
        static_category = np.full((MAX_STATIC,), STATIC_GENERIC, dtype=np.int8)
        static_valid_mask = np.zeros((MAX_STATIC,), dtype=bool)
        for slot, instance_token in enumerate(selected_static):
            track = dataset["track_data"][instance_token]
            ann_idx = track["sample_to_index"][t0_sample_token]
            static_position[slot] = track["positions"][ann_idx]
            static_heading[slot] = track["headings"][ann_idx]
            static_shape[slot] = np.array(
                [track["widths"][ann_idx], track["lengths"][ann_idx]],
                dtype=np.float64,
            )
            static_valid_mask[slot] = True

        if route_payload is None:
            route_payload = self._load_route_payload(parsed_path, t0_index, dataset)
        # issue #2: the frame's own log time decides which routing event
        # applies (t0 path and sim frame path share this selection; for the
        # t0 path it reproduces the parse-time t0 selection).
        frame_ts_ns = int(dataset["samples"][t0_index]["timestamp"])
        route_payload = self._route_payload_at_time(
            route_payload,
            frame_ts_ns,
            map_graph=map_graph,
            map_api=map_api,
            cache=route_event_cache,
        )
        if ego_override is None:
            ego_state = self._build_ego_state(dataset, t0_index)
        else:
            ego_state = ego_override["ego_state"]
        scenario_bundle = self._build_scenario_manager(
            map_graph=map_graph,
            map_name=map_name,
            route_payload=route_payload,
            ego_state=ego_state,
            map_api=map_api,
        )
        reference_line, route_debug = self._pack_reference_lines(
            scenario_bundle["reference_lines"]
        )
        route_debug["selected_route_event_timestamp_ns"] = route_payload.get(
            "selected_event_timestamp_ns"
        )
        route_debug["selected_route_event_topic"] = route_payload.get(
            "selected_event_topic"
        )
        route_debug["route_lane_count"] = len(scenario_bundle["route_lane_ids"])
        route_debug["route_event_frame_ts_ns"] = frame_ts_ns
        route_debug["route_event_count"] = len(self._route_events(route_payload))
        map_features, map_notes = self._build_map_features_global(
            map_graph=map_graph,
            ego_global_xy=origin_xy,
            reference_line=reference_line,
        )
        if ego_override is None:
            steering_angle = steering_pct_to_tire_angle(
                float(dataset["ego_steering"][t0_index]),
                dataset["calibration"],
            )
            current_state = np.array(
                [
                    float(origin_xy[0]),
                    float(origin_xy[1]),
                    angle,
                    float(dataset["ego_speed"][t0_index]),
                    float(dataset["ego_accel"][t0_index]),
                    steering_angle,
                    float(dataset["ego_ang_vel"][t0_index]),
                ],
                dtype=np.float64,
            )
        else:
            current_state = np.asarray(
                ego_override["current_state"], dtype=np.float64
            ).copy()

        raw_data = {
            "agent": {
                "position": agent_position,
                "heading": agent_heading,
                "velocity": agent_velocity,
                "shape": agent_shape,
                "category": agent_category,
                "valid_mask": agent_valid_mask,
            },
            "map": map_features,
            "reference_line": reference_line,
            "static_objects": {
                "position": static_position,
                "heading": static_heading,
                "shape": static_shape,
                "category": static_category,
                "valid_mask": static_valid_mask,
            },
            "current_state": current_state,
            "origin": origin_xy.astype(np.float64),
            "angle": np.array(angle, dtype=np.float64),
        }

        context = {
            "clip_id": dataset["clip_id"],
            "logfile": dataset["logfile"],
            "feature_vehicle": dataset["feature_vehicle"],
            "feature_ego_dims": list(dataset["feature_ego_dims"]),
            "physical_ego_dims": list(dataset["physical_ego_dims"]),
            "calibration_vehicle": str(dataset["calibration"]["vehicle"]),
            "imu_lat_sign": float(dataset["imu_lat_sign"]),
            "max_tire_angle_rad": float(dataset["max_tire_angle_rad"]),
            "t0_index": t0_index,
            "t0_time_sec": sample_ts_sec[t0_index],
            "t0_reason": t0_reason,
            "ego_source": "log" if ego_override is None else "sim_override",
            "origin_xy_global": origin_xy.tolist(),
            "angle_rad_global": angle,
            "history_sample_indices": history_indices.tolist(),
            "history_sample_tokens": [sample_tokens[index] for index in history_indices],
            "selected_agent_instance_tokens": selected_agents,
            "selected_static_instance_tokens": selected_static,
            "adapter_notes": [
                "Agent tensors use T=101 with history/present in slots [0:21] and future slots zero-filled with valid_mask=False.",
                "Static objects are approximated from low-speed tracked annotations at t0 and mapped to GENERIC static category.",
                "Reference lines come from the original pluto ScenarioManager/RouteManager driven through ApolloMap (C-SWM-017); the C-SWM-015 reimplementation was removed.",
                "Reference line is built from map+route only; ego future poses are not referenced.",
                "reference_line.future_projection is zero-filled intentionally to avoid ego future leakage.",
                "Traffic light status is unresolved from the Apollo feed and set to UNKNOWN for every map polygon.",
            ],
            "map_notes": map_notes,
            "route_debug": route_debug,
        }

        scene_context = self._build_scene_context(
            dataset=dataset,
            t0_index=t0_index,
            ego_state=ego_state,
            scenario_bundle=scenario_bundle,
            selected_agents=selected_agents,
            selected_static=selected_static,
        )
        return raw_data, context, scene_context

    def _build_ego_state(self, dataset: Dict[str, Any], t0_index: int):
        """Builds a nuPlan EgoState at t0 from parsed ego pose/dynamics.

        The parsed Apollo localization pose is treated as the rear-axle pose
        after dataloader-time IMU lateral correction. Longitudinal acceleration
        is a signed finite-difference of speed; the tire steering angle uses
        the measured steering percentage when available and otherwise falls
        back to a yaw-rate-based kinematic estimate.
        """
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import (
            StateSE2,
            StateVector2D,
            TimePoint,
        )
        xy = dataset["ego_positions"][t0_index]
        heading = float(dataset["ego_headings"][t0_index])
        speed = float(dataset["ego_speed"][t0_index])
        yaw_rate = float(dataset["ego_ang_vel"][t0_index])
        ts_ns = int(dataset["samples"][t0_index]["timestamp"])

        if t0_index >= 1:
            dt_sec = max(
                float(
                    dataset["sample_ts_sec"][t0_index]
                    - dataset["sample_ts_sec"][t0_index - 1]
                ),
                1e-3,
            )
            signed_accel = (
                speed - float(dataset["ego_speed"][t0_index - 1])
            ) / dt_sec
        else:
            signed_accel = float(dataset["ego_accel"][t0_index])

        vehicle_parameters = dataset["vehicle_parameters"]
        measured_pct = float(dataset["ego_steering"][t0_index])
        if np.isfinite(measured_pct):
            steering_angle = steering_pct_to_tire_angle(
                measured_pct,
                dataset["calibration"],
            )
        elif speed > 0.5:
            steering_angle = float(
                np.clip(
                    math.atan(vehicle_parameters.wheel_base * yaw_rate / speed),
                    -dataset["max_tire_angle_rad"],
                    dataset["max_tire_angle_rad"],
                )
            )
        else:
            steering_angle = 0.0

        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(float(xy[0]), float(xy[1]), heading),
            rear_axle_velocity_2d=StateVector2D(speed, 0.0),
            rear_axle_acceleration_2d=StateVector2D(signed_accel, 0.0),
            tire_steering_angle=steering_angle,
            time_point=TimePoint(ts_ns // 1000),
            vehicle_parameters=vehicle_parameters,
            angular_vel=yaw_rate,
        )

    def _build_scenario_manager(
        self,
        map_graph: dict,
        map_name: str,
        route_payload: Dict[str, Any],
        ego_state: Any,
        map_api: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Drives the original pluto ScenarioManager/RouteManager over ApolloMap.

        This replaces the C-SWM-015 reference-line reimplementation (which
        over-generated R by skipping the original "merge repeated lanes"
        candidate-pruning step). The original ScenarioManager output is
        authoritative.
        """
        from planning.map_adapter.apollo_map import ApolloMap

        ScenarioManager = self._import_scenario_manager()

        route_lane_ids = [
            str(lane_id) for lane_id in route_payload.get("route_lane_ids", [])
        ]
        if not route_lane_ids:
            raise ValueError(
                "route.json has no route_lane_ids; cannot build reference lines."
            )

        apollo_map = (
            map_api
            if map_api is not None
            else ApolloMap(map_name=map_name or None, payload=map_graph)
        )
        # issue #2: reuse the per-event lane->roadblock resolution when the
        # payload carries one (event-keyed cache in build_frame); otherwise
        # resolve from scratch (t0/build path, legacy payloads).
        public_roadblock_ids = route_payload.get("_route_public_roadblock_ids")
        if public_roadblock_ids is None:
            original_roadblock_ids = self._route_roadblocks_from_lane_ids(
                route_lane_ids, map_graph
            )
            public_roadblock_ids = [
                apollo_map.to_public_roadblock_id(rb_id)
                for rb_id in original_roadblock_ids
            ]
        if not public_roadblock_ids:
            raise ValueError(
                "route.json route_lane_ids do not map to any roadblocks in map_graph."
            )

        # Same radius formula as PlutoPlanner: eval_dt * eval_num_frames * 60 / 4.
        radius = 0.1 * 80 * 60.0 / 4.0
        scenario_manager = ScenarioManager(
            map_api=apollo_map,
            ego_state=ego_state,
            route_roadblocks_ids=public_roadblock_ids,
            radius=radius,
        )
        loaded_roadblock_ids = scenario_manager.get_route_roadblock_ids(process=True)
        scenario_manager.update_ego_state(ego_state)
        scenario_manager.update_drivable_area_map()
        reference_lines = scenario_manager.get_reference_lines(
            length=REF_STEPS * REF_SPACING
        )
        if not reference_lines:
            raise ValueError("Original ScenarioManager produced no reference lines.")

        return {
            "map_api": apollo_map,
            "scenario_manager": scenario_manager,
            "reference_lines": reference_lines,
            "route_lane_ids": route_lane_ids,
            "route_roadblock_ids_input": public_roadblock_ids,
            "route_roadblock_ids_loaded": list(loaded_roadblock_ids),
        }

    def _pack_reference_lines(
        self, reference_lines: List[np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Packs ScenarioManager reference lines exactly like the original
        `_get_reference_line_feature` (1 m x REF_STEPS via 0.25 m subsample[::4]).

        future_projection stays zero-filled: it is derived from ego future
        poses in the original trainer and would leak the future at inference.
        """
        merged_paths = [
            self._resample_path_quarter_meter(line) for line in reference_lines
        ]

        position = np.zeros((len(merged_paths), REF_STEPS, 2), dtype=np.float64)
        vector = np.zeros((len(merged_paths), REF_STEPS, 2), dtype=np.float64)
        orientation = np.zeros((len(merged_paths), REF_STEPS), dtype=np.float64)
        valid_mask = np.zeros((len(merged_paths), REF_STEPS), dtype=bool)
        future_projection = np.zeros((len(merged_paths), 8, 2), dtype=np.float64)

        packed_counts: List[int] = []
        for route_idx, line in enumerate(merged_paths):
            subsample = line[::4][: REF_STEPS + 1]
            n_valid = max(0, len(subsample) - 1)
            if n_valid == 0:
                continue
            position[route_idx, :n_valid] = subsample[:-1, :2]
            vector[route_idx, :n_valid] = np.diff(subsample[:, :2], axis=0)
            orientation[route_idx, :n_valid] = subsample[:-1, 2]
            valid_mask[route_idx, :n_valid] = True
            packed_counts.append(n_valid)

        return (
            {
                "position": position,
                "vector": vector,
                "orientation": orientation,
                "valid_mask": valid_mask,
                "future_projection": future_projection,
            },
            {
                "reference_line_source": "original ScenarioManager.get_reference_lines via ApolloMap",
                "reference_line_count": len(merged_paths),
                "reference_line_shapes": [list(line.shape) for line in reference_lines],
                "reference_line_valid_points": packed_counts,
                "leakage_free": True,
            },
        )

    @staticmethod
    def _resample_path_quarter_meter(line: np.ndarray) -> np.ndarray:
        """Resamples an (M, 3) [x, y, heading] polyline at 0.25 m spacing so the
        original 1 m pack (subsample[::4]) applies to ApolloMap's 0.5 m discrete
        paths the same way it applies to nuPlan's 0.25 m discrete paths."""
        if len(line) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        xy = np.asarray(line[:, :2], dtype=np.float64)
        cumulative = _cumulative_lengths(xy)
        total = float(cumulative[-1]) if len(cumulative) else 0.0
        targets = np.arange(0.0, total + 1e-6, 0.25, dtype=np.float64)
        if len(targets) == 0:
            targets = np.zeros((1,), dtype=np.float64)
        out = np.zeros((len(targets), 3), dtype=np.float64)
        out[:, :2] = ApolloPlutoDataloader._sample_polyline_at_progress(
            xy, cumulative, targets
        )
        out[:, 2] = ApolloPlutoDataloader._sample_heading_at_progress(
            np.asarray(line[:, 2], dtype=np.float64), cumulative, targets
        )
        return out

    def _build_scene_context(
        self,
        dataset: Dict[str, Any],
        t0_index: int,
        ego_state: Any,
        scenario_bundle: Dict[str, Any],
        selected_agents: List[str],
        selected_static: List[str],
    ) -> Dict[str, Any]:
        """Builds the nuPlan-typed scene context consumed by the deployment
        post-processor (original TrajectoryEvaluator surfaces)."""
        from nuplan.common.actor_state.agent import Agent
        from nuplan.common.actor_state.oriented_box import OrientedBox
        from nuplan.common.actor_state.scene_object import SceneObjectMetadata
        from nuplan.common.actor_state.state_representation import (
            StateSE2,
            StateVector2D,
        )
        from nuplan.common.actor_state.static_object import StaticObject
        from nuplan.common.actor_state.tracked_objects import TrackedObjects
        from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
        from nuplan.planning.simulation.observation.observation_type import (
            DetectionsTracks,
        )

        dynamic_type_by_category = {
            "vehicle": TrackedObjectType.VEHICLE,
            "pedestrian": TrackedObjectType.PEDESTRIAN,
            "bicycle": TrackedObjectType.BICYCLE,
        }

        sample_token = dataset["sample_tokens"][t0_index]
        timestamp_us = int(dataset["samples"][t0_index]["timestamp"]) // 1000
        track_data = dataset["track_data"]

        tracked_objects = []
        for instance_token in selected_agents:
            track = track_data[instance_token]
            ann_idx = track["sample_to_index"][sample_token]
            center = StateSE2(
                float(track["positions"][ann_idx][0]),
                float(track["positions"][ann_idx][1]),
                float(track["headings"][ann_idx]),
            )
            box = OrientedBox(
                center,
                length=float(track["lengths"][ann_idx]),
                width=float(track["widths"][ann_idx]),
                height=float(track["heights"][ann_idx]),
            )
            metadata = SceneObjectMetadata(
                timestamp_us=timestamp_us,
                token=instance_token,
                track_id=None,
                track_token=instance_token,
                category_name=str(track["category"]),
            )
            tracked_objects.append(
                Agent(
                    tracked_object_type=dynamic_type_by_category.get(
                        str(track["category"]), TrackedObjectType.VEHICLE
                    ),
                    oriented_box=box,
                    velocity=StateVector2D(
                        float(track["velocities"][ann_idx][0]),
                        float(track["velocities"][ann_idx][1]),
                    ),
                    metadata=metadata,
                )
            )

        for instance_token in selected_static:
            track = track_data[instance_token]
            ann_idx = track["sample_to_index"][sample_token]
            center = StateSE2(
                float(track["positions"][ann_idx][0]),
                float(track["positions"][ann_idx][1]),
                float(track["headings"][ann_idx]),
            )
            box = OrientedBox(
                center,
                length=float(track["lengths"][ann_idx]),
                width=float(track["widths"][ann_idx]),
                height=float(track["heights"][ann_idx]),
            )
            metadata = SceneObjectMetadata(
                timestamp_us=timestamp_us,
                token=instance_token,
                track_id=None,
                track_token=instance_token,
                category_name=str(track["category"]),
            )
            tracked_objects.append(
                StaticObject(
                    tracked_object_type=TrackedObjectType.GENERIC_OBJECT,
                    oriented_box=box,
                    metadata=metadata,
                )
            )

        scenario_manager = scenario_bundle["scenario_manager"]
        return {
            "ego_state": ego_state,
            "detections": DetectionsTracks(TrackedObjects(tracked_objects)),
            "traffic_light_data": [],
            "scenario_manager": scenario_manager,
            "map_api": scenario_bundle["map_api"],
            "route_lane_dict": scenario_manager.get_route_lane_dicts(),
            "drivable_area_map": scenario_manager.drivable_area_map,
            "reference_lines_global": scenario_bundle["reference_lines"],
            "agent_tokens": list(selected_agents),
            "agent_rows": list(range(1, len(selected_agents) + 1)),
            "static_tokens": list(selected_static),
            "hist_steps": HIST_STEPS,
            "t0_timestamp_us": timestamp_us,
            "notes": [
                "traffic_light_data is empty: no traffic light state in the parsed Apollo feed.",
                "Agent/static boxes come from t0 sample_annotation (center pose, w/l/h).",
                "Ego acceleration is a signed speed finite-difference; steering angle is a yaw-rate kinematic estimate.",
            ],
        }

    def _build_map_features_global(
        self,
        map_graph: dict,
        ego_global_xy: np.ndarray,
        reference_line: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        lane_candidates: List[Tuple[float, dict]] = []
        for lane in map_graph.get("lanes", []):
            center_xy = np.array([point[:2] for point in lane["central"]], dtype=np.float64)
            distance = _min_distance_to_polyline(ego_global_xy, center_xy)
            if distance <= RADIUS + MAP_SELECTION_MARGIN_M:
                lane_candidates.append((distance, lane))
        lane_candidates.sort(key=lambda item: item[0])

        crosswalk_candidates: List[Tuple[float, np.ndarray]] = []
        for polygon in map_graph.get("crosswalks", {}).values():
            points_xy = np.array([point[:2] for point in polygon], dtype=np.float64)
            distance = _min_distance_to_polyline(ego_global_xy, points_xy)
            if distance <= RADIUS + MAP_SELECTION_MARGIN_M:
                crosswalk_candidates.append((distance, points_xy))
        crosswalk_candidates.sort(key=lambda item: item[0])

        total_polygons = min(MAX_LANES, len(lane_candidates)) + min(
            MAX_CROSSWALKS, len(crosswalk_candidates)
        )
        point_position = np.zeros((total_polygons, 3, LANE_POINTS, 2), dtype=np.float64)
        point_vector = np.zeros((total_polygons, 3, LANE_POINTS, 2), dtype=np.float64)
        point_side = np.zeros((total_polygons, 3), dtype=np.int8)
        point_orientation = np.zeros((total_polygons, 3, LANE_POINTS), dtype=np.float64)
        polygon_center = np.zeros((total_polygons, 3), dtype=np.float64)
        polygon_position = np.zeros((total_polygons, 2), dtype=np.float64)
        polygon_orientation = np.zeros((total_polygons,), dtype=np.float64)
        polygon_type = np.zeros((total_polygons,), dtype=np.int8)
        polygon_on_route = np.zeros((total_polygons,), dtype=bool)
        polygon_tl_status = np.full((total_polygons,), TL_UNKNOWN, dtype=np.int8)
        polygon_has_speed_limit = np.zeros((total_polygons,), dtype=bool)
        polygon_speed_limit = np.zeros((total_polygons,), dtype=np.float64)
        polygon_road_block_id = np.zeros((total_polygons,), dtype=np.int32)
        ref_lines = [
            reference_line["position"][idx][reference_line["valid_mask"][idx]]
            for idx in range(reference_line["position"].shape[0])
        ]
        slot = 0

        for _, lane in lane_candidates[:MAX_LANES]:
            center = np.array(lane["central"], dtype=np.float64)[:, :2]
            left = np.array(lane["left"], dtype=np.float64)[:, :2]
            right = np.array(lane["right"], dtype=np.float64)[:, :2]
            centerline = _resample_polyline(center, LANE_POINTS + 1)
            left_bound = _resample_polyline(left, LANE_POINTS + 1)
            right_bound = _resample_polyline(right, LANE_POINTS + 1)
            edges = np.stack([centerline, left_bound, right_bound], axis=0)

            point_position[slot] = edges[:, :-1]
            point_vector[slot] = edges[:, 1:] - edges[:, :-1]
            point_orientation[slot] = np.arctan2(
                point_vector[slot, :, :, 1], point_vector[slot, :, :, 0]
            )
            point_side[slot] = np.arange(3)
            polygon_center[slot] = np.array(
                [
                    centerline[LANE_POINTS // 2, 0],
                    centerline[LANE_POINTS // 2, 1],
                    point_orientation[slot, 0, LANE_POINTS // 2],
                ],
                dtype=np.float64,
            )
            polygon_position[slot] = centerline[0]
            polygon_orientation[slot] = point_orientation[slot, 0, 0]
            polygon_type[slot] = self._infer_lane_polygon_type(lane)
            polygon_on_route[slot] = any(
                self._polyline_is_on_route(centerline[:-1], ref_valid_points)
                for ref_valid_points in ref_lines
            )
            speed_limit = float(lane.get("speed_limit") or 0.0)
            polygon_has_speed_limit[slot] = speed_limit > 0
            polygon_speed_limit[slot] = speed_limit
            polygon_road_block_id[slot] = self._safe_int(lane.get("road_id"))
            slot += 1

        for _, polygon_xy in crosswalk_candidates[:MAX_CROSSWALKS]:
            polygon_loop = polygon_xy
            if len(polygon_loop) >= 2 and not np.allclose(polygon_loop[0], polygon_loop[-1]):
                polygon_loop = np.vstack([polygon_loop, polygon_loop[0]])
            centerline = _resample_polyline(polygon_loop, LANE_POINTS + 1)
            left_bound = np.roll(centerline, -1, axis=0)
            right_bound = np.roll(centerline, 1, axis=0)
            edges = np.stack([centerline, left_bound, right_bound], axis=0)

            point_position[slot] = edges[:, :-1]
            point_vector[slot] = edges[:, 1:] - edges[:, :-1]
            point_orientation[slot] = np.arctan2(
                point_vector[slot, :, :, 1], point_vector[slot, :, :, 0]
            )
            point_side[slot] = np.arange(3)
            polygon_center[slot] = np.array(
                [
                    centerline[LANE_POINTS // 2, 0],
                    centerline[LANE_POINTS // 2, 1],
                    point_orientation[slot, 0, LANE_POINTS // 2],
                ],
                dtype=np.float64,
            )
            polygon_position[slot] = centerline[0]
            polygon_orientation[slot] = point_orientation[slot, 0, 0]
            polygon_type[slot] = POLYGON_TYPE_CROSSWALK
            slot += 1

        map_features = {
            "point_position": point_position,
            "point_vector": point_vector,
            "point_side": point_side,
            "point_orientation": point_orientation,
            "polygon_center": polygon_center,
            "polygon_position": polygon_position,
            "polygon_orientation": polygon_orientation,
            "polygon_type": polygon_type,
            "polygon_on_route": polygon_on_route,
            "polygon_tl_status": polygon_tl_status,
            "polygon_has_speed_limit": polygon_has_speed_limit,
            "polygon_speed_limit": polygon_speed_limit,
            "polygon_road_block_id": polygon_road_block_id,
        }
        notes = {
            "lane_candidate_count": len(lane_candidates),
            "lane_kept_count": min(MAX_LANES, len(lane_candidates)),
            "crosswalk_candidate_count": len(crosswalk_candidates),
            "crosswalk_kept_count": min(MAX_CROSSWALKS, len(crosswalk_candidates)),
            "lane_connector_heuristic": "junction_id or non-NO_TURN => lane_connector",
            "traffic_light_status": "all UNKNOWN (Apollo feed not wired to live TL state)",
        }
        return map_features, notes

    @staticmethod
    def _infer_lane_polygon_type(lane: dict) -> int:
        if lane.get("junction_id") or str(lane.get("turn") or "").upper() not in {"", "NO_TURN"}:
            return POLYGON_TYPE_LANE_CONNECTOR
        return POLYGON_TYPE_LANE

    @staticmethod
    def _polyline_is_on_route(points: np.ndarray, ref_valid_points: np.ndarray) -> bool:
        if len(ref_valid_points) == 0 or len(points) == 0:
            return False
        distances = np.linalg.norm(
            points[:, None, :] - ref_valid_points[None, :, :],
            axis=2,
        )
        return bool(float(distances.min()) <= ON_ROUTE_THRESHOLD_M)

    @staticmethod
    def _safe_int(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, (int, np.integer)):
            return int(value)
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return int(digits) if digits else 0

    # ------------------------------------------------- route events (issue #2)

    @staticmethod
    def _route_events(route_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chronologically sorted unique routing events from route.json
        ``all_sequences`` (empty for legacy payloads without event history)."""
        events = route_payload.get("all_sequences") or []
        return sorted(events, key=lambda event: int(event["timestamp_ns"]))

    @classmethod
    def _select_route_event(
        cls, route_payload: Dict[str, Any], frame_ts_ns: int
    ) -> Optional[Dict[str, Any]]:
        """Picks the routing event active at the frame's log time (issue #2).

        Latest event with ``timestamp_ns <= frame_ts_ns``; frames before the
        first event fall back to the earliest one.  Returns None for legacy
        payloads without ``all_sequences`` (callers then keep the clip-level
        payload unchanged)."""
        events = cls._route_events(route_payload)
        if not events:
            return None
        selected = events[0]
        for event in events:
            if int(event["timestamp_ns"]) <= int(frame_ts_ns):
                selected = event
            else:
                break
        return selected

    def _route_payload_at_time(
        self,
        route_payload: Dict[str, Any],
        frame_ts_ns: int,
        map_graph: Optional[dict] = None,
        map_api: Optional[Any] = None,
        cache: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Route payload as of the frame's log time (issue #2).

        Replaces the previous clip-level payload reuse: each frame re-selects
        the latest routing event at its own log timestamp, so a reroute
        published mid-clip switches the route once the sim passes its time.
        The per-event derivation (event fields + lane_ids -> public roadblock
        ids when map_graph/map_api are available) is cached by event
        ``timestamp_ns`` (``cache=clip["route_event_cache"]``): frames that
        keep the same event hit the cache, an event switch recomputes once.
        """
        event = self._select_route_event(route_payload, frame_ts_ns)
        if event is None:
            return route_payload
        key = int(event["timestamp_ns"])
        if cache is not None and key in cache:
            return cache[key]
        effective = dict(route_payload)
        effective["route_lane_ids"] = [
            str(lane_id) for lane_id in event.get("lane_ids", [])
        ]
        effective["destination_xy"] = event.get(
            "destination_xy", route_payload.get("destination_xy")
        )
        effective["selected_event_timestamp_ns"] = key
        effective["selected_event_topic"] = event.get("topic")
        if map_graph is not None and map_api is not None:
            original_roadblock_ids = self._route_roadblocks_from_lane_ids(
                effective["route_lane_ids"], map_graph
            )
            effective["_route_public_roadblock_ids"] = [
                map_api.to_public_roadblock_id(rb_id)
                for rb_id in original_roadblock_ids
            ]
        if cache is not None:
            cache[key] = effective
        return effective

    @staticmethod
    def _load_route_payload(
        parsed_path: Path,
        t0_index: int,
        dataset: Dict[str, Any],
    ) -> Dict[str, Any]:
        route_path = parsed_path / "route.json"
        if not route_path.exists():
            raise FileNotFoundError(
                f"Missing {route_path}; parse route extraction before inference."
            )
        payload = json.loads(route_path.read_text())
        route_t0_index = payload.get("t0_index")
        if route_t0_index is not None and int(route_t0_index) != int(t0_index):
            sample_ts = dataset["sample_ts_sec"]
            raise ValueError(
                "route.json t0 does not match current inference t0. "
                f"route_t0_index={route_t0_index}, infer_t0_index={t0_index}, "
                f"route_t0_sec={payload.get('t0_timestamp_ns', 0) / 1e9:.3f}, "
                f"infer_t0_sec={sample_ts[t0_index]:.3f}"
            )
        return payload

    @staticmethod
    def _route_roadblocks_from_lane_ids(
        route_lane_ids: List[str],
        map_graph: dict,
    ) -> List[str]:
        lane_to_roadblock = map_graph.get("lane_to_roadblock", {})
        ordered: List[str] = []
        seen = set()
        for lane_id in route_lane_ids:
            roadblock_id = lane_to_roadblock.get(lane_id)
            if roadblock_id and roadblock_id not in seen:
                ordered.append(str(roadblock_id))
                seen.add(str(roadblock_id))
        return ordered

    @staticmethod
    def _sample_polyline_at_progress(
        polyline_xy: np.ndarray,
        cumulative: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros((len(targets), 2), dtype=np.float64)
        seg_idx = 0
        for idx, target in enumerate(targets):
            while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < target:
                seg_idx += 1
            if seg_idx + 1 >= len(cumulative):
                out[idx] = polyline_xy[-1]
                continue
            span = cumulative[seg_idx + 1] - cumulative[seg_idx]
            if span <= 1e-9:
                out[idx] = polyline_xy[seg_idx]
                continue
            ratio = (target - cumulative[seg_idx]) / span
            out[idx] = polyline_xy[seg_idx] * (1.0 - ratio) + polyline_xy[seg_idx + 1] * ratio
        return out

    @staticmethod
    def _sample_heading_at_progress(
        headings: np.ndarray,
        cumulative: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros((len(targets),), dtype=np.float64)
        for idx, target in enumerate(targets):
            pos = int(np.searchsorted(cumulative, target, side="right") - 1)
            pos = max(0, min(pos, len(headings) - 1))
            out[idx] = headings[pos]
        return out

register_dataloader("pluto_feature", ApolloPlutoDataloader)
