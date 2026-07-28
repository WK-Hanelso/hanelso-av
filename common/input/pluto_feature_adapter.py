from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from common.input.pluto import (
    CATEGORY_CODES,
    CROSSWALK_POINTS,
    DT,
    HIST_STEPS,
    LANE_POINTS,
    MAP_SELECTION_MARGIN_M,
    MAX_AGENTS,
    MAX_CROSSWALKS,
    MAX_LANES,
    MAX_STATIC,
    ON_ROUTE_THRESHOLD_M,
    PACIFICA_DIMS,
    RADIUS,
    REF_SPACING,
    REF_STEPS,
    STATIC_SPEED_THRESHOLD_MPS,
    PlutoInputBuilder,
    _cumulative_lengths,
    _min_distance_to_polyline,
    _nearest_index,
    _polyline_length,
    _quat_to_yaw,
    _resample_by_spacing,
    _resample_polyline,
    _wrap_angle,
)


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


class ApolloPlutoFeatureAdapter:
    def __init__(self, pluto_root: str) -> None:
        self._builder = PlutoInputBuilder()
        self._pluto_root = Path(pluto_root)

    def _import_pluto_feature(self):
        import sys

        if str(self._pluto_root) not in sys.path:
            sys.path.insert(0, str(self._pluto_root))
        from src.features.pluto_feature import PlutoFeature

        return PlutoFeature

    def build(
        self,
        parsed_dir: str,
        map_graph: dict,
        config: dict,
        t0_time: float | None = None,
    ) -> AdapterBuildResult:
        PlutoFeature = self._import_pluto_feature()
        parsed_path = Path(parsed_dir)
        tables = self._builder._load_tables(parsed_path)
        dataset = self._builder._prepare_dataset(tables, config)
        t0_index, t0_reason = self._builder._select_t0_index(dataset, t0_time)

        raw_data, context = self._build_global_data(dataset, map_graph, t0_index, t0_reason)
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
        )

    def _build_global_data(
        self,
        dataset: Dict[str, Any],
        map_graph: dict,
        t0_index: int,
        t0_reason: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sample_tokens = dataset["sample_tokens"]
        sample_ts_sec = dataset["sample_ts_sec"]
        history_indices, history_deltas = self._builder._history_target_indices(sample_ts_sec, t0_index)
        origin_xy = dataset["ego_positions"][t0_index].copy()
        angle = float(dataset["ego_headings"][t0_index])

        agent_position = np.zeros((MAX_AGENTS, TOTAL_STEPS, 2), dtype=np.float64)
        agent_heading = np.zeros((MAX_AGENTS, TOTAL_STEPS), dtype=np.float64)
        agent_velocity = np.zeros((MAX_AGENTS, TOTAL_STEPS, 2), dtype=np.float64)
        agent_shape = np.zeros((MAX_AGENTS, TOTAL_STEPS, 2), dtype=np.float64)
        agent_category = np.zeros((MAX_AGENTS,), dtype=np.int8)
        agent_valid_mask = np.zeros((MAX_AGENTS, TOTAL_STEPS), dtype=bool)

        ego_positions_hist = dataset["ego_positions"][history_indices]
        ego_headings_hist = dataset["ego_headings"][history_indices]
        ego_speed_hist = dataset["ego_speed"][history_indices]
        ego_vectors_global = np.stack(
            [ego_speed_hist * np.cos(ego_headings_hist), ego_speed_hist * np.sin(ego_headings_hist)],
            axis=1,
        )

        agent_position[0, :HIST_STEPS] = ego_positions_hist
        agent_heading[0, :HIST_STEPS] = ego_headings_hist
        agent_velocity[0, :HIST_STEPS] = ego_vectors_global
        agent_shape[0] = np.array(dataset["ego_dims"], dtype=np.float64)
        agent_category[0] = CATEGORY_CODES["ego"]
        agent_valid_mask[0, :HIST_STEPS] = history_deltas <= (DT * 0.6)

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

        reference_line = self._build_reference_line_global(
            dataset["ego_positions"][t0_index:],
            dataset["ego_headings"][t0_index:],
        )
        map_features, map_notes = self._build_map_features_global(
            map_graph=map_graph,
            ego_global_xy=origin_xy,
            reference_line=reference_line,
        )
        current_state = np.array(
            [
                float(origin_xy[0]),
                float(origin_xy[1]),
                angle,
                float(dataset["ego_speed"][t0_index]),
                float(dataset["ego_accel"][t0_index]),
                float(dataset["ego_steering"][t0_index]),
                float(dataset["ego_ang_vel"][t0_index]),
            ],
            dtype=np.float64,
        )

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
            "vehicle_key": dataset["vehicle_key"],
            "ego_dims": list(dataset["ego_dims"] or PACIFICA_DIMS),
            "t0_index": t0_index,
            "t0_time_sec": sample_ts_sec[t0_index],
            "t0_reason": t0_reason,
            "origin_xy_global": origin_xy.tolist(),
            "angle_rad_global": angle,
            "history_sample_indices": history_indices.tolist(),
            "history_sample_tokens": [sample_tokens[index] for index in history_indices],
            "selected_agent_instance_tokens": selected_agents,
            "selected_static_instance_tokens": selected_static,
            "adapter_notes": [
                "Agent tensors use T=101 with history/present in slots [0:21] and future slots zero-filled with valid_mask=False.",
                "Static objects are approximated from low-speed tracked annotations at t0 and mapped to GENERIC static category.",
                "Reference line is a single R=1 future ego polyline resampled at 1m spacing when richer routing data is unavailable.",
                "Traffic light status is unresolved from the Apollo feed and set to UNKNOWN for every map polygon.",
            ],
            "map_notes": map_notes,
        }
        return raw_data, context

    def _build_reference_line_global(
        self,
        ego_future_xy: np.ndarray,
        ego_future_heading: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        position = np.zeros((1, REF_STEPS, 2), dtype=np.float64)
        vector = np.zeros((1, REF_STEPS, 2), dtype=np.float64)
        orientation = np.zeros((1, REF_STEPS), dtype=np.float64)
        valid_mask = np.zeros((1, REF_STEPS), dtype=bool)
        future_projection = np.zeros((1, 8, 2), dtype=np.float64)

        if len(ego_future_xy) == 0:
            return {
                "position": position,
                "vector": vector,
                "orientation": orientation,
                "valid_mask": valid_mask,
                "future_projection": future_projection,
            }

        sampled_position, sampled_valid = _resample_by_spacing(
            ego_future_xy,
            REF_STEPS,
            REF_SPACING,
        )
        position[0] = sampled_position
        valid_mask[0] = sampled_valid

        valid_indices = np.flatnonzero(sampled_valid)
        for idx in valid_indices:
            if idx + 1 in valid_indices:
                delta = sampled_position[idx + 1] - sampled_position[idx]
            elif idx - 1 in valid_indices:
                delta = sampled_position[idx] - sampled_position[idx - 1]
            else:
                delta = np.array([math.cos(ego_future_heading[0]), math.sin(ego_future_heading[0])])
            vector[0, idx] = delta
            orientation[0, idx] = math.atan2(delta[1], delta[0])

        linestring_cum = _cumulative_lengths(ego_future_xy)
        for proj_idx in range(8):
            future_idx = HIST_STEPS + proj_idx * 10
            if future_idx >= len(ego_future_xy):
                break
            point = ego_future_xy[future_idx]
            nearest = int(np.argmin(np.linalg.norm(ego_future_xy - point[None, :], axis=1)))
            future_projection[0, proj_idx, 0] = float(linestring_cum[nearest])
            future_projection[0, proj_idx, 1] = 0.0

        return {
            "position": position,
            "vector": vector,
            "orientation": orientation,
            "valid_mask": valid_mask,
            "future_projection": future_projection,
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

        ref_valid_points = reference_line["position"][0][reference_line["valid_mask"][0]]
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
            polygon_on_route[slot] = self._polyline_is_on_route(
                centerline[:-1],
                ref_valid_points,
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
