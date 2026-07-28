from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

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

        raw_data, context = self._build_global_data(
            parsed_path,
            dataset,
            map_graph,
            t0_index,
            t0_reason,
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
        )

    def _build_global_data(
        self,
        parsed_path: Path,
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

        route_payload = self._load_route_payload(parsed_path, t0_index, dataset)
        reference_line, route_debug = self._build_reference_line_global(
            map_graph=map_graph,
            route_payload=route_payload,
            ego_global_xy=dataset["ego_positions"][t0_index],
            ego_heading=dataset["ego_headings"][t0_index],
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
                "Reference line is built from map+route only; ego future poses are not referenced.",
                "reference_line.future_projection is zero-filled intentionally to avoid ego future leakage.",
                "Traffic light status is unresolved from the Apollo feed and set to UNKNOWN for every map polygon.",
            ],
            "map_notes": map_notes,
            "route_debug": route_debug,
        }
        return raw_data, context

    def _build_reference_line_global(
        self,
        map_graph: dict,
        route_payload: Dict[str, Any],
        ego_global_xy: np.ndarray,
        ego_heading: float,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        lane_lookup = {
            str(lane["id"]): lane for lane in map_graph.get("lanes", [])
        }
        route_lane_ids = [str(lane_id) for lane_id in route_payload.get("route_lane_ids", [])]
        if not route_lane_ids:
            raise ValueError(
                "route.json has no route_lane_ids; cannot build leakage-free reference lines."
            )

        route_roadblocks = self._route_roadblocks_from_lane_ids(route_lane_ids, map_graph)
        if not route_roadblocks:
            raise ValueError(
                "route.json route_lane_ids do not map to any roadblocks in map_graph."
            )

        ego_center_xy = ego_global_xy + np.array(
            [
                math.cos(ego_heading) * (PACIFICA_DIMS[1] * 0.5),
                math.sin(ego_heading) * (PACIFICA_DIMS[1] * 0.5),
            ],
            dtype=np.float64,
        )
        candidate_start_lanes = self._get_candidate_starting_lanes(
            ego_center_xy=ego_center_xy,
            ego_rear_axle_xy=ego_global_xy,
            ego_heading=ego_heading,
            map_graph=map_graph,
            route_roadblocks=route_roadblocks,
        )
        if not candidate_start_lanes:
            raise ValueError(
                "No candidate starting lane found within 3m on-route with heading error < pi/2. "
                f"route_roadblocks={route_roadblocks[:8]}"
            )

        discrete_paths: List[np.ndarray] = []
        for start_lane_id in candidate_start_lanes:
            discrete_paths.extend(
                self._find_all_candidate_routes(
                    ego_global_xy=ego_global_xy,
                    start_lane_id=start_lane_id,
                    lane_lookup=lane_lookup,
                    route_roadblocks=route_roadblocks,
                    max_length=120.0,
                    max_depth=15,
                )
            )

        trimmed_paths: List[np.ndarray] = []
        trimmed_lengths: List[float] = []
        for path in discrete_paths:
            trimmed_path, trimmed_length = self._trim_path_from_ego(
                ego_global_xy,
                path,
                length=120.0,
            )
            trimmed_paths.append(trimmed_path)
            trimmed_lengths.append(trimmed_length)

        length_mask = np.array(trimmed_lengths, dtype=np.float64) > 96.0
        if length_mask.any() and not length_mask.all():
            trimmed_paths = [trimmed_paths[i] for i in np.flatnonzero(length_mask)]
            trimmed_lengths = [trimmed_lengths[i] for i in np.flatnonzero(length_mask)]

        merged_paths = self._deduplicate_reference_paths(trimmed_paths)
        if not merged_paths:
            raise ValueError(
                "Reference line route search produced no valid trimmed paths after filtering/dedup."
            )

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
                "route_lane_count": len(route_lane_ids),
                "route_roadblock_count": len(route_roadblocks),
                "candidate_start_lane_ids": candidate_start_lanes,
                "candidate_path_count_raw": len(discrete_paths),
                "candidate_path_count_trimmed": len(trimmed_paths),
                "reference_line_count": len(merged_paths),
                "reference_line_valid_points": packed_counts,
                "selected_route_event_timestamp_ns": route_payload.get("selected_event_timestamp_ns"),
                "selected_route_event_topic": route_payload.get("selected_event_topic"),
                "leakage_free": True,
            },
        )

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

    def _get_candidate_starting_lanes(
        self,
        ego_center_xy: np.ndarray,
        ego_rear_axle_xy: np.ndarray,
        ego_heading: float,
        map_graph: dict,
        route_roadblocks: List[str],
    ) -> List[str]:
        lane_to_roadblock = map_graph.get("lane_to_roadblock", {})
        candidates: List[dict] = []
        for lane in map_graph.get("lanes", []):
            centerline = np.asarray(lane["central"], dtype=np.float64)
            if len(centerline) < 2:
                continue
            left = np.asarray(lane["left"], dtype=np.float64)[:, :2]
            right = np.asarray(lane["right"], dtype=np.float64)[:, :2]
            lane_polygon = np.vstack([left, right[::-1]])
            if len(lane_polygon) < 3:
                continue
            if Polygon(lane_polygon).distance(Point(*ego_center_xy)) > 3.1:
                continue
            if str(lane_to_roadblock.get(lane["id"], "")) not in route_roadblocks:
                continue
            if float(lane.get("length") or 0.0) <= 2.0:
                continue
            if (
                self._get_lane_angle_error(centerline, ego_rear_axle_xy, ego_heading)
                >= math.pi / 2.0
            ):
                continue
            candidates.append(lane)
        return [str(lane["id"]) for lane in candidates]

    def _find_all_candidate_routes(
        self,
        ego_global_xy: np.ndarray,
        start_lane_id: str,
        lane_lookup: Dict[str, dict],
        route_roadblocks: List[str],
        max_length: float,
        max_depth: int,
    ) -> List[np.ndarray]:
        candidate_routes: List[List[str]] = []
        start_lane = lane_lookup[start_lane_id]
        start_centerline = np.asarray(start_lane["central"], dtype=np.float64)
        start_progress = self._project_progress_along_polyline(ego_global_xy, start_centerline[:, :2])
        init_offset = -start_progress
        route_roadblock_set = set(route_roadblocks)
        lane_to_roadblock = {
            lane_id: f"{lane.get('road_id')}#{lane.get('section_id')}"
            for lane_id, lane in lane_lookup.items()
        }

        def dfs(cur_lane_id: str, visited: List[str], length_so_far: float) -> None:
            visited.append(cur_lane_id)
            cur_lane = lane_lookup[cur_lane_id]
            new_length = length_so_far + float(cur_lane.get("length") or 0.0)
            in_route_successors = [
                str(next_lane_id)
                for next_lane_id in cur_lane.get("successor_ids", [])
                if str(lane_to_roadblock.get(str(next_lane_id), "")) in route_roadblock_set
                and str(next_lane_id) in lane_lookup
            ]
            if (
                len(in_route_successors) == 0
                or len(visited) == max_depth
                or new_length > max_length
            ):
                candidate_routes.append(list(visited))
                return
            for next_lane_id in in_route_successors:
                dfs(next_lane_id, visited.copy(), new_length)

        dfs(start_lane_id, [], init_offset)

        candidate_paths: List[np.ndarray] = []
        for lane_ids in candidate_routes:
            discrete_path_parts = [
                self._lane_centerline_with_heading(lane_lookup[lane_id])
                for lane_id in lane_ids
            ]
            candidate_paths.append(self._concat_paths(discrete_path_parts))
        return candidate_paths

    def _trim_path_from_ego(
        self,
        ego_global_xy: np.ndarray,
        path_xyz_heading: np.ndarray,
        length: float,
    ) -> Tuple[np.ndarray, float]:
        if len(path_xyz_heading) == 0:
            return np.zeros((0, 3), dtype=np.float64), 0.0
        path_xy = path_xyz_heading[:, :2]
        cumulative = _cumulative_lengths(path_xy)
        start_progress = float(cumulative[0])
        end_progress = float(cumulative[-1])
        cur_progress = self._project_progress_along_polyline(ego_global_xy, path_xy)
        cut_start = max(start_progress, min(cur_progress, end_progress))
        cur_end = min(cur_progress + length, end_progress)
        path_length = max(0.0, cur_end - cut_start)
        if path_length <= 1e-6:
            return np.zeros((0, 3), dtype=np.float64), 0.0

        targets = np.arange(cut_start, cur_end + 1e-6, 0.25, dtype=np.float64)
        if targets[-1] < cur_end - 1e-6:
            targets = np.append(targets, cur_end)
        trimmed = np.zeros((len(targets), 3), dtype=np.float64)
        trimmed[:, :2] = self._sample_polyline_at_progress(path_xy, cumulative, targets)
        trimmed[:, 2] = self._sample_heading_at_progress(path_xyz_heading[:, 2], cumulative, targets)
        return trimmed, path_length

    @staticmethod
    def _deduplicate_reference_paths(paths: List[np.ndarray]) -> List[np.ndarray]:
        remove_index = set()
        for i in range(len(paths)):
            if i in remove_index:
                continue
            for j in range(i + 1, len(paths)):
                if j in remove_index:
                    continue
                min_len = min(len(paths[i]), len(paths[j]))
                if min_len == 0:
                    continue
                diff = np.abs(paths[i][:min_len, :2] - paths[j][:min_len, :2]).sum(-1)
                if float(np.max(diff)) < 0.5:
                    remove_index.add(j)
        return [paths[i] for i in range(len(paths)) if i not in remove_index]

    @staticmethod
    def _lane_centerline_with_heading(lane: dict) -> np.ndarray:
        centerline = np.asarray(lane["central"], dtype=np.float64)
        headings = np.zeros((len(centerline),), dtype=np.float64)
        if len(centerline) >= 2:
            deltas = np.diff(centerline[:, :2], axis=0)
            headings[:-1] = np.arctan2(deltas[:, 1], deltas[:, 0])
            headings[-1] = headings[-2]
        return np.column_stack([centerline[:, :2], headings])

    @staticmethod
    def _concat_paths(path_parts: List[np.ndarray]) -> np.ndarray:
        if not path_parts:
            return np.zeros((0, 3), dtype=np.float64)
        out = [path_parts[0]]
        for part in path_parts[1:]:
            if len(part) == 0:
                continue
            if len(out[-1]) and np.allclose(out[-1][-1, :2], part[0, :2]):
                out.append(part[1:])
            else:
                out.append(part)
        return np.concatenate(out, axis=0)

    @staticmethod
    def _get_lane_angle_error(
        centerline_xyz: np.ndarray,
        ego_global_xy: np.ndarray,
        ego_heading: float,
    ) -> float:
        subsample = centerline_xyz[::4]
        if len(subsample) < 2:
            subsample = centerline_xyz
        if len(subsample) == 0:
            return math.inf
        if len(subsample) == 1:
            closest_heading = 0.0
        else:
            deltas = np.diff(subsample[:, :2], axis=0)
            headings = np.arctan2(deltas[:, 1], deltas[:, 0])
            headings = np.append(headings, headings[-1])
            distances = np.linalg.norm(subsample[:, :2] - ego_global_xy[None, :], axis=1)
            closest_heading = float(headings[int(np.argmin(distances))])
        return abs(_wrap_angle(closest_heading - ego_heading))

    @staticmethod
    def _project_progress_along_polyline(point_xy: np.ndarray, polyline_xy: np.ndarray) -> float:
        if len(polyline_xy) < 2:
            return 0.0
        cumulative = _cumulative_lengths(polyline_xy)
        best_progress = 0.0
        best_distance = float("inf")
        for idx in range(len(polyline_xy) - 1):
            p0 = polyline_xy[idx]
            p1 = polyline_xy[idx + 1]
            segment = p1 - p0
            seg_len_sq = float(np.dot(segment, segment))
            if seg_len_sq <= 1e-9:
                continue
            t = float(np.dot(point_xy - p0, segment) / seg_len_sq)
            t = max(0.0, min(1.0, t))
            proj = p0 + t * segment
            distance = float(np.linalg.norm(point_xy - proj))
            if distance < best_distance:
                best_distance = distance
                best_progress = float(cumulative[idx] + t * math.sqrt(seg_len_sq))
        return best_progress

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
