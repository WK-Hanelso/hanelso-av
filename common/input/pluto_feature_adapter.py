from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common.input.pluto import (
    CATEGORY_CODES,
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
    _resample_polyline,
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
    scene_context: Optional[Dict[str, Any]] = None


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

    def _import_scenario_manager(self):
        import sys

        if str(self._pluto_root) not in sys.path:
            sys.path.insert(0, str(self._pluto_root))
        from src.scenario_manager.scenario_manager import ScenarioManager

        return ScenarioManager

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

    def _build_global_data(
        self,
        parsed_path: Path,
        dataset: Dict[str, Any],
        map_graph: dict,
        t0_index: int,
        t0_reason: str,
        map_name: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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
        ego_state = self._build_ego_state(dataset, t0_index)
        scenario_bundle = self._build_scenario_manager(
            map_graph=map_graph,
            map_name=map_name,
            route_payload=route_payload,
            ego_state=ego_state,
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
        (consistent with the whole feature pipeline). Longitudinal acceleration
        is a signed finite-difference of speed; the tire steering angle is a
        kinematic estimate from yaw rate (steering_percentage is not an angle).
        """
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import (
            StateSE2,
            StateVector2D,
            TimePoint,
        )
        from nuplan.common.actor_state.vehicle_parameters import (
            get_pacifica_parameters,
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

        vehicle_parameters = get_pacifica_parameters()
        if speed > 0.5:
            steering_angle = float(
                np.clip(
                    math.atan(vehicle_parameters.wheel_base * yaw_rate / speed),
                    -0.61,
                    0.61,
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
    ) -> Dict[str, Any]:
        """Drives the original pluto ScenarioManager/RouteManager over ApolloMap.

        This replaces the C-SWM-015 reference-line reimplementation (which
        over-generated R by skipping the original "merge repeated lanes"
        candidate-pruning step). The original ScenarioManager output is
        authoritative.
        """
        from common.map.apollo_map import ApolloMap

        ScenarioManager = self._import_scenario_manager()

        route_lane_ids = [
            str(lane_id) for lane_id in route_payload.get("route_lane_ids", [])
        ]
        if not route_lane_ids:
            raise ValueError(
                "route.json has no route_lane_ids; cannot build reference lines."
            )

        apollo_map = ApolloMap(map_name=map_name or None, payload=map_graph)
        original_roadblock_ids = self._route_roadblocks_from_lane_ids(
            route_lane_ids, map_graph
        )
        if not original_roadblock_ids:
            raise ValueError(
                "route.json route_lane_ids do not map to any roadblocks in map_graph."
            )
        public_roadblock_ids = [
            apollo_map.to_public_roadblock_id(rb_id)
            for rb_id in original_roadblock_ids
        ]

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
        out[:, :2] = ApolloPlutoFeatureAdapter._sample_polyline_at_progress(
            xy, cumulative, targets
        )
        out[:, 2] = ApolloPlutoFeatureAdapter._sample_heading_at_progress(
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
