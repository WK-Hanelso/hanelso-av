"""Deployment post-processing for the PLUTO policy (C-SWM-017).

Reuses the ORIGINAL pluto post-processing stack unchanged:
  - src.post_processing.trajectory_evaluator.TrajectoryEvaluator (8 rule metrics,
    internally rolls candidates out with ForwardSimulator/BatchLQR)
  - src.post_processing.emergency_brake.EmergencyBrake

Inputs are (a) the raw model output dict and (b) the scene context produced by
ApolloPlutoFeatureAdapter (nuPlan EgoState / DetectionsTracks / ScenarioManager
route_lane_dict + drivable_area_map + reference lines).

This is a deployment component shared by the real vehicle and the simulator; it
contains no simulation-loop code — only data in, best trajectory out.
The candidate trimming / agent-info / baseline-path logic is copied verbatim
from pluto/src/planners/pluto_planner.py (constants included) so the scoring
pipeline matches the original PlutoPlanner._run_planning_once.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.special import softmax


class PlutoPostProcessor:
    """Original PlutoPlanner post-processing (trim -> evaluate -> argmax ->
    emergency brake) rewired to our scene context."""

    def __init__(
        self,
        candidate_max_num: int = 20,
        learning_based_score_weight: float = 0.25,
        eval_dt: float = 0.1,
        eval_num_frames: int = 80,
    ) -> None:
        from planning.nuplan_common.post_processing.emergency_brake import EmergencyBrake
        from planning.nuplan_common.post_processing.trajectory_evaluator import TrajectoryEvaluator

        self._eval_dt = float(eval_dt)
        self._eval_num_frames = int(eval_num_frames)
        self._topk = int(candidate_max_num)
        self._learning_based_score_weight = float(learning_based_score_weight)

        self._trajectory_evaluator = TrajectoryEvaluator(eval_dt, eval_num_frames)
        self._emergency_brake = EmergencyBrake()

    # ------------------------------------------------------------------ API

    def run(
        self,
        model_output: Dict[str, Any],
        normalized_data: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        import torch

        def to_numpy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        candidate_trajectories = (
            to_numpy(model_output["candidate_trajectories"])[0].astype(np.float64)
        )
        probability = to_numpy(model_output["probability"])[0]
        predictions = to_numpy(model_output["output_prediction"])[0]
        ref_free_trajectory = (
            to_numpy(model_output["output_ref_free_trajectory"])[0].astype(np.float64)
            if "output_ref_free_trajectory" in model_output
            else None
        )

        ego_state = scene_context["ego_state"]
        hist_steps = int(scene_context["hist_steps"])
        agent_rows: List[int] = list(scene_context["agent_rows"])

        candidate_trajectories, learning_based_score = self._trim_candidates(
            candidate_trajectories,
            probability,
            ego_state,
            ref_free_trajectory,
        )

        agent_data, selected_predictions = self._select_valid_agents(
            normalized_data, predictions, agent_rows, hist_steps
        )
        agent_data["agent_tokens"] = ["ego"] + list(scene_context["agent_tokens"])
        agents_info = self._get_agent_info(agent_data, selected_predictions, ego_state)

        rule_based_scores = self._trajectory_evaluator.evaluate(
            candidate_trajectories=candidate_trajectories,
            init_ego_state=ego_state,
            detections=scene_context["detections"],
            traffic_light_data=scene_context["traffic_light_data"],
            agents_info=agents_info,
            route_lane_dict=scene_context["route_lane_dict"],
            drivable_area_map=scene_context["drivable_area_map"],
            baseline_path=self._get_ego_baseline_path(
                scene_context["reference_lines_global"], ego_state
            ),
        )

        final_scores = (
            rule_based_scores + self._learning_based_score_weight * learning_based_score
        )
        best_candidate_idx = int(final_scores.argmax())

        time_to_at_fault_collision = float(
            self._trajectory_evaluator.time_to_at_fault_collision(best_candidate_idx)
        )
        ebrake_trajectory = self._emergency_brake.brake_if_emergency(
            ego_state,
            time_to_at_fault_collision,
            candidate_trajectories[best_candidate_idx],
        )
        emergency = ebrake_trajectory is not None

        if emergency:
            best_trajectory_global = self._interpolated_trajectory_to_array(
                ebrake_trajectory
            )[1:]
        else:
            best_trajectory_global = candidate_trajectories[best_candidate_idx, 1:]

        best_trajectory_local = self._global_to_local(best_trajectory_global, ego_state)

        drivable_area_map = scene_context["drivable_area_map"]
        in_polygons = drivable_area_map.points_in_polygons(
            best_trajectory_global[:, :2]
        )
        best_in_drivable_fraction = float(in_polygons.any(axis=0).mean())

        multi_metrics = self._trajectory_evaluator._multi_metrics
        weighted_metrics = self._trajectory_evaluator._weighted_metrics

        return {
            "num_candidates": int(len(candidate_trajectories)),
            "best_candidate_idx": best_candidate_idx,
            "rule_based_scores": rule_based_scores,
            "learning_based_scores": np.asarray(learning_based_score, dtype=np.float64),
            "final_scores": final_scores,
            "multi_metrics": {
                "no_collision": multi_metrics[0].copy(),
                "drivable_area": multi_metrics[1].copy(),
                "driving_direction": multi_metrics[2].copy(),
            },
            "weighted_metrics": {
                "progress": weighted_metrics[0].copy(),
                "speed_limit": weighted_metrics[1].copy(),
                "comfortable": weighted_metrics[2].copy(),
                "ttc": weighted_metrics[3].copy(),
            },
            "ego_progress_m": np.asarray(
                self._trajectory_evaluator._ego_progress, dtype=np.float64
            ),
            "time_to_at_fault_collision_s": time_to_at_fault_collision,
            "emergency_brake": emergency,
            "best_trajectory_global": best_trajectory_global,
            "best_trajectory_local": best_trajectory_local,
            "candidate_trajectories_global": candidate_trajectories,
            "best_in_drivable_fraction": best_in_drivable_fraction,
            "num_agents_in_world": int(len(agents_info["tokens"])),
        }

    # ------------------------------------------ verbatim PlutoPlanner logic

    def _trim_candidates(
        self,
        candidate_trajectories: np.ndarray,
        probability: np.ndarray,
        ego_state,
        ref_free_trajectory: Optional[np.ndarray] = None,
    ):
        """Copied from PlutoPlanner._trim_candidates.

        candidate_trajectories: (n_ref, n_mode, 80, 3), probability: (n_ref, n_mode)
        """
        if len(candidate_trajectories.shape) == 4:
            n_ref, n_mode, T, C = candidate_trajectories.shape
            candidate_trajectories = candidate_trajectories.reshape(-1, T, C)
            probability = probability.reshape(-1)

        sorted_idx = np.argsort(-probability)
        sorted_candidate_trajectories = candidate_trajectories[sorted_idx][: self._topk]
        sorted_probability = probability[sorted_idx][: self._topk]
        sorted_probability = softmax(sorted_probability)

        if ref_free_trajectory is not None:
            sorted_candidate_trajectories = np.concatenate(
                [sorted_candidate_trajectories, ref_free_trajectory[None, ...]],
                axis=0,
            )
            sorted_probability = np.concatenate([sorted_probability, [0.25]], axis=0)

        # to global
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        sorted_candidate_trajectories[..., :2] = (
            np.matmul(sorted_candidate_trajectories[..., :2], rot_mat) + origin
        )
        sorted_candidate_trajectories[..., 2] += angle

        sorted_candidate_trajectories = np.concatenate(
            [sorted_candidate_trajectories[..., 0:1, :], sorted_candidate_trajectories],
            axis=-2,
        )

        return sorted_candidate_trajectories, sorted_probability

    @staticmethod
    def _select_valid_agents(
        normalized_data: Dict[str, Any],
        predictions: np.ndarray,
        agent_rows: List[int],
        hist_steps: int,
    ):
        """Builds a pluto-planner-shaped `data` dict from our padded feature
        arrays: row 0 = ego, rows 1..K = agents valid at t0; the time dimension
        is cut to the history window so index -1 is the current frame (the
        adapter pads T to 101 with zero future)."""
        agent = normalized_data["agent"]
        rows = np.array([0] + agent_rows, dtype=np.int64)

        agent_data = {
            "agent": {
                "position": np.asarray(agent["position"])[rows][:, :hist_steps],
                "heading": np.asarray(agent["heading"])[rows][:, :hist_steps],
                "velocity": np.asarray(agent["velocity"])[rows][:, :hist_steps],
                "shape": np.asarray(agent["shape"])[rows][:, :hist_steps],
                "category": np.asarray(agent["category"])[rows],
            },
        }
        selected_predictions = (
            predictions[np.asarray(agent_rows, dtype=np.int64) - 1]
            if len(agent_rows) > 0
            else predictions[:0]
        )
        return agent_data, selected_predictions

    def _get_agent_info(self, data, predictions, ego_state):
        """Copied from PlutoPlanner._get_agent_info (predictions: (n, 80, 2/3/5))."""
        current_velocity = np.linalg.norm(data["agent"]["velocity"][1:, -1], axis=-1)
        current_state = np.concatenate(
            [data["agent"]["position"][1:, -1], data["agent"]["heading"][1:, -1, None]],
            axis=-1,
        )
        velocity = None

        if predictions is None:  # constant velocity
            timesteps = np.linspace(0.1, 8, 80).reshape(1, 80, 1)
            displacement = data["agent"]["velocity"][1:, None, -1] * timesteps
            positions = current_state[:, None, :2] + displacement
            angles = current_state[:, None, 2:3].repeat(80, axis=1)
            predictions = np.concatenate([positions, angles], axis=-1)
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
            velocity = current_velocity[:, None].repeat(81, axis=1)
        elif predictions.shape[-1] == 2:
            predictions = np.concatenate(
                [current_state[:, None, :2], predictions], axis=1
            )
            diff = predictions[:, 1:] - predictions[:, :-1]
            start_end_dist = np.linalg.norm(
                predictions[:, -1, :2] - predictions[:, 0, :2], axis=-1
            )
            near_stop_mask = start_end_dist < 1.0
            angle = np.arctan2(diff[..., 1], diff[..., 0])
            angle = np.concatenate([current_state[:, None, -1], angle], axis=1)
            angle = np.where(
                near_stop_mask[:, None], current_state[:, 2:3].repeat(81, axis=1), angle
            )
            predictions = np.concatenate(
                [predictions[..., :2], angle[..., None]], axis=-1
            )
        elif predictions.shape[-1] == 3:
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
        elif predictions.shape[-1] == 5:
            velocity = np.linalg.norm(predictions[..., 3:5], axis=-1)
            predictions = np.concatenate(
                [current_state[:, None], predictions[..., :3]], axis=1
            )
            velocity = np.concatenate([current_velocity[:, None], velocity], axis=-1)
        else:
            raise ValueError("Invalid prediction shape")

        # to global
        predictions_global = self._local_to_global(predictions, ego_state)

        if velocity is None:
            velocity = (
                np.linalg.norm(np.diff(predictions_global[..., :2], axis=-2), axis=-1)
                / 0.1
            )
            velocity = np.concatenate([current_velocity[..., None], velocity], axis=-1)

        return {
            "tokens": data["agent_tokens"][1:]
            if "agent_tokens" in data
            else [f"agent_{i}" for i in range(len(current_state))],
            "shape": data["agent"]["shape"][1:, -1],
            "category": data["agent"]["category"][1:],
            "velocity": velocity,
            "predictions": predictions_global,
        }

    @staticmethod
    def _get_ego_baseline_path(reference_lines, ego_state):
        """Copied from PlutoPlanner._get_ego_baseline_path."""
        import shapely

        init_ref_points = np.array([r[0] for r in reference_lines], dtype=np.float64)

        init_distance = np.linalg.norm(
            init_ref_points[:, :2] - ego_state.rear_axle.array, axis=-1
        )
        nearest_idx = np.argmin(init_distance)
        reference_line = reference_lines[nearest_idx]
        baseline_path = shapely.LineString(reference_line[:, :2])

        return baseline_path

    @staticmethod
    def _local_to_global(local_trajectory: np.ndarray, ego_state):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(local_trajectory[..., :2], rot_mat) + origin
        heading = local_trajectory[..., 2] + angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    @staticmethod
    def _global_to_local(global_trajectory: np.ndarray, ego_state):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    @staticmethod
    def _interpolated_trajectory_to_array(trajectory) -> np.ndarray:
        states = trajectory.get_sampled_trajectory()
        return np.stack(
            [
                np.array(
                    [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading],
                    dtype=np.float64,
                )
                for state in states
            ],
            axis=0,
        )


from planning.interface import register_postprocessor

register_postprocessor("pluto", PlutoPostProcessor)
