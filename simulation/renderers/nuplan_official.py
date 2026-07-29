"""Original pluto NuplanScenarioRender wired as a Renderer (C-SWM-019).

Per frame:
- PlannerInput: SimulationHistoryBuffer accumulated from OUR EgoState /
  DetectionsTracks history (log ego in open_loop, sim ego in closed_loop;
  the renderer reads history[-1], the trail is kept internally) +
  traffic_light_data from scene_context (empty for Apollo clips).
- PlannerInitialization: map_api=ApolloMap, route_roadblock_ids from our
  ScenarioManager, mission_goal=StateSE2(route.json destination_xy).
- renderer.scenario_manager is re-injected with the adapter-built
  ScenarioManager, so `need_update` stays False and the original
  reference-line/route surfaces are reused as-is.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .base import Renderer, register_renderer


def _global_to_local(global_trajectory: np.ndarray, ego_state) -> np.ndarray:
    origin = ego_state.rear_axle.array
    angle = ego_state.rear_axle.heading
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
    heading = global_trajectory[..., 2] - angle
    return np.concatenate([position, heading[..., None]], axis=-1)


def nuplan_overlays(post, output, build):
    """(planning_local, candidates_local, predictions, candidate_index) in the
    ego-local frame, mirroring the PlutoPlanner render call: best trajectory +
    candidates with rule_based_score > 0 (global -> local) + raw model
    predictions for the valid agent rows."""
    ego_state = build.scene_context["ego_state"]
    if post is not None:
        planning_local = np.asarray(post["best_trajectory_local"], dtype=np.float64)
        keep = np.asarray(post["rule_based_scores"]) > 0
        candidates = np.asarray(
            post["candidate_trajectories_global"], dtype=np.float64
        )[keep]
        candidates_local = (
            _global_to_local(candidates, ego_state)
            if len(candidates)
            else None
        )
        candidate_index = int(post["best_candidate_idx"])
    else:
        planning_local = output["output_trajectory"].detach().cpu().numpy()[0]
        candidates_local = None
        candidate_index = None

    rows = np.asarray(build.scene_context["agent_rows"], dtype=np.int64) - 1
    preds_all = output["raw_output"]["output_prediction"].detach().cpu().numpy()[0]
    predictions = preds_all[rows] if len(rows) else None
    return planning_local, candidates_local, predictions, candidate_index


class NuplanOfficialRenderer(Renderer):
    suffix = "_nuplan"

    def __init__(self, clip: Dict[str, Any], map_graph: dict, sim_cfg: Dict[str, Any], mode: str) -> None:
        super().__init__(clip, map_graph, sim_cfg, mode)
        from nuplan.common.actor_state.state_representation import StateSE2
        from simulation.renderers.nuplan_scenario_render import NuplanScenarioRender

        mission_goal_xy = clip["route_payload"].get("destination_xy")
        if mission_goal_xy is None:
            raise ValueError(
                "route.json has no destination_xy: NuplanScenarioRender plots "
                "mission_goal unconditionally, so it cannot be None."
            )
        stride = int(sim_cfg.get("stride", 2))
        dt = float(clip["dt"])
        sample_interval = dt * stride if mode == "open_loop" else dt
        self._mission_goal = StateSE2(
            float(mission_goal_xy[0]), float(mission_goal_xy[1]), 0.0
        )
        self._sample_interval = float(sample_interval)
        self._history_size = int(clip["hist_steps"])
        self._renderer = NuplanScenarioRender(
            vehicle_parameters=clip["vehicle_parameters"]
        )
        self._ego_history: List[Any] = []
        self._obs_history: List[Any] = []

    # ---------------------------------------------------------------- frame

    def render_frame(self, out_path: Path, frame: Dict[str, Any]) -> None:
        build = frame["build"]
        planning_local, candidates_local, predictions, candidate_index = nuplan_overlays(
            frame.get("post"), frame["output"], build
        )
        self._render(
            out_path,
            scene_context=build.scene_context,
            iteration_index=int(frame["iteration"]),
            planning_local=planning_local,
            candidates_local=candidates_local,
            predictions=predictions,
            candidate_index=candidate_index,
        )

    def _render(
        self,
        out_path: Path,
        scene_context: Dict[str, Any],
        iteration_index: int,
        planning_local: Optional[np.ndarray] = None,
        candidates_local: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        candidate_index: Optional[int] = None,
    ) -> None:
        from nuplan.planning.simulation.history.simulation_history_buffer import (
            SimulationHistoryBuffer,
        )
        from nuplan.planning.simulation.planner.abstract_planner import (
            PlannerInitialization,
            PlannerInput,
        )
        from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
            SimulationIteration,
        )

        ego_state = scene_context["ego_state"]
        self._ego_history = (self._ego_history + [ego_state])[-self._history_size :]
        self._obs_history = (self._obs_history + [scene_context["detections"]])[
            -self._history_size :
        ]

        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=len(self._ego_history),
            ego_states=list(self._ego_history),
            observations=list(self._obs_history),
            sample_interval=self._sample_interval,
        )
        current_input = PlannerInput(
            iteration=SimulationIteration(ego_state.time_point, int(iteration_index)),
            history=history,
            traffic_light_data=list(scene_context["traffic_light_data"]),
        )
        scenario_manager = scene_context["scenario_manager"]
        route_roadblock_ids = scenario_manager.get_route_roadblock_ids()
        initialization = PlannerInitialization(
            route_roadblock_ids=route_roadblock_ids,
            mission_goal=self._mission_goal,
            map_api=scene_context["map_api"],
        )
        self._renderer.scenario_manager = scenario_manager

        img = self._renderer.render_from_simulation(
            current_input=current_input,
            initialization=initialization,
            route_roadblock_ids=route_roadblock_ids,
            planning_trajectory=planning_local,
            candidate_trajectories=candidates_local,
            predictions=predictions,
            candidate_index=candidate_index,
            return_img=True,
        )
        plt.imsave(out_path, img)


register_renderer("nuplan", NuplanOfficialRenderer)
