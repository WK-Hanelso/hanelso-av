"""C-SWM-018 simulation render driver: open-loop (log ego) / closed-loop (model ego).

Modes
-----
open_loop   : ego follows the bag (GT).  Every `stride`-th log frame becomes a
              t0; the adapter builds a feature, the policy runs forward, the
              post-processor picks a best trajectory, and a BEV frame is
              rendered with the raw/best predictions overlaid on the GT world.
closed_loop : ego is driven by the model.  Each step builds a feature from the
              injected sim ego-history (no ego log/future leakage), runs
              forward + post-processing, then propagates the ego one step with
              the ORIGINAL pluto ForwardSimulator (BatchLQR + kinematic
              bicycle).  Agents are re-fetched from the bag with the
              progress-aligned hybrid rule (PIPELINE.md 4.4):
                  matched = prog_j + argmin ||log_ego[prog_j:prog_j+W] - sim_ego||
                  prog_j  = max(matched, prog_j + 1)

Renderers (C-SWM-019)
---------------------
matplotlib : our BEVRenderer (default, unchanged).
nuplan     : the ORIGINAL pluto NuplanScenarioRender (official style).  Every
             frame assembles a real PlannerInput (SimulationHistoryBuffer from
             our ego/detection history + traffic_light_data) and a
             PlannerInitialization (map_api=ApolloMap, route_roadblock_ids,
             mission_goal=route.json destination_xy), injects our
             ScenarioManager, and calls render_from_simulation with the
             post-processor best trajectory / candidates / predictions in the
             ego-local frame exactly like PlutoPlanner._run_planning_once.

Outputs: work/sim/<clip>/<mode>[_nuplan]/frame_%05d.png
         + work/sim/<clip>/<mode>[_nuplan].mp4
         + <mode>[_nuplan]_metrics.json (per-step metrics, divergence, collisions).

Run with system python3 (torch 1.12 + natten + nuplan + shapely).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.input.pluto import DT, HIST_STEPS
from common.input.pluto_feature_adapter import ApolloPlutoFeatureAdapter
from common.policy import get_policy
import common.policy.pluto_torch  # noqa: F401  (registers pluto_torch)
from common.policy.pluto_postprocess import PlutoPostProcessor

PROGRESS_WINDOW = 80  # frames scanned ahead for progress alignment

CATEGORY_COLORS = {
    "vehicle": "#2563eb",
    "pedestrian": "#f97316",
    "bicycle": "#10b981",
}
STATIC_COLOR = "#b45309"
UNKNOWN_COLOR = "#9ca3af"


# --------------------------------------------------------------------- utils


def local_to_global(local_traj: np.ndarray, origin_xy: np.ndarray, angle: float) -> np.ndarray:
    """Ego-frame (x, y[, heading]) -> global, same convention as pluto."""
    rot = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    out = np.array(local_traj, dtype=np.float64, copy=True)
    out[..., :2] = local_traj[..., :2] @ rot + origin_xy
    if out.shape[-1] > 2:
        out[..., 2] = local_traj[..., 2] + angle
    return out


def oriented_box_corners(cx: float, cy: float, heading: float, width: float, length: float) -> np.ndarray:
    """(4, 2) corners of a box centered at (cx, cy)."""
    c, s = math.cos(heading), math.sin(heading)
    dx, dy = length / 2.0, width / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def ego_center_from_rear_axle(x: float, y: float, heading: float, rear_axle_to_center: float):
    return (
        x + rear_axle_to_center * math.cos(heading),
        y + rear_axle_to_center * math.sin(heading),
    )


def frame_agents(dataset: Dict[str, Any], frame_index: int, center_xy: np.ndarray, radius: float) -> List[Dict[str, Any]]:
    """All annotated tracks present at a log frame within `radius` of center."""
    token = dataset["sample_tokens"][frame_index]
    agents = []
    for instance_token, track in dataset["track_data"].items():
        ann_idx = track["sample_to_index"].get(token)
        if ann_idx is None:
            continue
        pos = track["positions"][ann_idx]
        if float(np.linalg.norm(pos - center_xy)) > radius:
            continue
        speed = float(np.linalg.norm(track["velocities"][ann_idx]))
        agents.append(
            {
                "token": instance_token,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "heading": float(track["headings"][ann_idx]),
                "width": float(track["widths"][ann_idx]),
                "length": float(track["lengths"][ann_idx]),
                "category": str(track["category"]),
                "speed": speed,
            }
        )
    return agents


def make_mp4(frames_dir: Path, out_path: Path, fps: int) -> Dict[str, Any]:
    """Stitches frame_%05d.png into an mp4 with the system ffmpeg binary."""
    pattern = str(frames_dir / "frame_%05d.png")
    attempts = []
    for codec in ("libx264", "mpeg4"):
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        attempts.append({"codec": codec, "returncode": proc.returncode, "stderr": proc.stderr[-500:]})
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return {"ok": True, "codec": codec, "path": str(out_path), "size_bytes": out_path.stat().st_size}
    return {"ok": False, "attempts": attempts}


# ------------------------------------------------------------------ renderer


class BEVRenderer:
    """Global-UTM BEV renderer with an ego-following crop for frame continuity."""

    def __init__(self, map_graph: dict, view_radius: float = 50.0, rear_axle_to_center: float = 1.461) -> None:
        self.view_radius = float(view_radius)
        self.rear_axle_to_center = float(rear_axle_to_center)
        self._lanes = []
        for lane in map_graph.get("lanes", []):
            central = np.asarray([p[:2] for p in lane["central"]], dtype=np.float64)
            left = np.asarray([p[:2] for p in lane["left"]], dtype=np.float64)
            right = np.asarray([p[:2] for p in lane["right"]], dtype=np.float64)
            bbox = (
                central[:, 0].min(),
                central[:, 0].max(),
                central[:, 1].min(),
                central[:, 1].max(),
            )
            self._lanes.append((central, left, right, bbox))
        self._crosswalks = []
        for polygon in map_graph.get("crosswalks", {}).values():
            pts = np.asarray([p[:2] for p in polygon], dtype=np.float64)
            if len(pts) >= 2 and not np.allclose(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[0]])
            bbox = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
            self._crosswalks.append((pts, bbox))

    def _visible(self, bbox, cx: float, cy: float) -> bool:
        margin = self.view_radius + 10.0
        return not (
            bbox[1] < cx - margin
            or bbox[0] > cx + margin
            or bbox[3] < cy - margin
            or bbox[2] > cy + margin
        )

    def render(
        self,
        out_path: Path,
        ego_pose: np.ndarray,
        ego_dims: np.ndarray,
        agents: List[Dict[str, Any]],
        reference_lines: Optional[List[np.ndarray]] = None,
        raw_traj_global: Optional[np.ndarray] = None,
        best_traj_global: Optional[np.ndarray] = None,
        log_ego_pose: Optional[np.ndarray] = None,
        sim_trace: Optional[np.ndarray] = None,
        title: str = "",
        info_lines: Optional[List[str]] = None,
        emergency: bool = False,
    ) -> None:
        ex, ey, eh = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])
        fig, ax = plt.subplots(figsize=(8, 8), dpi=110)

        for central, left, right, bbox in self._lanes:
            if not self._visible(bbox, ex, ey):
                continue
            ax.plot(central[:, 0], central[:, 1], color="#d8dee6", linewidth=0.8, alpha=0.8, zorder=1)
            ax.plot(left[:, 0], left[:, 1], color="#c3cad4", linewidth=0.6, alpha=0.6, zorder=1)
            ax.plot(right[:, 0], right[:, 1], color="#c3cad4", linewidth=0.6, alpha=0.6, zorder=1)
        for pts, bbox in self._crosswalks:
            if not self._visible(bbox, ex, ey):
                continue
            ax.fill(pts[:, 0], pts[:, 1], color="#e9d8fd", alpha=0.4, zorder=1)

        if reference_lines:
            for ref in reference_lines:
                ax.plot(ref[:, 0], ref[:, 1], color="#60a5fa", linewidth=1.6, alpha=0.85, zorder=2)

        for agent in agents:
            color = CATEGORY_COLORS.get(agent["category"], UNKNOWN_COLOR)
            if agent["speed"] <= 0.5:
                color = STATIC_COLOR if agent["category"] not in CATEGORY_COLORS else color
            corners = oriented_box_corners(
                agent["x"], agent["y"], agent["heading"], agent["width"], agent["length"]
            )
            ax.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.55, zorder=3)
            ax.plot(
                np.append(corners[:, 0], corners[0, 0]),
                np.append(corners[:, 1], corners[0, 1]),
                color=color,
                linewidth=1.0,
                zorder=3,
            )

        if sim_trace is not None and len(sim_trace) > 1:
            ax.plot(sim_trace[:, 0], sim_trace[:, 1], color="#111827", linewidth=1.2, alpha=0.6, zorder=4)

        if raw_traj_global is not None:
            ax.plot(
                raw_traj_global[:, 0],
                raw_traj_global[:, 1],
                color="#dc2626",
                linewidth=2.2,
                alpha=0.9,
                zorder=5,
                label="raw output_trajectory",
            )
        if best_traj_global is not None:
            label = "post best" + (" (E-BRAKE)" if emergency else "")
            ax.plot(
                best_traj_global[:, 0],
                best_traj_global[:, 1],
                color="#16a34a",
                linewidth=2.0,
                linestyle="--",
                alpha=0.95,
                zorder=6,
                label=label,
            )

        if log_ego_pose is not None:
            gx, gy = ego_center_from_rear_axle(
                float(log_ego_pose[0]), float(log_ego_pose[1]), float(log_ego_pose[2]), self.rear_axle_to_center
            )
            ghost = oriented_box_corners(gx, gy, float(log_ego_pose[2]), float(ego_dims[0]), float(ego_dims[1]))
            ax.plot(
                np.append(ghost[:, 0], ghost[0, 0]),
                np.append(ghost[:, 1], ghost[0, 1]),
                color="#6b7280",
                linewidth=1.2,
                linestyle=":",
                zorder=6,
                label="log ego (ghost)",
            )

        ecx, ecy = ego_center_from_rear_axle(ex, ey, eh, self.rear_axle_to_center)
        ego_corners = oriented_box_corners(ecx, ecy, eh, float(ego_dims[0]), float(ego_dims[1]))
        ego_color = "#b91c1c" if emergency else "#111827"
        ax.fill(ego_corners[:, 0], ego_corners[:, 1], color=ego_color, alpha=0.85, zorder=7)
        ax.plot([ex, ecx + 0.1 * math.cos(eh)], [ey, ecy + 0.1 * math.sin(eh)], color="#facc15", linewidth=1.2, zorder=8)

        ax.set_xlim(ex - self.view_radius, ex + self.view_radius)
        ax.set_ylim(ey - self.view_radius, ey + self.view_radius)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3, alpha=0.3)
        ax.set_title(title, fontsize=10)
        if info_lines:
            ax.text(
                0.02,
                0.98,
                "\n".join(info_lines),
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#d1d5db"},
            )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


# ----------------------------------------------- official nuplan renderer


class NuplanOfficialRenderer:
    """Drives the ORIGINAL pluto NuplanScenarioRender (C-SWM-019).

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

    def __init__(
        self,
        mission_goal_xy,
        sample_interval: float,
        history_size: int = HIST_STEPS,
    ) -> None:
        from nuplan.common.actor_state.state_representation import StateSE2
        from src.feature_builders.nuplan_scenario_render import NuplanScenarioRender

        if mission_goal_xy is None:
            raise ValueError(
                "route.json has no destination_xy: NuplanScenarioRender plots "
                "mission_goal unconditionally, so it cannot be None."
            )
        self._mission_goal = StateSE2(
            float(mission_goal_xy[0]), float(mission_goal_xy[1]), 0.0
        )
        self._sample_interval = float(sample_interval)
        self._history_size = int(history_size)
        self._renderer = NuplanScenarioRender()
        self._ego_history: List[Any] = []
        self._obs_history: List[Any] = []

    def render_frame(
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
            PlutoPostProcessor._global_to_local(candidates, ego_state)
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


# ------------------------------------------------------------ ego dynamics


def build_ego_state_from_array(state: np.ndarray, time_point_us: int):
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import (
        StateSE2,
        StateVector2D,
        TimePoint,
    )
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(float(state[0]), float(state[1]), float(state[2])),
        rear_axle_velocity_2d=StateVector2D(float(state[3]), float(state[4])),
        rear_axle_acceleration_2d=StateVector2D(float(state[5]), float(state[6])),
        tire_steering_angle=float(np.clip(state[7], -0.61, 0.61)),
        time_point=TimePoint(int(time_point_us)),
        vehicle_parameters=get_pacifica_parameters(),
        angular_vel=float(state[9]),
    )


def ego_state_to_pose(ego_state) -> np.ndarray:
    return np.array(
        [ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading],
        dtype=np.float64,
    )


def check_collision(ego_pose: np.ndarray, ego_dims: np.ndarray, rear_axle_to_center: float, agents: List[Dict[str, Any]]) -> List[str]:
    """Oriented-box overlap between the sim ego and log agents (shapely)."""
    import shapely

    ecx, ecy = ego_center_from_rear_axle(
        float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2]), rear_axle_to_center
    )
    ego_poly = shapely.Polygon(
        oriented_box_corners(ecx, ecy, float(ego_pose[2]), float(ego_dims[0]), float(ego_dims[1]))
    )
    hits = []
    for agent in agents:
        poly = shapely.Polygon(
            oriented_box_corners(agent["x"], agent["y"], agent["heading"], agent["width"], agent["length"])
        )
        if ego_poly.intersects(poly):
            hits.append(agent["token"])
    return hits


# ---------------------------------------------------------------- pipelines


def run_open_loop(args, adapter, clip, policy, postprocessor, renderer, out_dir: Path, official=None) -> Dict[str, Any]:
    dataset = clip["dataset"]
    n_samples = len(dataset["sample_tokens"])
    ego_dims = np.asarray(dataset["ego_dims"], dtype=np.float64)  # (width, length)
    mode_name = "open_loop_nuplan" if official is not None else "open_loop"
    frames_dir = out_dir / mode_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    start = args.start_index if args.start_index is not None else HIST_STEPS - 1
    indices = list(range(start, n_samples, args.stride))[: args.steps]

    records = []
    for frame_no, t0_idx in enumerate(indices):
        t_begin = time.time()
        build = adapter.build_frame(clip, t0_idx)
        with torch.inference_mode():
            output = policy.infer(build.feature)
        post = None
        if args.postprocess:
            post = postprocessor.run(
                model_output=output["raw_output"],
                normalized_data=build.normalized_numpy_data,
                scene_context=build.scene_context,
            )

        ego_pose = np.array(
            [
                dataset["ego_positions"][t0_idx][0],
                dataset["ego_positions"][t0_idx][1],
                dataset["ego_headings"][t0_idx],
            ],
            dtype=np.float64,
        )
        raw_local = output["output_trajectory"].detach().cpu().numpy()[0]
        raw_global = local_to_global(raw_local, ego_pose[:2], float(ego_pose[2]))
        best_global = post["best_trajectory_global"] if post is not None else None
        emergency = bool(post["emergency_brake"]) if post is not None else False

        agents = frame_agents(dataset, t0_idx, ego_pose[:2], renderer.view_radius * 1.5)
        speed = float(dataset["ego_speed"][t0_idx])
        t_sec = dataset["sample_ts_sec"][t0_idx] - dataset["sample_ts_sec"][0]
        if official is not None:
            planning_local, candidates_local, predictions, candidate_index = (
                nuplan_overlays(post, output, build)
            )
            official.render_frame(
                frames_dir / f"frame_{frame_no:05d}.png",
                scene_context=build.scene_context,
                iteration_index=frame_no,
                planning_local=planning_local,
                candidates_local=candidates_local,
                predictions=predictions,
                candidate_index=candidate_index,
            )
        else:
            renderer.render(
                frames_dir / f"frame_{frame_no:05d}.png",
                ego_pose=ego_pose,
                ego_dims=ego_dims,
                agents=agents,
                reference_lines=build.scene_context["reference_lines_global"],
                raw_traj_global=raw_global,
                best_traj_global=best_global,
                title=f"{clip['config'].get('clip_id', '')} open_loop (ego=log GT)",
                info_lines=[
                    f"frame {frame_no:4d}  log_idx {t0_idx:4d}  t={t_sec:6.1f}s",
                    f"ego speed {speed:5.2f} m/s  agents {len(agents):2d}",
                    f"emergency_brake={emergency}",
                ],
                emergency=emergency,
            )
        records.append(
            {
                "frame": frame_no,
                "log_index": t0_idx,
                "t_rel_sec": round(t_sec, 3),
                "ego_speed_mps": round(speed, 3),
                "num_agents_rendered": len(agents),
                "emergency_brake": emergency,
                "wall_time_sec": round(time.time() - t_begin, 3),
            }
        )
        print(
            f"[open_loop] frame {frame_no + 1}/{len(indices)} log_idx={t0_idx} "
            f"speed={speed:.2f} agents={len(agents)} ({records[-1]['wall_time_sec']}s)",
            flush=True,
        )

    fps = args.fps or max(1, round(10 / args.stride))
    mp4 = make_mp4(frames_dir, out_dir / f"{mode_name}.mp4", fps)
    return {"mode": mode_name, "num_frames": len(indices), "fps": fps, "mp4": mp4, "frames": records}


def run_closed_loop(args, adapter, clip, policy, postprocessor, renderer, out_dir: Path, official=None) -> Dict[str, Any]:
    from src.post_processing.forward_simulation.forward_simulator import ForwardSimulator

    dataset = clip["dataset"]
    n_samples = len(dataset["sample_tokens"])
    ego_dims = np.asarray(dataset["ego_dims"], dtype=np.float64)
    mode_name = "closed_loop_nuplan" if official is not None else "closed_loop"
    frames_dir = out_dir / mode_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    start = args.start_index if args.start_index is not None else HIST_STEPS - 1
    ts = dataset["sample_ts_sec"]

    # --- init sim ego history from the log (past only, ends at `start`)
    hist_idx, hist_deltas = adapter._builder._history_target_indices(ts, start)
    positions = [dataset["ego_positions"][i].copy() for i in hist_idx]
    headings = [float(dataset["ego_headings"][i]) for i in hist_idx]
    velocities = [
        np.array(
            [
                dataset["ego_speed"][i] * math.cos(dataset["ego_headings"][i]),
                dataset["ego_speed"][i] * math.sin(dataset["ego_headings"][i]),
            ],
            dtype=np.float64,
        )
        for i in hist_idx
    ]
    valid = list(hist_deltas <= (DT * 0.6))

    ego_state = adapter._build_ego_state(dataset, start)
    forward_sim = ForwardSimulator(dt=DT, num_frames=1)

    prog_j = start
    sim_trace = [ego_state_to_pose(ego_state)]
    records: List[Dict[str, Any]] = []
    collisions_total: List[Dict[str, Any]] = []

    for step in range(args.steps):
        t_begin = time.time()
        pose = ego_state_to_pose(ego_state)

        # --- progress-aligned agent fetch (PIPELINE.md 4.4 hybrid rule)
        if step > 0:
            hi = min(prog_j + PROGRESS_WINDOW, n_samples)
            window = dataset["ego_positions"][prog_j:hi]
            matched = prog_j + int(
                np.argmin(np.linalg.norm(window - pose[:2], axis=1))
            )
            prog_j = min(max(matched, prog_j + 1), n_samples - 1)

        # --- feature from sim ego history + log agents at prog_j
        dyn = ego_state.dynamic_car_state
        speed = float(np.hypot(dyn.rear_axle_velocity_2d.x, dyn.rear_axle_velocity_2d.y))
        sim_ego = {
            "position": np.asarray(positions, dtype=np.float64),
            "heading": np.asarray(headings, dtype=np.float64),
            "velocity_global": np.asarray(velocities, dtype=np.float64),
            "valid_mask": np.asarray(valid, dtype=bool),
            "current_state": np.array(
                [
                    pose[0],
                    pose[1],
                    pose[2],
                    speed,
                    float(dyn.rear_axle_acceleration_2d.x),
                    float(ego_state.tire_steering_angle),
                    float(dyn.angular_velocity),
                ],
                dtype=np.float64,
            ),
            "ego_state": ego_state,
        }
        build = adapter.build_frame(clip, prog_j, sim_ego=sim_ego)
        with torch.inference_mode():
            output = policy.infer(build.feature)
        post = postprocessor.run(
            model_output=output["raw_output"],
            normalized_data=build.normalized_numpy_data,
            scene_context=build.scene_context,
        )
        best_global = np.asarray(post["best_trajectory_global"], dtype=np.float64)
        emergency = bool(post["emergency_brake"])

        # official renderer draws at decision time (pre-step ego), exactly
        # like PlutoPlanner renders inside compute_planner_trajectory.
        if official is not None:
            planning_local, candidates_local, predictions, candidate_index = (
                nuplan_overlays(post, output, build)
            )
            official.render_frame(
                frames_dir / f"frame_{step:05d}.png",
                scene_context=build.scene_context,
                iteration_index=step,
                planning_local=planning_local,
                candidates_local=candidates_local,
                predictions=predictions,
                candidate_index=candidate_index,
            )

        # --- propagate ego 1 step with the original ForwardSimulator
        candidate = np.concatenate([pose[None, :], best_global[:, :3]], axis=0)
        if len(candidate) < 81:
            pad = np.repeat(candidate[-1:], 81 - len(candidate), axis=0)
            candidate = np.concatenate([candidate, pad], axis=0)
        rollout = forward_sim.forward(candidate[None, :81], ego_state)
        new_state = rollout[0, 1]
        new_time_us = ego_state.time_point.time_us + int(DT * 1e6)
        ego_state = build_ego_state_from_array(new_state, new_time_us)
        new_pose = ego_state_to_pose(ego_state)
        sim_trace.append(new_pose)

        # history: drop oldest, append the new sim state
        heading_new = float(new_state[2])
        rot = np.array(
            [
                [math.cos(heading_new), -math.sin(heading_new)],
                [math.sin(heading_new), math.cos(heading_new)],
            ]
        )
        vel_global = rot @ np.array([new_state[3], new_state[4]], dtype=np.float64)
        positions = positions[1:] + [new_pose[:2].copy()]
        headings = headings[1:] + [heading_new]
        velocities = velocities[1:] + [vel_global]
        valid = valid[1:] + [True]

        # --- metrics
        step_disp = float(np.linalg.norm(new_pose[:2] - pose[:2]))
        time_idx = min(start + step + 1, n_samples - 1)
        div_time = float(np.linalg.norm(new_pose[:2] - dataset["ego_positions"][time_idx]))
        div_prog = float(np.linalg.norm(new_pose[:2] - dataset["ego_positions"][prog_j]))
        dam = build.scene_context["drivable_area_map"]
        in_drivable = bool(dam.points_in_polygons(new_pose[None, :2]).any())
        agents = frame_agents(dataset, prog_j, new_pose[:2], renderer.view_radius * 1.5)
        hits = check_collision(new_pose, ego_dims, renderer.rear_axle_to_center, agents)
        if hits:
            collisions_total.append({"step": step, "tokens": hits})

        new_speed = float(np.hypot(new_state[3], new_state[4]))
        records.append(
            {
                "step": step,
                "prog_j": prog_j,
                "sim_xy": [round(float(new_pose[0]), 3), round(float(new_pose[1]), 3)],
                "speed_mps": round(new_speed, 3),
                "step_disp_m": round(step_disp, 4),
                "divergence_time_aligned_m": round(div_time, 3),
                "divergence_progress_aligned_m": round(div_prog, 3),
                "in_drivable": in_drivable,
                "emergency_brake": emergency,
                "collision_tokens": hits,
                "wall_time_sec": round(time.time() - t_begin, 3),
            }
        )

        if official is None:
            renderer.render(
                frames_dir / f"frame_{step:05d}.png",
                ego_pose=new_pose,
                ego_dims=ego_dims,
                agents=agents,
                reference_lines=build.scene_context["reference_lines_global"],
                raw_traj_global=None,
                best_traj_global=best_global,
                log_ego_pose=np.array(
                    [
                        dataset["ego_positions"][prog_j][0],
                        dataset["ego_positions"][prog_j][1],
                        dataset["ego_headings"][prog_j],
                    ]
                ),
                sim_trace=np.asarray(sim_trace),
                title=f"{clip['config'].get('clip_id', '')} closed_loop (ego=model)",
                info_lines=[
                    f"step {step:4d}  prog_j {prog_j:4d}/{n_samples}",
                    f"speed {new_speed:5.2f} m/s  step {step_disp:5.3f} m",
                    f"div(time) {div_time:6.2f} m  div(prog) {div_prog:6.2f} m",
                    f"in_drivable={in_drivable}  ebrake={emergency}  coll={len(hits)}",
                ],
                emergency=emergency,
            )
        print(
            f"[closed_loop] step {step + 1}/{args.steps} prog_j={prog_j} "
            f"speed={new_speed:.2f} step={step_disp:.3f}m div_t={div_time:.2f}m "
            f"drivable={in_drivable} ebrake={emergency} coll={len(hits)} "
            f"({records[-1]['wall_time_sec']}s)",
            flush=True,
        )

    fps = args.fps or 10
    mp4 = make_mp4(frames_dir, out_dir / f"{mode_name}.mp4", fps)

    step_disps = np.array([r["step_disp_m"] for r in records])
    summary = {
        "steps_completed": len(records),
        "final_divergence_time_aligned_m": records[-1]["divergence_time_aligned_m"] if records else None,
        "final_divergence_progress_aligned_m": records[-1]["divergence_progress_aligned_m"] if records else None,
        "max_divergence_time_aligned_m": max(r["divergence_time_aligned_m"] for r in records) if records else None,
        "max_step_disp_m": float(step_disps.max()) if len(step_disps) else None,
        "mean_step_disp_m": float(step_disps.mean()) if len(step_disps) else None,
        "steps_out_of_drivable": sum(1 for r in records if not r["in_drivable"]),
        "emergency_brake_steps": sum(1 for r in records if r["emergency_brake"]),
        "collision_events": collisions_total,
        "num_collision_steps": len(collisions_total),
    }
    return {
        "mode": mode_name,
        "num_frames": len(records),
        "fps": fps,
        "mp4": mp4,
        "summary": summary,
        "steps": records,
    }


# --------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["open_loop", "closed_loop"])
    parser.add_argument(
        "--renderer",
        default="matplotlib",
        choices=["matplotlib", "nuplan"],
        help="matplotlib: our BEVRenderer (default). nuplan: original pluto NuplanScenarioRender (official style).",
    )
    parser.add_argument("--parsed-dir", required=True)
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--vehicle", default="pacifica")
    parser.add_argument("--start-index", type=int, default=None, help="log frame index used as t_start (default: first frame with full history)")
    parser.add_argument("--steps", type=int, default=100, help="number of rendered frames (open) / sim steps (closed)")
    parser.add_argument("--stride", type=int, default=2, help="open_loop: log frames advanced per rendered frame")
    parser.add_argument("--fps", type=int, default=None, help="mp4 fps (default: 10/stride open, 10 closed)")
    parser.add_argument("--view-radius", type=float, default=50.0)
    parser.add_argument("--policy", default="pluto_torch")
    parser.add_argument("--config-path", default="code/hydra/config.yaml")
    parser.add_argument("--checkpoint-path", default="data/model/v3_pluto.ckpt")
    parser.add_argument("--pluto-root", default=str(REPO_ROOT / "third_party" / "pluto"))
    parser.add_argument("--out-root", default="work/sim")
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open_loop only: run post-processing for the best-trajectory overlay (closed_loop always does)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parsed_dir = Path(args.parsed_dir)
    clip_id = args.clip_id or parsed_dir.parent.name
    out_dir = Path(args.out_root) / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.map_path) as f:
        map_graph = json.load(f)

    config = {"clip_id": clip_id, "map_name": args.map_name, "vehicle": args.vehicle}
    adapter = ApolloPlutoFeatureAdapter(args.pluto_root)
    clip = adapter.prepare_clip(str(parsed_dir), map_graph, config)

    policy = get_policy(args.policy)(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        pluto_root=args.pluto_root,
        use_v3_planning_decoder=True,
    )
    postprocessor = PlutoPostProcessor(pluto_root=args.pluto_root)

    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    renderer = BEVRenderer(
        map_graph,
        view_radius=args.view_radius,
        rear_axle_to_center=float(get_pacifica_parameters().rear_axle_to_center),
    )

    official = None
    if args.renderer == "nuplan":
        sample_interval = DT * args.stride if args.mode == "open_loop" else DT
        official = NuplanOfficialRenderer(
            mission_goal_xy=clip["route_payload"].get("destination_xy"),
            sample_interval=sample_interval,
        )

    if args.mode == "open_loop":
        result = run_open_loop(args, adapter, clip, policy, postprocessor, renderer, out_dir, official=official)
    else:
        result = run_closed_loop(args, adapter, clip, policy, postprocessor, renderer, out_dir, official=official)

    result["clip_id"] = clip_id
    result["args"] = {k: str(v) for k, v in vars(args).items()}
    metrics_path = out_dir / f"{result['mode']}_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "steps")}, indent=2, ensure_ascii=False))
    print(f"metrics: {metrics_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
