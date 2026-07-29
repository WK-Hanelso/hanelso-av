"""model_driven ego-driver (closed_loop): ego is driven by the model.

Each step builds a feature from the injected sim ego-history (no ego
log/future leakage), runs forward + post-processing, then propagates the ego
one step with the ORIGINAL pluto ForwardSimulator (BatchLQR + kinematic
bicycle).  Agents are re-fetched from the bag with the progress-aligned
hybrid rule (PIPELINE.md 4.4):
    matched = prog_j + argmin ||log_ego[prog_j:prog_j+W] - sim_ego||
    prog_j  = max(matched, prog_j + 1)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from planning.models.pluto.input_builder import DT, HIST_STEPS
from planning.models.pluto.paths import ensure_pluto_on_path
from simulation.sim_utils import (
    PROGRESS_WINDOW,
    build_ego_state_from_array,
    check_collision,
    ego_state_to_pose,
    frame_agents,
    make_mp4,
    pacifica_rear_axle_to_center,
)
from .base import EgoDriver, register_ego_driver


class ModelDrivenDriver(EgoDriver):
    mode = "closed_loop"

    def run(self, renderer: Any, frames_dir: Path, mode_name: str, out_dir: Path) -> Dict[str, Any]:
        ensure_pluto_on_path()
        from src.post_processing.forward_simulation.forward_simulator import ForwardSimulator

        sim_cfg = self.sim_cfg
        clip = self.clip
        adapter = self.adapter
        dataset = clip["dataset"]
        n_samples = len(dataset["sample_tokens"])
        ego_dims = np.asarray(dataset["ego_dims"], dtype=np.float64)
        frames_dir.mkdir(parents=True, exist_ok=True)

        steps = int(sim_cfg["steps"])
        view_radius = float(sim_cfg.get("view_radius", 50.0))
        rear_axle_to_center = pacifica_rear_axle_to_center()
        start_index = sim_cfg.get("start_index")
        start = start_index if start_index is not None else HIST_STEPS - 1
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

        for step in range(steps):
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
                output = self.policy.infer(build.feature)
            post = self.postprocessor.run(
                model_output=output["raw_output"],
                normalized_data=build.normalized_numpy_data,
                scene_context=build.scene_context,
            )
            best_global = np.asarray(post["best_trajectory_global"], dtype=np.float64)
            emergency = bool(post["emergency_brake"])

            # --- propagate ego 1 step with the original ForwardSimulator
            candidate = np.concatenate([pose[None, :], best_global[:, :3]], axis=0)
            if len(candidate) < 81:
                pad = np.repeat(candidate[-1:], 81 - len(candidate), axis=0)
                candidate = np.concatenate([candidate, pad], axis=0)
            rollout = forward_sim.forward(candidate[None, :81], ego_state)
            new_state = rollout[0, 1]
            new_time_us = ego_state.time_point.time_us + int(DT * 1e6)
            new_ego_state = build_ego_state_from_array(new_state, new_time_us)
            new_pose = ego_state_to_pose(new_ego_state)
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
            agents = frame_agents(dataset, prog_j, new_pose[:2], view_radius * 1.5)
            hits = check_collision(new_pose, ego_dims, rear_axle_to_center, agents)
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

            # render at decision time (pre-step ego) for the official nuplan
            # renderer semantics; the matplotlib renderer draws the post-step
            # state with the log ghost + sim trace (unchanged from C-SWM-018).
            renderer.render_frame(
                frames_dir / f"frame_{step:05d}.png",
                {
                    "build": build,
                    "output": output,
                    "post": post,
                    "iteration": step,
                    "ego_pose": new_pose,
                    "ego_dims": ego_dims,
                    "agents": agents,
                    "best_traj_global": best_global,
                    "log_ego_pose": np.array(
                        [
                            dataset["ego_positions"][prog_j][0],
                            dataset["ego_positions"][prog_j][1],
                            dataset["ego_headings"][prog_j],
                        ]
                    ),
                    "sim_trace": np.asarray(sim_trace),
                    "title": f"{clip['config'].get('clip_id', '')} closed_loop (ego=model)",
                    "info_lines": [
                        f"step {step:4d}  prog_j {prog_j:4d}/{n_samples}",
                        f"speed {new_speed:5.2f} m/s  step {step_disp:5.3f} m",
                        f"div(time) {div_time:6.2f} m  div(prog) {div_prog:6.2f} m",
                        f"in_drivable={in_drivable}  ebrake={emergency}  coll={len(hits)}",
                    ],
                    "emergency": emergency,
                },
            )
            ego_state = new_ego_state
            print(
                f"[closed_loop] step {step + 1}/{steps} prog_j={prog_j} "
                f"speed={new_speed:.2f} step={step_disp:.3f}m div_t={div_time:.2f}m "
                f"drivable={in_drivable} ebrake={emergency} coll={len(hits)} "
                f"({records[-1]['wall_time_sec']}s)",
                flush=True,
            )

        fps = sim_cfg.get("fps") or 10
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


register_ego_driver("model_driven", ModelDrivenDriver)
