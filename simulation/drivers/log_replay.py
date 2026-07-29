"""log_replay ego-driver (open_loop): ego follows the bag (GT).

Every `stride`-th log frame becomes a t0; the adapter builds a feature, the
policy runs forward, the post-processor picks a best trajectory, and a frame
is rendered with the raw/best predictions overlaid on the GT world.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from planning.models.pluto.input_builder import HIST_STEPS
from simulation.sim_utils import frame_agents, local_to_global, make_mp4
from .base import EgoDriver, register_ego_driver


class LogReplayDriver(EgoDriver):
    mode = "open_loop"

    def run(self, renderer: Any, frames_dir: Path, mode_name: str, out_dir: Path) -> Dict[str, Any]:
        sim_cfg = self.sim_cfg
        clip = self.clip
        dataset = clip["dataset"]
        n_samples = len(dataset["sample_tokens"])
        ego_dims = np.asarray(dataset["ego_dims"], dtype=np.float64)  # (width, length)
        frames_dir.mkdir(parents=True, exist_ok=True)

        steps = int(sim_cfg["steps"])
        stride = int(sim_cfg.get("stride", 2))
        view_radius = float(sim_cfg.get("view_radius", 50.0))
        start_index = sim_cfg.get("start_index")
        start = start_index if start_index is not None else HIST_STEPS - 1
        indices = list(range(start, n_samples, stride))[:steps]

        records = []
        for frame_no, t0_idx in enumerate(indices):
            t_begin = time.time()
            build = self.adapter.build_frame(clip, t0_idx)
            with torch.inference_mode():
                output = self.policy.infer(build.feature)
            post = None
            if sim_cfg.get("postprocess", True):
                post = self.postprocessor.run(
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

            agents = frame_agents(dataset, t0_idx, ego_pose[:2], view_radius * 1.5)
            speed = float(dataset["ego_speed"][t0_idx])
            t_sec = dataset["sample_ts_sec"][t0_idx] - dataset["sample_ts_sec"][0]
            renderer.render_frame(
                frames_dir / f"frame_{frame_no:05d}.png",
                {
                    "build": build,
                    "output": output,
                    "post": post,
                    "iteration": frame_no,
                    "ego_pose": ego_pose,
                    "ego_dims": ego_dims,
                    "agents": agents,
                    "raw_traj_global": raw_global,
                    "best_traj_global": best_global,
                    "title": f"{clip['config'].get('clip_id', '')} open_loop (ego=log GT)",
                    "info_lines": [
                        f"frame {frame_no:4d}  log_idx {t0_idx:4d}  t={t_sec:6.1f}s",
                        f"ego speed {speed:5.2f} m/s  agents {len(agents):2d}",
                        f"emergency_brake={emergency}",
                    ],
                    "emergency": emergency,
                },
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

        fps = sim_cfg.get("fps") or max(1, round(10 / stride))
        mp4 = make_mp4(frames_dir, out_dir / f"{mode_name}.mp4", fps)
        return {"mode": mode_name, "num_frames": len(indices), "fps": fps, "mp4": mp4, "frames": records}


register_ego_driver("log_replay", LogReplayDriver)
