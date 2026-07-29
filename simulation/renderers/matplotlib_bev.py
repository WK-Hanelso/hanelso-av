"""matplotlib BEV renderer — global-UTM crop that follows the ego (C-SWM-018)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.sim_utils import (
    CATEGORY_COLORS,
    STATIC_COLOR,
    UNKNOWN_COLOR,
    calibration_rear_axle_to_center,
    ego_center_from_rear_axle,
    oriented_box_corners,
)
from .base import Renderer, register_renderer


class MatplotlibBEVRenderer(Renderer):
    """Our BEVRenderer with an ego-following crop for frame continuity."""

    suffix = ""

    def __init__(self, clip: Dict[str, Any], map_graph: dict, sim_cfg: Dict[str, Any], mode: str) -> None:
        super().__init__(clip, map_graph, sim_cfg, mode)
        self.view_radius = float(sim_cfg.get("view_radius", 50.0))
        self.rear_axle_to_center = calibration_rear_axle_to_center(
            clip["dataset"]["calibration"]
        )
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

    # ---------------------------------------------------------------- frame

    def render_frame(self, out_path: Path, frame: Dict[str, Any]) -> None:
        build = frame["build"]
        self._render(
            out_path,
            ego_pose=frame["ego_pose"],
            ego_dims=frame["ego_dims"],
            agents=frame["agents"],
            reference_lines=build.scene_context["reference_lines_global"],
            raw_traj_global=frame.get("raw_traj_global"),
            best_traj_global=frame.get("best_traj_global"),
            log_ego_pose=frame.get("log_ego_pose"),
            sim_trace=frame.get("sim_trace"),
            title=frame.get("title", ""),
            info_lines=frame.get("info_lines"),
            emergency=bool(frame.get("emergency", False)),
        )

    def _render(
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


register_renderer("matplotlib", MatplotlibBEVRenderer)
