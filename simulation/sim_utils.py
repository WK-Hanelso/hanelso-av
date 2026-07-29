"""Shared simulation helpers: geometry, log-agent fetch, ego dynamics, mp4."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from calibration.vehicle import max_tire_angle, to_vehicle_parameters

PROGRESS_WINDOW = 80  # frames scanned ahead for progress alignment

CATEGORY_COLORS = {
    "vehicle": "#2563eb",
    "pedestrian": "#f97316",
    "bicycle": "#10b981",
}
STATIC_COLOR = "#b45309"
UNKNOWN_COLOR = "#9ca3af"


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


def calibration_rear_axle_to_center(calib: Dict[str, Any]) -> float:
    return float(to_vehicle_parameters(calib).rear_axle_to_center)


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


def build_ego_state_from_array(
    state: np.ndarray,
    time_point_us: int,
    calib: Dict[str, Any],
):
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import (
        StateSE2,
        StateVector2D,
        TimePoint,
    )
    vehicle_parameters = to_vehicle_parameters(calib)

    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(float(state[0]), float(state[1]), float(state[2])),
        rear_axle_velocity_2d=StateVector2D(float(state[3]), float(state[4])),
        rear_axle_acceleration_2d=StateVector2D(float(state[5]), float(state[6])),
        tire_steering_angle=float(
            np.clip(state[7], -max_tire_angle(calib), max_tire_angle(calib))
        ),
        time_point=TimePoint(int(time_point_us)),
        vehicle_parameters=vehicle_parameters,
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
