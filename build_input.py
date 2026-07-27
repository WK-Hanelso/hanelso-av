import importlib.util
import json
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common.input  # noqa: F401
from common.input.config import BuildInputConfig
from common.input.base import get_input_builder


def _usage() -> str:
    return "Usage: python build_input.py <config.py>"


def _load_config_module(config_path: Path) -> ModuleType:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("build_input_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config module from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config(config_path_str: str) -> BuildInputConfig:
    config_path = Path(config_path_str).resolve()
    module = _load_config_module(config_path)
    if not hasattr(module, "config"):
        raise AttributeError(f"Config module must define `config`: {config_path}")
    config = module.config
    if not isinstance(config, BuildInputConfig):
        raise TypeError(
            f"`config` must be an instance of BuildInputConfig, got {type(config).__name__}: {config_path}"
        )
    return config


def _shape_str(array: np.ndarray) -> str:
    return "[" + ",".join(str(dim) for dim in array.shape) + "]"


def _inverse_transform(local_xy: np.ndarray, origin_xy: np.ndarray, angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return (local_xy @ rot.T) + origin_xy[None, :]


def _oriented_box(center_xy: np.ndarray, heading: float, width: float, length: float) -> np.ndarray:
    dx = length * 0.5
    dy = width * 0.5
    corners = np.array(
        [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]],
        dtype=np.float64,
    )
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    return (corners @ rot.T) + center_xy[None, :]


def _run_checks(feed: Dict[str, np.ndarray], context: Dict[str, object]) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []
    expected = {
        "agent_position": ((32, 21, 2), np.float32),
        "agent_heading": ((32, 21), np.float32),
        "agent_velocity": ((32, 21, 2), np.float32),
        "agent_shape": ((32, 2), np.float32),
        "agent_category": ((32,), np.int32),
        "agent_valid_mask": ((32, 21), np.bool_),
        "current_state": ((7,), np.float32),
        "static_position": ((32, 2), np.float32),
        "static_heading": ((32,), np.float32),
        "static_shape": ((32, 2), np.float32),
        "static_valid_mask": ((32,), np.bool_),
        "ref_position": ((120, 2), np.float32),
        "ref_vector": ((120, 2), np.float32),
        "ref_orientation": ((120,), np.float32),
        "ref_valid_mask": ((120,), np.bool_),
        "polygon_points": ((80, 3, 20, 2), np.float32),
        "map_valid_mask": ((80,), np.bool_),
        "on_route": ((80,), np.bool_),
        "point_position": ((32, 20, 2), np.float32),
        "point_vector": ((32, 20, 2), np.float32),
        "point_valid_mask": ((32, 20), np.bool_),
        "origin": ((2,), np.float64),
        "angle": ((1,), np.float64),
    }
    shape_errors: List[str] = []
    dtype_errors: List[str] = []
    finite_errors: List[str] = []
    for key, (shape, dtype) in expected.items():
        value = feed[key]
        if value.shape != shape:
            shape_errors.append(f"{key}: expected {shape}, got {value.shape}")
        if value.dtype != dtype:
            dtype_errors.append(f"{key}: expected {dtype}, got {value.dtype}")
        if value.dtype.kind in {"f", "i"} and not np.isfinite(value).all():
            finite_errors.append(key)
    checks.append(
        {
            "name": "shapes_dtypes_finite",
            "pass": not shape_errors and not dtype_errors and not finite_errors,
            "detail": {
                "shape_errors": shape_errors,
                "dtype_errors": dtype_errors,
                "non_finite": finite_errors,
            },
        }
    )

    ego_pos = feed["agent_position"][0, -1]
    ego_heading = feed["agent_heading"][0, -1]
    checks.append(
        {
            "name": "index0_ego_origin",
            "pass": bool(np.linalg.norm(ego_pos) <= 1e-4 and abs(float(ego_heading)) <= 1e-4),
            "detail": {
                "ego_position_t0": ego_pos.tolist(),
                "ego_heading_t0": float(ego_heading),
            },
        }
    )

    origin_xy = feed["origin"].astype(np.float64)
    angle = float(feed["angle"][0])
    ego_global_roundtrip = _inverse_transform(feed["agent_position"][0].astype(np.float64), origin_xy, angle)
    expected_global = np.array(context["ego_history_global_xy"], dtype=np.float64)
    roundtrip_err = np.linalg.norm(ego_global_roundtrip - expected_global, axis=1)
    checks.append(
        {
            "name": "ego_roundtrip",
            "pass": bool(roundtrip_err.max(initial=0.0) <= 1e-3),
            "detail": {
                "max_error_m": float(roundtrip_err.max(initial=0.0)),
                "mean_error_m": float(roundtrip_err.mean()) if len(roundtrip_err) else 0.0,
            },
        }
    )

    radius_margin = 5.0
    heading_ok = (
        np.abs(feed["agent_heading"][feed["agent_valid_mask"]]).max(initial=0.0)
        <= math.pi + 1e-5
    )
    pos_ok = (
        np.abs(feed["agent_position"][feed["agent_valid_mask"]]).max(initial=0.0)
        <= 120.0 + radius_margin
    )
    checks.append(
        {
            "name": "range_checks",
            "pass": bool(heading_ok and pos_ok),
            "detail": {
                "max_abs_agent_xy": float(np.abs(feed["agent_position"][feed["agent_valid_mask"]]).max(initial=0.0)),
                "max_abs_heading": float(np.abs(feed["agent_heading"][feed["agent_valid_mask"]]).max(initial=0.0)),
            },
        }
    )

    valid_ref = feed["ref_position"][feed["ref_valid_mask"]]
    if len(valid_ref) >= 2:
        ref_spacing = np.linalg.norm(np.diff(valid_ref, axis=0), axis=1)
        ref_ok = bool(((ref_spacing >= 0.9) & (ref_spacing <= 1.1)).all() and len(valid_ref) <= 120)
        ref_detail = {
            "valid_count": int(len(valid_ref)),
            "spacing_min_m": float(ref_spacing.min()),
            "spacing_max_m": float(ref_spacing.max()),
        }
    else:
        ref_ok = bool(len(valid_ref) <= 120)
        ref_detail = {"valid_count": int(len(valid_ref)), "spacing_min_m": None, "spacing_max_m": None}
    checks.append({"name": "reference_spacing", "pass": ref_ok, "detail": ref_detail})

    invalid_pos_zero = np.allclose(
        feed["agent_position"][~feed["agent_valid_mask"]],
        0.0,
    )
    invalid_vel_zero = np.allclose(
        feed["agent_velocity"][~feed["agent_valid_mask"]],
        0.0,
    )
    checks.append(
        {
            "name": "valid_mask_zero_padding",
            "pass": bool(invalid_pos_zero and invalid_vel_zero),
            "detail": {
                "invalid_position_zero": bool(invalid_pos_zero),
                "invalid_velocity_zero": bool(invalid_vel_zero),
            },
        }
    )
    return checks


def _render_bev(feed: Dict[str, np.ndarray], context: Dict[str, object], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    for idx, valid in enumerate(feed["map_valid_mask"]):
        if not valid:
            continue
        center, left, right = feed["polygon_points"][idx]
        ax.plot(center[:, 0], center[:, 1], color="#9aa0a6", linewidth=1.2, alpha=0.9)
        ax.plot(left[:, 0], left[:, 1], color="#c0c4c8", linewidth=0.7, alpha=0.8)
        ax.plot(right[:, 0], right[:, 1], color="#c0c4c8", linewidth=0.7, alpha=0.8)
    ref = feed["ref_position"][feed["ref_valid_mask"]]
    if len(ref) > 0:
        ax.plot(ref[:, 0], ref[:, 1], color="#ff7f11", linewidth=3.0)
        stride = max(1, len(ref) // 12)
        for idx in range(0, len(ref), stride):
            if idx >= len(ref) - 1:
                break
            delta = ref[min(idx + 1, len(ref) - 1)] - ref[idx]
            ax.arrow(
                ref[idx, 0],
                ref[idx, 1],
                delta[0],
                delta[1],
                color="#ff7f11",
                width=0.08,
                head_width=1.2,
                length_includes_head=True,
                alpha=0.85,
            )

    ego_box = _oriented_box(np.array([0.0, 0.0]), 0.0, float(feed["agent_shape"][0, 0]), float(feed["agent_shape"][0, 1]))
    ax.plot(ego_box[:, 0], ego_box[:, 1], color="#0b4f6c", linewidth=2.5)
    ax.arrow(0.0, 0.0, 3.0, 0.0, color="#0b4f6c", width=0.08, head_width=0.9, length_includes_head=True)

    agent_colors = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c", 4: "#9467bd"}
    valid_agents = 0
    for idx in range(1, feed["agent_position"].shape[0]):
        valid_steps = np.flatnonzero(feed["agent_valid_mask"][idx])
        if len(valid_steps) == 0:
            continue
        valid_agents += 1
        last = valid_steps[-1]
        center = feed["agent_position"][idx, last].astype(np.float64)
        heading = float(feed["agent_heading"][idx, last])
        width = float(feed["agent_shape"][idx, 0])
        length = float(feed["agent_shape"][idx, 1])
        box = _oriented_box(center, heading, width, length)
        color = agent_colors.get(int(feed["agent_category"][idx]), "#7f7f7f")
        ax.plot(box[:, 0], box[:, 1], color=color, linewidth=1.6)
        nose = center + np.array([math.cos(heading), math.sin(heading)]) * max(0.8, length * 0.35)
        ax.plot([center[0], nose[0]], [center[1], nose[1]], color=color, linewidth=1.4)

    ax.set_xlim(-125.0, 125.0)
    ax.set_ylim(-125.0, 125.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    map_name = context.get("map_name") or "unknown"
    title = (
        f"{context['clip_id']} | t0={context['t0_time_sec']:.3f}s | map={map_name} | "
        f"MA=32/{valid_agents + 1}"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _render_report(feed: Dict[str, np.ndarray], context: Dict[str, object], checks: List[Dict[str, object]]) -> str:
    overall_pass = all(check["pass"] for check in checks)
    lines: List[str] = []
    lines.append(f"Input Build: {'PASS' if overall_pass else 'FAIL'}")
    lines.append(f"Clip: {context['clip_id']}")
    lines.append(f"Parsed: {context['parsed_dir']}")
    lines.append(f"Map: {context.get('map_name') or 'unknown'} ({context['map_path']})")
    lines.append(f"Vehicle: {context['vehicle_key']}")
    lines.append(f"t0_index={context['t0_index']}, t0_time_sec={context['t0_time_sec']:.6f}, reason={context['t0_reason']}")
    lines.append("")
    lines.append("== Feed Summary ==")
    for key in sorted(feed):
        value = feed[key]
        lines.append(f"{key}: shape={_shape_str(value)}, dtype={value.dtype}")
    lines.append("")
    lines.append("== Input Checks ==")
    for check in checks:
        lines.append(f"[{'PASS' if check['pass'] else 'FAIL'}] {check['name']}")
        detail = json.dumps(check["detail"], ensure_ascii=True, sort_keys=True)
        lines.append(f"  {detail}")
    return "\n".join(lines) + "\n"


def _write_meta(path: Path, context: Dict[str, object], checks: List[Dict[str, object]], feed: Dict[str, np.ndarray]) -> None:
    payload = dict(context)
    payload["checks"] = checks
    payload["feed_shapes"] = {key: list(value.shape) for key, value in feed.items()}
    payload["feed_dtypes"] = {key: str(value.dtype) for key, value in feed.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or []
    if len(argv) != 1:
        raise SystemExit(_usage())
    try:
        config = _load_config(argv[0])
    except (AttributeError, FileNotFoundError, ImportError, TypeError) as exc:
        raise SystemExit(f"Config load error: {exc}")

    clip_id = config.clip_id or Path(config.parsed_dir).resolve().parent.name
    out_dir = Path(config.out_root) / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    map_graph = json.loads(Path(config.map_path).read_text())
    builder_cls = get_input_builder(config.builder)
    builder = builder_cls()
    build_dict = {
        "clip_id": clip_id,
        "map_name": config.map_name,
        "vehicle": config.vehicle,
    }
    feed = builder.build(config.parsed_dir, map_graph, config.t0_time, build_dict)
    context = dict(builder.last_context)
    context["clip_id"] = clip_id
    context["parsed_dir"] = config.parsed_dir
    context["map_path"] = config.map_path
    context["map_name"] = config.map_name or context.get("map_name") or ""

    checks = _run_checks(feed, context)

    np.savez(out_dir / "feed.npz", **feed)
    _write_meta(out_dir / "meta.json", context, checks, feed)
    _render_bev(feed, context, out_dir / "input_bev.png")
    report_text = _render_report(feed, context, checks)
    (out_dir / "input_report.txt").write_text(report_text)

    overall_pass = all(check["pass"] for check in checks)
    print(f"clip_id={clip_id}")
    print(f"out_dir={out_dir}")
    print(f"t0_time_sec={context['t0_time_sec']:.6f}")
    print(f"checks={'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
