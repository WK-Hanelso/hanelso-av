from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.input.pluto_feature_adapter import ApolloPlutoFeatureAdapter
from common.policy import get_policy
import common.policy.pluto_torch  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-dir", required=True)
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--vehicle", default="pacifica")
    parser.add_argument("--t0-time", type=float, default=None)
    parser.add_argument("--policy", default="pluto_torch")
    parser.add_argument("--config-path", default="code/hydra/config.yaml")
    parser.add_argument("--checkpoint-path", default="data/model/v3_pluto.ckpt")
    parser.add_argument("--pluto-root", default="/home/hanelso/hanelso/pluto_onnx")
    parser.add_argument("--out-root", default="work/inference")
    parser.add_argument(
        "--use-v3-planning-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_render(
    normalized_data: Dict[str, Any],
    output_trajectory: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9), dpi=140)

    map_data = normalized_data["map"]
    valid_polygons = map_data["valid_mask"].any(-1) if "valid_mask" in map_data else np.ones(
        map_data["point_position"].shape[0], dtype=bool
    )
    for idx in np.flatnonzero(valid_polygons):
        center = map_data["point_position"][idx, 0]
        left = map_data["point_position"][idx, 1]
        right = map_data["point_position"][idx, 2]
        ax.plot(center[:, 0], center[:, 1], color="#d0d7de", linewidth=1.0, alpha=0.7)
        ax.plot(left[:, 0], left[:, 1], color="#e5e7eb", linewidth=0.8, alpha=0.5)
        ax.plot(right[:, 0], right[:, 1], color="#e5e7eb", linewidth=0.8, alpha=0.5)

    ref = normalized_data["reference_line"]
    ref_colors = ["#2563eb", "#0f766e", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]
    for ref_idx in range(ref["position"].shape[0]):
        if not ref["valid_mask"][ref_idx].any():
            continue
        ref_points = ref["position"][ref_idx][ref["valid_mask"][ref_idx]]
        ax.plot(
            ref_points[:, 0],
            ref_points[:, 1],
            color=ref_colors[ref_idx % len(ref_colors)],
            linewidth=2.0,
            alpha=0.9,
        )

    agents = normalized_data["agent"]
    hist = agents["position"][:, :21]
    valid = agents["valid_mask"][:, :21]
    for idx in range(hist.shape[0]):
        if not valid[idx].any():
            continue
        points = hist[idx][valid[idx]]
        color = "#111827" if idx == 0 else "#6b7280"
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.0, alpha=0.8)
        if len(points) > 0:
            ax.scatter(points[-1, 0], points[-1, 1], s=8, color=color, alpha=0.8)

    for idx in np.flatnonzero(normalized_data["static_objects"]["valid_mask"]):
        pos = normalized_data["static_objects"]["position"][idx]
        ax.scatter(pos[0], pos[1], s=18, marker="s", color="#f59e0b", alpha=0.9)

    ax.plot(
        output_trajectory[:, 0],
        output_trajectory[:, 1],
        color="#dc2626",
        linewidth=3.0,
        alpha=0.95,
    )
    ax.scatter(output_trajectory[0, 0], output_trajectory[0, 1], s=20, color="#dc2626")

    ego_width, ego_length = normalized_data["agent"]["shape"][0, 20]
    rect = plt.Rectangle(
        (-ego_length / 2.0, -ego_width / 2.0),
        ego_length,
        ego_width,
        fill=False,
        edgecolor="#111827",
        linewidth=2.0,
    )
    ax.add_patch(rect)

    ax.set_aspect("equal")
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.set_title("PLUTO Torch Inference BEV (ego frame)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def summarize_output(output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            summary[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "finite": bool(torch.isfinite(tensor).all().item()),
            }
    return summary


def run_checks(normalized_data: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    traj = output["output_trajectory"].detach().cpu().numpy()[0]
    finite = {
        key: bool(torch.isfinite(value).all().item())
        for key, value in output.items()
        if isinstance(value, torch.Tensor)
    }
    step_disp = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)
    start_norm = float(np.linalg.norm(traj[0, :2]))

    ref = normalized_data["reference_line"]
    all_ref_points = [
        ref["position"][idx][ref["valid_mask"][idx]]
        for idx in range(ref["position"].shape[0])
        if ref["valid_mask"][idx].any()
    ]
    if all_ref_points:
        min_ref_distance = min(
            float(np.linalg.norm(traj[:, None, :2] - ref_points[None, :, :2], axis=2).min(axis=1).mean())
            for ref_points in all_ref_points
        )
    else:
        min_ref_distance = float("inf")

    return {
        "all_outputs_finite": all(finite.values()),
        "per_output_finite": finite,
        "trajectory_start_norm_m": start_norm,
        "trajectory_start_near_origin_pass": start_norm < 5.0,
        "trajectory_max_step_m": float(step_disp.max()) if len(step_disp) else 0.0,
        "trajectory_step_smooth_pass": bool(len(step_disp) == 0 or step_disp.max() < 2.0),
        "mean_min_distance_to_reference_m": min_ref_distance,
        "trajectory_reference_comment": (
            "trajectory stays reasonably close to reference"
            if np.isfinite(min_ref_distance) and min_ref_distance < 8.0
            else "trajectory departs materially from reference or reference is unavailable"
        ),
        "reference_line_count": int(ref["position"].shape[0]),
    }


def compare_decoders(
    feature: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    policy_cls = get_policy(args.policy)
    original_policy = policy_cls(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        pluto_root=args.pluto_root,
        use_v3_planning_decoder=False,
    )
    v3_policy = policy_cls(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        pluto_root=args.pluto_root,
        use_v3_planning_decoder=True,
    )
    with torch.inference_mode():
        original_output = original_policy.infer(feature)["raw_output"]
        v3_output = v3_policy.infer(feature)["raw_output"]

    original_traj = original_output["output_trajectory"].detach().cpu().numpy()
    v3_traj = v3_output["output_trajectory"].detach().cpu().numpy()
    abs_diff = np.abs(original_traj - v3_traj)
    return {
        "original_decoder": original_policy.load_report["decoder_swap"]["decoder"],
        "v3_decoder": v3_policy.load_report["decoder_swap"]["decoder"],
        "outputs_differ": bool(not np.allclose(original_traj, v3_traj, atol=1e-6)),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
    }


def main() -> int:
    args = parse_args()
    parsed_dir = Path(args.parsed_dir)
    out_dir = Path(args.out_root) / (args.clip_id or parsed_dir.parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.map_path) as f:
        map_graph = json.load(f)

    config = {
        "clip_id": args.clip_id or parsed_dir.parent.name,
        "map_name": args.map_name,
        "vehicle": args.vehicle,
    }

    adapter = ApolloPlutoFeatureAdapter(args.pluto_root)
    build_result = adapter.build(
        parsed_dir=str(parsed_dir),
        map_graph=map_graph,
        config=config,
        t0_time=args.t0_time,
    )

    policy_cls = get_policy(args.policy)
    policy = policy_cls(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        pluto_root=args.pluto_root,
        use_v3_planning_decoder=args.use_v3_planning_decoder,
    )
    with torch.inference_mode():
        output = policy.infer(build_result.feature)

    output_np = {
        key: value.detach().cpu().numpy()
        for key, value in output.items()
        if isinstance(value, torch.Tensor)
    }
    np.savez(out_dir / "outputs.npz", **output_np)

    render_path = out_dir / "infer_bev.png"
    build_render(
        build_result.normalized_numpy_data,
        output_np["output_trajectory"][0],
        render_path,
    )

    output_summary = summarize_output(output["raw_output"])
    checks = run_checks(build_result.normalized_numpy_data, output["raw_output"])
    decoder_comparison = compare_decoders(build_result.feature, args)
    report = {
        "clip_id": config["clip_id"],
        "parsed_dir": str(parsed_dir),
        "map_path": str(Path(args.map_path).resolve()),
        "policy": args.policy,
        "decoder": policy.load_report["decoder_swap"]["decoder"],
        "model_kwargs": policy.model_kwargs,
        "model_load_report": policy.load_report,
        "adapter_context": build_result.context,
        "output_summary": output_summary,
        "checks": checks,
        "decoder_comparison": decoder_comparison,
        "artifacts": {
            "outputs_npz": str((out_dir / "outputs.npz").resolve()),
            "infer_report": str((out_dir / "infer_report.txt").resolve()),
            "infer_bev_png": str(render_path.resolve()),
        },
    }
    (out_dir / "infer_report.txt").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )

    print(json.dumps(report["checks"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
