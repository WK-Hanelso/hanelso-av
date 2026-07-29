"""planning inference driver (C-SWM-025): root config 하나로 조립·실행.

    python planning/run_inference.py configs/e100bt25.py [--device cuda]

root config의 modules.planning 이름이 planning/configs/<이름>.py로 해석되고,
policy/dataloader는 registry 문자열로 조립된다.  모델 아티팩트는 bundle
(native config + checkpoint 쌍)에서 경로 참조로 읽는다.
출력: work/<clip_id>/inference/{outputs.npz, infer_report.txt, infer_bev.png}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_config, resolve_repo_path
from planning.interface import (
    get_dataloader,
    get_policy,
    get_postprocessor,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="ROOT config path (configs/*.py)")
    parser.add_argument(
        "--device",
        default=None,
        help="override planning.device (cpu|cuda). default: planning module config",
    )
    parser.add_argument("--t0-time", type=float, default=None)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="override output dir (default: work/<clip_id>/inference)",
    )
    return parser.parse_args()


def measure_forward_latency(policy, feature, repeats: int = 3) -> Dict[str, Any]:
    """Mean wall latency of policy.infer after one warmup (cuda-synchronized)."""
    is_cuda = getattr(policy, "device", torch.device("cpu")).type == "cuda"
    with torch.inference_mode():
        policy.infer(feature)  # warmup (cudnn autotune / lazy init)
        if is_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            begin = time.perf_counter()
            policy.infer(feature)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - begin)
    return {
        "repeats": repeats,
        "mean_sec": float(np.mean(times)),
        "min_sec": float(np.min(times)),
        "max_sec": float(np.max(times)),
    }


def build_render(
    normalized_data: Dict[str, Any],
    output_trajectory: np.ndarray,
    out_path: Path,
    post_best_trajectory: np.ndarray | None = None,
    post_emergency: bool = False,
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
        label="raw output_trajectory",
    )
    ax.scatter(output_trajectory[0, 0], output_trajectory[0, 1], s=20, color="#dc2626")

    if post_best_trajectory is not None:
        best_label = "post-processed best"
        if post_emergency:
            best_label += " (emergency brake)"
        ax.plot(
            post_best_trajectory[:, 0],
            post_best_trajectory[:, 1],
            color="#16a34a",
            linewidth=2.4,
            linestyle="--",
            alpha=0.95,
            label=best_label,
        )
        ax.scatter(
            post_best_trajectory[0, 0],
            post_best_trajectory[0, 1],
            s=20,
            color="#16a34a",
        )
    ax.legend(loc="upper right", fontsize=8)

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


def summarize_postprocess(post_result: Dict[str, Any]) -> Dict[str, Any]:
    def as_list(value: np.ndarray) -> list:
        return [round(float(v), 6) for v in np.asarray(value).reshape(-1)]

    ttc = post_result["time_to_at_fault_collision_s"]
    return {
        "num_candidates": post_result["num_candidates"],
        "best_candidate_idx": post_result["best_candidate_idx"],
        "emergency_brake": bool(post_result["emergency_brake"]),
        "time_to_at_fault_collision_s": ("inf" if np.isinf(ttc) else float(ttc)),
        "num_agents_in_world": post_result["num_agents_in_world"],
        "best_in_drivable_fraction": post_result["best_in_drivable_fraction"],
        "scores": {
            "rule_based": as_list(post_result["rule_based_scores"]),
            "learning_based": as_list(post_result["learning_based_scores"]),
            "final": as_list(post_result["final_scores"]),
        },
        "multi_metrics": {
            key: as_list(value) for key, value in post_result["multi_metrics"].items()
        },
        "weighted_metrics": {
            key: as_list(value)
            for key, value in post_result["weighted_metrics"].items()
        },
        "ego_progress_m": as_list(post_result["ego_progress_m"]),
    }


def run_postprocess_checks(post_result: Dict[str, Any]) -> Dict[str, Any]:
    best_local = post_result["best_trajectory_local"]
    step_disp = np.linalg.norm(np.diff(best_local[:, :2], axis=0), axis=1)
    start_norm = float(np.linalg.norm(best_local[0, :2]))
    rule_scores = np.asarray(post_result["rule_based_scores"], dtype=np.float64)
    learning_scores = np.asarray(post_result["learning_based_scores"], dtype=np.float64)
    return {
        "postprocess_best_finite": bool(np.isfinite(best_local).all()),
        "postprocess_scores_finite": bool(
            np.isfinite(rule_scores).all() and np.isfinite(learning_scores).all()
        ),
        "postprocess_best_start_norm_m": start_norm,
        "postprocess_best_start_near_origin_pass": start_norm < 5.0,
        "postprocess_best_max_step_m": float(step_disp.max()) if len(step_disp) else 0.0,
        "postprocess_best_step_smooth_pass": bool(
            len(step_disp) == 0 or step_disp.max() < 2.0
        ),
        "postprocess_best_in_drivable_fraction": post_result[
            "best_in_drivable_fraction"
        ],
        "postprocess_emergency_brake": bool(post_result["emergency_brake"]),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    plan_cfg = cfg.get("planning")
    if plan_cfg is None:
        raise SystemExit(
            f"root config {args.config} has no planning module "
            "(modules.planning is None)"
        )
    # 동적 로딩: modules.planning 이름 -> planning.models.<이름> import -> registry 등록.
    load_model((cfg.get("modules") or {}).get("planning"))

    clip_id = cfg["clip_id"]
    parsed_dir = resolve_repo_path(f"work/{clip_id}/parsed")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else resolve_repo_path(f"work/{clip_id}/inference")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(resolve_repo_path(cfg["map_path"])) as f:
        map_graph = json.load(f)

    config = {
        "clip_id": clip_id,
        "map_name": cfg["map_name"],
        "feature_vehicle": plan_cfg.get("feature_vehicle", "pacifica"),
        "calibration": cfg["calibration"],
        "data": cfg.get("data"),
    }

    adapter_cls = get_dataloader(plan_cfg["dataloader"])
    adapter = adapter_cls()
    build_result = adapter.build(
        parsed_dir=str(parsed_dir),
        map_graph=map_graph,
        config=config,
        t0_time=args.t0_time,
    )

    bundle = resolve_repo_path(plan_cfg["bundle"])
    device = args.device or plan_cfg.get("device", "cpu")
    policy_kwargs = {
        "config_path": str(bundle / plan_cfg["model_config"]),
        "checkpoint_path": str(bundle / plan_cfg["checkpoint"]),
        "device": device,
    }
    policy_cls = get_policy(plan_cfg["policy"])
    policy = policy_cls(**policy_kwargs)
    with torch.inference_mode():
        output = policy.infer(build_result.feature)
    forward_latency = measure_forward_latency(policy, build_result.feature)

    post_result = None
    post_cfg = plan_cfg.get("postprocess", {})
    if post_cfg.get("enabled", True):
        postprocessor = get_postprocessor(post_cfg["name"])(
            vehicle_parameters=build_result.scene_context["ego_state"].car_footprint.vehicle_parameters
        )
        post_result = postprocessor.run(
            model_output=output["raw_output"],
            normalized_data=build_result.normalized_numpy_data,
            scene_context=build_result.scene_context,
        )

    output_np = {
        key: value.detach().cpu().numpy()
        for key, value in output.items()
        if isinstance(value, torch.Tensor)
    }
    if post_result is not None:
        output_np["post_best_trajectory_local"] = post_result["best_trajectory_local"]
        output_np["post_best_trajectory_global"] = post_result["best_trajectory_global"]
        output_np["post_rule_based_scores"] = post_result["rule_based_scores"]
        output_np["post_learning_based_scores"] = post_result["learning_based_scores"]
        output_np["post_final_scores"] = post_result["final_scores"]
    np.savez(out_dir / "outputs.npz", **output_np)

    render_path = out_dir / "infer_bev.png"
    build_render(
        build_result.normalized_numpy_data,
        output_np["output_trajectory"][0],
        render_path,
        post_best_trajectory=(
            post_result["best_trajectory_local"] if post_result is not None else None
        ),
        post_emergency=(
            bool(post_result["emergency_brake"]) if post_result is not None else False
        ),
    )

    output_summary = summarize_output(output["raw_output"])
    checks = run_checks(build_result.normalized_numpy_data, output["raw_output"])
    if post_result is not None:
        checks.update(run_postprocess_checks(post_result))
    report = {
        "clip_id": config["clip_id"],
        "root_config": str(Path(args.config).resolve()),
        "parsed_dir": str(parsed_dir),
        "map_path": str(resolve_repo_path(cfg["map_path"])),
        "policy": plan_cfg["policy"],
        "bundle": str(bundle),
        "device": device,
        "model_param_device": policy.load_report["model_param_device"],
        "forward_latency": forward_latency,
        "decoder": policy.load_report["decoder_swap"]["decoder"],
        "model_kwargs": policy.model_kwargs,
        "model_load_report": policy.load_report,
        "adapter_context": build_result.context,
        "output_summary": output_summary,
        "checks": checks,
        "postprocess": (
            summarize_postprocess(post_result) if post_result is not None else None
        ),
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
