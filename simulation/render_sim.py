"""simulation driver (조립 전용, C-SWM-022): root config 하나로 실행.

    python simulation/render_sim.py configs/e100bt25.py \
        [--mode closed_loop] [--renderer nuplan] [--steps 3] ...

root config의 simulation 이름이 simulation/configs/<이름>.py로 해석되고, CLI
인자는 그 위에 덮어쓴다.  조립: feature adapter(planning/input registry) +
policy(planning/policy registry) + postprocessor + renderer(simulation/renderers
registry) + ego-driver(simulation/drivers registry; open_loop=log_replay,
closed_loop=model_driven).

sim은 오프라인 검증 경로라 CPU(swm-base)로 돈다 — GPU 실추론 경로는
planning/run_inference.py --device cuda (모델 소유 env: pluto-inf).

Outputs: work/<clip>/sim/<mode>[_nuplan]/frame_%05d.png
         + work/<clip>/sim/<mode>[_nuplan].mp4
         + <mode>[_nuplan]_metrics.json (per-step metrics, divergence, collisions).

Run with system python3 (torch 1.12 + natten + nuplan + shapely).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import load_config, resolve_repo_path
import planning.input  # noqa: F401  (registers input builders / feature adapters)
from planning.input.base import get_feature_adapter
from planning.policy import get_policy
import planning.policy.pluto_torch  # noqa: F401  (registers pluto_torch)
from planning.policy.pluto_postprocess import PlutoPostProcessor
from simulation.drivers import MODE_TO_DRIVER, get_ego_driver
from simulation.renderers import get_renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="ROOT config path (configs/*.py)")
    parser.add_argument("--mode", default=None, choices=["open_loop", "closed_loop"], help="override simulation.mode")
    parser.add_argument("--renderer", default=None, choices=["matplotlib", "nuplan"], help="override simulation.renderer")
    parser.add_argument("--steps", type=int, default=None, help="override simulation.steps")
    parser.add_argument("--stride", type=int, default=None, help="override simulation.stride")
    parser.add_argument("--fps", type=int, default=None, help="override simulation.fps")
    parser.add_argument("--start-index", type=int, default=None, help="override simulation.start_index")
    parser.add_argument("--view-radius", type=float, default=None, help="override simulation.view_radius")
    parser.add_argument("--device", default=None, help="override planning.device (sim 기본은 cpu)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    plan_cfg = cfg.get("planning")
    sim_cfg = dict(cfg.get("simulation") or {})
    if plan_cfg is None:
        raise SystemExit(f"root config {args.config} has no planning module")
    if not sim_cfg:
        raise SystemExit(f"root config {args.config} has no simulation config")
    for key in ("mode", "renderer", "steps", "stride", "fps", "start_index", "view_radius"):
        cli_value = getattr(args, key)
        if cli_value is not None:
            sim_cfg[key] = cli_value

    clip_id = cfg["clip_id"]
    parsed_dir = resolve_repo_path(f"work/{clip_id}/parsed")
    out_dir = resolve_repo_path(f"work/{clip_id}/sim")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(resolve_repo_path(cfg["map_path"])) as f:
        map_graph = json.load(f)

    config = {
        "clip_id": clip_id,
        "map_name": cfg["map_name"],
        "vehicle": cfg.get("vehicle", "pacifica"),
    }
    adapter = get_feature_adapter(plan_cfg["input_builder"])()
    clip = adapter.prepare_clip(str(parsed_dir), map_graph, config)

    bundle = resolve_repo_path(plan_cfg["bundle"])
    policy = get_policy(plan_cfg["policy"])(
        config_path=str(bundle / plan_cfg["model_config"]),
        checkpoint_path=str(bundle / plan_cfg["checkpoint"]),
        use_v3_planning_decoder=plan_cfg.get("use_v3_planning_decoder", True),
        device=args.device or plan_cfg.get("device", "cpu"),
    )
    postprocessor = PlutoPostProcessor()

    mode = sim_cfg["mode"]
    renderer_cls = get_renderer(sim_cfg["renderer"])
    renderer = renderer_cls(clip=clip, map_graph=map_graph, sim_cfg=sim_cfg, mode=mode)

    driver_cls = get_ego_driver(MODE_TO_DRIVER[mode])
    driver = driver_cls(
        adapter=adapter,
        policy=policy,
        postprocessor=postprocessor,
        clip=clip,
        sim_cfg=sim_cfg,
    )

    mode_name = f"{mode}{renderer.suffix}"
    frames_dir = out_dir / mode_name
    result = driver.run(renderer, frames_dir, mode_name, out_dir)

    result["clip_id"] = clip_id
    result["root_config"] = str(Path(args.config).resolve())
    result["sim_config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in sim_cfg.items()}
    metrics_path = out_dir / f"{mode_name}_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "steps")}, indent=2, ensure_ascii=False))
    print(f"metrics: {metrics_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
