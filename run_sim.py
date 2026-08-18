"""One-shot bag -> simulation mp4 orchestrator.

Usage:
    python3 run_sim.py <record 파일 경로> [--mode closed_loop|open_loop]
        [--renderer nuplan|matplotlib] [--steps N] [--device cpu|cuda] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent
WORK_ROOT = REPO_ROOT / "work"
MAPS_ROOT = WORK_ROOT / "maps"

# 기본: run_sim.py 를 실행한 인터프리터 하나로 전 스테이지를 돈다.
# docker av-base 처럼 파싱·추론·렌더 의존성이 한 환경에 모여 있는 경우가 기본 경로 —
# host 에 별도 파이썬 env 가 없어도 컨테이너 단일 환경에서 그대로 동작한다.
# 스테이지별로 다른 인터프리터가 필요하면 env 로 오버라이드:
#   RUN_SIM_PY_INSPECT_RECORD / RUN_SIM_PY_PARSE_CLIP / RUN_SIM_PY_RENDER_SIM
INTERPRETERS = {
    "inspect_record": os.environ.get("RUN_SIM_PY_INSPECT_RECORD", sys.executable),
    "parse_clip": os.environ.get("RUN_SIM_PY_PARSE_CLIP", sys.executable),
    "render_sim": os.environ.get("RUN_SIM_PY_RENDER_SIM", sys.executable),
}

DEFAULT_SIMULATION_MODULE = "closed_loop_nuplan"
DEFAULT_DATA_CFG = {"agents": "apollo_gt", "prediction": None}
REQUIRED_PARSED_ARTIFACTS = [
    "sample",
    "ego_pose",
    "ego_dynamics",
    "agent_tracks",
    "scene_log",
    "route",
    "map_graph",
]
CALIBRATION_PREFIX_RULES = (
    ("E100BT-", "e100"),
    ("U100", "u100"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", help="Apollo .record path")
    parser.add_argument("--mode", choices=["closed_loop", "open_loop"], default=None)
    parser.add_argument("--renderer", choices=["nuplan", "matplotlib"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--force", action="store_true", help="force re-parse even if parsed contract passes")
    return parser.parse_args()


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def relpath_str(path: Path) -> str:
    path = path if path.is_absolute() else (REPO_ROOT / path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def clip_id_from_record(record_path: Path) -> str:
    stem, suffix = record_path.name.split(".record.")
    return f"{record_path.parent.name}_{stem}_{suffix}"


def ensure_interpreter(name: str) -> str:
    value = INTERPRETERS[name]
    candidate = Path(value)
    if candidate.is_absolute():
        if not candidate.exists():
            raise RuntimeError(
                f"Required interpreter for stage '{name}' is missing: {candidate}\n"
                f"Action: create the environment first."
            )
        return str(candidate)
    return str(value)


def run_command(cmd: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def inspect_record(record_path: Path) -> Dict[str, object]:
    inspect_code = """
import json
from pathlib import Path
from tools.parser_validation.core import inspect_record_for_run_sim
result = inspect_record_for_run_sim(RECORD_PATH, Path(MAPS_ROOT))
print(json.dumps(result, ensure_ascii=False))
"""
    completed = run_command(
        [
            ensure_interpreter("inspect_record"),
            "-c",
            f"RECORD_PATH={str(record_path)!r}; MAPS_ROOT={str(MAPS_ROOT)!r}; {inspect_code}",
        ],
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Record inspection failed.\n"
            f"record: {record_path}\n"
            f"stderr:\n{completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Record inspection returned non-JSON output.\n"
            f"stdout:\n{completed.stdout}\nerror: {error}"
        ) from error


def detect_calibration(record_path: Path) -> str | None:
    tokens: List[str] = []
    for part in record_path.parts:
        tokens.append(part)
    tokens.append(record_path.stem)
    for token in tokens:
        for prefix, calibration_name in CALIBRATION_PREFIX_RULES:
            if token.startswith(prefix):
                return calibration_name
    return None


def detect_planning_module() -> str:
    models_dir = REPO_ROOT / "planning" / "models"
    candidates = sorted(
        path.name
        for path in models_dir.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    )
    if not candidates:
        raise RuntimeError(
            f"No planning models found under {models_dir}.\n"
            "Action: add planning/models/<name>/ and planning/configs/<name>.py first."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple planning models found; default cannot be inferred uniquely.\n"
            f"Candidates: {candidates}\n"
            "Action: keep exactly one model under planning/models/ or extend run_sim.py selection rules."
        )
    return candidates[0]


def write_resolved_config(
    *,
    clip_id: str,
    record_path: Path,
    map_name: str,
    map_path: Path,
    planning_module: str,
    calibration_name: str,
) -> Path:
    clip_root = WORK_ROOT / clip_id
    clip_root.mkdir(parents=True, exist_ok=True)
    resolved_config_path = clip_root / "resolved_config.py"
    config_payload = {
        "clip_id": clip_id,
        "record": relpath_str(record_path),
        "source": "apollo_record",
        "pose": "apollo_record",
        "map_name": map_name,
        "map_path": relpath_str(map_path),
        "data": dict(DEFAULT_DATA_CFG),
        "modules": {
            "planning": planning_module,
            "perception": None,
            "localization": None,
        },
        "calibration": calibration_name,
        "simulation": DEFAULT_SIMULATION_MODULE,
    }
    resolved_config_path.write_text(
        "# Auto-generated by run_sim.py. Reusable as a root config.\n"
        f"config = {pformat(config_payload, sort_dicts=False)}\n"
    )
    return resolved_config_path


def parsed_contract_ok(clip_id: str, map_name: str) -> bool:
    check_code = """
from data_devkit.contract import check, ContractError
try:
    check(
        clip_id=CLIP_ID,
        requires=REQUIRES,
        data_cfg=DATA_CFG,
        map_name=MAP_NAME,
    )
except ContractError as error:
    print(error)
    raise SystemExit(3)
"""
    completed = run_command(
        [
            ensure_interpreter("inspect_record"),
            "-c",
            (
                f"CLIP_ID={clip_id!r}; REQUIRES={REQUIRED_PARSED_ARTIFACTS!r}; "
                f"DATA_CFG={DEFAULT_DATA_CFG!r}; MAP_NAME={map_name!r}; {check_code}"
            ),
        ],
        cwd=REPO_ROOT,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 3:
        return False
    raise RuntimeError(
        "Parsed contract check failed unexpectedly.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def run_parse_clip(config_path: Path) -> None:
    completed = run_command(
        [ensure_interpreter("parse_clip"), "parse_clip.py", relpath_str(config_path)],
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "parse_clip.py failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())


def run_render_sim(
    config_path: Path,
    *,
    mode: str | None,
    renderer: str | None,
    steps: int | None,
    device: str | None,
) -> Dict[str, object]:
    cmd = [ensure_interpreter("render_sim"), "simulation/render_sim.py", relpath_str(config_path)]
    if mode:
        cmd.extend(["--mode", mode])
    if renderer:
        cmd.extend(["--renderer", renderer])
    if steps is not None:
        cmd.extend(["--steps", str(steps)])
    if device:
        cmd.extend(["--device", device])
    completed = run_command(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(
            "simulation/render_sim.py failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    stdout = completed.stdout.strip()
    if stdout:
        print(stdout)
    metrics_lines = [line for line in stdout.splitlines() if line.startswith("metrics: ")]
    if not metrics_lines:
        raise RuntimeError("render_sim.py succeeded but did not print metrics path.")
    metrics_path = Path(metrics_lines[-1].split("metrics: ", 1)[1].strip())
    if not metrics_path.exists():
        raise RuntimeError(f"render_sim.py reported missing metrics path: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    return {"metrics_path": metrics_path, "metrics": metrics}


def main() -> int:
    args = parse_args()
    record_arg_path = Path(args.record)
    if record_arg_path.is_absolute():
        record_path = record_arg_path.absolute()
    else:
        record_path = (REPO_ROOT / record_arg_path).absolute()
    if not record_path.exists():
        return fail(
            f"Record file not found: {record_path}\n"
            "Action: pass an existing Apollo .record file path."
        )

    try:
        inspection = inspect_record(record_path)
    except RuntimeError as error:
        return fail(str(error))

    missing_required_topics = inspection["missing_required_topics"]
    if missing_required_topics:
        return fail(
            "Record is missing required topic(s) for run_sim.py.\n"
            f"record: {record_path}\n"
            f"missing: {missing_required_topics}\n"
            "Action: re-record or export a bag that includes "
            "/apollo/localization/pose, /apollo/canbus/chassis, "
            "/apollo/perception/obstacles, and either "
            "/apollo/routing_response or /apollo/routing_response_history."
        )

    calibration_name = detect_calibration(record_path)
    if calibration_name is None:
        return fail(
            "Vehicle prefix could not be matched to a calibration config.\n"
            f"record: {record_path}\n"
            f"known prefix rules: {list(CALIBRATION_PREFIX_RULES)}\n"
            "Action: add calibration/configs/<차량>.py (calibration/configs/e100.py 형식) "
            "and extend the prefix rule table in run_sim.py."
        )
    calibration_config_path = REPO_ROOT / "calibration" / "configs" / f"{calibration_name}.py"
    if not calibration_config_path.exists():
        return fail(
            "Vehicle prefix matched, but calibration config is missing.\n"
            f"expected: {calibration_config_path}\n"
            "Action: add calibration/configs/<차량>.py (calibration/configs/e100.py 형식)."
        )

    map_match = inspection["map_match"]
    map_name = map_match["matched_map"]
    if map_name is None:
        known_maps = sorted(path.parent.name for path in MAPS_ROOT.glob("*/map_graph.json"))
        return fail(
            "No available map covers this record trajectory.\n"
            f"record: {record_path}\n"
            f"available maps: {known_maps}\n"
            "Action: obtain the source HD map for this area and run "
            "`docker run --rm -v \"$PWD\":/workspace -w /workspace av-base:latest "
            "python parse_map.py --map <base_map.bin> --name <map_name>`."
        )
    map_path = Path(map_match["matched_map_path"]).resolve()
    if not map_path.exists():
        return fail(
            "Map was detected but map_graph.json is missing.\n"
            f"expected: {map_path}\n"
            "Action: regenerate the map with parse_map.py."
        )

    try:
        planning_module = detect_planning_module()
    except RuntimeError as error:
        return fail(str(error))
    planning_config_path = REPO_ROOT / "planning" / "configs" / f"{planning_module}.py"
    if not planning_config_path.exists():
        return fail(
            "Planning model directory exists but planning config is missing.\n"
            f"expected: {planning_config_path}\n"
            "Action: add planning/configs/<model>.py."
        )

    clip_id = clip_id_from_record(record_path)
    resolved_config_path = write_resolved_config(
        clip_id=clip_id,
        record_path=record_path,
        map_name=str(map_name),
        map_path=map_path,
        planning_module=planning_module,
        calibration_name=calibration_name,
    )

    should_parse = args.force or not parsed_contract_ok(clip_id, str(map_name))
    if should_parse:
        try:
            run_parse_clip(resolved_config_path)
        except RuntimeError as error:
            return fail(str(error))
    else:
        print(f"parse: skip (existing parsed contract satisfied for clip_id={clip_id})")

    try:
        render_result = run_render_sim(
            resolved_config_path,
            mode=args.mode,
            renderer=args.renderer,
            steps=args.steps,
            device=args.device,
        )
    except RuntimeError as error:
        return fail(str(error))

    metrics_path = Path(render_result["metrics_path"]).resolve()
    metrics = render_result["metrics"]
    mp4_info = metrics["mp4"]
    mp4_value = mp4_info["path"] if isinstance(mp4_info, dict) else mp4_info
    mp4_path = Path(mp4_value).resolve()
    print(f"mp4: {mp4_path}")
    print(f"metrics: {metrics_path}")
    print(
        f"resolved: map={map_name} calibration={calibration_name} model={planning_module} "
        f"config={resolved_config_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
