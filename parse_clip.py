import importlib.util
from pathlib import Path
from types import ModuleType
from typing import List, Optional

import common.io.apollo  # noqa: F401
import common.io.pose  # noqa: F401
from common.io.config import ParseConfig
from common.io.registry import get_parser, get_pose_provider


def _default_clip_id(record_path: str) -> str:
    path = Path(record_path)
    vehicle = path.parent.name
    stem, suffix = path.name.split(".record.")
    return f"{vehicle}_{stem}_{suffix}"


def _usage() -> str:
    return "Usage: python parse_clip.py <config.py>"


def _load_config_module(config_path: Path) -> ModuleType:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("parse_clip_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config module from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config(config_path_str: str) -> ParseConfig:
    config_path = Path(config_path_str).resolve()
    module = _load_config_module(config_path)
    if not hasattr(module, "config"):
        raise AttributeError(f"Config module must define `config`: {config_path}")
    config = module.config
    if not isinstance(config, ParseConfig):
        raise TypeError(
            f"`config` must be an instance of ParseConfig, got {type(config).__name__}: {config_path}"
        )
    return config


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or []
    if not argv:
        raise SystemExit(_usage())
    if len(argv) != 1:
        raise SystemExit(f"{_usage()}\nReceived unexpected arguments: {' '.join(argv)}")

    try:
        config = _load_config(argv[0])
    except (AttributeError, FileNotFoundError, ImportError, TypeError) as exc:
        raise SystemExit(f"Config load error: {exc}")
    clip_id = config.clip_id or _default_clip_id(config.record)
    out_dir = str(Path(config.out_root) / clip_id / "parsed")
    parser_cls = get_parser(config.source)
    pose_cls = get_pose_provider(config.pose)
    manifest = parser_cls().parse(config.record, out_dir, clip_id, pose_cls(), config)
    print(f"clip_id={manifest['clip_id']}")
    print(f"out_dir={out_dir}")
    print(f"n_samples={manifest['n_samples']}")
    for table_name, count in sorted(manifest["tables"].items()):
        print(f"{table_name}={count}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
