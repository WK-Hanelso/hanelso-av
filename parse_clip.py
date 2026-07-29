"""parse driver: 원본 record -> work/<clip_id>/parsed (공용 파이프라인, .venv-apollo).

    python parse_clip.py configs/e100bt25.py

root config의 record/source/pose(+선택 keyframe·keyframe_hz)를 읽어 파싱한다.
"""

from pathlib import Path
from typing import List, Optional

import data_devkit.parsers.apollo  # noqa: F401
import data_devkit.parsers.pose  # noqa: F401
from common.config import load_config_file, resolve_repo_path
from data_devkit.parsers.config import ParseConfig
from data_devkit.parsers.registry import get_parser, get_pose_provider


def _default_clip_id(record_path: str) -> str:
    path = Path(record_path)
    vehicle = path.parent.name
    stem, suffix = path.name.split(".record.")
    return f"{vehicle}_{stem}_{suffix}"


def _usage() -> str:
    return "Usage: python parse_clip.py <root_config.py>"


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or []
    if len(argv) != 1:
        raise SystemExit(_usage())

    # 파싱은 root 필드만 소비 — 모듈 도메인 해석이 필요 없어 load_config_file 사용
    # (.venv-apollo에서도 stdlib만으로 동작).
    cfg = load_config_file(resolve_repo_path(argv[0]))
    parse_kwargs = {
        "record": cfg["record"],
        "source": cfg.get("source", "apollo_record"),
        "pose": cfg.get("pose", "apollo_record"),
        "clip_id": cfg.get("clip_id"),
    }
    for optional_key in ("out_root", "keyframe", "keyframe_hz"):
        if optional_key in cfg:
            parse_kwargs[optional_key] = cfg[optional_key]
    config = ParseConfig(**parse_kwargs)

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
