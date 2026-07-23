import argparse
from pathlib import Path

import common.io.apollo  # noqa: F401
import common.io.pose  # noqa: F401
from common.io.registry import get_parser, get_pose_provider


def _default_clip_id(record_path: str) -> str:
    path = Path(record_path)
    vehicle = path.parent.name
    stem, suffix = path.name.split(".record.")
    return f"{vehicle}_{stem}_{suffix}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse one clip into unified JSON tables.")
    parser.add_argument("--record", required=True, help="Path to the source record file.")
    parser.add_argument("--source", required=True, help="Registered source parser name.")
    parser.add_argument("--pose", required=True, help="Registered pose provider name.")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to work/<clip_id>/parsed.")
    parser.add_argument("--clip-id", default=None, help="Override clip id.")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    clip_id = args.clip_id or _default_clip_id(args.record)
    out_dir = args.out or str(Path("work") / clip_id / "parsed")
    parser_cls = get_parser(args.source)
    pose_cls = get_pose_provider(args.pose)
    manifest = parser_cls().parse(args.record, out_dir, clip_id, pose_cls())
    print(f"clip_id={manifest['clip_id']}")
    print(f"out_dir={out_dir}")
    print(f"n_samples={manifest['n_samples']}")
    for table_name, count in sorted(manifest["tables"].items()):
        print(f"{table_name}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

