import argparse
from pathlib import Path
from typing import Optional, Sequence

import data_devkit.parsers.apollo  # noqa: F401
from data_devkit.parsers.registry import get_map_parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse one source HDMap into work/maps/<name>/map_graph.json."
    )
    parser.add_argument("--map", required=True, dest="map_path", help="Input base_map.bin path.")
    parser.add_argument("--name", required=True, dest="map_name", help="Map name.")
    parser.add_argument(
        "--source",
        default="apollo",
        help="Registered map parser key. Default: apollo.",
    )
    parser.add_argument(
        "--out-root",
        default="work/maps",
        help="Output root. Final output is <out-root>/<name>/map_graph.json.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out_root) / args.map_name
    parser_cls = get_map_parser(args.source)
    manifest = parser_cls().parse(args.map_path, str(out_dir), args.map_name)
    print(f"map_name={manifest['map_name']}")
    print(f"map_path={manifest['map_path']}")
    print(f"out_path={manifest['out_path']}")
    print(f"proj={manifest['proj']}")
    for name, count in sorted(manifest["counts"].items()):
        print(f"{name}={count}")
    for family, counts in sorted(manifest["distributions"].items()):
        for name, count in counts.items():
            print(f"{family}.{name}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
