import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.parser_validation.core import (
    default_clip_id,
    render_bev,
    render_map_bev,
    render_text_report,
    summarize_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one Apollo cyber record and write a report.")
    parser.add_argument("--record", help="Path to Apollo .record file")
    parser.add_argument("--map-only", help="Render only work/maps/<map_name>/map_bev.png from map_graph.json")
    parser.add_argument(
        "--maps-root",
        default="work/maps",
        help="Directory that contains */map_graph.json candidates",
    )
    parser.add_argument(
        "--out-root",
        default="work/validation",
        help="Root directory for validator outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.record) == bool(args.map_only):
        raise SystemExit("Provide exactly one of --record or --map-only.")

    maps_root = Path(args.maps_root)
    if args.map_only:
        map_name = str(args.map_only)
        map_bev_path = render_map_bev(map_name, maps_root)
        print(f"map_name={map_name}")
        print(f"map_bev_png={map_bev_path}")
        return

    record_path = Path(args.record)
    clip_id = default_clip_id(str(record_path))
    out_dir = Path(args.out_root) / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_record(str(record_path), maps_root)
    matched_map = summary["map_match"]["matched_map"]
    if matched_map:
        summary["visualization"]["map_bev_png"] = render_map_bev(str(matched_map), maps_root)
    bev_outputs = render_bev(summary, out_dir)
    summary["visualization"].update(bev_outputs)
    text_report = render_text_report(summary)

    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "report.txt").write_text(text_report)

    print(f"clip_id={clip_id}")
    print(f"out_dir={out_dir}")
    print(f"matched_map={matched_map}")
    print(f"reroute={summary['channel_summaries']['routing']['reroute']}")
    print(f"validation={summary['validation']['status']}")
    print(f"scene_bev_png={summary['visualization']['bev_png']}")
    print(f"map_bev_png={summary['visualization'].get('map_bev_png')}")


if __name__ == "__main__":
    main()
