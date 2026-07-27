import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.parser_validation.core import default_clip_id, render_text_report, summarize_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one Apollo cyber record and write a report.")
    parser.add_argument("--record", required=True, help="Path to Apollo .record file")
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
    record_path = Path(args.record)
    clip_id = default_clip_id(str(record_path))
    out_dir = Path(args.out_root) / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_record(str(record_path), Path(args.maps_root))
    text_report = render_text_report(summary)

    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "report.txt").write_text(text_report)

    print(f"clip_id={clip_id}")
    print(f"out_dir={out_dir}")
    print(f"matched_map={summary['map_match']['matched_map']}")
    print(f"reroute={summary['channel_summaries']['routing']['reroute']}")


if __name__ == "__main__":
    main()
