"""OpenScene 로그 → counterfactual.json 생성 CLI 드라이버 (FORMAT_SPEC §11).

실행은 항상 docker run (av-base — numpy만 사용):
    docker run --rm -v "$PWD":/workspace -w /workspace \\
      -v <openscene_root>:/data:ro av-base:latest \\
      python -m data_devkit.counterfactual.generate_openscene \\
        --logs /data/mini_navsim_logs/mini --out work/counterfactual/openscene_mini \\
        [--vocab data/model/sparsedrive_v2/kmeans/trajectory_1024_256.npz] [--stride 10]

산출: <out>/counterfactual.json (§11) + <out>/report.json (§11.2 생성 리포트).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from data_devkit.counterfactual.generator import (
    LANE_OFFSET_M,
    N_STEPS,
    VocabCoverage,
    build_payload,
    coverage_report,
    lane_transplant,
    time_transplant,
)

MIN_SPEED_MPS = 1.5       # 저속 앵커 제외 (차선 변경형 반사실이 무의미)
LOW_SPEED_FLAG_MPS = 3.0
DONOR_SPEED_TOL = 1.5
DONOR_LAT_DIFF_MIN = 2.0


def ego_future(frames: List[dict], idx: int) -> np.ndarray:
    """미래 N_STEPS 프레임의 pose를 현재 ego frame으로 (0.5s 간격, [x,y,heading])."""
    e0_inv = np.linalg.inv(np.asarray(frames[idx]["ego2global"]))
    rows = []
    for k in range(1, N_STEPS + 1):
        m = e0_inv @ np.asarray(frames[idx + k]["ego2global"])
        rows.append([m[0, 3], m[1, 3], float(np.arctan2(m[1, 0], m[0, 0]))])
    return np.asarray(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", required=True, help="OpenScene 로그 pkl 디렉토리")
    parser.add_argument("--out", required=True, help="산출 디렉토리")
    parser.add_argument("--vocab", default=None, help="scoring vocabulary npz (§11.2, 선택)")
    parser.add_argument("--stride", type=int, default=10, help="앵커 프레임 간격")
    parser.add_argument("--seed", type=int, default=7, help="donor 선택 시드")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = VocabCoverage.from_npz(args.vocab) if args.vocab else None
    rng = np.random.default_rng(args.seed)

    anchors_by_log: Dict[str, List[dict]] = {}
    for pkl_path in sorted(Path(args.logs).glob("*.pkl")):
        frames = pickle.load(open(pkl_path, "rb"))
        anchors = []
        for idx in range(4, len(frames) - N_STEPS - 2, args.stride):
            fr = frames[idx]
            v0 = float(np.linalg.norm(np.asarray(fr["ego_dynamic_state"][:2])))
            if v0 < MIN_SPEED_MPS:
                continue
            anchors.append(
                {"token": fr["token"], "v0": v0, "gt": ego_future(frames, idx)}
            )
        if anchors:
            anchors_by_log[pkl_path.stem] = anchors

    entries = []
    for log_name, anchors in anchors_by_log.items():
        for a in anchors:
            flags = ["low_speed_source"] if a["v0"] < LOW_SPEED_FLAG_MPS else []
            for sign in (+1.0, -1.0):
                entries.append(
                    lane_transplant(a["token"], a["gt"], sign * LANE_OFFSET_M, flags, vocab)
                )
            donors = [
                d for d in anchors
                if d["token"] != a["token"]
                and abs(d["v0"] - a["v0"]) < DONOR_SPEED_TOL
                and abs(float(d["gt"][-1, 1]) - float(a["gt"][-1, 1])) > DONOR_LAT_DIFF_MIN
            ]
            if donors:
                d = donors[int(rng.integers(len(donors)))]
                entries.append(
                    time_transplant(
                        a["token"], d["gt"], d["token"], log_name,
                        flags + [f"donor_speed_diff_{abs(d['v0'] - a['v0']):.1f}"], vocab,
                    )
                )

    (out_dir / "counterfactual.json").write_text(json.dumps(build_payload(entries)))
    report = coverage_report(entries)
    report["logs"] = len(anchors_by_log)
    report["anchors"] = sum(len(v) for v in anchors_by_log.values())
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
