import argparse
import json
from pathlib import Path


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify parsed unified JSON tables.")
    parser.add_argument("--out", required=True, help="Parsed output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    samples = _load_json(out_dir / "sample.json")
    ego_poses = _load_json(out_dir / "ego_pose.json")
    annotations = _load_json(out_dir / "sample_annotation.json")
    instances = _load_json(out_dir / "instance.json")
    categories = _load_json(out_dir / "category.json")
    scenes = _load_json(out_dir / "scene.json")
    ego_dynamics = _load_json(out_dir / "ego_dynamics.json")

    assert samples, "sample.json is empty"
    assert ego_poses, "ego_pose.json is empty"
    assert len(samples) == len(ego_poses), "sample and ego_pose counts must match"
    assert len(samples) == len(ego_dynamics), "sample and ego_dynamics counts must match"
    assert len(scenes) == 1, "v1 expects exactly one scene"

    sample_tokens = {row["token"] for row in samples}
    ego_pose_tokens = {row["token"] for row in ego_poses}
    instance_by_token = {row["token"]: row for row in instances}
    category_tokens = {row["token"] for row in categories}
    annotation_by_token = {row["token"]: row for row in annotations}

    for scene in scenes:
        assert scene["first_sample_token"] in sample_tokens
        assert scene["last_sample_token"] in sample_tokens
        assert scene["nbr_samples"] == len(samples)

    for index, sample in enumerate(samples):
        if index == 0:
            assert sample["prev"] == ""
        else:
            assert sample["prev"] == samples[index - 1]["token"]
        if index == len(samples) - 1:
            assert sample["next"] == ""
        else:
            assert sample["next"] == samples[index + 1]["token"]

    for row in ego_dynamics:
        assert row["sample_token"] in sample_tokens
        assert row["ego_pose_token"] in ego_pose_tokens

    for annotation in annotations:
        assert annotation["sample_token"] in sample_tokens
        assert annotation["instance_token"] in instance_by_token
        if annotation["prev"]:
            assert annotation["prev"] in annotation_by_token
        if annotation["next"]:
            assert annotation["next"] in annotation_by_token

    for instance in instances:
        assert instance["category_token"] in category_tokens
        assert instance["first_annotation_token"] in annotation_by_token
        assert instance["last_annotation_token"] in annotation_by_token
        count = sum(
            1 for annotation in annotations if annotation["instance_token"] == instance["token"]
        )
        assert count == instance["nbr_annotations"]

    print(f"verified_samples={len(samples)}")
    print(f"verified_ego_poses={len(ego_poses)}")
    print(f"verified_annotations={len(annotations)}")
    print(f"verified_instances={len(instances)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
