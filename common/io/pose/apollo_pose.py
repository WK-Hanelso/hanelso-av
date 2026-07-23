from bisect import bisect_left
from math import cos, sin
from typing import Dict, List

from cyber_record.record import Record

from common.io.base import EgoPoseProvider
from common.io.registry import register_pose

POSE_TOPIC = "/apollo/localization/pose"


def _nearest_index(timestamps: List[int], target: int) -> int:
    if not timestamps:
        raise ValueError("Apollo pose provider has no poses loaded.")
    index = bisect_left(timestamps, target)
    if index == 0:
        return 0
    if index == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[index - 1]
    after = timestamps[index]
    if target - before <= after - target:
        return index - 1
    return index


def _yaw_to_quaternion(yaw: float) -> List[float]:
    half = yaw * 0.5
    return [cos(half), 0.0, 0.0, sin(half)]


class ApolloRecordPoseProvider(EgoPoseProvider):
    def __init__(self) -> None:
        self._timestamps: List[int] = []
        self._poses: List[Dict[str, List[float]]] = []

    def prepare(self, record_path: str) -> None:
        self._timestamps = []
        self._poses = []
        for topic, msg, timestamp_ns in Record(record_path).read_messages():
            if topic != POSE_TOPIC:
                continue
            self._timestamps.append(int(timestamp_ns))
            self._poses.append(
                {
                    "translation": [
                        float(msg.pose.position.x),
                        float(msg.pose.position.y),
                        float(msg.pose.position.z),
                    ],
                    "rotation": _yaw_to_quaternion(float(msg.pose.heading)),
                }
            )
        if not self._timestamps:
            raise ValueError("No Apollo localization poses found in record.")

    def pose_at(self, timestamp_ns: int) -> dict:
        index = _nearest_index(self._timestamps, timestamp_ns)
        return self._poses[index]


register_pose("apollo_record", ApolloRecordPoseProvider)

