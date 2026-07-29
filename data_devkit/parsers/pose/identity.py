from data_devkit.parsers.base import EgoPoseProvider
from data_devkit.parsers.registry import register_pose


class IdentityPoseProvider(EgoPoseProvider):
    def prepare(self, record_path: str) -> None:
        return None

    def pose_at(self, timestamp_ns: int) -> dict:
        return {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}


register_pose("identity", IdentityPoseProvider)

