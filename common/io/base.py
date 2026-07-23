from abc import ABC, abstractmethod
from typing import Dict


class SourceParser(ABC):
    @abstractmethod
    def parse(
        self,
        record_path: str,
        out_dir: str,
        clip_id: str,
        pose_provider: "EgoPoseProvider",
    ) -> Dict[str, object]:
        """Parse one source clip into the unified table set."""


class EgoPoseProvider(ABC):
    @abstractmethod
    def prepare(self, record_path: str) -> None:
        """Load source data needed to answer pose queries."""

    @abstractmethod
    def pose_at(self, timestamp_ns: int) -> Dict[str, object]:
        """Return a pose dict with translation and quaternion rotation."""

