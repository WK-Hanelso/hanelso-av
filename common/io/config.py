from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParseConfig:
    record: str
    source: str = "apollo_record"
    pose: str = "apollo_record"
    clip_id: Optional[str] = None
    out_root: str = "work"
