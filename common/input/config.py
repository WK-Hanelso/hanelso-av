from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BuildInputConfig:
    parsed_dir: str
    map_path: str
    builder: str = "pluto"
    clip_id: Optional[str] = None
    map_name: Optional[str] = None
    vehicle: str = "pacifica"
    t0_time: Optional[float] = None
    out_root: str = "work/input"
