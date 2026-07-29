"""EgoDriver ABC + registry.

ego를 누가 움직이는가의 추상화:
    log_replay   : ego가 로그(GT)를 따라감 — open_loop
    model_driven : ego를 모델이 운전 (원본 pluto ForwardSimulator 전파) — closed_loop

드라이버는 렌더러 구현과 무관하게 매 프레임 frame dict(renderers/base.py 계약)를
만들어 renderer.render_frame()에 넘긴다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Type

_DRIVERS: Dict[str, Type["EgoDriver"]] = {}

#: root/simulation config의 mode 문자열 -> 드라이버 registry key
MODE_TO_DRIVER = {"open_loop": "log_replay", "closed_loop": "model_driven"}


class EgoDriver(ABC):
    def __init__(
        self,
        adapter: Any,
        policy: Any,
        postprocessor: Any,
        clip: Dict[str, Any],
        sim_cfg: Dict[str, Any],
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.postprocessor = postprocessor
        self.clip = clip
        self.sim_cfg = sim_cfg

    @abstractmethod
    def run(self, renderer: Any, frames_dir: Path, mode_name: str, out_dir: Path) -> Dict[str, Any]:
        """Runs the loop, renders frames + mp4, returns the result/metrics dict."""


def register_ego_driver(name: str, driver_cls: Type[EgoDriver]) -> None:
    _DRIVERS[name] = driver_cls


def get_ego_driver(name: str) -> Type[EgoDriver]:
    if name not in _DRIVERS:
        raise KeyError(f"Unknown ego driver '{name}'. Available: {sorted(_DRIVERS)}")
    return _DRIVERS[name]
