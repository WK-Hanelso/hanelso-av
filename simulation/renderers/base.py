"""Renderer ABC + registry.

드라이버(EgoDriver)는 렌더러 구현을 모른다: 매 프레임 frame dict 하나를
render_frame()에 넘길 뿐이다.

frame dict 계약 (드라이버가 채우고, 각 렌더러는 필요한 키만 소비):
    build            AdapterBuildResult (feature/scene_context/normalized data)
    output           policy.infer() 반환 dict
    post             PlutoPostProcessor.run() 결과 (없으면 None)
    iteration        int — 렌더 프레임/스텝 번호
    ego_pose         (3,) [x, y, heading] — 현재 렌더 기준 ego (rear-axle)
    ego_dims         (2,) [width, length]
    agents           frame_agents() 반환 리스트
    title            str
    info_lines       List[str]
    emergency        bool
    raw_traj_global  optional (T, >=2) — open_loop raw 궤적 (global)
    best_traj_global optional (T, >=2) — 후처리 best 궤적 (global)
    log_ego_pose     optional (3,) — closed_loop의 log ghost
    sim_trace        optional (N, 3) — closed_loop 누적 sim 경로
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Type

_RENDERERS: Dict[str, Type["Renderer"]] = {}


class Renderer(ABC):
    #: appended to the mode directory name (e.g. "_nuplan" -> closed_loop_nuplan/)
    suffix: str = ""

    def __init__(self, clip: Dict[str, Any], map_graph: dict, sim_cfg: Dict[str, Any], mode: str) -> None:
        self.clip = clip
        self.map_graph = map_graph
        self.sim_cfg = sim_cfg
        self.mode = mode

    @abstractmethod
    def render_frame(self, out_path: Path, frame: Dict[str, Any]) -> None:
        """Renders one frame dict (contract above) to out_path."""


def register_renderer(name: str, renderer_cls: Type[Renderer]) -> None:
    _RENDERERS[name] = renderer_cls


def get_renderer(name: str) -> Type[Renderer]:
    if name not in _RENDERERS:
        raise KeyError(f"Unknown renderer '{name}'. Available: {sorted(_RENDERERS)}")
    return _RENDERERS[name]
