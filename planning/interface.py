"""planning 인터페이스 — ABC + registry + 동적 로딩 (C-SWM-025).

계약
----
- 드라이버(``planning/run_inference.py``, ``simulation/render_sim.py``)는
  concrete 모델을 모른다. root config의 ``modules.planning`` 이름으로
  :func:`load_model` 이 ``planning.models.<이름>`` 패키지를 import하고,
  그 패키지 ``__init__`` 이 registry에 자기 구현을 등록한다.
- 모델의 모든 코드는 ``planning/models/<이름>/`` 한 디렉토리가 소유한다.
- registry 키는 모듈 config(``planning/configs/<이름>.py``)가 정한다
  (policy / dataloader / postprocess.name).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Type

MODELS_DIR = Path(__file__).resolve().parent / "models"


def available_models() -> List[str]:
    """planning/models/ 스캔 — __init__.py 있는 하위 디렉토리 = 모델."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and (p / "__init__.py").exists()
    )


def load_model(name: Optional[str]) -> ModuleType:
    """``planning.models.<name>`` 을 import해 registry 등록을 일으킨다."""
    models = available_models()
    if not name or name not in models:
        raise KeyError(f"Unknown planning model '{name}'. Available: {models}")
    return import_module(f"planning.models.{name}")


_POLICIES: Dict[str, Type["Policy"]] = {}


class Policy(ABC):
    @abstractmethod
    def infer(self, feature: Any) -> Dict[str, Any]:
        """Run one forward pass on one already-built feature batch."""


def register_policy(name: str, policy_cls: Type[Policy]) -> None:
    _POLICIES[name] = policy_cls


def get_policy(name: str) -> Type[Policy]:
    if name not in _POLICIES:
        raise KeyError(f"Unknown policy '{name}'. Available: {sorted(_POLICIES)}")
    return _POLICIES[name]


_DATALOADERS: Dict[str, Type["Dataloader"]] = {}


class Dataloader(ABC):
    REQUIRES: List[str]

    @abstractmethod
    def build(
        self,
        parsed_dir: str,
        map_graph: dict,
        config: dict,
        t0_time: Optional[float] = None,
    ) -> Any:
        """Build one inference input from one parsed clip.

        Returns an object that carries:
        - ``feature``: policy-ready collated feature batch.
        - ``context``: JSON-serializable adapter metadata for reporting.
        - ``normalized_numpy_data``: numpy dict used by rendering/checks.
        - ``scene_context``: postprocess inputs, or ``None`` when unused.
        """

    @abstractmethod
    def prepare_clip(
        self,
        parsed_dir: str,
        map_graph: dict,
        config: dict,
    ) -> Dict[str, Any]:
        """Load clip-scoped simulation inputs once.

        Returns a dict bundle that ``build_frame`` can reuse across frames
        (parsed tables, route payload, map API, config, and timing metadata).
        """

    @abstractmethod
    def build_frame(
        self,
        clip: Dict[str, Any],
        t0_index: int,
        sim_ego: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build one simulation frame input from a ``prepare_clip`` bundle.

        Returns the same contract shape as :meth:`build` for one frame.
        """


def register_dataloader(name: str, dataloader_cls: Type[Dataloader]) -> None:
    _DATALOADERS[name] = dataloader_cls


def get_dataloader(name: str) -> Type[Dataloader]:
    if name not in _DATALOADERS:
        raise KeyError(
            f"Unknown dataloader '{name}'. Available: {sorted(_DATALOADERS)}"
        )
    return _DATALOADERS[name]


_POSTPROCESSORS: Dict[str, Type["Postprocessor"]] = {}


class Postprocessor(ABC):
    @abstractmethod
    def run(
        self,
        model_output: Dict[str, Any],
        normalized_data: Dict[str, Any],
        scene_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Select the deployed trajectory from model outputs.

        Returns a dict that includes the chosen local/global trajectory, score
        vectors, safety metrics, emergency-brake flag, and evaluator byproducts
        consumed by reports or simulation drivers.
        """


def register_postprocessor(
    name: str, postprocessor_cls: Type[Postprocessor]
) -> None:
    _POSTPROCESSORS[name] = postprocessor_cls


def get_postprocessor(name: str) -> Type[Postprocessor]:
    if name not in _POSTPROCESSORS:
        raise KeyError(
            f"Unknown postprocessor '{name}'. Available: {sorted(_POSTPROCESSORS)}"
        )
    return _POSTPROCESSORS[name]
