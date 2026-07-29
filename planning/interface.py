"""planning 인터페이스 — ABC + registry + 동적 로딩 (C-SWM-023).

구 planning/policy/base.py + planning/input/base.py 통합.

계약
----
- 드라이버(run_inference.py, simulation/render_sim.py, build_input.py)는
  concrete 모델을 모른다.  root config의 ``modules.planning`` 이름으로
  :func:`load_model` 이 ``planning.models.<이름>`` 패키지를 import하고,
  그 패키지 ``__init__`` 이 registry에 자기 구현을 등록한다.
- 모델의 모든 코드는 ``planning/models/<이름>/`` 한 디렉토리가 소유한다.
- registry 키는 모듈 config(``planning/configs/<이름>.py``)가 정한다
  (policy / input_builder(feature adapter) / feed_builder / postprocess.name).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Type

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"


# --------------------------------------------------------------- dynamic load
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
        raise KeyError(
            f"Unknown planning model '{name}'. Available: {models}"
        )
    return import_module(f"planning.models.{name}")


# --------------------------------------------------------------------- policy
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


# -------------------------------------------------------------- input builder
_BUILDERS: Dict[str, Type["InputBuilder"]] = {}


class InputBuilder(ABC):
    """Plain numpy feed 빌더 (build_input.py 드라이버 대상)."""

    def __init__(self) -> None:
        self.last_context: Dict[str, object] = {}

    @abstractmethod
    def build(
        self,
        parsed_dir: str,
        map_graph: dict,
        t0_time: Optional[float],
        config: dict,
    ) -> Dict[str, np.ndarray]:
        """Build a model-ready feed dict from one parsed clip."""


def register_input_builder(name: str, builder_cls: Type[InputBuilder]) -> None:
    _BUILDERS[name] = builder_cls


def get_input_builder(name: str) -> Type[InputBuilder]:
    if name not in _BUILDERS:
        raise KeyError(
            f"Unknown input builder '{name}'. Available: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name]


# ------------------------------------------------------- feature adapter (=dataloader)
# 모델-facing dataloader: 완성된 feature 객체(예: collated PlutoFeature)와
# scene context를 만든다.  각 구현은 REQUIRES(데이터 계약 아티팩트 이름
# 목록)를 선언하고 data_devkit.contract.check로 fail-fast 검증해야 한다.
_FEATURE_ADAPTERS: Dict[str, type] = {}


def register_feature_adapter(name: str, adapter_cls: type) -> None:
    _FEATURE_ADAPTERS[name] = adapter_cls


def get_feature_adapter(name: str) -> type:
    if name not in _FEATURE_ADAPTERS:
        raise KeyError(
            f"Unknown feature adapter '{name}'. Available: {sorted(_FEATURE_ADAPTERS)}"
        )
    return _FEATURE_ADAPTERS[name]


# -------------------------------------------------------------- postprocessor
_POSTPROCESSORS: Dict[str, type] = {}


def register_postprocessor(name: str, postprocessor_cls: type) -> None:
    _POSTPROCESSORS[name] = postprocessor_cls


def get_postprocessor(name: str) -> type:
    if name not in _POSTPROCESSORS:
        raise KeyError(
            f"Unknown postprocessor '{name}'. Available: {sorted(_POSTPROCESSORS)}"
        )
    return _POSTPROCESSORS[name]
