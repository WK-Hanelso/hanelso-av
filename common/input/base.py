from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np

_BUILDERS: Dict[str, Type["InputBuilder"]] = {}


class InputBuilder(ABC):
    def __init__(self) -> None:
        self.last_context: Dict[str, object] = {}

    @abstractmethod
    def build(
        self,
        parsed_dir: str,
        map_graph: dict,
        t0_time: Optional[float],
        config: dict,
    ) -> dict[str, np.ndarray]:
        """Build a model-ready feed dict from one parsed clip."""


def register_input_builder(name: str, builder_cls: Type[InputBuilder]) -> None:
    _BUILDERS[name] = builder_cls


def get_input_builder(name: str) -> Type[InputBuilder]:
    if name not in _BUILDERS:
        raise KeyError(
            f"Unknown input builder '{name}'. Available: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name]
