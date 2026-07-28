from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

_POLICIES: Dict[str, Type["Policy"]] = {}


class Policy(ABC):
    @abstractmethod
    def infer(self, feature: Any) -> dict[str, Any]:
        """Run one forward pass on one already-built feature batch."""


def register_policy(name: str, policy_cls: Type[Policy]) -> None:
    _POLICIES[name] = policy_cls


def get_policy(name: str) -> Type[Policy]:
    if name not in _POLICIES:
        raise KeyError(f"Unknown policy '{name}'. Available: {sorted(_POLICIES)}")
    return _POLICIES[name]
