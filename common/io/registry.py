from typing import Dict, Type

from common.io.base import EgoPoseProvider, SourceParser

_PARSERS: Dict[str, Type[SourceParser]] = {}
_POSE_PROVIDERS: Dict[str, Type[EgoPoseProvider]] = {}


def register_parser(name: str, parser_cls: Type[SourceParser]) -> None:
    _PARSERS[name] = parser_cls


def get_parser(name: str) -> Type[SourceParser]:
    if name not in _PARSERS:
        raise KeyError(f"Unknown parser '{name}'. Available: {sorted(_PARSERS)}")
    return _PARSERS[name]


def register_pose(name: str, pose_cls: Type[EgoPoseProvider]) -> None:
    _POSE_PROVIDERS[name] = pose_cls


def get_pose_provider(name: str) -> Type[EgoPoseProvider]:
    if name not in _POSE_PROVIDERS:
        raise KeyError(
            f"Unknown pose provider '{name}'. Available: {sorted(_POSE_PROVIDERS)}"
        )
    return _POSE_PROVIDERS[name]

