from typing import Dict, Type

from data_devkit.parsers.base import EgoPoseProvider, MapParser, SourceParser

_PARSERS: Dict[str, Type[SourceParser]] = {}
_MAP_PARSERS: Dict[str, Type[MapParser]] = {}
_POSE_PROVIDERS: Dict[str, Type[EgoPoseProvider]] = {}


def register_parser(name: str, parser_cls: Type[SourceParser]) -> None:
    _PARSERS[name] = parser_cls


def get_parser(name: str) -> Type[SourceParser]:
    if name not in _PARSERS:
        raise KeyError(f"Unknown parser '{name}'. Available: {sorted(_PARSERS)}")
    return _PARSERS[name]


def register_map_parser(name: str, parser_cls: Type[MapParser]) -> None:
    _MAP_PARSERS[name] = parser_cls


def get_map_parser(name: str) -> Type[MapParser]:
    if name not in _MAP_PARSERS:
        raise KeyError(
            f"Unknown map parser '{name}'. Available: {sorted(_MAP_PARSERS)}"
        )
    return _MAP_PARSERS[name]


def register_pose(name: str, pose_cls: Type[EgoPoseProvider]) -> None:
    _POSE_PROVIDERS[name] = pose_cls


def get_pose_provider(name: str) -> Type[EgoPoseProvider]:
    if name not in _POSE_PROVIDERS:
        raise KeyError(
            f"Unknown pose provider '{name}'. Available: {sorted(_POSE_PROVIDERS)}"
        )
    return _POSE_PROVIDERS[name]
