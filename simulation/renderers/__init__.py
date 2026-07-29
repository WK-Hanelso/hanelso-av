"""Sim renderers: register implementations on import."""

from .base import Renderer, get_renderer, register_renderer
from . import matplotlib_bev  # noqa: F401  (registers "matplotlib")
from . import nuplan_official  # noqa: F401  (registers "nuplan")

__all__ = ["Renderer", "get_renderer", "register_renderer"]
