"""Sim ego-drivers: register implementations on import."""

from .base import MODE_TO_DRIVER, EgoDriver, get_ego_driver, register_ego_driver
from . import log_replay  # noqa: F401  (registers "log_replay")
from . import model_driven  # noqa: F401  (registers "model_driven")

__all__ = ["MODE_TO_DRIVER", "EgoDriver", "get_ego_driver", "register_ego_driver"]
