"""PLUTO 모델 패키지 — import가 registry 등록 진입점 (동적 로딩).

등록 키: policy="pluto_torch", dataloader="pluto_feature",
postprocessor="pluto".
"""

from planning.models.pluto import dataloader  # noqa: F401  (registers "pluto_feature")
from planning.models.pluto import policy  # noqa: F401  (registers "pluto_torch")
from planning.models.pluto import postprocess  # noqa: F401  (registers "pluto")
