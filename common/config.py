"""계층 config 조합 로더.

계약
----
- ROOT config (``configs/*.py``) = 유일 진입점. "무엇을 쓸지"(이름)만 적는다::

      config = dict(
          clip_id=..., record="data/bag/...", map_name="AYG",
          modules=dict(planning="pluto", perception=None, localization=None),
          calibration="e100",
          simulation="closed_loop_nuplan",
      )

- 모듈 config (``<domain>/configs/<이름>.py``)는 그 이름으로 해석돼 병합된다.
- 학습이 결정한 값은 bundle 안의 native config(hydra 산출물)를 **경로로 참조**
  한다(전사 금지). 우리가 결정하는 값만 우리 config에 둔다.
- ``_base_`` (mmdet식, 모듈 전역 변수, str 또는 list)로 다른 config .py를
  상속할 수 있다. dict는 재귀 병합, 그 외 타입은 자식이 덮어쓴다.

사용
----
    from common.config import load_config
    cfg = load_config("configs/e100bt25.py")
    cfg["planning"]["policy"]   # -> "pluto_torch"
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[1]

# 모듈 이름을 해석하는 도메인들.  root config의 modules=dict(...) 항목은
# _MODULE_DOMAINS, 단일 문자열 항목은 _SINGLETON_DOMAINS 로 해석된다.
_MODULE_DOMAINS = ("planning", "perception", "localization")
_SINGLETON_DOMAINS = ("calibration", "simulation")


def resolve_repo_path(path: Union[str, Path]) -> Path:
    """repo-상대 경로를 절대 경로로. 이미 절대면 그대로."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """config .py 하나를 실행해 `config` dict를 반환 (``_base_`` 상속 지원)."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"_config_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config") or not isinstance(module.config, dict):
        raise TypeError(f"{path}: must define a dict named `config`")
    cfg = dict(module.config)

    bases = getattr(module, "_base_", None)
    if bases:
        if isinstance(bases, (str, Path)):
            bases = [bases]
        merged: Dict[str, Any] = {}
        for base_rel in bases:
            merged = _deep_merge(merged, load_config_file(path.parent / base_rel))
        cfg = _deep_merge(merged, cfg)
    return cfg


def _available_names(domain: str) -> list:
    config_dir = REPO_ROOT / domain / "configs"
    if not config_dir.is_dir():
        return []
    return sorted(p.stem for p in config_dir.glob("*.py") if p.stem != "__init__")


def resolve_module_config(domain: str, name: Optional[str]) -> Optional[Dict[str, Any]]:
    """<domain>/configs/<name>.py 를 로드. name=None이면 None(도메인 미사용)."""
    if name is None:
        return None
    path = REPO_ROOT / domain / "configs" / f"{name}.py"
    if not path.exists():
        raise KeyError(
            f"Unknown {domain} config '{name}': {path} not found. "
            f"Available: {_available_names(domain)}"
        )
    return load_config_file(path)


def load_config(
    root_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """ROOT config를 읽고 모듈 이름들을 모듈 config로 해석·병합해 반환.

    반환 dict = root 필드 그대로 + 각 도메인 키(planning/perception/
    localization/calibration/simulation)에 해석된 모듈 config dict(미사용
    도메인은 None).  ``overrides``는 최종 dict 위에 재귀 병합된다(CLI용).
    """
    root = load_config_file(resolve_repo_path(root_path))
    resolved = dict(root)
    modules = dict(root.get("modules") or {})
    unknown = set(modules) - set(_MODULE_DOMAINS)
    if unknown:
        raise KeyError(
            f"Unknown module domains in root config: {sorted(unknown)}. "
            f"Expected subset of {_MODULE_DOMAINS}"
        )
    for domain in _MODULE_DOMAINS:
        resolved[domain] = resolve_module_config(domain, modules.get(domain))
    for domain in _SINGLETON_DOMAINS:
        resolved[domain] = resolve_module_config(domain, root.get(domain))
    if overrides:
        resolved = _deep_merge(resolved, overrides)
    return resolved
