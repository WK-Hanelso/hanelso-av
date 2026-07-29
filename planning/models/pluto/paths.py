"""Single place that puts the vendored third_party/pluto on sys.path.

All planning-side imports of the original PLUTO package (``src.*``) must go
through :func:`ensure_pluto_on_path`.  External pluto checkouts (absolute
paths outside this repo) are forbidden — the repo is self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLUTO_ROOT = REPO_ROOT / "third_party" / "pluto"


def ensure_pluto_on_path(pluto_root: Optional[Union[str, Path]] = None) -> Path:
    """Inserts the vendored pluto root into sys.path (idempotent).

    Returns the resolved pluto root.  ``pluto_root=None`` uses the vendored
    default ``<repo>/third_party/pluto``.
    """
    root = Path(pluto_root) if pluto_root else DEFAULT_PLUTO_ROOT
    if not (root / "src").exists():
        raise FileNotFoundError(
            f"pluto root not found or has no src/: {root} "
            "(expected vendored third_party/pluto)"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
