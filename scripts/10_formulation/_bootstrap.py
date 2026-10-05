"""Path bootstrap so scripts can `import src...` when run directly.

Every script does::

    import _bootstrap  # noqa: F401  (adds repo root to sys.path)

placed before the `from src...` imports. It walks upward from this file until
it finds the directory that contains ``src/``, so it works no matter how deep
the script lives under ``scripts/`` and regardless of the working directory.
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve()
_repo = None
for parent in _here.parents:
    if (parent / "src").is_dir() and (parent / "config").is_dir():
        _repo = parent
        break
if _repo is None:
    _repo = _here.parents[1]  # fallback

if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
