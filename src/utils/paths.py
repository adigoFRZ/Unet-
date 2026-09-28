"""Project-root-relative path resolution.

Why this exists: a config value like ``cache/baseline_v1`` is only meaningful
relative to the project root. Resolving it with ``Path(...).resolve()`` would
anchor it to the *current working directory* instead, so launching the trainer
from anywhere other than the project root would silently read the wrong
directory -- or no directory, if one happened to exist at that path.

Every path that comes out of a config goes through :func:`resolve_path`, which
anchors relative paths to an explicit project root. The default project root is
derived from this file's own location, so it is correct regardless of the shell's
working directory.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, derived from this module's location:
#: <root>/src/utils/paths.py -> parents[2] == <root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve_project_root(value: str | Path | None = None) -> Path:
    """Resolve a configured project root.

    Absolute values are used as given. Relative values (including the common
    default ``"."``) are anchored to :data:`REPO_ROOT`, *not* to the current
    working directory.
    """
    if value is None:
        return REPO_ROOT
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if str(value).strip() in ("", "."):
        return REPO_ROOT
    return (REPO_ROOT / path).resolve()


def resolve_path(value: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve a possibly-relative path against ``project_root``."""
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root = Path(project_root).resolve() if project_root is not None else REPO_ROOT
    return (root / path).resolve()
