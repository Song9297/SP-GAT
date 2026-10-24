"""Portable paths for the source-only research release.

CLI paths are rooted at the directory containing this module. Persisted artifact
references use forward slashes and are relative to an explicit containing folder.
"""
import os
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent


def resolve_path(value):
    """Resolve a CLI path independently of the process working directory."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def path_reference(value, relative_to=None):
    """Serialize an artifact path relative to its manifest folder or project root."""
    path = resolve_path(value)
    base = resolve_path(relative_to) if relative_to is not None else ROOT
    try:
        reference = os.path.relpath(str(path), str(base))
    except ValueError as exc:
        raise ValueError(
            'Portable artifact references require the project, checkpoints and results on the '
            'same drive. Keep their directory tree together when moving it.'
        ) from exc
    return Path(reference).as_posix()


def resolve_reference(value, relative_to=None):
    """Read a persisted relative reference; never silently use an old absolute path."""
    reference = str(value)
    windows = PureWindowsPath(reference)
    if not reference or Path(reference).is_absolute() or windows.drive or windows.root:
        raise ValueError(
            'Expected a relative artifact reference. Regenerate legacy manifests '
            'with this release instead of copying machine-specific records.'
        )
    base = resolve_path(relative_to) if relative_to is not None else ROOT
    return (base / Path(reference.replace('\\', '/'))).resolve()


def dataset_reference(value):
    """Return a non-identifying dataset label; the content hash establishes identity."""
    path = resolve_path(value)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def require_data(value):
    """Fail before fitting when the privately supplied dataset is absent."""
    path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(
            'Research data are not distributed in this source release. Place the '
            'authorized dataset at data/AGB.csv, or pass --data PATH. '
            'See data/README.md for the required format.'
        )
    return path
