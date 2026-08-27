"""Shared filesystem paths for OCP runtime artifacts.

The default behavior remains compatible with the original project: runtime
artifacts are written under ``<repo>/files``. Set ``OCP_FILES_DIR`` to redirect
generated models, trails, implementations, and temporary modeling files.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES_DIR = PROJECT_ROOT / "files"

# Subdirectories under the files directory for cached constraint templates.
SBOX_MODELING_SUBDIR = "sbox_modeling"
MATRIX_MODELING_SUBDIR = "matrix_modeling"


def get_files_dir(*parts, create=True):
    """Return the configured OCP files directory, optionally with subpaths."""
    base = Path(os.environ.get("OCP_FILES_DIR", DEFAULT_FILES_DIR)).expanduser()
    path = base.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_sbox_constraints_files_dir(create=True):
    """Return the directory holding cached S-box constraint templates."""
    return get_files_dir(SBOX_MODELING_SUBDIR, create=create)


def get_matrix_constraints_files_dir(create=True):
    """Return the directory holding cached matrix constraint templates."""
    return get_files_dir(MATRIX_MODELING_SUBDIR, create=create)
