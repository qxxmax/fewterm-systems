"""Repo-relative locations of vendored code (third_party/) and reference data (data/).

Every path can be overridden by an environment variable; nothing outside the repository is used by default.
  FEWTERM_DATA        root of the small reference data (default <repo>/data)
  FEWTERM_PHI4_REF    stored 2D phi^4 L = 8 reference run (default <FEWTERM_DATA>/phi4_regress_L8_k0.270)
"""
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO / 'third_party'


def data_dir():
    return Path(os.environ.get('FEWTERM_DATA', REPO / 'data'))


def klein_dir():
    return Path(os.environ.get('FEWTERM_KLEIN_DATA', data_dir() / 'klein'))


def phi4_ref_dir():
    return Path(os.environ.get('FEWTERM_PHI4_REF', data_dir() / 'phi4_regress_L8_k0.270'))


def third_party(*parts, add=True):
    """Path of a vendored code folder (see third_party/SOURCES.md); optionally prepended to sys.path."""
    p = THIRD_PARTY.joinpath(*parts)
    if not p.is_dir():
        raise FileNotFoundError(f'vendored code folder missing: {p}')
    if add and str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p
