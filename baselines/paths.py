"""Repo-relative locations for the convolutional flow of the paper (baselines/README.md).

Both locations can be overridden by an environment variable:
  FEWTERM_NF_GRID         trained convolutional flows, <dir>/k{kappa}/nf_L{L}_{single|mix}_s{seed}.{pt,meta.json}
  FEWTERM_BASELINE_OUT    outputs of training and evaluation (default <repo>/baselines/runs)
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def nf_grid():
    return Path(os.environ.get('FEWTERM_NF_GRID', HERE / 'runs'))


def out():
    return Path(os.environ.get('FEWTERM_BASELINE_OUT', HERE / 'runs'))
