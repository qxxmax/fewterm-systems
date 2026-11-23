"""HMC + Wolff reference chains for one (d, L, kappa, lambda) point (fewterm.wolff.mcmc, 4 chains).

    python scripts/ref_mcmc.py --d 3 --L 8 --kappa 0.17 --seed S --out DIR [--n-rec 65536]

Writes DIR/chains.npz (traces of M, phi2, energy per update) and DIR/record.json; the IMH chains of
fewterm.phi4.flow are compared against these.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewterm.phi4 import LAM  # noqa: E402
from fewterm.wolff import mcmc  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--d', type=int, default=3)
p.add_argument('--L', type=int, required=True)
p.add_argument('--kappa', type=float, required=True)
p.add_argument('--lam', type=float, default=LAM)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--n-rec', type=int, default=65536)
p.add_argument('--warm', type=int, default=4096)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(1)
_, info = mcmc(a.L, a.kappa, 'hmcwolff', a.seed, d=a.d, lam=a.lam, warm=a.warm, n_rec=a.n_rec, record=a.out)
print(info, flush=True)
