# Few-parameter flow for lattice phi^4 theory

Code for the paper "A few-parameter flow for lattice phi^4 theory fitted by linear least squares"
(M. Qian).

The flow transports a base distribution to the Boltzmann distribution of lattice phi^4 theory. Its velocity field
is a linear combination of 16 field functions, and flow matching reduces the training to linear least squares on
about one thousand Monte Carlo configurations. The density of the flow is exact, so reweighting and an independence
Metropolis-Hastings chain correct the proposals to the target distribution.

## Quick start

```bash
conda env create -f environment.yml && conda activate fewterm     # or: pip install -r requirements.txt
jupyter notebook notebooks/phi4_2d_main_results.ipynb
```

The notebook reruns the main results at L = 8, 12 and 16 for four couplings and compares them with long HMC chains
(about 15 minutes on 4 CPU threads). It shows the stored output of such a run.

## Where things are in the code

| paper | code |
|---|---|
| action, observables | `fewterm/phi4.py`: `action`, `observables` |
| the 16 field functions | `fewterm/phi4.py` (`OperatorCNF.basis`, `RadialCNF`, `ExtCNF.new_basis`) and `fewterm/phi4u.py` (`CoupledCNF.features`) |
| zero-mode base distribution | `fewterm/landau.py`: `LandauBase`, `from_pool` |
| zero-mode pairing | `fewterm/phi4u.py`: `zm_ot_pair` |
| least-squares fit | `fewterm/phi4u.py`: `fit_coupled` |
| Strang splitting, implicit midpoint, log-Jacobian | `fewterm/phi4u.py`: `CoupledCNF.transform`, `coupling_substep`, `UnboundedCNF.substep` |
| fitting configurations (HMC with cluster updates) | `fewterm/wolff.py`: `fit_pool` |
| HMC reference chains, autocorrelation times | `scripts/hmc_tau.py`, `scripts/tau_compare.py`, `scripts/imh_single.py` |
| thermodynamic integration | `scripts/ti_node.py`, `scripts/free_energy_ti.py` |
| base distributions without and with pairing (Appendix C) | `scripts/landau_compare.py`, `scripts/base_is.py` |
| number of fitting configurations (Discussion) | `scripts/datasize_landau.py` |
| convolutional flow | `baselines/` (see `baselines/README.md`) |
| figures and tables | `paper/` (see `paper/README.md`) |

The code works in d dimensions.

## Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -m pytest fewterm/tests -q
```

The tests check the log-Jacobian against automatic differentiation, the splitting against a fine Runge-Kutta
integration, the HMC and Wolff kernels against the original code (`third_party/`), and a stored L = 8 fit bit for
bit (`data/`). `FEWTERM_SLOW=1` adds a small three-dimensional end-to-end run.

## License

MIT (see `LICENSE`). The convolutional-flow training scripts in `baselines/nf/` are adapted from the
tutorial of Albergo et al., arXiv:2101.08176, as noted in its header.
