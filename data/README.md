# Reference data

`phi4_regress_L8_k0.270/` is a stored 2D phi^4 run at L = 8, kappa = 0.270, lambda = 0.022, used by
`fewterm/tests/test_regression_2d.py` to check that the code reproduces it bit for bit.

- `pool_0/fit_pool.pt`: 1024 fitting configurations (1024 x 8 x 8, float64) and `pool_0/fit_pool_setup.json`.
- `flow/*.json`: coefficients of the stored fits (`D192` and `D224` are the 12- and 14-term stages, see
  `fewterm/phi4.py`) and their step-convergence checks.
- `flow/proposals_head1024.npz`: the first 1024 proposals of the stored run.

The location can be changed with `FEWTERM_DATA` or `FEWTERM_PHI4_REF`.
