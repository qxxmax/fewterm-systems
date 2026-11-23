# Vendored code

Two small packages from the author's earlier 2D phi^4 code, used only by `fewterm/tests/test_wolff.py` to check
the HMC, Wolff and observable kernels of `fewterm.wolff` bit for bit. The code is unchanged; the hashes below are those of the files in this repository.
`phi4_nf/__init__.py` is new (the original imports modules that are not needed here).

| folder | files |
|---|---|
| `phi4_wolff/src/` | `action.py`, `hmc_wolff_tau_phi4_fullsave.py` |
| `phi4_nf/phi4_nf/` | `kernels.py`, `diagnostics.py`, `physics.py` |

SHA-256
```
dac58f01ad98082cdb0540fb38c4396432f0c6b878781e5cc351ad152d5e0646  phi4_wolff/src/action.py
9d53ddcc01a4bc605076bccdbe29dd72cad9aed48a479c8b1957c749c6d1e25d  phi4_wolff/src/hmc_wolff_tau_phi4_fullsave.py
ebd65b5595961cfb8cfdb834044682a342f5d006d292598767948e6908e5c50a  phi4_nf/phi4_nf/kernels.py
d40ae762d2de847a31aa42d28de70f6b167abd66bff306b69f4fad061f73e36e  phi4_nf/phi4_nf/diagnostics.py
00c9792cbe15d7d2e1e73dbfd5a655a02cb72cd9e0bd70bd430f430314fa2078  phi4_nf/phi4_nf/physics.py
```
