# Figures and tables of the paper

`scripts/` holds the scripts that make the figures and tables, `data/` the summary data they produce or read.

| output | script | input |
|---|---|---|
| Fig. 2 (observables) and Fig. 3 (ESS) | `fig_main.py` | raw runs, `cache_main.pkl` |
| Fig. 4 (autocorrelation times) | `tau_ours_imh.py`, `fig_tau_single.py` | raw runs, `tau_ours_imh.json`, `tau_single.json` |
| Fig. 5 (free energy) | `free_energy.py` | raw runs, `free_energy.json` |
| Fig. 6 (ablation, Appendix C) | `fig_ablation.py` | raw runs, `ablation.json` |
| Fig. 7 (convolutional flow, Appendix D) | `fig_main.py` | raw runs, `cache_main.pkl`, `tau_nf_imh.json` |
| Fig. S1 (-log ESS / V) | `fig_essvol.py` | raw runs, `essvol.json` |
| Fig. S2 (running autocorrelation times) | `fig_acf.py` | raw runs |
| Tables S1-S6 | `table_main.py` | `table_main.json` |

Fig. 1 is a diagram drawn in the LaTeX source. `paper_figs.py` holds shared helpers and `style.py` sets the fonts.
The scripts are run from the folder that holds the raw runs (chains and proposals, about 2.6 GB), which are not in
the repository. The summary files in `data/` contain every number shown in the paper. Figures are written to
`paper/figures` or to `$FEWTERM_FIG_OUT`.

To rerun the main computation at L = 8, 12 and 16 on a laptop, use `notebooks/phi4_2d_main_results.ipynb`.
