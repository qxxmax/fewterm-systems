"""Appendix tables from table_main.json (written by fig_main.py): per L, rows kappa; columns ESS and
<|M|>, chi/V, U4 for HMC-Wolff / ours IS / ours IMH, value(error) notation.
    python scripts/table_main.py  -> paper2/sections_zh/appendix_tables.tex"""
import json, math, os

T = json.load(open('table_main.json'))
OUT = os.path.join(os.environ.get('FEWTERM_FIG_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')), 'tables_zh.tex')


def ve(v, e):
    if not e or not math.isfinite(e) or e <= 0:
        return f'{v:.3f}'
    p = max(0, 1 - int(math.floor(math.log10(e))))          # two significant digits of the error
    p = min(p, 7)
    if round(e * 10 ** p) >= 100:                            # e.g. 0.000996 -> (10), not (100)
        p -= 1
    return f'{v:.{p}f}({round(e * 10 ** p):d})'


rows = {}
for key, r in T.items():
    L = int(key.split('_')[0][1:]); k = float(key.split('_k')[1])
    rows.setdefault(L, []).append((k, r))
SCALING = r'''图~\ref{fig:scaling} 给出正文图~\ref{fig:akappa} 所用的 $\mathrm{Var}(\log w)$ 与幂律拟合。

\begin{figure}[!htb]
\centering
\includegraphics[width=\columnwidth]{fig_scaling.pdf}
\caption{若干 $\kappa$ 处 $\mathrm{Var}(\log w)$ 随 $L$ 的变化，以及给出图~\ref{fig:akappa} 的幂律拟合；图例中为拟合的指数 $a$，黑色为 $\kappa_c$。}
\label{fig:scaling}
\end{figure}
'''
s = [r'\section{数值结果}', r'\label{app:tables}', '', SCALING,
     r'表~\ref{tab:L8}--\ref{tab:L32} 给出图~\ref{fig:main} 中各点的数值。HMC 为一条 $2\times10^6$ 轨迹的 HMC 链（$\Gamma$ 方法误差）。IS 与 IMH 为少参数流的重加权与 IMH 估计，括号内为末位的一个标准差。$\kappa\in\{0.26,0.2705,0.28,0.30\}$ 有三个种子，观测量按误差平方倒数加权平均，ESS 取平均值，其余点为单个种子。', '']
for L in sorted(rows):
    s += [r'\begin{table*}[p]', rf'\caption{{$L={L}$ 的观测量。}}', rf'\label{{tab:L{L}}}', r'\scriptsize',
          r'\begin{ruledtabular}', r'\begin{tabular}{lr' + 'rrr' * 3 + '}',
          r'& & \multicolumn{3}{c}{$\langle|M|\rangle$} & \multicolumn{3}{c}{$\chi/V$} & \multicolumn{3}{c}{$U_4$}\\',
          r'$\kappa$ & ESS & HMC & IS & IMH & HMC & IS & IMH & HMC & IS & IMH\\', r'\hline']
    for k, r in sorted(rows[L], key=lambda x: x[0]):
        cells = [f'{k:.4f}'.rstrip('0').ljust(4, '0'), f'{r["ess"]:.3f}' if r['ess'] >= .01 else f'{r["ess"]:.4f}']
        for q in range(3):
            for m in ('hmc', 'IS', 'IMH'):
                cells.append(ve(r[m][0][q], r[m][1][q]))
        s.append(' & '.join(cells) + r'\\')
    s += [r'\end{tabular}', r'\end{ruledtabular}', r'\end{table*}', '']
open(OUT, 'w').write('\n'.join(s))
import sys
if '--en' in sys.argv:
    e = '\n'.join(s).replace(SCALING, '').replace(r'\section{数值结果}', '').replace(r'\label{app:tables}', '')
    e = e.replace(e[e.index('表~'):e.index('\n', e.index('表~'))],
                  r'Tables~\ref{tab:L8}--\ref{tab:L32} give the numerical values of the points in Fig.~\ref{fig:main} for all six volumes. HMC is one chain of $2\times10^6$ trajectories (errors from the $\Gamma$ method). IS and IMH are the reweighting and IMH estimates of the few-parameter flow. Numbers in parentheses are one standard deviation on the last digits. At $\kappa\in\{0.26,0.2705,0.28,0.30\}$ three seeds are combined, the observables by the inverse-variance weighted mean and the ESS by the plain mean. The other points have one seed.')
    import re
    e = re.sub(r'\\caption\{\$L=(\d+)\$ 的观测量。\}', r'\\caption{Observables at $L=\1$.}', e)
    e = e.replace(r'\begin{table*}[p]', r'\begin{table}[!htb]').replace(r'\end{table*}', r'\end{table}').replace(r'\scriptsize', r'\footnotesize')
    open(os.path.join(os.path.dirname(OUT), 'supp_tables.tex'), 'w').write(e.strip() + '\n')
print(OUT, sorted(rows))
