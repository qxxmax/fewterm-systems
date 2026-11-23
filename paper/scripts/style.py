"""Shared figure style for paper2: one colour per method across all figures, Dark2 for L, 9 pt, lw 1.0."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = {'ours': '#1f77b4', 'nf': '#d62728', 'wolff': 'k', 'hmc': '#ff7f0e', 'raw': '#7f7f7f'}
LCOL = dict(zip((8, 12, 16, 20, 24, 32), ('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#a6761d')))
W1, W2 = 3.4, 7.0           # single / double column width, inch
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'lines.linewidth': 1.0, 'lines.markersize': 3.5, 'axes.linewidth': .7,
                     'errorbar.capsize': 0, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02, 'pdf.fonttype': 42,
                     'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix'})


def panel(ax, s, x=.03, y=.96):
    ax.text(x, y, f'({s})', transform=ax.transAxes, va='top', ha='left', fontsize=8)
