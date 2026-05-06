"""nb3_cspA — reproduces the cspA 37°C / 10°C two-panel arcplot figure.

Loads cached NPZs from data/cache/cspA/ and produces one PDF per fit
(fix_1 … fix_6): each file contains a 1×2 arcplot with MERGE-RNA BPP
arcs above the baseline and Zhang et al. reference arcs below.

Run from the repo root:
    python notebooks/nb3_cspA.py

Outputs written to: data/cache/cspA/figures/

To regenerate the cached NPZs (when fits change):
    python notebooks/_regenerate_cache_nb3.py
"""
# %% Imports & paths
import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.lines import Line2D

try:
    REPO = pathlib.Path(__file__).resolve().parents[2]
except NameError:
    _p = pathlib.Path.cwd().resolve()
    REPO = next(p for p in [_p, *_p.parents] if (p / 'merge_rna').is_dir())

def _parse_args():
    p = argparse.ArgumentParser(description='Plot nb3 cspA figures')
    p.add_argument('--cache-dir', default=None,
                   help='Override cache directory (default: data/cache/cspA)')
    p.add_argument('--fig-out', default=None,
                   help='Override figures output directory (default: <cache>/figures)')
    args, _ = p.parse_known_args()
    return args

_ARGS = _parse_args()

def _relpath(path):
    try:
        return str(pathlib.Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)

CACHE = pathlib.Path(_ARGS.cache_dir) if _ARGS.cache_dir else REPO / 'data' / 'cache' / 'cspA'
FIG_OUT = pathlib.Path(_ARGS.fig_out) if _ARGS.fig_out else CACHE / 'figures'
FIG_OUT.mkdir(parents=True, exist_ok=True)


def _show_or_note(path):
    try:
        get_ipython()
        from IPython.display import display, Image
        if pathlib.Path(path).suffix.lower() in ('.png', '.jpg', '.jpeg'):
            display(Image(str(path)))
        else:
            print(f"  saved -> {_relpath(path)}")
    except NameError:
        print(f"  saved -> {_relpath(path)}")


# %% Config
SUBFOLDERS = sorted(p.stem for p in CACHE.glob('*.npz'))
THRESHOLD = 0.05
MAX_LW = 0.7
ARC_H_RATIO = 0.8
CONDITIONS = [
    dict(key='10', name='10°C', color='black',  sublabel='a'),
    dict(key='37', name='37°C', color='red',    sublabel='b'),
]

# %% Plotting helper

def _render_arcplot(ax, bpp_model, bpp_ref, n, color, name, sublabel):
    label_kw = dict(ha='left', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.7, edgecolor='none'))
    ax.text(0.02, 0.96, sublabel, transform=ax.transAxes, **label_kw)

    ax.plot([-1, n], [0, 0], 'k-', lw=1.0)

    for i in range(n):
        for j in range(i + 1, n):
            prob = bpp_model[i, j]
            if prob > THRESHOLD:
                cx = (i + j) / 2.0
                w = j - i
                ax.add_patch(Arc((cx, 0), w, w * ARC_H_RATIO,
                                 theta1=0, theta2=180,
                                 edgecolor=color, lw=MAX_LW * prob, fill=False))

    for i in range(n):
        for j in range(i + 1, n):
            prob = bpp_ref[i, j]
            if prob > THRESHOLD:
                cx = (i + j) / 2.0
                w = j - i
                ax.add_patch(Arc((cx, 0), w, w * ARC_H_RATIO,
                                 theta1=180, theta2=360,
                                 edgecolor='royalblue', lw=MAX_LW * prob, fill=False))

    ax.set_xlim(-1, n + 1)
    ax.set_ylim(-n / 2, n / 2)
    ax.set_yticks([])
    ax.set_xlabel('Position (nt)', fontsize=10)
    ax.set_title(name)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend(handles=[
        Line2D([0], [0], color=color,       lw=2, label='MERGE-RNA'),
        Line2D([0], [0], color='royalblue', lw=2, label='Prediction from Zhang et al.'),
    ], loc='best', fontsize=10)


# %% Plot loop

for sf in SUBFOLDERS:
    npz_path = CACHE / f'{sf}.npz'
    if not npz_path.exists():
        print(f'SKIP {sf}: {_relpath(npz_path)} not found')
        continue

    print(f'Plotting {sf}...')
    d = np.load(npz_path, allow_pickle=True)
    n = len(str(d['seq']))

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.8), dpi=300)

    for ax, cond in zip(axs, CONDITIONS):
        _render_arcplot(
            ax,
            bpp_model=d[f'bpp_model_{cond["key"]}'],
            bpp_ref=d[f'bpp_ref_{cond["key"]}'],
            n=n,
            color=cond['color'],
            name=cond['name'],
            sublabel=cond['sublabel'],
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIG_OUT / f'cspA_{sf}.pdf'
    plt.savefig(out_path, dpi=450, bbox_inches='tight')
    plt.close('all')
    _show_or_note(out_path)
    print(f'  saved: {_relpath(out_path)}')

print(f'\nDone. Figures written to {_relpath(FIG_OUT)}')
