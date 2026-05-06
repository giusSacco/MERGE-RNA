"""Regenerate cached NPZs for nb3_cspA.

Reads fitted parameters from fits_paper/cspA/{fix_1..fix_6} and the raw
data from data/cspA/10C/SRR6123774/ and data/cspA/37C/SRR6123773/,
recomputes BPP matrices with ViennaRNA, and writes one .npz per fit to
data/cache/cspA/.

Run from the repo root:
    python scripts/analysis/_regenerate_cache_nb3.py

Run with no args to reproduce committed cache.  Pass --help for workstation-fit
override flags used by scripts/verify_workstation.sh.

Source provenance: helpers extracted from scripts/analysis/maintext_figs.py
lines 4384–4411; combination logic from merge_rna/experiment.py
create_combined_cspA_df (lines 653–701). Synthetic/noise branches stripped.
"""
# %% Imports & path setup
import argparse
import os
import sys
import pathlib

import numpy as np
import pandas as pd
import RNA

REPO = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('RNA_STRUCT_HOME', str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from merge_rna import Experiment, MultiSystemsFit  # noqa: E402

# %% CLI overrides (defaults reproduce committed cache)
_ap = argparse.ArgumentParser(description=__doc__, add_help=True)
_ap.add_argument('--cache-out', default=None,
                 help='Write cache files here instead of data/cache/cspA')
_ap.add_argument('--fits-root', default=None,
                 help='Root directory containing fit subfolders (default: fits_paper/cspA)')
_ap.add_argument('--subfolders', default=None,
                 help='Comma-separated subfolder names (default: fix_1,…,fix_6)')
_ap.add_argument('--params-10-template', default='{subfolder}/cspA_comb_100_with_lambdas/params1D.txt',
                 help='Path template for 100%% 10°C params, relative to --fits-root; '
                      'placeholder: {subfolder}')
_ap.add_argument('--params-37-template', default='{subfolder}/cspA_comb_0_with_lambdas/params1D.txt',
                 help='Path template for 0%% (37°C) params, relative to --fits-root; '
                      'placeholder: {subfolder}')
_args, _ = _ap.parse_known_args()

CACHE_DIR  = pathlib.Path(_args.cache_out) if _args.cache_out else REPO / 'data' / 'cache' / 'cspA'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FITS_ROOT  = pathlib.Path(_args.fits_root) if _args.fits_root else REPO / 'fits_paper' / 'cspA'
SUBFOLDERS = _args.subfolders.split(',') if _args.subfolders else [f'fix_{i}' for i in range(1, 7)]
PARAMS_10_TEMPLATE = _args.params_10_template
PARAMS_37_TEMPLATE = _args.params_37_template

def _rel(p):
    try:
        return str(pathlib.Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)

# %% Key-inputs summary (printed on every run so you can verify correct sources)
print("=" * 60)
print("_regenerate_cache_nb3 — key inputs")
print(f"  Fits root   : {_rel(FITS_ROOT)}")
print(f"  Subfolders  : {SUBFOLDERS}")
print(f"  10°C template: {PARAMS_10_TEMPLATE}")
print(f"  37°C template: {PARAMS_37_TEMPLATE}")
print(f"  Cache out   : {_rel(CACHE_DIR)}")
print("=" * 60)

# %% Reference structures
_ref_path = REPO / 'data' / 'cspA' / 'reference_structures.txt'
_refs = dict(line.split('=', 1) for line in _ref_path.read_text().splitlines() if line)
REF37SS = _refs['ref37ss']
REF10SS = _refs['ref10ss']

# %% Helpers (verbatim from maintext_figs.py:4384-4411)

def get_bpp(exp, params_1D):
    multi_exp = MultiSystemsFit([exp], validation_exps=None, infer_1D_sc=True, skip_output_setup=True)
    params_dict = multi_exp.pack_params(params_1D, multi_exp.systems[0])
    exp_fit = multi_exp.systems[0].exp_fits_all[0]
    RNA.cvar.temperature = exp.temp_C
    fc = RNA.fold_compound(exp_fit.seq)
    exp_fit.apply_soft_constraints(params_dict['lambda_sc'], fc)
    fc.pf()
    return np.array(fc.bpp())[1:, 1:]


def get_bpp_from_lambda(exp, lambda_sc):
    """Compute BPP with an explicit lambda_sc vector (not extracted from a full params1D)."""
    multi_exp = MultiSystemsFit([exp], validation_exps=None, infer_1D_sc=True, skip_output_setup=True)
    exp_fit = multi_exp.systems[0].exp_fits_all[0]
    RNA.cvar.temperature = exp.temp_C
    fc = RNA.fold_compound(exp_fit.seq)
    exp_fit.apply_soft_constraints(lambda_sc, fc)
    fc.pf()
    return np.array(fc.bpp())[1:, 1:]


def dotbracket_to_bpp(dotbracket):
    n = len(dotbracket)
    bpp_matrix = np.zeros((n, n))
    stack = []
    for i, char in enumerate(dotbracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                bpp_matrix[i, j] = 1
                bpp_matrix[j, i] = 1
    return bpp_matrix


# %% Build combined experiments directly from committed data
# Replicates create_combined_cspA_df(pop_10, normalise_mut_rate=True)
# using only SRR6123774 (10C) and SRR6123773 (37C), no synthetic data.

exp_10_raw = Experiment(str(REPO / 'data' / 'cspA' / '10C' / 'SRR6123774' / 'info.txt'))
exp_37_raw = Experiment(str(REPO / 'data' / 'cspA' / '37C' / 'SRR6123773' / 'info.txt'))


def _make_combined_exp(pop_10, exp_10_raw, exp_37_raw):
    pop_37 = 1.0 - pop_10
    new_df = pd.DataFrame()
    new_df['total_count'] = (exp_10_raw.df['total_count'] + exp_37_raw.df['total_count']) // 2
    new_df['mut_count'] = (
        exp_10_raw.df['mut_count'] * (new_df['total_count'] / exp_10_raw.df['total_count']) * pop_10
        + exp_37_raw.df['mut_count'] * (new_df['total_count'] / exp_37_raw.df['total_count']) * pop_37
    )
    new_df['mut_count'] = np.round(new_df['mut_count']).astype(int)
    new_df['mut_rate'] = new_df['mut_count'] / new_df['total_count']
    new_df['ref_nt'] = exp_10_raw.df['ref_nt']
    corrective_factor = 0.012180 / new_df['mut_rate'].mean()
    new_df['mut_count'] = np.round(new_df['mut_count'] * corrective_factor)
    new_df['mut_rate'] = new_df['mut_count'] / new_df['total_count']
    exp = Experiment(
        seq=''.join(new_df['ref_nt'].values).replace('T', 'U'),
        temp_C=(10 + 37) // 2,
        reagent='DMS combined in vitro',
        system_name=f'cspA_combined_{pop_10 * 100:.0f}%',
    )
    exp.df = new_df
    return exp


exp_10 = _make_combined_exp(pop_10=1.0, exp_10_raw=exp_10_raw, exp_37_raw=exp_37_raw)
exp_37 = _make_combined_exp(pop_10=0.0, exp_10_raw=exp_10_raw, exp_37_raw=exp_37_raw)

# %% Reference BPP matrices (same for every fit)
bpp_ref_10 = dotbracket_to_bpp(REF10SS)
bpp_ref_37 = dotbracket_to_bpp(REF37SS)
seq = exp_37.seq

# %% Compute and save one .npz per fit
for sf in SUBFOLDERS:
    params_10_path = FITS_ROOT / PARAMS_10_TEMPLATE.format(subfolder=sf)
    params_37_path = FITS_ROOT / PARAMS_37_TEMPLATE.format(subfolder=sf)

    if not params_10_path.exists() or not params_37_path.exists():
        print(f'SKIP {sf}: params not found '
              f'(10°C: {params_10_path.exists()}, 37°C: {params_37_path.exists()})')
        continue

    print(f'Computing {sf}...')
    if params_10_path == params_37_path:
        # Joint fit: params1D contains lambda_sc for both systems, but the position
        # ordering depends on Python set() hashing (non-deterministic across processes).
        # Load per-system lambda files from detailed/ instead — they are always
        # correctly named by system_name regardless of set ordering.
        detailed_dir = params_10_path.parent / 'detailed'
        lam_10_path = detailed_dir / f'{exp_10.system_name}_lambda_sc.txt'
        lam_37_path = detailed_dir / f'{exp_37.system_name}_lambda_sc.txt'
        bpp_model_10 = get_bpp_from_lambda(exp_10, np.loadtxt(lam_10_path, comments='#'))
        bpp_model_37 = get_bpp_from_lambda(exp_37, np.loadtxt(lam_37_path, comments='#'))
    else:
        bpp_model_10 = get_bpp(exp_10, np.loadtxt(params_10_path))
        bpp_model_37 = get_bpp(exp_37, np.loadtxt(params_37_path))

    out_path = CACHE_DIR / f'{sf}.npz'
    np.savez_compressed(
        out_path,
        bpp_model_10=bpp_model_10,
        bpp_model_37=bpp_model_37,
        bpp_ref_10=bpp_ref_10,
        bpp_ref_37=bpp_ref_37,
        seq=seq,
    )
    print(f'  saved: {out_path}')

npz_written = len(list(CACHE_DIR.glob('*.npz')))
print(f'\nDone. {npz_written} NPZ(s) written to {CACHE_DIR}')
