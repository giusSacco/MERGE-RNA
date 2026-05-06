"""Regenerate cached BPP matrices for nb2_adenine_riboswitch.

Loads the committed fit parameters under
fits_paper/adenine_riboswitch/{noade_r6,noade_r7}/ade_*/params1D.txt,
recomputes the 0-DMS-extrapolation BPP matrix for each fit using
ViennaRNA, packs everything into one NPZ for the notebook.

Run from the repo root:
    python scripts/analysis/_regenerate_cache_nb2.py

Run with no args to reproduce committed cache.  Pass --help for workstation-fit
override flags used by scripts/verify_workstation.sh.

Source provenance: BPP-from-params logic adapted from
scripts/analysis/olson_analysis.py:123-214 (OTF fallback in load_fit_data).
"""
# %% Imports & path setup
import argparse
import os
import sys
import pathlib

import numpy as np
import RNA

REPO = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('RNA_STRUCT_HOME', str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from merge_rna import ExperimentFit  # noqa: E402

# %% CLI overrides (defaults reproduce committed cache)
_ap = argparse.ArgumentParser(description=__doc__, add_help=True)
_ap.add_argument('--cache-out', default=None,
                 help='Write cache files here instead of data/cache/adenine_riboswitch')
_ap.add_argument('--fits-root', default=None,
                 help='Root directory containing replica subfolders '
                      '(default: fits_paper/adenine_riboswitch)')
_ap.add_argument('--replicas', default=None,
                 help='Comma-separated replica folder names '
                      '(default: noade_r7,noade_r6).  '
                      'First name → r7 role (full BPP + arcplots); '
                      'second name → r6 role (P1 pair only for K_D).  '
                      'Single replica: r6 values derived from the same data.')
_ap.add_argument('--params-template', default='{replica}/ade_{conc}/params1D.txt',
                 help='Path template relative to --fits-root; '
                      'placeholders: {replica}, {conc}')
_args, _ = _ap.parse_known_args()

# %% Config
ADE_CONCS = [0, 0.7, 2.1, 6.2, 18.5, 55.6, 166.7, 500]
REPLICAS = _args.replicas.split(',') if _args.replicas else ['noade_r7', 'noade_r6']
TEMP_C = 30.0

# Construct sequence (176 nt). Identical across all 16 fits; verified from
# fits/olson_phase2_linear_noade_r{6,7}/ade_*/detailed/fit_metadata.json.
SEQ = ('GGCCUUCGGGCCAAGAUCAACGCUUCAUAUAAUCCUAAUGAUAUGGUUUGGGAGUUUCUACCAAGAGCCUUAA'
       'ACUCUUGAUUAUGAAGUCUGUCGCUUUAUCCGAAAUUUUAUAAAGAGAAGACUCAUGAAUUCGAUCCGGUUC'
       'GCCGGAUCCAAAUCGGGCUUCGGUCCGGUUC')
N = len(SEQ)
assert N == 176, f"Expected 176 nt, got {N}"

# P1 representative pair labels -> 0-based indices, for compact r6 cache.
MAINTEXT_INDEX_ANCHOR = 5
MAINTEXT_START_1BASED = 14
P1_REP_PAIR_LABELS = (17, 79)

def _label_to_abs0b(label):
    return (label - MAINTEXT_INDEX_ANCHOR - 1 + MAINTEXT_START_1BASED) - 1

P1_I_ABS, P1_J_ABS = sorted(_label_to_abs0b(l) for l in P1_REP_PAIR_LABELS)

FITS_PAPER = pathlib.Path(_args.fits_root) if _args.fits_root else REPO / 'fits_paper' / 'adenine_riboswitch'
PARAMS_TEMPLATE = _args.params_template
CACHE_DIR = pathlib.Path(_args.cache_out) if _args.cache_out else REPO / 'data' / 'cache' / 'adenine_riboswitch'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / 'figures').mkdir(parents=True, exist_ok=True)

def _rel(p):
    try:
        return str(pathlib.Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)

# %% Key-inputs summary (printed on every run so you can verify correct sources)
print("=" * 60)
print("_regenerate_cache_nb2 — key inputs")
print(f"  Fits root   : {_rel(FITS_PAPER)}")
print(f"  Replicas    : {REPLICAS}")
print(f"  Template    : {PARAMS_TEMPLATE}")
print(f"  Cache out   : {_rel(CACHE_DIR)}")
print("=" * 60)


# %% BPP computation
def _compute_bpp(params_path):
    """Recompute 0-DMS-extrapolation BPP matrix from a params1D.txt file."""
    params = np.loadtxt(params_path)
    lambda_sc = params[8:] if len(params) > 8 else None

    exp_fit = ExperimentFit.__new__(ExperimentFit)
    exp_fit.seq = SEQ
    exp_fit.N_seq = N
    exp_fit.temp_C = TEMP_C
    RNA.cvar.temperature = TEMP_C
    return exp_fit.get_bpp_matrix(0.0, lambda_sc=lambda_sc)


# %% Loop over (replica, conc)
# Role assignment: REPLICAS[0] → 'r7 role' (full BPP stored for arcplots)
#                  REPLICAS[1] → 'r6 role' (P1 pair only, for K_D second rep.)
# With a single replica both roles are filled from the same data, producing
# two identical Hill curves in the notebook (visually fine for verification).
print(f"Recomputing BPPs for {len(REPLICAS)} replica(s) x {len(ADE_CONCS)} concs ...")
bpp_r7 = np.zeros((len(ADE_CONCS), N, N), dtype=np.float64)
bpp_r6_p1pair = np.zeros(len(ADE_CONCS), dtype=np.float64)

for role_idx, replica in enumerate(REPLICAS[:2]):  # use at most 2
    for k, conc in enumerate(ADE_CONCS):
        params_path = FITS_PAPER / PARAMS_TEMPLATE.format(replica=replica, conc=conc)
        if not params_path.exists():
            raise FileNotFoundError(params_path)
        bpp = _compute_bpp(params_path)
        if role_idx == 0:  # r7 role
            bpp_r7[k] = bpp
            print(f"  {replica} (r7-role) ade_{conc}: BPP({P1_I_ABS},{P1_J_ABS}) = {bpp[P1_I_ABS, P1_J_ABS]:.3f}")
        else:  # r6 role
            bpp_r6_p1pair[k] = bpp[P1_I_ABS, P1_J_ABS]
            print(f"  {replica} (r6-role) ade_{conc}: BPP({P1_I_ABS},{P1_J_ABS}) = {bpp_r6_p1pair[k]:.3f}")

if len(REPLICAS) == 1:
    bpp_r6_p1pair = bpp_r7[:, P1_I_ABS, P1_J_ABS].copy()
    print(f"  Single-replica mode: bpp_r6_p1pair derived from bpp_r7")


# %% Pack & save
out_path = CACHE_DIR / 'bpp_matrices.npz'
np.savez_compressed(
    out_path,
    concentrations=np.asarray(ADE_CONCS, dtype=np.float64),
    bpp_r7=bpp_r7,
    bpp_r6_p1pair=bpp_r6_p1pair,
    seq=SEQ,
    temp_C=TEMP_C,
)
print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
print("Done.")
