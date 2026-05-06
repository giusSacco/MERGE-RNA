"""Regenerate cached CSVs for nb1_structured_rnas.

Reads raw .rc files (via Experiment.paths_to_redmond_ivt_data_txt) and the
committed fit parameters under fits_paper/structured_rnas/physical_params_only_crossval/ and
fits_paper/redmond_mfe/, recomputes everything, writes 4 CSVs to
data/cache/structured_rnas/.

Run from the repo root:
    python scripts/analysis/_regenerate_cache_nb1.py

Run with no args to reproduce committed cache.  Pass --help for workstation-fit
override flags used by scripts/verify_workstation.sh.

Slow (a few minutes). Only needed when the underlying fits change or
when verifying the cache against fresh computations.

Source provenance: this script is a literal extraction from
scripts/analysis/maintext_figs.py — the "FIGURE 2" cell (lines
1259-2340) and its pre-computation cells. All exploratory branches
have been collapsed to the canonical paper values.
"""
# %% Imports & path setup
import argparse
import os
import sys
import pathlib
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm
import RNA
from scipy.stats import pearsonr

REPO = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('RNA_STRUCT_HOME', str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from merge_rna import Experiment, MultiSystemsFit  # noqa: E402

# %% CLI overrides (defaults reproduce committed cache)
_ap = argparse.ArgumentParser(description=__doc__, add_help=True)
_ap.add_argument('--cache-out', default=None,
                 help='Write cache files here instead of data/cache/structured_rnas')
_ap.add_argument('--fits-paper-dir', default=None,
                 help='Root of fits_paper/ (default: <repo>/fits_paper)')
_ap.add_argument('--cv-folder', default=None,
                 help='Folder containing CV param subdirs (default: …/physical_params_only_crossval). '
                      'Verify mode: …/outputs/turner/redmond_together_phase1')
_ap.add_argument('--cv-params-template', default='red_crossval_{fold_id}/params1D.txt',
                 help='Path template relative to --cv-folder; placeholder: {fold_id}. '
                      'Verify mode: "params1D.txt" (no subfolder)')
_ap.add_argument('--cv-folds', type=int, default=10,
                 help='Number of CV combinations to use (default: 10 = all C(5,3))')
_ap.add_argument('--cv-combo', default=None,
                 help='Use a single specific CV combo, e.g. "hc16,bact_RNaseP_typeA,tetrahymena_ribozyme"')
_ap.add_argument('--mfe-template', default=None,
                 help='Path template for Turner per-system params; placeholders: {system}, {run_i}. '
                      'Default: …/mergerna_turner/{system}/run_{run_i}/params1D.txt. '
                      'Verify mode: absolute path to shared params1D.txt (no placeholders)')
_ap.add_argument('--andr-template', default=None,
                 help='Path template for Andronescu per-system params (same placeholders). '
                      'Pass empty string "" to skip Andronescu entirely.')
_ap.add_argument('--n-runs', type=int, default=3,
                 help='Number of repeated runs per system (default: 3)')
_args, _ = _ap.parse_known_args()

# %% Config (canonical values from maintext_figs.py:55-60, 346, 642, 724, 759, 785, 812)
SYSTEMS = ['hc16', 'bact_RNaseP_typeA', 'tetrahymena_ribozyme',
           'HCV_IRES', 'V_chol_gly_riboswitch']
SYSTEMS_WITH_PDB = [s for s in SYSTEMS if s != 'bact_RNaseP_typeA']

FITS_PAPER_DIR = pathlib.Path(_args.fits_paper_dir) if _args.fits_paper_dir else REPO / 'fits_paper'
CV_FOLDER_PHYS = pathlib.Path(_args.cv_folder) if _args.cv_folder else (
    FITS_PAPER_DIR / 'structured_rnas' / 'physical_params_only_crossval')
CV_PARAMS_TEMPLATE = _args.cv_params_template
N_RUNS = _args.n_runs
SKIP_ANDR = (_args.andr_template == '')

_DEFAULT_MFE = str(FITS_PAPER_DIR / 'structured_rnas' / 'mergerna_turner' / '{system}' / 'run_{run_i}' / 'params1D.txt')
_DEFAULT_ANDR = str(FITS_PAPER_DIR / 'structured_rnas' / 'mergerna_andr' / '{system}' / 'run_{run_i}' / 'params1D.txt')
MFE_TEMPLATE  = _args.mfe_template  if _args.mfe_template  is not None else _DEFAULT_MFE
ANDR_TEMPLATE = _args.andr_template if _args.andr_template is not None else _DEFAULT_ANDR


def _resolve(template, system, run_i):
    """Resolve a path template, ignoring unknown placeholders (fixed-path case)."""
    try:
        return pathlib.Path(template.format(system=system, run_i=run_i))
    except KeyError:
        return pathlib.Path(template)


NON_ZERO_CONCS = [8, 17, 34, 57]
DEIGAN_M, DEIGAN_B = 1.8, -0.6
PP_PAIRED_THR = 0.75
PP_UNPAIRED_THR = 0.25

CACHE_DIR = pathlib.Path(_args.cache_out) if _args.cache_out else REPO / 'data' / 'cache' / 'structured_rnas'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if _args.cv_combo:
    _CV_COMBOS = [tuple(_args.cv_combo.split(','))]
else:
    _CV_COMBOS = list(combinations(SYSTEMS, 3))[:_args.cv_folds]

def _rel(p):
    try:
        return str(pathlib.Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)

# %% Key-inputs summary (printed on every run so you can verify correct sources)
print("=" * 60)
print("_regenerate_cache_nb1 — key inputs")
print(f"  CV folder   : {_rel(CV_FOLDER_PHYS)}")
print(f"  CV template : {CV_PARAMS_TEMPLATE}")
print(f"  CV combos   : {len(_CV_COMBOS)} fold(s), n_runs={N_RUNS}")
print(f"  MFE template: {_rel(MFE_TEMPLATE)}")
print(f"  Andr        : {'<skipped>' if SKIP_ANDR else _rel(ANDR_TEMPLATE)}")
print(f"  Cache out   : {_rel(CACHE_DIR)}")
print("=" * 60)


# %% Helpers (lifted verbatim from maintext_figs.py:566, 588, 647)
def dotbracket_to_matrix(dotbracket):
    """Pairing matrix from dotbracket string (- means NaN row/col)."""
    matrix = np.zeros((len(dotbracket), len(dotbracket)))
    stack = []
    for i, c in enumerate(dotbracket):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop()
            matrix[i][j] = 1
            matrix[j][i] = 1
        elif c == '.':
            pass
        elif c == '-':
            matrix[i][:] = np.nan
            matrix[:, i] = np.nan
        else:
            raise ValueError('Invalid character in dotbracket string')
    assert len(stack) == 0
    return matrix


def compute_distance(bpp1, bpp2):
    """Frobenius distance over upper triangle, NaN-safe."""
    dist = 0.0
    n = 0
    for i in range(bpp1.shape[0]):
        for j in range(i + 1, bpp1.shape[1]):
            if not np.isnan(bpp1[i, j]) and not np.isnan(bpp2[i, j]):
                dist += (bpp1[i, j] - bpp2[i, j]) ** 2
                n += 1
    return float(np.sqrt(dist / n))


def normalize_reactivities_deigan(reactivities):
    """Deigan 2009 normalisation: divide by mean of top 10% of non-outliers."""
    q1, q3 = np.percentile(reactivities, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    non_outliers = reactivities[reactivities <= threshold]
    n_top10 = max(1, int(np.ceil(0.10 * len(non_outliers))))
    top10 = np.sort(non_outliers)[::-1][:n_top10]
    norm = np.mean(top10)
    return reactivities / norm if norm > 0 else reactivities


def initialize_exps(sys1, sys2, sys3):
    """Train/val split of Redmond experiments for one CV fold."""
    train = [Experiment(p) for p in Experiment.paths_to_redmond_ivt_data_txt
             if Experiment(p).system in (sys1, sys2, sys3)]
    train = [e for e in train if e.conc_mM != 85]
    val = [Experiment(p) for p in Experiment.paths_to_redmond_ivt_data_txt
           if Experiment(p).system not in (sys1, sys2, sys3)]
    val = [e for e in val if e.conc_mM != 85]
    return train, val


# %% Panel A — cross-validation losses per system per fold
print(f"Building MultiSystemsFit for {len(_CV_COMBOS)} CV combination(s)...")
multi_exp_dict = {}
for combo in tqdm(_CV_COMBOS):
    train, val = initialize_exps(*combo)
    multi_exp_dict[combo] = MultiSystemsFit(train, validation_exps=val,
                                            skip_output_setup=True)

print("Computing per-system normalised losses...")
panel_a_rows = []
for combo in tqdm(_CV_COMBOS):
    multi = multi_exp_dict[combo]
    fold_id = '_'.join(combo)
    params_file = CV_FOLDER_PHYS / CV_PARAMS_TEMPLATE.format(fold_id=fold_id)
    multi.multisys_loss_and_grad(np.loadtxt(params_file), compute_gradient=False)

    for sys_obj in multi.systems:
        loss_total = 0.0
        is_validation = None
        for exp_fit in sys_obj.exp_fits_all:
            loss_exp = multi.losses_exp_fit[exp_fit.ID]
            normalised = loss_exp / (exp_fit.N_seq - 50) / (5 * 2)
            loss_total += normalised
            is_validation = not exp_fit.is_training
        panel_a_rows.append({
            'system': sys_obj.sys_name,
            'fold_id': fold_id,
            'is_validation': bool(is_validation),
            'loss': float(loss_total),
        })

panel_a = pd.DataFrame(panel_a_rows)
panel_a.to_csv(CACHE_DIR / 'panel_a_losses.csv', index=False)
print(f"✓ panel_a_losses.csv: {len(panel_a)} rows")


# %% Load all_exps + reference (PDB) BPPs + Vienna BPPs
print("\nLoading all experiments + PDB structures...")
all_exps = [Experiment(p) for p in Experiment.paths_to_redmond_ivt_data_txt]

bpp_pdb_dict = {}
bpp_vienna_dict = {}
bpp_pdb_full_plot_dict = {}
arcplot_meta = {}
distance_vienna_bpp_per_system = {}
distance_vienna_mfe_per_system = {}

for system in tqdm(SYSTEMS_WITH_PDB):
    exps = [e for e in all_exps if e.system == system and e.conc_mM != 85]
    exps[0].add_pdb_ss_to_df(exps[0].raw_df)
    ss_pdb = ''.join(exps[0].raw_df['pdb_ss'])
    bpp_pdb_full = dotbracket_to_matrix(ss_pdb)
    indices = exps[0].df.index
    bpp_pdb = bpp_pdb_full[indices][:, indices]
    bpp_pdb_dict[system] = bpp_pdb
    bpp_pdb_full_plot_dict[system] = np.nan_to_num(bpp_pdb_full, nan=0.0)
    arcplot_meta[system] = {
        'df_start_idx': int(indices[0]),
        'n_total': len(exps[0].seq),
        'n_full': len(exps[0].raw_df),
        'sequence': exps[0].seq,
    }

    fc = RNA.fold_compound(exps[0].seq)
    RNA.cvar.temperature = exps[0].temp_C
    ss_vienna_mfe = fc.mfe()[0]
    bpp_vienna_mfe = dotbracket_to_matrix(ss_vienna_mfe)
    fc.pf()
    bpp_vienna = np.array(fc.bpp())[1:, 1:]
    bpp_vienna_dict[system] = bpp_vienna

    distance_vienna_bpp_per_system[system] = compute_distance(bpp_pdb, bpp_vienna)
    distance_vienna_mfe_per_system[system] = compute_distance(bpp_pdb, bpp_vienna_mfe)


# %% Deigan distances + pairing probs (Turner energy params)
print("\nComputing Deigan (Turner) BPPs...")
deigan_bpp_dist_turner = {s: {} for s in SYSTEMS_WITH_PDB}
deigan_pp_turner = {s: {} for s in SYSTEMS_WITH_PDB}

for system in tqdm(SYSTEMS_WITH_PDB):
    exps = [e for e in all_exps if e.system == system and e.conc_mM != 85]
    exp_0mM = [e for e in exps if e.conc_mM == 0]
    bg = np.mean([np.array(e.df['mut_rate']) for e in exp_0mM], axis=0)

    for conc in NON_ZERO_CONCS:
        exp_conc = [e for e in exps if e.conc_mM == conc]
        if not exp_conc:
            continue
        mut = np.mean([np.array(e.df['mut_rate']) for e in exp_conc], axis=0)
        react = np.clip(mut - bg, 0, None)
        react = normalize_reactivities_deigan(react)

        fc = RNA.fold_compound(exp_conc[0].seq)
        RNA.cvar.temperature = exp_conc[0].temp_C
        fc.sc_add_SHAPE_deigan(react.tolist(), DEIGAN_M, DEIGAN_B)
        fc.pf()
        bpp = np.array(fc.bpp())[1:, 1:]
        deigan_bpp_dist_turner[system][conc] = compute_distance(bpp, bpp_pdb_dict[system])
        deigan_pp_turner[system][conc] = np.sum(bpp + bpp.T, axis=0)


# %% Deigan distances + pairing probs (Andronescu energy params)
print("\nComputing Deigan (Andronescu) BPPs...")
RNA.params_load_RNA_Andronescu2007()
deigan_bpp_dist_andr = {s: {} for s in SYSTEMS_WITH_PDB}
deigan_pp_andr = {s: {} for s in SYSTEMS_WITH_PDB}

for system in tqdm(SYSTEMS_WITH_PDB):
    exps = [e for e in all_exps if e.system == system and e.conc_mM != 85]
    exp_0mM = [e for e in exps if e.conc_mM == 0]
    bg = np.mean([np.array(e.df['mut_rate']) for e in exp_0mM], axis=0)

    for conc in NON_ZERO_CONCS:
        exp_conc = [e for e in exps if e.conc_mM == conc]
        if not exp_conc:
            continue
        mut = np.mean([np.array(e.df['mut_rate']) for e in exp_conc], axis=0)
        react = np.clip(mut - bg, 0, None)
        react = normalize_reactivities_deigan(react)

        fc = RNA.fold_compound(exp_conc[0].seq)
        RNA.cvar.temperature = exp_conc[0].temp_C
        fc.sc_add_SHAPE_deigan(react.tolist(), DEIGAN_M, DEIGAN_B)
        fc.pf()
        bpp = np.array(fc.bpp())[1:, 1:]
        deigan_bpp_dist_andr[system][conc] = compute_distance(bpp, bpp_pdb_dict[system])
        deigan_pp_andr[system][conc] = np.sum(bpp + bpp.T, axis=0)

RNA.params_load_RNA_Turner2004()


# %% MERGE-RNA (Turner) BPPs from per-system fits
print(f"\nComputing MERGE-RNA (Turner) BPPs ({N_RUNS} run(s) per system)...")
single_system_bpps = {s: {} for s in SYSTEMS_WITH_PDB}
for system in tqdm(SYSTEMS_WITH_PDB):
    for run_i in range(1, N_RUNS + 1):
        exp_zero = [Experiment(p) for p in Experiment.paths_to_redmond_ivt_data_txt
                    if Experiment(p).system == system and Experiment(p).conc_mM == 0][0]
        m = MultiSystemsFit([exp_zero], infer_1D_sc=True, skip_output_setup=True)
        ef = m.systems[0].exp_fits_all[0]
        params_file = _resolve(MFE_TEMPLATE, system, run_i)
        if not params_file.exists():
            print(f"  WARN: {params_file} missing, skipping")
            continue
        params = np.loadtxt(params_file)
        fc = RNA.fold_compound(exp_zero.seq)
        ef.apply_soft_constraints(params[m.lambdas_indices[system]], fc)
        fc.pf()
        single_system_bpps[system][run_i] = np.array(fc.bpp())[1:, 1:]


# %% MERGE-RNA (Andronescu) BPPs from per-system fits
if SKIP_ANDR:
    print("\nSkipping Andronescu BPPs (--andr-template '')")
    new_andr_bpps = {s: {} for s in SYSTEMS_WITH_PDB}
else:
    print(f"\nComputing MERGE-RNA (Andronescu) BPPs ({N_RUNS} run(s) per system)...")
    RNA.params_load_RNA_Andronescu2007()
    new_andr_bpps = {s: {} for s in SYSTEMS_WITH_PDB}
    for system in tqdm(SYSTEMS_WITH_PDB):
        for run_i in range(1, N_RUNS + 1):
            exp_zero = [Experiment(p) for p in Experiment.paths_to_redmond_ivt_data_txt
                        if Experiment(p).system == system and Experiment(p).conc_mM == 0][0]
            m = MultiSystemsFit([exp_zero], infer_1D_sc=True, skip_output_setup=True)
            ef = m.systems[0].exp_fits_all[0]
            params_file = _resolve(ANDR_TEMPLATE, system, run_i)
            if not params_file.exists():
                print(f"  WARN: {params_file} missing, skipping")
                continue
            params = np.loadtxt(params_file)
            fc = RNA.fold_compound(exp_zero.seq)
            ef.apply_soft_constraints(params[m.lambdas_indices[system]], fc)
            fc.pf()
            new_andr_bpps[system][run_i] = np.array(fc.bpp())[1:, 1:]
    RNA.params_load_RNA_Turner2004()


# %% Panel B — 6 distance methods per system, sub-bars per concentration/fold
print("\nWriting panel_b_distances.csv...")
panel_b_rows = []
for system in SYSTEMS_WITH_PDB:
    panel_b_rows.append({'system': system, 'method': 'Vienna_MFE',
                         'sub_id': 0, 'distance': distance_vienna_mfe_per_system[system]})
    panel_b_rows.append({'system': system, 'method': 'Vienna_BPP',
                         'sub_id': 0, 'distance': distance_vienna_bpp_per_system[system]})
    for j, c in enumerate(sorted(deigan_bpp_dist_turner[system].keys())):
        panel_b_rows.append({'system': system, 'method': 'Deigan_Turner',
                             'sub_id': j, 'distance': deigan_bpp_dist_turner[system][c]})
    for j, c in enumerate(sorted(deigan_bpp_dist_andr[system].keys())):
        panel_b_rows.append({'system': system, 'method': 'Deigan_Andronescu',
                             'sub_id': j, 'distance': deigan_bpp_dist_andr[system][c]})
    for run_i in range(1, N_RUNS + 1):
        if run_i in single_system_bpps[system]:
            d = compute_distance(single_system_bpps[system][run_i], bpp_pdb_dict[system])
            panel_b_rows.append({'system': system, 'method': 'MERGE_Turner',
                                 'sub_id': run_i - 1, 'distance': d})
    for run_i in range(1, N_RUNS + 1):
        if run_i in new_andr_bpps[system]:
            d = compute_distance(new_andr_bpps[system][run_i], bpp_pdb_dict[system])
            panel_b_rows.append({'system': system, 'method': 'MERGE_Andronescu',
                                 'sub_id': run_i - 1, 'distance': d})

panel_b = pd.DataFrame(panel_b_rows)
panel_b.to_csv(CACHE_DIR / 'panel_b_distances.csv', index=False)
print(f"✓ panel_b_distances.csv: {len(panel_b)} rows")


# %% Panel C — Pearson R² between mut_rate(57mM, baseline-subtracted) and pairing prob
print("\nWriting panel_c_r2.csv...")

def safe_r2(pp_values, nt_mask, mut_rates):
    x = np.asarray(pp_values)
    valid = nt_mask & np.isfinite(x) & np.isfinite(mut_rates)
    if np.sum(valid) > 2:
        r, _ = pearsonr(x[valid], mut_rates[valid])
        return r ** 2
    return np.nan


panel_c_rows = []
for system in tqdm(SYSTEMS_WITH_PDB):
    exps = [e for e in all_exps if e.system == system and e.conc_mM != 85]
    exp = exps[0]
    exp_57mM = [e for e in exps if e.conc_mM == 57]
    exp_0mM = [e for e in exps if e.conc_mM == 0]
    mut_rates = np.mean([np.array(e.df['mut_rate']) for e in exp_57mM], axis=0)
    if len(exp_0mM):
        mut_rates = mut_rates - np.mean([np.array(e.df['mut_rate']) for e in exp_0mM], axis=0)

    ref_nts = np.array(list(exp.seq))
    mut_mask = np.ones(len(ref_nts), dtype=bool)
    if len(mut_mask) > 50:
        mut_mask[:25] = False
        mut_mask[-25:] = False
    a_mask = (ref_nts == 'A') & mut_mask
    c_mask = (ref_nts == 'C') & mut_mask

    fc = RNA.fold_compound(exp.seq)
    RNA.cvar.temperature = exp.temp_C
    ss_mfe = fc.mfe()[0]
    pp_mfe = np.array([1.0 if c in '()' else 0.0 for c in ss_mfe])
    fc.pf()
    bpp_v = np.array(fc.bpp())[1:, 1:]
    pp_bpp = np.sum(bpp_v + bpp_v.T, axis=0)

    def add_row(method, sub_id, pp):
        r2_a = safe_r2(pp, a_mask, mut_rates)
        r2_c = safe_r2(pp, c_mask, mut_rates)
        vals = [v for v in (r2_a, r2_c) if np.isfinite(v)]
        r2_mean = float(np.mean(vals)) if vals else np.nan
        panel_c_rows.append({
            'system': system,
            'method': method,
            'sub_id': sub_id,
            'r2_mean': r2_mean,
            'r2_A': float(r2_a) if np.isfinite(r2_a) else np.nan,
            'r2_C': float(r2_c) if np.isfinite(r2_c) else np.nan,
        })

    add_row('Vienna_MFE', 0, pp_mfe)
    add_row('Vienna_BPP', 0, pp_bpp)
    for j, c in enumerate(sorted(deigan_pp_turner[system].keys())):
        add_row('Deigan_Turner', j, deigan_pp_turner[system][c])
    for j, c in enumerate(sorted(deigan_pp_andr[system].keys())):
        add_row('Deigan_Andronescu', j, deigan_pp_andr[system][c])
    for run_i in range(1, N_RUNS + 1):
        if run_i in single_system_bpps[system]:
            bpp = single_system_bpps[system][run_i]
            add_row('MERGE_Turner', run_i - 1, np.sum(bpp + bpp.T, axis=0))
    for run_i in range(1, N_RUNS + 1):
        if run_i in new_andr_bpps[system]:
            bpp = new_andr_bpps[system][run_i]
            add_row('MERGE_Andronescu', run_i - 1, np.sum(bpp + bpp.T, axis=0))

panel_c = pd.DataFrame(panel_c_rows)
panel_c.to_csv(CACHE_DIR / 'panel_c_r2.csv', index=False)
print(f"✓ panel_c_r2.csv: {len(panel_c)} rows")


# %% Panel D — per-position dots (A and C nucleotides only, classified by PDB pairing)
print("\nWriting panel_d_dots.csv...")

panel_d_rows = []
for system in tqdm(SYSTEMS_WITH_PDB):
    exps = [e for e in all_exps if e.system == system and e.conc_mM != 85]
    exp = exps[0]

    exp.add_pdb_ss_to_df(exp.raw_df)
    ss_pdb_full = ''.join(exp.raw_df['pdb_ss'])
    bpp_pdb_full = dotbracket_to_matrix(ss_pdb_full)

    multi = MultiSystemsFit(exps, infer_1D_sc=True, skip_output_setup=True)
    ef = multi.systems[0].exp_fits_all[0]
    params_file = _resolve(MFE_TEMPLATE, system, 1)
    params = np.loadtxt(params_file)
    fc_box = RNA.fold_compound(ef.seq)
    ef.apply_soft_constraints(params[multi.lambdas_indices[system]], fc_box)
    fc_box.pf()
    bpp_model = np.array(fc_box.bpp())[1:, 1:]
    pp_model = np.sum(bpp_model + bpp_model.T, axis=0)

    bpp_pdb_full_plot = np.nan_to_num(bpp_pdb_full, nan=0.0)
    pp_ref_full = np.sum(bpp_pdb_full_plot + bpp_pdb_full_plot.T, axis=0) / 2
    df_idx = exps[0].df.index.values
    pp_ref = pp_ref_full[df_idx]

    exp_57mM = [e for e in exps if e.conc_mM == 57][0]
    mut_rates = np.array(exp_57mM.df['mut_rate'])
    exp_0mM = [e for e in exps if e.conc_mM == 0 and e.rep_number == exp_57mM.rep_number]
    if exp_0mM:
        mut_rates = mut_rates - np.array(exp_0mM[0].df['mut_rate'])

    sequence = exps[0].seq
    ref_nts = np.array(list(sequence))
    n = len(ref_nts)
    mut_mask = np.ones(n, dtype=bool)
    mut_mask[:25] = False
    mut_mask[-25:] = False

    for nt in ('A', 'C'):
        nt_mask = (ref_nts == nt) & mut_mask
        for i in np.where(nt_mask)[0]:
            ref_pp = float(pp_ref[i])
            if ref_pp > PP_PAIRED_THR:
                pdb_paired = True
            elif ref_pp < PP_UNPAIRED_THR:
                pdb_paired = False
            else:
                continue  # intermediate reference - excluded from panel D
            panel_d_rows.append({
                'system': system,
                'position': int(i),
                'ref_nt': nt,
                'mut_rate': float(mut_rates[i]),  # 57 mM minus 0 mM
                'model_pp': float(pp_model[i]),
                'pdb_paired': bool(pdb_paired),
            })

panel_d = pd.DataFrame(panel_d_rows)
panel_d.to_csv(CACHE_DIR / 'panel_d_dots.csv', index=False)
print(f"✓ panel_d_dots.csv: {len(panel_d)} rows")


# %% Arcplot BPP matrices (one NPZ per system)
print("\nSaving arcplot BPP matrices...")
ARCPLOT_DIR = CACHE_DIR / 'arcplot_bpps'
ARCPLOT_DIR.mkdir(exist_ok=True)
for system in SYSTEMS_WITH_PDB:
    bpp_model = single_system_bpps[system].get(1)
    if bpp_model is None:
        print(f"  WARN: no MERGE-RNA BPP for {system}, skipping")
        continue
    meta = arcplot_meta[system]
    np.savez_compressed(
        ARCPLOT_DIR / f'{system}.npz',
        bpp_vienna=bpp_vienna_dict[system],
        bpp_model=bpp_model,
        bpp_pdb_full=bpp_pdb_full_plot_dict[system],
        df_start_idx=np.array(meta['df_start_idx']),
        n_total=np.array(meta['n_total']),
        n_full=np.array(meta['n_full']),
    )
    print(f"  ✓ {system}.npz")


# %% Summary
print(f"\nAll caches written to: {CACHE_DIR}")
for csv in sorted(CACHE_DIR.glob('*.csv')):
    df = pd.read_csv(csv)
    print(f"  {csv.name:30s} {len(df):5d} rows  "
          f"finite={int(df.select_dtypes(include='number').apply(np.isfinite).all().all())}")
