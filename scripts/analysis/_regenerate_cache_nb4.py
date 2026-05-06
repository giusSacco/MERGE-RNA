"""Regenerate cached CSVs/NPZs for nb4_designed_sequence.

Reads raw data and fitted parameters from:
  fits_paper/designed_sequence/{synthetic,experimental}/
  data/designed_sequence/experimental/
Writes synthetic mutation-rate CSVs to data/designed_sequence/synthetic/
and all derived cache to data/cache/designed_sequence/.

Run from the repo root:
    python scripts/analysis/_regenerate_cache_nb4.py

Run with no args to reproduce committed cache.  Pass --help for workstation-fit
override flags used by scripts/verify_workstation.sh.

Source provenance: extracted from scripts/analysis/maintext_figs.py,
"Fig 3" (lines 3622–3777) and "Fig 3.1" (lines 3780–3998). All exploratory
branches and the Deigan panel have been stripped; only the canonical paper
version is reproduced.
"""
# %% Imports & path setup
import argparse
import os
import sys
import pathlib
import shutil

import numpy as np
import pandas as pd
import RNA

REPO = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('RNA_STRUCT_HOME', str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from merge_rna import Experiment, MultiSystemsFit, create_exp_synthetic_comb  # noqa: E402

SYNTH_FITS_DIR = REPO / 'fits_paper' / 'designed_sequence' / 'synthetic'
EXP_FITS_DIR   = REPO / 'fits_paper' / 'designed_sequence' / 'experimental'
DATA_EXP_DIR   = REPO / 'data' / 'designed_sequence' / 'experimental'
DATA_SYNTH_DIR = REPO / 'data' / 'designed_sequence' / 'synthetic'
DATA_SYNTH_DIR.mkdir(parents=True, exist_ok=True)

# %% CLI overrides (defaults reproduce committed cache)
_ap = argparse.ArgumentParser(description=__doc__, add_help=True)
_ap.add_argument('--cache-out', default=None,
                 help='Write cache files here instead of data/cache/designed_sequence')
_ap.add_argument('--synth-fits-dir', default=None,
                 help='Directory containing synthetic pop params (default: fits_paper/…/synthetic)')
_ap.add_argument('--synth-glob', default='pop*.txt',
                 help='Glob pattern for synthetic params inside --synth-fits-dir '
                      '(default: "pop*.txt"; verify mode: "synthetic_bistable_*/params1D.txt")')
_ap.add_argument('--exp-andr-params', default=None,
                 help='Path to strand-displacement Andronescu params1D.txt (workstation re-fit)')
_args, _ = _ap.parse_known_args()

CACHE_DIR = pathlib.Path(_args.cache_out) if _args.cache_out else REPO / 'data' / 'cache' / 'designed_sequence'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %% Config (canonical values from maintext_figs.py)
COMPL_SEQ = "CCCCACUUCAACUCACAACUAUUCC"  # used to locate both helices

REF_PARAMS_FILE  = SYNTH_FITS_DIR / 'reference_physical_params.txt'
SYNTH_PARAMS_DIR = pathlib.Path(_args.synth_fits_dir) if _args.synth_fits_dir else SYNTH_FITS_DIR
SYNTH_GLOB       = _args.synth_glob
PARAMS_TURNER    = EXP_FITS_DIR / 'strand_disp_turner.txt'
PARAMS_ANDRONESCU = (pathlib.Path(_args.exp_andr_params) if _args.exp_andr_params
                     else EXP_FITS_DIR / 'strand_disp_andronescu.txt')
SD_DATA_PATHS    = sorted(DATA_EXP_DIR.glob('*/*_sorted.txt'))

SCHEMATIC_SRC = SYNTH_FITS_DIR / 'schematic_result.png'


def _rel(p):
    try:
        return str(pathlib.Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# %% Key-inputs summary (printed on every run so you can verify correct sources)
print("=" * 60)
print("_regenerate_cache_nb4 — key inputs")
print(f"  Synth fits  : {_rel(SYNTH_PARAMS_DIR)}")
print(f"  Synth glob  : {SYNTH_GLOB}")
print(f"  Exp Turner  : {_rel(PARAMS_TURNER)}")
print(f"  Exp Andr    : {_rel(PARAMS_ANDRONESCU)}")
print(f"  Cache out   : {_rel(CACHE_DIR)}")
print("=" * 60)


# %% Helpers

def _extract_pop_int(params_file):
    """Extract integer population % (0-100) from params file path.

    Handles two naming conventions:
      flat file:  synthetic/pop020.txt         → 20
      subdir:     synthetic_bistable_20/params1D.txt → 20
    """
    if params_file.stem == 'params1D':
        return int(params_file.parent.name.split('_')[-1])
    return int(params_file.stem.replace('pop', ''))


def _get_helix_indices(seq, compl_seq):
    """Return (indices1, indices2) for the two occurrences of compl_seq in seq."""
    start1 = seq.find(compl_seq)
    if start1 == -1:
        raise ValueError(f"Could not find first occurrence of helix motif in sequence")
    end1 = start1 + len(compl_seq) - 1
    start2 = seq.find(compl_seq, start1 + 1)
    if start2 == -1:
        raise ValueError(f"Could not find second occurrence of helix motif in sequence")
    end2 = start2 + len(compl_seq) - 1
    return list(range(start1, end1 + 1)), list(range(start2, end2 + 1))


# %% Build params_dict_synth (shared reference physical parameters)
print("Building params_dict_synth from reference physical params...")
params_1D_ref = np.loadtxt(REF_PARAMS_FILE)
exp_ref_0mM = None
for path_ in Experiment.paths_to_redmond_ivt_data_txt:
    exp_candidate = Experiment(path_)
    if exp_candidate.conc_mM == 0:
        exp_ref_0mM = exp_candidate
        break
if exp_ref_0mM is None:
    raise ValueError("Could not find a 0 mM Redmond experiment")
multi_ref = MultiSystemsFit([exp_ref_0mM], validation_exps=None, infer_1D_sc=False, skip_output_setup=True)
params_dict_synth = multi_ref.pack_params(params_1D_ref, multi_ref.systems[0])
print("  done.")


# %% Synthetic: pp curves + inset table + arcplot extremes
print("\nProcessing synthetic bistable data...")
synth_params_files = sorted(SYNTH_PARAMS_DIR.glob(SYNTH_GLOB))

pp_curve_rows = []
inset_rows = []
bpp_extremes = {}
seq_synth = None
indices_helix1 = None
indices_helix2 = None

for params_file in synth_params_files:
    pop_int = _extract_pop_int(params_file)
    pop1 = pop_int / 100
    params_1D = np.loadtxt(params_file)

    exp = create_exp_synthetic_comb(pop1, params_dict=params_dict_synth)

    # Save synthetic mutation rates (what we fit against)
    (exp.df.reset_index()[['pos', 'ref_nt', 'mut_count', 'total_count']]
        .to_csv(DATA_SYNTH_DIR / f'pop{pop_int:03d}.csv', index=False))

    multi_exp = MultiSystemsFit([exp], infer_1D_sc=True, skip_output_setup=True)
    exp_fit = multi_exp.systems[0].exp_fits_all[0]
    lambda_sc = params_1D[multi_exp.lambdas_indices[exp.system_name]]
    pp_model = exp_fit.get_ps(0, lambda_sc)

    if seq_synth is None:
        seq_synth = exp.seq
        indices_helix1, indices_helix2 = _get_helix_indices(seq_synth, COMPL_SEQ)

    for pos, pp in enumerate(pp_model):
        pp_curve_rows.append({'pop1': pop1, 'pos': pos, 'pp': float(pp)})

    inset_rows.append({
        'pop1': pop1,
        'median_pp_helix1': float(np.median(pp_model[indices_helix1])),
        'median_pp_helix2': float(np.median(pp_model[indices_helix2])),
    })
    print(f"  pop1={pop1:.2f}: pp computed, helix1 median={np.median(pp_model[indices_helix1]):.3f}")

    if pop1 in (0.0, 1.0):
        fc = RNA.fold_compound(exp.seq)
        RNA.cvar.temperature = exp_fit.temp_C
        exp_fit.apply_soft_constraints(lambda_sc, fc)
        fc.pf()
        bpp_extremes[pop1] = np.array(fc.bpp())[1:, 1:]
        print(f"    BPP matrix computed for pop1={pop1:.1f}")

pd.DataFrame(pp_curve_rows).to_csv(CACHE_DIR / 'synthetic_pp_curves.csv', index=False)
pd.DataFrame(inset_rows).to_csv(CACHE_DIR / 'synthetic_inset_table.csv', index=False)
print(f"✓ synthetic_pp_curves.csv: {len(pp_curve_rows)} rows")
print(f"✓ synthetic_inset_table.csv: {len(inset_rows)} rows")

np.savez_compressed(
    CACHE_DIR / 'synthetic_arcplot_extremes.npz',
    bpp_pop0=bpp_extremes[0.0],
    bpp_pop1=bpp_extremes[1.0],
    seq=np.array(seq_synth),
    indices_helix1=np.array(indices_helix1),
    indices_helix2=np.array(indices_helix2),
)
print("✓ synthetic_arcplot_extremes.npz")


# %% Strand displacement: mutation rates
print("\nProcessing strand-displacement mutation rates...")
mut_rate_rows = []
exps_sd = [Experiment(str(p)) for p in SD_DATA_PATHS]
for exp in exps_sd:
    df = exp.df[exp.df['ref_nt'].isin(['A', 'C', 'G', 'U'])].copy()
    for pos, row in df.iterrows():
        mut_rate_rows.append({
            'rep': exp.rep_number,
            'conc_mM': exp.conc_mM,
            'pos': int(pos),
            'ref_nt': row['ref_nt'],
            'mut_rate': float(row['mut_rate']),
        })
pd.DataFrame(mut_rate_rows).to_csv(CACHE_DIR / 'strand_disp_mut_rates.csv', index=False)
print(f"✓ strand_disp_mut_rates.csv: {len(mut_rate_rows)} rows")


# %% Strand displacement: pairing probability curves
print("\nComputing strand-displacement pairing probability curves...")
multi_fit = MultiSystemsFit(
    experiments=exps_sd,
    infer_1D_sc=True,
    use_interpolated_ps=False,
    skip_output_setup=True,
)
exp_fit_sd = multi_fit.systems[0].exp_fits_all[0]
sys_name_sd = exps_sd[0].system_name

# Turner baseline + MERGE (Turner params are the default)
params_turner = np.loadtxt(PARAMS_TURNER)
ps_baseline_turner = exp_fit_sd.get_ps(0, np.zeros(len(exps_sd[0].seq)))
ps_merge_turner = exp_fit_sd.get_ps(0, params_turner[multi_fit.lambdas_indices[sys_name_sd]])

# Andronescu baseline + MERGE
RNA.params_load_RNA_Andronescu2007()
params_andronescu = np.loadtxt(PARAMS_ANDRONESCU)
ps_baseline_andronescu = exp_fit_sd.get_ps(0, np.zeros(len(exps_sd[0].seq)))
ps_merge_andronescu = exp_fit_sd.get_ps(0, params_andronescu[multi_fit.lambdas_indices[sys_name_sd]])
RNA.params_load_RNA_Turner2004()

pp_idx = exp_fit_sd.df.index.values
pd.DataFrame({
    'pos': pp_idx,
    'baseline_turner': ps_baseline_turner,
    'baseline_andronescu': ps_baseline_andronescu,
    'merge_turner': ps_merge_turner,
    'merge_andronescu': ps_merge_andronescu,
}).to_csv(CACHE_DIR / 'strand_disp_pp_curves.csv', index=False)
print(f"✓ strand_disp_pp_curves.csv: {len(pp_idx)} rows")


# %% Copy schematic image
if SCHEMATIC_SRC.exists():
    shutil.copy2(SCHEMATIC_SRC, CACHE_DIR / 'schematic_result.png')
    print(f"✓ schematic_result.png (copied from {SCHEMATIC_SRC.relative_to(REPO)})")
else:
    print(f"  WARN: schematic not found at {SCHEMATIC_SRC}")

print(f"\nAll caches written to: {CACHE_DIR}")
