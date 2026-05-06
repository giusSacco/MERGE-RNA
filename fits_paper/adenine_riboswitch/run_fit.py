#!/usr/bin/env python3
"""
Paper fit: Olson et al. (2022) adenine riboswitch (DANCE-MaP DMS).

Two-phase fit of the adenine riboswitch (175 nt amplicon) across a
titration of adenine concentrations from 0 to 500 µM.

Phase 1: fit shared physical parameters using the ethanol control +
         ade=0 µM DMS experiment (or all concentrations with --joint-phase1).
Phase 2: fix physical parameters and fit per-concentration lambda_sc
         (run in parallel, one worker per concentration).

Data: .rc files at $RNA_STRUCT_HOME/data/adenine_riboswitch/{SRR}/
      (processed from BioProject PRJNA756782 / SRP333600).

Usage:
    cd $RNA_STRUCT_HOME
    # Both phases (default):
    python fits_paper/adenine_riboswitch/run_fit.py --phase both

    # Phase 1 only:
    python fits_paper/adenine_riboswitch/run_fit.py --phase 1

    # Phase 2 only (requires phase 1 output):
    python fits_paper/adenine_riboswitch/run_fit.py --phase 2 \\
        --phase1-params fits_paper/adenine_riboswitch/phase1/add_riboswitch_WT/params1D.txt
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

from tqdm import tqdm

if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

import RNA
from merge_rna import Experiment, MultiSystemsFit

# =============================================================================
# Sample metadata (Olson et al. 2022, rep 1 WT adenine titration)
# =============================================================================
SAMPLES_DMS_REP1 = {
    'SRR15560843': ('WT adenine 0 µM rep 1',     0),
    'SRR15560844': ('WT adenine 0.7 µM rep 1',   0.7),
    'SRR15560845': ('WT adenine 2.1 µM rep 1',   2.1),
    'SRR15560846': ('WT adenine 6.2 µM rep 1',   6.2),
    'SRR15560847': ('WT adenine 18.5 µM rep 1',  18.5),
    'SRR15560848': ('WT adenine 55.6 µM rep 1',  55.6),
    'SRR15560849': ('WT adenine 166.7 µM rep 1', 166.7),
    'SRR15560850': ('WT adenine 500 µM rep 1',   500),
}
CONTROL_REP1_SRR = 'SRR15560865'

_rna_home = os.environ.get('RNA_STRUCT_HOME', '.')
_data_root = os.environ.get('MERGE_RNA_DATA', _rna_home)
RESULTS_DIR = os.path.join(_data_root, 'data', 'adenine_riboswitch')

# =============================================================================
# Masking
# Aligned to REF_SEQ (subsequence of the 175 nt amplicon).
# Convention: '1' = exclude from fit loss, '0' = include.
# (Excludes positions with severely low coverage at 5' cassette.)
# =============================================================================
REF_SEQ = 'AAGAUCAACGCUUCAUAUAAUCCUAAUGAUAUGGUUUGGGAGUUUCUACCAAGAGCCUUAAACUCUUGAUUAUGAAGUCUGUCGCUUUAUCCGAAAUUUUAUAAAGAGAAGACUCAUGAAUUC'
MASKING  = '000000000000000001110000001111111110000001111111100000011111110000001100000000000000000000000000000000000000000000000000000'

DEFAULT_EDGE_MASK = 25


def build_custom_mask(full_seq):
    """Return a '0'/'1' string (ExperimentFit convention: '1' = include).

    Inside REF_SEQ: MASKING '0' → '1' (keep), MASKING '1' → '0' (exclude).
    Outside REF_SEQ: standard edge mask of DEFAULT_EDGE_MASK positions.
    """
    n = len(full_seq)
    start = full_seq.find(REF_SEQ)
    if start < 0:
        raise RuntimeError('REF_SEQ not found in full amplicon sequence')

    mask = ['0' if (i < DEFAULT_EDGE_MASK or i >= n - DEFAULT_EDGE_MASK) else '1'
            for i in range(n)]
    for i, c in enumerate(MASKING):
        mask[start + i] = '0' if c == '1' else '1'
    return ''.join(mask)


def set_rna_params(name):
    if name == 'andronescu':
        RNA.params_load_RNA_Andronescu2007()
    elif name == 'turner':
        RNA.params_load_RNA_Turner2004()
    else:
        raise ValueError(f'Unknown RNA parameter set: {name}')


def find_info_txt(srr):
    pattern = os.path.join(RESULTS_DIR, srr, '*.txt')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f'No .txt info file for {srr} at {pattern}')
    return matches[0]


def load_experiment(srr, verbose=True):
    path = find_info_txt(srr)
    exp = Experiment(path, keep_all_positions=True)
    if verbose:
        print(f'  {srr}: {exp.short_description}, N={exp.N_seq}, '
              f'conc={exp.conc_mM} mM, NaN={exp.df["total_count"].isnull().sum()}')
    return exp


# =============================================================================
# Phase 1
# =============================================================================
def run_phase1(root_dir, rna_params, use_custom_mask, linear_mode, overwrite,
               do_plots, print_to_std_out, guess=None):
    print('=' * 60)
    print('PHASE 1: fitting physical parameters (ctrl + ade=0 µM)')
    print(f'  RNA params:  {rna_params}')
    print('=' * 60)

    set_rna_params(rna_params)
    ctrl = load_experiment(CONTROL_REP1_SRR)
    experiments = [ctrl, load_experiment('SRR15560843')]

    custom_mask = build_custom_mask(ctrl.seq) if use_custom_mask else None
    if custom_mask:
        print(f'  Custom mask: {custom_mask.count("1")}/{len(custom_mask)} positions kept')

    MultiSystemsFit(
        experiments=experiments,
        output_suffix='add_riboswitch_WT',
        root_dir=root_dir,
        fit_mode='physical_only',
        infer_1D_sc=False,
        custom_mask=custom_mask,
        linear_mode=linear_mode,
        overwrite=overwrite,
        do_plots=do_plots,
        print_to_std_out=print_to_std_out,
        guess=guess,
    ).fit()

    params_path = os.path.join(root_dir, 'add_riboswitch_WT', 'params1D.txt')
    print(f'\nPhase 1 done. Params: {params_path}')
    return params_path


# =============================================================================
# Phase 2 (parallel)
# =============================================================================
def _phase2_job(args):
    srr, ade_uM, phase1_params, root_dir, rna_params, use_custom_mask, \
        resume_dir, overwrite, do_plots, print_to_std_out = args
    try:
        set_rna_params(rna_params)
        ctrl = load_experiment(CONTROL_REP1_SRR, verbose=False)
        dms  = load_experiment(srr, verbose=False)
        custom_mask = build_custom_mask(ctrl.seq) if use_custom_mask else None
        suffix = f'ade_{ade_uM}'
        if resume_dir:
            candidate = os.path.join(resume_dir, suffix, 'params1D.txt')
            guess = candidate if os.path.exists(candidate) else phase1_params
        else:
            guess = phase1_params
        MultiSystemsFit(
            experiments=[ctrl, dms],
            output_suffix=suffix,
            root_dir=root_dir,
            fit_mode='lambda_only',
            infer_1D_sc=True,
            custom_mask=custom_mask,
            guess=guess,
            overwrite=overwrite,
            do_plots=do_plots,
            print_to_std_out=print_to_std_out,
        ).fit()
        return {'srr': srr, 'ade_uM': ade_uM, 'ok': True,  'error': None}
    except Exception as exc:
        return {'srr': srr, 'ade_uM': ade_uM, 'ok': False, 'error': str(exc)}


def run_phase2(phase1_params, root_dir, rna_params, use_custom_mask,
               resume_dir, workers, overwrite, do_plots, print_to_std_out):
    print('=' * 60)
    print('PHASE 2: fitting lambda_sc per adenine concentration (parallel)')
    print(f'  Physical params: {phase1_params}')
    print(f'  RNA params:      {rna_params}')
    print('=' * 60)

    if not os.path.exists(phase1_params):
        raise FileNotFoundError(f'Phase 1 params not found: {phase1_params}')

    jobs = [
        (srr, ade_uM, phase1_params, root_dir, rna_params, use_custom_mask,
         resume_dir, overwrite, do_plots, print_to_std_out)
        for srr, (_, ade_uM) in SAMPLES_DMS_REP1.items()
    ]
    n_workers = workers or min(len(jobs), os.cpu_count() or 1)
    print(f'  Launching {len(jobs)} jobs with {n_workers} worker(s)')

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_phase2_job, jobs), total=len(jobs)))

    for r in results:
        status = 'OK' if r['ok'] else 'FAIL'
        print(f'  [{status}] ade={r["ade_uM"]} µM  {r["srr"]}')
        if r['error']:
            print(f'         {r["error"]}')

    failures = [r for r in results if not r['ok']]
    if failures:
        raise RuntimeError(f'Phase 2: {len(failures)} job(s) failed')

    print(f'\nPhase 2 done. Results in {root_dir}/')


# =============================================================================
# CLI
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Olson adenine riboswitch fit')
    parser.add_argument('--phase', choices=['1', '2', 'both'], default='both')
    parser.add_argument('--phase1-params', default=None,
                        help='Path to phase 1 params1D.txt (required for --phase 2)')
    parser.add_argument('--root-dir-phase1', default='fits_paper/adenine_riboswitch/phase1')
    parser.add_argument('--root-dir-phase2', default='fits_paper/adenine_riboswitch/phase2')
    parser.add_argument('--rna-params', choices=['turner', 'andronescu'], default='turner')
    parser.add_argument('--no-custom-mask', action='store_false', dest='custom_mask',
                        help='Disable masking for low-coverage 5\' cassette positions')
    parser.set_defaults(custom_mask=True)
    parser.add_argument('--no-linear', action='store_false', dest='linear_mode',
                        help='Disable linear mode (mu_r≤0, p_b=0) in phase 1')
    parser.set_defaults(linear_mode=True)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--resume-phase2-dir', default=None)
    parser.add_argument('--initial-guess', default=None)
    parser.add_argument('--overwrite', action='store_true', default=True)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    do_plots        = not args.no_plots
    print_to_stdout = not args.quiet

    phase1_params = args.phase1_params
    if args.phase in ('1', 'both'):
        phase1_params = run_phase1(
            root_dir=args.root_dir_phase1,
            rna_params=args.rna_params,
            use_custom_mask=args.custom_mask,
            linear_mode=args.linear_mode,
            overwrite=args.overwrite,
            do_plots=do_plots,
            print_to_std_out=print_to_stdout,
            guess=args.initial_guess,
        )

    if args.phase in ('2', 'both'):
        if args.phase == '2' and phase1_params is None:
            parser.error('--phase1-params required when --phase is 2')
        run_phase2(
            phase1_params=phase1_params,
            root_dir=args.root_dir_phase2,
            rna_params=args.rna_params,
            use_custom_mask=args.custom_mask,
            resume_dir=args.resume_phase2_dir,
            workers=args.workers,
            overwrite=args.overwrite,
            do_plots=do_plots,
            print_to_std_out=print_to_stdout,
        )
