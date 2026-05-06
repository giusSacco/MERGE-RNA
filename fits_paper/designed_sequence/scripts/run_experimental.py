#!/usr/bin/env python3
"""
Paper fit: newseq WT experimental bistable data.

Two-phase fit:
  Phase 1: physical parameters only (no interpolated pairing probs).
  Phase 2: fix physical params, fit lambda_sc with use_interpolated_ps=True.

Usage:
    cd $RNA_STRUCT_HOME
    python fits_paper/designed_sequence/scripts/run_experimental.py --phase both
    python fits_paper/designed_sequence/scripts/run_experimental.py --phase 2 \\
        --phase1-params fits_paper/designed_sequence/outputs/experimental_turner/phase1/newseqWT/params1D.txt
"""
import argparse
import os
import sys

if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

import RNA
from merge_rna import Experiment, MultiSystemsFit

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(description='newseq WT experimental bistable fit')
parser.add_argument('--phase', choices=['1', '2', 'both'], default='both')
parser.add_argument('--phase1-params', default=None,
                    help='Path to phase 1 params1D.txt (required for --phase 2)')
parser.add_argument('--rna-params', choices=['turner', 'andronescu'], default='turner',
                    help='ViennaRNA thermodynamic parameter set (default: turner)')
parser.add_argument('--max-iter', type=int, default=None)
parser.add_argument('--no-plots', action='store_true')
parser.add_argument('--overwrite', action='store_true', default=True)
parser.add_argument('--output-dir', default=None,
                    help='Root output directory (default: fits_paper/designed_sequence/outputs/experimental_<rna-params>)')
args = parser.parse_args()

if args.rna_params == 'andronescu':
    RNA.params_load_RNA_Andronescu2007()
    print("Loaded Andronescu (2007) RNA parameters")
else:
    RNA.params_load_RNA_Turner2004()
    print("Loaded Turner (2004) RNA parameters")

# =============================================================================
# Configuration
# =============================================================================
_base = args.output_dir if args.output_dir else os.path.join('fits_paper', 'designed_sequence', 'outputs', f'experimental_{args.rna_params}')
ROOT_DIR_PHASE1 = os.path.join(_base, 'phase1')
ROOT_DIR_PHASE2 = os.path.join(_base, 'phase2')

# =============================================================================
# Load experiments
# =============================================================================
experiments = [Experiment(path) for path in Experiment.paths_to_newseq_WT_data_txt]

print(f"Loaded {len(experiments)} newseq WT experiments:")
for exp in experiments:
    print(f"  - {exp.system} @ {exp.conc_mM}mM, {exp.temp_C}°C, rep {exp.rep_number}")

_common = dict(
    experiments=experiments,
    output_suffix='newseqWT',
    strict_convergence=True,
    do_plots=not args.no_plots,
    print_to_std_out=True,
    max_iter=args.max_iter,
    overwrite=args.overwrite,
)

# =============================================================================
# Phase 1: physical parameters only, no interpolated pairing probs
# =============================================================================
phase1_params = args.phase1_params
if args.phase in ('1', 'both'):
    print(f"\n{'='*60}")
    print(f"PHASE 1: fitting physical parameters")
    print(f"{'='*60}")

    MultiSystemsFit(
        root_dir=ROOT_DIR_PHASE1,
        fit_mode='physical_only',
        infer_1D_sc=False,
        use_interpolated_ps=False,
        **_common,
    ).fit()

    phase1_params = os.path.join(ROOT_DIR_PHASE1, 'newseqWT', 'params1D.txt')
    print(f'\nPhase 1 done. Params: {phase1_params}')

# =============================================================================
# Phase 2: lambda_sc only, with interpolated pairing probs
# =============================================================================
if args.phase in ('2', 'both'):
    if args.phase == '2' and phase1_params is None:
        parser.error('--phase1-params required when --phase is 2')
    if not os.path.exists(phase1_params):
        raise FileNotFoundError(f'Phase 1 params not found: {phase1_params}')

    print(f"\n{'='*60}")
    print(f"PHASE 2: fitting lambda_sc (use_interpolated_ps=True)")
    print(f"  Physical params: {phase1_params}")
    print(f"{'='*60}")

    MultiSystemsFit(
        root_dir=ROOT_DIR_PHASE2,
        fit_mode='lambda_only',
        infer_1D_sc=True,
        use_interpolated_ps=True,
        guess=phase1_params,
        **_common,
    ).fit()

    print(f'\nPhase 2 done. Results in {ROOT_DIR_PHASE2}/newseqWT/')
