#!/usr/bin/env python3
"""
Paper fit: IVT structured RNAs (all systems together).

Trains on 3 systems, validates on 2 (hc16, bact_RNaseP_typeA).
Tests generalization of physical parameters across different RNAs.

Systems: hc16, bact_RNaseP_typeA, tetrahymena_ribozyme, HCV_IRES, V_chol_gly_riboswitch

Usage:
    cd $RNA_STRUCT_HOME
    python fits_paper/structured_rnas/scripts/run_structured_rnas.py [--rna-params andronescu]
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
parser = argparse.ArgumentParser(description='Structured RNAs joint fit')
parser.add_argument('--rna-params', choices=['turner', 'andronescu'], default='turner',
                    help='ViennaRNA thermodynamic parameter set (default: turner)')
parser.add_argument('--max-iter', type=int, default=None)
parser.add_argument('--no-plots', action='store_true')
parser.add_argument('--overwrite', action='store_true', default=True)
parser.add_argument('--output-suffix', default='redmond_together',
                    help='Base output suffix (phase1 uses <suffix>_phase1)')
parser.add_argument('--output-dir', default=None,
                    help='Root output directory (default: fits_paper/structured_rnas/outputs/<rna-params>)')
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
OUTPUT_DIR = args.output_dir if args.output_dir else os.path.join('fits_paper', 'structured_rnas', 'outputs', args.rna_params)
VALIDATION_SYSTEMS = ['hc16', 'bact_RNaseP_typeA']
PHASE1_SUFFIX = f"{args.output_suffix}_phase1"
PHASE2_SUFFIX = args.output_suffix

# =============================================================================
# Load experiments
# =============================================================================
all_experiments = [Experiment(path) for path in Experiment.paths_to_redmond_ivt_data_txt]
all_experiments = [exp for exp in all_experiments if exp.conc_mM != 85]

train_experiments = [exp for exp in all_experiments if exp.system not in VALIDATION_SYSTEMS]
val_experiments = [exp for exp in all_experiments if exp.system in VALIDATION_SYSTEMS]

print(f"Training on {len(train_experiments)} experiments:")
for exp in train_experiments:
    print(f"  - {exp.system} @ {exp.conc_mM}mM, rep {exp.rep_number}")

print(f"\nValidation on {len(val_experiments)} experiments:")
for exp in val_experiments:
    print(f"  - {exp.system} @ {exp.conc_mM}mM, rep {exp.rep_number}")

# =============================================================================
# Run fit
# =============================================================================
print(f"\nRunning phase 1 (physical-only with validation)...")
print(f"Phase 1 output: {OUTPUT_DIR}/{PHASE1_SUFFIX}/")

phase1 = MultiSystemsFit(
    experiments=train_experiments,
    validation_exps=val_experiments,
    output_suffix=PHASE1_SUFFIX,
    root_dir=OUTPUT_DIR,
    infer_1D_sc=False,
    fit_mode='physical_only',
    do_plots=not args.no_plots,
    print_to_std_out=True,
    max_iter=args.max_iter,
    overwrite=args.overwrite,
)

phase1.fit()

phase1_params = os.path.join(OUTPUT_DIR, PHASE1_SUFFIX, 'params1D.txt')

print(f"\nRunning phase 2 (lambda-only, all systems training)...")
print(f"Phase 2 output: {OUTPUT_DIR}/{PHASE2_SUFFIX}/")

phase2 = MultiSystemsFit(
    experiments=all_experiments,
    validation_exps=None,
    output_suffix=PHASE2_SUFFIX,
    root_dir=OUTPUT_DIR,
    infer_1D_sc=True,
    fit_mode='lambda_only',
    do_plots=not args.no_plots,
    print_to_std_out=True,
    max_iter=args.max_iter,
    overwrite=args.overwrite,
    guess=phase1_params,
)

phase2.fit()

print(f"\nFit complete! Results saved to {OUTPUT_DIR}/{PHASE2_SUFFIX}/")
