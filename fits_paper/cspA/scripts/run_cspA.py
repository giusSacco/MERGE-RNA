#!/usr/bin/env python3
"""
Paper fit: cspA mRNA — joint fit of 10°C and 37°C combined experiments.

Creates two combined experiments via initialise_combined_cspA_exp:
  - pop_10=1.0  → pure 10°C measurement
  - pop_10=0.0  → pure 37°C measurement

Physical parameters are shared across both temperatures (phase 1);
per-sequence lambda_sc are refined independently (phase 2).

Usage:
    cd $RNA_STRUCT_HOME
    python fits_paper/cspA/scripts/run_cspA.py [--rna-params andronescu]
"""
import argparse
import os
import sys

if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

import RNA
from merge_rna import Experiment, MultiSystemsFit, initialise_combined_cspA_exp

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(description='cspA joint fit (10°C + 37°C)')
parser.add_argument('--rna-params', choices=['turner', 'andronescu'], default='turner',
                    help='ViennaRNA thermodynamic parameter set (default: turner)')
parser.add_argument('--max-iter', type=int, default=None)
parser.add_argument('--no-plots', action='store_true')
parser.add_argument('--overwrite', action='store_true', default=True)
parser.add_argument('--output-dir', default=None,
                    help='Root output directory (default: fits_paper/cspA/outputs/<rna-params>)')
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
OUTPUT_DIR = args.output_dir if args.output_dir else os.path.join('fits_paper', 'cspA', 'outputs', args.rna_params)

# =============================================================================
# Build combined experiments (normalised so rates are on the same scale)
# =============================================================================
exp_10C = initialise_combined_cspA_exp(pop_10=1.0, normalise_mut_rate=True)
exp_37C = initialise_combined_cspA_exp(pop_10=0.0, normalise_mut_rate=True)

print(f"10°C combined experiment: system={exp_10C.system_name}, n={len(exp_10C.df)}")
print(f"37°C combined experiment: system={exp_37C.system_name}, n={len(exp_37C.df)}")

# =============================================================================
# Run joint fit
# =============================================================================
print(f"\nRunning joint fit (sequential mode)...")
print(f"Output will be saved to: {OUTPUT_DIR}/")

multi_sys = MultiSystemsFit(
    experiments=[exp_10C, exp_37C],
    output_suffix='cspA_combined',
    root_dir=OUTPUT_DIR,
    infer_1D_sc=True,
    fit_mode='sequential',
    do_plots=not args.no_plots,
    print_to_std_out=True,
    max_iter=args.max_iter,
    overwrite=args.overwrite,
)

result = multi_sys.fit()

print(f"\nFit complete! Results saved to {OUTPUT_DIR}/cspA_combined/")
