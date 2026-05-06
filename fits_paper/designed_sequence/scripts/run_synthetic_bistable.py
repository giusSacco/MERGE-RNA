#!/usr/bin/env python3
"""
Paper fit: synthetic bistable RNA.

Creates a synthetic bistable RNA sequence where two conformations
compete, testing the model's ability to recover mixed structures.

Usage:
    cd $RNA_STRUCT_HOME
    python fits_paper/designed_sequence/scripts/run_synthetic_bistable.py
"""
import argparse
import os
import sys

if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

import numpy as np
from merge_rna import Experiment, create_exp_synthetic_comb, MultiSystemsFit

parser = argparse.ArgumentParser(description='Synthetic bistable RNA fit')
parser.add_argument('--output-dir', default=None,
                    help='Root output directory (default: fits_paper/designed_sequence/outputs/synthetic_bistable)')
parser.add_argument('--ref-params', default=None,
                    help='Path to reference_physical_params.txt '
                         '(default: fits_paper/designed_sequence/synthetic/reference_physical_params.txt)')
args = parser.parse_args()

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = args.output_dir if args.output_dir else os.path.join('fits_paper', 'designed_sequence', 'outputs', 'synthetic_bistable')
POP1_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # fraction of population 1

_ref_params_path = args.ref_params if args.ref_params else os.path.join(
    'fits_paper', 'designed_sequence', 'synthetic', 'reference_physical_params.txt')
_params_1D_ref = np.loadtxt(_ref_params_path)
# Build params_dict using a real 0 mM Redmond experiment (same approach as _regenerate_cache_nb4.py)
_exp_ref_0mM = next(
    (Experiment(p) for p in Experiment.paths_to_redmond_ivt_data_txt if Experiment(p).conc_mM == 0),
    None)
if _exp_ref_0mM is None:
    raise RuntimeError("Could not find a 0 mM Redmond experiment to build reference params_dict")
_multi_ref = MultiSystemsFit([_exp_ref_0mM], validation_exps=None, infer_1D_sc=False, skip_output_setup=True)
_params_dict_ref = _multi_ref.pack_params(_params_1D_ref, _multi_ref.systems[0])

# =============================================================================
# Fit each population mixture
# =============================================================================
for pop1 in POP1_VALUES:
    print(f"\n{'='*60}")
    print(f"Fitting synthetic bistable with {pop1*100:.0f}% population 1")
    print(f"{'='*60}")

    exp = create_exp_synthetic_comb(pop1=pop1, same_system=False, params_dict=_params_dict_ref)
    
    print(f"  Sequence length: {exp.N_seq}")
    print(f"  Temperature: {exp.temp_C}°C")
    
    multi_sys = MultiSystemsFit(
        experiments=[exp],
        output_suffix=f'synthetic_bistable_{pop1*100:.0f}',
        root_dir=OUTPUT_DIR,
        infer_1D_sc=True,
        fit_mode='sequential',
        do_plots=True,
        print_to_std_out=True
    )
    
    result = multi_sys.fit()
    print(f"Fit complete for pop1={pop1}")

print(f"\n{'='*60}")
print(f"All fits complete! Results saved to {OUTPUT_DIR}/")
print(f"{'='*60}")
