#!/usr/bin/env python3
"""
Preset: Fit synthetic bistable RNA.

This creates a synthetic bistable RNA sequence where two conformations
compete, testing the model's ability to recover mixed structures.

Usage:
    cd $RNA_STRUCT_HOME
    python examples/preset_synthetic_bistable.py
"""
import os
import sys

if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

from class_experiment import create_exp_synthetic_comb
from class_experimentfit import MultiSystemsFit

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = 'preset_synthetic_bistable'
POP1_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # fraction of population 1

# =============================================================================
# Fit each population mixture
# =============================================================================
for pop1 in POP1_VALUES:
    print(f"\n{'='*60}")
    print(f"Fitting synthetic bistable with {pop1*100:.0f}% population 1")
    print(f"{'='*60}")
    
    exp = create_exp_synthetic_comb(pop1=pop1, same_system=False)
    
    print(f"  Sequence length: {exp.seq_length}")
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
