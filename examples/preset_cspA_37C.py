#!/usr/bin/env python3
"""
Preset: Fit cspA RNA at 37°C.

Usage:
    cd $RNA_STRUCT_HOME
    python examples/preset_cspA_37C.py
"""
import os
import sys

if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

from class_experiment import Experiment
from class_experimentfit import MultiSystemsFit

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = 'preset_cspA_37C'

# =============================================================================
# Load experiments
# =============================================================================
experiments = [Experiment(path) for path in Experiment.paths_to_cspa_37C_data_txt]

print(f"Loaded {len(experiments)} cspA 37°C experiments:")
for exp in experiments:
    print(f"  - {exp.system} @ {exp.conc_mM}mM, {exp.temp_C}°C, rep {exp.replicate}")

# =============================================================================
# Run fit
# =============================================================================
print(f"\nRunning fit...")
print(f"Output will be saved to: {OUTPUT_DIR}/")

multi_sys = MultiSystemsFit(
    experiments=experiments,
    output_suffix='cspA_37C',
    root_dir=OUTPUT_DIR,
    infer_1D_sc=True,
    fit_mode='sequential',
    do_plots=True,
    print_to_std_out=True
)

result = multi_sys.fit()

print(f"\nFit complete! Results saved to {OUTPUT_DIR}/cspA_37C/")
