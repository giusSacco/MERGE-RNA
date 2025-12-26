#!/usr/bin/env python3
"""
Preset: Fit newseq WT experimental bistable data with Turner (2004) parameters.

This is the default ViennaRNA parameter set.

Usage:
    cd $RNA_STRUCT_HOME
    python examples/preset_experimental_bistable_turner.py
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
OUTPUT_DIR = 'preset_experimental_bistable_turner'

# =============================================================================
# Load experiments
# =============================================================================
experiments = [Experiment(path) for path in Experiment.paths_to_newseq_WT_data_txt]

print(f"Loaded {len(experiments)} newseq WT experiments:")
for exp in experiments:
    print(f"  - {exp.system} @ {exp.conc_mM}mM, {exp.temp_C}°C, rep {exp.rep_number}")

# =============================================================================
# Fit with Turner (default) parameters
# =============================================================================
print(f"\n{'='*60}")
print("Fitting with Turner (2004) parameters (default)")
print(f"{'='*60}")

multi_sys = MultiSystemsFit(
    experiments=experiments,
    output_suffix='newseqWT',
    root_dir=OUTPUT_DIR,
    infer_1D_sc=True,
    use_interpolated_ps=True,
    fit_mode='sequential',
    strict_convergence=True,
    do_plots=True,
    print_to_std_out=True
)

result = multi_sys.fit()

print(f"\n{'='*60}")
print(f"Fit complete! Results saved to {OUTPUT_DIR}/newseqWT/")
print(f"{'='*60}")
