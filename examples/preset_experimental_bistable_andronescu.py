#!/usr/bin/env python3
"""
Preset: Fit newseq WT experimental bistable data with Andronescu (2007) parameters.

This uses an alternative thermodynamic parameter set from ViennaRNA.

Usage:
    cd $RNA_STRUCT_HOME
    python examples/preset_experimental_bistable_andronescu.py
"""
import os
import sys

if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

import RNA
from class_experiment import Experiment
from class_experimentfit import MultiSystemsFit

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = 'preset_experimental_bistable_andronescu'

# Load Andronescu (2007) parameters BEFORE creating experiments
RNA.params_load_RNA_Andronescu2007()
print("Loaded Andronescu (2007) RNA parameters")

# =============================================================================
# Load experiments
# =============================================================================
experiments = [Experiment(path) for path in Experiment.paths_to_newseq_WT_data_txt]

print(f"Loaded {len(experiments)} newseq WT experiments:")
for exp in experiments:
    print(f"  - {exp.system} @ {exp.conc_mM}mM, {exp.temp_C}°C, rep {exp.rep_number}")

# =============================================================================
# Fit with Andronescu (2007) parameters
# =============================================================================
print(f"\n{'='*60}")
print("Fitting with Andronescu (2007) parameters")
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
