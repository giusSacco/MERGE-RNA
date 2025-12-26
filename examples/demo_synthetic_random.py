#!/usr/bin/env python3
"""
Demo: Fit a random synthetic RNA sequence.

This example generates a random 30nt RNA sequence with synthetic mutation data
and runs the full fitting pipeline. Useful for testing the installation and
understanding the fitting workflow.

Usage:
    cd $RNA_STRUCT_HOME
    python examples/demo_synthetic_random.py
"""
import os
import sys
import numpy as np

# Ensure we're in the right directory
if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

from class_experiment import Experiment
from class_experimentfit import ExperimentFit, System, MultiSystemsFit

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = 'demo_synthetic_random'
SEQUENCE_LENGTH = 30
RANDOM_SEED = 42
MASK_EDGES = (5, 5)  # NaN at first/last 5 positions

# =============================================================================
# Generate random sequence
# =============================================================================
np.random.seed(RANDOM_SEED)
sequence = ''.join(np.random.choice(['A', 'C', 'G', 'U'], SEQUENCE_LENGTH))
print(f"Generated sequence: {sequence}")

# =============================================================================
# Create synthetic experiment
# =============================================================================
import pandas as pd

# Create experiment object
exp = Experiment.__new__(Experiment)
exp.sequence = sequence
exp.system = f'random_{SEQUENCE_LENGTH}nt'
exp.conc_mM = 100.0
exp.temp_C = 25.0
exp.replicate = 1
exp.seq_length = SEQUENCE_LENGTH
exp.pdb = None

# Create dataframe
coverage = 10000
exp.df = pd.DataFrame({
    'position': np.arange(1, SEQUENCE_LENGTH + 1),
    'nucleotide': list(sequence),
    'total_count': coverage,
    'mut_count': np.zeros(SEQUENCE_LENGTH, dtype=int),
    'wt_count': coverage * np.ones(SEQUENCE_LENGTH, dtype=int),
    'mut_rate': np.zeros(SEQUENCE_LENGTH)
})
exp.raw_df = exp.df.copy()

# Generate synthetic data
exp_fit = ExperimentFit(exp)
params_dict = exp_fit.default_params_dict.copy()
exp.df = exp_fit.generate_synthetic_data(params_dict=params_dict, coverage=coverage, noise=True)

# Apply edge masking
left_mask, right_mask = MASK_EDGES
if left_mask > 0:
    exp.df.loc[:left_mask - 1, 'mut_rate'] = np.nan
if right_mask > 0:
    exp.df.loc[SEQUENCE_LENGTH - right_mask:, 'mut_rate'] = np.nan
exp.raw_df = exp.df.copy()

# =============================================================================
# Run fit
# =============================================================================
print(f"\nRunning fit on {SEQUENCE_LENGTH}nt random sequence...")
print(f"Output will be saved to: {OUTPUT_DIR}/")

multi_sys = MultiSystemsFit(
    experiments=[exp],
    output_suffix='random_seq',
    root_dir=OUTPUT_DIR,
    infer_1D_sc=True,
    mask_edges=MASK_EDGES,
    fit_mode='sequential',
    do_plots=True,
    print_to_std_out=True
)

result = multi_sys.fit()

print(f"\nFit complete! Results saved to {OUTPUT_DIR}/random_seq/")
print(f"  - params1D.txt: fitted parameters")
print(f"  - *.png: diagnostic plots")
