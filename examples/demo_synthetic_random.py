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
OUTPUT_DIR = os.path.join('examples', 'outputs', 'demo_synthetic_random')
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

# Create experiment object using proper Experiment constructor
exp = Experiment(None, seq=sequence, system_name=f'random_{SEQUENCE_LENGTH}nt', 
                 conc_mM=100.0, temp_C=25.0, rep_number=1)

# Create dataframe
coverage = 10000
exp.df = pd.DataFrame({
    'pos': np.arange(1, SEQUENCE_LENGTH + 1),
    'ref_nt': list(sequence),
    'total_count': coverage,
    'mut_count': np.zeros(SEQUENCE_LENGTH, dtype=int),
    'wt_count': coverage * np.ones(SEQUENCE_LENGTH, dtype=int),
    'mut_rate': np.zeros(SEQUENCE_LENGTH),
    'Sample': f'random_{SEQUENCE_LENGTH}nt'
})
exp.raw_df = exp.df.copy()

# Generate synthetic data
eps_b = np.zeros(SEQUENCE_LENGTH)  # No background for synthetic data

# Create default params_dict for synthetic data generation
# These are typical values for DMS probing
params_dict = {
    'mu_r': 0.5,      # chemical potential
    'p_b': 1.0,       # penalty for paired bases  
    'p_bind': {       # binding probabilities per nucleotide type and pairing state
        (0, 'A'): 0.3,  # unpaired A
        (0, 'C'): 0.3,  # unpaired C
        (0, 'G'): 0.05, # unpaired G
        (0, 'U'): 0.05, # unpaired U
        (1, 'A'): 0.0,  # paired A (DMS doesn't react with paired A)
        (1, 'C'): 0.0,  # paired C (DMS doesn't react with paired C)
        (1, 'G'): 0.05, # paired G
        (1, 'U'): 0.05, # paired U
    },
    'm0': 0.001,
    'm1': 1.0,
    'lambda_sc': np.zeros(SEQUENCE_LENGTH),  # soft constraints (none initially)
}

# Generate synthetic mutation data using Experiment's method
exp.df = exp.generate_synthetic_data(params_dict=params_dict, coverage=coverage, noise=True, eps_b=eps_b)

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
