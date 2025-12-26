#!/usr/bin/env python3
"""
Test script: Generate synthetic data for a random short RNA sequence.
Creates a 30nt random sequence with synthetic mutation profiles,
with NaN mutations at positions 1-5 and 26-30 (mimicking real data masking).
"""

import numpy as np
import pandas as pd
import os

# Ensure RNA_STRUCT_HOME is set
if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.abspath(__file__))

from class_experiment import Experiment
from class_experimentfit import ExperimentFit, System, MultiSystemsFit

def generate_random_sequence(length=30, seed=42):
    """Generate a random RNA sequence."""
    np.random.seed(seed)
    nucleotides = ['A', 'C', 'G', 'U']
    return ''.join(np.random.choice(nucleotides, size=length))


def create_short_synthetic_experiment(seq=None, conc_mM=50, coverage=10000, 
                                       noise=True, mask_ends=5, seed=42):
    """
    Create a synthetic experiment for a short RNA sequence.
    
    Args:
        seq: RNA sequence (if None, generates random 30nt sequence)
        conc_mM: concentration in mM
        coverage: read depth
        noise: whether to add binomial noise
        mask_ends: number of nucleotides at each end to set as NaN
        seed: random seed for reproducibility
    
    Returns:
        Experiment object with synthetic data
    """
    if seq is None:
        seq = generate_random_sequence(30, seed=seed)
    
    print(f"Sequence ({len(seq)} nt): {seq}")
    
    # Load reference parameters from a previous fit
    params_1D = np.loadtxt("fits/red_crossval_1/red_crossval_bact_RNaseP_typeA_tetrahymena_ribozyme_V_chol_gly_riboswitch/params1D.txt")
    
    # Create a reference experiment just to get the params_dict structure
    ref_exp = [Experiment(path_) for path_ in Experiment.paths_to_redmond_ivt_data_txt 
               if Experiment(path_).conc_mM == 0][0]
    multi_exp = MultiSystemsFit([ref_exp], validation_exps=None, infer_1D_sc=False)
    params_dict = multi_exp.pack_params(params_1D, multi_exp.systems[0])
    
    # Clean up
    del params_1D, ref_exp, multi_exp
    
    # Create the synthetic experiment
    exp = Experiment(
        seq=seq, 
        temp_C=37, 
        reagent='DMS synthetic', 
        system_name='test_short_30nt',
        conc_mM=conc_mM
    )
    
    # Generate synthetic data
    exp.df = exp.generate_synthetic_data(
        params_dict=params_dict, 
        coverage=coverage, 
        noise=noise
    )
    
    # Mask first and last N nucleotides with NaN (mimicking real data)
    # Store raw_df with full data for reference
    exp.raw_df = exp.df.copy()
    
    if mask_ends > 0:
        mask_indices = list(range(mask_ends)) + list(range(len(seq) - mask_ends, len(seq)))
        exp.df.loc[mask_indices, 'mut_rate'] = np.nan
        print(f"Masked positions 1-{mask_ends} and {len(seq)-mask_ends+1}-{len(seq)} with NaN")
    
    # NOTE: We keep df at full length (30 rows) - the ExperimentFit mask handles NaN positions
    # This matches how real data works: df has same length as sequence
    
    return exp


def main():
    print("=" * 60)
    print("Creating synthetic experiment for a random 30nt sequence")
    print("=" * 60)
    
    # Create the synthetic experiment
    exp = create_short_synthetic_experiment(
        seq=None,  # Will generate random sequence
        conc_mM=50,
        coverage=10000,
        noise=True,
        mask_ends=5,
        seed=42
    )
    
    print("\n" + "=" * 60)
    print("DataFrame preview:")
    print("=" * 60)
    print(exp.df.to_string())
    
    print("\n" + "=" * 60)
    print("Summary statistics:")
    print("=" * 60)
    print(f"  Total positions: {len(exp.df)}")
    print(f"  Non-NaN positions: {exp.df['mut_rate'].notna().sum()}")
    print(f"  Mean mutation rate (non-NaN): {exp.df['mut_rate'].mean():.4f}")
    print(f"  Min mutation rate: {exp.df['mut_rate'].min():.4f}")
    print(f"  Max mutation rate: {exp.df['mut_rate'].max():.4f}")
    
    # Quick test: create an ExperimentFit
    print("\n" + "=" * 60)
    print("Testing ExperimentFit creation...")
    print("=" * 60)
    
    eps_b = np.zeros(len(exp.seq))
    exp_fit = ExperimentFit(exp, infer_1D_sc=False, eps_b=eps_b, is_training=True)
    print(f"  ExperimentFit created successfully")
    print(f"  Position mask shape: {exp_fit.position_mask.shape}")
    print(f"  Masked positions (excluded): {np.sum(~exp_fit.position_mask)}")
    
    # Check how NaN positions are handled in loss calculation
    nan_positions = exp.df['mut_rate'].isna()
    print(f"  NaN positions in data: {np.sum(nan_positions)} (indices: {np.where(nan_positions)[0].tolist()})")
    
    # Test fitting on this synthetic data
    print("\n" + "=" * 60)
    print("Running fit on synthetic data...")
    print("=" * 60)
    
    # Create a System with this experiment (need 0mM control for eps_b)
    # For synthetic data, we'll create a 0mM experiment with zero mutations
    exp_0mM = create_short_synthetic_experiment(
        seq=exp.seq, conc_mM=0, coverage=10000, noise=False, mask_ends=5, seed=42
    )
    # Override with zero mutations for 0mM control
    exp_0mM.df['mut_count'] = 0
    exp_0mM.df['mut_rate'] = 0.0
    
    # Run MultiSystemsFit with mask_edges to handle the NaN positions
    multi_fit = MultiSystemsFit(
        experiments=[exp_0mM, exp],  # 0mM first for eps_b calculation
        validation_exps=None,
        infer_1D_sc=False,  # Start without lambda_sc
        do_plots=True,  # Generate diagnostic plots
        strict_convergence=False,  # Use scipy defaults for faster testing
        mask_edges=(5, 5),  # Mask first 5 and last 5 positions (matching our NaN mask)
        root_dir='demo_synthetic',  # Custom output directory
    )
    
    print(f"  Number of systems: {len(multi_fit.systems)}")
    
    # Run the fit
    print("\n  Starting optimization...")
    result = multi_fit.fit()
    
    print(f"\n  Optimization completed!")
    print(f"  Final loss: {result.fun:.6f}")
    print(f"  Number of iterations: {result.nit}")
    print(f"  Converged: {result.success}")
    
    # Unpack and display fitted parameters
    params_dict = multi_fit.pack_params(result.x, multi_fit.systems[0])
    print(f"\n  Fitted parameters:")
    print(f"    mu_r: {params_dict['mu_r']:.4f}")
    print(f"    p_b: {params_dict['p_b']:.4f}")
    print(f"    m0: {params_dict['m0']:.6f}")
    print(f"    m1: {params_dict['m1']:.6f}")
    print(f"    p_bind: {params_dict['p_bind']}")
    
    return exp


if __name__ == "__main__":
    exp = main()
