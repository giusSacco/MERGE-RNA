#!/usr/bin/env python3
"""
Preset: Fit Redmond IVT structured RNAs (all systems together).

Trains on 3 systems, validates on 2 (hc16, bact_RNaseP_typeA).
This tests generalization of physical parameters across different RNAs.

Systems: hc16, bact_RNaseP_typeA, tetrahymena_ribozyme, HCV_IRES, V_chol_gly_riboswitch

Usage:
    cd $RNA_STRUCT_HOME
    python examples/preset_structured_rnas.py
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
OUTPUT_DIR = 'preset_structured_rnas'
VALIDATION_SYSTEMS = ['hc16', 'bact_RNaseP_typeA']

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
print(f"\nRunning fit...")
print(f"Output will be saved to: {OUTPUT_DIR}/")

multi_sys = MultiSystemsFit(
    experiments=train_experiments,
    validation_exps=val_experiments,
    output_suffix='redmond_together',
    root_dir=OUTPUT_DIR,
    infer_1D_sc=True,
    fit_mode='sequential',
    do_plots=True,
    print_to_std_out=True
)

result = multi_sys.fit()

print(f"\nFit complete! Results saved to {OUTPUT_DIR}/redmond_together/")
