#!/usr/bin/env python3
"""
MERGE-RNA demo: multi-system joint fit on synthetic sequences.

Demonstrates:
- Multi-system joint fitting (3 independent 30 nt sequences)
- Concentration series: background (0 mM) + 50 mM + 100 mM per system
- Edge masking (first/last 5 positions excluded from the loss)
- Train/test split (seq_A + seq_B = training; seq_C = test)
- Sequential fit mode: Phase 1 learns shared physical parameters,
  Phase 2 refines per-sequence soft constraints (lambda_sc)

No external data required. All mutation profiles are generated synthetically.
Total runtime: ~2-3 minutes on a laptop.

Usage:
    python examples/demo.py
"""
import os
import sys
import numpy as np
import pandas as pd

if 'RNA_STRUCT_HOME' not in os.environ:
    os.environ['RNA_STRUCT_HOME'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

from merge_rna import Experiment, MultiSystemsFit

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR   = os.path.join('examples', 'outputs', 'demo')
MASK_EDGES   = (5, 5)       # NaN first/last 5 positions
COVERAGE     = 10_000
SEED         = int(os.environ.get('MERGE_RNA_SEED', 42))
SEQ_LEN      = 30
CONCS_mM     = [0, 50, 100]  # 0 mM = background; 50 and 100 mM = DMS
LAMBDA_RANGE = 0.05          # lambda_sc drawn uniformly from [-LAMBDA_RANGE, +LAMBDA_RANGE]

# Ground-truth shared physical parameters.
TRUE_PARAMS = {
    'mu_r':   0.0,
    'p_b':    1.0,
    'p_bind': {
        (0, 'A'): 0.45, (0, 'C'): 0.30, (0, 'G'): 0.05, (0, 'U'): 0.05,
        (1, 'A'): 0.00, (1, 'C'): 0.00, (1, 'G'): 0.05, (1, 'U'): 0.05,
    },
    'm0': 0.001,
    'm1': 1.0,
}

# =============================================================================
# Generate sequences and per-sequence lambda_sc (reproducible via SEED)
# =============================================================================
rng = np.random.default_rng(SEED)

SYSTEM_NAMES = ('seq_A', 'seq_B', 'seq_C')
SEQUENCES = {
    name: ''.join(rng.choice(['A', 'C', 'G', 'U'], SEQ_LEN))
    for name in SYSTEM_NAMES
}
LAMBDAS = {
    name: rng.uniform(-LAMBDA_RANGE, LAMBDA_RANGE, SEQ_LEN)
    for name in SYSTEM_NAMES
}

print("\nGenerated sequences and ground-truth lambda_sc:")
for name in SYSTEM_NAMES:
    print(f"  {name}: {SEQUENCES[name]}  "
          f"lambda_sc=[{LAMBDAS[name].min():.3f}, {LAMBDAS[name].max():.3f}]")


# =============================================================================
# Helper: create one synthetic experiment
# =============================================================================
def make_experiment(system_name, conc_mM):
    """Return an Experiment with binomial-noise synthetic mutation data."""
    sequence = SEQUENCES[system_name]
    n = len(sequence)
    exp = Experiment(None, seq=sequence, system_name=system_name,
                     conc_mM=conc_mM, temp_C=37.0)
    exp.df = pd.DataFrame({
        'pos':         np.arange(1, n + 1),
        'ref_nt':      list(sequence),
        'total_count': COVERAGE,
        'mut_count':   np.zeros(n, dtype=int),
        'wt_count':    np.full(n, COVERAGE, dtype=int),
        'mut_rate':    np.zeros(n),
        'Sample':      f'{system_name}_{conc_mM}mM',
    })
    exp.raw_df = exp.df.copy()

    params = {**TRUE_PARAMS, 'lambda_sc': LAMBDAS[system_name]}
    exp.df = exp.generate_synthetic_data(params_dict=params,
                                          coverage=COVERAGE, noise=True)

    left, right = MASK_EDGES
    if left  > 0: exp.df.loc[:left - 1,  'mut_rate'] = np.nan
    if right > 0: exp.df.loc[n - right:, 'mut_rate'] = np.nan
    exp.raw_df = exp.df.copy()
    return exp


# =============================================================================
# Build experiment sets
# seq_A, seq_B → training (all concentrations)
# seq_C        → test/validation (all concentrations)
# =============================================================================
train_exps = [make_experiment(name, c) for name in ('seq_A', 'seq_B') for c in CONCS_mM]
val_exps   = [make_experiment('seq_C', c) for c in CONCS_mM]

train_sys = sorted({e.system_name for e in train_exps})
val_sys   = sorted({e.system_name for e in val_exps})
print(f"\nTraining   ({len(train_exps)} experiments, systems: {train_sys})")
print(f"Validation ({len(val_exps)} experiments, systems: {val_sys})")
print(f"Concentrations: {CONCS_mM} mM  |  mask edges: {MASK_EDGES}  |  length: 30 nt\n")

# =============================================================================
# Run joint fit
# =============================================================================
multi_sys = MultiSystemsFit(
    experiments=train_exps,
    validation_exps=val_exps,
    output_suffix='demo',
    root_dir=OUTPUT_DIR,
    infer_1D_sc=True,
    mask_edges=MASK_EDGES,
    fit_mode='sequential',    # Phase 1: physical params; Phase 2: lambda_sc
    do_plots=True,
    print_to_std_out=True,
    overwrite=True,
)

result = multi_sys.fit()

print(f"\nFit complete! Results saved to {OUTPUT_DIR}/demo/")
print(f"  params1D.txt        — fitted parameters")
print(f"  *_mut_profile.png   — observed vs. predicted mutation rates")
print(f"  loss_history.png    — convergence curve")
