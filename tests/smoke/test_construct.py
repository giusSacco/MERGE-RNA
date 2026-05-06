#!/usr/bin/env python3
"""Smoke test: object construction.

Builds a small synthetic Experiment + ExperimentFit and verifies the
basic invariants (DataFrame columns, sequence length, mask non-empty).
Does NOT run a fit.
"""
import os
import sys

os.environ.setdefault('RNA_STRUCT_HOME',
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

import numpy as np
import pandas as pd

from merge_rna import Experiment, ExperimentFit

SEQ_LEN = 30
np.random.seed(42)
sequence = ''.join(np.random.choice(['A', 'C', 'G', 'U'], SEQ_LEN))

exp = Experiment(None, seq=sequence, system_name='smoke_test',
                 conc_mM=100.0, temp_C=25.0)

coverage = 10000
exp.df = pd.DataFrame({
    'pos': np.arange(1, SEQ_LEN + 1),
    'ref_nt': list(sequence),
    'total_count': coverage,
    'mut_count': np.zeros(SEQ_LEN, dtype=int),
    'wt_count': coverage * np.ones(SEQ_LEN, dtype=int),
    'mut_rate': np.zeros(SEQ_LEN),
    'Sample': 'smoke_test',
})
exp.raw_df = exp.df.copy()

# DataFrame invariants
for col in ['pos', 'ref_nt', 'total_count', 'mut_count', 'mut_rate']:
    assert col in exp.df.columns, f"missing column: {col}"
assert len(exp.df) == SEQ_LEN
assert exp.seq == sequence

# Construct ExperimentFit
eps_b = np.zeros(SEQ_LEN)
fit = ExperimentFit(
    exp,
    infer_1D_sc=True,
    eps_b=eps_b,
    is_training=True,
    mask_edges=(5, 5),
)

# Mask invariants
assert fit.position_mask is not None
assert fit.position_mask.dtype == bool
assert fit.position_mask.sum() > 0, "mask must include at least one position"
# Edge masking should exclude first/last 5 positions
assert not fit.position_mask[:5].any(), "first 5 positions should be masked out"
assert not fit.position_mask[-5:].any(), "last 5 positions should be masked out"

print("OK construct")
