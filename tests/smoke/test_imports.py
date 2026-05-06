#!/usr/bin/env python3
"""Smoke test: import sanity.

Confirms that core library symbols can be imported from the package path.
"""
import os
import sys

os.environ.setdefault('RNA_STRUCT_HOME',
                      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.environ['RNA_STRUCT_HOME'])

import merge_rna
from merge_rna import (
    Experiment,
    ExperimentFit,
    System,
    MultiSystemsFit,
    clocked,
    load_rc,
    initialise_combined_cspA_exp,
    create_exp_synthetic_comb,
)

assert Experiment is not None
assert ExperimentFit is not None
assert System is not None
assert MultiSystemsFit is not None
assert callable(clocked)
assert callable(load_rc)
assert callable(initialise_combined_cspA_exp)
assert callable(create_exp_synthetic_comb)

print("OK imports (merge_rna)")
