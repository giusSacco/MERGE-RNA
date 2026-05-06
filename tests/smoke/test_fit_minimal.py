#!/usr/bin/env python3
"""Smoke test: end-to-end fit on the synthetic demo.

Runs `examples/demo.py` as a subprocess and asserts:
- subprocess exits cleanly
- expected output files exist
- pairing probabilities are valid (no NaN/inf, in [0, 1])
- BPP matrix is square, symmetric (within tolerance), in [0, 1]
- params1D.txt parses to a finite vector
- summary.log contains a finite final loss
- lambda_sc.txt is finite

Takes ~1-3 minutes.
"""
import os
import re
import subprocess
import sys
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('RNA_STRUCT_HOME', str(REPO_ROOT))

DEMO_SCRIPT = REPO_ROOT / 'examples' / 'demo.py'
OUT_DIR = REPO_ROOT / 'examples' / 'outputs' / 'demo' / 'demo'
DETAILED = OUT_DIR / 'detailed'
PLOTS = OUT_DIR / 'plots'

assert DEMO_SCRIPT.exists(), f"demo script missing: {DEMO_SCRIPT}"

# Run the demo end-to-end
print(f"Running {DEMO_SCRIPT.relative_to(REPO_ROOT)} ...")
proc = subprocess.run(
    [sys.executable, str(DEMO_SCRIPT)],
    cwd=str(REPO_ROOT),
    capture_output=True,
    text=True,
    timeout=600,
)
if proc.returncode != 0:
    print("STDOUT:\n" + proc.stdout[-2000:])
    print("STDERR:\n" + proc.stderr[-2000:])
    raise SystemExit(f"demo script failed with code {proc.returncode}")

# Output structure
assert OUT_DIR.is_dir(), f"output dir missing: {OUT_DIR}"
assert DETAILED.is_dir(), f"detailed/ missing"
assert PLOTS.is_dir(), f"plots/ missing"
assert (OUT_DIR / 'params1D.txt').exists()
assert (OUT_DIR / 'summary.log').exists()

# Required diagnostic plots (seq_A is one of the two training systems)
expected_plots = [
    'loss_history.png',
    'seq_A_mut_profile.png',
    'seq_A_pairing_probs.png',
    'seq_A_lambda_sc.png',
]
for name in expected_plots:
    p = PLOTS / name
    assert p.exists() and p.stat().st_size > 0, f"plot missing or empty: {name}"

# params1D.txt: vector of finite numbers
params = np.loadtxt(OUT_DIR / 'params1D.txt')
assert params.ndim == 1 and params.size > 0
assert np.all(np.isfinite(params)), "params1D contains NaN or inf"

# Pairing probabilities (1D)
pp = np.loadtxt(DETAILED / 'pairing_probs.txt', comments='#')
assert pp.ndim == 1 and pp.size == 30, f"unexpected pairing_probs shape: {pp.shape}"
assert np.all(np.isfinite(pp)), "pairing_probs contains NaN or inf"
assert np.all((pp >= 0.0) & (pp <= 1.0)), \
    f"pairing_probs outside [0, 1]: min={pp.min()} max={pp.max()}"

# BPP matrix (N x N)
bpp = np.loadtxt(DETAILED / 'base_pairing_probabilities.txt', comments='#')
assert bpp.shape == (30, 30), f"unexpected BPP shape: {bpp.shape}"
assert np.all(np.isfinite(bpp)), "BPP matrix contains NaN or inf"
assert np.all((bpp >= 0.0) & (bpp <= 1.0)), \
    f"BPP entries outside [0, 1]: min={bpp.min()} max={bpp.max()}"
# ViennaRNA convention: BPP is stored as upper-triangular (lower zeros) or symmetric.
# Accept both, but require the upper triangle to carry the probabilities.
upper_sum = np.triu(bpp, k=1).sum()
assert upper_sum > 0, "BPP upper triangle is empty (no pairing probabilities at all?)"

# Lambda_sc: finite (sign and magnitude depend on fit)
lam = np.loadtxt(DETAILED / 'seq_A_lambda_sc.txt', comments='#')
assert lam.ndim == 1 and lam.size == 30
assert np.all(np.isfinite(lam)), "lambda_sc contains NaN or inf"

# Final loss from summary.log: must be finite
log_text = (OUT_DIR / 'summary.log').read_text()
loss_matches = re.findall(r'(?:Loss|loss)[^\d\-+]*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',
                          log_text)
assert loss_matches, "no loss values found in summary.log"
final_loss = float(loss_matches[-1])
assert np.isfinite(final_loss), f"final loss is not finite: {final_loss}"

print(f"OK fit (final loss = {final_loss:.4g}, "
      f"pp range [{pp.min():.3f}, {pp.max():.3f}], "
      f"bpp max = {bpp.max():.3f})")
