# MERGE-RNA Examples

Self-contained demo script — no external data required.

## Quick Start

```bash
git clone https://github.com/giusSacco/MERGE-RNA.git
cd MERGE-RNA
conda create -f environment.yml
conda activate merge-rna
python examples/demo.py
```

## What the demo shows

`demo.py` runs a joint fit on three synthetic 30 nt sequences (<1 min on a laptop):

| Feature | Detail |
|---------|--------|
| Multi-system | 3 independent sequences fitted jointly |
| Concentration series | 0 mM (background) + 50 mM + 100 mM per system |
| Edge masking | First/last 5 positions excluded from the loss |
| Train/test split | seq_A + seq_B = training; seq_C = test |
| Fit mode | Sequential: Phase 1 physical params → Phase 2 lambda_sc |

## Output

Results are written to `examples/outputs/demo/demo/`:
- `params1D.txt` — fitted parameters
- `*_mut_profile.png` — observed vs. predicted mutation rates
- `loss_history.png` — convergence curve

## Paper fits

Scripts that reproduce the paper results (require real data and `RNA_STRUCT_HOME`) live in `fits_paper/`:

| Script | Description |
|--------|-------------|
| `fits_paper/cspA/scripts/run_cspA.py` | cspA mRNA — joint 10°C + 37°C fit |
| `fits_paper/structured_rnas/scripts/run_structured_rnas.py` | IVT structured RNAs (joint fit with validation) |
| `fits_paper/designed_sequence/scripts/run_synthetic_bistable.py` | Synthetic bistable RNA |
| `fits_paper/designed_sequence/scripts/run_experimental.py` | Designed bistable sequence (Turner default; `--rna-params andronescu` for Andronescu 2007) |
| `fits_paper/adenine_riboswitch/run_fit.py` | Olson et al. adenine riboswitch (two-phase fit) |
