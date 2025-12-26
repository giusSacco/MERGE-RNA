# MERGE-RNA Examples

Standalone example scripts for running fits on different RNA systems.

## Quick Start

```bash
cd $RNA_STRUCT_HOME
conda activate vienna
python examples/demo_synthetic_random.py
```

## Available Presets

| Script | Description |
|--------|-------------|
| `demo_synthetic_random.py` | Random 30nt sequence with synthetic data |
| `preset_cspA_10C.py` | cspA mRNA at 10°C |
| `preset_cspA_37C.py` | cspA mRNA at 37°C |
| `preset_structured_rnas.py` | Redmond IVT systems (joint fit with validation) |
| `preset_synthetic_bistable.py` | Synthetic bistable RNA mixtures |
| `preset_experimental_bistable_turner.py` | newseq WT bistable with Turner (2004) params |
| `preset_experimental_bistable_andronescu.py` | newseq WT bistable with Andronescu (2007) params |

## Output

Each script creates its own output directory (e.g., `demo_synthetic_random/`, `preset_cspA_10C/`) containing:
- `params1D.txt` - Fitted parameters
- `*.log` - Fitting log
- `*_mut_profile.png` - Mutation rate profiles
- `*_pairing_probs.png` - Pairing probability comparisons
- `*_lambda_sc.png` - Inferred soft constraints
- `loss_history.png` - Loss during optimization

## Customizing

Each script follows the same pattern:
1. Load or create experiments
2. Configure `MultiSystemsFit`
3. Run `multi_sys.fit()`

Key options for `MultiSystemsFit`:
- `infer_1D_sc=True` - Infer position-specific soft constraints (lambda_sc)
- `fit_mode='sequential'` - Fit physical params first, then lambda_sc
- `do_plots=True` - Generate diagnostic plots
- `mask_edges=(left, right)` - Mask edge positions
