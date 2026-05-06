# fits_paper

Scripts that reproduce the fits reported in the paper. 

```bash
cd $RNA_STRUCT_HOME
conda activate merge-rna
python fits_paper/<system>/scripts/<script>.py [flags]
```

All scripts accept `--rna-params turner|andronescu` (default: Turner 2004). Outputs are written to `fits_paper/<system>/outputs/`.

## Adenine riboswitch — `adenine_riboswitch/run_fit.py`

Two-phase fit of the [Olson et al. (2022)](https://www.cell.com/molecular-cell/abstract/S1097-2765(22)00114-9) adenine riboswitch DMS-MaP (175 nt, 8 adenine concentrations).

**Phase 1** (`--phase 1`): fits shared physical parameters using the ethanol control + ade=0 µM experiment. Runs in `linear_mode` (constrains µ_r ≤ 0, fixes p_b = 0) and applies a custom mask to reproduce the protocol of the paper.

**Phase 2** (`--phase 2`): fixes physical parameters from phase 1 and fits per-concentration λ_sc independently, parallelised over adenine concentrations.

---

## Structured RNAs — `structured_rnas/scripts/run_structured_rnas.py`

Joint fit of five RNAs (hc16, bact_RNaseP_typeA, tetrahymena_ribozyme, HCV_IRES, V_chol_gly_riboswitch). DMS-MaP performed by [Bohn et al. 2023](https://www.nature.com/articles/s41592-023-01862-7). Three systems train the shared physical parameters; two (hc16, bact_RNaseP_typeA) serve as validation. Uses `fit_mode='sequential'`: phase 1 learns shared parameters, phase 2 refines per-sequence λ_sc for each system independently.

---

## cspA — `cspA/scripts/run_cspA.py`

Joint fit of the cspA 5' UTR measured at 10 °C and 37 °C ([Zhang et al. 2018](https://www.cell.com/molecular-cell/abstract/S1097-2765(18)30180-1)).  The two experiments undergo mutation-rate normalisation because of different probing protocols, then fitted jointly with `fit_mode='sequential'`.

---

## Designed bistable sequence — `designed_sequence/scripts/run_experimental.py`

Two-phase fit of the experimentally measured bistable RNA ([Sacco et al. 2025](https://arxiv.org/abs/2512.20581)).

| Flag | Default | Effect |
|------|---------|--------|
| `--phase 1\|2\|both` | `both` | Run one or both phases |
| `--phase1-params PATH` | — | Required when `--phase 2` only |
| `--rna-params turner\|andronescu` | `turner` | Thermodynamic parameter set |

Also in this folder: `run_synthetic_bistable.py` fits on synthetic data generated for the same sequence with known gound-truth populations.

---


Key flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--phase 1\|2\|both` | `both` | Run one or both phases |
| `--phase1-params PATH` | — | Required when `--phase 2` only |
| `--no-custom-mask` | off | Disable 5′ cassette masking |
| `--no-linear` | off | Disable linear mode in phase 1 |
| `--workers N` | all CPUs | Parallelism for phase 2 |
| `--rna-params turner\|andronescu` | `turner` | Thermodynamic parameter set |
