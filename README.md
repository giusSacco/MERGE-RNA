# MERGE-RNA

**M**ulti-system **E**nsemble **R**efinement via **G**eneralizable parameters **E**stimation

[MERGE-RNA (Sacco et al., 2025)](https://arxiv.org/abs/2512.20581) predicts the RNA secondary structure ensemble that best explains chemical probing data across a joint analysis of data from multiple sequences, replicates, and probe concentrations in a single unified fit.

By explicitly modeling the underlying physics, the model learns shared parameters that tie concentration-dependent pairing probabilities to observed mutation rates.
Our model employs a maximum-entropy principle to predict thermodynamic populations, introducing only the minimal sequence-specific adjustments necessary to align the ensemble with experimental data.

## Installation

### Prerequisites

- Python ≥ 3.9
- ViennaRNA ≥ 2.7 (with Python bindings)
- Optional: Jupyter (for notebook execution)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/giusSacco/MERGE-RNA.git
   cd MERGE-RNA
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate merge-rna
   ```

3. **Set the environment variable (needed for paper fits and info.txt-based datasets):**
   ```bash
   export RNA_STRUCT_HOME=$(pwd)
   # or: export MERGE_RNA_DATA=/path/to/data_root
   ```

If you only run the demo, you can skip this. Some scripts auto-detect the repo root,
but setting `RNA_STRUCT_HOME` avoids unresolved `$RNA_STRUCT_HOME` paths in `info.txt`.


## Quick Start

```bash
python examples/demo.py
```

### What the demo shows

`demo.py` runs a simple joint fit on three synthetic 30 nt sequences:

| Feature | Detail |
|---------|--------|
| Multi-system | 3 independent sequences fitted jointly |
| Concentration series | 0 mM (background) + 50 mM + 100 mM per system |
| Edge masking | First/last 5 positions excluded from the loss |
| Train/test split | seq_A + seq_B = training; seq_C = test |
| Fit mode | Sequential: Phase 1 physical params → Phase 2 lambda_sc |

## Paper Fits

Scripts that reproduce the fits reported in the paper live in [fits_paper/](fits_paper/). They use the datasets in `data/` and resolve
paths via `RNA_STRUCT_HOME` (or `MERGE_RNA_DATA` if the data folder lives elsewhere).

You can reproduce the analysis we presented for the V. vulnificus add adenine riboswitch (expect <1 hour on CPU) on data from [Olson et al. 2023](https://www.cell.com/molecular-cell/abstract/S1097-2765(22)00114-9) :

```bash
python fits_paper/adenine_riboswitch/run_fit.py --phase both
```

Other systems (briefly):

- `fits_paper/structured_rnas/scripts/run_structured_rnas.py` — joint fit of five structured RNAs. Original raw data from [Bohn et al. 2023](https://www.nature.com/articles/s41592-023-01862-7).
- `fits_paper/cspA/scripts/run_cspA.py` — cspA 10C/37C joint fit. Original unaligned data from [Zhang et al. 2018](https://www.cell.com/molecular-cell/abstract/S1097-2765(18)30180-1), processed as described in the manuscript to obtain the mutation-rate made available here.
- `fits_paper/designed_sequence/scripts/run_experimental.py` — designed bistable RNA (experimental). Original data from [Sacco et al. 2025](https://arxiv.org/abs/2512.20581).
- `fits_paper/designed_sequence/scripts/run_synthetic_bistable.py` — synthetic bistable control.

See [fits_paper/README.md](fits_paper/README.md) for the workflow and script list.

## Notebooks

The notebooks in [notebooks/](notebooks/) reproduce the manuscript figures (structured RNAs, adenine riboswitch, cspA, and designed sequence). They read cached data from `data/cache/<notebook>/`.

To regenerate the results plotted in the figures (requires ViennaRNA):

```bash
python scripts/analysis/_regenerate_cache_nb<N>.py
```

To re-export a notebook from the source `.py` file:

```bash
jupyter nbconvert --to notebook --execute scripts/analysis/nb<N>_<name>.py --output notebooks/nb<N>_<name>.ipynb
```

or directly edit the `.ipynb` file in Jupyter and run all cells.

## Core Components

- **merge_rna/experiment.py** – `Experiment` class for loading data, attaching experiment metadata, and synthetic dataset helpers.
- **merge_rna/fit.py** – `ExperimentFit`, `System`, and `MultiSystemsFit` classes for mutation-rate prediction, gradients, soft-constraint handling, and optimisation orchestration.

## Citation

If you use MERGE-RNA, please cite the paper:

> G. Sacco, J. Li, R.P. Smyth, G. Sanguinetti, G. Bussi.  
> **MERGE-RNA: a physics-based model to predict RNA secondary structure ensembles with chemical probing**  
> arXiv:2512.20581, 2025.  
> [https://arxiv.org/abs/2512.20581](https://arxiv.org/abs/2512.20581)

```bibtex
@article{sacco2025mergeRNA,
  title   = {{MERGE-RNA}: a physics-based model to predict {RNA} secondary structure ensembles with chemical probing},
  author  = {Sacco, Giuseppe and Li, Jianhui and Smyth, Redmond P. and Sanguinetti, Guido and Bussi, Giovanni},
  journal = {arXiv preprint arXiv:2512.20581},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.20581},
  doi     = {10.48550/arXiv.2512.20581}
}
```

A `CITATION.cff` file is also provided for use with GitHub's *Cite this repository* button.

## License

This repository is released under the terms of the [LICENSE](LICENSE) file.
