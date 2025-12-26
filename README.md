# MERGE-RNA

**M**ulti-system **E**nsemble **R**efinement via **G**eneralizable parameters **E**stimation

MERGE-RNA models chemical-probing mutation profiles for RNA constructs across different reagents and concentrations.
The toolkit fits shared physical parameters and sequence-specific soft constraints to explain observed mutation rates.
By explicitly modeling the underlying physics, it enables the joint analysis of data from multiple sequences, replicates, and probe concentrations in a single unified fit.
The soft constraints capture sequence-specific corrections to the thermodynamic model, allowing the computation of ensemble populations of secondary structures that best explain the data.

## Installation

### Prerequisites

- Python ≥ 3.9
- ViennaRNA ≥ 2.7 (with Python bindings)

### Setup

1. **Create a conda environment with ViennaRNA:**
   ```bash
   conda create -n merge-rna -c conda-forge -c bioconda python=3.11 viennarna
   conda activate merge-rna
   ```

2. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/your-username/MERGE-RNA.git
   cd MERGE-RNA
   pip install -r requirements.txt
   ```

3. **Set the environment variable:**
   ```bash
   export RNA_STRUCT_HOME=$(pwd)
   ```
   Add this to your shell profile for persistence.

## Quick Start

Run any example from the `examples/` directory:

```bash
cd $RNA_STRUCT_HOME
python examples/demo_synthetic_random.py
```

This generates synthetic DMS-MaP data for a random 30-nt RNA and fits physical parameters to recover the true structure.

### Available Examples

| Script | Description |
|--------|-------------|
| `demo_synthetic_random.py` | Random 30-nt sequence with synthetic data |
| `preset_cspA_10C.py` | *E. coli* cspA mRNA at 10°C |
| `preset_cspA_37C.py` | *E. coli* cspA mRNA at 37°C |
| `preset_structured_rnas.py` | Joint fit of 5 structured RNAs (ribozymes, riboswitches) |
| `preset_synthetic_bistable.py` | Synthetic bistable RNA construct |
| `preset_experimental_bistable_turner.py` | Experimental bistable RNA (Turner parameters) |
| `preset_experimental_bistable_andronescu.py` | Experimental bistable RNA (Andronescu parameters) |

See [examples/README.md](examples/README.md) for details.

## Core Components

- **class_experiment.py** – `Experiment` class for loading `.rc` count files, normalising coverage, attaching metadata, and synthetic dataset helpers.
- **class_experimentfit.py** – `ExperimentFit`, `System`, and `MultiSystemsFit` classes for mutation-rate prediction, gradients, soft-constraint handling, and optimisation orchestration.

## Reference

If you use MERGE-RNA, please cite:

> G. Sacco, J. Li, R.P. Smyth, G. Sanguinetti, G. Bussi.  
> **MERGE-RNA: a physics-based model to predict RNA secondary structure ensembles with chemical probing**  
> arXiv preprint arXiv:2512.20581, 2025.  
> [https://arxiv.org/abs/2512.20581](https://arxiv.org/abs/2512.20581)

## License

This repository is released under the terms of the [LICENSE](LICENSE) file.
