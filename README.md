# MERGE-RNA
[MERGE-RNA](https://arxiv.org/abs/2512.20581) models chemical-probing mutation profiles for RNA constructs across different reagents and concentrations.
The toolkit fits shared physical parameters and sequence-specific soft constraints to explain observed mutation rates.
By explicitly modeling the underlying physics, it enables the joint analysis of data from multiple sequences, replicates, and probe concentrations in a single unified fit.
The soft constraints capture sequence-specific corrections to the thermodynamic model, allowing the computation of ensemble population of secondary structures that best explain the data.

# Reference
For the scientific background and detailed methodology, see the original work: [MERGE-RNA: a physics-based model to predict RNA secondary structure ensembles with chemical probing](https://arxiv.org/abs/2512.20581).

## Work in progress
MERGE-RNA models chemical-probing mutation profiles for RNA constructs across different reagents and concentrations. The toolkit fits shared physical parameters and sequence-specific soft constraints to explain observed mutation rates. By explicitly modeling the underlying physics, it enables the joint analysis of data from multiple sequences, replicates, and probe concentrations in a single unified fit.

## Core Components
- **class_experiment.py** – Defines the `Experiment` class. Loads `.rc` count files, normalises coverage, attaches metadata, and provides synthetic dataset helpers.
- **class_experimentfit.py** – Extends `Experiment` with mutation-rate prediction, gradients, and soft-constraint handling; hosts `System` and `MultiSystemsFit` orchestration.
- **run_multiple_fits.py** – Command-line entry point for launching pre-defined multi-system optimisation sweeps.

## License
This repository is released under the terms of the [LICENSE](LICENSE) file.
