# Notebooks

The `.ipynb` files here are executed notebooks that reproduce the manuscript figures, one for each system analyzed in the paper:
- set of structured RNAs
- adenine riboswitch
- cspA at 10°C and 37°C
- designed sequence (synthetic data and new experimental data)

Each notebook is generated from the corresponding `.py` source in `scripts/analysis/` to aid versioning and reads
pre-computed data from `data/cache/<notebook>/`.

To regenerate the cache (requires ViennaRNA and local fit parameters in `fits_paper/`):

```bash
python scripts/analysis/_regenerate_cache_nb<N>.py
```

To re-export a notebook from the source `.py` file:

```bash
jupyter nbconvert --to notebook --execute scripts/analysis/nb<N>_<name>.py --output notebooks/nb<N>_<name>.ipynb
```
or directly edit the `.ipynb` file in Jupyter and run all cells.