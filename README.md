## Learning with Differentially Private (Sliced) Wasserstein Gradients

This repository contains code to reproduce the experiments from the article *Learning with Differentially Private Sliced Wasserstein Gradients*.

#### Folder Structure

To run the experiments, download the entire folder to your local machine. The folder includes:

#### Notebooks

- `notebook_swae.ipynb`: Main experiments using the Sliced Wasserstein Autoencoder.
- `notebook_AccFid.ipynb`: Compute the Accuracy and FID in the SWAE data generation experiment. 
- `notebook_appendix.ipynb`: Additional experiments as described in the appendix (fairness and private matching).
- `notebook_baseline.ipynb`: Baseline in the distribution matching experiment.
- `notebook_regression.ipynb`: 1d private and fair regression, comparison with Xian et al. (2024).

#### Supporting Files

- `accountants/`: Functions for moment accounting, adapted from Meta's [Opacus](https://github.com/pytorch/opacus) (Apache 2.0 License). Required by both notebooks.
- `bias_measure_fcts.py`: Fairness metrics from Confidence Intervals for Testing Disparate Impact in Fair Learning
 (https://arxiv.org/abs/1807.06362). Used in `notebook_appendix.ipynb`.
- `aux_swae.py`: Helper functions used in `notebook_swae.ipynb`.
- `aux_appendix.py`: Helper functions used in `notebook_appendix.ipynb`.

#### Reproducibility

Each notebook has been precompiled to show the results that should be obtained. To facilitate reproducibility, each notebook lists all dependencies installed at the time of compilation.  To rerun the experiments:

1. Ensure all required libraries are installed (imports listed at the top of each notebook).
2. If version issues arise, check the `!pip freeze` output included at the start of each notebook. It lists the package versions used during the original runs.
