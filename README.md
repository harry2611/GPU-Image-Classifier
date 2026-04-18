# GPU-Accelerated Image Classification System

A production-quality, resume-ready project that combines classical machine learning, deep learning, and custom GPU kernels into one reproducible image-classification pipeline.

The project is being built in milestones so each layer stays understandable and maintainable:

1. Project setup
2. Dataset loading and preprocessing
3. Classical ML baselines
4. PyTorch CNN training
5. Pretrained ResNet18 fine-tuning
6. Custom CUDA kernel
7. Triton kernel
8. Benchmarking and comparison
9. Optional demo app

The first milestone implemented here covers steps 1-3.

## Why Fashion-MNIST First?

Fashion-MNIST is the default dataset because it is:

- fast to download and iterate on
- easy to flatten for scikit-learn baselines
- still realistic enough for CNN and benchmarking work

The code is structured so you can switch to `cifar10` from the CLI without changing the pipeline.

## Current Features

- deterministic dataset download and stratified train/validation/test handling
- reusable preprocessing utilities for both classical ML and future PyTorch training
- classical baselines:
  - Logistic Regression
  - Linear SVM
  - Random Forest
- evaluation outputs:
  - accuracy
  - precision
  - recall
  - F1-score
  - confusion matrix
  - ROC-AUC when the model exposes usable scores
- artifact generation:
  - JSON metrics
  - CSV summary tables
  - confusion-matrix plots
  - serialized sklearn models

## Project Structure

```text
project/
│── app/
│── benchmarking/
│── cuda_kernels/
│── data/
│   ├── raw/
│   ├── processed/
│   └── dataset_manager.py
│── evaluation/
│── models/
│── training/
│── triton_kernels/
│── utils/
│── outputs/
│── main.py
│── pyproject.toml
│── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If you want linting and tests as well:

```bash
pip install -e ".[dev]"
```

## Run The Current Milestone

Run the classical baseline pipeline on Fashion-MNIST:

```bash
python3 main.py classical-baselines --dataset fashion_mnist
```

For faster local iteration:

```bash
python3 main.py classical-baselines --dataset fashion_mnist --train-subset 12000 --test-subset 3000
```

To switch datasets:

```bash
python3 main.py classical-baselines --dataset cifar10 --train-subset 15000
```

## Outputs

Artifacts are written under `outputs/`:

- `outputs/metrics/classical_summary.csv`
- `outputs/metrics/classical_results.json`
- `outputs/figures/*_confusion_matrix.png`
- `outputs/models/*.joblib`

## Architecture Overview

### Data Layer

`data/dataset_manager.py` handles dataset download, split creation, split caching, flattening for scikit-learn, and PyTorch-ready transform helpers for the later CNN stages.

### Model Layer

`models/classical_models.py` centralizes estimator construction so the training loop stays clean and adding new baselines is low-risk.

### Training Layer

`training/classical_pipeline.py` runs fitting, validation, testing, artifact persistence, and benchmark-style timing for training and inference.

### Evaluation Layer

`evaluation/metrics.py` keeps metrics logic in one place, which avoids scattered scoring code and makes later model comparisons much easier.

## Notes About GPU Work

This milestone is portable and works on CPU-only machines. The CUDA and Triton milestones will require an NVIDIA CUDA environment to run end-to-end. The codebase will be structured to degrade gracefully when a compatible GPU is not available.

## Next Steps

1. Add a custom CNN in PyTorch with GPU-aware training.
2. Add a pretrained ResNet18 path.
3. Implement a CUDA kernel for image normalization or matrix multiplication.
4. Implement the same operation in Triton.
5. Add benchmark reports comparing CPU, PyTorch GPU, CUDA, and Triton.
6. Optionally expose inference through FastAPI or Streamlit.

