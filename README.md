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

The current implementation covers steps 1-8. The repo now includes an operator-level benchmarking path for image normalization across CPU PyTorch, PyTorch CUDA, a custom CUDA extension, and Triton.

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
- deep learning models:
  - custom CNN built from scratch in PyTorch
  - ResNet18 with optional pretrained weights and backbone freezing
- GPU-aware PyTorch training with automatic device selection:
  - CUDA when available
  - MPS fallback on Apple Silicon
  - CPU fallback otherwise
- operator-level GPU optimization:
  - custom CUDA kernel for NCHW image normalization
  - matching Triton kernel for the same operation
  - benchmark harness that validates correctness and compares latency/bandwidth
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
  - kernel benchmark CSV and JSON reports
  - confusion-matrix plots
  - training-history curves
  - kernel benchmark comparison plots
  - serialized sklearn models
  - best-model PyTorch checkpoints

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

If you want the GPU benchmarking extras on a Linux machine with NVIDIA CUDA:

```bash
pip install -e ".[gpu]"
```

The custom CUDA extension also requires a working CUDA toolkit with `nvcc` available through `CUDA_HOME`.

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

Train the custom CNN:

```bash
python3 main.py train-pytorch --dataset fashion_mnist --model simple_cnn --epochs 10
```

Train ResNet18 with pretrained weights:

```bash
python3 main.py train-pytorch --dataset cifar10 --model resnet18 --use-pretrained --epochs 12
```

Freeze the ResNet18 backbone for a fast transfer-learning baseline:

```bash
python3 main.py train-pytorch --dataset cifar10 --model resnet18 --use-pretrained --freeze-backbone --epochs 8
```

Run the image-normalization benchmark:

```bash
python3 main.py benchmark-kernels --operation image_normalization
```

Use a smaller tensor while you are iterating locally:

```bash
python3 main.py benchmark-kernels --batch-size 16 --channels 3 --height 64 --width 64 --benchmark-iterations 20
```

## Outputs

Artifacts are written under `outputs/`:

- `outputs/metrics/classical_summary.csv`
- `outputs/metrics/classical_results.json`
- `outputs/metrics/*_history.csv`
- `outputs/metrics/*_summary.csv`
- `outputs/metrics/*_results.json`
- `outputs/metrics/image_normalization_*_benchmark.csv`
- `outputs/metrics/image_normalization_*_benchmark.json`
- `outputs/figures/*_confusion_matrix.png`
- `outputs/figures/*_training_curves.png`
- `outputs/figures/image_normalization_*_benchmark.png`
- `outputs/models/*.joblib`
- `outputs/models/*.pt`

## Architecture Overview

### Data Layer

`data/dataset_manager.py` handles dataset download, split creation, split caching, flattening for scikit-learn, train/eval transforms, and PyTorch DataLoader creation with optional stratified subsampling.

### Model Layer

`models/classical_models.py` centralizes sklearn estimator construction, and `models/pytorch_models.py` exposes the custom CNN plus ResNet18 adaptation logic for grayscale or RGB datasets.

### Training Layer

`training/classical_pipeline.py` runs the sklearn baselines, while `training/pytorch_pipeline.py` handles GPU-aware training, validation-based model selection, checkpointing, and final test evaluation.

### Evaluation Layer

`evaluation/metrics.py` keeps metrics logic in one place, which avoids scattered scoring code and makes later model comparisons much easier.

### GPU Kernel Layer

`cuda_kernels/image_normalization.py` loads a custom CUDA extension for per-channel image normalization, while `triton_kernels/image_normalization.py` implements the same NCHW operation in Triton for side-by-side benchmarking.

### Benchmark Layer

`benchmarking/image_normalization_benchmark.py` generates synthetic image tensors, validates each backend against the PyTorch reference implementation, times warmup and measured iterations, computes effective bandwidth, and saves CSV, JSON, and plot artifacts.

## Notes About GPU Work

The classical and PyTorch milestones work on CPU-only machines. The kernel benchmark command also works on CPU-only machines, but it will mark the CUDA and Triton backends as skipped until you run it on a CUDA-capable NVIDIA environment. Apple Silicon MPS helps for model training, but it does not replace CUDA for the custom kernel and Triton parts.

## Next Steps

1. Optionally add a second optimized operator such as matrix multiplication or depthwise convolution.
2. Integrate benchmark findings into the training pipeline summary report.
3. Optionally expose inference through FastAPI or Streamlit.

