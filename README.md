# Trustworthy Decentralized Learning under Non-IID Data and Byzantine Participants

This project develops a reproducible PyTorch framework for studying how heterogeneous data and unreliable participants affect decentralized learning, with an emphasis on transparent experiments suitable for research and extension.

## Current scope

The current codebase provides a small Fashion-MNIST convolutional model, a reproducible centralized training baseline, and reproducible Dirichlet client partitioning of training data. Federated and decentralized learning, Byzantine behavior, and robust peer selection are outside the current scope.

> **Active development:** This repository is being built milestone by milestone and does not yet contain final research results.

## Installation

Create a Python 3.11 or newer virtual environment, then install the project and test dependencies:

```bash
python -m pip install -e ".[dev]"
```

## GPU setup

The centralized configuration uses `device: auto`, which automatically selects CUDA when PyTorch detects a supported NVIDIA GPU. Install an appropriate CUDA-enabled PyTorch wheel inside the project virtual environment; use the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to choose a build for your hardware and driver.

This development machine was verified with the official `cu130` wheel:

```bash
python -m pip uninstall torch torchvision -y
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev]"
```

The `cu130` build is the tested choice for this machine, not a requirement for every NVIDIA GPU. The CUDA version shown by `nvidia-smi` indicates driver compatibility and does not need to exactly match `torch.version.cuda`. Normal training with PyTorch binaries uses their bundled CUDA runtime and the installed NVIDIA driver; the full CUDA Toolkit and `nvcc` are not required.

Verify GPU detection, then run the existing development baseline:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
python -m tdl.training.centralized --config configs/centralized.yaml
```

GPU verification should report `True` for CUDA availability and the detected GPU name. Keep `device: auto`; the baseline should print `Device: cuda` when CUDA is available.

## Inspect client partitions

Partition Fashion-MNIST training labels without training a model:

```bash
python -m tdl.data.inspect_partition --config configs/partition.yaml
```

The default is 10 clients, seed 42, alpha 0.3, and at least one sample per client. The command prints per-client sample and class counts, verifies exact coverage and repeatability, and writes `results/partition_summary.json`. Use `--alpha` and `--output` to inspect other concentrations and retain separate local JSON summaries. Generated results and datasets remain ignored by Git.

Each class receives an independent symmetric Dirichlet allocation. Larger alpha tends toward similar class allocations; smaller alpha favors label skew. Client sizes are not fixed. Allocations that violate the minimum are rejected using the same seeded RNG, with a default limit of 1000 attempts and a clear error if none succeeds. The JSON also reports mean label total variation from the global distribution (smaller values indicate more similar label proportions).
