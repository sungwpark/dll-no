# Generative Neural Operators through Diffusion Last Layer

Official implementation for the ICML 2026 paper **Generative Neural Operators through Diffusion Last Layer**.

**Authors: Sungwon Park, Anthony Zhou, Hongjoong Kim, Amir Barati Farimani**

This repository contains PyTorch Lightning and Hydra implementations of:

- `DLL`: Diffusion Last Layer for probabilistic neural operators
- deterministic neural-operator baselines
- MC-dropout baselines
- probabilistic neural operator (`PNO`) baselines
- pixel-space diffusion (`DM`) baselines
- latent diffusion (`LDM`) baselines
- dataset generation code for the benchmarks used in the paper

## What This Repo Covers

The main paper experiments in the current codebase are:

- stochastic Burgers
- stochastic Darcy flow
- Kuramoto-Sivashinsky rollout forecasting
- Kolmogorov flow rollout forecasting

In the paper, DLL models an input-conditioned distribution over a compact coefficient representation learned by an operator encoder. In the code, this is implemented as a two-stage pipeline:

1. train an operator encoder that reconstructs outputs from input-conditioned features and learned coefficients
2. train a conditional diffusion model on those coefficients

## Repository Layout

```text
conf/
  config.yaml                  # Hydra root config
  dataset/                     # Dataset generation configs
  model/                       # Backbone / diffusion / encoder configs
  optimizer/                   # Optimizer configs
  scheduler/                   # Scheduler configs
  task/                        # Task-level experiment configs
src/
  data/                        # Dataset builders and dataloaders
  models/                      # FNO, U-Net, encoder, diffusion modules
  train/                       # Training and evaluation entrypoints
  main.py                      # Hydra entrypoint
```

## Environment Setup

The code assumes you run commands from the repository root.

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

A typical Conda setup is:

```bash
conda create -n dll python=3.12
conda activate dll
pip install torch lightning hydra-core numpy matplotlib tqdm wandb neuralop
```

Optional dataset-generation dependencies:

```bash
pip install apebench jax jaxlib
```

Notes:

- `hydra-core` already installs `omegaconf` as a dependency.
- `wandb` is required by the training scripts for logging.
- `neuralop` is required for the FNO backbones and embedding modules.
- `apebench` is required for the KS and Kolmogorov-flow datasets.
- `jax` and `jaxlib` are required for the stochastic Burgers and stochastic Darcy dataset generators.
- Install the CUDA-specific PyTorch build appropriate for your machine.
- If you do not want online logging, set `WANDB_MODE=offline` or `WANDB_MODE=disabled` before running experiments.

## Running Experiments

All runs use the Hydra entrypoint:

```bash
python src/main.py task=<task_config> dataset=<dataset_config>
```

### 1. Generate datasets

Stochastic operator-learning datasets:

```bash
python src/main.py task=generate_dataset dataset=sburgers
python src/main.py task=generate_dataset dataset=sdarcy
```

Deterministic rollout datasets:

```bash
python src/main.py task=generate_dataset dataset=ks
python src/main.py task=generate_dataset dataset=kmflow
```

Generated datasets are cached under `datasets/` by default.

### 2. Train the main paper models

DLL:

```bash
python src/main.py task=sburgers/dll dataset=sburgers
python src/main.py task=sdarcy/dll dataset=sdarcy
python src/main.py task=ks/dll dataset=ks
python src/main.py task=kmflow/dll dataset=kmflow
```

Deterministic FNO baselines:

```bash
python src/main.py task=sburgers/regression dataset=sburgers
python src/main.py task=sdarcy/regression dataset=sdarcy
python src/main.py task=ks/regression dataset=ks
python src/main.py task=kmflow/regression dataset=kmflow
```

MC-dropout baselines:

```bash
python src/main.py task=sburgers/dropout dataset=sburgers
python src/main.py task=sdarcy/dropout dataset=sdarcy
python src/main.py task=ks/dropout dataset=ks
python src/main.py task=kmflow/dropout dataset=kmflow
```

PNO baselines:

```bash
python src/main.py task=sburgers/pno dataset=sburgers
python src/main.py task=sdarcy/pno dataset=sdarcy
python src/main.py task=ks/pno dataset=ks
python src/main.py task=kmflow/pno dataset=kmflow
```

Diffusion baselines:

```bash
python src/main.py task=sburgers/dm dataset=sburgers
python src/main.py task=sdarcy/dm dataset=sdarcy
python src/main.py task=ks/dm dataset=ks
python src/main.py task=kmflow/dm dataset=kmflow

python src/main.py task=sburgers/ldm dataset=sburgers
python src/main.py task=sdarcy/ldm dataset=sdarcy
python src/main.py task=ks/ldm dataset=ks
python src/main.py task=kmflow/ldm dataset=kmflow
```

## Configuration Notes

- The root Hydra config is [`conf/config.yaml`](conf/config.yaml).
- Each dataset-specific task under `conf/task/<dataset>/` selects the model family and evaluation mode.
- DLL uses two stages:
  - `training_oe`: operator-encoder training
  - `training_dll`: coefficient-space diffusion training
- Common evaluation modes are `stochastic` and `rollout`.

## Outputs

By default the code writes:

- cached datasets to `datasets/`
- checkpoints to `checkpoints/<project>/<run_name>/`
- W&B logs to `wandb_logs/`


## Citation

```bibtex
@inproceedings{park2026generative,
  title     = {Generative Neural Operators through Diffusion Last Layer},
  author    = {Park, Sungwon and Zhou, Anthony and Kim, Hongjoong and Barati Farimani, Amir},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```