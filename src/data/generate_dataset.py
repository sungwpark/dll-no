from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
from torch.utils.data import TensorDataset
from hydra.utils import instantiate
from omegaconf import DictConfig


@dataclass(frozen=True)
class XYMeta:
    X: torch.Tensor
    Y: torch.Tensor
    meta: Dict[str, Any]


def _split_xy(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    total_len = len(X)
    total_requested = n_train + n_val + n_test

    if total_requested > total_len:
        raise ValueError(
            f"Requested split ({n_train} train + {n_val} val + {n_test} test = {total_requested}) "
            f"exceeds total dataset size ({total_len})"
        )

    gen = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(total_len, generator=gen)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val : n_train + n_val + n_test]

    train_set = TensorDataset(X[train_indices], Y[train_indices])
    val_set = TensorDataset(X[val_indices], Y[val_indices])
    test_set = TensorDataset(X[test_indices], Y[test_indices])
    return train_set, val_set, test_set


def cached_xy_dataset(
    *,
    dataset_dir: str | Path,
    filename: str,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    save_data: bool,
    verbose: bool,
    overwrite: bool = False,
    generator: Callable[[], XYMeta],
    log_prefix: str = "dataset",
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Shared cache/load logic for datasets whose on-disk payload is:
      {"X": <tensor>, "Y": <tensor>, "meta": <dict>}
    """
    dataset_path = Path(dataset_dir).expanduser()
    dataset_path.mkdir(parents=True, exist_ok=True)
    file_path = dataset_path / filename

    if file_path.exists() and not overwrite:
        if verbose:
            print(f"[{log_prefix}] Loading existing dataset from {file_path}")
        payload = torch.load(file_path, map_location="cpu")
        X = payload["X"]
        Y = payload["Y"]
        return _split_xy(X, Y, seed=seed, n_train=n_train, n_val=n_val, n_test=n_test)

    out = generator()
    X, Y, meta = out.X, out.Y, out.meta

    if save_data:
        payload: Dict[str, Any] = {"X": X, "Y": Y, "meta": meta}
        torch.save(payload, file_path)
        if verbose:
            print(f"[{log_prefix}] Saved to {file_path} (Shape: {tuple(X.shape)})")

    return _split_xy(X, Y, seed=seed, n_train=n_train, n_val=n_val, n_test=n_test)


def generate_dataset(cfg: DictConfig) -> None:
    """Hydra entrypoint to generate and cache the configured dataset to disk.

    This is intentionally single-process / single-GPU (whatever CUDA_VISIBLE_DEVICES exposes).
    """
    instantiate(cfg.dataset)
