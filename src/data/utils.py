from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import TensorDataset

XYFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def split_test_trajectories_in_half(
    test_trjs: torch.Tensor,
    *,
    seed: int,
    name: str = "test_trjs",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically shuffle trajectories and split into (val, test) halves."""
    n_total = int(test_trjs.shape[0])
    half = n_total // 2
    if half == 0:
        raise ValueError(
            f"Need at least 2 trajectories to split, got {n_total} for {name}."
        )

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=gen)
    shuffled = test_trjs[perm]
    return shuffled[:half], shuffled[half:]


def try_load_cached_traj_dataset(
    *,
    cache_path: Path,
    overwrite: bool,
    verbose: bool,
    log_prefix: str,
    return_trajectories: bool,
    to_xy: XYFn,
) -> (
    None
    | tuple[TensorDataset, TensorDataset, TensorDataset]
    | tuple[tuple[TensorDataset, TensorDataset, TensorDataset], dict[str, torch.Tensor]]
):
    """Load cached trajectory dataset (trajectory-only cache format).

    Cache format supported:
    - {"train_trjs": ..., "val_trjs": ..., "test_trjs": ..., "meta": ...}

    Returns:
    - None if cache does not exist, overwrite=True, or does not satisfy requested format.
    - (train_set, val_set, test_set) if usable and return_trajectories=False.
    - ((train_set, val_set, test_set), {"val_trjs": ..., "test_trjs": ...}) if return_trajectories=True.
    """
    if not cache_path.exists() or overwrite:
        return None

    if verbose:
        print(f"[{log_prefix}] Loading cached dataset from: {cache_path}")

    payload: dict[str, Any] = torch.load(cache_path, map_location="cpu")
    has_trjs = all(k in payload for k in ("train_trjs", "val_trjs", "test_trjs"))
    if not has_trjs:
        if verbose:
            print(
                f"[{log_prefix}] Cache missing required trajectory keys, regenerating..."
            )
        return None

    train_trjs = payload["train_trjs"]
    val_trjs = payload["val_trjs"]
    test_trjs = payload["test_trjs"]

    X_train, Y_train = to_xy(train_trjs)
    X_val, Y_val = to_xy(val_trjs)
    X_test, Y_test = to_xy(test_trjs)

    datasets = (
        TensorDataset(X_train, Y_train),
        TensorDataset(X_val, Y_val),
        TensorDataset(X_test, Y_test),
    )

    if return_trajectories:
        trajectories = {"val_trjs": val_trjs, "test_trjs": test_trjs}
        return datasets, trajectories

    return datasets
