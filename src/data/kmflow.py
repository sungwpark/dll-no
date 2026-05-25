from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .utils import split_test_trajectories_in_half, try_load_cached_traj_dataset


def _to_torch_4d(trjs: Any, *, name: str) -> torch.Tensor:
    """Coerce trajectory container into float32 CPU tensor shaped (N_traj, T, H, W)."""
    if isinstance(trjs, dict):
        try:
            trjs = trjs["u"]  # common apebench convention
        except KeyError as e:
            raise TypeError(
                f"{name} is a dict, but no 'u' key was found. Keys={list(trjs.keys())}"
            ) from e

    if isinstance(trjs, torch.Tensor):
        t = trjs
    else:
        try:
            arr = np.asarray(trjs)
            if not arr.flags.writeable:
                arr = arr.copy()
            t = torch.from_numpy(arr)
        except Exception:
            try:
                t = torch.as_tensor(trjs)
            except Exception as e:
                raise TypeError(
                    f"Could not convert {name} (type={type(trjs)}) to torch.Tensor: {e}"
                ) from e
    if t.ndim == 4:
        pass  # Already correct shape
    elif t.ndim == 5 and t.shape[2] == 1:
        t = t.squeeze(2)
    else:
        raise ValueError(
            f"Expected {name} to have shape (N_traj, T, H, W); got {tuple(t.shape)}"
        )

    return t.to(device="cpu", dtype=torch.float32)


def _trjs_to_xy(trjs_4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert trajectories (N_traj, T, H, W) into next-step pairs (N_traj*(T-1), 1, H, W)."""
    x, y = trjs_4d[:, :-1, :, :], trjs_4d[:, 1:, :, :]
    h, w = int(trjs_4d.shape[-2]), int(trjs_4d.shape[-1])
    x = x.reshape(x.shape[0] * x.shape[1], 1, h, w).contiguous()
    y = y.reshape(y.shape[0] * y.shape[1], 1, h, w).contiguous()
    return x, y


def build_kmflow_dataset(
    dataset_dir: str = "datasets",
    filename: str = "kmflow.pt",
    save_data: bool = False,
    verbose: bool = True,
    overwrite: bool = False,
    data_name: str = "KMFlow",
    return_trajectories: bool = False,
    num_spatial_dims: int = 2,
    num_points: int = 128,
    dt: float = 1.0,
    num_substeps: int = 100,
    num_train_samples: int = 256,
    train_temporal_horizon: int = 50,
    num_test_samples: int = 64,
    test_temporal_horizon: int = 100,
    num_warmup_steps: int = 100,
    train_seed: int = 0,
    test_seed: int = 773,
    **_: Any,  # ignore unknown Hydra keys
) -> (
    tuple[TensorDataset, TensorDataset, TensorDataset]
    | tuple[tuple[TensorDataset, TensorDataset, TensorDataset], dict[str, torch.Tensor]]
):
    """
    KolmogorovFlow dataset builder backed by `apebench`.

    Behavior (as requested):
    - Train set comes from `sc.get_train_data()`
    - Validation set is **half** of `sc.get_test_data()`
    - Test set is the remaining half of `sc.get_test_data()`

    Output:
    - If `return_trajectories=False`: Returns (train_set, val_set, test_set) where each is a `TensorDataset(X, Y)`
      with X/Y built from next-step pairs along time.
    - If `return_trajectories=True`: Returns ((train_set, val_set, test_set), trajectories) where
      trajectories is a dict with keys 'val_trjs' and 'test_trjs' containing the full trajectory tensors.
    - If `save_data=True`, writes a `.pt` with keys: train_trjs/val_trjs/test_trjs/meta
    """
    dataset_path = Path(dataset_dir).expanduser()
    dataset_path.mkdir(parents=True, exist_ok=True)
    cache_path = dataset_path / filename

    cached = try_load_cached_traj_dataset(
        cache_path=cache_path,
        overwrite=overwrite,
        verbose=verbose,
        log_prefix="build_kmflow_dataset",
        return_trajectories=return_trajectories,
        to_xy=_trjs_to_xy,
    )
    if cached is not None:
        return cached
    if num_spatial_dims != 2:
        raise ValueError("KolmogorovFlow requires num_spatial_dims=2")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if num_substeps <= 0:
        raise ValueError("num_substeps must be positive")
    if num_train_samples <= 0:
        raise ValueError("num_train_samples must be positive")
    if train_temporal_horizon <= 0:
        raise ValueError("train_temporal_horizon must be positive")
    if num_test_samples <= 0:
        raise ValueError("num_test_samples must be positive")
    if test_temporal_horizon <= 0:
        raise ValueError("test_temporal_horizon must be positive")

    try:
        import apebench  # type: ignore
    except Exception as e:
        raise ModuleNotFoundError(
            "Missing dependency 'apebench'. Install it (e.g. `pip install apebench`) "
            "or add it to your environment before generating KolmogorovFlow datasets."
        ) from e
    sc = apebench.scenarios.physical.KolmogorovFlow(
        num_spatial_dims=num_spatial_dims,
        num_points=num_points,
        dt=dt,
        num_substeps=num_substeps,
        num_train_samples=num_train_samples,
        train_temporal_horizon=train_temporal_horizon,
        num_test_samples=num_test_samples,
        test_temporal_horizon=test_temporal_horizon,
        num_warmup_steps=num_warmup_steps,
        train_seed=train_seed,
        test_seed=test_seed,
    )

    train_trjs = _to_torch_4d(sc.get_train_data(), name="train_trjs")
    test_trjs = _to_torch_4d(sc.get_test_data(), name="test_trjs")

    val_trjs, test_trjs2 = split_test_trajectories_in_half(test_trjs, seed=test_seed)

    X_train, Y_train = _trjs_to_xy(train_trjs)
    X_val, Y_val = _trjs_to_xy(val_trjs)
    X_test, Y_test = _trjs_to_xy(test_trjs2)
    shapes = {
        k: tuple(v.shape)
        for k, v in {
            "train_trjs": train_trjs,
            "val_trjs": val_trjs,
            "test_trjs": test_trjs2,
        }.items()
    }

    meta: dict[str, Any] = {
        "data_name": data_name,
        "backend": "apebench",
        "num_spatial_dims": num_spatial_dims,
        "num_points": num_points,
        "dt": dt,
        "num_substeps": num_substeps,
        "dt_sim": dt / num_substeps,
        "num_train_samples": num_train_samples,
        "train_temporal_horizon": train_temporal_horizon,
        "num_test_samples": num_test_samples,
        "test_temporal_horizon": test_temporal_horizon,
        "num_warmup_steps": num_warmup_steps,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "note": "val/test are generated by splitting apebench test trajectories into halves after deterministic shuffle",
        "shapes": shapes,
    }

    if save_data:
        payload = {
            "train_trjs": train_trjs,
            "val_trjs": val_trjs,
            "test_trjs": test_trjs2,
            "meta": meta,
        }
        torch.save(payload, cache_path)
        if verbose:
            print(f"[build_kmflow_dataset] Saved cached dataset to: {cache_path}")

    datasets = (
        TensorDataset(X_train, Y_train),
        TensorDataset(X_val, Y_val),
        TensorDataset(X_test, Y_test),
    )

    if return_trajectories:
        trajectories = {
            "val_trjs": val_trjs,
            "test_trjs": test_trjs2,
        }
        return datasets, trajectories
    else:
        return datasets
