from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .utils import split_test_trajectories_in_half, try_load_cached_traj_dataset


def _to_torch_3d(trjs: Any, *, name: str) -> torch.Tensor:
    """Coerce trajectory container into float32 CPU tensor shaped (N_traj, T, X)."""
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
        if t.shape[2] == 1:
            t = t.squeeze(2)
        elif t.shape[1] == 1:
            t = t.squeeze(1)
        else:
            raise ValueError(
                f"Expected {name} to be (N_traj, T, X) or (N_traj, T, 1, X); got {tuple(t.shape)}"
            )

    if t.ndim != 3:
        raise ValueError(
            f"Expected {name} to have shape (N_traj, T, X) but got {tuple(t.shape)}"
        )

    return t.to(device="cpu", dtype=torch.float32)


def _trjs_to_xy(trjs_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert trajectories (N_traj, T, X) into next-step pairs (N_traj*(T-1), X)."""
    x, y = trjs_3d[:, :-1, :], trjs_3d[:, 1:, :]
    x = x.reshape(-1, trjs_3d.shape[-1]).contiguous()
    y = y.reshape(-1, trjs_3d.shape[-1]).contiguous()
    return x, y


def build_ks_dataset(
    dataset_dir: str = "datasets",
    filename: str = "ks.pt",
    save_data: bool = False,
    verbose: bool = True,
    overwrite: bool = False,
    data_name: str = "KS",
    return_trajectories: bool = False,
    spatial_resolution: int = 256,
    seed: int = 42,
    dt_data: float = 1.0,
    n_train_traj: int = 1024,
    n_val_traj: int = 64,
    n_test_traj: int = 64,
    dt: float | None = None,
    dt_out: float | None = None,  # legacy: output spacing; prefer `dt` + `num_substeps`
    num_substeps: int | None = None,
    num_train_samples: int | None = None,
    num_test_samples: int | None = None,
    train_temporal_horizon: int | None = None,
    test_temporal_horizon: int | None = None,
    num_warmup_steps: int = 100,
    train_seed: int | None = None,
    test_seed: int = 773,
    **_: Any,  # ignore unknown Hydra keys (e.g., dt_sim, L, etc.)
) -> (
    tuple[TensorDataset, TensorDataset, TensorDataset]
    | tuple[tuple[TensorDataset, TensorDataset, TensorDataset], dict[str, torch.Tensor]]
):
    """
    KS dataset builder backed by `apebench`.

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
        log_prefix="build_ks_dataset",
        return_trajectories=return_trajectories,
        to_xy=_trjs_to_xy,
    )
    if cached is not None:
        return cached
    dt_in = float(dt if dt is not None else dt_data)
    dt_out_cfg = float(dt_out) if dt_out is not None else None

    if dt_in <= 0:
        raise ValueError("dt must be positive.")
    dt_out_eff = dt_out_cfg if dt_out_cfg is not None else dt_in
    if dt_out_eff <= 0:
        raise ValueError("dt_out must be positive.")

    if num_substeps is None:
        if dt_out_cfg is None:
            num_substeps_eff = 1
        else:
            ratio = dt_out_eff / dt_in
            if abs(ratio - round(ratio)) > 1e-6:
                raise ValueError(
                    "When providing legacy (dt, dt_out), dt_out/dt must be an integer so it can be "
                    f"expressed via num_substeps. Got dt={dt_in}, dt_out={dt_out_eff} -> ratio={ratio}."
                )
            num_substeps_eff = int(round(ratio))
    else:
        num_substeps_eff = int(num_substeps)
        if num_substeps_eff <= 0:
            raise ValueError("num_substeps must be a positive integer.")

    dt_sim_eff = dt_out_eff / num_substeps_eff
    num_train_samples = int(
        num_train_samples if num_train_samples is not None else n_train_traj
    )
    num_test_samples = int(
        num_test_samples if num_test_samples is not None else (n_val_traj + n_test_traj)
    )
    if num_test_samples <= 0:
        raise ValueError(
            "num_test_samples must be positive (it is used for val/test split)."
        )
    if train_temporal_horizon is None:
        raise ValueError("train_temporal_horizon must be specified (not None)")
    if train_temporal_horizon <= 0:
        raise ValueError("train_temporal_horizon must be positive.")

    if test_temporal_horizon is None:
        test_temporal_horizon = int(2 * train_temporal_horizon)
    if test_temporal_horizon <= 0:
        raise ValueError("test_temporal_horizon must be positive.")

    train_seed = int(seed if train_seed is None else train_seed)

    try:
        import apebench  # type: ignore
    except Exception as e:
        raise ModuleNotFoundError(
            "Missing dependency 'apebench'. Install it (e.g. `pip install apebench`) "
            "or add it to your environment before generating KS datasets."
        ) from e
    sc = apebench.scenarios.physical.KuramotoSivashinsky(
        num_points=int(spatial_resolution),
        dt=float(dt_out_eff),  # output spacing of returned trajectories
        num_substeps=int(
            num_substeps_eff
        ),  # internal integrator step = dt/num_substeps
        num_train_samples=int(num_train_samples),
        train_temporal_horizon=int(
            train_temporal_horizon
        ),  # <-- time-steps per trajectory (excluding initial)
        num_test_samples=int(num_test_samples),
        test_temporal_horizon=int(test_temporal_horizon),
        num_warmup_steps=int(num_warmup_steps),
        train_seed=int(train_seed),
        test_seed=int(test_seed),
    )

    train_trjs = _to_torch_3d(sc.get_train_data(), name="train_trjs")
    test_trjs = _to_torch_3d(sc.get_test_data(), name="test_trjs")

    val_trjs, test_trjs2 = split_test_trajectories_in_half(
        test_trjs, seed=int(test_seed)
    )

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
        "dt": float(dt_out_eff),
        "dt_out": float(dt_out_eff),
        "num_substeps": int(num_substeps_eff),
        "dt_sim": float(dt_sim_eff),
        "downsample_factor": int(num_substeps_eff),
        "num_points": int(spatial_resolution),
        "num_train_samples": int(num_train_samples),
        "train_temporal_horizon": int(train_temporal_horizon),
        "num_test_samples": int(num_test_samples),
        "test_temporal_horizon": int(test_temporal_horizon),
        "num_warmup_steps": int(num_warmup_steps),
        "train_seed": int(train_seed),
        "test_seed": int(test_seed),
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
            print(f"[build_ks_dataset] Saved cached dataset to: {cache_path}")

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
