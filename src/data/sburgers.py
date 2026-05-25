from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset


def _steps_per_macro(*, dt_macro: float, dt_sim: float) -> int:
    ratio = float(dt_macro) / float(dt_sim)
    steps = int(round(ratio))
    if abs(ratio - steps) > 1e-12:
        raise ValueError(
            f"dt_macro/dt_sim must be integer. Got {dt_macro}/{dt_sim}={ratio}."
        )
    if steps < 1:
        raise ValueError("dt_macro must be >= dt_sim.")
    return steps


def _xy_from_u0_u1(
    u0: torch.Tensor, u1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten all paired samples.

    u0: (K,N), u1: (K,M,N) -> X,Y: (K*M, N)
    """
    if u0.ndim != 2 or u1.ndim != 3:
        raise ValueError(
            f"Expected u0 (K,N) and u1 (K,M,N); got u0={tuple(u0.shape)}, u1={tuple(u1.shape)}"
        )
    if u1.shape[0] != u0.shape[0] or u1.shape[-1] != u0.shape[-1]:
        raise ValueError(
            f"Shape mismatch: u0={tuple(u0.shape)} vs u1={tuple(u1.shape)}"
        )

    K, M, N = u1.shape
    x = u0.repeat_interleave(M, dim=0)
    y = u1.reshape(K * M, N)
    return x, y


def _jax_make_single_step_dataset(
    *,
    N: int,
    L: float,
    dt_macro: float,
    dt_sim: float,
    nu: float,
    sigma: float,
    noise_num_modes: int,
    noise_modes: list[int] | None,
    noise_mode_weights: list[float] | None,
    dealias: bool,
    kmax_ic: int,
    n_train_inputs: int,
    n_val_inputs: int,
    n_val_outputs_per_input: int,
    n_test_inputs: int,
    n_test_outputs_per_input: int,
    n_train_outputs_per_input: int,
    seed: int,
) -> dict[str, Any]:
    """
    Generates a single-step stochastic Burgers dataset using:
      - Pseudo-spectral spatial discretization (FFT), optional 2/3 de-aliasing
      - Drift integrator: ETDRK4 (stronger than explicit Euler)
      - Noise integrator: Euler–Maruyama (additive noise)
    """
    import jax
    import jax.numpy as jnp

    N = int(N)
    L = float(L)
    dt_macro = float(dt_macro)
    dt_sim = float(dt_sim)
    nu = float(nu)
    sigma = float(sigma)
    noise_num_modes = int(noise_num_modes)
    noise_modes = list(noise_modes) if noise_modes is not None else None
    noise_mode_weights = (
        list(noise_mode_weights) if noise_mode_weights is not None else None
    )
    dealias = bool(dealias)
    kmax_ic = int(kmax_ic)

    if noise_modes is not None:
        if len(noise_modes) < 1:
            raise ValueError("noise_modes must be a non-empty list.")
        if any(int(k) < 1 for k in noise_modes):
            raise ValueError(
                f"noise_modes must be positive integers, got {noise_modes}."
            )
        if noise_mode_weights is not None and len(noise_mode_weights) != len(
            noise_modes
        ):
            raise ValueError(
                f"noise_mode_weights must have same length as noise_modes; got "
                f"{len(noise_mode_weights)} vs {len(noise_modes)}."
            )
    else:
        if noise_mode_weights is not None:
            raise ValueError(
                "noise_mode_weights was provided but noise_modes is None. Provide noise_modes too."
            )
        if noise_num_modes < 1:
            raise ValueError(f"noise_num_modes must be >= 1, got {noise_num_modes}.")

    steps = _steps_per_macro(dt_macro=dt_macro, dt_sim=dt_sim)

    def make_grid_and_wavenumbers() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = jnp.linspace(0.0, L, N, endpoint=False)
        dx = L / N
        k = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=dx)

        if dealias:
            n = jnp.fft.fftfreq(N, d=1.0) * N
            cutoff = N // 3
            dealias_mask = (jnp.abs(n) <= cutoff).astype(jnp.float32)
        else:
            dealias_mask = jnp.ones((N,), dtype=jnp.float32)

        return x.astype(jnp.float32), k.astype(jnp.float32), dealias_mask

    x, k, dealias_mask = make_grid_and_wavenumbers()
    if noise_modes is None:
        m = int(noise_num_modes)
        noise_modes_arr = jnp.arange(1, m + 1, dtype=jnp.float32)  # (m,)
        noise_weights_arr = jnp.ones((m,), dtype=jnp.float32)  # (m,)
    else:
        noise_modes_arr = jnp.asarray(noise_modes, dtype=jnp.float32)  # (m,)
        if noise_mode_weights is None:
            noise_weights_arr = jnp.ones((noise_modes_arr.shape[0],), dtype=jnp.float32)
        else:
            noise_weights_arr = jnp.asarray(noise_mode_weights, dtype=jnp.float32)
        m = int(noise_modes_arr.shape[0])
    dt = jnp.float32(dt_sim)
    Lhat = (-jnp.float32(nu) * (k**2)).astype(jnp.complex64)  # diagonal in Fourier
    E = jnp.exp(Lhat * dt).astype(jnp.complex64)
    E2 = jnp.exp(Lhat * (dt / 2.0)).astype(jnp.complex64)
    M = 16
    r = jnp.exp(
        1j * jnp.pi * (jnp.arange(1, M + 1, dtype=jnp.float32) - 0.5) / M
    ).astype(jnp.complex64)
    LR = (dt * Lhat)[:, None] + r[None, :]  # (N, M)
    Q = dt * jnp.mean((jnp.exp(LR / 2.0) - 1.0) / LR, axis=1)
    f1 = dt * jnp.mean(
        (-4.0 - LR + jnp.exp(LR) * (4.0 - 3.0 * LR + LR * LR)) / (LR**3), axis=1
    )
    f2 = dt * jnp.mean((2.0 + LR + jnp.exp(LR) * (-2.0 + LR)) / (LR**3), axis=1)
    f3 = dt * jnp.mean(
        (-4.0 - 3.0 * LR - LR * LR + jnp.exp(LR) * (4.0 - LR)) / (LR**3), axis=1
    )

    Q = Q.astype(jnp.complex64)
    f1 = f1.astype(jnp.complex64)
    f2 = f2.astype(jnp.complex64)
    f3 = f3.astype(jnp.complex64)

    def _nonlinear_hat_from_u(u: jnp.ndarray) -> jnp.ndarray:
        u2_hat = jnp.fft.fft(u * u).astype(jnp.complex64)
        if dealias:
            u2_hat = u2_hat * dealias_mask  # mask in Fourier (float mask is fine)
        return (-0.5j * k * u2_hat).astype(jnp.complex64)

    def _etdrk4_drift_step(u: jnp.ndarray) -> jnp.ndarray:
        v_hat = jnp.fft.fft(u).astype(jnp.complex64)
        Nv = _nonlinear_hat_from_u(u)

        a_hat = E2 * v_hat + Q * Nv
        a = jnp.fft.ifft(a_hat).real.astype(jnp.float32)
        Na = _nonlinear_hat_from_u(a)

        b_hat = E2 * v_hat + Q * Na
        b = jnp.fft.ifft(b_hat).real.astype(jnp.float32)
        Nb = _nonlinear_hat_from_u(b)

        c_hat = E2 * a_hat + Q * (2.0 * Nb - Nv)
        c = jnp.fft.ifft(c_hat).real.astype(jnp.float32)
        Nc = _nonlinear_hat_from_u(c)

        v_hat_next = E * v_hat + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        u_det = jnp.fft.ifft(v_hat_next).real.astype(jnp.float32)
        return u_det

    def burgers_micro_step(
        u: jnp.ndarray,
        key: jax.Array,
        x: jnp.ndarray,
        k: jnp.ndarray,
        dealias_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        u_det = _etdrk4_drift_step(u)
        z = jax.random.normal(key, shape=(m,), dtype=jnp.float32)  # (m,)
        dW = jnp.sqrt(dt) * z  # (m,)
        sigma_x_modes = jnp.cos(noise_modes_arr[:, None] * x[None, :]).astype(
            jnp.float32
        )  # (m, N)
        sigma_x_modes = (noise_weights_arr[:, None] * sigma_x_modes).astype(
            jnp.float32
        )  # (m, N)
        noise_x = jnp.tensordot(dW, sigma_x_modes, axes=(0, 0)).astype(
            jnp.float32
        )  # (N,)
        u_next = u_det + jnp.float32(sigma) * noise_x
        return u_next.astype(jnp.float32)

    def burgers_macro_step(
        u0: jnp.ndarray,
        base_key: jax.Array,
        x: jnp.ndarray,
        k: jnp.ndarray,
        dealias_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        def body(u, i):
            key_i = jax.random.fold_in(base_key, i)
            u_next = burgers_micro_step(u, key_i, x, k, dealias_mask)
            return u_next, None

        uS, _ = jax.lax.scan(body, u0, jnp.arange(steps))
        return uS

    burgers_macro_step = jax.jit(burgers_macro_step)

    def sample_initial_condition(key: jax.Array, amp: float = 1.0) -> jnp.ndarray:
        key_re, key_im = jax.random.split(key)

        re = jax.random.normal(key_re, (kmax_ic,), dtype=jnp.float32)
        im = jax.random.normal(key_im, (kmax_ic,), dtype=jnp.float32)
        coeff = (re + 1j * im) / jnp.sqrt(2.0)

        modes = jnp.arange(1, kmax_ic + 1, dtype=jnp.float32)
        decay = (1.0 / (modes**2)).astype(jnp.complex64)
        coeff = coeff.astype(jnp.complex64) * decay

        u_hat = jnp.zeros((N,), dtype=jnp.complex64)
        u_hat = u_hat.at[1 : kmax_ic + 1].set(coeff)
        u_hat = u_hat.at[-kmax_ic:].set(jnp.conj(coeff[::-1]))
        u_hat = u_hat.at[0].set(0.0 + 0j)

        u0 = jnp.fft.ifft(u_hat).real.astype(jnp.float32)
        u0 = amp * u0 / (jnp.std(u0) + 1e-6)
        return u0

    master = jax.random.PRNGKey(int(seed))
    key_ic_train, key_noise_train, key_rest = jax.random.split(master, 3)

    def ic_fn(kk):
        return sample_initial_condition(kk, amp=1.0)

    keys_train_ic = jax.random.split(key_ic_train, int(n_train_inputs))
    train_u0 = jax.vmap(ic_fn)(keys_train_ic)  # (K,N)

    def multi_outputs_for_one_u0(
        u0: jnp.ndarray, base_key: jax.Array, M: int
    ) -> jnp.ndarray:
        keys = jax.random.split(base_key, M)
        return jax.vmap(lambda kn: burgers_macro_step(u0, kn, x, k, dealias_mask))(
            keys
        )  # (M,N)

    keys_train_base = jax.random.split(key_noise_train, int(n_train_inputs))
    train_u1 = jax.vmap(
        lambda u0, bk: multi_outputs_for_one_u0(u0, bk, int(n_train_outputs_per_input))
    )(
        train_u0, keys_train_base
    )  # (K,M,N)
    key_ic_val, key_noise_val, key_ic_test, key_noise_test = jax.random.split(
        key_rest, 4
    )

    keys_val_ic = jax.random.split(key_ic_val, int(n_val_inputs))
    val_u0 = jax.vmap(ic_fn)(keys_val_ic)
    keys_val_base = jax.random.split(key_noise_val, int(n_val_inputs))
    val_u1 = jax.vmap(
        lambda u0, bk: multi_outputs_for_one_u0(u0, bk, int(n_val_outputs_per_input))
    )(val_u0, keys_val_base)

    keys_test_ic = jax.random.split(key_ic_test, int(n_test_inputs))
    test_u0 = jax.vmap(ic_fn)(keys_test_ic)
    keys_test_base = jax.random.split(key_noise_test, int(n_test_inputs))
    test_u1 = jax.vmap(
        lambda u0, bk: multi_outputs_for_one_u0(u0, bk, int(n_test_outputs_per_input))
    )(test_u0, keys_test_base)

    return {
        "x": x,
        "train_u0": train_u0,
        "train_u1": train_u1,
        "val_u0": val_u0,
        "val_u1": val_u1,
        "test_u0": test_u0,
        "test_u1": test_u1,
        "steps_per_macro": steps,
    }


def build_sburgers_dataset(
    dataset_dir: str | Path = "datasets",
    filename: str = "sburgers_single_step.pt",
    *,
    verbose: bool = True,
    save_data: bool = False,
    return_stochastic: bool = True,
    overwrite: bool = False,
    seed: int = 42,
    data_name: str = "SBurgers",
    num_points: int = 256,
    dt: float = 1.0,
    dt_sim: float = 1e-4,
    L: float = 2.0 * math.pi,
    nu: float = 0.01,
    noise_scale: float = 0.5,
    noise_num_modes: int = 1,
    noise_modes: list[int] | None = None,
    noise_mode_weights: list[float] | None = None,
    dealias: bool = True,
    n_train_inputs: int = 1024,
    n_val_inputs: int = 64,
    n_test_inputs: int = 64,
    n_outputs_per_input_train: int = 1,
    n_outputs_per_input_eval: int = 32,
    u0_amp: float = 1.0,
    u0_length_scale: float = 0.15,
    kmax_ic: int | None = None,
) -> (
    tuple[TensorDataset, TensorDataset, TensorDataset]
    | tuple[tuple[TensorDataset, TensorDataset, TensorDataset], dict[str, torch.Tensor]]
):
    """
    Stochastic Burgers (single-step) dataset builder with simple on-disk caching.

    Cache format (`.pt`) (torch tensors, CPU):
      - x: (N,)
      - train_u0: (Ktr,N), train_u1: (Ktr,Mtr,N)
      - val_u0: (Kva,N),   val_u1: (Kva,Mev,N)
      - test_u0: (Kte,N),  test_u1: (Kte,Mev,N)
      - meta: dict[str, Any]

    Outputs:
      - train_set/val_set/test_set are TensorDataset(X,Y) with X,Y shaped (B,1,N).
      - If return_stochastic=True, also returns extras dict with generic keys:
          {"val_stochastic": {"x": X0, "y": Yset}, "test_stochastic": {...}}
        where:
          - X0 is original-scale conditioning inputs: (K, 1, N)
          - Yset is original-scale output samples: (K, M, 1, N)
    """
    dataset_path = Path(dataset_dir).expanduser()
    dataset_path.mkdir(parents=True, exist_ok=True)
    path = dataset_path / filename

    if path.is_file() and not overwrite:
        if verbose:
            print(f"[build_sburgers_dataset] Loading cached dataset from: {path}")
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("meta") if isinstance(payload, dict) else None
        if isinstance(meta, dict):
            for _legacy_key in ("k0", "alpha", "zero_mean_noise", "noise_length_scale"):
                meta.pop(_legacy_key, None)
    else:
        if verbose:
            print(
                f"[build_sburgers_dataset] Generating dataset (save_data={bool(save_data)}) -> {path}"
            )
        N = int(num_points)
        dt_macro = float(dt)
        dt_sim_f = float(dt_sim)
        if dt_macro <= 0 or dt_sim_f <= 0:
            raise ValueError("dt and dt_sim must be positive.")
        kmax_ic_eff = (
            int(kmax_ic)
            if kmax_ic is not None
            else int(max(1, min(32, round(1.0 / max(float(u0_length_scale), 1e-6)))))
        )

        data_jax = _jax_make_single_step_dataset(
            N=N,
            L=float(L),
            dt_macro=dt_macro,
            dt_sim=dt_sim_f,
            nu=float(nu),
            sigma=float(noise_scale),
            noise_num_modes=int(noise_num_modes),
            noise_modes=noise_modes,
            noise_mode_weights=noise_mode_weights,
            dealias=bool(dealias),
            kmax_ic=int(kmax_ic_eff),
            n_train_inputs=int(n_train_inputs),
            n_val_inputs=int(n_val_inputs),
            n_val_outputs_per_input=int(n_outputs_per_input_eval),
            n_test_inputs=int(n_test_inputs),
            n_test_outputs_per_input=int(n_outputs_per_input_eval),
            n_train_outputs_per_input=int(n_outputs_per_input_train),
            seed=int(seed),
        )

        def to_torch(a) -> torch.Tensor:
            import jax

            return torch.from_numpy(np.array(jax.device_get(a))).to(
                dtype=torch.float32, device="cpu"
            )

        payload = {
            "x": to_torch(data_jax["x"]),
            "train_u0": to_torch(data_jax["train_u0"]) * float(u0_amp),
            "train_u1": to_torch(data_jax["train_u1"]) * float(u0_amp),
            "val_u0": to_torch(data_jax["val_u0"]) * float(u0_amp),
            "val_u1": to_torch(data_jax["val_u1"]) * float(u0_amp),
            "test_u0": to_torch(data_jax["test_u0"]) * float(u0_amp),
            "test_u1": to_torch(data_jax["test_u1"]) * float(u0_amp),
            "meta": {
                "data_name": data_name,
                "backend": "jax_macro_step",
                "num_points": N,
                "L": float(L),
                "dt_macro": dt_macro,
                "dt_sim": dt_sim_f,
                "steps_per_macro": int(data_jax["steps_per_macro"]),
                "nu": float(nu),
                "sigma": float(noise_scale),
                "noise_num_modes": int(noise_num_modes),
                "noise_modes": (
                    None if noise_modes is None else list(map(int, noise_modes))
                ),
                "noise_mode_weights": (
                    None
                    if noise_mode_weights is None
                    else list(map(float, noise_mode_weights))
                ),
                "noise_term": "sigma * sum_j w_j * cos(k_j x) dW_t^j (independent Brownian motions)",
                "dealias": bool(dealias),
                "kmax_ic": int(kmax_ic_eff),
                "u0_amp": float(u0_amp),
                "u0_length_scale": float(u0_length_scale),
                "seed": int(seed),
                "n_train_inputs": int(n_train_inputs),
                "n_val_inputs": int(n_val_inputs),
                "n_test_inputs": int(n_test_inputs),
                "n_outputs_per_input_train": int(n_outputs_per_input_train),
                "n_outputs_per_input_eval": int(n_outputs_per_input_eval),
                "drift_integrator": "ETDRK4",
            },
        }

        if save_data:
            torch.save(payload, path)
            if verbose:
                print(f"[build_sburgers_dataset] Saved cache to: {path}")

    if not isinstance(payload, dict):
        raise TypeError(f"Expected cached payload dict, got {type(payload)}")
    x = payload["x"].to(dtype=torch.float32, device="cpu")
    train_u0 = payload["train_u0"].to(dtype=torch.float32, device="cpu")
    train_u1 = payload["train_u1"].to(dtype=torch.float32, device="cpu")
    val_u0 = payload["val_u0"].to(dtype=torch.float32, device="cpu")
    val_u1 = payload["val_u1"].to(dtype=torch.float32, device="cpu")
    test_u0 = payload["test_u0"].to(dtype=torch.float32, device="cpu")
    test_u1 = payload["test_u1"].to(dtype=torch.float32, device="cpu")
    X_train, Y_train = _xy_from_u0_u1(train_u0, train_u1)
    X_val, Y_val = _xy_from_u0_u1(val_u0, val_u1)
    X_test, Y_test = _xy_from_u0_u1(test_u0, test_u1)

    train_set = TensorDataset(
        X_train.unsqueeze(1).contiguous(), Y_train.unsqueeze(1).contiguous()
    )
    val_set = TensorDataset(
        X_val.unsqueeze(1).contiguous(), Y_val.unsqueeze(1).contiguous()
    )
    test_set = TensorDataset(
        X_test.unsqueeze(1).contiguous(), Y_test.unsqueeze(1).contiguous()
    )

    if verbose:
        print(
            f"[build_sburgers_dataset] Shapes: "
            f"train_u0={train_u0.shape}, train_u1={train_u1.shape} -> train_pairs={len(train_set)}; "
            f"val_u0={val_u0.shape}, val_u1={val_u1.shape} -> val_pairs={len(val_set)}; "
            f"test_u0={test_u0.shape}, test_u1={test_u1.shape} -> test_pairs={len(test_set)}"
        )

    datasets = (train_set, val_set, test_set)

    if not return_stochastic:
        return datasets
    val_x0 = val_u0.unsqueeze(1).contiguous()
    test_x0 = test_u0.unsqueeze(1).contiguous()
    val_yset = val_u1.unsqueeze(2).contiguous()
    test_yset = test_u1.unsqueeze(2).contiguous()

    extras: dict[str, Any] = {
        "val_stochastic": {"x": val_x0, "y": val_yset},
        "test_stochastic": {"x": test_x0, "y": test_yset},
    }
    return datasets, extras
