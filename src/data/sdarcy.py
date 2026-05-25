from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def _xy_from_a_u(a: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 3 or u.ndim != 4:
        raise ValueError(
            f"Expected a (K,H,W) and u (K,M,H,W); got a={tuple(a.shape)}, u={tuple(u.shape)}"
        )
    if u.shape[0] != a.shape[0] or u.shape[-2:] != a.shape[-2:]:
        raise ValueError(f"Shape mismatch: a={tuple(a.shape)} vs u={tuple(u.shape)}")

    K, M, H, W = u.shape
    x = a.repeat_interleave(M, dim=0)
    y = u.reshape(K * M, H, W)
    return x, y


def _downsample_2d(
    t: torch.Tensor, size: int, mode: str, *, antialias: bool
) -> torch.Tensor:
    if int(t.shape[-2]) == int(size) and int(t.shape[-1]) == int(size):
        return t
    kwargs: dict[str, Any] = {"mode": str(mode)}
    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = False

    if bool(antialias) and mode in ("bilinear", "bicubic"):
        try:
            return F.interpolate(
                t, size=(int(size), int(size)), antialias=True, **kwargs
            )
        except TypeError:
            pass
    return F.interpolate(t, size=(int(size), int(size)), **kwargs)


def _jax_make_sdarcy_dataset(
    *,
    N: int,
    n_train_inputs: int,
    n_val_inputs: int,
    n_test_inputs: int,
    n_outputs_per_input_train: int,
    n_outputs_per_input_eval: int,
    solve_batch_size: int,
    seed: int,
    tau_a: float,
    alpha_a: float,
    mu_f: float,
    sigma_f_ln: float,
    ell_f_ln: float,
    sigma_gp_f_ln: float,
    jitter_f_ln: float,
    sigma_f_gp: float,
    ell_f_gp: float,
    sigma_gp_f_gp: float,
    jitter_f_gp: float,
    tol: float,
    maxiter: int,
    verbose: bool,
) -> dict[str, Any]:
    """
    Dataset builder for forcing that is the sum of a log-normal GP and a mean-zero GP.
    The component amplitudes are controlled directly by `sigma_f_ln` and `sigma_f_gp`:
      f = sigma_f_ln * f_lognormal + sigma_f_gp * f_mean_zero

    `forcing_norm` semantics are ignored in this variant (normalization removed).
    """
    import jax
    import jax.numpy as jnp
    from jax import jit, random, vmap
    from jax.scipy.sparse.linalg import cg

    N = int(N)
    h = float(1.0 / float(N))
    solve_batch_size = int(solve_batch_size)
    if solve_batch_size <= 0:
        raise ValueError(f"solve_batch_size must be positive; got {solve_batch_size}")

    def _idct2(x: jax.Array) -> jax.Array:
        fft = jax.scipy.fft
        if hasattr(fft, "idct"):
            try:
                y = fft.idct(x, type=2, norm="ortho", axis=0)
                y = fft.idct(y, type=2, norm="ortho", axis=1)
                return y
            except TypeError:
                y = fft.idct(x, norm="ortho", axis=0)
                y = fft.idct(y, norm="ortho", axis=1)
                return y
        raise NotImplementedError(
            "JAX version does not provide jax.scipy.fft.idct; cannot invert DCT-II."
        )

    k = jnp.arange(N, dtype=jnp.float32)
    kx, ky = jnp.meshgrid(k, k, indexing="ij")
    lam = (jnp.pi**2) * (kx**2 + ky**2)
    psd = (lam + jnp.float32(tau_a)) ** (-jnp.float32(alpha_a))
    sqrt_psd = jnp.sqrt(psd).astype(jnp.float32)

    @jit
    def sample_permeability_a(key_a: jax.Array) -> jax.Array:
        z = random.normal(key_a, (N, N), dtype=jnp.float32)
        coeffs = z * sqrt_psd
        g = _idct2(coeffs).astype(jnp.float32)
        a = jnp.where(g >= 0, jnp.float32(12.0), jnp.float32(3.0))
        return a.astype(jnp.float32)

    def rbf_kernel_1d(x: jax.Array, ell: float, sigma_gp: float) -> jax.Array:
        x = x.astype(jnp.float32)
        ell = jnp.float32(ell)
        sigma_gp = jnp.float32(sigma_gp)
        d2 = (x[:, None] - x[None, :]) ** 2
        return (sigma_gp**2) * jnp.exp(-0.5 * d2 / (ell**2))

    xgrid = jnp.linspace(jnp.float32(0.0), jnp.float32(1.0), N)

    def _make_cholesky(
        ell: float, sigma_gp: float, jitter: float
    ) -> tuple[jax.Array, jax.Array]:
        Kx = rbf_kernel_1d(xgrid, ell=float(ell), sigma_gp=float(sigma_gp)).astype(
            jnp.float32
        )
        Ky = rbf_kernel_1d(xgrid, ell=float(ell), sigma_gp=float(sigma_gp)).astype(
            jnp.float32
        )
        Kx = Kx + jnp.float32(jitter) * jnp.eye(N, dtype=jnp.float32)
        Ky = Ky + jnp.float32(jitter) * jnp.eye(N, dtype=jnp.float32)
        return jnp.linalg.cholesky(Kx).astype(jnp.float32), jnp.linalg.cholesky(
            Ky
        ).astype(jnp.float32)

    sigma_f_ln_v = float(sigma_f_ln)
    ell_f_ln_v = float(ell_f_ln)
    sigma_gp_f_ln_v = float(sigma_gp_f_ln)
    jitter_f_ln_v = float(jitter_f_ln)

    sigma_f_gp_v = float(sigma_f_gp)
    ell_f_gp_v = float(ell_f_gp)
    sigma_gp_f_gp_v = float(sigma_gp_f_gp)
    jitter_f_gp_v = float(jitter_f_gp)

    Lx_ln, Ly_ln = _make_cholesky(ell_f_ln_v, sigma_gp_f_ln_v, jitter_f_ln_v)
    Lx_gp, Ly_gp = _make_cholesky(ell_f_gp_v, sigma_gp_f_gp_v, jitter_f_gp_v)

    @jit
    def sample_gp_rbf_separable_2d(
        key: jax.Array, Lx: jax.Array, Ly: jax.Array
    ) -> jax.Array:
        Z = random.normal(key, (N, N), dtype=jnp.float32)
        return ((Lx @ Z) @ Ly.T).astype(jnp.float32)

    @jit
    def _mean_zero(t: jax.Array) -> jax.Array:
        return (t - jnp.mean(t)).astype(jnp.float32)

    @jit
    def sample_forcing_f(key: jax.Array) -> jax.Array:
        k1, k2 = random.split(key)
        G1 = _mean_zero(sample_gp_rbf_separable_2d(k1, Lx_ln, Ly_ln))
        ln_comp = jnp.exp(G1)

        G2 = _mean_zero(sample_gp_rbf_separable_2d(k2, Lx_gp, Ly_gp))
        zero_comp = G2

        f = (jnp.float32(sigma_f_ln_v) * ln_comp) + (
            jnp.float32(sigma_f_gp_v) * zero_comp
        )
        return f.astype(jnp.float32)

    @jit
    def apply_darcy_operator(u: jax.Array, a: jax.Array, h: float) -> jax.Array:
        u_pad = jnp.pad(
            u, ((1, 1), (1, 1)), mode="constant", constant_values=jnp.float32(0.0)
        )
        uc = u_pad[1:-1, 1:-1]
        u_r, u_l = u_pad[2:, 1:-1], u_pad[:-2, 1:-1]
        u_u, u_d = u_pad[1:-1, 2:], u_pad[1:-1, :-2]

        a_pad = jnp.pad(a, ((1, 1), (1, 1)), mode="edge")
        ax_p = jnp.float32(0.5) * (a_pad[1:-1, 1:-1] + a_pad[2:, 1:-1])
        ax_m = jnp.float32(0.5) * (a_pad[1:-1, 1:-1] + a_pad[:-2, 1:-1])
        ay_p = jnp.float32(0.5) * (a_pad[1:-1, 1:-1] + a_pad[1:-1, 2:])
        ay_m = jnp.float32(0.5) * (a_pad[1:-1, 1:-1] + a_pad[1:-1, :-2])

        term_x = ax_p * (u_r - uc) - ax_m * (uc - u_l)
        term_y = ay_p * (u_u - uc) - ay_m * (uc - u_d)
        return -(term_x + term_y) / (h**2)

    def batched_cg(
        matvec, b: jax.Array, *, tol: float, maxiter: int
    ) -> tuple[jax.Array, jax.Array]:
        eps = jnp.float32(1e-12)
        tol_f = jnp.float32(tol)
        maxiter_i = int(maxiter)

        x = jnp.zeros_like(b, dtype=jnp.float32)
        r = (b - matvec(x)).astype(jnp.float32)
        p = r

        b2 = jnp.sum(b * b, axis=1).astype(jnp.float32) + eps
        rs = jnp.sum(r * r, axis=1).astype(jnp.float32)

        def converged(rs_):
            return rs_ <= (tol_f**2) * b2

        def cond(state):
            i, x, r, p, rs = state
            return jnp.logical_and(
                i < maxiter_i, jnp.logical_not(jnp.all(converged(rs)))
            )

        def body(state):
            i, x, r, p, rs = state
            Ap = matvec(p).astype(jnp.float32)
            pAp = jnp.sum(p * Ap, axis=1).astype(jnp.float32) + eps
            alpha = rs / pAp

            x_new = x + alpha[:, None] * p
            r_new = r - alpha[:, None] * Ap
            rs_new = jnp.sum(r_new * r_new, axis=1).astype(jnp.float32)

            beta = rs_new / (rs + eps)
            p_new = r_new + beta[:, None] * p

            active = jnp.logical_not(converged(rs_new))
            active2 = active[:, None]
            x = jnp.where(active2, x_new, x)
            r = jnp.where(active2, r_new, r)
            p = jnp.where(active2, p_new, p)
            rs = jnp.where(active, rs_new, rs)
            return (i + 1, x, r, p, rs)

        _, x, _, _, rs = jax.lax.while_loop(cond, body, (0, x, r, p, rs))
        info = jnp.where(converged(rs), jnp.int32(0), jnp.int32(1))
        return x.astype(jnp.float32), info

    def solve_darcy_batched(
        a_b: jax.Array, f_b: jax.Array, h: float, tol: float, maxiter: int
    ) -> jax.Array:
        B = int(a_b.shape[0])
        n = int(a_b.shape[1] * a_b.shape[2])
        b = f_b.reshape(B, n).astype(jnp.float32)

        def matvec(x_flat: jax.Array) -> jax.Array:
            x = x_flat.reshape(B, N, N)
            Ax = vmap(lambda uu, aa: apply_darcy_operator(uu, aa, h), in_axes=(0, 0))(
                x, a_b
            )
            return Ax.reshape(B, n)

        x_flat, _info = batched_cg(matvec, b, tol=float(tol), maxiter=int(maxiter))
        return x_flat.reshape(B, N, N).astype(jnp.float32)

    master = random.PRNGKey(int(seed))
    key_train_a, key_train_f, key_val_a, key_val_f, key_test_a, key_test_f = (
        random.split(master, 6)
    )

    def _sample_split(
        *, key_a: jax.Array, key_f: jax.Array, K: int, M: int, split_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        K = int(K)
        M = int(M)

        a_out = np.empty((K, N, N), dtype=np.float32)
        u_out = np.empty((K, M, N, N), dtype=np.float32)

        B_in = int(min(solve_batch_size, K))
        if verbose:
            print(
                f"[sdarcy:{split_name}] solve_batch_size={B_in} (inputs per chunk), M={M}"
            )

        for i0 in range(0, K, B_in):
            take = int(min(B_in, K - i0))
            pad = int(B_in - take)

            keys_a = random.split(random.fold_in(key_a, i0), B_in)
            a_b = vmap(sample_permeability_a)(keys_a)

            keys_f_base = random.split(random.fold_in(key_f, i0), B_in)
            keys_f = vmap(lambda k: random.split(k, M))(keys_f_base)
            keys_f_flat = keys_f.reshape(B_in * M, -1)

            f_flat = vmap(sample_forcing_f)(keys_f_flat).reshape(B_in * M, N, N)

            a_flat = jnp.repeat(a_b, repeats=M, axis=0)
            u_flat = solve_darcy_batched(
                a_flat, f_flat, h=h, tol=float(tol), maxiter=int(maxiter)
            )
            u_flat = u_flat.block_until_ready()

            a_host = np.array(jax.device_get(a_b), dtype=np.float32)[:take]
            u_host = np.array(jax.device_get(u_flat), dtype=np.float32)[
                : take * M
            ].reshape(take, M, N, N)
            a_out[i0 : i0 + take] = a_host
            u_out[i0 : i0 + take] = u_host

            if verbose and (i0 % max(1, (K // 10)) == 0):
                print(f"[sdarcy:{split_name}] {i0}/{K} inputs done")

        return a_out, u_out

    train_a, train_u = _sample_split(
        key_a=key_train_a,
        key_f=key_train_f,
        K=int(n_train_inputs),
        M=int(n_outputs_per_input_train),
        split_name="train",
    )
    val_a, val_u = _sample_split(
        key_a=key_val_a,
        key_f=key_val_f,
        K=int(n_val_inputs),
        M=int(n_outputs_per_input_eval),
        split_name="val",
    )
    test_a, test_u = _sample_split(
        key_a=key_test_a,
        key_f=key_test_f,
        K=int(n_test_inputs),
        M=int(n_outputs_per_input_eval),
        split_name="test",
    )

    return {
        "train_a": train_a,
        "train_u": train_u,
        "val_a": val_a,
        "val_u": val_u,
        "test_a": test_a,
        "test_u": test_u,
        "meta": {
            "backend": "jax_sdarcy",
            "forcing_type": "sum_lognormal_plus_gp",
            "N": int(N),
            "h": float(h),
            "seed": int(seed),
            "tau_a": float(tau_a),
            "alpha_a": float(alpha_a),
            "mu_f": float(mu_f),
            "mu_f_used": False,
            "sigma_f_ln": float(sigma_f_ln_v),
            "ell_f_ln": float(ell_f_ln_v),
            "sigma_gp_f_ln": float(sigma_gp_f_ln_v),
            "jitter_f_ln": float(jitter_f_ln_v),
            "sigma_f_gp": float(sigma_f_gp_v),
            "ell_f_gp": float(ell_f_gp_v),
            "sigma_gp_f_gp": float(sigma_gp_f_gp_v),
            "jitter_f_gp": float(jitter_f_gp_v),
            "tol": float(tol),
            "maxiter": int(maxiter),
            "solve_batch_size": int(solve_batch_size),
            "n_train_inputs": int(n_train_inputs),
            "n_val_inputs": int(n_val_inputs),
            "n_test_inputs": int(n_test_inputs),
            "n_outputs_per_input_train": int(n_outputs_per_input_train),
            "n_outputs_per_input_eval": int(n_outputs_per_input_eval),
        },
    }


def build_sdarcy_dataset(
    dataset_dir: str | Path = "datasets",
    filename: str = "sdarcy.pt",
    *,
    verbose: bool = True,
    save_data: bool = False,
    return_stochastic: bool = True,
    overwrite: bool = False,
    seed: int = 42,
    data_name: str = "SDarcy",
    N: int = 128,
    out_N: int | None = None,
    downsample_mode_a: str = "area",
    downsample_mode_u: str = "bilinear",
    downsample_antialias: bool = True,
    n_train_inputs: int = 10000,
    n_val_inputs: int = 32,
    n_test_inputs: int = 32,
    n_outputs_per_input_train: int = 1,
    n_outputs_per_input_eval: int = 64,
    solve_batch_size: int = 256,
    tau_a: float = 9.0,
    alpha_a: float = 2.0,
    mu_f: float = 0.0,
    sigma_f_ln: float = 1.0,
    ell_f_ln: float = 0.25,
    sigma_gp_f_ln: float = 1.0,
    jitter_f_ln: float = 1e-5,
    sigma_f_gp: float = 1.0,
    ell_f_gp: float = 0.25,
    sigma_gp_f_gp: float = 1.0,
    jitter_f_gp: float = 1e-5,
    tol: float = 1e-6,
    maxiter: int = 5000,
) -> (
    tuple[TensorDataset, TensorDataset, TensorDataset]
    | tuple[tuple[TensorDataset, TensorDataset, TensorDataset], dict[str, Any]]
):
    dataset_path = Path(dataset_dir).expanduser()
    dataset_path.mkdir(parents=True, exist_ok=True)
    path = dataset_path / filename

    if path.is_file() and not overwrite:
        if verbose:
            print(f"[build_sdarcy_dataset] Loading cached dataset from: {path}")
        payload = torch.load(path, map_location="cpu")
    else:
        if verbose:
            print(
                f"[build_sdarcy_dataset] Generating dataset (save_data={bool(save_data)}) -> {path}"
            )

        data_jax = _jax_make_sdarcy_dataset(
            N=int(N),
            n_train_inputs=int(n_train_inputs),
            n_val_inputs=int(n_val_inputs),
            n_test_inputs=int(n_test_inputs),
            n_outputs_per_input_train=int(n_outputs_per_input_train),
            n_outputs_per_input_eval=int(n_outputs_per_input_eval),
            solve_batch_size=int(solve_batch_size),
            seed=int(seed),
            tau_a=float(tau_a),
            alpha_a=float(alpha_a),
            mu_f=float(mu_f),
            sigma_f_ln=float(sigma_f_ln),
            ell_f_ln=float(ell_f_ln),
            sigma_gp_f_ln=float(sigma_gp_f_ln),
            jitter_f_ln=float(jitter_f_ln),
            sigma_f_gp=float(sigma_f_gp),
            ell_f_gp=float(ell_f_gp),
            sigma_gp_f_gp=float(sigma_gp_f_gp),
            jitter_f_gp=float(jitter_f_gp),
            tol=float(tol),
            maxiter=int(maxiter),
            verbose=bool(verbose),
        )

        def to_torch(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.array(a, dtype=np.float32)).to(
                dtype=torch.float32, device="cpu"
            )

        payload = {
            "train_a": to_torch(data_jax["train_a"]),
            "train_u": to_torch(data_jax["train_u"]),
            "val_a": to_torch(data_jax["val_a"]),
            "val_u": to_torch(data_jax["val_u"]),
            "test_a": to_torch(data_jax["test_a"]),
            "test_u": to_torch(data_jax["test_u"]),
            "meta": {
                "data_name": data_name,
                **(data_jax.get("meta") or {}),
                "out_N": None if out_N is None else int(out_N),
                "downsample_mode_a": str(downsample_mode_a),
                "downsample_mode_u": str(downsample_mode_u),
                "downsample_antialias": bool(downsample_antialias),
            },
        }

        if out_N is not None:
            out_sz = int(out_N)
            if verbose:
                print(
                    f"[build_sdarcy_dataset] Downsampling from {int(N)}x{int(N)} -> {out_sz}x{out_sz}"
                )

            def ds_a(a: torch.Tensor) -> torch.Tensor:
                return _downsample_2d(
                    a.unsqueeze(1),
                    out_sz,
                    str(downsample_mode_a),
                    antialias=bool(downsample_antialias),
                ).squeeze(1)

            def ds_u(u: torch.Tensor) -> torch.Tensor:
                K, M, H, W = u.shape
                u2 = u.reshape(K * M, 1, H, W)
                u2 = _downsample_2d(
                    u2,
                    out_sz,
                    str(downsample_mode_u),
                    antialias=bool(downsample_antialias),
                )
                return u2.reshape(K, M, out_sz, out_sz)

            payload["train_a"] = ds_a(payload["train_a"])
            payload["val_a"] = ds_a(payload["val_a"])
            payload["test_a"] = ds_a(payload["test_a"])
            payload["train_u"] = ds_u(payload["train_u"])
            payload["val_u"] = ds_u(payload["val_u"])
            payload["test_u"] = ds_u(payload["test_u"])

        if save_data:
            torch.save(payload, path)
            if verbose:
                print(f"[build_sdarcy_dataset] Saved cache to: {path}")

    if not isinstance(payload, dict):
        raise TypeError(f"Expected cached payload dict, got {type(payload)}")

    train_a = payload["train_a"].to(dtype=torch.float32, device="cpu")
    train_u = payload["train_u"].to(dtype=torch.float32, device="cpu")
    val_a = payload["val_a"].to(dtype=torch.float32, device="cpu")
    val_u = payload["val_u"].to(dtype=torch.float32, device="cpu")
    test_a = payload["test_a"].to(dtype=torch.float32, device="cpu")
    test_u = payload["test_u"].to(dtype=torch.float32, device="cpu")

    X_train, Y_train = _xy_from_a_u(train_a, train_u)
    X_val, Y_val = _xy_from_a_u(val_a, val_u)
    X_test, Y_test = _xy_from_a_u(test_a, test_u)

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
            f"[build_sdarcy_dataset] Shapes: "
            f"train_a={train_a.shape}, train_u={train_u.shape} -> train_pairs={len(train_set)}; "
            f"val_a={val_a.shape}, val_u={val_u.shape} -> val_pairs={len(val_set)}; "
            f"test_a={test_a.shape}, test_u={test_u.shape} -> test_pairs={len(test_set)}"
        )

    datasets = (train_set, val_set, test_set)
    if not return_stochastic:
        return datasets

    val_x0 = val_a.unsqueeze(1).contiguous()
    test_x0 = test_a.unsqueeze(1).contiguous()
    val_yset = val_u.unsqueeze(2).contiguous()
    test_yset = test_u.unsqueeze(2).contiguous()

    extras: dict[str, Any] = {
        "val_stochastic": {"x": val_x0, "y": val_yset},
        "test_stochastic": {"x": test_x0, "y": test_yset},
    }
    return datasets, extras
