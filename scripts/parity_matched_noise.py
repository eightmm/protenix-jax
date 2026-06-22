"""Noise-matched parity harness: JAX Protenix vs captured torch noise.

Loads the torch-captured diffusion noise (init + per-step churn) and runs the
JAX sampler with that exact noise injected and centre-only augmentation, so the
two trajectories are directly comparable. Reports raw and Kabsch-aligned atom
RMSD (all-atom + CA) against the torch coordinates, plus a determinism check
and GPU wall-clock + peak-VRAM timing.

Throwaway harness; no src changes. Run from protenix_jax with GPU jax:
    uv run --extra cuda13 python scripts/parity_matched_noise.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

SC = Path(
    "/tmp/claude-1000/-home-jaemin-non-project-optimizing/"
    "d7543612-73cf-404f-8252-997982139f04/scratchpad/e2e"
)
INPUT_JSON = SC / "input.json"
WEIGHTS = SC / "native_base.pkl"
TORCH_NOISE = SC / "torch_noise.npz"
TORCH_COORD = SC / "torch_coord.npy"
N_STEP = 20
N_CYCLE = 10


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD after optimal rigid alignment of p onto q."""
    pc = p - p.mean(0)
    qc = q - q.mean(0)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    pr = pc @ r.T
    return float(np.sqrt(((pr - qc) ** 2).sum(1).mean()))


def raw_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(((p - q) ** 2).sum(1).mean()))


def main() -> None:
    print("backend:", jax.default_backend())

    import protenix_jax.models.diffusion.diffusion as _jdiff
    from protenix_jax.bridge.weights_io import load_native_weights
    from protenix_jax.data.featurize_json import featurize_protein_json, load_first_job
    from protenix_jax.models.predict import protenix_predict_static

    # capture JAX first denoiser output (step-0 x_denoised)
    _jax_denoise0 = []
    _orig_jdm = _jdiff.diffusion_module_forward

    def _jdm_capture(*a, **k):
        out = _orig_jdm(*a, **k)
        if not _jax_denoise0:
            _jax_denoise0.append(np.asarray(jax.block_until_ready(out)))
        return out

    _jdiff.diffusion_module_forward = _jdm_capture

    features = featurize_protein_json(
        load_first_job(INPUT_JSON), base_dir=INPUT_JSON.parent
    )
    params = load_native_weights(WEIGHTS)
    n_atom = int(features["atom_to_token_idx"].shape[-1])
    print("jax n_atom:", n_atom)

    nz = np.load(TORCH_NOISE)
    init_np = nz["init"].astype(np.float32)  # [n_sample, n_atom, 3] (3-D)
    steps_np = nz["steps"].astype(np.float32)  # [n_step, n_sample, n_atom, 3]
    torch_coord = np.load(TORCH_COORD).astype(np.float32)  # [n_sample, n_atom, 3]
    print(
        "torch init", init_np.shape, "steps", steps_np.shape, "coord", torch_coord.shape
    )
    assert init_np.shape[-2] == n_atom, (init_np.shape, n_atom)
    assert steps_np.shape[0] == N_STEP, steps_np.shape

    init_noise = jnp.asarray(init_np)
    step_noises = tuple(jnp.asarray(s) for s in steps_np)

    def run(init_n, step_n):
        out = protenix_predict_static(
            params,
            features,
            key=jax.random.PRNGKey(0),
            n_sample=init_np.shape[0],
            num_sampling_steps=N_STEP,
            recycling_steps=N_CYCLE,
            init_noise=init_n,
            step_noises=step_n,
            run_confidence=False,
            run_confidence_scores=False,
            centre_each_step=True,
        )
        return np.asarray(jax.block_until_ready(out["coordinate"]))

    # ---- matched-noise run (timed for GPU wall-clock) ----
    coord1 = run(init_noise, step_noises)  # warm/compile
    t0 = time.perf_counter()
    coord_timed = run(init_noise, step_noises)
    dt = time.perf_counter() - t0
    jax_coord = coord_timed[0]
    torch_c = torch_coord[0]

    peak = None
    try:
        peak = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception:
        pass

    print("\n=== MATCHED-NOISE RMSD (JAX vs torch) ===")
    print(f"all-atom raw RMSD     : {raw_rmsd(jax_coord, torch_c):.4f} A")
    print(f"all-atom Kabsch RMSD  : {kabsch_rmsd(jax_coord, torch_c):.4f} A")

    # CA atoms: need atom-level mask; use is_ca-like feature if present.
    ca_idx = _ca_indices(features)
    if ca_idx is not None and len(ca_idx) > 0:
        ja, ta = jax_coord[ca_idx], torch_c[ca_idx]
        print(f"CA raw RMSD           : {raw_rmsd(ja, ta):.4f} A  ({len(ca_idx)} CA)")
        print(f"CA Kabsch RMSD        : {kabsch_rmsd(ja, ta):.4f} A")
    else:
        print("CA indices unavailable in features; skipped CA RMSD")

    # ---- determinism: same noise twice ----
    det = raw_rmsd(coord1[0], jax_coord)
    print(
        f"\ndeterminism (same noise twice) max|diff|: "
        f"{np.abs(coord1[0] - jax_coord).max():.2e} A, rmsd {det:.2e}"
    )

    # ---- sensitivity: different noise must change output ----
    rng = np.random.default_rng(1)
    alt_init = jnp.asarray(rng.standard_normal(init_np.shape).astype(np.float32))
    alt_steps = tuple(
        jnp.asarray(rng.standard_normal(s.shape).astype(np.float32)) for s in steps_np
    )
    coord_alt = run(alt_init, alt_steps)[0]
    print(
        f"sensitivity (diff noise) rmsd vs matched: "
        f"{raw_rmsd(coord_alt, jax_coord):.3f} A"
    )

    # ---- step-0 pre-model parity: analytic x_noisy vs torch captured ----
    d0 = SC / "torch_denoise0.npz"
    if d0.exists():
        dz = np.load(d0)
        sched = np.asarray(
            __import__(
                "protenix_jax.models.diffusion.diffusion", fromlist=["x"]
            ).inference_noise_schedule(n_step=N_STEP)
        )
        gamma0, gamma_min, lam = 0.8, 1.0, 1.003
        c0 = sched[0]
        x_l0 = c0 * init_np[0]
        x_l0 = x_l0 - x_l0.mean(0, keepdims=True)  # centre_only step 0
        gamma = gamma0 if c0 > gamma_min else 0.0
        t_hat = c0 * (gamma + 1.0)
        dnl = np.sqrt(max(t_hat**2 - c0**2, 0.0))
        x_noisy0 = x_l0 + lam * dnl * steps_np[0, 0]
        tx = dz["x_noisy"][0]
        print("\n=== STEP-0 PRE-MODEL PARITY ===")
        print(
            f"analytic x_noisy vs torch x_noisy raw RMSD: "
            f"{raw_rmsd(x_noisy0, tx):.2e} A (t_hat jax {t_hat:.4f} "
            f"torch {float(dz['t_hat'].reshape(-1)[0]):.4f})"
        )

    # ---- step-0 denoiser-output parity (model+trunk) ----
    if d0.exists() and _jax_denoise0:
        jden = _jax_denoise0[0]
        jden = jden[0] if jden.ndim == 3 else jden[0, 0]
        tden = dz["x_denoised"]
        tden = tden[0] if tden.ndim == 3 else tden[0, 0]
        print("=== STEP-0 DENOISER OUTPUT PARITY (model+trunk) ===")
        print(
            f"jax x_denoised vs torch x_denoised raw RMSD : "
            f"{raw_rmsd(jden, tden):.4f} A"
        )
        print(
            f"jax x_denoised vs torch x_denoised Kabsch   : "
            f"{kabsch_rmsd(jden, tden):.4f} A"
        )

    print("\n=== JAX GPU TIMING ===")
    print(f"wall-clock (warm) : {dt:.3f} s  (n_step={N_STEP}, n_cycle={N_CYCLE})")
    if peak is not None:
        print(f"peak VRAM         : {peak / 1e9:.3f} GB")


def _ca_indices(features):
    """Best-effort CA atom indices from ref_atom_name_chars if available."""
    names = features.get("ref_atom_name_chars")
    if names is None:
        return None
    arr = np.asarray(names)
    # ref_atom_name_chars: [n_atom, 4, 64] one-hot of 4 chars (offset 32).
    # "CA" -> chars 'C','A' then padding.
    if arr.ndim == 3:
        chars = arr.argmax(-1) + 32  # back to ascii
        is_ca = (
            (chars[:, 0] == ord("C"))
            & (chars[:, 1] == ord("A"))
            & (chars[:, 2] == ord(" "))
        )
        return np.where(is_ca)[0]
    return None


if __name__ == "__main__":
    main()
