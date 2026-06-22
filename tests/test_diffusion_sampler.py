from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from protenix_jax.models.diffusion.diffusion import (
    inference_noise_schedule,
    sample_diffusion,
)


def test_inference_noise_schedule_matches_protenix_formula() -> None:
    schedule = inference_noise_schedule(
        n_step=4,
        s_max=10.0,
        s_min=0.1,
        rho=2.0,
        sigma_data=3.0,
        dtype=jnp.float32,
    )
    indices = np.arange(5, dtype=np.float32)
    expected = 3.0 * (
        10.0 ** 0.5 + indices / 4.0 * (0.1**0.5 - 10.0**0.5)
    ) ** 2
    expected[-1] = 0.0

    np.testing.assert_allclose(np.asarray(schedule), expected, rtol=1e-6, atol=1e-6)


def test_sample_diffusion_zero_denoiser_matches_euler_formula() -> None:
    noise_schedule = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    init_noise = jnp.asarray([[[[1.0, -1.0, 0.5], [0.0, 2.0, -2.0]]]])
    step_noise = jnp.zeros_like(init_noise)

    def zero_denoiser(x_noisy, t_hat):
        assert x_noisy.shape == init_noise.shape
        assert t_hat.shape == (1, 1)
        return jnp.zeros_like(x_noisy)

    out = sample_diffusion(
        zero_denoiser,
        noise_schedule,
        n_sample=1,
        n_atom=2,
        key=None,
        init_noise=init_noise,
        step_noises=(step_noise,),
        gamma0=0.5,
        gamma_min=0.5,
        noise_scale_lambda=1.0,
        step_scale_eta=1.5,
        centre_each_step=False,
    )

    x_l = 2.0 * np.asarray(init_noise)
    t_hat = 2.0 * 1.5
    x_noisy = x_l
    delta = x_noisy / t_hat
    expected = x_noisy + 1.5 * (1.0 - t_hat) * delta
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-6, atol=1e-6)


def test_sample_diffusion_chunks_samples() -> None:
    noise_schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.ones((3, 2, 3), dtype=jnp.float32)
    step_noise = jnp.zeros_like(init_noise)

    def identity_denoiser(x_noisy, t_hat):
        del t_hat
        return x_noisy

    out = sample_diffusion(
        identity_denoiser,
        noise_schedule,
        n_sample=3,
        n_atom=2,
        key=None,
        init_noise=init_noise,
        step_noises=(step_noise,),
        diffusion_chunk_size=2,
        centre_each_step=False,
    )

    assert out.shape == (3, 2, 3)
    np.testing.assert_allclose(np.asarray(out), np.asarray(init_noise), atol=1e-6)
