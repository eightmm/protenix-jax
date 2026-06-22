"""Single Protenix inference wrapper.

This module is the torch-free library API equivalent of the static inference
CLI: static features + JAX params + PRNG key -> one prediction dictionary.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from protenix_jax.models.diffusion.diffusion import inference_noise_schedule
from protenix_jax.models.model import ProtenixInferenceParams, protenix_infer_static


def protenix_predict_static(
    params: ProtenixInferenceParams,
    features: Mapping[str, jnp.ndarray | Mapping[str, jnp.ndarray]],
    key: jax.Array | None,
    *,
    n_sample: int = 1,
    num_sampling_steps: int = 200,
    s_max: float = 160.0,
    s_min: float = 4.0e-4,
    rho: float = 7.0,
    sigma_data: float = 16.0,
    recycling_steps: int = 10,
    init_noise: jnp.ndarray | None = None,
    step_noises: tuple[jnp.ndarray, ...] | None = None,
    pair_mask: jnp.ndarray | None = None,
    input_atom_heads: int = 4,
    atom_encoder_heads: int = 4,
    token_heads: int = 16,
    atom_decoder_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    use_pairformer_scan: bool = True,
    use_confidence_scan: bool = True,
    use_diffusion_scan: bool = False,
    use_confidence_embedding: bool = True,
    run_confidence: bool = True,
    run_confidence_scores: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    token_q_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    centre_each_step: bool = True,
) -> dict[str, jnp.ndarray]:
    """Run the static-feature Protenix inference graph."""

    noise_schedule = inference_noise_schedule(
        n_step=num_sampling_steps,
        s_max=s_max,
        s_min=s_min,
        rho=rho,
        sigma_data=sigma_data,
    )
    return protenix_infer_static(
        dict(features),
        params,
        noise_schedule,
        key=key,
        n_sample=n_sample,
        init_noise=init_noise,
        step_noises=step_noises,
        n_cycle=recycling_steps,
        pair_mask=pair_mask,
        input_atom_heads=input_atom_heads,
        atom_encoder_heads=atom_encoder_heads,
        token_heads=token_heads,
        atom_decoder_heads=atom_decoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        sigma_data=sigma_data,
        use_pairformer_scan=use_pairformer_scan,
        use_confidence_scan=use_confidence_scan,
        use_diffusion_scan=use_diffusion_scan,
        use_confidence_embedding=use_confidence_embedding,
        run_confidence=run_confidence,
        run_confidence_scores=run_confidence_scores,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        single_att_q_chunk_size=single_att_q_chunk_size,
        token_q_chunk_size=token_q_chunk_size,
        diffusion_chunk_size=diffusion_chunk_size,
        gamma0=gamma0,
        gamma_min=gamma_min,
        noise_scale_lambda=noise_scale_lambda,
        step_scale_eta=step_scale_eta,
        centre_each_step=centre_each_step,
    )

