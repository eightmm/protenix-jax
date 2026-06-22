"""Diffusion conditioning blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from protenix_jax.models.diffusion.atom import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderParams,
    atom_attention_decoder,
    atom_attention_encoder,
)
from protenix_jax.models.diffusion.transformer import (
    DiffusionTransformerStackParams,
    diffusion_transformer_stack,
)
from protenix_jax.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    TransitionParams,
    layer_norm,
    linear,
    transition,
)
from protenix_jax.models.trunk_blocks.embedders import (
    FourierParams,
    RelativePositionParams,
    fourier_embedding,
    relative_position_encoding,
)


class DiffusionConditioningParams(NamedTuple):
    """Parameters for Protenix ``DiffusionConditioning``."""

    relpe: RelativePositionParams
    layernorm_z: LayerNormParams
    linear_z: LinearParams
    transition_z1: TransitionParams
    transition_z2: TransitionParams
    layernorm_s: LayerNormParams
    linear_s: LinearParams
    fourier: FourierParams
    layernorm_n: LayerNormParams
    linear_n: LinearParams
    transition_s1: TransitionParams
    transition_s2: TransitionParams


class DiffusionModuleParams(NamedTuple):
    """Parameters for infer-only Protenix ``DiffusionModule``."""

    conditioning: DiffusionConditioningParams
    atom_encoder: AtomAttentionEncoderParams
    layernorm_s: LayerNormParams
    linear_s: LinearParams
    diffusion_transformer: DiffusionTransformerStackParams
    layernorm_a: LayerNormParams
    atom_decoder: AtomAttentionDecoderParams


def inference_noise_schedule(
    *,
    n_step: int = 200,
    s_max: float = 160.0,
    s_min: float = 4.0e-4,
    rho: float = 7.0,
    sigma_data: float = 16.0,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Return the Protenix inference noise schedule."""

    step_indices = jnp.arange(n_step + 1, dtype=dtype)
    schedule = sigma_data * (
        s_max ** (1.0 / rho)
        + step_indices
        / jnp.asarray(n_step, dtype=dtype)
        * (s_min ** (1.0 / rho) - s_max ** (1.0 / rho))
    ) ** rho
    return schedule.at[-1].set(0.0)


def centre_random_augmentation(x: jnp.ndarray) -> jnp.ndarray:
    """Center coordinates over the atom axis, matching inference centering."""

    return x - jnp.mean(x, axis=-2, keepdims=True)


def sample_diffusion(
    denoise_fn,
    noise_schedule: jnp.ndarray,
    *,
    n_sample: int,
    n_atom: int,
    key: jax.Array | None,
    init_noise: jnp.ndarray | None = None,
    step_noises: tuple[jnp.ndarray, ...] | None = None,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: int | None = None,
    dtype: jnp.dtype = jnp.float32,
    centre_each_step: bool = True,
) -> jnp.ndarray:
    """Run Protenix Algorithm 18 diffusion sampling with a JAX denoiser."""

    if diffusion_chunk_size is None or diffusion_chunk_size <= 0:
        return _sample_diffusion_chunk(
            denoise_fn,
            noise_schedule,
            n_sample=n_sample,
            n_atom=n_atom,
            key=key,
            init_noise=init_noise,
            step_noises=step_noises,
            gamma0=gamma0,
            gamma_min=gamma_min,
            noise_scale_lambda=noise_scale_lambda,
            step_scale_eta=step_scale_eta,
            dtype=dtype,
            centre_each_step=centre_each_step,
        )

    outputs = []
    keys = None
    if key is not None:
        n_chunks = (n_sample + diffusion_chunk_size - 1) // diffusion_chunk_size
        keys = jax.random.split(key, n_chunks)
    for chunk_index, start in enumerate(range(0, n_sample, diffusion_chunk_size)):
        chunk_n = min(diffusion_chunk_size, n_sample - start)
        init_chunk = None
        if init_noise is not None:
            init_chunk = _slice_sample_axis(init_noise, start, chunk_n)
        step_chunks = None
        if step_noises is not None:
            step_chunks = tuple(
                _slice_sample_axis(noise, start, chunk_n) for noise in step_noises
            )
        outputs.append(
            _sample_diffusion_chunk(
                denoise_fn,
                noise_schedule,
                n_sample=chunk_n,
                n_atom=n_atom,
                key=None if keys is None else keys[chunk_index],
                init_noise=init_chunk,
                step_noises=step_chunks,
                gamma0=gamma0,
                gamma_min=gamma_min,
                noise_scale_lambda=noise_scale_lambda,
                step_scale_eta=step_scale_eta,
                dtype=dtype,
                centre_each_step=centre_each_step,
            )
        )
    return jnp.concatenate(outputs, axis=-3)


def sample_diffusion_with_module(
    input_feature_dict: dict[str, jnp.ndarray | dict[str, jnp.ndarray]],
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionModuleParams,
    noise_schedule: jnp.ndarray,
    *,
    n_sample: int,
    key: jax.Array | None,
    init_noise: jnp.ndarray | None = None,
    step_noises: tuple[jnp.ndarray, ...] | None = None,
    pair_z: jnp.ndarray | None = None,
    p_lm: jnp.ndarray | None = None,
    c_l: jnp.ndarray | None = None,
    atom_encoder_heads: int = 4,
    token_heads: int = 16,
    atom_decoder_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    sigma_data: float = 16.0,
    use_conditioning: bool = True,
    use_scan: bool = False,
    token_q_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    dtype: jnp.dtype = jnp.float32,
    centre_each_step: bool = True,
) -> jnp.ndarray:
    """Sample coordinates using ``DiffusionModuleParams`` and static features."""

    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    n_atom = int(atom_to_token_idx.shape[-1])
    n_token = int(s_inputs.shape[-2])

    def denoise_fn(x_noisy: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
        return diffusion_module_forward(
            atom_to_token_idx,
            input_feature_dict["ref_pos"],
            input_feature_dict["ref_charge"],
            input_feature_dict["ref_mask"],
            input_feature_dict["ref_atom_name_chars"],
            input_feature_dict["ref_element"],
            input_feature_dict["d_lm"],
            input_feature_dict["v_lm"],
            input_feature_dict["pad_info"],
            x_noisy,
            t_hat,
            input_feature_dict["relp"],
            s_inputs,
            s_trunk,
            z_trunk,
            params,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            n_token=n_token,
            atom_encoder_heads=atom_encoder_heads,
            token_heads=token_heads,
            atom_decoder_heads=atom_decoder_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            sigma_data=sigma_data,
            use_conditioning=use_conditioning,
            use_scan=use_scan,
            token_q_chunk_size=token_q_chunk_size,
        )

    return sample_diffusion(
        denoise_fn,
        noise_schedule,
        n_sample=n_sample,
        n_atom=n_atom,
        key=key,
        init_noise=init_noise,
        step_noises=step_noises,
        gamma0=gamma0,
        gamma_min=gamma_min,
        noise_scale_lambda=noise_scale_lambda,
        step_scale_eta=step_scale_eta,
        diffusion_chunk_size=diffusion_chunk_size,
        dtype=dtype,
        centre_each_step=centre_each_step,
    )


def diffusion_conditioning_prepare_cache(
    relp_feature: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionConditioningParams,
) -> jnp.ndarray:
    """Build diffusion pair conditioning cache."""

    pair_z = jnp.concatenate(
        [z_trunk, relative_position_encoding(relp_feature, params.relpe)],
        axis=-1,
    )
    pair_z = linear(layer_norm(pair_z, params.layernorm_z), params.linear_z)
    pair_z = pair_z + transition(pair_z, params.transition_z1)
    return pair_z + transition(pair_z, params.transition_z2)


def diffusion_conditioning(
    t_hat_noise_level: jnp.ndarray,
    relp_feature: jnp.ndarray,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionConditioningParams,
    *,
    pair_z: jnp.ndarray | None = None,
    sigma_data: float = 16.0,
    use_conditioning: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Protenix ``DiffusionConditioning`` in inference mode."""

    if pair_z is None:
        if not use_conditioning:
            s_trunk = jnp.zeros_like(s_trunk)
            z_trunk = jnp.zeros_like(z_trunk)
        pair_z = diffusion_conditioning_prepare_cache(relp_feature, z_trunk, params)

    single_s = jnp.concatenate([s_trunk, s_inputs], axis=-1)
    single_s = linear(layer_norm(single_s, params.layernorm_s), params.linear_s)
    noise = jnp.log(t_hat_noise_level / sigma_data) / 4.0
    noise = fourier_embedding(noise, params.fourier).astype(single_s.dtype)
    noise = linear(layer_norm(noise, params.layernorm_n), params.linear_n)
    single_s = single_s[..., None, :, :] + noise[..., :, None, :]
    single_s = single_s + transition(single_s, params.transition_s1)
    single_s = single_s + transition(single_s, params.transition_s2)
    return single_s, pair_z


def diffusion_module_f_forward(
    atom_to_token_idx: jnp.ndarray,
    ref_pos: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_mask: jnp.ndarray,
    ref_atom_name_chars: jnp.ndarray,
    ref_element: jnp.ndarray,
    d_lm: jnp.ndarray,
    v_lm: jnp.ndarray,
    pad_info: dict[str, jnp.ndarray],
    r_noisy: jnp.ndarray,
    t_hat_noise_level: jnp.ndarray,
    relp_feature: jnp.ndarray,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionModuleParams,
    *,
    pair_z: jnp.ndarray | None = None,
    p_lm: jnp.ndarray | None = None,
    c_l: jnp.ndarray | None = None,
    n_token: int,
    atom_encoder_heads: int,
    token_heads: int,
    atom_decoder_heads: int,
    n_queries: int,
    n_keys: int,
    sigma_data: float = 16.0,
    use_conditioning: bool = True,
    use_scan: bool = False,
    token_q_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run the raw Protenix denoising network ``F`` for one noise level."""

    single_s, pair_z = diffusion_conditioning(
        t_hat_noise_level,
        relp_feature,
        s_inputs,
        s_trunk,
        z_trunk,
        params.conditioning,
        pair_z=pair_z,
        sigma_data=sigma_data,
        use_conditioning=use_conditioning,
    )
    s_trunk_sample = jnp.expand_dims(s_trunk, axis=-3)
    z_pair_sample = jnp.expand_dims(pair_z, axis=-4)
    a_token, q_skip, c_skip, p_skip = atom_attention_encoder(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        params.atom_encoder,
        r_l=r_noisy,
        s=s_trunk_sample,
        z=z_pair_sample,
        p_lm=p_lm,
        c_l=c_l,
        n_token=n_token,
        n_heads=atom_encoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_scan,
    )
    a_token = a_token.astype(jnp.float32)
    a_token = a_token + linear(
        layer_norm(single_s, params.layernorm_s),
        params.linear_s,
    )
    a_token = diffusion_transformer_stack(
        a_token.astype(jnp.float32),
        single_s.astype(jnp.float32),
        z_pair_sample.astype(jnp.float32),
        params.diffusion_transformer,
        num_heads=token_heads,
        use_scan=use_scan,
        global_q_chunk_size=token_q_chunk_size,
    )
    a_token = layer_norm(a_token, params.layernorm_a)
    return atom_attention_decoder(
        atom_to_token_idx,
        a_token,
        q_skip,
        c_skip,
        p_skip,
        params.atom_decoder,
        n_heads=atom_decoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_scan,
    )


def diffusion_module_forward(
    atom_to_token_idx: jnp.ndarray,
    ref_pos: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_mask: jnp.ndarray,
    ref_atom_name_chars: jnp.ndarray,
    ref_element: jnp.ndarray,
    d_lm: jnp.ndarray,
    v_lm: jnp.ndarray,
    pad_info: dict[str, jnp.ndarray],
    x_noisy: jnp.ndarray,
    t_hat_noise_level: jnp.ndarray,
    relp_feature: jnp.ndarray,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionModuleParams,
    *,
    pair_z: jnp.ndarray | None = None,
    p_lm: jnp.ndarray | None = None,
    c_l: jnp.ndarray | None = None,
    n_token: int,
    atom_encoder_heads: int,
    token_heads: int,
    atom_decoder_heads: int,
    n_queries: int,
    n_keys: int,
    sigma_data: float = 16.0,
    use_conditioning: bool = True,
    use_scan: bool = False,
    token_q_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run one Protenix EDM denoising step."""

    scale = jnp.sqrt(sigma_data**2 + t_hat_noise_level**2)[..., None, None]
    r_noisy = x_noisy / scale
    r_update = diffusion_module_f_forward(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        r_noisy,
        t_hat_noise_level,
        relp_feature,
        s_inputs,
        s_trunk,
        z_trunk,
        params,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        n_token=n_token,
        atom_encoder_heads=atom_encoder_heads,
        token_heads=token_heads,
        atom_decoder_heads=atom_decoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        sigma_data=sigma_data,
        use_conditioning=use_conditioning,
        use_scan=use_scan,
        token_q_chunk_size=token_q_chunk_size,
    )
    s_ratio = (t_hat_noise_level / sigma_data)[..., None, None].astype(r_update.dtype)
    return (
        x_noisy / (1.0 + s_ratio**2)
        + t_hat_noise_level[..., None, None] / jnp.sqrt(1.0 + s_ratio**2) * r_update
    ).astype(r_update.dtype)


def _sample_diffusion_chunk(
    denoise_fn,
    noise_schedule: jnp.ndarray,
    *,
    n_sample: int,
    n_atom: int,
    key: jax.Array | None,
    init_noise: jnp.ndarray | None,
    step_noises: tuple[jnp.ndarray, ...] | None,
    gamma0: float,
    gamma_min: float,
    noise_scale_lambda: float,
    step_scale_eta: float,
    dtype: jnp.dtype,
    centre_each_step: bool,
) -> jnp.ndarray:
    n_steps = int(noise_schedule.shape[0]) - 1
    if init_noise is None:
        if key is None:
            raise ValueError("key is required when init_noise is not provided")
        key, init_key = jax.random.split(key)
        init_noise = jax.random.normal(init_key, (n_sample, n_atom, 3), dtype=dtype)
    x_l = noise_schedule[0].astype(dtype) * init_noise.astype(dtype)

    if step_noises is None:
        if key is None:
            raise ValueError("key is required when step_noises is not provided")
        step_keys = jax.random.split(key, n_steps)
        step_noises = tuple(
            jax.random.normal(step_key, x_l.shape, dtype=dtype)
            for step_key in step_keys
        )
    if len(step_noises) != n_steps:
        raise ValueError("step_noises length must equal len(noise_schedule) - 1")

    for step_i in range(n_steps):
        c_tau_last = noise_schedule[step_i].astype(dtype)
        c_tau = noise_schedule[step_i + 1].astype(dtype)
        if centre_each_step:
            x_l = centre_random_augmentation(x_l)
        gamma = jnp.where(c_tau > gamma_min, gamma0, 0.0).astype(dtype)
        t_hat_scalar = c_tau_last * (gamma + 1.0)
        delta_noise_level = jnp.sqrt(t_hat_scalar**2 - c_tau_last**2)
        x_noisy = x_l + noise_scale_lambda * delta_noise_level * step_noises[step_i]
        t_hat = jnp.full(x_noisy.shape[:-2], t_hat_scalar, dtype=dtype)
        x_denoised = denoise_fn(x_noisy, t_hat)
        delta = (x_noisy - x_denoised) / t_hat[..., None, None]
        dt = c_tau - t_hat
        x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta
    return x_l


def _slice_sample_axis(x: jnp.ndarray, start: int, size: int) -> jnp.ndarray:
    if x.ndim < 3:
        raise ValueError("sample noise must have at least sample/atom/coord axes")
    return x[..., start : start + size, :, :] if x.ndim > 3 else x[start : start + size]
