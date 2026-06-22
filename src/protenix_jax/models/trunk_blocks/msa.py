"""MSA trunk blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import nn as jnn

from protenix_jax.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    TransitionParams,
    layer_norm,
    linear,
    sigmoid,
    transition,
)
from protenix_jax.models.trunk_blocks.pairformer import (
    PairformerBlockParams,
    pairformer_block,
)


class OuterProductMeanParams(NamedTuple):
    """Parameters for Protenix ``OuterProductMean``."""

    layer_norm: LayerNormParams
    linear_1: LinearParams
    linear_2: LinearParams
    linear_out: LinearParams


class MSAPairWeightedAveragingParams(NamedTuple):
    """Parameters for ``MSAPairWeightedAveraging``."""

    layernorm_m: LayerNormParams
    linear_mv: LinearParams
    layernorm_z: LayerNormParams
    linear_z: LinearParams
    linear_mg: LinearParams
    linear_out: LinearParams


class MSABlockParams(NamedTuple):
    """Parameters for one Protenix MSA block."""

    outer_product_mean: OuterProductMeanParams
    msa_pair_weighted_averaging: MSAPairWeightedAveragingParams | None
    msa_transition: TransitionParams | None
    pair_stack: PairformerBlockParams


class MSAModuleParams(NamedTuple):
    """Parameters for Protenix ``MSAModule``."""

    linear_m: LinearParams
    linear_s: LinearParams
    blocks: tuple[MSABlockParams, ...]


def outer_product_mean(
    m: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: OuterProductMeanParams,
    *,
    eps: float = 1e-3,
) -> jnp.ndarray:
    """Apply Protenix ``OuterProductMean`` in dense inference mode."""

    if mask is None:
        mask = jnp.ones(m.shape[:-1], dtype=m.dtype)
    mask = mask.astype(m.dtype)

    m_norm = layer_norm(m, params.layer_norm)
    a = linear(m_norm, params.linear_1) * mask[..., None]
    b = linear(m_norm, params.linear_2) * mask[..., None]
    outer = jnp.einsum("...mic,...mjd->...ijcd", a, b)
    outer = outer.reshape(outer.shape[:-2] + (-1,))
    outer = linear(outer, params.linear_out)
    norm = jnp.einsum("...mi,...mj->...ij", mask, mask)[..., None] + eps
    return outer / norm


def msa_pair_weighted_averaging(
    m: jnp.ndarray,
    z: jnp.ndarray,
    params: MSAPairWeightedAveragingParams,
) -> jnp.ndarray:
    """Apply inference-mode ``MSAPairWeightedAveraging``."""

    m_norm = layer_norm(m, params.layernorm_m)
    num_heads = int(params.linear_z.weight.shape[0])
    v = linear(m_norm, params.linear_mv)
    v = v.reshape(v.shape[:-1] + (num_heads, -1))
    b = linear(layer_norm(z, params.layernorm_z), params.linear_z)
    weights = jnn.softmax(b.astype(jnp.float32), axis=-2).astype(v.dtype)
    gate = sigmoid(linear(m_norm, params.linear_mg))
    gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
    out = gate * jnp.einsum("...ijh,...mjhc->...mihc", weights, v)
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.linear_out)


def msa_block(
    m: jnp.ndarray | None,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    params: MSABlockParams,
    *,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
) -> tuple[jnp.ndarray | None, jnp.ndarray]:
    """Apply one inference-mode Protenix MSA block."""

    if m is None:
        raise ValueError("MSABlock requires m before the final block output")
    z = z + outer_product_mean(m, None, params.outer_product_mean)
    if params.msa_pair_weighted_averaging is not None:
        m = m + msa_pair_weighted_averaging(
            m,
            z,
            params.msa_pair_weighted_averaging,
        )
        if params.msa_transition is None:
            raise ValueError("missing MSA transition for non-final MSA block")
        m = m + transition(m, params.msa_transition)
    _, z = pairformer_block(
        None,
        z,
        pair_mask,
        params.pair_stack,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
    )
    if params.msa_pair_weighted_averaging is None:
        return None, z
    return m, z


def msa_module(
    input_feature_dict: dict[str, jnp.ndarray],
    z: jnp.ndarray,
    s_inputs: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    params: MSAModuleParams,
    *,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply Protenix ``MSAModule`` to already-materialized MSA features."""

    if not params.blocks or "msa" not in input_feature_dict:
        return z
    msa = input_feature_dict["msa"]
    if msa.ndim < 2:
        return z

    msa_one_hot = jnp.eye(32, dtype=s_inputs.dtype)[msa]
    target_shape = msa_one_hot.shape[:-1]
    msa_sample = jnp.concatenate(
        [
            msa_one_hot,
            input_feature_dict["has_deletion"].reshape(target_shape + (1,)),
            input_feature_dict["deletion_value"].reshape(target_shape + (1,)),
        ],
        axis=-1,
    )
    m = linear(msa_sample, params.linear_m)
    m = m + linear(s_inputs, params.linear_s)[..., None, :, :]

    for block_params in params.blocks:
        m, z = msa_block(
            m,
            z,
            pair_mask,
            block_params,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        )
    return z
