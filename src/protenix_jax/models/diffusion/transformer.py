"""Transformer leaf blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from protenix_jax.models.primitives.attention import (
    AttentionPairBiasParams,
    attention_pair_bias,
    local_attention_pair_bias,
)
from protenix_jax.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LinearParams,
    adaptive_layer_norm,
    linear,
    sigmoid,
    silu,
)


class ConditionedTransitionParams(NamedTuple):
    """Parameters for ``ConditionedTransitionBlock``."""

    adaln: AdaptiveLayerNormParams
    linear_a1: LinearParams
    linear_a2: LinearParams
    linear_b: LinearParams
    linear_s: LinearParams


class DiffusionTransformerBlockParams(NamedTuple):
    """Parameters for one Protenix ``DiffusionTransformerBlock``."""

    attention_pair_bias: AttentionPairBiasParams
    conditioned_transition: ConditionedTransitionParams


class DiffusionTransformerStackParams(NamedTuple):
    """Parameters for a Protenix ``DiffusionTransformer`` stack."""

    blocks: tuple[DiffusionTransformerBlockParams, ...]


def conditioned_transition_block(
    a: jnp.ndarray,
    s: jnp.ndarray,
    params: ConditionedTransitionParams,
) -> jnp.ndarray:
    """Apply Protenix ``ConditionedTransitionBlock``."""

    a = adaptive_layer_norm(a, s, params.adaln)
    hidden = silu(linear(a, params.linear_a1)) * linear(a, params.linear_a2)
    return sigmoid(linear(s, params.linear_s)) * linear(hidden, params.linear_b)


def diffusion_transformer_block(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: DiffusionTransformerBlockParams,
    *,
    num_heads: int,
    n_queries: int | None = None,
    n_keys: int | None = None,
    global_q_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply one inference-mode DiffusionTransformer block."""

    if n_queries is not None and n_keys is not None:
        attn_out = local_attention_pair_bias(
            a,
            s,
            z,
            params.attention_pair_bias,
            num_heads=num_heads,
            n_queries=n_queries,
            n_keys=n_keys,
        )
    else:
        attn_out = attention_pair_bias(
            a,
            s,
            z,
            params.attention_pair_bias,
            num_heads=num_heads,
            q_chunk_size=global_q_chunk_size,
        )
    a = a + attn_out
    return a + conditioned_transition_block(a, s, params.conditioned_transition)


def diffusion_transformer_stack(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: DiffusionTransformerStackParams,
    *,
    num_heads: int,
    n_queries: int | None = None,
    n_keys: int | None = None,
    use_scan: bool = False,
    global_q_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply a DiffusionTransformer stack in inference mode."""

    if not params.blocks:
        raise ValueError("DiffusionTransformerStack requires at least one block")
    if not use_scan:
        for block in params.blocks:
            a = diffusion_transformer_block(
                a,
                s,
                z,
                block,
                num_heads=num_heads,
                n_queries=n_queries,
                n_keys=n_keys,
                global_q_chunk_size=global_q_chunk_size,
            )
        return a

    stacked = stack_diffusion_transformer_block_params(params.blocks)

    def body(a_carry, block_params):
        return (
            diffusion_transformer_block(
                a_carry,
                s,
                z,
                block_params,
                num_heads=num_heads,
                n_queries=n_queries,
                n_keys=n_keys,
                global_q_chunk_size=global_q_chunk_size,
            ),
            None,
        )

    a, _ = jax.lax.scan(body, a, stacked)
    return a


def stack_diffusion_transformer_block_params(
    blocks: tuple[DiffusionTransformerBlockParams, ...],
) -> DiffusionTransformerBlockParams:
    """Stack DiffusionTransformer block params for ``lax.scan``."""

    if not blocks:
        raise ValueError("stack_diffusion_transformer_block_params requires blocks")
    return jax.tree.map(_stack_param_leaf, *blocks, is_leaf=lambda x: x is None)


def _stack_param_leaf(*leaves):
    first = leaves[0]
    if first is None:
        if any(leaf is not None for leaf in leaves):
            raise ValueError("cannot stack mixed None/non-None leaves")
        return None
    if isinstance(first, bool):
        return jnp.asarray(leaves, dtype=bool)
    return jnp.stack(leaves, axis=0)
