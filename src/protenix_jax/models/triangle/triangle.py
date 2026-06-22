"""Triangle pair-update blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from protenix_jax.models.primitives.attention import AttentionParams
from protenix_jax.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
    sigmoid,
)

TriangleDirection = Literal["outgoing", "incoming"]


class TriangleMultiplicationParams(NamedTuple):
    """Parameters for ``TriangleMultiplicativeUpdate``."""

    layer_norm_in: LayerNormParams
    layer_norm_out: LayerNormParams
    linear_a_p: LinearParams
    linear_a_g: LinearParams
    linear_b_p: LinearParams
    linear_b_g: LinearParams
    linear_z: LinearParams
    linear_g: LinearParams


class TriangleAttentionParams(NamedTuple):
    """Parameters for Protenix ``TriangleAttention``."""

    layer_norm: LayerNormParams
    linear: LinearParams
    attention: AttentionParams


def triangle_multiplication(
    z: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: TriangleMultiplicationParams,
    direction: TriangleDirection,
    *,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply Protenix triangle multiplication without eval in-place mutation."""

    if mask is None:
        mask = jnp.ones(z.shape[:-1], dtype=z.dtype)
    mask = mask.astype(z.dtype)[..., None]

    z_norm = layer_norm(z, params.layer_norm_in)
    a = mask * sigmoid(linear(z_norm, params.linear_a_g))
    a = a * linear(z_norm, params.linear_a_p)
    b = mask * sigmoid(linear(z_norm, params.linear_b_g))
    b = b * linear(z_norm, params.linear_b_p)

    out = _triangle_contract(
        a.astype(jnp.float32),
        b.astype(jnp.float32),
        direction,
        chunk_size,
    )
    out = out.astype(z.dtype)
    out = layer_norm(out, params.layer_norm_out)
    out = linear(out, params.linear_z)
    return out * sigmoid(linear(z_norm, params.linear_g))


def _triangle_contract(
    a: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
    chunk_size: int | None,
) -> jnp.ndarray:
    if chunk_size is None or chunk_size <= 0:
        return _triangle_contract_block(a, b, direction)

    n = a.shape[-3]
    if chunk_size >= n:
        return _triangle_contract_block(a, b, direction)

    out = jnp.zeros(a.shape[:-3] + (n, n, a.shape[-1]), dtype=a.dtype)
    for start in range(0, n, chunk_size):
        size = min(chunk_size, n - start)
        if direction == "outgoing":
            a_block = jax.lax.dynamic_slice_in_dim(a, start, size, axis=-3)
            block = jnp.einsum("...ikd,...jkd->...ijd", a_block, b)
        elif direction == "incoming":
            a_block = jax.lax.dynamic_slice_in_dim(a, start, size, axis=-2)
            block = jnp.einsum("...kid,...kjd->...ijd", a_block, b)
        else:
            raise ValueError(f"unsupported triangle direction: {direction!r}")
        out = out.at[..., start : start + size, :, :].set(block)
    return out


def _triangle_contract_block(
    a: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
) -> jnp.ndarray:
    if direction == "outgoing":
        return jnp.einsum("...ikd,...jkd->...ijd", a, b)
    if direction == "incoming":
        return jnp.einsum("...kid,...kjd->...ijd", a, b)
    raise ValueError(f"unsupported triangle direction: {direction!r}")


def triangle_attention(
    x: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: TriangleAttentionParams,
    *,
    num_heads: int,
    starting: bool = True,
    inf: float = 1e9,
    q_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply Protenix triangle attention in the dense XLA path."""

    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    x = layer_norm(x, params.layer_norm)
    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)
    mask_bias = mask_bias[..., :, None, None, :]
    triangle_bias = linear(x, params.linear)
    triangle_bias = jnp.moveaxis(triangle_bias, -1, -3)
    triangle_bias = jnp.expand_dims(triangle_bias, axis=-4)

    out = _triangle_attention_dense(
        x,
        params,
        num_heads,
        mask_bias,
        triangle_bias,
        q_chunk_size,
    )
    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return out


def _triangle_attention_dense(
    x: jnp.ndarray,
    params: TriangleAttentionParams,
    num_heads: int,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    q_chunk_size: int | None,
) -> jnp.ndarray:
    q = _project_heads(x, params.attention.linear_q, num_heads)
    k = _project_heads(x, params.attention.linear_k, num_heads)
    v = _project_heads(x, params.attention.linear_v, num_heads)
    q = q / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype))

    if q_chunk_size is None or q_chunk_size <= 0 or q_chunk_size >= q.shape[-2]:
        out = _triangle_attention_block(q, k, v, mask_bias, triangle_bias)
    else:
        out = jnp.zeros(q.shape, dtype=q.dtype)
        for start in range(0, q.shape[-2], q_chunk_size):
            size = min(q_chunk_size, q.shape[-2] - start)
            q_block = jax.lax.dynamic_slice_in_dim(q, start, size, axis=-2)
            mask_block = jax.lax.dynamic_slice_in_dim(
                mask_bias,
                start,
                size,
                axis=-2,
            )
            tri_block = jax.lax.dynamic_slice_in_dim(
                triangle_bias,
                start,
                size,
                axis=-2,
            )
            block = _triangle_attention_block(q_block, k, v, mask_block, tri_block)
            out = out.at[..., start : start + size, :].set(block)

    out = jnp.swapaxes(out, -2, -3)
    if params.attention.linear_g is not None:
        gate = sigmoid(linear(x, params.attention.linear_g))
        gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.attention.linear_o)


def _triangle_attention_block(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
) -> jnp.ndarray:
    logits = jnp.einsum("...hid,...hjd->...hij", q, k)
    logits = logits + mask_bias + triangle_bias
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
    return jnp.einsum("...hij,...hjd->...hid", probs, v)


def _project_heads(
    x: jnp.ndarray,
    params: LinearParams,
    num_heads: int,
) -> jnp.ndarray:
    y = linear(x, params)
    y = y.reshape(y.shape[:-1] + (num_heads, -1))
    return jnp.swapaxes(y, -2, -3)
