"""JAX attention blocks matching Protenix inference modules."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from protenix_jax.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    adaptive_layer_norm,
    layer_norm,
    linear,
    sigmoid,
)


class AttentionParams(NamedTuple):
    """Parameters for ``protenix.model.modules.primitives.Attention``."""

    linear_q: LinearParams
    linear_k: LinearParams
    linear_v: LinearParams
    linear_o: LinearParams
    linear_g: LinearParams | None


class AttentionPairBiasParams(NamedTuple):
    """Parameters for standard ``AttentionPairBias``."""

    layernorm_a: LayerNormParams | AdaptiveLayerNormParams
    layernorm_kv: LayerNormParams | AdaptiveLayerNormParams | None
    attention: AttentionParams
    layernorm_z: LayerNormParams
    linear_z: LinearParams
    linear_a_last: LinearParams | None = None
    has_s: bool = False
    cross_attention_mode: bool = False


def prepare_qkv(
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    params: AttentionParams,
    num_heads: int,
    *,
    apply_scale: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare q/k/v in Protenix layout: ``[..., H, Q/K/V, C_hidden]``."""

    q = _project_heads(q_x, params.linear_q, num_heads)
    k = _project_heads(kv_x, params.linear_k, num_heads)
    v = _project_heads(kv_x, params.linear_v, num_heads)
    if apply_scale:
        q = q / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype))
    return q, k, v


def attention(
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    params: AttentionParams,
    num_heads: int,
    attn_bias: jnp.ndarray | None = None,
    *,
    q_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run standard full attention with optional pair bias and gating."""

    q, k, v = prepare_qkv(q_x, kv_x, params, num_heads, apply_scale=True)
    out = _attention_qkv(q, k, v, attn_bias, q_chunk_size=q_chunk_size)
    out = jnp.swapaxes(out, -2, -3)
    if params.linear_g is not None:
        gate = sigmoid(linear(q_x, params.linear_g))
        gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.linear_o)


def attention_pair_bias(
    a: jnp.ndarray,
    s: jnp.ndarray | None,
    z: jnp.ndarray,
    params: AttentionPairBiasParams,
    *,
    num_heads: int,
    q_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run standard global ``AttentionPairBias``.

    This implements the non-local path used by token Pairformer/Confidence
    blocks. Atom local attention is intentionally separate.
    """

    q = _apply_apb_norm(a, s, params.layernorm_a, params.has_s)
    if params.cross_attention_mode:
        kv = _apply_apb_norm(a, s, params.layernorm_kv, params.has_s)
    else:
        kv = q
    bias = linear(layer_norm(z, params.layernorm_z), params.linear_z)
    bias = jnp.moveaxis(bias, -1, -3)
    out = attention(
        q,
        kv,
        params.attention,
        num_heads,
        bias,
        q_chunk_size=q_chunk_size,
    )
    if params.linear_a_last is not None:
        if s is None:
            raise ValueError("linear_a_last requires conditioning tensor s")
        out = sigmoid(linear(s, params.linear_a_last)) * out
    return out


def local_attention(
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    params: AttentionParams,
    num_heads: int,
    *,
    trunked_attn_bias: jnp.ndarray | None,
    n_queries: int,
    n_keys: int,
    inf: float = 1.0e10,
) -> jnp.ndarray:
    """Run local blocked attention used by AtomTransformer."""

    q, k, v = prepare_qkv(q_x, kv_x, params, num_heads, apply_scale=True)
    q_trunked, k_trunked, mask, q_pad = _local_qk_trunks(
        q,
        k,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    _, v_trunked, _, _ = _local_qk_trunks(
        q,
        v,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    logits = jnp.einsum("...htqd,...htkd->...htqk", q_trunked, k_trunked)
    mask = mask.reshape((1,) * (logits.ndim - 3) + mask.shape)
    logits = logits + jnp.where(mask, 0.0, -inf)
    if trunked_attn_bias is not None:
        if trunked_attn_bias.ndim == logits.ndim - 1:
            trunked_attn_bias = jnp.expand_dims(trunked_attn_bias, axis=-4)
        logits = logits + trunked_attn_bias
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
    out = jnp.einsum("...htqk,...htkd->...htqd", probs, v_trunked)
    out = out.reshape(out.shape[:-4] + (num_heads, -1, out.shape[-1]))
    if q_pad > 0:
        out = out[..., :-q_pad, :]
    out = jnp.swapaxes(out, -2, -3)
    if params.linear_g is not None:
        gate = sigmoid(linear(q_x, params.linear_g))
        gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.linear_o)


def local_attention_pair_bias(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: AttentionPairBiasParams,
    *,
    num_heads: int,
    n_queries: int,
    n_keys: int,
) -> jnp.ndarray:
    """Run adaptive local ``AttentionPairBias`` for AtomTransformer."""

    q = _apply_apb_norm(a, s, params.layernorm_a, params.has_s)
    kv = (
        _apply_apb_norm(a, s, params.layernorm_kv, params.has_s)
        if params.cross_attention_mode
        else q
    )
    bias = linear(layer_norm(z, params.layernorm_z), params.linear_z)
    bias = jnp.moveaxis(bias, -1, -4)
    out = local_attention(
        q,
        kv,
        params.attention,
        num_heads,
        trunked_attn_bias=bias,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    if params.linear_a_last is not None:
        out = sigmoid(linear(s, params.linear_a_last)) * out
    return out


def _project_heads(
    x: jnp.ndarray,
    params: LinearParams,
    num_heads: int,
) -> jnp.ndarray:
    y = linear(x, params)
    y = y.reshape(y.shape[:-1] + (num_heads, -1))
    return jnp.swapaxes(y, -2, -3)


def _attention_qkv(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    attn_bias: jnp.ndarray | None,
    *,
    q_chunk_size: int | None,
) -> jnp.ndarray:
    n_q = q.shape[-2]
    if q_chunk_size is None or q_chunk_size <= 0 or q_chunk_size >= n_q:
        return _attention_qkv_chunk(q, k, v, attn_bias)
    chunks = []
    for start in range(0, n_q, q_chunk_size):
        end = min(start + q_chunk_size, n_q)
        q_chunk = q[..., start:end, :]
        bias_chunk = None
        if attn_bias is not None:
            bias_chunk = _normalize_attention_bias(attn_bias, q.ndim)[
                ..., start:end, :
            ]
        chunks.append(_attention_qkv_chunk(q_chunk, k, v, bias_chunk))
    return jnp.concatenate(chunks, axis=-2)


def _attention_qkv_chunk(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    attn_bias: jnp.ndarray | None,
) -> jnp.ndarray:
    logits = jnp.einsum("...hid,...hjd->...hij", q, k)
    if attn_bias is not None:
        logits = logits + _normalize_attention_bias(attn_bias, logits.ndim)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
    return jnp.einsum("...hij,...hjd->...hid", probs, v)


def _normalize_attention_bias(
    attn_bias: jnp.ndarray,
    logits_ndim: int,
) -> jnp.ndarray:
    if attn_bias.ndim == logits_ndim:
        return attn_bias
    if attn_bias.ndim == logits_ndim - 1:
        return jnp.expand_dims(attn_bias, axis=-3)
    raise ValueError("attention bias rank must match logits rank or omit head axis")


def _apply_apb_norm(
    a: jnp.ndarray,
    s: jnp.ndarray | None,
    params: LayerNormParams | AdaptiveLayerNormParams | None,
    has_s: bool,
) -> jnp.ndarray:
    if params is None:
        raise ValueError("missing attention pair-bias normalization params")
    if has_s:
        if s is None:
            raise ValueError("adaptive attention pair-bias normalization requires s")
        if not isinstance(params, AdaptiveLayerNormParams):
            raise TypeError("expected AdaptiveLayerNormParams when has_s=True")
        return adaptive_layer_norm(a, s, params)
    if not isinstance(params, LayerNormParams):
        raise TypeError("expected LayerNormParams when has_s=False")
    return layer_norm(a, params)


def _local_qk_trunks(
    q: jnp.ndarray,
    k: jnp.ndarray,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    if q.shape != k.shape:
        raise ValueError("local attention requires q and kv to share shape")
    if n_keys < n_queries or n_queries % 2 or n_keys % 2:
        raise ValueError("invalid local attention window sizes")
    n = q.shape[-2]
    n_trunks = (n + n_queries - 1) // n_queries
    q_pad = n_trunks * n_queries - n
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 0.5) * n_queries + n_keys / 2 - n + 0.5)
    pad_q = ((0, 0),) * (q.ndim - 2) + ((0, q_pad), (0, 0))
    pad_k = ((0, 0),) * (k.ndim - 2) + ((pad_left, pad_right), (0, 0))
    q_padded = jnp.pad(q, pad_q)
    k_padded = jnp.pad(k, pad_k)
    q_trunked = q_padded.reshape(
        q.shape[:-2] + (n_trunks, n_queries, q.shape[-1])
    )
    k_trunked = jnp.stack(
        [
            k_padded[..., i * n_queries : i * n_queries + n_keys, :]
            for i in range(n_trunks)
        ],
        axis=-3,
    )
    q_abs = jnp.arange(n_trunks * n_queries).reshape(n_trunks, n_queries)
    k_abs = (
        jnp.arange(n_keys)[None, :]
        + jnp.arange(n_trunks)[:, None] * n_queries
        - pad_left
    )
    mask = (q_abs[..., None] < n) & (k_abs[:, None, :] >= 0) & (
        k_abs[:, None, :] < n
    )
    return q_trunked, k_trunked, mask, q_pad
