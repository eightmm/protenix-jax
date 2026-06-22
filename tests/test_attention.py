from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from protenix_jax.bridge.torch_mapping import (
    map_attention_pair_bias_state_dict,
    map_attention_state_dict,
)
from protenix_jax.models.primitives.attention import (
    attention,
    attention_pair_bias,
    prepare_qkv,
)
from protenix_jax.models.primitives.primitives import layer_norm


def test_prepare_qkv_uses_torch_layout_and_scaling() -> None:
    x = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32)
    state = {
        "attn.linear_q.weight": np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float32
        ),
        "attn.linear_q.bias": np.array([0.5, 0.0, 0.0, -0.5], dtype=np.float32),
        "attn.linear_k.weight": np.ones((4, 2), dtype=np.float32),
        "attn.linear_v.weight": np.full((4, 2), 0.25, dtype=np.float32),
        "attn.linear_o.weight": np.ones((2, 4), dtype=np.float32),
        "attn.linear_g.weight": np.zeros((4, 2), dtype=np.float32),
    }
    params = map_attention_state_dict(state, "attn")

    q, k, v = prepare_qkv(x, x, params, num_heads=2, apply_scale=True)

    expected_q_flat = np.asarray(x) @ state["attn.linear_q.weight"].T
    expected_q_flat = expected_q_flat + state["attn.linear_q.bias"]
    expected_q = expected_q_flat.reshape(1, 2, 2, 2).transpose(0, 2, 1, 3)
    expected_q = expected_q / np.sqrt(2.0)
    np.testing.assert_allclose(np.asarray(q), expected_q, rtol=1e-6, atol=1e-6)
    assert k.shape == (1, 2, 2, 2)
    assert v.shape == (1, 2, 2, 2)


def test_attention_matches_reference_formula_with_bias_and_gating() -> None:
    rng = np.random.default_rng(4)
    q_x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    kv_x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    bias = rng.normal(size=(1, 2, 3, 3)).astype(np.float32)
    state = {
        "attn.linear_q.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_q.bias": rng.normal(size=(4,)).astype(np.float32),
        "attn.linear_k.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_v.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_o.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_g.weight": rng.normal(size=(4, 4)).astype(np.float32),
    }
    params = map_attention_state_dict(state, "attn")

    actual = np.asarray(
        attention(jnp.asarray(q_x), jnp.asarray(kv_x), params, 2, jnp.asarray(bias))
    )

    q, k, v = prepare_qkv(jnp.asarray(q_x), jnp.asarray(kv_x), params, 2)
    logits = np.einsum("bhid,bhjd->bhij", np.asarray(q), np.asarray(k))
    logits = logits + bias
    probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
    out = np.einsum("bhij,bhjd->bhid", probs, np.asarray(v))
    out = out.transpose(0, 2, 1, 3)
    gate = 1.0 / (1.0 + np.exp(-(q_x @ state["attn.linear_g.weight"].T)))
    out = (out * gate.reshape(1, 3, 2, 2)).reshape(1, 3, 4)
    expected = out @ state["attn.linear_o.weight"].T

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_attention_query_chunk_matches_unchunked() -> None:
    rng = np.random.default_rng(44)
    q_x = rng.normal(size=(2, 5, 4)).astype(np.float32)
    kv_x = rng.normal(size=(2, 5, 4)).astype(np.float32)
    bias = rng.normal(size=(2, 2, 5, 5)).astype(np.float32)
    state = {
        "attn.linear_q.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_q.bias": rng.normal(size=(4,)).astype(np.float32),
        "attn.linear_k.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_v.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_o.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_g.weight": rng.normal(size=(4, 4)).astype(np.float32),
    }
    params = map_attention_state_dict(state, "attn")

    unchunked = attention(
        jnp.asarray(q_x),
        jnp.asarray(kv_x),
        params,
        2,
        jnp.asarray(bias),
    )
    chunked = attention(
        jnp.asarray(q_x),
        jnp.asarray(kv_x),
        params,
        2,
        jnp.asarray(bias),
        q_chunk_size=2,
    )

    np.testing.assert_allclose(
        np.asarray(chunked),
        np.asarray(unchunked),
        rtol=1e-5,
        atol=1e-5,
    )


def test_attention_pair_bias_has_s_false_matches_reference_formula() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(size=(1, 3, 4)).astype(np.float32)
    z = rng.normal(size=(1, 3, 3, 5)).astype(np.float32)
    state = {
        "apb.layernorm_a.weight": rng.normal(size=(4,)).astype(np.float32),
        "apb.layernorm_a.bias": rng.normal(size=(4,)).astype(np.float32),
        "apb.layernorm_z.weight": rng.normal(size=(5,)).astype(np.float32),
        "apb.layernorm_z.bias": rng.normal(size=(5,)).astype(np.float32),
        "apb.linear_nobias_z.weight": rng.normal(size=(2, 5)).astype(np.float32),
        "apb.attention.linear_q.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_q.bias": rng.normal(size=(4,)).astype(np.float32),
        "apb.attention.linear_k.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_v.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_o.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_g.weight": rng.normal(size=(4, 4)).astype(np.float32),
    }
    params = map_attention_pair_bias_state_dict(state, "apb", has_s=False)

    actual = np.asarray(
        attention_pair_bias(jnp.asarray(a), None, jnp.asarray(z), params, num_heads=2)
    )

    a_norm = layer_norm(jnp.asarray(a), params.layernorm_a)
    z_norm = layer_norm(jnp.asarray(z), params.layernorm_z)
    bias = np.asarray(z_norm) @ state["apb.linear_nobias_z.weight"].T
    bias = bias.transpose(0, 3, 1, 2)
    expected = np.asarray(
        attention(a_norm, a_norm, params.attention, 2, jnp.asarray(bias))
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_attention_pair_bias_query_chunk_matches_unchunked() -> None:
    rng = np.random.default_rng(55)
    a = rng.normal(size=(1, 5, 4)).astype(np.float32)
    z = rng.normal(size=(1, 5, 5, 5)).astype(np.float32)
    state = {
        "apb.layernorm_a.weight": rng.normal(size=(4,)).astype(np.float32),
        "apb.layernorm_a.bias": rng.normal(size=(4,)).astype(np.float32),
        "apb.layernorm_z.weight": rng.normal(size=(5,)).astype(np.float32),
        "apb.layernorm_z.bias": rng.normal(size=(5,)).astype(np.float32),
        "apb.linear_nobias_z.weight": rng.normal(size=(2, 5)).astype(np.float32),
        "apb.attention.linear_q.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_q.bias": rng.normal(size=(4,)).astype(np.float32),
        "apb.attention.linear_k.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_v.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_o.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_g.weight": rng.normal(size=(4, 4)).astype(np.float32),
    }
    params = map_attention_pair_bias_state_dict(state, "apb", has_s=False)

    unchunked = attention_pair_bias(
        jnp.asarray(a),
        None,
        jnp.asarray(z),
        params,
        num_heads=2,
    )
    chunked = attention_pair_bias(
        jnp.asarray(a),
        None,
        jnp.asarray(z),
        params,
        num_heads=2,
        q_chunk_size=2,
    )

    np.testing.assert_allclose(
        np.asarray(chunked),
        np.asarray(unchunked),
        rtol=1e-5,
        atol=1e-5,
    )
