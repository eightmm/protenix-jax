from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from protenix_jax.bridge.torch_mapping import (
    map_msa_block_state_dict,
    map_msa_module_state_dict,
    map_msa_pair_weighted_averaging_state_dict,
    map_outer_product_mean_state_dict,
)
from protenix_jax.models.primitives.primitives import transition
from protenix_jax.models.trunk_blocks.msa import (
    msa_block,
    msa_module,
    msa_pair_weighted_averaging,
    outer_product_mean,
)
from protenix_jax.models.trunk_blocks.pairformer import pairformer_block


def test_outer_product_mean_matches_reference_formula() -> None:
    rng = np.random.default_rng(21)
    state = _outer_product_mean_state(rng, "opm", c_m=3, c_hidden=2, c_z=4)
    params = map_outer_product_mean_state_dict(state, "opm")
    m = rng.normal(size=(2, 3, 3)).astype(np.float32)
    mask = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)

    actual = outer_product_mean(jnp.asarray(m), jnp.asarray(mask), params)

    ln = _layer_norm_np(m, state["opm.layer_norm.weight"], state["opm.layer_norm.bias"])
    a = (ln @ state["opm.linear_1.weight"].T) * mask[..., None]
    b = (ln @ state["opm.linear_2.weight"].T) * mask[..., None]
    outer = np.einsum("mic,mjd->ijcd", a, b).reshape(3, 3, 4)
    expected = outer @ state["opm.linear_out.weight"].T
    expected = expected + state["opm.linear_out.bias"]
    norm = np.einsum("mi,mj->ij", mask, mask)[..., None] + 1e-3
    expected = expected / norm

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def test_msa_pair_weighted_averaging_matches_reference_formula() -> None:
    rng = np.random.default_rng(22)
    state = _msa_pair_weighted_state(rng, "mpwa", c_m=4, c_z=3, heads=2, c=2)
    params = map_msa_pair_weighted_averaging_state_dict(state, "mpwa")
    m = rng.normal(size=(2, 3, 4)).astype(np.float32)
    z = rng.normal(size=(3, 3, 3)).astype(np.float32)

    actual = msa_pair_weighted_averaging(jnp.asarray(m), jnp.asarray(z), params)

    m_norm = _layer_norm_np(
        m,
        state["mpwa.layernorm_m.weight"],
        state["mpwa.layernorm_m.bias"],
    )
    v = (m_norm @ state["mpwa.linear_no_bias_mv.weight"].T).reshape(2, 3, 2, 2)
    z_norm = _layer_norm_np(
        z,
        state["mpwa.layernorm_z.weight"],
        state["mpwa.layernorm_z.bias"],
    )
    b = z_norm @ state["mpwa.linear_no_bias_z.weight"].T
    w = _softmax_np(b, axis=-2)
    g = _sigmoid_np(m_norm @ state["mpwa.linear_no_bias_mg.weight"].T).reshape(
        2,
        3,
        2,
        2,
    )
    o = g * np.einsum("ijh,mjhc->mihc", w, v)
    expected = o.reshape(2, 3, 4) @ state["mpwa.linear_no_bias_out.weight"].T

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def test_msa_block_matches_reference_composition() -> None:
    rng = np.random.default_rng(23)
    state = _msa_block_state(
        rng,
        "msa.blocks.0",
        c_m=4,
        c_z=4,
        msa_heads=2,
        msa_c=2,
        pair_heads=2,
        is_last=False,
    )
    params = map_msa_block_state_dict(state, "msa.blocks.0")
    m = rng.normal(size=(2, 3, 4)).astype(np.float32)
    z = rng.normal(size=(3, 3, 4)).astype(np.float32)
    pair_mask = np.ones((3, 3), dtype=np.float32)

    actual_m, actual_z = msa_block(
        jnp.asarray(m),
        jnp.asarray(z),
        jnp.asarray(pair_mask),
        params,
    )

    expected_z = jnp.asarray(z) + outer_product_mean(
        jnp.asarray(m),
        None,
        params.outer_product_mean,
    )
    expected_m = jnp.asarray(m) + msa_pair_weighted_averaging(
        jnp.asarray(m),
        expected_z,
        params.msa_pair_weighted_averaging,
    )
    expected_m = expected_m + transition(expected_m, params.msa_transition)
    _, expected_z = pairformer_block(
        None,
        expected_z,
        jnp.asarray(pair_mask),
        params.pair_stack,
    )

    np.testing.assert_allclose(np.asarray(actual_m), np.asarray(expected_m), atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual_z), np.asarray(expected_z), atol=1e-5)


def test_msa_module_builds_msa_embedding_and_discovers_blocks() -> None:
    rng = np.random.default_rng(24)
    state = {
        "msa_module.linear_no_bias_m.weight": rng.normal(size=(4, 34)).astype(
            np.float32
        ),
        "msa_module.linear_no_bias_s.weight": rng.normal(size=(4, 5)).astype(
            np.float32
        ),
    }
    state.update(
        _msa_block_state(
            rng,
            "msa_module.blocks.0",
            c_m=4,
            c_z=4,
            msa_heads=2,
            msa_c=2,
            pair_heads=2,
            is_last=False,
        )
    )
    state.update(
        _msa_block_state(
            rng,
            "msa_module.blocks.1",
            c_m=4,
            c_z=4,
            msa_heads=2,
            msa_c=2,
            pair_heads=2,
            is_last=True,
        )
    )
    params = map_msa_module_state_dict(state, "msa_module")
    features = {
        "msa": jnp.asarray([[0, 1, 2], [3, 4, 5]]),
        "has_deletion": jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        "deletion_value": jnp.asarray([[0.0, 0.5, 0.0], [0.25, 0.0, 1.0]]),
    }
    z = rng.normal(size=(3, 3, 4)).astype(np.float32)
    s_inputs = rng.normal(size=(3, 5)).astype(np.float32)

    actual = msa_module(features, jnp.asarray(z), jnp.asarray(s_inputs), None, params)

    assert len(params.blocks) == 2
    assert params.blocks[-1].msa_pair_weighted_averaging is None
    assert actual.shape == (3, 3, 4)


def _outer_product_mean_state(
    rng: np.random.Generator,
    prefix: str,
    *,
    c_m: int,
    c_hidden: int,
    c_z: int,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.layer_norm.weight": rng.normal(size=(c_m,)).astype(np.float32),
        f"{prefix}.layer_norm.bias": rng.normal(size=(c_m,)).astype(np.float32),
        f"{prefix}.linear_1.weight": rng.normal(size=(c_hidden, c_m)).astype(
            np.float32
        ),
        f"{prefix}.linear_2.weight": rng.normal(size=(c_hidden, c_m)).astype(
            np.float32
        ),
        f"{prefix}.linear_out.weight": rng.normal(
            size=(c_z, c_hidden * c_hidden)
        ).astype(np.float32),
        f"{prefix}.linear_out.bias": rng.normal(size=(c_z,)).astype(np.float32),
    }


def _msa_pair_weighted_state(
    rng: np.random.Generator,
    prefix: str,
    *,
    c_m: int,
    c_z: int,
    heads: int,
    c: int,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.layernorm_m.weight": rng.normal(size=(c_m,)).astype(np.float32),
        f"{prefix}.layernorm_m.bias": rng.normal(size=(c_m,)).astype(np.float32),
        f"{prefix}.linear_no_bias_mv.weight": rng.normal(
            size=(heads * c, c_m)
        ).astype(np.float32),
        f"{prefix}.layernorm_z.weight": rng.normal(size=(c_z,)).astype(np.float32),
        f"{prefix}.layernorm_z.bias": rng.normal(size=(c_z,)).astype(np.float32),
        f"{prefix}.linear_no_bias_z.weight": rng.normal(size=(heads, c_z)).astype(
            np.float32
        ),
        f"{prefix}.linear_no_bias_mg.weight": rng.normal(
            size=(heads * c, c_m)
        ).astype(np.float32),
        f"{prefix}.linear_no_bias_out.weight": rng.normal(
            size=(c_m, heads * c)
        ).astype(np.float32),
    }


def _msa_block_state(
    rng: np.random.Generator,
    prefix: str,
    *,
    c_m: int,
    c_z: int,
    msa_heads: int,
    msa_c: int,
    pair_heads: int,
    is_last: bool,
) -> dict[str, np.ndarray]:
    state = _outer_product_mean_state(
        rng,
        f"{prefix}.outer_product_mean_msa",
        c_m=c_m,
        c_hidden=2,
        c_z=c_z,
    )
    if not is_last:
        state.update(
            _msa_pair_weighted_state(
                rng,
                f"{prefix}.msa_stack.msa_pair_weighted_averaging",
                c_m=c_m,
                c_z=c_z,
                heads=msa_heads,
                c=msa_c,
            )
        )
        _add_transition_state(state, rng, f"{prefix}.msa_stack.transition_m", c_m)
    _add_pair_stack_state(state, rng, f"{prefix}.pair_stack", c_z, pair_heads)
    return state


def _add_pair_stack_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_z: int,
    heads: int,
) -> None:
    _add_triangle_multiplication_state(state, rng, f"{prefix}.tri_mul_out", c_z)
    _add_triangle_multiplication_state(state, rng, f"{prefix}.tri_mul_in", c_z)
    _add_triangle_attention_state(state, rng, f"{prefix}.tri_att_start", c_z, heads)
    _add_triangle_attention_state(state, rng, f"{prefix}.tri_att_end", c_z, heads)
    _add_transition_state(state, rng, f"{prefix}.pair_transition", c_z)


def _add_triangle_multiplication_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_z: int,
) -> None:
    state[f"{prefix}.layer_norm_in.weight"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layer_norm_in.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layer_norm_out.weight"] = rng.normal(size=(c_z,)).astype(
        np.float32
    )
    state[f"{prefix}.layer_norm_out.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    for name in ("linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g"):
        state[f"{prefix}.{name}.weight"] = rng.normal(size=(c_z, c_z)).astype(
            np.float32
        )
    state[f"{prefix}.linear_z.weight"] = rng.normal(size=(c_z, c_z)).astype(np.float32)
    state[f"{prefix}.linear_g.weight"] = rng.normal(size=(c_z, c_z)).astype(np.float32)


def _add_triangle_attention_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_z: int,
    heads: int,
) -> None:
    state[f"{prefix}.layer_norm.weight"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.layer_norm.bias"] = rng.normal(size=(c_z,)).astype(np.float32)
    state[f"{prefix}.linear.weight"] = rng.normal(size=(heads, c_z)).astype(np.float32)
    for name in ("linear_q", "linear_k", "linear_v", "linear_o", "linear_g"):
        state[f"{prefix}.mha.{name}.weight"] = rng.normal(size=(c_z, c_z)).astype(
            np.float32
        )


def _add_transition_state(
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
    prefix: str,
    c_in: int,
) -> None:
    hidden = c_in * 4
    state[f"{prefix}.layernorm1.weight"] = rng.normal(size=(c_in,)).astype(np.float32)
    state[f"{prefix}.layernorm1.bias"] = rng.normal(size=(c_in,)).astype(np.float32)
    state[f"{prefix}.linear_no_bias_a.weight"] = rng.normal(
        size=(hidden, c_in)
    ).astype(np.float32)
    state[f"{prefix}.linear_no_bias_b.weight"] = rng.normal(
        size=(hidden, c_in)
    ).astype(np.float32)
    state[f"{prefix}.linear_no_bias.weight"] = rng.normal(
        size=(c_in, hidden)
    ).astype(np.float32)


def _layer_norm_np(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_np(x: np.ndarray, axis: int) -> np.ndarray:
    y = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(y)
    return y / np.sum(y, axis=axis, keepdims=True)
