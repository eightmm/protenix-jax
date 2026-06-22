from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from protenix_jax.bridge.weights_io import load_native_weights, save_native_weights
from protenix_jax.models.primitives.primitives import LinearParams


def test_native_weights_roundtrip_namedtuple_tree(tmp_path) -> None:
    path = tmp_path / "weights.pkl.gz"
    params = {
        "linear": LinearParams(
            weight=jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
            bias=None,
        ),
        "flag": True,
    }

    save_native_weights(path, params)
    restored = load_native_weights(path)

    assert isinstance(restored["linear"], LinearParams)
    assert restored["linear"].bias is None
    assert restored["flag"] is True
    np.testing.assert_array_equal(
        np.asarray(restored["linear"].weight),
        np.asarray([[1.0, 2.0]], dtype=np.float32),
    )


def test_native_weights_roundtrip_uncompressed(tmp_path) -> None:
    path = tmp_path / "weights.pkl"
    params = LinearParams(
        weight=jnp.asarray([[3.0]], dtype=jnp.float32),
        bias=jnp.asarray([4.0], dtype=jnp.float32),
    )

    save_native_weights(path, params, compress=False)
    restored = load_native_weights(path)

    np.testing.assert_array_equal(np.asarray(restored.weight), [[3.0]])
    np.testing.assert_array_equal(np.asarray(restored.bias), [4.0])
