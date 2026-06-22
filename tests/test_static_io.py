from __future__ import annotations

import numpy as np

from protenix_jax.data.static_io import (
    flatten_output_dict,
    load_static_feature_npz,
    save_output_npz,
    save_static_feature_npz,
)


def test_load_static_feature_npz_groups_pad_info(tmp_path) -> None:
    path = tmp_path / "features.npz"
    np.savez(
        path,
        restype=np.zeros((2, 32), dtype=np.float32),
        **{"pad_info.mask_trunked": np.ones((2, 2, 4), dtype=bool)},
    )

    features = load_static_feature_npz(path)

    assert set(features) == {"restype", "pad_info"}
    np.testing.assert_array_equal(features["restype"], np.zeros((2, 32)))
    np.testing.assert_array_equal(
        features["pad_info"]["mask_trunked"],
        np.ones((2, 2, 4), dtype=bool),
    )


def test_save_static_feature_npz_roundtrips_pad_info(tmp_path) -> None:
    path = tmp_path / "features.npz"
    save_static_feature_npz(
        path,
        {
            "restype": np.zeros((2, 32), dtype=np.float32),
            "pad_info": {"mask_trunked": np.ones((2, 2, 4), dtype=bool)},
        },
    )

    features = load_static_feature_npz(path)

    assert set(features) == {"restype", "pad_info"}
    np.testing.assert_array_equal(features["restype"], np.zeros((2, 32)))
    np.testing.assert_array_equal(
        features["pad_info"]["mask_trunked"],
        np.ones((2, 2, 4), dtype=bool),
    )


def test_flatten_output_dict_skips_trunk_by_default() -> None:
    output = {
        "coordinate": np.ones((1, 3, 3)),
        "s_inputs": np.ones((2, 67)),
        "nested": {"value": np.asarray([1.0])},
    }

    flat = flatten_output_dict(output)

    assert set(flat) == {"coordinate", "nested.value"}


def test_save_output_npz_writes_compressed_arrays(tmp_path) -> None:
    path = tmp_path / "out.npz"
    save_output_npz(
        path,
        {
            "coordinate": np.ones((1, 2, 3)),
            "s_trunk": np.ones((2, 2)),
        },
    )

    with np.load(path) as data:
        assert set(data.files) == {"coordinate"}
        np.testing.assert_array_equal(data["coordinate"], np.ones((1, 2, 3)))
