from __future__ import annotations

import pytest

from protenix_jax.runtime_policy import (
    infer_model_name_from_path,
    validate_inference_limits,
)


def test_infer_model_name_from_path_detects_known_checkpoints() -> None:
    assert infer_model_name_from_path("protenix-v2.pt") == "protenix-v2"
    assert (
        infer_model_name_from_path("protenix_base_default_v1.0.0.pt")
        == "protenix_base_default_v1.0.0"
    )
    assert infer_model_name_from_path("weights.pkl") is None


def test_validate_inference_limits_rejects_large_v2_inputs() -> None:
    with pytest.raises(ValueError, match="protenix-v2"):
        validate_inference_limits(model_name="protenix-v2", n_token=2561)


def test_validate_inference_limits_allows_v2_boundary_and_unknown_models() -> None:
    validate_inference_limits(model_name="protenix-v2", n_token=2560)
    validate_inference_limits(model_name=None, n_token=10000)
    validate_inference_limits(
        model_name="protenix_base_default_v1.0.0",
        n_token=10000,
    )
