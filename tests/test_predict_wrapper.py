from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from test_model import _toy_features, _toy_params

from protenix_jax.models.diffusion.diffusion import inference_noise_schedule
from protenix_jax.models.model import protenix_infer_static
from protenix_jax.models.predict import protenix_predict_static


def test_predict_wrapper_matches_static_infer_direct_call() -> None:
    params = _toy_params()
    features = _toy_features()
    init_noise = jnp.ones((1, 3, 3), dtype=jnp.float32)
    step_noise = jnp.zeros_like(init_noise)

    actual = protenix_predict_static(
        params,
        features,
        key=None,
        n_sample=1,
        num_sampling_steps=1,
        recycling_steps=1,
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        centre_each_step=False,
        init_noise=init_noise,
        step_noises=(step_noise,),
    )
    expected = protenix_infer_static(
        features,
        params,
        inference_noise_schedule(n_step=1, sigma_data=4.0),
        key=None,
        n_sample=1,
        init_noise=init_noise,
        step_noises=(step_noise,),
        n_cycle=1,
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        centre_each_step=False,
    )

    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(np.asarray(actual[key]), np.asarray(value))

