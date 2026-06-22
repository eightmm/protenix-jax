from __future__ import annotations

import numpy as np
import pytest
from test_model import _toy_features, _toy_params

from protenix_jax.bridge.weights_io import save_native_weights
from protenix_jax.cli.static_infer import main
from protenix_jax.data.featurize_json import featurize_protein_json
from protenix_jax.data.static_io import save_static_feature_npz
from protenix_jax.models.primitives.primitives import LinearParams
from protenix_jax.models.trunk_blocks.embedders import RelativePositionParams


def test_static_infer_help_exits_without_runtime_imports(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--features" in out
    assert "--input-json" in out
    assert "--weights" in out
    assert "--checkpoint" not in out
    assert "--chunk-policy" in out
    assert "--model-name" in out
    assert "--max-msa-rows" in out


def test_static_infer_runs_with_native_weights(tmp_path) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    features_path = tmp_path / "toy_features.npz"
    out_path = tmp_path / "out.npz"
    save_native_weights(weights_path, _toy_params(), compress=False)
    save_static_feature_npz(features_path, _toy_features())

    main(
        [
            "--weights",
            str(weights_path),
            "--features",
            str(features_path),
            "--out",
            str(out_path),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--input-atom-heads",
            "1",
            "--atom-encoder-heads",
            "1",
            "--token-heads",
            "1",
            "--atom-decoder-heads",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--sigma-data",
            "4.0",
            "--cpu-only",
        ]
    )

    with np.load(out_path) as data:
        assert data["coordinate"].shape == (1, 3, 3)
        assert data["distogram_logits"].shape == (2, 2, 3)
        assert data["summary_plddt"].shape == (1,)


def test_static_infer_runs_from_sequence_json_features(tmp_path) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    features_path = tmp_path / "json_features.npz"
    out_path = tmp_path / "out.npz"
    save_native_weights(weights_path, _toy_params(), compress=False)
    features = featurize_protein_json(
        {"sequences": [{"proteinChain": {"sequence": "AG", "count": 1}}]},
        n_queries=2,
        n_keys=4,
    )
    features["relp"] = np.zeros((2, 2, 2), dtype=np.float32)
    save_static_feature_npz(features_path, features)

    main(
        [
            "--weights",
            str(weights_path),
            "--features",
            str(features_path),
            "--out",
            str(out_path),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--input-atom-heads",
            "1",
            "--atom-encoder-heads",
            "1",
            "--token-heads",
            "1",
            "--atom-decoder-heads",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--sigma-data",
            "4.0",
            "--cpu-only",
        ]
    )

    with np.load(out_path) as data:
        assert data["coordinate"].shape == (1, 10, 3)
        assert data["distogram_logits"].shape == (2, 2, 3)
        assert data["summary_plddt"].shape == (1,)


def test_static_infer_runs_directly_from_sequence_json(tmp_path) -> None:
    weights_path = tmp_path / "toy_weights.pkl"
    input_json = tmp_path / "input.json"
    out_path = tmp_path / "out.npz"
    input_json.write_text(
        '[{"sequences": [{"proteinChain": {"sequence": "AG", "count": 1}}]}]'
    )
    save_native_weights(weights_path, _toy_params_with_relp_dim(139), compress=False)

    main(
        [
            "--weights",
            str(weights_path),
            "--input-json",
            str(input_json),
            "--out",
            str(out_path),
            "--n-sample",
            "1",
            "--n-step",
            "1",
            "--n-cycle",
            "1",
            "--input-atom-heads",
            "1",
            "--atom-encoder-heads",
            "1",
            "--token-heads",
            "1",
            "--atom-decoder-heads",
            "1",
            "--n-queries",
            "2",
            "--n-keys",
            "4",
            "--sigma-data",
            "4.0",
            "--cpu-only",
        ]
    )

    with np.load(out_path) as data:
        assert data["coordinate"].shape == (1, 10, 3)
        assert data["summary_plddt"].shape == (1,)


def test_static_infer_rejects_large_v2_before_weight_load(tmp_path) -> None:
    features_path = tmp_path / "large_features.npz"
    np.savez_compressed(
        features_path,
        restype=np.zeros((2561, 32), dtype=np.float32),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--weights",
                str(tmp_path / "missing.pkl"),
                "--features",
                str(features_path),
                "--out",
                str(tmp_path / "out.npz"),
                "--model-name",
                "protenix-v2",
                "--cpu-only",
            ]
        )

    assert exc_info.value.code != 0
    assert "does not support n_token > 2560" in str(exc_info.value)


def _toy_params_with_relp_dim(relp_dim: int):
    params = _toy_params()
    pairformer_output = params.pairformer_output
    trunk = pairformer_output.trunk
    initial = trunk.initial
    initial = initial._replace(
        relative_position=RelativePositionParams(
            linear_no_bias=LinearParams(
                weight=np.zeros((2, relp_dim), dtype=np.float32),
                bias=None,
            )
        )
    )
    trunk = trunk._replace(initial=initial)
    pairformer_output = pairformer_output._replace(trunk=trunk)
    diffusion = params.diffusion
    conditioning = diffusion.conditioning
    conditioning = conditioning._replace(
        relpe=RelativePositionParams(
            linear_no_bias=LinearParams(
                weight=np.zeros((2, relp_dim), dtype=np.float32),
                bias=None,
            )
        )
    )
    diffusion = diffusion._replace(conditioning=conditioning)
    return params._replace(pairformer_output=pairformer_output, diffusion=diffusion)
