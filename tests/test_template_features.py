"""Template featurization parity tests against torch Protenix goldens.

Goldens were produced by ``InferenceTemplateFeaturizer`` (JSON template path,
``parse_json_templates`` -> ``Templates.as_protenix_dict``) on the embedded
template and query, and stored in ``tests/fixtures``.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

from protenix_jax.data.featurize_json import featurize_protein_json

_FIXTURES = Path(__file__).parent / "fixtures"
_TEMPLATE_JSON = _FIXTURES / "template_small.json"
_GOLDEN_B64 = _FIXTURES / "template_small_golden.b64"
_QUERY = "AGSRF"
_TEMPLATE_KEYS = (
    "template_aatype",
    "template_atom_positions",
    "template_atom_mask",
    "template_pseudo_beta_mask",
    "template_distogram",
    "template_unit_vector",
    "template_backbone_frame_mask",
)


def _load_golden() -> dict[str, np.ndarray]:
    raw = base64.b64decode(_GOLDEN_B64.read_text())
    with np.load(io.BytesIO(raw)) as data:
        return {k: data[k] for k in data.files}


def test_template_features_match_torch_golden() -> None:
    job = {
        "name": "tpl",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": _QUERY,
                    "count": 1,
                    "templatesPath": str(_TEMPLATE_JSON),
                },
            },
        ],
    }
    features = featurize_protein_json(job)
    golden = _load_golden()

    for key in _TEMPLATE_KEYS:
        assert key in features, key
        got = np.asarray(features[key]).astype(np.float64)
        ref = golden[key].astype(np.float64)
        assert got.shape == ref.shape, (key, got.shape, ref.shape)
        np.testing.assert_allclose(got, ref, atol=1e-5, err_msg=key)

    assert features["template_aatype"].shape == (4, 5)
    assert features["template_distogram"].shape == (4, 5, 5, 39)
    assert features["template_unit_vector"].shape == (4, 5, 5, 3)


def test_template_features_absent_without_templates() -> None:
    job = {
        "name": "no_tpl",
        "sequences": [
            {"proteinChain": {"sequence": "AGSRF", "count": 1}},
        ],
    }
    features = featurize_protein_json(job)
    for key in _TEMPLATE_KEYS:
        assert key not in features


def test_template_short_chain_skipped() -> None:
    # Length <= 4 protein chains are skipped (torch behaviour); since the only
    # chain is skipped, no real templates remain -> features omitted.
    job = {
        "name": "short",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AGSR",
                    "count": 1,
                    "templatesPath": str(_TEMPLATE_JSON),
                },
            },
        ],
    }
    features = featurize_protein_json(job)
    assert "template_aatype" not in features


def test_template_rejects_non_json_path(tmp_path) -> None:
    bad = tmp_path / "tpl.a3m"
    bad.write_text(">x\nAAAAA\n")
    job = {
        "name": "bad",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AGSRF",
                    "count": 1,
                    "templatesPath": str(bad),
                },
            },
        ],
    }
    try:
        featurize_protein_json(job)
    except ValueError as exc:
        assert "JSON templates" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-JSON template path")


def test_template_json_fixture_is_self_contained() -> None:
    data = json.loads(_TEMPLATE_JSON.read_text())
    assert data[0]["mmcif"]
    assert len(data[0]["queryIndices"]) == len(data[0]["templateIndices"])
