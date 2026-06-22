from __future__ import annotations

import json

import numpy as np
import pytest

from protenix_jax.data.featurize_json import (
    featurize_protein_json,
    load_first_job,
    main,
    parse_a3m_profile,
    parse_a3m_rows,
)
from protenix_jax.data.static_io import load_static_feature_npz


def test_featurize_sequence_only_protein_json() -> None:
    job = {
        "name": "toy",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AGX",
                    "count": 2,
                    "id": ["A", "B"],
                },
            },
        ],
    }

    features = featurize_protein_json(job, n_queries=2, n_keys=4)

    assert features["restype"].shape == (6, 32)
    np.testing.assert_array_equal(features["restype"].argmax(axis=-1), [0, 7, 20] * 2)
    np.testing.assert_array_equal(features["profile"], features["restype"])
    np.testing.assert_array_equal(features["deletion_mean"], np.zeros(6))
    np.testing.assert_array_equal(features["residue_index"], [1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(features["asym_id"], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(features["entity_id"], [0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(features["sym_id"], [0, 0, 0, 1, 1, 1])
    assert features["token_bonds"].shape == (6, 6)
    assert features["atom_to_token_idx"].shape == (32,)
    assert features["ref_pos"].shape == (32, 3)
    assert features["ref_element"].shape == (32, 128)
    assert features["ref_atom_name_chars"].shape == (32, 4, 64)
    assert features["d_lm"].shape[-1] == 3
    assert features["v_lm"].shape[-1] == 1
    assert features["pad_info"]["mask_trunked"].shape == features["v_lm"].shape[:-1]
    assert features["has_frame"].tolist() == [1] * 6
    assert features["distogram_rep_atom_mask"].sum() == 6


def test_parse_a3m_profile_maps_insertions_and_ambiguous_codes() -> None:
    profile, deletion_mean = parse_a3m_profile(
        "ACD",
        ">query\nACD\n>hit1\nAc-D\n>hit2\nAZJ\n",
    )

    assert profile.shape == (3, 32)
    np.testing.assert_allclose(deletion_mean, [0.0, 1 / 3, 0.0])
    np.testing.assert_allclose(profile[0, 0], 1.0)
    np.testing.assert_allclose(profile[1, 4], 1 / 3)
    np.testing.assert_allclose(profile[1, 6], 1 / 3)
    np.testing.assert_allclose(profile[1, 31], 1 / 3)
    np.testing.assert_allclose(profile[2, 3], 2 / 3)
    np.testing.assert_allclose(profile[2, 20], 1 / 3)


def test_parse_a3m_rows_matches_protenix_deletion_encoding() -> None:
    msa, deletion_matrix = parse_a3m_rows(
        "AG",
        ">query\nAG\n>hit\nAc-\n",
    )

    np.testing.assert_array_equal(msa, [[0, 7], [0, 31]])
    np.testing.assert_array_equal(deletion_matrix, [[0, 0], [0, 1]])


def test_parse_a3m_profile_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="aligned length"):
        parse_a3m_profile("ACD", ">query\nACD\n>bad\nAC\n")


def test_featurize_json_uses_msa_profile(tmp_path) -> None:
    msa_path = tmp_path / "toy.a3m"
    msa_path.write_text(">query\nAG\n>hit\nA-\n")
    job = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AG",
                    "unpairedMsaPath": "toy.a3m",
                }
            }
        ]
    }

    features = featurize_protein_json(job, base_dir=tmp_path, n_queries=2, n_keys=4)

    np.testing.assert_allclose(features["profile"][0, 0], 1.0)
    np.testing.assert_allclose(features["profile"][1, 7], 0.5)
    np.testing.assert_allclose(features["profile"][1, 31], 0.5)
    np.testing.assert_allclose(features["deletion_mean"], [0.0, 0.0])


def test_featurize_json_emits_global_msa_rows(tmp_path) -> None:
    msa_path = tmp_path / "a.a3m"
    msa_path.write_text(">query\nAG\n>hit\nAc-\n")
    job = {
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "AG",
                    "unpairedMsaPath": "a.a3m",
                }
            },
            {"proteinChain": {"sequence": "CD"}},
        ]
    }

    features = featurize_protein_json(job, base_dir=tmp_path, n_queries=2, n_keys=4)

    np.testing.assert_array_equal(
        features["msa"],
        [
            [0, 7, 4, 3],
            [0, 31, 31, 31],
        ],
    )
    np.testing.assert_array_equal(
        features["has_deletion"],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
    )
    expected_deletion = np.zeros((2, 4), dtype=np.float32)
    expected_deletion[1, 1] = np.arctan(1 / 3) * (2 / np.pi)
    np.testing.assert_allclose(features["deletion_value"], expected_deletion)


def test_featurize_json_rejects_unsupported_entities() -> None:
    job = {
        "name": "bad",
        "sequences": [{"ligand": {"ligand": "CCD_ATP", "count": 1}}],
    }

    with pytest.raises(ValueError, match="Only proteinChain"):
        featurize_protein_json(job)


def test_featurize_json_rejects_unsupported_raw_features() -> None:
    with pytest.raises(ValueError, match="covalent_bonds"):
        featurize_protein_json(
            {
                "sequences": [{"proteinChain": {"sequence": "A"}}],
                "covalent_bonds": [{"entity1": "1"}],
            }
        )
    with pytest.raises(ValueError, match="template"):
        featurize_protein_json(
            {
                "sequences": [
                    {"proteinChain": {"sequence": "A", "templatesPath": "x.hhr"}}
                ]
            }
        )
    with pytest.raises(ValueError, match="modifications"):
        featurize_protein_json(
            {
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "A",
                            "modifications": [{"ptmType": "CCD_MSE", "ptmPosition": 1}],
                        },
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="unsupported residue"):
        featurize_protein_json({"sequences": [{"proteinChain": {"sequence": "AJ"}}]})


def test_load_first_job_and_cli_write_static_npz(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    out_path = tmp_path / "features.npz"
    input_path.write_text(
        json.dumps(
            [
                {
                    "name": "toy",
                    "sequences": [{"proteinChain": {"sequence": "AG", "count": 1}}],
                }
            ]
        )
    )

    assert load_first_job(input_path)["name"] == "toy"
    main(["--input", str(input_path), "--out", str(out_path), "--n-queries", "2"])

    features = load_static_feature_npz(out_path)
    assert features["restype"].shape == (2, 32)
    assert features["pad_info"]["mask_trunked"].shape == features["v_lm"].shape[:-1]
