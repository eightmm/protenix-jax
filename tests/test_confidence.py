from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from protenix_jax.models.heads.confidence import (
    ConfidenceDistanceEmbeddingParams,
    ConfidenceHeadParams,
    ConfidenceOutputParams,
    calculate_chain_based_gpde,
    calculate_chain_based_plddt,
    calculate_chain_pair_pae,
    calculate_clash,
    calculate_iptm,
    calculate_normalization,
    calculate_ptm,
    calculate_vdw_clash,
    compute_contact_prob,
    confidence_distance_embedding,
    confidence_head,
    confidence_one_hot,
    confidence_output_logits,
    confidence_scores_from_logits,
    get_bin_centers,
    logits_to_score,
)
from protenix_jax.models.primitives.primitives import LayerNormParams, LinearParams
from protenix_jax.models.trunk_blocks.pairformer import PairformerStackParams


def test_confidence_one_hot_uses_open_bin_edges() -> None:
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    lower = jnp.asarray([1.0, 2.0])
    upper = jnp.asarray([2.0, 4.0])

    result = confidence_one_hot(x, lower, upper)

    np.testing.assert_array_equal(
        np.asarray(result),
        np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
                [False, False],
            ],
        ),
    )


def test_confidence_distance_embedding_combines_binned_and_scalar_distance() -> None:
    params = ConfidenceDistanceEmbeddingParams(
        lower_bins=jnp.asarray([0.0, 1.5]),
        upper_bins=jnp.asarray([1.5, 10.0]),
        linear_d=LinearParams(
            weight=jnp.asarray([[10.0, 1.0], [100.0, 2.0]]),
            bias=None,
        ),
        linear_d_wo_onehot=LinearParams(
            weight=jnp.asarray([[0.5], [2.0]]),
            bias=None,
        ),
    )
    coords = jnp.asarray([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])

    result = confidence_distance_embedding(coords, params)

    expected = np.asarray(
        [
            [[0.0, 0.0], [1.0 + 2.5, 2.0 + 10.0]],
            [[1.0 + 2.5, 2.0 + 10.0], [0.0, 0.0]],
        ],
    )
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6, atol=1e-6)


def test_confidence_output_logits_project_pair_and_atom_outputs() -> None:
    params = ConfidenceOutputParams(
        pae_ln=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        pde_ln=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        plddt_ln=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        resolved_ln=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        linear_pae=LinearParams(weight=jnp.asarray([[1.0, -1.0]]), bias=None),
        linear_pde=LinearParams(weight=jnp.asarray([[2.0, 1.0]]), bias=None),
        plddt_weight=jnp.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 3.0]],
            ],
        ),
        resolved_weight=jnp.asarray(
            [
                [[0.5], [1.5]],
                [[1.0], [2.0]],
            ],
        ),
    )
    s_single = jnp.asarray([[1.0, 3.0], [2.0, 6.0]])
    z_pair = jnp.asarray(
        [
            [[1.0, 3.0], [2.0, 4.0]],
            [[5.0, 7.0], [8.0, 10.0]],
        ],
    )
    atom_to_token_idx = jnp.asarray([0, 1])
    atom_to_tokatom_idx = jnp.asarray([0, 1])

    output = confidence_output_logits(
        s_single,
        z_pair,
        atom_to_token_idx,
        atom_to_tokatom_idx,
        params,
    )

    assert set(output) == {"plddt", "pae", "pde", "resolved"}
    assert output["pae"].shape == (2, 2, 1)
    assert output["pde"].shape == (2, 2, 1)
    assert output["plddt"].shape == (2, 2)
    assert output["resolved"].shape == (2, 1)
    np.testing.assert_allclose(np.asarray(output["plddt"][0]), [-1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(output["resolved"][1]), [1.0], atol=1e-5)


def test_get_bin_centers_matches_protenix_formula() -> None:
    centers = get_bin_centers(min_bin=2.0, max_bin=10.0, no_bins=4)

    np.testing.assert_allclose(
        np.asarray(centers),
        np.asarray([3.0, 5.0, 7.0, 9.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_logits_to_score_uses_softmax_weighted_bin_centers() -> None:
    logits = jnp.asarray([[0.0, 0.0], [-20.0, 20.0]], dtype=jnp.float32)

    score, prob = logits_to_score(
        logits,
        min_bin=0.0,
        max_bin=2.0,
        no_bins=2,
        return_prob=True,
    )

    np.testing.assert_allclose(np.asarray(score), [1.0, 1.5], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(prob[0]),
        [0.5, 0.5],
        rtol=1e-6,
        atol=1e-6,
    )


def test_compute_contact_prob_sums_bins_below_threshold() -> None:
    logits = jnp.zeros((1, 1, 4), dtype=jnp.float32)

    contact = compute_contact_prob(
        logits,
        min_bin=0.0,
        max_bin=4.0,
        no_bins=4,
        thres=2.0,
    )

    np.testing.assert_allclose(
        np.asarray(contact),
        np.asarray([[0.5]]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_confidence_scores_from_logits_returns_basic_full_data() -> None:
    plddt = jnp.asarray([[[0.0, 0.0], [-20.0, 20.0]]], dtype=jnp.float32)
    pae = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    pde = jnp.ones((1, 2, 2, 2), dtype=jnp.float32)
    distogram = jnp.asarray(
        [
            [[20.0, -20.0], [20.0, -20.0]],
            [[20.0, -20.0], [20.0, -20.0]],
        ],
        dtype=jnp.float32,
    )

    scores = confidence_scores_from_logits(
        plddt_logits=plddt,
        pae_logits=pae,
        pde_logits=pde,
        distogram_logits=distogram,
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        contact_threshold=1.0,
    )

    assert scores["atom_plddt"].shape == (1, 2)
    assert scores["token_pair_pae"].shape == (1, 2, 2)
    assert scores["token_pair_pde"].shape == (1, 2, 2)
    assert scores["contact_probs"].shape == (2, 2)
    assert scores["summary_plddt"].shape == (1,)
    assert scores["summary_gpde"].shape == (1,)
    np.testing.assert_allclose(
        np.asarray(scores["summary_plddt"]),
        [125.0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_calculate_ptm_matches_protenix_row_mean_then_frame_max() -> None:
    pae_prob = jnp.asarray(
        [
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    has_frame = jnp.asarray([True, True])
    norm = calculate_normalization(2)
    weights = 1.0 / (1.0 + (np.asarray([0.5, 1.5]) / norm) ** 2)
    expected = max(weights[0], weights[1])

    ptm = calculate_ptm(
        pae_prob,
        has_frame,
        min_bin=0.0,
        max_bin=2.0,
        no_bins=2,
    )

    np.testing.assert_allclose(np.asarray(ptm), [expected], rtol=1e-6, atol=1e-6)


def test_calculate_iptm_uses_inter_chain_columns_only() -> None:
    pae_prob = jnp.asarray(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    has_frame = jnp.asarray([True, True])
    asym_id = jnp.asarray([0, 1])
    norm = calculate_normalization(2)
    weights = 1.0 / (1.0 + (np.asarray([0.5, 1.5]) / norm) ** 2)
    expected = max(weights[1], weights[0])

    iptm = calculate_iptm(
        pae_prob,
        has_frame,
        asym_id,
        min_bin=0.0,
        max_bin=2.0,
        no_bins=2,
    )

    np.testing.assert_allclose(np.asarray(iptm), [expected], rtol=1e-6, atol=1e-6)


def test_confidence_scores_adds_ptm_iptm_when_masks_are_given() -> None:
    logits = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    scores = confidence_scores_from_logits(
        plddt_logits=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        pae_logits=logits,
        pde_logits=logits,
        distogram_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        token_has_frame=jnp.asarray([True, True]),
        token_asym_id=jnp.asarray([0, 1]),
    )

    assert scores["summary_ptm"].shape == (1,)
    assert scores["summary_iptm"].shape == (1,)
    assert scores["summary_ranking_score"].shape == (1,)


def test_calculate_chain_based_plddt_matches_token_chain_masks() -> None:
    atom_plddt = jnp.asarray([[10.0, 20.0, 30.0]], dtype=jnp.float32)
    asym_id = jnp.asarray([0, 0, 1])
    atom_to_token_idx = jnp.asarray([0, 1, 2])

    out = calculate_chain_based_plddt(atom_plddt, asym_id, atom_to_token_idx)

    np.testing.assert_allclose(np.asarray(out["chain_plddt"]), [[15.0, 30.0]])
    np.testing.assert_allclose(
        np.asarray(out["chain_pair_plddt"]),
        [[[0.0, 20.0], [20.0, 0.0]]],
    )


def test_calculate_chain_pair_pae_returns_weighted_mean_and_min() -> None:
    token_pair_pae = jnp.asarray(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
        dtype=jnp.float32,
    )
    asym_id = jnp.asarray([0, 0, 1])
    token_has_frame = jnp.asarray([True, True, True])

    out = calculate_chain_pair_pae(token_pair_pae, asym_id, token_has_frame)

    np.testing.assert_allclose(
        np.asarray(out["chain_pair_pae_mean"]),
        [[[3.0, 4.5], [7.5, 9.0]]],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(out["chain_pair_pae_min"]),
        [[[1.0, 3.0], [7.0, 9.0]]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_calculate_chain_based_gpde_returns_intra_and_interface_values() -> None:
    token_pair_pde = jnp.asarray(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
        dtype=jnp.float32,
    )
    contact_probs = jnp.ones((3, 3), dtype=jnp.float32)
    asym_id = jnp.asarray([0, 0, 1])

    out = calculate_chain_based_gpde(token_pair_pde, contact_probs, asym_id)

    np.testing.assert_allclose(np.asarray(out["chain_gpde"]), [[3.0, 9.0]])
    np.testing.assert_allclose(
        np.asarray(out["chain_pair_gpde"]),
        [[[0.0, 4.5], [4.5, 0.0]]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_confidence_scores_adds_chain_metrics_when_chain_inputs_are_given() -> None:
    logits = jnp.zeros((1, 3, 3, 2), dtype=jnp.float32)
    scores = confidence_scores_from_logits(
        plddt_logits=jnp.zeros((1, 3, 2), dtype=jnp.float32),
        pae_logits=logits,
        pde_logits=logits,
        distogram_logits=jnp.zeros((3, 3, 2), dtype=jnp.float32),
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        token_has_frame=jnp.asarray([True, True, True]),
        token_asym_id=jnp.asarray([0, 0, 1]),
        atom_to_token_idx=jnp.asarray([0, 1, 2]),
    )

    assert scores["chain_plddt"].shape == (1, 2)
    assert scores["chain_pair_plddt"].shape == (1, 2, 2)
    assert scores["chain_pair_pae_mean"].shape == (1, 2, 2)
    assert scores["chain_pair_pae_min"].shape == (1, 2, 2)
    assert scores["chain_gpde"].shape == (1, 2)
    assert scores["chain_pair_gpde"].shape == (1, 2, 2)


def test_calculate_clash_flags_dense_inter_chain_contacts() -> None:
    coords = jnp.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.6],
            ],
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
            ],
        ],
        dtype=jnp.float32,
    )
    asym_id = jnp.asarray([0, 0, 1, 1])
    atom_to_token_idx = jnp.asarray([0, 1, 2, 3])

    has_clash = calculate_clash(
        coords,
        asym_id,
        atom_to_token_idx,
        threshold=1.1,
    )

    np.testing.assert_array_equal(np.asarray(has_clash), np.asarray([True, False]))


def test_confidence_scores_adds_clash_penalized_ranking() -> None:
    logits = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    coords = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    scores = confidence_scores_from_logits(
        plddt_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        pae_logits=logits,
        pde_logits=logits,
        distogram_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        token_has_frame=jnp.asarray([True, True]),
        token_asym_id=jnp.asarray([0, 1]),
        atom_to_token_idx=jnp.asarray([0, 1]),
        atom_coordinate=coords,
    )

    assert scores["has_clash"].shape == (2,)
    assert scores["summary_ranking_score"].shape == (2,)
    assert scores["summary_ranking_score"][0] < scores["summary_ranking_score"][1]


def test_calculate_vdw_clash_uses_element_radii() -> None:
    coords = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    asym_id = jnp.asarray([0, 1])
    atom_to_token_idx = jnp.asarray([0, 1])
    elements_one_hot = jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    has_vdw = calculate_vdw_clash(
        coords,
        asym_id,
        atom_to_token_idx,
        elements_one_hot,
        threshold=0.75,
    )

    np.testing.assert_array_equal(np.asarray(has_vdw), np.asarray([True, False]))


def test_calculate_vdw_clash_skips_same_molecule_pairs() -> None:
    coords = jnp.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=jnp.float32)
    asym_id = jnp.asarray([0, 1])
    atom_to_token_idx = jnp.asarray([0, 1])
    elements_one_hot = jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    has_vdw = calculate_vdw_clash(
        coords,
        asym_id,
        atom_to_token_idx,
        elements_one_hot,
        mol_id=jnp.asarray([5, 5]),
        threshold=0.75,
    )

    np.testing.assert_array_equal(np.asarray(has_vdw), np.asarray([False]))


def test_confidence_scores_adds_vdw_penalized_ranking_when_inputs_are_given() -> None:
    logits = jnp.zeros((2, 2, 2, 2), dtype=jnp.float32)
    coords = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    scores = confidence_scores_from_logits(
        plddt_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        pae_logits=logits,
        pde_logits=logits,
        distogram_logits=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        plddt_max_bin=2.0,
        pae_max_bin=2.0,
        pde_max_bin=2.0,
        distogram_min_bin=0.0,
        distogram_max_bin=2.0,
        token_has_frame=jnp.asarray([True, True]),
        token_asym_id=jnp.asarray([0, 1]),
        atom_to_token_idx=jnp.asarray([0, 1]),
        atom_coordinate=coords,
        elements_one_hot=jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
    )

    assert scores["has_vdw_clash"].shape == (2,)
    assert scores["summary_ranking_score_vdw_penalized"].shape == (2,)
    assert (
        scores["summary_ranking_score_vdw_penalized"][0]
        < scores["summary_ranking_score_vdw_penalized"][1]
    )


def test_confidence_head_stacks_sample_axis_like_protenix() -> None:
    params = _empty_confidence_params(c_s_inputs=3, c_s=2, c_z=2)
    features = {
        "distogram_rep_atom_mask": jnp.asarray([1, 1, 0]),
        "atom_to_token_idx": jnp.asarray([0, 1, 1]),
        "atom_to_tokatom_idx": jnp.asarray([0, 0, 1]),
    }
    coordinates = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [6.0, 8.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    output = confidence_head(
        features,
        s_inputs=jnp.zeros((2, 3), dtype=jnp.float32),
        s_trunk=jnp.zeros((2, 2), dtype=jnp.float32),
        z_trunk=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        pair_mask=None,
        x_pred_coords=coordinates,
        params=params,
    )

    assert output["plddt"].shape == (2, 3, 2)
    assert output["pae"].shape == (2, 2, 2, 1)
    assert output["pde"].shape == (2, 2, 2, 1)
    assert output["resolved"].shape == (2, 3, 1)


def _empty_confidence_params(
    *,
    c_s_inputs: int,
    c_s: int,
    c_z: int,
) -> ConfidenceHeadParams:
    return ConfidenceHeadParams(
        input_strunk_ln=LayerNormParams(
            weight=jnp.ones((c_s,)),
            bias=jnp.zeros((c_s,)),
        ),
        linear_s1=LinearParams(weight=jnp.zeros((c_z, c_s_inputs)), bias=None),
        linear_s2=LinearParams(weight=jnp.zeros((c_z, c_s_inputs)), bias=None),
        distance_embedding=ConfidenceDistanceEmbeddingParams(
            lower_bins=jnp.asarray([0.0]),
            upper_bins=jnp.asarray([10.0]),
            linear_d=LinearParams(weight=jnp.zeros((c_z, 1)), bias=None),
            linear_d_wo_onehot=LinearParams(weight=jnp.zeros((c_z, 1)), bias=None),
        ),
        pairformer_stack=PairformerStackParams(blocks=()),
        output=ConfidenceOutputParams(
            pae_ln=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
            pde_ln=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
            plddt_ln=LayerNormParams(weight=jnp.ones((c_s,)), bias=jnp.zeros((c_s,))),
            resolved_ln=LayerNormParams(
                weight=jnp.ones((c_s,)),
                bias=jnp.zeros((c_s,)),
            ),
            linear_pae=LinearParams(weight=jnp.zeros((1, c_z)), bias=None),
            linear_pde=LinearParams(weight=jnp.zeros((1, c_z)), bias=None),
            plddt_weight=jnp.zeros((2, c_s, 2)),
            resolved_weight=jnp.zeros((2, c_s, 1)),
        ),
    )
