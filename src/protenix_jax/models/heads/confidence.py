"""Confidence head pieces for the Protenix JAX inference port."""

from __future__ import annotations

from typing import NamedTuple

import jax.nn as jnn
import jax.numpy as jnp

from protenix_jax.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)
from protenix_jax.models.trunk_blocks.pairformer import (
    PairformerStackParams,
    pairformer_stack,
)


class ConfidenceDistanceEmbeddingParams(NamedTuple):
    """Parameters for confidence-head pair distance embedding."""

    lower_bins: jnp.ndarray
    upper_bins: jnp.ndarray
    linear_d: LinearParams
    linear_d_wo_onehot: LinearParams


class ConfidenceOutputParams(NamedTuple):
    """Parameters for confidence-head output projections."""

    pae_ln: LayerNormParams
    pde_ln: LayerNormParams
    plddt_ln: LayerNormParams
    resolved_ln: LayerNormParams
    linear_pae: LinearParams
    linear_pde: LinearParams
    plddt_weight: jnp.ndarray
    resolved_weight: jnp.ndarray


class ConfidenceHeadParams(NamedTuple):
    """Parameters for the single-sample confidence head path."""

    input_strunk_ln: LayerNormParams
    linear_s1: LinearParams
    linear_s2: LinearParams
    distance_embedding: ConfidenceDistanceEmbeddingParams
    pairformer_stack: PairformerStackParams
    output: ConfidenceOutputParams


RDKIT_VDWS = jnp.asarray(
    [
        1.2,
        1.4,
        2.2,
        1.9,
        1.8,
        1.7,
        1.6,
        1.55,
        1.5,
        1.54,
        2.4,
        2.2,
        2.1,
        2.1,
        1.95,
        1.8,
        1.8,
        1.88,
        2.8,
        2.4,
        2.3,
        2.15,
        2.05,
        2.05,
        2.05,
        2.05,
        2.0,
        2.0,
        2.0,
        2.1,
        2.1,
        2.1,
        2.05,
        1.9,
        1.9,
        2.02,
        2.9,
        2.55,
        2.4,
        2.3,
        2.15,
        2.1,
        2.05,
        2.05,
        2.0,
        2.05,
        2.1,
        2.2,
        2.2,
        2.25,
        2.2,
        2.2,
        2.1,
        2.1,
        2.16,
        3.0,
        2.7,
        2.5,
        2.48,
        2.47,
        2.45,
        2.43,
        2.42,
        2.4,
        2.38,
        2.37,
        2.35,
        2.33,
        2.32,
        2.3,
        2.28,
        2.27,
        2.25,
        2.2,
        2.1,
        2.05,
        2.0,
        2.0,
        2.05,
        2.1,
        2.05,
        2.2,
        2.3,
        2.3,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.4,
        2.0,
        2.3,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ],
    dtype=jnp.float32,
)


def get_bin_centers(
    min_bin: float,
    max_bin: float,
    no_bins: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Return Protenix score bin centers."""

    bin_width = (max_bin - min_bin) / no_bins
    boundaries = jnp.linspace(
        min_bin,
        max_bin - bin_width,
        no_bins,
        dtype=dtype,
    )
    return boundaries + 0.5 * bin_width


def logits_to_score(
    logits: jnp.ndarray,
    *,
    min_bin: float,
    max_bin: float,
    no_bins: int | None = None,
    return_prob: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Convert binned logits to Protenix-style expected scores."""

    if no_bins is None:
        no_bins = int(logits.shape[-1])
    prob = jnn.softmax(logits.astype(jnp.float32), axis=-1)
    bin_centers = get_bin_centers(
        min_bin,
        max_bin,
        no_bins,
        dtype=prob.dtype,
    )
    score = prob @ bin_centers
    if return_prob:
        return score, prob
    return score


def calculate_normalization(n_token: int | jnp.ndarray) -> jnp.ndarray:
    """TM-score normalization constant used by Protenix."""

    n = jnp.asarray(n_token, dtype=jnp.float32)
    return 1.24 * (jnp.maximum(n, 19.0) - 15.0) ** (1.0 / 3.0) - 1.8


def calculate_ptm(
    pae_prob: jnp.ndarray,
    has_frame: jnp.ndarray,
    *,
    min_bin: float,
    max_bin: float,
    no_bins: int | None = None,
    token_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute Protenix pTM with static-shape masking."""

    n_token = int(pae_prob.shape[-3])
    if no_bins is None:
        no_bins = int(pae_prob.shape[-1])
    if token_mask is None:
        token_mask = jnp.ones((n_token,), dtype=bool)
    else:
        token_mask = token_mask.astype(bool)
    has_frame = has_frame.astype(bool)
    valid_rows = token_mask & has_frame

    n_d = jnp.sum(token_mask.astype(jnp.float32))
    norm = calculate_normalization(n_d)
    centers = get_bin_centers(min_bin, max_bin, no_bins, dtype=jnp.float32)
    per_bin_weight = 1.0 / (1.0 + (centers / norm) ** 2)
    token_token_ptm = jnp.sum(pae_prob.astype(jnp.float32) * per_bin_weight, axis=-1)

    col_mask = token_mask.astype(token_token_ptm.dtype)
    denom = jnp.maximum(jnp.sum(col_mask), 1.0)
    row_mean = jnp.sum(token_token_ptm * col_mask, axis=-1) / denom
    row_score = jnp.where(valid_rows, row_mean, -jnp.inf)
    score = jnp.max(row_score, axis=-1)
    return jnp.where(jnp.any(valid_rows), score, jnp.zeros_like(score))


def calculate_iptm(
    pae_prob: jnp.ndarray,
    has_frame: jnp.ndarray,
    asym_id: jnp.ndarray,
    *,
    min_bin: float,
    max_bin: float,
    no_bins: int | None = None,
    token_mask: jnp.ndarray | None = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute Protenix ipTM with static-shape masking."""

    n_token = int(pae_prob.shape[-3])
    if no_bins is None:
        no_bins = int(pae_prob.shape[-1])
    if token_mask is None:
        token_mask = jnp.ones((n_token,), dtype=bool)
    else:
        token_mask = token_mask.astype(bool)
    has_frame = has_frame.astype(bool)
    asym_id = asym_id.astype(jnp.int32)
    valid_rows = token_mask & has_frame

    n_d = jnp.sum(token_mask.astype(jnp.float32))
    norm = calculate_normalization(n_d)
    centers = get_bin_centers(min_bin, max_bin, no_bins, dtype=jnp.float32)
    per_bin_weight = 1.0 / (1.0 + (centers / norm) ** 2)
    token_token_ptm = jnp.sum(pae_prob.astype(jnp.float32) * per_bin_weight, axis=-1)

    col_mask = token_mask.astype(token_token_ptm.dtype)
    is_diff_chain = (asym_id[None, :] != asym_id[:, None]).astype(token_token_ptm.dtype)
    denom = jnp.sum(is_diff_chain * col_mask[None, :], axis=-1)
    row_score = jnp.sum(token_token_ptm * is_diff_chain * col_mask, axis=-1) / (
        eps + denom
    )
    row_score = jnp.where(valid_rows, row_score, -jnp.inf)
    score = jnp.max(row_score, axis=-1)
    return jnp.where(jnp.any(valid_rows), score, jnp.zeros_like(score))


def calculate_chain_based_plddt(
    atom_plddt: jnp.ndarray,
    asym_id: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    *,
    n_chain: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute Protenix chain pLDDT summaries."""

    if n_chain is None:
        n_chain = int(jnp.max(asym_id)) + 1
    atom_chain_id = asym_id.astype(jnp.int32)[atom_to_token_idx]

    chain_vals = []
    for aid in range(n_chain):
        atom_mask = (atom_chain_id == aid).astype(atom_plddt.dtype)
        denom = jnp.maximum(jnp.sum(atom_mask), 1.0)
        chain_vals.append(jnp.sum(atom_plddt * atom_mask, axis=-1) / denom)
    chain_plddt = jnp.stack(chain_vals, axis=-1)

    pair_rows = []
    for aid_1 in range(n_chain):
        pair_cols = []
        for aid_2 in range(n_chain):
            if aid_1 == aid_2:
                pair_cols.append(jnp.zeros(atom_plddt.shape[:-1], dtype=jnp.float32))
            else:
                atom_mask = (
                    (atom_chain_id == aid_1) | (atom_chain_id == aid_2)
                ).astype(atom_plddt.dtype)
                denom = jnp.maximum(jnp.sum(atom_mask), 1.0)
                pair_cols.append(jnp.sum(atom_plddt * atom_mask, axis=-1) / denom)
        pair_rows.append(jnp.stack(pair_cols, axis=-1))
    chain_pair_plddt = jnp.stack(pair_rows, axis=-2)
    return {
        "chain_plddt": chain_plddt.astype(jnp.float32),
        "chain_pair_plddt": chain_pair_plddt.astype(jnp.float32),
    }


def calculate_chain_based_gpde(
    token_pair_pde: jnp.ndarray,
    contact_probs: jnp.ndarray,
    asym_id: jnp.ndarray,
    *,
    n_chain: int | None = None,
    eps: float = 1e-8,
) -> dict[str, jnp.ndarray]:
    """Compute Protenix chain and chain-pair gPDE summaries."""

    if n_chain is None:
        n_chain = int(jnp.max(asym_id)) + 1
    asym_id = asym_id.astype(jnp.int32)
    contact_probs = contact_probs.astype(token_pair_pde.dtype)

    def _weighted_mean(mask_1, mask_2):
        pair_mask = (mask_1[:, None] & mask_2[None, :]).astype(token_pair_pde.dtype)
        weights = contact_probs * pair_mask
        return jnp.sum(token_pair_pde * weights, axis=(-1, -2)) / (
            jnp.sum(weights, axis=(-1, -2)) + eps
        )

    chain_vals = []
    for aid in range(n_chain):
        mask = asym_id == aid
        chain_vals.append(_weighted_mean(mask, mask))
    chain_gpde = jnp.stack(chain_vals, axis=-1)

    pair_rows = []
    for aid_1 in range(n_chain):
        pair_cols = []
        for aid_2 in range(n_chain):
            if aid_1 == aid_2:
                pair_cols.append(
                    jnp.zeros(token_pair_pde.shape[:-2], dtype=jnp.float32)
                )
            elif aid_2 < aid_1:
                pair_cols.append(pair_rows[aid_2][..., aid_1])
            else:
                pair_cols.append(_weighted_mean(asym_id == aid_1, asym_id == aid_2))
        pair_rows.append(jnp.stack(pair_cols, axis=-1))
    chain_pair_gpde = jnp.stack(pair_rows, axis=-2)
    return {
        "chain_gpde": chain_gpde.astype(jnp.float32),
        "chain_pair_gpde": chain_pair_gpde.astype(jnp.float32),
    }


def calculate_chain_pair_pae(
    token_pair_pae: jnp.ndarray,
    asym_id: jnp.ndarray,
    token_has_frame: jnp.ndarray,
    contact_probs: jnp.ndarray | None = None,
    *,
    n_chain: int | None = None,
    eps: float = 1e-8,
) -> dict[str, jnp.ndarray]:
    """Compute Protenix chain-pair PAE mean and minimum summaries."""

    if n_chain is None:
        n_chain = int(jnp.max(asym_id)) + 1
    asym_id = asym_id.astype(jnp.int32)
    token_has_frame = token_has_frame.astype(bool)
    if contact_probs is None:
        contact_probs = jnp.ones(token_pair_pae.shape[-2:], dtype=token_pair_pae.dtype)
    else:
        contact_probs = contact_probs.astype(token_pair_pae.dtype)

    frame_mask = token_has_frame[:, None] & token_has_frame[None, :]
    mean_rows = []
    min_rows = []
    for aid_1 in range(n_chain):
        mean_cols = []
        min_cols = []
        for aid_2 in range(n_chain):
            pair_mask = (
                (asym_id[:, None] == aid_1)
                & (asym_id[None, :] == aid_2)
                & frame_mask
            )
            pair_mask_f = pair_mask.astype(token_pair_pae.dtype)
            weights = contact_probs * pair_mask_f
            weight_sum = jnp.sum(weights, axis=(-1, -2))
            mean_cols.append(
                jnp.where(
                    weight_sum > 0,
                    jnp.sum(token_pair_pae * weights, axis=(-1, -2))
                    / (weight_sum + eps),
                    jnp.nan,
                )
            )
            min_cols.append(
                jnp.min(
                    jnp.where(pair_mask, token_pair_pae, jnp.inf),
                    axis=(-1, -2),
                )
            )
        mean_rows.append(jnp.stack(mean_cols, axis=-1))
        min_rows.append(jnp.stack(min_cols, axis=-1))
    chain_pair_pae_mean = jnp.stack(mean_rows, axis=-2)
    chain_pair_pae_min = jnp.stack(min_rows, axis=-2)
    chain_pair_pae_min = jnp.where(
        jnp.isfinite(chain_pair_pae_min),
        chain_pair_pae_min,
        jnp.nan,
    )
    return {
        "chain_pair_pae_mean": chain_pair_pae_mean.astype(jnp.float32),
        "chain_pair_pae_min": chain_pair_pae_min.astype(jnp.float32),
    }


def calculate_clash(
    atom_coordinate: jnp.ndarray,
    asym_id: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    *,
    threshold: float = 1.1,
    n_chain: int | None = None,
) -> jnp.ndarray:
    """Compute a Protenix AF3-style inter-chain clash flag per sample.

    This covers the AF3 polymer-style clash penalty used in ranking. VDW
    ligand/polymer clash requires element radii and remains separate.
    """

    if n_chain is None:
        n_chain = int(jnp.max(asym_id)) + 1
    atom_chain_id = asym_id.astype(jnp.int32)[atom_to_token_idx]
    diff = atom_coordinate[..., :, None, :] - atom_coordinate[..., None, :, :]
    dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1))

    sample_clashes = []
    for aid_1 in range(n_chain):
        mask_1 = atom_chain_id == aid_1
        n_1 = jnp.sum(mask_1.astype(jnp.float32))
        for aid_2 in range(aid_1 + 1, n_chain):
            mask_2 = atom_chain_id == aid_2
            n_2 = jnp.sum(mask_2.astype(jnp.float32))
            pair_mask = mask_1[:, None] & mask_2[None, :]
            total_clash = jnp.sum(
                ((dist < threshold) & pair_mask).astype(jnp.float32),
                axis=(-1, -2),
            )
            relative_clash = total_clash / jnp.maximum(jnp.minimum(n_1, n_2), 1.0)
            sample_clashes.append((total_clash > 100.0) | (relative_clash > 0.5))
    if not sample_clashes:
        return jnp.zeros(atom_coordinate.shape[:-2], dtype=bool)
    return jnp.any(jnp.stack(sample_clashes, axis=-1), axis=-1)


def calculate_vdw_clash(
    atom_coordinate: jnp.ndarray,
    asym_id: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    elements_one_hot: jnp.ndarray,
    *,
    mol_id: jnp.ndarray | None = None,
    threshold: float = 0.75,
    n_chain: int | None = None,
) -> jnp.ndarray:
    """Compute inter-chain VDW clash flags using Protenix/RDKit radii."""

    if n_chain is None:
        n_chain = int(jnp.max(asym_id)) + 1
    atom_chain_id = asym_id.astype(jnp.int32)[atom_to_token_idx]
    element_order = jnp.argmax(elements_one_hot, axis=-1)
    radii = RDKIT_VDWS[element_order]
    diff = atom_coordinate[..., :, None, :] - atom_coordinate[..., None, :, :]
    dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1))
    vdw_sum = radii[:, None] + radii[None, :]
    relative_vdw_distance = dist / jnp.maximum(vdw_sum, 1e-8)

    sample_clashes = []
    for aid_1 in range(n_chain):
        mask_1 = atom_chain_id == aid_1
        skip_pair = jnp.asarray(False)
        mol_1 = jnp.asarray(-1)
        if mol_id is not None:
            mol_1 = jnp.max(jnp.where(mask_1, mol_id, -1))
        for aid_2 in range(aid_1 + 1, n_chain):
            mask_2 = atom_chain_id == aid_2
            skip_pair = jnp.asarray(False)
            if mol_id is not None:
                mol_2 = jnp.max(jnp.where(mask_2, mol_id, -2))
                skip_pair = mol_1 == mol_2
            pair_mask = mask_1[:, None] & mask_2[None, :]
            pair_clash = jnp.any(
                (relative_vdw_distance < threshold) & pair_mask,
                axis=(-1, -2),
            )
            pair_clash = jnp.where(skip_pair, jnp.zeros_like(pair_clash), pair_clash)
            sample_clashes.append(pair_clash)
    if not sample_clashes:
        return jnp.zeros(atom_coordinate.shape[:-2], dtype=bool)
    return jnp.any(jnp.stack(sample_clashes, axis=-1), axis=-1)


def compute_contact_prob(
    distogram_logits: jnp.ndarray,
    *,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    no_bins: int | None = None,
    thres: float = 8.0,
) -> jnp.ndarray:
    """Compute Protenix contact probabilities from distogram logits."""

    if no_bins is None:
        no_bins = int(distogram_logits.shape[-1])
    prob = jnn.softmax(distogram_logits.astype(jnp.float32), axis=-1)
    centers = get_bin_centers(min_bin, max_bin, no_bins, dtype=prob.dtype)
    contact_mask = centers < thres
    return jnp.sum(prob * contact_mask.astype(prob.dtype), axis=-1)


def confidence_scores_from_logits(
    *,
    plddt_logits: jnp.ndarray,
    pae_logits: jnp.ndarray,
    pde_logits: jnp.ndarray,
    distogram_logits: jnp.ndarray,
    plddt_min_bin: float = 0.0,
    plddt_max_bin: float = 1.0,
    plddt_no_bins: int | None = None,
    pae_min_bin: float = 0.0,
    pae_max_bin: float = 32.0,
    pae_no_bins: int | None = None,
    pde_min_bin: float = 0.0,
    pde_max_bin: float = 32.0,
    pde_no_bins: int | None = None,
    distogram_min_bin: float = 2.3125,
    distogram_max_bin: float = 21.6875,
    distogram_no_bins: int | None = None,
    contact_threshold: float = 8.0,
    token_has_frame: jnp.ndarray | None = None,
    token_asym_id: jnp.ndarray | None = None,
    atom_to_token_idx: jnp.ndarray | None = None,
    atom_coordinate: jnp.ndarray | None = None,
    elements_one_hot: jnp.ndarray | None = None,
    mol_id: jnp.ndarray | None = None,
    clash_threshold: float = 1.1,
    vdw_clash_threshold: float = 0.75,
    token_mask: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute the basic full-data confidence scores used in inference."""

    atom_plddt = logits_to_score(
        plddt_logits,
        min_bin=plddt_min_bin,
        max_bin=plddt_max_bin,
        no_bins=plddt_no_bins,
    )
    token_pair_pde = logits_to_score(
        pde_logits,
        min_bin=pde_min_bin,
        max_bin=pde_max_bin,
        no_bins=pde_no_bins,
    )
    token_pair_pae, pae_prob = logits_to_score(
        pae_logits,
        min_bin=pae_min_bin,
        max_bin=pae_max_bin,
        no_bins=pae_no_bins,
        return_prob=True,
    )
    contact_probs = compute_contact_prob(
        distogram_logits,
        min_bin=distogram_min_bin,
        max_bin=distogram_max_bin,
        no_bins=distogram_no_bins,
        thres=contact_threshold,
    )
    summary_plddt = jnp.mean(atom_plddt, axis=-1) * 100.0
    gpde_numer = jnp.sum(token_pair_pde * contact_probs, axis=(-1, -2))
    gpde_denom = jnp.sum(contact_probs, axis=(-1, -2))
    summary_gpde = jnp.where(gpde_denom > 0, gpde_numer / gpde_denom, 0.0)
    scores = {
        "atom_plddt": atom_plddt.astype(jnp.float32),
        "token_pair_pde": token_pair_pde.astype(jnp.float32),
        "token_pair_pae": token_pair_pae.astype(jnp.float32),
        "contact_probs": contact_probs.astype(jnp.float32),
        "summary_plddt": summary_plddt.astype(jnp.float32),
        "summary_gpde": summary_gpde.astype(jnp.float32),
    }
    if token_has_frame is not None and token_asym_id is not None:
        summary_ptm = calculate_ptm(
            pae_prob,
            token_has_frame,
            min_bin=pae_min_bin,
            max_bin=pae_max_bin,
            no_bins=pae_no_bins,
            token_mask=token_mask,
        )
        summary_iptm = calculate_iptm(
            pae_prob,
            token_has_frame,
            token_asym_id,
            min_bin=pae_min_bin,
            max_bin=pae_max_bin,
            no_bins=pae_no_bins,
            token_mask=token_mask,
        )
        ranking_score = 0.8 * summary_iptm + 0.2 * summary_ptm
        if atom_to_token_idx is not None and atom_coordinate is not None:
            has_clash = calculate_clash(
                atom_coordinate,
                token_asym_id,
                atom_to_token_idx,
                threshold=clash_threshold,
            )
            ranking_score = ranking_score - 100.0 * has_clash.astype(
                ranking_score.dtype
            )
            scores["has_clash"] = has_clash
            if elements_one_hot is not None:
                has_vdw_clash = calculate_vdw_clash(
                    atom_coordinate,
                    token_asym_id,
                    atom_to_token_idx,
                    elements_one_hot,
                    mol_id=mol_id,
                    threshold=vdw_clash_threshold,
                )
                scores["has_vdw_clash"] = has_vdw_clash
                scores["summary_ranking_score_vdw_penalized"] = (
                    ranking_score
                    - 100.0 * has_vdw_clash.astype(ranking_score.dtype)
                ).astype(jnp.float32)
        scores.update(
            {
                "summary_ptm": summary_ptm.astype(jnp.float32),
                "summary_iptm": summary_iptm.astype(jnp.float32),
                "summary_ranking_score": ranking_score.astype(jnp.float32),
            }
        )
        scores.update(
            calculate_chain_based_gpde(
                token_pair_pde,
                contact_probs,
                token_asym_id,
            )
        )
        scores.update(
            calculate_chain_pair_pae(
                token_pair_pae,
                token_asym_id,
                token_has_frame,
                contact_probs,
            )
        )
        if atom_to_token_idx is not None:
            scores.update(
                calculate_chain_based_plddt(
                    atom_plddt,
                    token_asym_id,
                    atom_to_token_idx,
                )
            )
    return scores


def confidence_one_hot(
    x: jnp.ndarray,
    lower_bins: jnp.ndarray,
    upper_bins: jnp.ndarray,
) -> jnp.ndarray:
    """Open-interval distance binning matching Protenix ``one_hot``."""

    return ((x[..., None] > lower_bins) & (x[..., None] < upper_bins)).astype(x.dtype)


def confidence_distance_embedding(
    x_pred_rep_coords: jnp.ndarray,
    params: ConfidenceDistanceEmbeddingParams,
) -> jnp.ndarray:
    """Embed representative-atom pair distances for ConfidenceHead."""

    coords = x_pred_rep_coords.astype(jnp.float32)
    diff = coords[..., :, None, :] - coords[..., None, :, :]
    distance = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1))
    return linear(
        confidence_one_hot(distance, params.lower_bins, params.upper_bins),
        params.linear_d,
    ) + linear(distance[..., None], params.linear_d_wo_onehot)


def confidence_output_logits(
    s_single: jnp.ndarray,
    z_pair: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    atom_to_tokatom_idx: jnp.ndarray,
    params: ConfidenceOutputParams,
) -> dict[str, jnp.ndarray]:
    """Project pair and atom confidence logits."""

    pae = linear(layer_norm(z_pair, params.pae_ln), params.linear_pae)
    pde = linear(
        layer_norm(z_pair + jnp.swapaxes(z_pair, -2, -3), params.pde_ln),
        params.linear_pde,
    )
    atom_single = s_single[..., atom_to_token_idx, :]
    plddt_weight = params.plddt_weight[atom_to_tokatom_idx]
    resolved_weight = params.resolved_weight[atom_to_tokatom_idx]
    plddt = jnp.einsum(
        "...nc,ncb->...nb",
        layer_norm(atom_single, params.plddt_ln),
        plddt_weight,
    )
    resolved = jnp.einsum(
        "...nc,ncb->...nb",
        layer_norm(atom_single, params.resolved_ln),
        resolved_weight,
    )
    return {
        "plddt": plddt.astype(jnp.float32),
        "pae": pae.astype(jnp.float32),
        "pde": pde.astype(jnp.float32),
        "resolved": resolved.astype(jnp.float32),
    }


def confidence_head_single_sample(
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    x_pred_rep_coords: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    atom_to_tokatom_idx: jnp.ndarray,
    params: ConfidenceHeadParams,
    *,
    use_embedding: bool = True,
    use_scan: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Run the ConfidenceHead inference path for one predicted sample."""

    s_trunk = layer_norm(jnp.clip(s_trunk, -512.0, 512.0), params.input_strunk_ln)
    z_base = z_trunk if use_embedding else jnp.zeros_like(z_trunk)
    z_init = linear(s_inputs, params.linear_s1)[..., :, None, :] + linear(
        s_inputs,
        params.linear_s2,
    )[..., None, :, :]
    z_pair = z_base + z_init + confidence_distance_embedding(
        x_pred_rep_coords,
        params.distance_embedding,
    )
    if params.pairformer_stack.blocks:
        s_single, z_pair = pairformer_stack(
            s_trunk,
            z_pair,
            pair_mask,
            params.pairformer_stack,
            use_scan=use_scan,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            single_att_q_chunk_size=single_att_q_chunk_size,
        )
        if s_single is None:
            raise ValueError("ConfidenceHead requires PairformerStack single output")
    else:
        s_single = s_trunk
    return confidence_output_logits(
        s_single.astype(jnp.float32),
        z_pair.astype(jnp.float32),
        atom_to_token_idx,
        atom_to_tokatom_idx,
        params.output,
    )


def confidence_head(
    input_feature_dict: dict[str, jnp.ndarray | dict[str, jnp.ndarray]],
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    x_pred_coords: jnp.ndarray,
    params: ConfidenceHeadParams,
    *,
    use_embedding: bool = True,
    use_scan: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Run the Protenix confidence head over the sample axis.

    The original PyTorch inference path loops over samples to reduce peak pair
    memory. This JAX path keeps that contract and stacks outputs on the same
    axes as Protenix.
    """

    n_token = int(s_inputs.shape[-2])
    rep_atom_idx = jnp.nonzero(
        input_feature_dict["distogram_rep_atom_mask"].astype(bool),
        size=n_token,
    )[0]
    x_pred_rep_coords = jnp.take(x_pred_coords, rep_atom_idx, axis=-2)
    n_sample = int(x_pred_rep_coords.shape[-3])
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    atom_to_tokatom_idx = input_feature_dict["atom_to_tokatom_idx"]

    outputs = []
    for sample_index in range(n_sample):
        outputs.append(
            confidence_head_single_sample(
                s_inputs,
                s_trunk,
                z_trunk,
                pair_mask,
                jnp.take(x_pred_rep_coords, sample_index, axis=-3),
                atom_to_token_idx,
                atom_to_tokatom_idx,
                params,
                use_embedding=use_embedding,
                use_scan=use_scan,
                triangle_mul_chunk_size=triangle_mul_chunk_size,
                triangle_att_q_chunk_size=triangle_att_q_chunk_size,
                single_att_q_chunk_size=single_att_q_chunk_size,
            )
        )

    return {
        "plddt": jnp.stack([out["plddt"] for out in outputs], axis=-3),
        "pae": jnp.stack([out["pae"] for out in outputs], axis=-4),
        "pde": jnp.stack([out["pde"] for out in outputs], axis=-4),
        "resolved": jnp.stack([out["resolved"] for out in outputs], axis=-3),
    }
