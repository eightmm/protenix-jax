"""Top-level trunk initialization helpers for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from protenix_jax.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)
from protenix_jax.models.trunk_blocks.embedders import (
    ConstraintEmbedderParams,
    RelativePositionParams,
    constraint_embedder,
    relative_position_encoding,
    relative_position_features,
)
from protenix_jax.models.trunk_blocks.msa import MSAModuleParams, msa_module
from protenix_jax.models.trunk_blocks.pairformer import (
    PairformerStackParams,
    pairformer_stack,
)
from protenix_jax.models.trunk_blocks.template import (
    TemplateEmbedderParams,
    template_embedder,
)


class TrunkInitializationParams(NamedTuple):
    """Parameters for Protenix ``s_init`` and ``z_init`` construction."""

    linear_sinit: LinearParams
    linear_zinit1: LinearParams
    linear_zinit2: LinearParams
    relative_position: RelativePositionParams
    linear_token_bond: LinearParams


class RecyclingProjectionParams(NamedTuple):
    """Parameters for the root-level recycling projections."""

    layernorm_z: LayerNormParams
    linear_z: LinearParams
    layernorm_s: LayerNormParams
    linear_s: LinearParams


class TrunkParams(NamedTuple):
    """Root-level Protenix trunk parameters before the Pairformer stack."""

    initial: TrunkInitializationParams
    recycling: RecyclingProjectionParams


class PairformerOutputParams(NamedTuple):
    """Parameters for Protenix ``get_pairformer_output`` after input embedding."""

    trunk: TrunkParams
    constraint: ConstraintEmbedderParams
    template: TemplateEmbedderParams
    msa: MSAModuleParams
    pairformer_stack: PairformerStackParams


def trunk_initial_embeddings(
    s_inputs: jnp.ndarray,
    relp: jnp.ndarray,
    token_bonds: jnp.ndarray,
    params: TrunkInitializationParams,
    *,
    z_constraint: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build ``s_init`` and ``z_init`` as in Protenix ``get_pairformer_output``."""

    s_init = linear(s_inputs, params.linear_sinit)
    z_init = linear(s_init, params.linear_zinit1)[..., :, None, :] + linear(
        s_init,
        params.linear_zinit2,
    )[..., None, :, :]
    z_init = z_init + relative_position_encoding(relp, params.relative_position)
    z_init = z_init + linear(token_bonds[..., None], params.linear_token_bond)
    if z_constraint is not None:
        z_init = z_init + z_constraint
    return s_init, z_init


def recycle_embeddings(
    s_init: jnp.ndarray,
    z_init: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: RecyclingProjectionParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Protenix root-level recycling projections without dropout."""

    z_out = z_init + linear(layer_norm(z, params.layernorm_z), params.linear_z)
    s_out = s_init + linear(layer_norm(s, params.layernorm_s), params.linear_s)
    return s_out, z_out


def pairformer_output_from_s_inputs(
    input_feature_dict: dict[str, jnp.ndarray],
    s_inputs: jnp.ndarray,
    params: PairformerOutputParams,
    *,
    n_cycle: int = 1,
    pair_mask: jnp.ndarray | None = None,
    use_pairformer_scan: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply Protenix trunk from precomputed ``InputFeatureEmbedder`` output."""

    z_constraint = None
    if "constraint_feature" in input_feature_dict:
        z_constraint = constraint_embedder(
            input_feature_dict["constraint_feature"],
            params.constraint,
        )

    relp = input_feature_dict.get("relp")
    if relp is None:
        relp = relative_position_features(input_feature_dict)
    s_init, z_init = trunk_initial_embeddings(
        s_inputs,
        relp,
        input_feature_dict["token_bonds"],
        params.trunk.initial,
        z_constraint=z_constraint,
    )

    s = jnp.zeros_like(s_init)
    z = jnp.zeros_like(z_init)
    for _ in range(n_cycle):
        s, z = recycle_embeddings(
            s_init,
            z_init,
            s,
            z,
            params.trunk.recycling,
        )
        z = z + template_embedder(
            input_feature_dict,
            z,
            pair_mask,
            params.template,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        )
        z = msa_module(
            input_feature_dict,
            z,
            s_inputs,
            pair_mask,
            params.msa,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        )
        if params.pairformer_stack.blocks:
            s, z = pairformer_stack(
                s,
                z,
                pair_mask,
                params.pairformer_stack,
                use_scan=use_pairformer_scan,
                triangle_mul_chunk_size=triangle_mul_chunk_size,
                triangle_att_q_chunk_size=triangle_att_q_chunk_size,
                single_att_q_chunk_size=single_att_q_chunk_size,
            )
    return s_inputs, s, z
