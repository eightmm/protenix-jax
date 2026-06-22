"""Top-level infer-only Protenix JAX orchestration."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from protenix_jax.models.diffusion.diffusion import (
    DiffusionModuleParams,
    diffusion_conditioning_prepare_cache,
    sample_diffusion_with_module,
)
from protenix_jax.models.heads.confidence import (
    ConfidenceHeadParams,
    confidence_head,
    confidence_scores_from_logits,
)
from protenix_jax.models.heads.head import DistogramParams, distogram_head
from protenix_jax.models.trunk_blocks.embedders import (
    InputFeatureEmbedderParams,
    input_feature_embedder,
)
from protenix_jax.models.trunk_blocks.trunk import (
    PairformerOutputParams,
    pairformer_output_from_s_inputs,
)


class ProtenixInferenceParams(NamedTuple):
    """Parameters for the current infer-only Protenix wrapper."""

    input_embedder: InputFeatureEmbedderParams
    pairformer_output: PairformerOutputParams
    diffusion: DiffusionModuleParams
    distogram: DistogramParams
    confidence: ConfidenceHeadParams


def protenix_infer_static(
    input_feature_dict: dict[str, jnp.ndarray | dict[str, jnp.ndarray]],
    params: ProtenixInferenceParams,
    noise_schedule: jnp.ndarray,
    *,
    key: jax.Array | None,
    n_sample: int,
    init_noise: jnp.ndarray | None = None,
    step_noises: tuple[jnp.ndarray, ...] | None = None,
    n_cycle: int = 1,
    pair_mask: jnp.ndarray | None = None,
    input_atom_heads: int = 4,
    atom_encoder_heads: int = 4,
    token_heads: int = 16,
    atom_decoder_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    sigma_data: float = 16.0,
    use_pairformer_scan: bool = True,
    use_confidence_scan: bool = True,
    use_diffusion_scan: bool = False,
    use_confidence_embedding: bool = True,
    run_confidence: bool = True,
    run_confidence_scores: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    token_q_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    centre_each_step: bool = True,
) -> dict[str, jnp.ndarray]:
    """Run the currently ported static-feature Protenix inference path."""

    n_token = int(input_feature_dict["restype"].shape[-2])
    s_inputs = input_feature_embedder(
        input_feature_dict,
        params.input_embedder,
        n_token=n_token,
        n_heads=input_atom_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_diffusion_scan,
    )
    s_inputs, s_trunk, z_trunk = pairformer_output_from_s_inputs(
        input_feature_dict,
        s_inputs,
        params.pairformer_output,
        n_cycle=n_cycle,
        pair_mask=pair_mask,
        use_pairformer_scan=use_pairformer_scan,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        single_att_q_chunk_size=single_att_q_chunk_size,
    )
    relp = input_feature_dict["relp"]
    pair_z = diffusion_conditioning_prepare_cache(
        relp,
        z_trunk,
        params.diffusion.conditioning,
    )
    coordinates = sample_diffusion_with_module(
        input_feature_dict,
        s_inputs,
        s_trunk,
        z_trunk,
        params.diffusion,
        noise_schedule,
        n_sample=n_sample,
        key=key,
        pair_z=pair_z,
        atom_encoder_heads=atom_encoder_heads,
        token_heads=token_heads,
        atom_decoder_heads=atom_decoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        sigma_data=sigma_data,
        use_scan=use_diffusion_scan,
        token_q_chunk_size=token_q_chunk_size,
        diffusion_chunk_size=diffusion_chunk_size,
        gamma0=gamma0,
        gamma_min=gamma_min,
        noise_scale_lambda=noise_scale_lambda,
        step_scale_eta=step_scale_eta,
        centre_each_step=centre_each_step,
        init_noise=init_noise,
        step_noises=step_noises,
    )
    output = {
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
        "coordinate": coordinates,
        "distogram_logits": distogram_head(z_trunk, params.distogram),
    }
    if run_confidence:
        confidence_logits = confidence_head(
            input_feature_dict,
            s_inputs,
            s_trunk,
            z_trunk,
            pair_mask,
            coordinates,
            params.confidence,
            use_embedding=use_confidence_embedding,
            use_scan=use_confidence_scan,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            single_att_q_chunk_size=single_att_q_chunk_size,
        )
        output.update(confidence_logits)
        if run_confidence_scores:
            output.update(
                confidence_scores_from_logits(
                    plddt_logits=confidence_logits["plddt"],
                    pae_logits=confidence_logits["pae"],
                    pde_logits=confidence_logits["pde"],
                    distogram_logits=output["distogram_logits"],
                    token_has_frame=input_feature_dict.get("has_frame"),
                    token_asym_id=input_feature_dict.get("asym_id"),
                    atom_to_token_idx=input_feature_dict.get("atom_to_token_idx"),
                    atom_coordinate=coordinates,
                    elements_one_hot=input_feature_dict.get("ref_element"),
                    mol_id=input_feature_dict.get("mol_id"),
                )
            )
    return output
