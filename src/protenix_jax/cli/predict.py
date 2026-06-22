"""Standalone Protenix JAX prediction from native weights."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--features", type=Path)
    feature_group.add_argument("--input-json", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-sample", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=200)
    parser.add_argument("--s-max", type=float, default=160.0)
    parser.add_argument("--s-min", type=float, default=4e-4)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-data", type=float, default=16.0)
    parser.add_argument("--n-cycle", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)
    parser.add_argument("--max-msa-rows", type=int, default=16384)
    parser.add_argument("--input-atom-heads", type=int, default=4)
    parser.add_argument("--atom-encoder-heads", type=int, default=4)
    parser.add_argument("--token-heads", type=int, default=16)
    parser.add_argument("--atom-decoder-heads", type=int, default=4)
    parser.add_argument("--triangle-mul-chunk-size", type=int)
    parser.add_argument("--triangle-att-q-chunk-size", type=int)
    parser.add_argument("--single-att-q-chunk-size", type=int)
    parser.add_argument("--token-q-chunk-size", type=int)
    parser.add_argument("--diffusion-chunk-size", type=int)
    parser.add_argument(
        "--chunk-policy",
        choices=("auto", "manual", "off"),
        default="auto",
        help="Resolve chunk knobs automatically, manually, or disable chunking.",
    )
    parser.add_argument("--no-pairformer-scan", action="store_true")
    parser.add_argument("--diffusion-scan", action="store_true")
    parser.add_argument("--no-confidence-scan", action="store_true")
    parser.add_argument("--no-confidence", action="store_true")
    parser.add_argument("--no-confidence-scores", action="store_true")
    parser.add_argument("--include-trunk", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--compile-cache",
        type=Path,
        default=Path("outputs/compile_cache"),
        help="Persistent XLA compilation cache dir (compile-time only, "
        "output-invariant).",
    )
    parser.add_argument(
        "--no-compile-cache",
        action="store_true",
        help="Disable the persistent compilation cache.",
    )
    parser.add_argument(
        "--model-name",
        default="auto",
        help="Known Protenix model name for runtime limits, or 'auto'.",
    )
    args = parser.parse_args(argv)

    if args.cpu_only:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax

    if not args.no_compile_cache:
        cache = args.compile_cache.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        print(f"compile cache: {cache}")

    from protenix_jax.bridge.weights_io import load_native_weights
    from protenix_jax.chunking import resolve_chunk_config
    from protenix_jax.data.featurize_json import featurize_protein_json, load_first_job
    from protenix_jax.data.static_io import load_static_feature_npz, save_output_npz
    from protenix_jax.models.predict import protenix_predict_static
    from protenix_jax.runtime_policy import (
        infer_model_name_from_path,
        validate_inference_limits,
    )

    if args.features is not None:
        features = load_static_feature_npz(args.features)
    else:
        try:
            features = featurize_protein_json(
                load_first_job(args.input_json),
                base_dir=args.input_json.parent,
                n_queries=args.n_queries,
                n_keys=args.n_keys,
                max_msa_rows=args.max_msa_rows,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    model_name = args.model_name
    if model_name == "auto":
        model_name = infer_model_name_from_path(args.weights)
    n_token = int(features["restype"].shape[-2])
    try:
        validate_inference_limits(model_name=model_name, n_token=n_token)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    params = load_native_weights(args.weights)
    chunk_config = resolve_chunk_config(
        n_token=n_token,
        n_sample=args.n_sample,
        policy=args.chunk_policy,
        triangle_mul_chunk_size=args.triangle_mul_chunk_size,
        triangle_att_q_chunk_size=args.triangle_att_q_chunk_size,
        single_att_q_chunk_size=args.single_att_q_chunk_size,
        token_q_chunk_size=args.token_q_chunk_size,
        diffusion_chunk_size=args.diffusion_chunk_size,
    )
    output = protenix_predict_static(
        params,
        features,
        key=jax.random.PRNGKey(args.seed),
        n_sample=args.n_sample,
        num_sampling_steps=args.n_step,
        s_max=args.s_max,
        s_min=args.s_min,
        rho=args.rho,
        sigma_data=args.sigma_data,
        recycling_steps=args.n_cycle,
        input_atom_heads=args.input_atom_heads,
        atom_encoder_heads=args.atom_encoder_heads,
        token_heads=args.token_heads,
        atom_decoder_heads=args.atom_decoder_heads,
        n_queries=args.n_queries,
        n_keys=args.n_keys,
        use_pairformer_scan=not args.no_pairformer_scan,
        use_confidence_scan=not args.no_confidence_scan,
        use_diffusion_scan=args.diffusion_scan,
        run_confidence=not args.no_confidence,
        run_confidence_scores=not args.no_confidence_scores,
        triangle_mul_chunk_size=chunk_config.triangle_mul_chunk_size,
        triangle_att_q_chunk_size=chunk_config.triangle_att_q_chunk_size,
        single_att_q_chunk_size=chunk_config.single_att_q_chunk_size,
        token_q_chunk_size=chunk_config.token_q_chunk_size,
        diffusion_chunk_size=chunk_config.diffusion_chunk_size,
    )
    save_output_npz(args.out, output, include_trunk=args.include_trunk)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()

