# protenix_jax

Experimental JAX inference port for Protenix.

This package is a standalone JAX runtime for Protenix inference. The runtime
loads native Protenix-JAX weights and does not import PyTorch or the canonical
`protenix` package.

## Scope

- Inference only.
- Static/precomputed features first.
- Runtime inference uses native Protenix-JAX weights only.
- Checkpoint inspection/conversion is not part of this standalone package.
- No training or preprocessing rewrite.
- No model weights committed to this repository.

## Commands

```bash
cd /home/jaemin/non-project/optimizing/protenix_jax
uv sync --extra dev
JAX_PLATFORMS=cpu uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

Build a static feature bundle from sequence-only Protenix JSON:

```bash
JAX_PLATFORMS=cpu uv run protenix-jax-featurize-json \
  --input /path/to/input.json \
  --out outputs/static_features.npz \
  --max-msa-rows 16384
```

Run standalone inference from a static `.npz` feature bundle:

```bash
JAX_PLATFORMS=cpu uv run protenix-jax-predict \
  --weights outputs/protenix-v2-jax-weights.pkl \
  --features /path/to/static_features.npz \
  --out outputs/static_infer.npz \
  --n-sample 1 \
  --n-step 200 \
  --model-name auto \
  --chunk-policy auto \
  --diffusion-chunk-size 1
```

Or pass a sequence-only Protenix JSON job directly:

```bash
JAX_PLATFORMS=cpu uv run protenix-jax-predict \
  --weights outputs/protenix-v2-jax-weights.pkl \
  --input-json /path/to/input.json \
  --out outputs/static_infer.npz \
  --n-sample 1 \
  --n-step 200 \
  --model-name auto \
  --chunk-policy auto \
  --max-msa-rows 16384
```

Nested `pad_info` feature entries are represented as `pad_info.<name>` keys in
the `.npz`. The package also exposes `save_static_feature_npz` for writing this
format from an in-memory feature dictionary.

## Current Port Coverage

- Primitive layers, attention leaves, triangle leaves.
- Input feature embedding and atom encoder input path.
- Template, MSA, trunk Pairformer, confidence head.
- Diffusion conditioning, diffusion atom encoder/decoder, and one-step
  denoising wrapper.
- Inference noise schedule, diffusion sampling loop, and static top-level
  inference wrapper with confidence logits.
- Basic confidence postprocessing: contact probabilities, atom pLDDT, token-pair
  PAE/PDE, summary pLDDT, summary GPDE, pTM, ipTM, and unpenalized ranking
  score.
- Chain/clash confidence postprocessing: chain pLDDT, chain-pair pLDDT, chain
  GPDE, chain-pair GPDE, chain-pair PAE mean/min, and AF3-style clash penalty.
- VDW clash postprocessing with Protenix/RDKit radii and same-molecule pair
  skipping.
- Native JAX weight load path; runtime inference runs without torch or upstream
  Protenix.
- Sequence-only `proteinChain` JSON to static-feature `.npz` CLI. This path
  supports `pairedMsaPath`/`unpairedMsaPath` A3M profile, deletion mean, and
  row-level `msa`/`has_deletion`/`deletion_value` features with
  `--max-msa-rows` depth control. It rejects ligands, nucleic acids, covalent
  bonds, templates, and modified residues instead of silently producing partial
  features.
- Static inference CLI from either `.npz` features or direct sequence-only
  Protenix JSON; `protenix-jax-static-infer` is a native-only alias for
  `protenix-jax-predict`.
- Standalone native-weight prediction CLI: `protenix-jax-predict`. This CLI has
  no checkpoint option and is tested to import neither `torch` nor upstream
  `protenix`.
- Torch-free library prediction wrapper:
  `protenix_jax.models.predict.protenix_predict_static`.
- Memory hooks: local atom windows, triangle chunks, Pairformer scan, global
  attention query chunks, diffusion sample chunks, and Protenix-style automatic
  token chunk thresholds.
- Runtime limits: `protenix-v2` rejects `n_token > 2560`, matching the
  canonical Protenix inference guard.

The automatic chunk policy follows Protenix's inference thresholds:
`<=1024: no chunk`, `<=1536: 512`, `<=2048: 256`, `<=2560: 128`,
and `>2560: 32`. Explicit CLI chunk arguments override the policy.
`--model-name auto` infers known Protenix model names from native weight
filenames when possible; pass `--model-name protenix-v2` for renamed weights.

Full Protenix raw-feature parity, including ligand/nucleic-acid featurization,
species-based paired-MSA row assembly, template ingestion, TFG, and efficient
fusion, is still in progress.
