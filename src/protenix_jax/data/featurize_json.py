"""Sequence-only Protenix JSON to static feature conversion."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from protenix_jax.data.static_io import save_static_feature_npz

RESTYPE_INDEX = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    "X": 20,
}
MSA_PROTEIN_INDEX = {
    **RESTYPE_INDEX,
    "B": RESTYPE_INDEX["D"],
    "J": RESTYPE_INDEX["X"],
    "O": RESTYPE_INDEX["X"],
    "U": RESTYPE_INDEX["C"],
    "Z": RESTYPE_INDEX["E"],
    "-": 31,
}
ELEMENT_INDEX = {"H": 0, "C": 5, "N": 6, "O": 7, "S": 15}
ATOM_TO_TOKATOM_INDEX = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}


def load_first_job(path: str | Path) -> dict[str, Any]:
    """Load the first Protenix JSON job."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError("input JSON must be a non-empty top-level list")
    job = data[0]
    if not isinstance(job, dict):
        raise ValueError("first input JSON entry must be an object")
    return job


def featurize_protein_json(
    job: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
    n_queries: int = 32,
    n_keys: int = 128,
    max_msa_rows: int = 16384,
) -> dict[str, Any]:
    """Build static features for proteinChain inputs."""

    if n_keys < n_queries or n_queries % 2 or n_keys % 2:
        raise ValueError("n_keys must be >= n_queries and both must be even")
    if max_msa_rows <= 0:
        raise ValueError("max_msa_rows must be positive")
    chains = _expand_protein_chains(
        job,
        base_dir=base_dir,
        max_msa_rows=max_msa_rows,
    )
    n_token = sum(len(chain["sequence"]) for chain in chains)
    if n_token <= 0:
        raise ValueError("at least one protein residue is required")

    restype = np.zeros((n_token, 32), dtype=np.float32)
    residue_index = np.zeros((n_token,), dtype=np.int64)
    token_index = np.zeros((n_token,), dtype=np.int64)
    asym_id = np.zeros((n_token,), dtype=np.int64)
    entity_id = np.zeros((n_token,), dtype=np.int64)
    sym_id = np.zeros((n_token,), dtype=np.int64)
    has_frame = np.ones((n_token,), dtype=np.int64)
    profile = np.zeros((n_token, 32), dtype=np.float32)
    deletion_mean = np.zeros((n_token,), dtype=np.float32)

    atom_to_token_idx: list[int] = []
    atom_to_tokatom_idx: list[int] = []
    ref_pos: list[tuple[float, float, float]] = []
    ref_element: list[str] = []
    ref_atom_names: list[str] = []
    distogram_rep_atom_mask: list[int] = []
    mol_id: list[int] = []

    token_i = 0
    for chain in chains:
        sequence = chain["sequence"]
        chain_profile, chain_deletion_mean = _chain_msa_features(chain)
        for pos, aa in enumerate(sequence, start=1):
            restype[token_i, RESTYPE_INDEX[aa]] = 1.0
            profile[token_i] = chain_profile[pos - 1]
            deletion_mean[token_i] = chain_deletion_mean[pos - 1]
            residue_index[token_i] = pos
            asym_id[token_i] = chain["asym_id"]
            entity_id[token_i] = chain["entity_id"]
            sym_id[token_i] = chain["sym_id"]
            atom_names = _residue_atom_names(aa)
            for atom_name in atom_names:
                atom_to_token_idx.append(token_i)
                atom_to_tokatom_idx.append(ATOM_TO_TOKATOM_INDEX[atom_name])
                ref_pos.append(_dummy_atom_position(pos, atom_name, chain["asym_id"]))
                ref_element.append(_element_from_atom_name(atom_name))
                ref_atom_names.append(atom_name)
                distogram_rep_atom_mask.append(
                    int(atom_name == _distogram_rep_atom_name(aa))
                )
                mol_id.append(chain["asym_id"])
            token_i += 1

    atom_to_token = np.asarray(atom_to_token_idx, dtype=np.int64)
    ref_pos_arr = np.asarray(ref_pos, dtype=np.float32)
    d_lm, v_lm, pad_info = _local_atom_geometry(
        ref_pos_arr,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    relp = _relative_position_features(
        asym_id=asym_id,
        residue_index=residue_index,
        entity_id=entity_id,
        sym_id=sym_id,
        token_index=token_index,
    )
    msa, deletion_matrix = _merge_chain_msa_features(
        chains,
        max_msa_rows=max_msa_rows,
    )
    return {
        "atom_to_token_idx": atom_to_token,
        "ref_pos": ref_pos_arr,
        "ref_charge": np.zeros((len(atom_to_token),), dtype=np.float32),
        "ref_mask": np.ones((len(atom_to_token),), dtype=np.float32),
        "ref_atom_name_chars": _encode_atom_name_chars(ref_atom_names),
        "ref_element": _encode_elements(ref_element),
        "d_lm": d_lm,
        "v_lm": v_lm,
        "pad_info": pad_info,
        "restype": restype,
        "profile": profile,
        "deletion_mean": deletion_mean,
        "msa": msa,
        "has_deletion": np.clip(deletion_matrix, 0.0, 1.0).astype(np.float32),
        "deletion_value": (
            np.arctan(deletion_matrix.astype(np.float32) / 3.0) * (2.0 / np.pi)
        ).astype(np.float32),
        "relp": relp,
        "token_bonds": np.zeros((n_token, n_token), dtype=np.float32),
        "residue_index": residue_index,
        "token_index": token_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "has_frame": has_frame,
        "distogram_rep_atom_mask": np.asarray(
            distogram_rep_atom_mask,
            dtype=np.float32,
        ),
        "atom_to_tokatom_idx": np.asarray(atom_to_tokatom_idx, dtype=np.int64),
        "mol_id": np.asarray(mol_id, dtype=np.int64),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)
    parser.add_argument("--max-msa-rows", type=int, default=16384)
    args = parser.parse_args(argv)

    features = featurize_protein_json(
        load_first_job(args.input),
        base_dir=args.input.parent,
        n_queries=args.n_queries,
        n_keys=args.n_keys,
        max_msa_rows=args.max_msa_rows,
    )
    save_static_feature_npz(args.out, features)
    print(f"wrote: {args.out}")


def parse_a3m_profile(
    query_sequence: str,
    a3m: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse protein A3M content into Protenix profile and deletion mean."""

    msa, deletion_matrix = parse_a3m_rows(query_sequence, a3m)
    profile = (msa[..., None] == np.arange(32)).sum(axis=0) / msa.shape[0]
    return profile.astype(np.float32), deletion_matrix.mean(axis=0)


def parse_a3m_rows(
    query_sequence: str,
    a3m: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse protein A3M content into Protenix MSA rows and deletion matrix."""

    records = _parse_a3m_records(a3m)
    if not records:
        records = [query_sequence]
    rows = []
    deletions = []
    for sequence in records:
        row, deletion = _aligned_protein_row(sequence)
        if len(row) != len(query_sequence):
            raise ValueError(
                "A3M aligned length must match query length: "
                f"{len(row)} != {len(query_sequence)}"
        )
        rows.append(row)
        deletions.append(deletion)
    msa = np.asarray(rows, dtype=np.int64)
    deletion_matrix = np.asarray(deletions, dtype=np.float32)
    return msa, deletion_matrix


def _expand_protein_chains(
    job: dict[str, Any],
    *,
    base_dir: str | Path | None,
    max_msa_rows: int,
) -> list[dict[str, Any]]:
    if job.get("covalent_bonds"):
        raise ValueError("covalent_bonds are not supported")
    sequences = job.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("job must contain a non-empty sequences list")
    chains: list[dict[str, Any]] = []
    entity_by_sequence: dict[str, int] = {}
    entity_sym_counts: dict[int, int] = {}
    for entry in sequences:
        if not isinstance(entry, dict) or set(entry) != {"proteinChain"}:
            raise ValueError("Only proteinChain inputs are supported")
        chain = entry["proteinChain"]
        if not isinstance(chain, dict):
            raise ValueError("proteinChain entry must be an object")
        if "templatesPath" in chain:
            raise ValueError("template paths are not supported")
        if chain.get("modifications"):
            raise ValueError("proteinChain modifications are not supported")
        sequence = _normalize_sequence(chain.get("sequence"))
        msa_profile, msa_deletion_mean, msa, deletion_matrix = _load_chain_msa_features(
            sequence,
            chain,
            base_dir=base_dir,
            max_msa_rows=max_msa_rows,
        )
        count = int(chain.get("count", 1))
        if count <= 0:
            raise ValueError("proteinChain count must be positive")
        ids = chain.get("id")
        if ids is not None and len(ids) != count:
            raise ValueError("proteinChain id length must match count")
        entity = entity_by_sequence.setdefault(sequence, len(entity_by_sequence))
        for _ in range(count):
            sym = entity_sym_counts.get(entity, 0)
            entity_sym_counts[entity] = sym + 1
            chains.append(
                {
                    "sequence": sequence,
                    "entity_id": entity,
                    "asym_id": len(chains),
                    "sym_id": sym,
                    "profile": msa_profile,
                    "deletion_mean": msa_deletion_mean,
                    "msa": msa,
                    "deletion_matrix": deletion_matrix,
                }
            )
    return chains


def _load_chain_msa_features(
    sequence: str,
    chain: dict[str, Any],
    *,
    base_dir: str | Path | None,
    max_msa_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = [
        chain[key]
        for key in ("pairedMsaPath", "unpairedMsaPath")
        if key in chain and chain[key]
    ]
    profiles = []
    deletion_means = []
    msa_rows = []
    deletion_rows = []
    for path in paths:
        a3m_path = _resolve_path(path, base_dir=base_dir)
        a3m = a3m_path.read_text(encoding="utf-8")
        msa, deletion_matrix = parse_a3m_rows(sequence, a3m)
        profile = (msa[..., None] == np.arange(32)).sum(axis=0) / msa.shape[0]
        deletion_mean = deletion_matrix.mean(axis=0)
        profiles.append(profile)
        deletion_means.append(deletion_mean)
        msa_rows.append(msa)
        deletion_rows.append(deletion_matrix)
    if not profiles:
        restype = np.zeros((len(sequence), 32), dtype=np.float32)
        for i, aa in enumerate(sequence):
            restype[i, RESTYPE_INDEX[aa]] = 1.0
        msa, deletion_matrix = parse_a3m_rows(sequence, "")
        return (
            restype,
            np.zeros((len(sequence),), dtype=np.float32),
            msa,
            deletion_matrix,
        )
    merged_msa, merged_deletion_matrix = _deduplicate_msa_rows(
        np.concatenate(msa_rows, axis=0),
        np.concatenate(deletion_rows, axis=0),
    )
    merged_msa = merged_msa[:max_msa_rows]
    merged_deletion_matrix = merged_deletion_matrix[:max_msa_rows]
    return (
        np.mean(np.stack(profiles), axis=0).astype(np.float32),
        np.mean(np.stack(deletion_means), axis=0).astype(np.float32),
        merged_msa,
        merged_deletion_matrix,
    )


def _chain_msa_features(chain: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return chain["profile"], chain["deletion_mean"]


def _deduplicate_msa_rows(
    msa: np.ndarray,
    deletion_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    keep = []
    seen = set()
    for i, row in enumerate(msa.astype(np.int8)):
        key = row.tobytes()
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return msa[keep], deletion_matrix[keep]


def _merge_chain_msa_features(
    chains: list[dict[str, Any]],
    *,
    max_msa_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_depth = min(max(chain["msa"].shape[0] for chain in chains), max_msa_rows)
    msa_parts = []
    deletion_parts = []
    for chain in chains:
        msa = chain["msa"][:max_depth]
        deletion_matrix = chain["deletion_matrix"][:max_depth]
        pad_depth = max_depth - msa.shape[0]
        if pad_depth:
            width = len(chain["sequence"])
            msa = np.pad(
                msa,
                ((0, pad_depth), (0, 0)),
                constant_values=MSA_PROTEIN_INDEX["-"],
            )
            deletion_matrix = np.pad(
                deletion_matrix,
                ((0, pad_depth), (0, 0)),
                constant_values=0,
            )
            if msa.shape[1] != width or deletion_matrix.shape[1] != width:
                raise ValueError("MSA row width must match chain sequence length")
        msa_parts.append(msa)
        deletion_parts.append(deletion_matrix)
    return (
        np.concatenate(msa_parts, axis=1).astype(np.int64),
        np.concatenate(deletion_parts, axis=1).astype(np.float32),
    )


def _resolve_path(path: str | Path, *, base_dir: str | Path | None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    if not resolved.exists():
        raise ValueError(f"MSA path does not exist: {resolved}")
    return resolved


def _parse_a3m_records(a3m: str) -> list[str]:
    records: list[str] = []
    current: list[str] = []
    for raw_line in a3m.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                records.append("".join(current))
                current = []
            continue
        current.append(line)
    if current:
        records.append("".join(current))
    return records


def _aligned_protein_row(sequence: str) -> tuple[list[int], list[int]]:
    row: list[int] = []
    deletion: list[int] = []
    pending_insertions = 0
    for char in sequence:
        if char.islower() or char == ".":
            pending_insertions += 1
            continue
        upper = char.upper()
        if upper not in MSA_PROTEIN_INDEX:
            raise ValueError(f"unsupported MSA residue: {char}")
        row.append(MSA_PROTEIN_INDEX[upper])
        deletion.append(pending_insertions)
        pending_insertions = 0
    return row, deletion


def _normalize_sequence(sequence: Any) -> str:
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("proteinChain sequence must be a non-empty string")
    sequence = sequence.upper()
    for aa in sequence:
        if aa not in RESTYPE_INDEX:
            raise ValueError(f"unsupported residue: {aa}")
    return sequence


def _residue_atom_names(aa: str) -> tuple[str, ...]:
    if aa == "G":
        return ("N", "CA", "C", "O")
    return ("N", "CA", "C", "O", "CB")


def _distogram_rep_atom_name(aa: str) -> str:
    if aa == "G":
        return "CA"
    return "CB"


def _dummy_atom_position(
    residue_index: int,
    atom_name: str,
    asym_id: int,
) -> tuple[float, float, float]:
    base_x = float((residue_index - 1) * 3.8)
    chain_y = float(asym_id * 8.0)
    offsets = {
        "N": (0.0, 0.0, 0.0),
        "CA": (1.45, 0.0, 0.0),
        "C": (2.9, 0.0, 0.0),
        "O": (3.5, 0.4, 0.0),
        "CB": (1.45, 1.5, 0.0),
    }
    dx, dy, dz = offsets[atom_name]
    return base_x + dx, chain_y + dy, dz


def _element_from_atom_name(atom_name: str) -> str:
    if atom_name.startswith("C"):
        return "C"
    return atom_name[0]


def _encode_elements(elements: list[str]) -> np.ndarray:
    encoded = np.zeros((len(elements), 128), dtype=np.float32)
    for i, element in enumerate(elements):
        encoded[i, ELEMENT_INDEX[element]] = 1.0
    return encoded


def _encode_atom_name_chars(atom_names: list[str]) -> np.ndarray:
    encoded = np.zeros((len(atom_names), 4, 64), dtype=np.float32)
    for i, name in enumerate(atom_names):
        for j, char in enumerate(name.ljust(4)[:4]):
            encoded[i, j, min(max(ord(char) - 32, 0), 63)] = 1.0
    return encoded


def _local_atom_geometry(
    ref_pos: np.ndarray,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n_atom = ref_pos.shape[0]
    n_trunks = int(math.ceil(n_atom / n_queries))
    q_pad = n_trunks * n_queries - n_atom
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 0.5) * n_queries + n_keys / 2 - n_atom + 0.5)

    q_padded = np.pad(ref_pos, ((0, q_pad), (0, 0)))
    k_padded = np.pad(ref_pos, ((pad_left, pad_right), (0, 0)))
    q = q_padded.reshape(n_trunks, n_queries, 3)
    k = np.stack(
        [k_padded[i * n_queries : i * n_queries + n_keys] for i in range(n_trunks)],
        axis=0,
    )
    q_abs = np.arange(n_trunks * n_queries).reshape(n_trunks, n_queries)
    k_abs = (
        np.arange(n_keys)[None, :]
        + np.arange(n_trunks)[:, None] * n_queries
        - pad_left
    )
    mask = (q_abs[..., None] < n_atom) & (k_abs[:, None, :] >= 0) & (
        k_abs[:, None, :] < n_atom
    )
    d_lm = q[:, :, None, :] - k[:, None, :, :]
    d_lm = d_lm.astype(np.float32)
    v_lm = mask[..., None].astype(np.float32)
    return d_lm, v_lm, {"mask_trunked": mask}


def _relative_position_features(
    *,
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    entity_id: np.ndarray,
    sym_id: np.ndarray,
    token_index: np.ndarray,
    r_max: int = 32,
    s_max: int = 2,
) -> np.ndarray:
    same_chain = asym_id[:, None] == asym_id[None, :]
    same_residue = residue_index[:, None] == residue_index[None, :]
    same_entity = entity_id[:, None] == entity_id[None, :]

    residue_delta = np.clip(
        residue_index[:, None] - residue_index[None, :] + r_max,
        0,
        2 * r_max,
    )
    residue_bins = np.where(same_chain, residue_delta, 2 * r_max + 1)
    token_delta = np.clip(
        token_index[:, None] - token_index[None, :] + r_max,
        0,
        2 * r_max,
    )
    token_bins = np.where(same_chain & same_residue, token_delta, 2 * r_max + 1)
    chain_delta = np.clip(sym_id[:, None] - sym_id[None, :] + s_max, 0, 2 * s_max)
    chain_bins = np.where(same_entity, chain_delta, 2 * s_max + 1)

    rel_pos = np.eye(2 * (r_max + 1), dtype=np.float32)[residue_bins]
    rel_token = np.eye(2 * (r_max + 1), dtype=np.float32)[token_bins]
    rel_chain = np.eye(2 * (s_max + 1), dtype=np.float32)[chain_bins]
    return np.concatenate(
        [rel_pos, rel_token, same_entity[..., None].astype(np.float32), rel_chain],
        axis=-1,
    )


if __name__ == "__main__":
    main()
