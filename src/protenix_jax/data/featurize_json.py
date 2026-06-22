"""Sequence-only Protenix JSON to static feature conversion."""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from protenix_jax.data.static_io import save_static_feature_npz
from protenix_jax.data.template_features import (
    assemble_template_features,
    chain_template_dense,
)

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
# Periodic table order used by torch ``get_all_elems`` (index == Z - 1).
_PERIODIC_TABLE = (
    "H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO "
    "NI CU ZN GA GE AS SE BR KR RB SR Y ZR NB MO TC RU RH PD AG CD IN SN SB "
    "TE I XE CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB LU HF TA W RE OS "
    "IR PT AU HG TL PB BI PO AT RN FR RA AC TH PA U NP PU AM CM BK CF ES FM "
    "MD NO LR RF DB SG BH HS MT DS RG CN NH FL MC LV TS OG"
).split()
ELEMENT_INDEX = {elem: i for i, elem in enumerate(_PERIODIC_TABLE)}
for _i in range(119, 129):
    ELEMENT_INDEX[f"UNK_ELEM_{_i}"] = _i - 1
# Nucleotide restype indices (torch STD_RESIDUES); RNA then DNA.
RNA_RESTYPE_INDEX = {"A": 21, "G": 22, "C": 23, "U": 24}
DNA_RESTYPE_INDEX = {"DA": 26, "DG": 27, "DC": 28, "DT": 29}
RNA_CODES = {"A": "A", "G": "G", "C": "C", "U": "U"}
DNA_CODES = {"A": "DA", "G": "DG", "C": "DC", "T": "DT"}
# Distogram representative atom: purine -> C4, pyrimidine -> C2.
_PURINE_CODES = {"DA", "DG", "A", "G"}
_PYRIMIDINE_CODES = {"DC", "DT", "C", "U"}
AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}
_CCD_TABLE_PATH = Path(__file__).with_name("ccd_std_residues.npz")
_CCD_TABLE: dict[str, dict[str, np.ndarray]] | None = None
_LIGAND_TABLE_PATH = Path(__file__).with_name("ccd_ligands.npz")
_LIGAND_TABLE: dict[str, dict[str, np.ndarray]] | None = None
_NUCLEIC_TABLE_PATH = Path(__file__).with_name("ccd_nucleotides.npz")
_NUCLEIC_TABLE: dict[str, dict[str, np.ndarray]] | None = None


def _ccd_ligands() -> dict[str, dict[str, np.ndarray]]:
    """Lazy-load the vendored ligand CCD reference table.

    Each CCD code maps to ``names``/``coord``/``charge``/``mask``/``elem``
    arrays in the order produced by torch ``get_component_atom_array``
    (keep_leaving_atoms=True, keep_hydrogens=False).
    """

    global _LIGAND_TABLE
    if _LIGAND_TABLE is None:
        raw = np.load(_LIGAND_TABLE_PATH, allow_pickle=False)
        codes = [str(c) for c in raw["_codes"]]
        table: dict[str, dict[str, np.ndarray]] = {}
        for code in codes:
            table[code] = {
                "names": raw[f"{code}/names"],
                "coord": raw[f"{code}/coord"].astype(np.float32),
                "charge": raw[f"{code}/charge"].astype(np.float32),
                "mask": raw[f"{code}/mask"].astype(np.float32),
                "elem": raw[f"{code}/elem"],
            }
        _LIGAND_TABLE = table
    return _LIGAND_TABLE


def _ccd_nucleotides() -> dict[str, dict[str, np.ndarray]]:
    """Lazy-load the vendored nucleotide CCD reference table.

    Each CCD code (DA/DC/DG/DT, A/C/G/U) maps to ``names``/``coord``/
    ``charge``/``mask``/``elem`` arrays in RES_ATOMS order (OP3 first). OP3
    is the 5'-terminal leaving atom: kept only for the first residue.
    """

    global _NUCLEIC_TABLE
    if _NUCLEIC_TABLE is None:
        raw = np.load(_NUCLEIC_TABLE_PATH, allow_pickle=False)
        codes = [str(c) for c in raw["_codes"]]
        table: dict[str, dict[str, np.ndarray]] = {}
        for code in codes:
            table[code] = {
                "names": raw[f"{code}/names"],
                "coord": raw[f"{code}/coord"].astype(np.float32),
                "charge": raw[f"{code}/charge"].astype(np.float32),
                "mask": raw[f"{code}/mask"].astype(np.float32),
                "elem": raw[f"{code}/elem"],
            }
        _NUCLEIC_TABLE = table
    return _NUCLEIC_TABLE


def _ccd_std_residues() -> dict[str, dict[str, np.ndarray]]:
    """Lazy-load the vendored CCD reference-conformer table (20 std residues).

    Each residue maps to ``names``/``coord``/``charge``/``elem`` arrays in the
    canonical RES_ATOMS order (N, CA, C, O, sidechain..., OXT last). OXT is
    kept only for the C-terminal residue of a chain.
    """

    global _CCD_TABLE
    if _CCD_TABLE is None:
        raw = np.load(_CCD_TABLE_PATH, allow_pickle=False)
        table: dict[str, dict[str, np.ndarray]] = {}
        for aa3 in AA1_TO_AA3.values():
            table[aa3] = {
                "names": raw[f"{aa3}/names"],
                "coord": raw[f"{aa3}/coord"].astype(np.float32),
                "charge": raw[f"{aa3}/charge"].astype(np.float32),
                "elem": raw[f"{aa3}/elem"],
            }
        _CCD_TABLE = table
    return _CCD_TABLE


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
    chains = _expand_chains(job, base_dir=base_dir)
    n_token = sum(_chain_token_count(chain) for chain in chains)
    if n_token <= 0:
        raise ValueError("at least one token is required")

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
    ref_charge: list[float] = []
    ref_element: list[str] = []
    ref_atom_names: list[str] = []
    distogram_rep_atom_mask: list[int] = []
    ref_mask_list: list[float] = []
    mol_id: list[int] = []

    state = {
        "token_i": 0,
        "restype": restype,
        "profile": profile,
        "deletion_mean": deletion_mean,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": atom_to_tokatom_idx,
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_element": ref_element,
        "ref_atom_names": ref_atom_names,
        "distogram_rep_atom_mask": distogram_rep_atom_mask,
        "ref_mask": ref_mask_list,
        "mol_id": mol_id,
    }
    for chain in chains:
        if chain["kind"] == "protein":
            _emit_protein_tokens(chain, state)
        elif chain["kind"] == "nucleic":
            _emit_nucleic_tokens(chain, state)
        else:
            _emit_ligand_tokens(chain, state)

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
    msa, deletion_matrix, assembled_profile, assembled_deletion_mean = (
        _assemble_msa_features(chains, max_msa_rows=max_msa_rows)
    )
    profile[:] = assembled_profile
    deletion_mean[:] = assembled_deletion_mean
    template_features = _assemble_chain_templates(chains)
    out = {
        "atom_to_token_idx": atom_to_token,
        "ref_pos": ref_pos_arr,
        "ref_charge": np.asarray(ref_charge, dtype=np.float32),
        "ref_mask": np.asarray(ref_mask_list, dtype=np.float32),
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
    if template_features is not None:
        out.update(template_features)
    return out


def _assemble_chain_templates(
    chains: list[dict[str, Any]],
) -> dict[str, np.ndarray] | None:
    """Build per-token template features in chain/token order (torch parity)."""

    chain_dense = []
    for chain in chains:
        num_res = _chain_token_count(chain)
        if chain["kind"] == "protein":
            sequence = chain["sequence"]
            chain_dense.append(
                chain_template_dense(
                    chain.get("templates_path"),
                    sequence=sequence,
                    skip=len(sequence) <= 4,
                )
            )
        else:
            chain_dense.append(
                chain_template_dense(None, sequence="X" * num_res, skip=True)
            )
    return assemble_template_features(chain_dense)


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

    records, _descs = _parse_a3m_records(a3m)
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


def _expand_chains(
    job: dict[str, Any],
    *,
    base_dir: str | Path | None,
) -> list[dict[str, Any]]:
    if job.get("covalent_bonds"):
        raise ValueError("covalent_bonds are not supported")
    sequences = job.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("job must contain a non-empty sequences list")
    chains: list[dict[str, Any]] = []
    entity_by_key: dict[str, int] = {}
    entity_sym_counts: dict[int, int] = {}
    for entry in sequences:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("each sequence entry must have one entity key")
        (kind,) = entry.keys()
        if kind == "proteinChain":
            built = _build_protein_chain(entry[kind], base_dir=base_dir)
        elif kind in ("dnaSequence", "rnaSequence"):
            built = _build_nucleic_chain(entry[kind], kind=kind)
        elif kind in ("ligand", "ion"):
            built = _build_ligand_chain(entry[kind], kind=kind)
        else:
            raise ValueError(f"unsupported entity kind: {kind}")
        entity = entity_by_key.setdefault(
            built["entity_key"], len(entity_by_key)
        )
        for _ in range(built["count"]):
            sym = entity_sym_counts.get(entity, 0)
            entity_sym_counts[entity] = sym + 1
            chains.append(
                {
                    **built["chain"],
                    "entity_id": entity,
                    "asym_id": len(chains),
                    "sym_id": sym,
                }
            )
    return chains


def _build_protein_chain(
    chain: dict[str, Any],
    *,
    base_dir: str | Path | None,
) -> dict[str, Any]:
    if not isinstance(chain, dict):
        raise ValueError("proteinChain entry must be an object")
    templates_path = chain.get("templatesPath") or None
    if templates_path:
        templates_path = str(
            _resolve_path(templates_path, base_dir=base_dir)
        )
    if chain.get("modifications"):
        raise ValueError("proteinChain modifications are not supported")
    sequence = _normalize_sequence(chain.get("sequence"))
    paired_a3m, unpaired_a3m = _read_chain_a3m(chain, base_dir=base_dir)
    count = int(chain.get("count", 1))
    if count <= 0:
        raise ValueError("proteinChain count must be positive")
    ids = chain.get("id")
    if ids is not None and len(ids) != count:
        raise ValueError("proteinChain id length must match count")
    return {
        "entity_key": f"protein:{sequence}",
        "count": count,
        "chain": {
            "kind": "protein",
            "sequence": sequence,
            "paired_a3m": paired_a3m,
            "unpaired_a3m": unpaired_a3m,
            "templates_path": templates_path,
        },
    }


def _build_ligand_chain(info: dict[str, Any], *, kind: str) -> dict[str, Any]:
    """Build a ligand/ion chain from CCD codes (one token per heavy atom)."""

    if not isinstance(info, dict):
        raise ValueError(f"{kind} entry must be an object")
    if kind == "ion":
        ligand_str = f"CCD_{info['ion']}"
    else:
        ligand_str = info["ligand"]
        if not isinstance(ligand_str, str) or not ligand_str:
            raise ValueError("ligand string must be non-empty")
    if not ligand_str.startswith("CCD_"):
        raise ValueError(
            "only CCD_ ligands are supported; SMILES/FILE require rdkit "
            "(not a JAX runtime dependency)"
        )
    codes = ligand_str[4:].split("_")
    table = _ccd_ligands()
    residues = []
    for res_id, code in enumerate(codes, start=1):
        if code not in table:
            raise ValueError(
                f"ligand CCD code {code!r} not in vendored ccd_ligands.npz; "
                "regenerate it with scripts/build_ccd_ligands_npz.py"
            )
        residues.append((res_id, table[code]))
    count = int(info.get("count", 1))
    if count <= 0:
        raise ValueError(f"{kind} count must be positive")
    n_tok = sum(len(entry["names"]) for _res_id, entry in residues)
    gap = MSA_PROTEIN_INDEX["-"]
    msa = np.full((1, n_tok), gap, dtype=np.int64)
    deletion_matrix = np.zeros((1, n_tok), dtype=np.float32)
    return {
        "entity_key": f"ligand:{ligand_str}",
        "count": count,
        "chain": {
            "kind": "ligand",
            "residues": residues,
            "msa": msa,
            "deletion_matrix": deletion_matrix,
        },
    }


def _build_nucleic_chain(
    info: dict[str, Any], *, kind: str
) -> dict[str, Any]:
    """Build a DNA/RNA chain (one per-residue token like protein)."""

    if not isinstance(info, dict):
        raise ValueError(f"{kind} entry must be an object")
    if info.get("modifications"):
        raise ValueError(f"{kind} modifications are not supported")
    sequence = info.get("sequence")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError(f"{kind} sequence must be a non-empty string")
    sequence = sequence.upper()
    code_map = DNA_CODES if kind == "dnaSequence" else RNA_CODES
    restype_map = DNA_RESTYPE_INDEX if kind == "dnaSequence" else RNA_RESTYPE_INDEX
    table = _ccd_nucleotides()
    codes = []
    for base in sequence:
        if base not in code_map:
            raise ValueError(f"unsupported {kind} base: {base}")
        codes.append(code_map[base])
    count = int(info.get("count", 1))
    if count <= 0:
        raise ValueError(f"{kind} count must be positive")
    label = "dna" if kind == "dnaSequence" else "rna"
    n_tok = len(sequence)
    gap = MSA_PROTEIN_INDEX["-"]
    msa = np.full((1, n_tok), gap, dtype=np.int64)
    deletion_matrix = np.zeros((1, n_tok), dtype=np.float32)
    return {
        "entity_key": f"{label}:{sequence}",
        "count": count,
        "chain": {
            "kind": "nucleic",
            "codes": codes,
            "restype_map": restype_map,
            "table": table,
            "msa": msa,
            "deletion_matrix": deletion_matrix,
        },
    }


def _chain_token_count(chain: dict[str, Any]) -> int:
    if chain["kind"] == "protein":
        return len(chain["sequence"])
    if chain["kind"] == "nucleic":
        return len(chain["codes"])
    return sum(len(entry["names"]) for _res_id, entry in chain["residues"])


def _emit_protein_tokens(
    chain: dict[str, Any], state: dict[str, Any]
) -> None:
    sequence = chain["sequence"]
    for pos, aa in enumerate(sequence, start=1):
        token_i = state["token_i"]
        state["restype"][token_i, RESTYPE_INDEX[aa]] = 1.0
        state["residue_index"][token_i] = pos
        state["asym_id"][token_i] = chain["asym_id"]
        state["entity_id"][token_i] = chain["entity_id"]
        state["sym_id"][token_i] = chain["sym_id"]
        entry = _ccd_std_residues()[AA1_TO_AA3[aa]]
        names = entry["names"]
        is_cterm = pos == len(sequence)
        rep = _distogram_rep_atom_name(aa)
        for j, atom_name in enumerate(names):
            atom_name = str(atom_name)
            if atom_name == "OXT" and not is_cterm:
                continue
            state["atom_to_token_idx"].append(token_i)
            state["atom_to_tokatom_idx"].append(j)
            xyz = entry["coord"][j]
            state["ref_pos"].append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            state["ref_charge"].append(float(entry["charge"][j]))
            state["ref_element"].append(str(entry["elem"][j]))
            state["ref_atom_names"].append(atom_name)
            state["distogram_rep_atom_mask"].append(int(atom_name == rep))
            state["ref_mask"].append(1.0)
            state["mol_id"].append(chain["asym_id"])
        state["token_i"] = token_i + 1


def _emit_ligand_tokens(
    chain: dict[str, Any], state: dict[str, Any]
) -> None:
    """One token per ligand heavy atom; restype=UNK(20), tokatom_idx=0."""

    for res_id, entry in chain["residues"]:
        names = entry["names"]
        for j in range(len(names)):
            token_i = state["token_i"]
            state["restype"][token_i, RESTYPE_INDEX["X"]] = 1.0
            state["residue_index"][token_i] = res_id
            state["asym_id"][token_i] = chain["asym_id"]
            state["entity_id"][token_i] = chain["entity_id"]
            state["sym_id"][token_i] = chain["sym_id"]
            xyz = entry["coord"][j]
            state["atom_to_token_idx"].append(token_i)
            state["atom_to_tokatom_idx"].append(0)
            state["ref_pos"].append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            state["ref_charge"].append(float(entry["charge"][j]))
            state["ref_element"].append(str(entry["elem"][j]))
            state["ref_atom_names"].append(str(names[j]))
            state["distogram_rep_atom_mask"].append(1)
            state["ref_mask"].append(float(entry["mask"][j]))
            state["mol_id"].append(chain["asym_id"])
            state["token_i"] = token_i + 1


def _emit_nucleic_tokens(
    chain: dict[str, Any], state: dict[str, Any]
) -> None:
    """One token per nucleotide; OP3 kept only on the 5'-terminal residue."""

    table = chain["table"]
    restype_map = chain["restype_map"]
    codes = chain["codes"]
    for pos, code in enumerate(codes, start=1):
        token_i = state["token_i"]
        state["restype"][token_i, restype_map[code]] = 1.0
        state["residue_index"][token_i] = pos
        state["asym_id"][token_i] = chain["asym_id"]
        state["entity_id"][token_i] = chain["entity_id"]
        state["sym_id"][token_i] = chain["sym_id"]
        entry = table[code]
        names = entry["names"]
        is_first = pos == 1
        rep = "C4" if code in _PURINE_CODES else "C2"
        for j, atom_name in enumerate(names):
            atom_name = str(atom_name)
            if atom_name == "OP3" and not is_first:
                continue
            state["atom_to_token_idx"].append(token_i)
            state["atom_to_tokatom_idx"].append(j)
            xyz = entry["coord"][j]
            state["ref_pos"].append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            state["ref_charge"].append(float(entry["charge"][j]))
            state["ref_element"].append(str(entry["elem"][j]))
            state["ref_atom_names"].append(atom_name)
            state["distogram_rep_atom_mask"].append(int(atom_name == rep))
            state["ref_mask"].append(float(entry["mask"][j]))
            state["mol_id"].append(chain["asym_id"])
        state["token_i"] = token_i + 1


_GAP_IDX = MSA_PROTEIN_INDEX["-"]
_MSA_PAD_VALUES = {"msa": _GAP_IDX, "deletion_matrix": 0}
_UNIPROT_REGEX = re.compile(
    r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|"
    r"(?:[A-Z0-9]{1,10}_)(?P<SpeciesId>[A-Z0-9]{1,5})"
)
_UNIREF_REGEX = re.compile(r"^UniRef100_[^_]+_([^_/]+)")


def _read_chain_a3m(
    chain: dict[str, Any],
    *,
    base_dir: str | Path | None,
) -> tuple[str, str]:
    """Read paired/unpaired A3M for a protein chain (inline or precomputed)."""

    def _resolve(inline_key: str, path_key: str) -> str:
        inline = chain.get(inline_key)
        if inline:
            return inline
        path = chain.get(path_key)
        if path:
            return _resolve_path(path, base_dir=base_dir).read_text(encoding="utf-8")
        return ""

    paired = _resolve("pairedMsa", "pairedMsaPath")
    unpaired = _resolve("unpairedMsa", "unpairedMsaPath")
    return paired, unpaired


def _species_id(description: str) -> str:
    """Extract a species identifier from a UniProt/UniRef description line."""

    desc = description.strip()
    m = _UNIPROT_REGEX.match(desc) or _UNIREF_REGEX.match(desc)
    if not m:
        return ""
    return m.group("SpeciesId") if "SpeciesId" in m.groupdict() else m.group(1)


def _featurize_a3m(
    query: str,
    a3m: str,
    *,
    dedup: bool,
) -> dict[str, np.ndarray]:
    """Port of torch ``RawMsa(...).featurize`` for protein chains."""

    seqs, descs = _parse_a3m_records(a3m)
    if dedup:
        seqs, descs = _dedup_sequences(seqs, descs)
    if not seqs:
        seqs, descs = [query], ["Original query"]
    cols = len(query)
    msa = np.zeros((len(seqs), cols), dtype=np.int64)
    deletion = np.zeros((len(seqs), cols), dtype=np.int64)
    for i, sequence in enumerate(seqs):
        row, dels = _aligned_protein_row(sequence)
        n = min(len(row), cols)
        msa[i, :n] = row[:n]
        deletion[i, :n] = dels[:n]
    return {
        "msa": msa,
        "deletion_matrix": deletion,
        "species": np.array([_species_id(d) for d in descs], dtype=object),
    }


def _dedup_sequences(
    seqs: list[str], descs: list[str]
) -> tuple[list[str], list[str]]:
    """Remove duplicate sequences ignoring insertion (lowercase) columns."""

    table = str.maketrans("", "", string.ascii_lowercase)
    u_seqs: list[str] = []
    u_descs: list[str] = []
    seen: set[str] = set()
    for s, d in zip(seqs, descs):
        stripped = s.translate(table)
        if stripped not in seen:
            seen.add(stripped)
            u_seqs.append(s)
            u_descs.append(d)
    return u_seqs, u_descs


def _gap_only_chain_features(width: int) -> dict[str, np.ndarray]:
    """Single gap row used for ligand/nucleic chains (torch placeholder)."""

    return {
        "msa": np.full((1, width), _GAP_IDX, dtype=np.int64),
        "deletion_matrix": np.zeros((1, width), dtype=np.int64),
        "species": np.array([""], dtype=object),
    }


def _align_species(
    all_species: list[str],
    chain_species_map: list[dict[str, np.ndarray]],
    species_min_hits: dict[str, int],
) -> np.ndarray:
    blocks = []
    for species in all_species:
        rows = []
        for s2r in chain_species_map:
            n = species_min_hits[species]
            if species not in s2r:
                rows.append(np.full(n, -1, dtype=np.int32))
            else:
                rows.append(s2r[species][:n])
        blocks.append(np.stack(rows, axis=1))
    return np.concatenate(blocks, axis=0)


def _pair_chains_by_species(
    chains: list[dict[str, np.ndarray]],
    max_paired: int,
    active: set[int],
    max_per_species: int,
) -> list[dict[str, np.ndarray]]:
    """Port of torch ``MSAPairingEngine.pair_chains_by_species``."""

    chain_species_map: list[dict[str, np.ndarray]] = []
    all_counts: dict[str, int] = {}
    min_hits: dict[str, int] = {}
    for c in chains:
        ids = c.get("species_all_seq", np.array([], dtype=object))
        no_species = ids.size == 0 or (ids.size == 1 and not ids[0])
        if no_species or c["asym_id"] not in active:
            chain_species_map.append({})
            continue
        row_idx = np.arange(len(ids))
        order = ids.argsort(kind="stable")
        ids_s = ids[order]
        row_idx = row_idx[order]
        species, uniq = np.unique(ids_s, return_index=True)
        grouped = np.split(row_idx, uniq[1:])
        mapping = dict(zip(species, grouped))
        chain_species_map.append(mapping)
        for s in species:
            all_counts[s] = all_counts.get(s, 0) + 1
        for s, idxs in mapping.items():
            min_hits[s] = min(min_hits.get(s, max_per_species), len(idxs))

    ranked: dict[int, list[str]] = {}
    for s, count in all_counts.items():
        if not s or count <= 1:
            continue
        ranked.setdefault(count, []).append(s)

    pair_idxs = [np.zeros((1, len(chains)), dtype=np.int32)]
    total = 0
    for count in sorted(ranked.keys(), reverse=True):
        rows = _align_species(ranked[count], chain_species_map, min_hits)
        rank = np.sum(np.log(np.abs(rows.astype(np.float32)) + 1e-10), axis=1)
        pair_idxs.append(rows[np.argsort(rank)])
        total += rows.shape[0]
        if total >= max_paired:
            break
    final_idxs = np.concatenate(pair_idxs, axis=0)[:max_paired]

    new_chains = []
    for i, c in enumerate(chains):
        nc = {k: v for k, v in c.items() if "all_seq" not in k}
        sel = final_idxs[:, i]
        for f in ["msa", "deletion_matrix"]:
            src = c[f"{f}_all_seq"]
            pad = np.full((1, src.shape[1]), _MSA_PAD_VALUES[f], src.dtype)
            padded = np.concatenate([src, pad], axis=0)
            nc[f"{f}_all_seq"] = padded[sel]
        new_chains.append(nc)
    return new_chains


def _cleanup_unpaired(
    chains: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Drop unpaired rows already present in the paired MSA (torch port)."""

    for c in chains:
        paired_bytes = {row.tobytes() for row in c["msa_all_seq"].astype(np.int8)}
        keep = [
            i
            for i, row in enumerate(c["msa"].astype(np.int8))
            if row.tobytes() not in paired_bytes
        ]
        c["msa"] = c["msa"][keep]
        c["deletion_matrix"] = c["deletion_matrix"][keep]
    return chains


def _filter_all_gapped_rows(
    chains: list[dict[str, np.ndarray]],
    active: set[int],
) -> list[dict[str, np.ndarray]]:
    """Remove all-gap rows from the paired MSA across active chains."""

    subset = [c["msa_all_seq"] for c in chains if c["asym_id"] in active]
    if not subset:
        return chains
    non_gap = np.any(np.concatenate(subset, axis=1) != _GAP_IDX, axis=1)
    for c in chains:
        c["msa_all_seq"] = c["msa_all_seq"][non_gap]
        c["deletion_matrix_all_seq"] = c["deletion_matrix_all_seq"][non_gap]
    return chains


def _merge_feature(chains: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    if "_all_seq" in key:
        return np.concatenate([c[key] for c in chains], axis=1)
    max_d = max(c[key].shape[0] for c in chains)
    base = key
    pads = [
        np.pad(
            c[key],
            ((0, max_d - c[key].shape[0]), (0, 0)),
            constant_values=_MSA_PAD_VALUES.get(base, 0),
        )
        for c in chains
    ]
    return np.concatenate(pads, axis=1)


def _assemble_msa_features(
    chains: list[dict[str, Any]],
    *,
    max_msa_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of torch ``FeatureAssemblyLine.assemble`` for inference inputs."""

    unique_prot_seqs = {
        c["sequence"] for c in chains if c["kind"] == "protein"
    }
    need_pairing = len(unique_prot_seqs) > 1
    active = {c["asym_id"] for c in chains}

    raw_chains: list[dict[str, np.ndarray]] = []
    for c in chains:
        width = _chain_token_count(c)
        if c["kind"] == "protein":
            skip = len(c["sequence"]) <= 4
            up = _featurize_a3m(
                c["sequence"],
                "" if skip else c["unpaired_a3m"],
                dedup=True,
            )
            p = _featurize_a3m(
                c["sequence"],
                c["paired_a3m"] if (need_pairing and not skip) else "",
                dedup=False,
            )
        else:
            up = _gap_only_chain_features(width)
            p = _gap_only_chain_features(width)
        feat: dict[str, np.ndarray] = dict(up)
        feat.update({f"{k}_all_seq": v for k, v in p.items()})
        feat["asym_id"] = c["asym_id"]
        msa = feat["msa"]
        prof = (msa[..., None] == np.arange(32)).sum(axis=0) / msa.shape[0]
        feat["profile"] = prof.astype(np.float32)
        feat["deletion_mean"] = np.mean(feat["deletion_matrix"], axis=0)
        raw_chains.append(feat)

    max_p = max_msa_rows // 2
    if need_pairing:
        raw_chains = _pair_chains_by_species(raw_chains, max_p, active, 600)
        raw_chains = _cleanup_unpaired(raw_chains)
    if "msa_all_seq" in raw_chains[0]:
        raw_chains = _filter_all_gapped_rows(raw_chains, active)

    cropped = []
    for c in raw_chains:
        p_msa = c.get("msa_all_seq")
        ps = min(p_msa.shape[0], max_p) if p_msa is not None else 0
        us = max(0, min(c["msa"].shape[0], max_msa_rows - ps))
        cr: dict[str, np.ndarray] = {
            "asym_id": c["asym_id"],
            "profile": c["profile"],
            "deletion_mean": c["deletion_mean"],
        }
        for k in ("msa", "deletion_matrix"):
            cr[k] = c[k][:us]
            if f"{k}_all_seq" in c:
                cr[f"{k}_all_seq"] = c[f"{k}_all_seq"][:ps]
        cropped.append(cr)

    merged: dict[str, np.ndarray] = {}
    for base in ("msa", "deletion_matrix"):
        for f in (base, f"{base}_all_seq"):
            if f in cropped[0]:
                merged[f] = _merge_feature(cropped, f)
    profile = np.concatenate([c["profile"] for c in cropped], axis=0)
    deletion_mean = np.concatenate([c["deletion_mean"] for c in cropped], axis=0)

    max_u = max(c["msa"].shape[0] for c in cropped if c["asym_id"] in active)
    merged["msa"] = merged["msa"][:max_u]
    merged["deletion_matrix"] = merged["deletion_matrix"][:max_u]
    if "msa_all_seq" in merged:
        max_pa = max(
            c["msa_all_seq"].shape[0] for c in cropped if c["asym_id"] in active
        )
        merged["msa_all_seq"] = merged["msa_all_seq"][:max_pa]
        merged["deletion_matrix_all_seq"] = merged["deletion_matrix_all_seq"][:max_pa]
        for base in ("msa", "deletion_matrix"):
            merged[base] = np.concatenate(
                [merged[f"{base}_all_seq"], merged[base]], axis=0
            )

    msa = merged["msa"].astype(np.int64)
    deletion_matrix = merged["deletion_matrix"].astype(np.float32)
    return (
        msa,
        deletion_matrix,
        profile.astype(np.float32),
        deletion_mean.astype(np.float32),
    )


def _resolve_path(path: str | Path, *, base_dir: str | Path | None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    if not resolved.exists():
        raise ValueError(f"MSA path does not exist: {resolved}")
    return resolved


def _parse_a3m_records(a3m: str) -> tuple[list[str], list[str]]:
    """Parse FASTA/A3M into (sequences, descriptions); torch ``parse_fasta``."""

    sequences: list[str] = []
    descriptions: list[str] = []
    index = -1
    for raw_line in a3m.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])
            sequences.append("")
        elif index >= 0:
            sequences[index] += line
    return sequences, descriptions


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


def _distogram_rep_atom_name(aa: str) -> str:
    if aa == "G":
        return "CA"
    return "CB"


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
