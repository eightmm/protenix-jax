"""Static feature and inference output IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_static_feature_npz(path: str | Path) -> dict[str, Any]:
    """Load a static Protenix feature dictionary from an ``.npz`` file."""

    features: dict[str, Any] = {}
    pad_info: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            if key.startswith("pad_info."):
                pad_info[key.removeprefix("pad_info.")] = value
            else:
                features[key] = value
    if pad_info:
        features["pad_info"] = pad_info
    return features


def save_static_feature_npz(
    path: str | Path,
    features: dict[str, Any],
) -> None:
    """Save a static Protenix feature dictionary as an ``.npz`` file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    for key, value in features.items():
        if key == "pad_info" and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"pad_info.{sub_key}"] = np.asarray(sub_value)
        else:
            flat[key] = np.asarray(value)
    np.savez_compressed(path, **flat)


def flatten_output_dict(
    output: dict[str, Any],
    *,
    include_trunk: bool = False,
) -> dict[str, np.ndarray]:
    """Convert JAX/NumPy output values into a flat NumPy mapping."""

    flat = {}
    skip = set() if include_trunk else {"s_inputs", "s_trunk", "z_trunk"}
    for key, value in output.items():
        if key in skip:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = np.asarray(sub_value)
        else:
            flat[key] = np.asarray(value)
    return flat


def save_output_npz(
    path: str | Path,
    output: dict[str, Any],
    *,
    include_trunk: bool = False,
) -> None:
    """Save inference output arrays to compressed ``.npz``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_output = flatten_output_dict(output, include_trunk=include_trunk)
    np.savez_compressed(path, **flat_output)
