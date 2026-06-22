"""Native Protenix JAX weight serialization."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def save_native_weights(
    path: str | Path,
    params: Any,
    *,
    compress: bool = True,
) -> None:
    """Save a JAX parameter pytree without requiring torch at load time."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_tree = jax.tree.map(_leaf_to_numpy, params)
    opener = gzip.open if compress else open
    with opener(path, "wb") as fh:
        pickle.dump(numpy_tree, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_native_weights(path: str | Path) -> Any:
    """Load native JAX weights produced by ``save_native_weights``."""

    path = Path(path)
    opener = gzip.open if _is_gzip_file(path) else open
    with opener(path, "rb") as fh:
        numpy_tree = pickle.load(fh)
    return jax.tree.map(_leaf_to_jax, numpy_tree)


def _is_gzip_file(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def _leaf_to_numpy(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value)
    return value


def _leaf_to_jax(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    return value
