# This file is part of IMAS-Python.
# You should have received the IMAS-Python LICENSE file with this project.
"""Read a Parquet file and populate an IDS (Parquet → PyArrow Table → IDS).

Design
------
The Parquet file is read as a :class:`pyarrow.Table` with a single row.
:py:meth:`pa.Table.to_pylist` converts the entire row tree into nested Python
dicts and lists which are then walked recursively to populate the IDS.

NBC (non-backward-compatible) path maps are applied at the leaf level: when
the on-disk path is remapped to a different path in the target DD, the value
is written to the new path.  Paths that have no equivalent in the target (new
path is ``None``) are silently dropped.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

from imas.ids_convert import NBCPathMap
from imas.ids_data_type import IDSDataType
from imas.ids_metadata import IDSMetadata
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure
from imas.ids_toplevel import IDSToplevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parquet_to_ids(
    path: str,
    ids: IDSToplevel,
    ids_metadata: IDSMetadata,
    nbc_map: Optional[NBCPathMap],
) -> None:
    """Read *path* and populate *ids*.

    Args:
        path: Parquet file path to read.
        ids: IDS toplevel to populate.
        ids_metadata: Metadata describing the DD version *as stored on disk*.
            This may differ from the in-memory DD when NBC conversion is needed.
        nbc_map: Path map for implicit DD version conversion.  Pass ``None``
            when no conversion is needed.
    """
    table = pq.read_table(path)
    # The table has exactly one row (one occurrence per file).
    row_dict: dict = table.to_pylist()[0]
    _dict_to_structure(row_dict, ids, ids_metadata, nbc_map)


# ---------------------------------------------------------------------------
# Recursive population helpers
# ---------------------------------------------------------------------------


def _dict_to_structure(
    data: dict,
    node: IDSStructure,
    meta: IDSMetadata,
    nbc_map: Optional[NBCPathMap],
) -> None:
    """Populate *node* from *data* (a Python dict matching *meta*'s children).

    *meta* describes the **on-disk** DD version.  *node* belongs to the
    **in-memory** DD version (which may differ when NBC conversion is active).
    """
    for name, child_meta in meta._children.items():
        value = data.get(name)

        # ----------------------------------------------------------------
        # NBC mapping – resolve the on-disk path to a target path
        # ----------------------------------------------------------------
        target_meta = child_meta
        if nbc_map is not None:
            ps = child_meta.path_string
            if ps in nbc_map:
                new_path = nbc_map.path.get(ps)
                if new_path is None:
                    logger.info(
                        "Skipping %s: no equivalent in target Data Dictionary.", ps
                    )
                    continue
                try:
                    # Walk from the IDS root to the new path
                    target_meta = node.metadata[new_path]
                except KeyError:
                    logger.info(
                        "Skipping %s: target path %s not found.", ps, new_path
                    )
                    continue
            elif ps in nbc_map.type_change:
                logger.info(
                    "Skipping %s: type-change path, cannot convert automatically.", ps
                )
                continue

        dt = child_meta.data_type

        # Null primitive or null struct → leave the node unset
        if value is None and dt not in (
            IDSDataType.STRUCTURE,
            IDSDataType.STRUCT_ARRAY,
        ):
            continue

        child_node = node[target_meta.name]
        _python_to_node(value, child_node, child_meta, target_meta, nbc_map)


def _python_to_node(
    value,
    node,
    src_meta: IDSMetadata,
    tgt_meta: IDSMetadata,
    nbc_map: Optional[NBCPathMap],
) -> None:
    """Populate *node* from a Python *value*.

    Args:
        value: Python value extracted from the Parquet data.
        node: IDS node to write into (IDSPrimitive, IDSStructure, IDSStructArray).
        src_meta: Metadata of the *stored* DD version (governs iteration).
        tgt_meta: Metadata of the *target* in-memory DD (governs the set path).
        nbc_map: NBC path map (may be ``None``).
    """
    dt = src_meta.data_type

    if dt is IDSDataType.STRUCT_ARRAY:
        if not value:
            return  # empty list → leave node empty
        node.resize(len(value))
        for i, item_dict in enumerate(value):
            if item_dict is not None:
                # src_meta for a STRUCT_ARRAY holds the element-level children
                _dict_to_structure(item_dict, node[i], src_meta, nbc_map)

    elif dt is IDSDataType.STRUCTURE:
        if value is None:
            return
        _dict_to_structure(value, node, src_meta, nbc_map)

    else:
        # IDSPrimitive
        if value is None:
            return
        node.value = _python_to_primitive(value, tgt_meta)


# ---------------------------------------------------------------------------
# Primitive conversion helpers
# ---------------------------------------------------------------------------


def _python_to_primitive(value, meta: IDSMetadata):
    """Convert a Python/Arrow value back to a native IDS primitive.

    Args:
        value: Python object as returned by ``pa.Table.to_pylist()``.
        meta: Metadata of the *target* IDS node (used for its data_type/ndim).

    Returns:
        A scalar, numpy array, or list suitable for assigning to
        ``IDSPrimitive.value``.
    """
    dt = meta.data_type
    ndim = meta.ndim

    if dt is IDSDataType.CPX:
        return _python_to_complex(value, ndim)

    if dt is IDSDataType.FLT:
        if ndim == 0:
            return float(value)
        return np.array(value, dtype=np.float64)

    if dt is IDSDataType.INT:
        if ndim == 0:
            return int(value)
        return np.array(value, dtype=np.int32)

    # IDSDataType.STR
    if ndim == 0:
        return str(value)
    return list(value)


def _python_to_complex(value, ndim: int):
    """Convert ``[real, imag]``-formatted Python lists back to complex.

    * ndim 0  → a single :class:`complex` scalar.
    * ndim N  → a :class:`numpy.ndarray` of dtype ``complex128``.
    """
    if ndim == 0:
        return complex(float(value[0]), float(value[1]))
    return _nested_to_complex_array(value, ndim)


def _nested_to_complex_array(value, ndim: int) -> np.ndarray:
    """Recursively rebuild a complex ndarray from nested ``[real, imag]`` lists."""
    if ndim == 1:
        return np.array(
            [complex(float(v[0]), float(v[1])) for v in value],
            dtype=np.complex128,
        )
    sub = [_nested_to_complex_array(v, ndim - 1) for v in value]
    return np.array(sub, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Utility: collect filled paths from a row dict
# ---------------------------------------------------------------------------


def collect_filled_paths(row_dict: dict, prefix: str = "") -> list[str]:
    """Recursively find all non-null leaf paths in *row_dict*.

    Returns a list of slash-separated IDS paths
    (e.g. ``"ids_properties/comment"``, ``"profiles_1d/time"``).

    Only primitive (leaf) paths are returned – STRUCTURE containers and
    STRUCT_ARRAY paths are omitted, matching the convention used by the
    netCDF backend.

    Args:
        row_dict: Nested Python dict as produced by ``pa.Table.to_pylist()[0]``.
        prefix: Internal recursion prefix; leave empty at the call site.
    """
    result: list[str] = []
    _collect(row_dict, prefix, result)
    return result


def _collect(data, prefix: str, result: list[str]) -> None:
    """Recursive worker for :func:`collect_filled_paths`."""
    if data is None:
        return

    if isinstance(data, dict):
        for name, value in data.items():
            child_path = f"{prefix}/{name}" if prefix else name
            _collect(value, child_path, result)

    elif isinstance(data, list):
        # Variable-length list → could be STRUCT_ARRAY or primitive array
        if not data:
            return
        first = data[0]
        if isinstance(first, dict):
            # STRUCT_ARRAY: recurse into each element's fields
            seen: set[str] = set()
            for item in data:
                if item is None:
                    continue
                for name, value in item.items():
                    child_path = f"{prefix}/{name}" if prefix else name
                    if child_path not in seen:
                        seen.add(child_path)
                        _collect(value, child_path, result)
        else:
            # Leaf numeric / string array (or FixedSizeList for complex)
            result.append(prefix)

    else:
        # Scalar leaf (int, float, str, or complex represented as list[2])
        result.append(prefix)
