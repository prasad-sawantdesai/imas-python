# This file is part of IMAS-Python.
# You should have received the IMAS-Python LICENSE file with this project.
"""Write an IDS to a Parquet file (IDS → PyArrow Table → Parquet).

Design
------
The IDS is serialised as a **single-row** :class:`pyarrow.Table` whose schema
is generated upfront from the Data Dictionary via :mod:`schema`.  Each top-
level column corresponds to one direct child of the IDS root.

* Missing primitives   → ``None`` (Arrow null)
* Empty STRUCT_ARRAY   → ``[]``   (empty Arrow list, not null)
* Empty STRUCTURE      → ``None`` (null Arrow struct)
* Complex numbers      → ``[real, imag]``     (FixedSizeList[float64; 2])
* Multi-dim arrays     → nested Python lists  (Arrow variable-length lists)
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from imas.backends.parquet.schema import ids_schema
from imas.ids_data_type import IDSDataType
from imas.ids_metadata import IDSMetadata
from imas.ids_struct_array import IDSStructArray
from imas.ids_structure import IDSStructure
from imas.ids_toplevel import IDSToplevel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ids_to_parquet(
    ids: IDSToplevel,
    path: str,
    dd_version: str,
    occurrence: int,
    *,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    """Serialise *ids* to a Parquet file at *path*.

    Args:
        ids: The IDS toplevel to write.
        path: Destination file path (will be created or overwritten).
        dd_version: Data Dictionary version string; stored in the Parquet
            schema metadata so it can be verified on read.
        occurrence: IDS occurrence number; stored in schema metadata.
        compression: Parquet compression codec (default ``"zstd"``).
        compression_level: Compression level (default ``3``).
    """
    schema = ids_schema(ids.metadata, dd_version=dd_version, occurrence=occurrence)
    table = _ids_to_table(ids, schema)
    pq.write_table(
        table,
        path,
        compression=compression,
        compression_level=compression_level,
    )


# ---------------------------------------------------------------------------
# IDS → PyArrow Table
# ---------------------------------------------------------------------------


def _ids_to_table(ids: IDSToplevel, schema: pa.Schema) -> pa.Table:
    """Build a one-row :class:`pa.Table` from *ids* using *schema*."""
    arrays: list[pa.Array] = []
    for i in range(len(schema)):
        field = schema.field(i)
        child_meta = ids.metadata._children[field.name]
        child_node = ids[field.name]
        python_value = _node_to_python(child_node, child_meta)
        # Wrap in list because pa.array expects an iterable of rows
        arrays.append(pa.array([python_value], type=field.type))

    return pa.table(
        {schema.field(i).name: arrays[i] for i in range(len(schema))},
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Recursive node → Python value
# ---------------------------------------------------------------------------


def _node_to_python(node, meta: IDSMetadata):
    """Recursively convert an IDS node to a Python value for Arrow.

    The returned value is always compatible with the Arrow type produced by
    :func:`schema._metadata_to_type` for the same *meta*.

    Returns:
        * ``None``  – field is unset (Arrow null).
        * ``[]``    – STRUCT_ARRAY is empty (Arrow empty list).
        * ``dict``  – STRUCTURE is set.
        * ``list``  – STRUCT_ARRAY elements or numeric/string arrays.
        * scalar    – 0D primitive value.
    """
    dt = meta.data_type

    if dt is IDSDataType.STRUCT_ARRAY:
        return _struct_array_to_python(node, meta)

    if dt is IDSDataType.STRUCTURE:
        if not node.has_value:
            return None
        return _structure_to_dict(node, meta)

    # IDSPrimitive
    if not node.has_value:
        return None
    return _primitive_to_python(node.value, meta.data_type, meta.ndim)


def _struct_array_to_python(node: IDSStructArray, meta: IDSMetadata) -> list:
    """Convert an IDSStructArray to a Python list of dicts."""
    if not node.has_value:
        return []
    return [_structure_to_dict(item, meta) for item in node]


def _structure_to_dict(node: IDSStructure, meta: IDSMetadata) -> dict:
    """Convert an IDSStructure to a Python dict of child values.

    Every child from the DD is included (absent ones become ``None``).
    This keeps the dict shape consistent with the Arrow struct schema.
    """
    return {
        name: _node_to_python(node[name], child_meta)
        for name, child_meta in meta._children.items()
    }


# ---------------------------------------------------------------------------
# Primitive conversion helpers
# ---------------------------------------------------------------------------


def _primitive_to_python(value, dt: IDSDataType, ndim: int):
    """Convert a raw IDS primitive value to a Python object for Arrow.

    Scalars (ndim=0) become plain Python scalars.
    Arrays are converted to nested Python lists (Arrow serialises these as
    variable-length lists, or FixedSizeList for complex scalars).
    """
    if dt is IDSDataType.CPX:
        return _complex_to_python(value, ndim)

    if dt in (IDSDataType.FLT, IDSDataType.INT):
        if ndim == 0:
            # Unwrap numpy scalar to native Python type for Arrow
            return value.item() if hasattr(value, "item") else float(value)
        return np.asarray(value).tolist()

    # IDSDataType.STR
    if ndim == 0:
        return str(value)
    return list(value)


def _complex_to_python(value, ndim: int):
    """Convert a complex IDS value to Arrow-compatible representation.

    A single complex number becomes ``[real, imag]`` (FixedSizeList[float64]).
    Higher-dimensional arrays become nested lists with the same leaf format.
    """
    if ndim == 0:
        v = complex(value)
        return [float(v.real), float(v.imag)]
    return _complex_ndarray_to_nested(np.asarray(value, dtype=np.complex128), ndim)


def _complex_ndarray_to_nested(arr: np.ndarray, remaining_dims: int) -> list:
    """Recursively convert a complex ndarray to nested ``[real, imag]`` lists."""
    if remaining_dims == 1:
        return [[float(v.real), float(v.imag)] for v in arr]
    return [_complex_ndarray_to_nested(row, remaining_dims - 1) for row in arr]
