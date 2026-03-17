# This file is part of IMAS-Python.
# You should have received the IMAS-Python LICENSE file with this project.
"""Build PyArrow schemas from IDS metadata.

The schema is derived entirely from the Data Dictionary, encoding every field
(even unset ones) so that the Arrow type is stable and type-checked on write.

Type mapping
------------
+----------------------------+-------------------------------------+
| IDS type                   | Arrow type                          |
+============================+=====================================+
| STRUCTURE                  | pa.struct([child_fields...])        |
+----------------------------+-------------------------------------+
| STRUCT_ARRAY               | pa.list_(pa.struct([child_fields])) |
+----------------------------+-------------------------------------+
| FLT_0D                     | pa.float64() (nullable)             |
+----------------------------+-------------------------------------+
| FLT_ND (N >= 1)            | pa.list_(... pa.float64())          |
+----------------------------+-------------------------------------+
| INT_0D                     | pa.int32()  (nullable)              |
+----------------------------+-------------------------------------+
| INT_ND (N >= 1)            | pa.list_(... pa.int32())            |
+----------------------------+-------------------------------------+
| STR_0D                     | pa.string() (nullable)              |
+----------------------------+-------------------------------------+
| STR_1D                     | pa.list_(pa.string())               |
+----------------------------+-------------------------------------+
| CPX_0D                     | pa.list_(pa.float64(), 2)           |
|                            |  (FixedSizeList[float64; 2])        |
+----------------------------+-------------------------------------+
| CPX_ND (N >= 1)            | pa.list_(... CPX_0D)                |
+----------------------------+-------------------------------------+

All fields are nullable so that absent data can be represented as null
without padding or fill values.
"""

import pyarrow as pa

from imas.ids_data_type import IDSDataType
from imas.ids_metadata import IDSMetadata

# Arrow type representing a single complex number as [real, imag]
COMPLEX_ARROW_TYPE: pa.DataType = pa.list_(pa.float64(), 2)
"""Fixed-size list of 2 float64 values: [real, imag]."""


def ids_schema(
    ids_metadata: IDSMetadata,
    dd_version: str = "",
    occurrence: int = 0,
) -> pa.Schema:
    """Build a top-level PyArrow schema for an IDS.

    Each direct child of the IDS becomes a top-level column.  Structural
    nesting (STRUCTURE / STRUCT_ARRAY) is encoded using Arrow struct and list
    types respectively.

    Args:
        ids_metadata: Toplevel IDS metadata (must satisfy
            ``ids_metadata._parent is None``).
        dd_version: Data Dictionary version string stored in schema metadata.
        occurrence: IDS occurrence number stored in schema metadata.

    Returns:
        A :class:`pyarrow.Schema` with one field per IDS child.
    """
    if ids_metadata._parent is not None:
        raise ValueError("ids_schema() requires top-level IDS metadata.")

    fields = [
        _metadata_to_field(name, child_meta)
        for name, child_meta in ids_metadata._children.items()
    ]

    schema_meta: dict[bytes, bytes] = {
        b"ids_name": ids_metadata.name.encode(),
        b"data_dictionary_version": dd_version.encode(),
        b"occurrence": str(occurrence).encode(),
    }
    return pa.schema(fields, metadata=schema_meta)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _metadata_to_field(name: str, meta: IDSMetadata) -> pa.Field:
    """Convert a single :class:`IDSMetadata` node to a nullable Arrow field.

    Arrow field metadata (documentation, units, path) is embedded as bytes
    key/value pairs so it round-trips through Parquet without loss.
    """
    arrow_type = _metadata_to_type(meta)

    fmeta: dict[bytes, bytes] = {}
    if meta.documentation:
        fmeta[b"documentation"] = meta.documentation.encode()
    if meta.units:
        fmeta[b"units"] = meta.units.encode()
    if meta.path_string:
        fmeta[b"path"] = meta.path_string.encode()

    return pa.field(name, arrow_type, nullable=True, metadata=fmeta or None)


def _metadata_to_type(meta: IDSMetadata) -> pa.DataType:
    """Recursively map an IDS metadata node to an Arrow DataType."""
    dt = meta.data_type

    if dt is IDSDataType.STRUCTURE:
        return pa.struct(
            [_metadata_to_field(n, c) for n, c in meta._children.items()]
        )

    if dt is IDSDataType.STRUCT_ARRAY:
        inner_struct = pa.struct(
            [_metadata_to_field(n, c) for n, c in meta._children.items()]
        )
        return pa.list_(inner_struct)

    # Primitive type: build up nested pa.list_ for each dimension
    base: pa.DataType = _primitive_base_type(dt)
    for _ in range(meta.ndim):
        base = pa.list_(base)
    return base


def _primitive_base_type(dt: IDSDataType) -> pa.DataType:
    """Return the innermost Arrow scalar type for a primitive IDS data type."""
    if dt is IDSDataType.FLT:
        return pa.float64()
    if dt is IDSDataType.INT:
        return pa.int32()
    if dt is IDSDataType.STR:
        return pa.string()
    if dt is IDSDataType.CPX:
        return COMPLEX_ARROW_TYPE  # FixedSizeList<2>[float64]
    raise ValueError(f"Unknown primitive IDS data type: {dt!r}")
