# This file is part of IMAS-Python.
# You should have received the IMAS-Python LICENSE file with this project.
"""DBEntry implementation backed by Parquet files.

Storage layout
--------------
Each :class:`imas.db_entry.DBEntry` corresponds to a **directory** on disk::

    <base_dir>/
    ├── core_profiles/
    │   ├── 0.parquet
    │   └── 1.parquet
    └── equilibrium/
        └── 0.parquet

One Parquet file per (IDS name, occurrence) pair.

URI format
----------
Use the ``imas:parquet?path=<dir>`` URI with
:class:`imas.db_entry.DBEntry`::

    import imas
    with imas.DBEntry("imas:parquet?path=/tmp/mydb", "w") as entry:
        entry.put(cp)

Multiple query parameters can be separated by ``;``::

    imas:parquet?path=/tmp/mydb;compression=snappy

Supported query parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``path``
    (Required) Base directory for the data entry.
``compression``
    Parquet compression codec.  Defaults to ``"zstd"``.
``compression_level``
    Compression level (integer).  Defaults to ``3``.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import List, Optional, Union
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq

from imas.backends.db_entry_impl import (
    DBEntryImpl,
    GetSampleParameters,
    GetSliceParameters,
)
from imas.backends.parquet.ids2pq import ids_to_parquet
from imas.backends.parquet.pq2ids import collect_filled_paths, parquet_to_ids
from imas.exception import DataEntryException
from imas.ids_convert import NBCPathMap, dd_version_map_from_factories
from imas.ids_factory import IDSFactory
from imas.ids_toplevel import IDSToplevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level guard: check pyarrow is available at import time (not lazily)
# ---------------------------------------------------------------------------
try:
    import pyarrow  # noqa: F401
except ImportError as _err:
    raise ImportError(
        "The `pyarrow` Python package is required for the Parquet backend. "
        "Install it with:  pip install imas-python[parquet]"
    ) from _err

_SCHEMA_META_KEY = b"data_dictionary_version"


# ---------------------------------------------------------------------------
# URI parsing helper
# ---------------------------------------------------------------------------


def _parse_uri(uri: str) -> tuple[str, dict[str, str]]:
    """Parse ``imas:parquet?path=...`` into ``(base_dir, params)``.

    Also accepts a bare directory path (no ``imas:parquet?`` prefix) for
    convenience, e.g. when used programmatically.
    """
    if uri.startswith("imas:parquet"):
        # Strip the scheme prefix and parse remaining as a query string.
        # e.g. "imas:parquet?path=/tmp/db;compression=snappy"
        rest = uri[len("imas:parquet"):]
        if rest.startswith("?"):
            rest = rest[1:]
        # Split on ";" (IMAS convention) or "&" (URL convention)
        params = {}
        for pair in rest.replace("&", ";").split(";"):
            if "=" in pair:
                k, _, v = pair.partition("=")
                params[k.strip()] = v.strip()

        base_dir = params.pop("path", "")
        if not base_dir:
            raise ValueError(
                f"Missing 'path' parameter in Parquet URI: {uri!r}"
            )
        return base_dir, params

    # Plain path treated as base directory
    return uri, {}


# ---------------------------------------------------------------------------
# DBEntryImpl implementation
# ---------------------------------------------------------------------------


class PQDBEntryImpl(DBEntryImpl):
    """DBEntry implementation using a directory of Parquet files.

    One ``.parquet`` file is written per (IDS name, occurrence) pair under
    *base_dir*.  The Data Dictionary version and occurrence are stored in the
    Parquet schema metadata.
    """

    def __init__(
        self,
        base_dir: str,
        mode: str,
        factory: IDSFactory,
        *,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> None:
        """Open or create a Parquet-backed data entry.

        Args:
            base_dir: Root directory for this data entry.
            mode: Open mode: ``"r"``, ``"a"``, ``"w"``, or ``"x"``.
            factory: IDS factory (determines the desired in-memory DD version).
            compression: Parquet compression codec.
            compression_level: Parquet compression level.
        """
        self._base_dir = os.path.abspath(base_dir)
        self._factory = factory
        self._compression = compression
        self._compression_level = compression_level

        self._init_directory(mode)

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def _init_directory(self, mode: str) -> None:
        """Validate / create *base_dir* according to *mode*."""
        exists = os.path.isdir(self._base_dir)

        if mode == "r":
            if not exists:
                raise DataEntryException(
                    f"Parquet data entry does not exist: {self._base_dir!r}"
                )
        elif mode == "x":
            if exists and any(
                fname.endswith(".parquet")
                for _, _, files in os.walk(self._base_dir)
                for fname in files
            ):
                raise DataEntryException(
                    f"Parquet data entry already contains data: {self._base_dir!r}"
                )
            os.makedirs(self._base_dir, exist_ok=True)
        elif mode == "w":
            # Wipe existing content and re-create
            if exists:
                shutil.rmtree(self._base_dir)
            os.makedirs(self._base_dir, exist_ok=True)
        else:  # "a" or anything else
            os.makedirs(self._base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _ids_dir(self, ids_name: str) -> str:
        return os.path.join(self._base_dir, ids_name)

    def _ids_path(self, ids_name: str, occurrence: int) -> str:
        return os.path.join(self._base_dir, ids_name, f"{occurrence}.parquet")

    # ------------------------------------------------------------------
    # DBEntryImpl interface
    # ------------------------------------------------------------------

    @classmethod
    def from_uri(
        cls, uri: str, mode: str, factory: IDSFactory
    ) -> "PQDBEntryImpl":
        """Create a :class:`PQDBEntryImpl` from an IMAS-style Parquet URI."""
        base_dir, params = _parse_uri(uri)
        compression = params.get("compression", "zstd")
        try:
            compression_level = int(params.get("compression_level", "3"))
        except ValueError:
            compression_level = 3
        return cls(
            base_dir,
            mode,
            factory,
            compression=compression,
            compression_level=compression_level,
        )

    def close(self, *, erase: bool = False) -> None:
        """Close the data entry.  Optionally erase all files on disk.

        Keyword Args:
            erase: When ``True``, the entire base directory is deleted.
        """
        if erase:
            shutil.rmtree(self._base_dir, ignore_errors=True)
            logger.info("Erased Parquet data entry at %s", self._base_dir)

    def get(
        self,
        ids_name: str,
        occurrence: int,
        parameters: Union[None, GetSliceParameters, GetSampleParameters],
        destination: IDSToplevel,
        lazy: bool,
        nbc_map: Optional[NBCPathMap],
    ) -> IDSToplevel:
        """Load an IDS from a Parquet file.

        Note:
            ``get_slice`` and ``get_sample`` are not supported by the Parquet
            backend.  ``lazy`` loading is also not supported: the full IDS is
            read from disk on every ``get()`` call.
        """
        if parameters is not None:
            func = (
                "get_slice"
                if isinstance(parameters, GetSliceParameters)
                else "get_sample"
            )
            raise NotImplementedError(
                f"`{func}` is not available for the Parquet backend."
            )
        if lazy:
            logger.warning(
                "Lazy loading is not supported by the Parquet backend; "
                "returning a fully loaded IDS."
            )

        pq_path = self._ids_path(ids_name, occurrence)
        if not os.path.isfile(pq_path):
            raise DataEntryException(
                f"IDS {ids_name!r}, occurrence {occurrence} is not found "
                f"at {pq_path!r}."
            )

        # Determine the DD version stored in the file
        ds_factory = self._factory
        stored_dd_version = self._read_stored_dd_version(pq_path)
        if stored_dd_version and stored_dd_version != self._factory.dd_version:
            ds_factory = IDSFactory(stored_dd_version)

        if ds_factory.dd_version == destination._dd_version:
            parquet_to_ids(pq_path, destination, destination.metadata, None)
        else:
            # Build an NBC map from the on-disk DD to the in-memory DD
            ddmap, source_is_older = dd_version_map_from_factories(
                ids_name, ds_factory, self._factory
            )
            conversion_map = (
                ddmap.old_to_new if source_is_older else ddmap.new_to_old
            )
            parquet_to_ids(
                pq_path,
                destination,
                ds_factory.new(ids_name).metadata,
                conversion_map,
            )

        return destination

    def put(
        self,
        ids: IDSToplevel,
        occurrence: int,
        is_slice: bool,
    ) -> None:
        """Write an IDS to a Parquet file.

        Note:
            ``put_slice`` is not supported by the Parquet backend.
        """
        if is_slice:
            raise NotImplementedError(
                "`put_slice` is not available for the Parquet backend."
            )

        ids_name = ids.metadata.name
        pq_path = self._ids_path(ids_name, occurrence)

        # Create the IDS sub-directory on demand
        os.makedirs(self._ids_dir(ids_name), exist_ok=True)

        # Stamp the stored DD version in ids_properties.version_put if present
        if hasattr(ids.ids_properties, "version_put"):
            ids.ids_properties.version_put.data_dictionary = (
                self._factory.dd_version
            )

        ids_to_parquet(
            ids,
            pq_path,
            dd_version=self._factory.dd_version,
            occurrence=occurrence,
            compression=self._compression,
            compression_level=self._compression_level,
        )
        logger.debug(
            "Wrote IDS %s occurrence %d to %s", ids_name, occurrence, pq_path
        )

    def read_dd_version(self, ids_name: str, occurrence: int) -> str:
        """Return the DD version stored in the Parquet file metadata."""
        pq_path = self._ids_path(ids_name, occurrence)
        if not os.path.isfile(pq_path):
            raise DataEntryException(
                f"IDS {ids_name!r}, occurrence {occurrence} is not found."
            )
        version = self._read_stored_dd_version(pq_path)
        if not version:
            raise DataEntryException(
                f"Parquet file {pq_path!r} is missing 'data_dictionary_version' "
                "metadata."
            )
        return version

    def access_layer_version(self) -> str:
        return "N/A"  # Parquet backend does not use the Access Layer

    def delete_data(self, ids_name: str, occurrence: int) -> None:
        """Delete an IDS occurrence by removing its Parquet file.

        If the IDS sub-directory becomes empty after deletion it is also
        removed.
        """
        pq_path = self._ids_path(ids_name, occurrence)
        if not os.path.isfile(pq_path):
            raise DataEntryException(
                f"IDS {ids_name!r}, occurrence {occurrence} not found; "
                "nothing to delete."
            )
        os.remove(pq_path)
        logger.debug(
            "Deleted IDS %s occurrence %d (%s)", ids_name, occurrence, pq_path
        )
        # Clean up empty IDS directory
        ids_dir = self._ids_dir(ids_name)
        if os.path.isdir(ids_dir) and not os.listdir(ids_dir):
            os.rmdir(ids_dir)

    def list_all_occurrences(self, ids_name: str) -> List[int]:
        """Return a sorted list of all occurrence numbers for *ids_name*."""
        ids_dir = self._ids_dir(ids_name)
        if not os.path.isdir(ids_dir):
            return []

        occurrences: list[int] = []
        for fname in os.listdir(ids_dir):
            if fname.endswith(".parquet"):
                stem = fname[: -len(".parquet")]
                try:
                    occurrences.append(int(stem))
                except ValueError:
                    logger.warning(
                        "Unexpected filename %r in %s (expected '<int>.parquet')",
                        fname,
                        ids_dir,
                    )
        occurrences.sort()
        return occurrences

    def list_filled_paths(self, ids_name: str, occurrence: int) -> List[str]:
        """Return all non-null leaf data paths stored for the given IDS.

        Paths use ``/`` as separator and are relative to the IDS root,
        e.g. ``"ids_properties/comment"`` or ``"profiles_1d/time"``.

        The Parquet schema stores all DD fields (including absent ones as
        null).  This method reads the data to determine which fields are
        actually populated.
        """
        pq_path = self._ids_path(ids_name, occurrence)
        if not os.path.isfile(pq_path):
            raise DataEntryException(
                f"IDS {ids_name!r}, occurrence {occurrence} is not found."
            )
        table = pq.read_table(pq_path)
        row_dict: dict = table.to_pylist()[0]
        return collect_filled_paths(row_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_stored_dd_version(pq_path: str) -> str:
        """Read the ``data_dictionary_version`` from Parquet schema metadata.

        Returns an empty string if the key is absent.
        """
        meta = pq.read_schema(pq_path).metadata or {}
        raw = meta.get(_SCHEMA_META_KEY, b"")
        return raw.decode() if isinstance(raw, bytes) else str(raw)
