"""Plain floats are one DOUBLE column (storage protocol "3"): no ``*_exact`` text, signed zero stored as 0.0, NaN."""

import math
import pathlib
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pytest
from httk.core.storage import IdentitySkip, StorageInfo, content_id

from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql.layout import actual_columns

pytestmark = pytest.mark.xdist_group("bulk-heavy")


@dataclass(frozen=True)
class Measured:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="float_measured")

    label: str
    energy: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class Noted:
    """NaN is storable only outside content identity: an ``IdentitySkip`` float."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="float_noted")

    label: str
    note: Annotated[float | None, IdentitySkip()] = None


@dataclass(frozen=True)
class Series:
    """NaN in a float-sequence element: a non-deduplicated record computes no content identity."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="float_series", dedup="none")

    label: str
    values: tuple[float, ...]


MODES = ("serial", "parity", "deferred", "parallel")


def _store(backend: str, path: pathlib.Path | None = None) -> SqlStore:
    """A fresh store; a file store when ``path`` is given (each call reopens it)."""
    if backend == "duckdb":
        pytest.importorskip("duckdb_engine")
        return SqlStore(Backend.duckdb(path), entry_records={})
    return SqlStore(Backend.sqlite(path), entry_records={})


def _save(store: SqlStore, mode: str, *records: Any) -> None:
    if mode == "serial":
        for record in records:
            store.save(record)
        return
    if mode == "parallel" and store.backend_facts.parallel_shard_format == "parquet":
        pytest.importorskip("pyarrow")
    options: dict[str, Any] = {
        "parity": {"finalize": "parity"},
        "deferred": {"finalize": "deferred"},
        "parallel": {"workers": 2},
    }[mode]
    with store.bulk_ingest(**options) as bulk:
        for record in records:
            bulk.save(record)


def _rows(store: SqlStore, record_type: type) -> list[Any]:
    searcher = store.searcher()
    return [row.record for row in searcher.results(record=searcher.variable(record_type))]


def _only(store: SqlStore, record_type: type) -> Any:
    (record,) = _rows(store, record_type)
    return record


def test_schema_has_no_exact_float_columns() -> None:
    schema = resolve_schema(Measured)
    assert [column.name for column in schema.field("energy").columns] == ["energy"]
    child = schema.field("values").child
    assert child is not None and [column.name for column in child.element_columns] == ["values"]


@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_store_has_no_exact_float_columns(backend: str) -> None:
    store = _store(backend)
    store.save(Measured("m", 1.5, (2.5,)))
    with store._database.engine.connect() as connection:
        parent = actual_columns(connection, "float_measured")
        child = actual_columns(connection, "float_measured_values")
    assert "energy" in parent and "values" in child
    assert not any(name.endswith("_exact") for name in parent | child)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_signed_zero_is_zero_and_int_valued_floats_read_back_as_floats(
    backend: str, mode: str, tmp_path: pathlib.Path
) -> None:
    """Signed zero is not content: -0.0 is stored as 0.0, so a fetched record keeps its id and deduplicates.

    File stores across a reopen: DuckDB's on-disk compression would otherwise fold mixed-sign zeros.
    """
    record = Measured("m", -0.0, (-0.0, 0.0, 2.0))
    path = tmp_path / f"store.{backend}"
    _save(_store(backend, path), mode, record, Measured("i", 3, (2,)))  # type: ignore[arg-type]  # ints in float fields
    store = _store(backend, path)
    integral, found = sorted(_rows(store, Measured), key=lambda row: row.label)
    assert [value.hex() for value in (found.energy, *found.values)] == ["0x0.0p+0"] * 3 + [(2.0).hex()]
    assert [type(value) for value in (integral.energy, *integral.values)] == [float, float]
    assert content_id(found) == content_id(record)
    sid = store.save(found)
    assert store.save(record) == sid
    assert len(_rows(store, Measured)) == 2  # no new row


@pytest.mark.parametrize("mode", MODES)
def test_sqlite_rejects_nan(mode: str) -> None:
    with pytest.raises(ValueError, match=r"float_noted\.note is NaN: SQLite cannot store NaN"):
        _save(_store("sqlite"), mode, Noted("n", math.nan))
    with pytest.raises(ValueError, match=r"float_series_values\.values is NaN: SQLite cannot store NaN"):
        _save(_store("sqlite"), mode, Series("s", (1.0, math.nan)))


def test_sqlite_rejects_nan_keep_stored_fill() -> None:
    from httk.core.storage import KeepStored

    # ponytail: a test-local record class; module level if another test needs it.
    @dataclass(frozen=True)
    class Kept:
        __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="float_kept")

        label: str
        note: Annotated[float | None, IdentitySkip(), KeepStored()] = None

    store = _store("sqlite")
    store.save(Kept("k"))
    with pytest.raises(ValueError, match=r"float_kept\.note is NaN"):
        store.save(Kept("k", math.nan))


@pytest.mark.parametrize("mode", MODES)
def test_duckdb_stores_nan(mode: str) -> None:
    store = _store("duckdb")
    _save(store, mode, Noted("n", math.nan))
    note = _only(store, Noted).note
    assert note is not None and math.isnan(note)
    store = _store("duckdb")
    _save(store, mode, Series("s", (1.0, math.nan)))
    first, second = _only(store, Series).values
    assert first == 1.0 and math.isnan(second)
