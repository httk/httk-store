"""Float tensors (``Annotated[tuple[tuple[float, ...], ...], Shape(r, c)]``): DOUBLE columns only, no ``*_exact``."""

import math
import pathlib
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

import pytest
from httk.core import FracVector
from httk.core.storage import IdentitySkip, Indexed, KeepStored, Shape, StorageInfo, Unique, content_id, float_matrix
from postgres_support import postgres_database

from httk.store.backend.schema import SchemaError, _is_float_tensor, resolve_schema
from httk.store.backend.sql import Backend, SqlStore, StoreEntryProvider
from httk.store.backend.sql.layout import actual_columns
from httk.store.backend.sql.searcher import SqlColumn
from httk.store.backend.sql.stored_properties import _SqlQueryContext

pytestmark = pytest.mark.xdist_group("bulk-heavy")

type Matrix = tuple[tuple[float, ...], ...]

MODES = ("serial", "parity", "deferred", "parallel")


@dataclass(frozen=True)
class Lattice:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="float_lattice")

    label: str
    basis: Annotated[tuple[tuple[float, ...], ...], Shape(3, 3)]
    coords: Annotated[tuple[tuple[float, ...], ...], Shape(0, 3)]
    strain: Annotated[tuple[tuple[float, ...], ...], Shape(1, 2), Indexed()] | None = None
    note: Annotated[tuple[tuple[float, ...], ...] | None, Shape(1, 2), IdentitySkip()] = None


@dataclass(frozen=True)
class Noisy:
    """NaN is storable only where no content identity is computed from the field."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="float_noisy", dedup="none")

    label: str
    fixed: Annotated[tuple[tuple[float, ...], ...], Shape(1, 2)]
    rows: Annotated[tuple[tuple[float, ...], ...], Shape(0, 2)]


@dataclass(frozen=True)
class ByValue:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="float_by_value", dedup="by_value")

    basis: Annotated[tuple[tuple[float, ...], ...], Shape(2, 2)]


@dataclass(frozen=True)
class Served:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="float_served")

    basis: Annotated[tuple[tuple[float, ...], ...], Shape(2, 2)]
    coords: Annotated[tuple[tuple[float, ...], ...], Shape(0, 2)]
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


BASIS = ((1.0, -0.0, 0.0), (0.0, 2.5, 0.0), (0.0, 0.0, 3.0))  # deliberately not normalised: -0.0
COORDS = ((0.0, 0.5, 0.25), (0.5, -0.0, 0.75))
LATTICE = Lattice("a", BASIS, COORDS, strain=((0.01, -0.02),))
BARE = Lattice("b", float_matrix([[2, 0, 0], [0, 2, 0], [0, 0, 2]], 3), ())


def _store(backend: str, path: pathlib.Path | None = None) -> SqlStore:
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


def _hex(matrix: Matrix) -> list[list[str]]:
    return [[value.hex() for value in row] for row in matrix]


def test_schema_columns_and_discriminator() -> None:
    schema = resolve_schema(Lattice)
    basis, coords, strain = (schema.field(name) for name in ("basis", "coords", "strain"))
    assert (basis.role, basis.codec_name, basis.shape) == ("fixed_array", "float", Shape(3, 3))
    assert [column.name for column in basis.columns] == [f"basis_{i}" for i in range(9)]
    assert {column.kind for column in basis.columns} == {"float"}
    assert all(column.nullable for column in strain.columns) and all(column.indexed for column in strain.columns)
    assert (coords.role, coords.codec_name, coords.columns) == ("child", "float", ())
    assert coords.child is not None
    assert [column.name for column in coords.child.element_columns] == ["coords_0", "coords_1", "coords_2"]
    assert all(_is_float_tensor(spec) for spec in (basis, coords, strain))

    @dataclass(frozen=True)
    class Other:
        values: tuple[float, ...]
        exact: Annotated[FracVector, Shape(2, 2)]

    other = resolve_schema(Other)
    assert other.field("values").codec_name == "float"
    assert not _is_float_tensor(other.field("values")) and not _is_float_tensor(other.field("exact"))


@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_store_has_no_exact_columns(backend: str) -> None:
    store = _store(backend)
    store.save(LATTICE)
    with store._database.engine.connect() as connection:
        parent = actual_columns(connection, "float_lattice")
        child = actual_columns(connection, "float_lattice_coords")
    assert {"basis_0", "basis_8", "strain_1", "note_0"} <= parent and {"coords_0", "coords_2"} <= child
    assert not any(name.endswith("_exact") for name in parent | child)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_round_trip_none_and_signed_zero(backend: str, mode: str, tmp_path: pathlib.Path) -> None:
    """-0.0 is stored as 0.0, None and empty children round-trip, and a fetched record deduplicates."""
    path = tmp_path / f"store.{backend}"
    _save(_store(backend, path), mode, LATTICE, BARE)
    store = _store(backend, path)
    found, bare = sorted(_rows(store, Lattice), key=lambda row: row.label)
    assert found == Lattice("a", float_matrix(BASIS, 3), float_matrix(COORDS, 3), strain=((0.01, -0.02),))
    assert _hex(found.basis)[0][1] == _hex(found.coords)[1][1] == "0x0.0p+0"
    assert all(type(value) is float for row in (*found.basis, *found.coords) for value in row)
    assert (bare.strain, bare.note, bare.coords) == (None, None, ())
    assert content_id(found) == content_id(LATTICE)
    sid = store.save(found)
    assert store.save(LATTICE) == sid
    assert len(_rows(store, Lattice)) == 2
    eager = store.fetch(Lattice, sid, eager=True)
    assert eager == found and found._httk_stored_floats("coords") is None


@pytest.mark.parametrize("mode", MODES)
def test_sqlite_rejects_nan_in_tensors(mode: str) -> None:
    with pytest.raises(ValueError, match=r"float_noisy\.fixed_1 is NaN: SQLite cannot store NaN"):
        _save(_store("sqlite"), mode, Noisy("f", ((1.0, math.nan),), ()))
    with pytest.raises(ValueError, match=r"float_noisy_rows\.rows_0 is NaN: SQLite cannot store NaN"):
        _save(_store("sqlite"), mode, Noisy("r", ((1.0, 2.0),), ((math.nan, 1.0),)))
    with pytest.raises(ValueError, match=r"float_lattice\.note_0 is NaN"):
        _save(_store("sqlite"), mode, Lattice("n", BASIS, (), note=((math.nan, 0.0),)))


@pytest.mark.parametrize("mode", MODES)
def test_duckdb_stores_nan_in_tensors(mode: str) -> None:
    store = _store("duckdb")
    _save(store, mode, Noisy("n", ((1.0, math.nan),), ((math.nan, 2.0),)))
    (found,) = _rows(store, Noisy)
    assert found.fixed[0][0] == 1.0 and math.isnan(found.fixed[0][1])
    assert math.isnan(found.rows[0][0]) and found.rows[0][1] == 2.0


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (Lattice("l", [list(row) for row in BASIS], ()), r"Lattice\.basis: expected 3 rows of 3 floats"),  # type: ignore[arg-type]
        (Lattice("i", ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ()), r"Lattice\.basis: expected 3 rows"),  # type: ignore[arg-type]
        (Lattice("r", BASIS[:2], ()), r"Lattice\.basis: expected 3 rows"),
        (Lattice("w", BASIS, ((0.0, 0.5),)), r"Lattice\.coords: expected any number of rows of 3 floats"),
    ],
)
@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_non_canonical_or_misshapen_values_raise(record: Lattice, message: str, mode: str) -> None:
    with pytest.raises(ValueError, match=message):
        _save(_store("sqlite"), mode, record)


def test_unique_and_keep_stored_are_rejected() -> None:
    @dataclass(frozen=True)
    class UniqueTensor:
        basis: Annotated[tuple[tuple[float, ...], ...], Shape(2, 2), Unique()]

    @dataclass(frozen=True)
    class KeptFloat:
        label: str
        rows: Annotated[tuple[tuple[float, ...], ...] | None, Shape(0, 2), IdentitySkip(), KeepStored()] = None

    @dataclass(frozen=True)
    class KeptExact:
        label: str
        basis: Annotated[FracVector | None, Shape(2, 2), IdentitySkip(), KeepStored()] = None

    with pytest.raises(SchemaError, match=r"UniqueTensor\.basis: Unique does not apply to a float tensor"):
        resolve_schema(UniqueTensor)
    for cls in (KeptFloat, KeptExact):
        with pytest.raises(SchemaError, match=rf"{cls.__name__}\.\w+: KeepStored does not apply"):
            resolve_schema(cls)


@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_by_value_dedup(backend: str) -> None:
    store = _store(backend)
    sid = store.save(ByValue(((1.0, 0.0), (0.0, 1.0))))
    assert store.save(ByValue(((1.0, -0.0), (0.0, 1.0)))) == sid
    assert store.save(ByValue(((1.0, 0.0), (0.0, 2.0)))) != sid
    assert len(_rows(store, ByValue)) == 2


def test_served_json_and_fulltype() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.save(Served(((1.0, 0.5), (0.0, 1.0)), ((0.25, 0.75),), id="x-1", immutable_id="x-1~1"))
        provider = StoreEntryProvider(store, {"served": Served})
        properties = provider.entry_types()["served"].properties
        basis = properties["_httk_custom_basis"].as_optimade()
        assert basis["items"]["items"]["x-optimade-type"] == "float"
        assert basis["x-optimade-dimensions"]["sizes"] == [2, 2]
        assert properties["_httk_custom_coords"].as_optimade()["items"]["items"]["x-optimade-type"] == "float"
        (row,) = provider.records("served")
        assert row["_httk_custom_basis"] == [[1.0, 0.5], [0.0, 1.0]]
        assert row["_httk_custom_coords"] == [[0.25, 0.75]]


@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_child_column_queries_and_counts(backend: str) -> None:
    """The searcher and stored-property contexts read a float-tensor child through its first element column."""
    store = _store(backend)
    _save(store, "serial", LATTICE, BARE, Lattice("c", BASIS, ((0.9, 0.0, 0.0),)))

    def labels(build: Any) -> set[str]:
        searcher = store.searcher()
        variable = searcher.variable(Lattice)
        searcher.add(build(searcher, variable))
        searcher._output(variable, "record")
        return {row[0].label for row in searcher.results()}

    assert labels(lambda _searcher, v: v.coords > 0.4) == {"a", "c"}

    def counted(searcher: Any, variable: Any) -> Any:
        ctx = _SqlQueryContext(searcher, variable)
        return ctx.compare(ctx.count(ctx.scope("coords")), "=", ctx.constant(2))

    assert labels(counted) == {"a"}

    def filtered(searcher: Any, variable: Any) -> Any:
        ctx = _SqlQueryContext(searcher, variable)
        coords = ctx.scope("coords")
        large = ctx.filtered(coords, ctx.compare(coords.field("value"), ">", ctx.constant(0.4)))
        return ctx.compare(ctx.count(large), "=", ctx.constant(1))

    assert labels(filtered) == {"a", "c"}


def test_fixed_tensor_result_projection() -> None:
    store = _store("sqlite")
    _save(store, "serial", LATTICE, BARE)
    searcher = store.searcher()
    record = searcher.variable(Lattice)
    spec = resolve_schema(Lattice).field("strain")
    strain = SqlColumn(searcher, record._alias.c["strain_0"], variable=record, spec=spec)
    searcher._output(record.label, "label")
    searcher._output(strain, "strain")
    assert {row.label: row.strain for row in searcher.results()} == {"a": ((0.01, -0.02),), "b": None}


def test_mongo_documents_round_trip_and_mapping() -> None:
    pytest.importorskip("bson")
    from httk.core.storage import project_storage_record

    from httk.store.backend.mongo.documents import decode_record, encode_record
    from httk.store.backend.mongo.mapping import index_specs_for, validator_for

    schema = resolve_schema(Lattice)
    for record in (LATTICE, BARE):
        document = encode_record(schema, project_storage_record(Lattice, record), record, Lattice, int)
        assert not any(key.endswith("_exact") for key in document)
        assert all(not any(key.endswith("_exact") for key in item) for item in document["coords"])
        assert decode_record(schema, {"f": document}, int) == Lattice(
            record.label, float_matrix(record.basis, 3), float_matrix(record.coords, 3), record.strain
        )
    document = encode_record(schema, project_storage_record(Lattice, LATTICE), LATTICE, Lattice, int)
    assert document["basis_1"].hex() == "0x0.0p+0" and document["coords"][1] == {
        "coords_0": 0.5,
        "coords_1": 0.0,
        "coords_2": 0.75,
    }
    assert "strain_0" not in encode_record(schema, project_storage_record(Lattice, BARE), BARE, Lattice, int)
    with pytest.raises(ValueError, match=r"Lattice\.coords: expected any number of rows of 3 floats"):
        bad = Lattice("w", BASIS, ((0.0, 0.5),))
        encode_record(schema, project_storage_record(Lattice, bad), bad, Lattice, int)

    validator = validator_for(schema)["$jsonSchema"]["properties"]["f"]
    assert "dependencies" not in validator
    assert validator["properties"]["basis_0"] == {"bsonType": "double"}
    assert {"basis_0", "basis_8"} <= set(validator["required"]) and "strain_0" not in validator["required"]
    item = validator["properties"]["coords"]["items"]
    assert item["required"] == ["coords_0", "coords_1", "coords_2"] and "dependencies" not in item
    names = {spec.name for spec in index_specs_for(schema)}
    assert {"ix_float_lattice_strain_0", "ix_float_lattice_strain_1"} <= names
    assert not any("basis" in name or "_exact" in name for name in names)


@pytest.mark.parametrize("mode", ["serial", "parallel"])
def test_postgresql_round_trip_and_nan(mode: str) -> None:
    with postgres_database() as database:
        store = SqlStore(database, entry_records={})
        _save(store, mode, LATTICE, BARE, Noisy("n", ((1.0, math.nan),), ((math.nan, 2.0), (3.0, 4.0))))
        store._clear_identity_caches()
        found, bare = sorted(_rows(store, Lattice), key=lambda row: row.label)
        assert found == Lattice("a", float_matrix(BASIS, 3), float_matrix(COORDS, 3), strain=((0.01, -0.02),))
        assert (bare.strain, bare.coords) == (None, ())
        (noisy,) = _rows(store, Noisy)
        assert math.isnan(noisy.fixed[0][1]) and math.isnan(noisy.rows[0][0]) and noisy.rows[1] == (3.0, 4.0)
