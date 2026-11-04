"""Tensor ``*_exact`` columns hold ``""`` when the float columns already determine the value."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Annotated, ClassVar

import httk.core
import pytest
import sqlalchemy
from httk.core import FracVector
from httk.core.storage import Indexed, Shape, StorageInfo, Unique, content_id

from httk.store import EntryIdScheme
from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql.searcher import SqlColumn

pytest.importorskip("httk.atomistic")

pytestmark = pytest.mark.xdist_group("bulk-heavy")

COD = Path(__file__).parent / "fixtures" / "cod" / "1000008.cif"
SITES = "atomistic_sites_reduced_coords"


@dataclass(frozen=True)
class MatrixRecord:
    name: str
    matrix: Annotated[FracVector, Shape(2, 2)] | None = None


@dataclass(frozen=True)
class ByValueMatrix:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    matrix: Annotated[FracVector, Shape(2, 2)]


@dataclass(frozen=True)
class UniqueMatrix:
    matrix: Annotated[FracVector, Shape(2, 2), Unique()]
    label: str = ""


def _database(kind: str) -> Backend:
    if kind == "duckdb":
        pytest.importorskip("duckdb_engine")
    return getattr(Backend, kind)()


def _column(database: Backend, table: str, column: str) -> list[object]:
    with database.engine.connect() as connection:
        return [row[0] for row in connection.execute(sqlalchemy.text(f'SELECT "{column}" FROM "{table}"'))]


def _structure_store(database: Backend) -> SqlStore:
    from httk.atomistic import StructureEntry, UnitcellStructureRecord

    return SqlStore(
        database, entry_records={StructureEntry: UnitcellStructureRecord}, entry_ids=EntryIdScheme("httk.test", "1")
    )


def _round_trip(kind: str, view: object) -> tuple[object, list[object]]:
    from httk.atomistic import UnitcellStructureRecord, UnitcellStructureView

    database = _database(kind)
    try:
        store = _structure_store(database)
        sid = store.save(view)
        store._clear_identity_caches()
        lazy = UnitcellStructureView(store.fetch(UnitcellStructureRecord, sid))
        eager = UnitcellStructureView(store.fetch(UnitcellStructureRecord, sid, eager=True))
        assert lazy == view
        assert eager == view
        assert content_id(lazy) == content_id(view)
        return lazy, _column(database, SITES, "reduced_coords_exact")
    finally:
        database.dispose()


@pytest.mark.parametrize("kind", ["sqlite", "duckdb"])
def test_decimal_coordinates_store_the_sentinel(kind: str) -> None:
    from httk.atomistic import UnitcellStructureView

    view = UnitcellStructureView(httk.core.load(str(COD)))
    _fetched, texts = _round_trip(kind, view)
    assert texts and set(texts) == {""}


def _thirds() -> object:
    from httk.atomistic import Cell, Sites, Species, UnitcellStructure, UnitcellStructureView

    return UnitcellStructureView(
        UnitcellStructure(
            Cell([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
            Sites([[0, 0, 0], ["1/3", "2/3", "1/4"]]),
            (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
            ("Na", "Cl"),
        )
    )


@pytest.mark.parametrize("kind", ["sqlite", "duckdb"])
def test_thirds_keep_the_full_text(kind: str) -> None:
    _fetched, texts = _round_trip(kind, _thirds())
    assert texts == ["", "12;4,8,3"]


@pytest.mark.parametrize("kind", ["sqlite", "duckdb"])
def test_fixed_array_sentinel_none_and_projection(kind: str) -> None:
    database = _database(kind)
    try:
        store = SqlStore(database, entry_records={})
        decimal = FracVector([["0.1", 2], [-3, "1/4"]])
        third = FracVector([["1/3", 0], [0, 1]])
        sids = [store.save(MatrixRecord(name, value)) for name, value in (("d", decimal), ("t", third), ("n", None))]
        texts = dict(
            zip(_column(database, "matrix_record", "name"), _column(database, "matrix_record", "matrix_exact"))
        )
        assert texts == {"d": "", "t": "3;1,0,0,3", "n": None}
        store._clear_identity_caches()
        assert [store.fetch(MatrixRecord, sid).matrix for sid in sids] == [decimal, third, None]
        # Fixed-shape tensors are not publicly queryable yet; build the projection column the
        # result layer decodes (first float column as element, the rest plus exact as extras).
        searcher = store.searcher()
        record = searcher.variable(MatrixRecord)
        spec = resolve_schema(MatrixRecord).field("matrix")
        matrix = SqlColumn(searcher, record._alias.c["matrix_0"], variable=record, spec=spec)
        searcher._output(record.name, "name")
        searcher._output(matrix, "matrix")
        projected = {row.name: row.matrix for row in searcher.results()}
        assert projected == {"d": decimal, "t": third, "n": None}
    finally:
        database.dispose()


def test_by_value_fixed_array_keeps_the_full_text() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        matrix = FracVector([[1, 0], [0, 1]])
        first = store.save(ByValueMatrix(matrix))
        assert _column(database, "by_value_matrix", "matrix_exact") == ["1;1,0,0,1"]
        assert store.save(ByValueMatrix(matrix)) == first


def test_bulk_serial_and_parallel_write_the_same_sentinel() -> None:
    from httk.atomistic import UnitcellStructureView

    cod = UnitcellStructureView(httk.core.load(str(COD)))
    texts = []
    for workers in (1, 2):
        database = Backend.sqlite()
        try:
            store = _structure_store(database)
            with store.bulk_ingest(workers=workers, chunk_size=1) as bulk:
                # The repeated structure exercises dedupe across the workers' shards.
                for view in (cod, _thirds(), cod):
                    bulk.save(view)
            texts.append(_column(database, SITES, "reduced_coords_exact"))
        finally:
            database.dispose()
    assert sorted(texts[0]) == sorted(texts[1])
    assert set(texts[0]) == {"", "12;4,8,3"}


def test_legacy_full_text_reads_like_the_sentinel() -> None:
    database = Backend.sqlite()
    try:
        store = SqlStore(database, entry_records={})
        value = FracVector([["0.1", 2], [-3, "1/4"]])
        sid = store.save(MatrixRecord("d", value))
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("UPDATE matrix_record SET matrix_exact = '20;2,40,-60,5'"))
        store._clear_identity_caches()
        assert store.fetch(MatrixRecord, sid).matrix == value
        assert store.fetch(MatrixRecord, sid).matrix.to_fractions() == [[Fraction(1, 10), 2], [-3, Fraction(1, 4)]]
    finally:
        database.dispose()


@dataclass(frozen=True)
class TensorDocument:
    matrix: Annotated[FracVector, Shape(2, 2)] | None
    rows: Annotated[FracVector, Shape(0, 3)]


def test_mongo_documents_carry_floats_and_the_sentinel() -> None:
    pytest.importorskip("bson")
    from httk.store.backend.mongo.documents import decode_record, encode_record

    schema = resolve_schema(TensorDocument)
    record = TensorDocument(FracVector([["0.1", 2], [3, "1/4"]]), FracVector([["0.5", "1/3", 0], [0, 0, "0.125"]]))
    document = encode_record(schema, {"matrix": record.matrix, "rows": record.rows}, record, TensorDocument, int)
    assert document["matrix_exact"] == ""
    assert [element["rows_exact"] for element in document["rows"]] == ["6;3,2,0", ""]
    assert decode_record(schema, {"f": document}, int) == record
    empty = TensorDocument(None, FracVector([[1, 2, 3]]))
    assert (
        decode_record(
            schema, {"f": encode_record(schema, {"matrix": None, "rows": empty.rows}, empty, TensorDocument, int)}, int
        )
        == empty
    )


@pytest.mark.parametrize("kind", ["sqlite", "duckdb"])
def test_unique_fixed_array_is_unique_as_a_whole_tensor(kind: str) -> None:
    database = _database(kind)
    try:
        store = SqlStore(database, entry_records={})
        # The two tensors share their first element; only the whole tensor is unique.
        values = [FracVector([["0.1", "0.2"], ["0.3", "0.4"]]), FracVector([["0.1", "0.6"], ["0.7", "0.8"]])]
        sids = [store.save(UniqueMatrix(value)) for value in values]
        assert sorted(_column(database, "unique_matrix", "matrix_exact")) == ["10;1,2,3,4", "10;1,6,7,8"]
        store._clear_identity_caches()
        assert [store.fetch(UniqueMatrix, sid).matrix for sid in sids] == values
        # As for a Unique scalar: an equal record deduplicates, a different one with the same tensor is rejected.
        assert store.save(UniqueMatrix(values[0])) == sids[0]
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            store.save(UniqueMatrix(values[0], "other"))
    finally:
        database.dispose()


def test_mongo_unique_fixed_array_is_unique_as_a_whole_tensor(mongo_test_database) -> None:
    from pymongo.errors import DuplicateKeyError

    from httk.store.backend.mongo import MongoStore
    from httk.store.backend.mongo.mapping import collection_name_for

    store = MongoStore(mongo_test_database, entry_records={})
    values = [FracVector([["0.1", "0.2"], ["0.3", "0.4"]]), FracVector([["0.1", "0.6"], ["0.7", "0.8"]])]
    sids = [store.save(UniqueMatrix(value)) for value in values]
    collection = mongo_test_database.database[collection_name_for(resolve_schema(UniqueMatrix))]
    assert sorted(document["f"]["matrix_exact"] for document in collection.find()) == ["10;1,2,3,4", "10;1,6,7,8"]
    store._clear_identity_caches()
    assert [store.fetch(UniqueMatrix, sid).matrix for sid in sids] == values
    assert store.save(UniqueMatrix(values[0])) == sids[0]
    with pytest.raises(DuplicateKeyError):
        store.save(UniqueMatrix(values[0], "other"))


def test_mongo_tensor_indexes_cover_the_exact_key_only() -> None:
    pytest.importorskip("pymongo")
    from httk.store.backend.mongo.mapping import index_specs_for

    @dataclass(frozen=True)
    class IndexedMatrix:
        matrix: Annotated[FracVector, Shape(1, 2), Indexed()]

    def field_indexes(cls: type) -> dict[str, bool]:
        specs = index_specs_for(resolve_schema(cls))
        return {spec.keys[0][0]: spec.unique for spec in specs if spec.keys[0][0].startswith("f.matrix")}

    assert field_indexes(UniqueMatrix) == {"f.matrix_exact": True}
    assert field_indexes(IndexedMatrix) == {"f.matrix_0": False, "f.matrix_1": False, "f.matrix_exact": False}
