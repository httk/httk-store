"""``Inline()`` child rows: one child table carrying the element dataclass's own columns.

Layout, round trips, schema refusals, fingerprint, fsck, bulk ingest, and parity with the
link-table layout of the same value (reconstruction, served entries, content identity).
"""

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Any, ClassVar

import pytest
import sqlalchemy
from clickhouse_read_support import CLICKHOUSE_PARAM, bulk_store
from httk.core import FracVector
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import (
    IdentitySkip,
    Indexed,
    Inline,
    KeepStored,
    Shape,
    Skip,
    StorageInfo,
    Unique,
    content_id,
    stored_property,
)
from postgres_support import POSTGRES_PARAM, postgres_database

from httk.store import FederatedStore
from httk.store.backend.mongo.entry_provider import StoreEntryProvider as MongoEntryProvider
from httk.store.backend.schema import SchemaError, resolve_schema
from httk.store.backend.sql import Backend, SqlStore, StoredPropertySqlConfigurationError
from httk.store.backend.sql.entry_provider import StoreEntryProvider
from httk.store.backend.sql.stored_properties import _SqlQueryContext
from httk.store.query import UnsupportedQueryError
from httk.store.storage_layout import _table_fingerprint, classify_schema_upgrade
from httk.store.store_common import EntryMetadataConflictError

ORIGIN = FracVector(((0, 0, 0),))


@dataclass(frozen=True)
class Site:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="ignored_for_inline", dedup="none")

    symbol: Annotated[str, Indexed()]
    x: float
    occupancy: Fraction | None = None
    position: Annotated[FracVector, Shape(1, 3)] = ORIGIN
    moment: Annotated[tuple[tuple[float, ...], ...], Shape(1, 2)] = ((0.0, 0.0),)
    note: Annotated[str, Skip()] = "not stored"


@dataclass(frozen=True)
class Crystal:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(identity_name="test.inline.crystal")

    name: str
    sites: Annotated[tuple[Site, ...], Inline()]
    extra: Annotated[list[Site] | None, Inline()] = None


@dataclass(frozen=True)
class LinkedCrystal:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(identity_name="test.inline.crystal")

    name: str
    sites: tuple[Site, ...]
    extra: list[Site] | None = None


SI = Site("Si", 0.25, Fraction(1, 3), FracVector(((Fraction(1, 2), 0, Fraction(1, 4)),)), ((0.5, -1.5),))
O = Site("O", 0.75)
CRYSTALS = [
    Crystal("a", (SI, O), [O]),
    Crystal("b", ()),
    Crystal("c", (O,), []),
    Crystal("d", (Site("Na", 0.5, Fraction(1)),), None),
]


def _linked(crystal: Crystal) -> LinkedCrystal:
    return LinkedCrystal(crystal.name, crystal.sites, crystal.extra)


@contextmanager
def _sql_database(backend: str) -> Iterator[Backend]:
    if backend == "postgresql":
        with postgres_database() as postgres:
            yield postgres
        return
    if backend == "duckdb":
        pytest.importorskip("duckdb_engine")
    with Backend.duckdb() if backend == "duckdb" else Backend.sqlite() as local:
        yield local


@pytest.fixture(params=["sqlite", "duckdb", POSTGRES_PARAM])
def database(request) -> Iterator[Backend]:
    with _sql_database(request.param) as database:
        yield database


@pytest.fixture(params=["sqlite", "duckdb", POSTGRES_PARAM, "mongo"])
def store(request) -> Iterator[Any]:
    """A fresh store per backend, for the tests that hold for every backend."""
    if request.param == "mongo":
        from httk.store.backend.mongo import MongoDatabase, MongoStore

        client = request.getfixturevalue("mongo_test_client")
        name = f"httk_test_{uuid.uuid4().hex}"
        try:
            yield MongoStore(MongoDatabase(client, name), entry_records={})
        finally:
            client.drop_database(name)
        return
    with _sql_database(request.param) as database:
        yield SqlStore(database, entry_records={})


def _sql_child_row_count(store: Any) -> int | None:
    return len(_child_rows(store, "crystal_sites", "sites")) if isinstance(store, SqlStore) else None


def _inspect(store: SqlStore) -> sqlalchemy.Inspector:
    return sqlalchemy.inspect(store._database.engine)


def _child_rows(store: SqlStore, table: str, field: str) -> list[tuple[Any, ...]]:
    """Child rows as (parent name, row without the parent sid), in a sid-independent order."""
    parent = store._table("crystal")
    child = store._table(table)
    with store._read_connection() as connection:
        rows = connection.execute(
            sqlalchemy.select(parent.c["name"], *(column for column in child.c if column.name != "crystal_sid"))
            .select_from(child.join(parent, child.c["crystal_sid"] == parent.c["sid"]))
            .order_by(parent.c["name"], child.c[f"{field}_index"])
        ).all()
    return [tuple(row) for row in rows]


def test_physical_layout(database: Backend) -> None:
    store = SqlStore(database, entry_records={})
    store.save(CRYSTALS[0])
    inspector = _inspect(store)
    assert set(inspector.get_table_names()) - {
        name for name in inspector.get_table_names() if name.startswith("_httk")
    } == {
        "crystal",
        "crystal_sites",
        "crystal_extra",
    }
    columns = [column["name"] for column in inspector.get_columns("crystal_sites")]
    assert columns == [
        "crystal_sid",
        "sites_index",
        "symbol",
        "x",
        "occupancy",
        "occupancy_exact",
        "position_0",
        "position_1",
        "position_2",
        "position_exact",
        "moment_0",
        "moment_1",
    ]
    assert "extra_present" in [column["name"] for column in inspector.get_columns("crystal")]
    # DuckDB does not reflect indexes; the built table carries the one Indexed() requested.
    indexed = {tuple(column.name for column in index.columns) for index in store._table("crystal_sites").indexes}
    assert ("symbol",) in indexed


@pytest.mark.parametrize("eager", [False, True])
def test_round_trip(store: Any, eager: bool) -> None:
    sids = [store.save(crystal) for crystal in CRYSTALS]
    fetched = store.fetch_many(Crystal, sids, eager=eager)
    assert fetched == CRYSTALS
    assert type(fetched[0].sites) is tuple and type(fetched[0].extra) is list
    assert fetched[0].sites[0].position == SI.position and type(fetched[0].sites[0].occupancy) is Fraction
    assert fetched[1].sites == () and fetched[1].extra is None and fetched[2].extra == []
    assert store.save(CRYSTALS[0]) == sids[0]  # content-id dedup, no second set of rows
    assert _sql_child_row_count(store) in (4, None)


@pytest.mark.parametrize("backend", [CLICKHOUSE_PARAM])
def test_clickhouse_bulk_round_trip(backend: str) -> None:
    with bulk_store(CRYSTALS) as store:
        search = store.searcher()
        crystal = search.variable(Crystal)
        rows = search.results(crystal=crystal)
        assert sorted((row.crystal for row in rows), key=lambda value: value.name) == CRYSTALS
        search = store.searcher()
        crystal = search.variable(Crystal)
        search.add(crystal.sites.symbol == "Si")
        assert [row.crystal.name for row in search.results(crystal=crystal)] == ["a"]


def test_link_and_inline_layouts_agree(store: Any) -> None:
    for crystal in CRYSTALS:
        assert content_id(crystal) == content_id(_linked(crystal))
        store.save(crystal)
        store.save(_linked(crystal))
    inline = store.fetch_many(Crystal, [store.sid_of(crystal) for crystal in CRYSTALS], eager=True)
    linked = store.fetch_many(LinkedCrystal, [store.sid_of(_linked(c)) for c in CRYSTALS], eager=True)
    assert [(c.name, c.sites, c.extra) for c in inline] == [(c.name, c.sites, c.extra) for c in linked]

    if isinstance(store, SqlStore):  # MongoDB serves configured families only: see test_mongo_served_entries_agree
        assert _served(store, StoreEntryProvider, Crystal) == _served(store, StoreEntryProvider, LinkedCrystal)


def _served(store: Any, provider_class: type, cls: type) -> tuple[Any, list[dict[str, Any]]]:
    provider = provider_class(store, {"crystals": cls}, id_of=lambda _type, _sid, value: value.name)
    rows = sorted(provider.records("crystals"), key=lambda row: row["__id"])
    definition = provider.entry_types()["crystals"]
    return sorted(definition.properties), rows


class InlineCrystals:
    type = "crystals"


class LinkedCrystals:
    type = "crystals"


for _family, _record in ((InlineCrystals, Crystal), (LinkedCrystals, LinkedCrystal)):
    register_entry_family(name=f"inline-child-rows-{_family.__name__}", family=f"{__name__}:{_family.__name__}")
    register_entry_record(
        name=f"inline-child-rows-{_record.__name__}",
        family=f"inline-child-rows-{_family.__name__}",
        record=f"{__name__}:{_record.__name__}",
    )


def test_mongo_served_entries_agree(request) -> None:
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    client = request.getfixturevalue("mongo_test_client")
    served = []
    for family, record, values in ((InlineCrystals, Crystal, CRYSTALS), (LinkedCrystals, LinkedCrystal, None)):
        name = f"httk_test_{uuid.uuid4().hex}"
        try:
            store = MongoStore(MongoDatabase(client, name), entry_records={family: record})
            for crystal in CRYSTALS:
                store.save(crystal if values is not None else _linked(crystal))
            served.append(_served(store, MongoEntryProvider, record))
        finally:
            client.drop_database(name)
    assert served[0] == served[1]


def test_search_through_inline_rows(store: Any) -> None:
    for crystal in CRYSTALS:
        store.save(crystal)

    def names(build: Any) -> list[str]:
        search = store.searcher()
        crystal = search.variable(Crystal)
        search.add(build(crystal))
        return sorted(row.crystal.name for row in search.results(crystal=crystal))

    assert names(lambda c: c.sites.symbol == "O") == ["a", "c"]
    assert names(lambda c: c.sites.occupancy < Fraction(1, 2)) == ["a"]
    assert names(lambda c: ~c.sites.symbol.has_any("O")) == ["b", "d"]
    assert names(lambda c: c.extra.symbol == "O") == ["a"]

    def same_site(c: Any) -> Any:
        site = c.sites
        return (site.symbol == "O") & (site.x < 0.5)

    assert names(same_site) == []
    assert names(lambda c: (c.sites.symbol == "O") & (c.sites.x < 0.5)) == ["a"]


def test_federation_through_inline_rows() -> None:
    with Backend.sqlite() as first_database, Backend.sqlite() as second_database:
        first = SqlStore(first_database, entry_records={})
        second = SqlStore(second_database, entry_records={})
        for index, crystal in enumerate(CRYSTALS):
            (first if index % 2 else second).save(crystal)
        federation = FederatedStore({"first": first, "second": second})
        search = federation.searcher()
        crystal = search.variable(Crystal)
        search.add(crystal.sites.symbol == "O")
        assert sorted(row.crystal.name for row in search.results(crystal=crystal)) == ["a", "c"]


def test_stored_property_scope_reaches_inline_rows(database: Backend) -> None:
    store = SqlStore(database, entry_records={})
    for crystal in CRYSTALS:
        store.save(crystal)

    def names(build: Any) -> list[str]:
        search = store.searcher()
        crystal = search.variable(Crystal)
        search.add(build(_SqlQueryContext(search, crystal)))
        return sorted(row.crystal.name for row in search.results(crystal=crystal))

    def heavy_sites(ctx: Any, count: int) -> Any:
        sites = ctx.scope("sites")
        heavy = ctx.filtered(sites, ctx.not_(ctx.equal(sites.field("symbol"), ctx.constant("O"))))
        return ctx.equal(ctx.count(heavy), ctx.constant(count))

    assert names(lambda ctx: heavy_sites(ctx, 1)) == ["a", "d"]
    assert names(lambda ctx: heavy_sites(ctx, 0)) == ["b", "c"]

    def aligned(ctx: Any) -> Any:
        sites, extra = ctx.aligned(ctx.scope("sites"), ctx.scope("extra"))
        return ctx.exists(sites, ctx.equal(extra.field("symbol"), ctx.constant("O")))

    assert names(aligned) == ["a"]

    def lineage(ctx: Any) -> Any:
        return ctx.scope("sites").field("logical_id")

    with pytest.raises(StoredPropertySqlConfigurationError, match="only available on parent scopes"):
        names(lineage)


def test_replace_keeps_revision_rows(store: Any) -> None:
    first_sid = store.save(Crystal("r", (SI, O)))
    first = store.fetch(Crystal, first_sid)
    second = store.fetch(Crystal, store.replace(first, Crystal("r", (O,))))
    assert store.fetch(Crystal, first_sid).sites == (SI, O)
    assert second.sites == (O,)
    assert [revision.sites for revision in store.history(second)] == [(SI, O), (O,)]
    assert _sql_child_row_count(store) in (3, None)


def test_mongo_document_layout_and_scopes(request) -> None:
    from httk.store.backend.mongo import MongoDatabase, MongoStore
    from httk.store.backend.mongo.evaluator import evaluate
    from httk.store.backend.mongo.stored_properties import (
        MongoStoredPropertyConfigurationError,
        _MongoQueryContext,
    )

    client = request.getfixturevalue("mongo_test_client")
    name = f"httk_test_{uuid.uuid4().hex}"
    try:
        store = MongoStore(MongoDatabase(client, name), entry_records={})
        sids = [store.save(crystal) for crystal in CRYSTALS]
        collection = client[name]["crystal"]
        document = collection.find_one({"_id": sids[0]})
        assert document is not None
        site = document["f"]["sites"][0]
        assert site == {
            "symbol": "Si",
            "x": 0.25,
            "occupancy": 1 / 3,
            "occupancy_exact": "1/3",
            "position_0": 0.5,
            "position_1": 0.0,
            "position_2": 0.25,
            "position_exact": site["position_exact"],
            "moment_0": 0.5,
            "moment_1": -1.5,
        }
        assert "occupancy" not in document["f"]["sites"][1]  # None is an absent key, as on a parent document
        assert "extra" not in collection.find_one({"_id": sids[1]})["f"]  # None field; () is an empty array
        assert collection.find_one({"_id": sids[2]})["f"]["extra"] == []
        assert "ixe_crystal_sites_symbol" in collection.index_information()
        assert store.fsck().violations == ()  # embedded elements: nothing to orphan

        ctx = _MongoQueryContext(Crystal, store)
        sites = ctx.scope("sites")
        heavy = ctx.count(ctx.filtered(sites, ctx.not_(ctx.equal(sites.field("symbol"), ctx.constant("O")))))
        predicate = ctx.equal(heavy, ctx.constant(1))
        assert [c.name for c in CRYSTALS if evaluate(predicate, store.fetch(Crystal, store.sid_of(c)))] == ["a", "d"]
        sodium = ctx.exists(sites, ctx.equal(sites.field("symbol"), ctx.constant("Na")))
        assert [c.name for c in CRYSTALS if evaluate(sodium, c)] == ["d"]
        with pytest.raises(MongoStoredPropertyConfigurationError, match="only available on parent scopes"):
            ctx.scope("sites").field("logical_id")
    finally:
        client.drop_database(name)


@dataclass(frozen=True)
class Holder:
    crystal: Crystal


def test_fsck_removes_orphan_and_unreachable_inline_rows() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.save(Holder(CRYSTALS[0]))
        holder = store._table("holder")
        child = store._table("crystal_sites")
        with store._database.engine.begin() as connection:
            row = dict(connection.execute(sqlalchemy.select(child)).mappings().first() or {})
            connection.execute(sqlalchemy.insert(child).values({**row, "crystal_sid": 999}))
        report = store.fsck(known_types=(Holder,))
        assert report.violations == ()
        assert report.tables["crystal_sites"].deleted == 1
        assert len(_child_rows(store, "crystal_sites", "sites")) == 2

        with store._database.engine.begin() as connection:
            connection.execute(sqlalchemy.delete(holder))  # the nested crystal becomes unreachable
        report = store.fsck(known_types=(Holder,))
        assert report.tables["crystal"].deleted == 1
        assert report.tables["crystal_sites"].deleted == 2
        assert report.tables["crystal_extra"].deleted == 1


@pytest.mark.parametrize(
    ("mode", "options"),
    [
        ("sequential", {"finalize": "parity"}),
        ("deferred", {"finalize": "deferred"}),
        ("parallel", {"workers": 2}),
    ],
)
def test_bulk_ingest_matches_save(database: Backend, mode: str, options: dict[str, Any]) -> None:
    tables = (("crystal_sites", "sites"), ("crystal_extra", "extra"))
    with Backend.sqlite() as saved_database:
        saved = SqlStore(saved_database, entry_records={})
        for crystal in CRYSTALS:
            saved.save(crystal)
        expected = {name: _child_rows(saved, name, field) for name, field in tables}
    bulk = SqlStore(database, entry_records={})
    with bulk.bulk_ingest(chunk_size=2, **options) as ingest:
        for crystal in CRYSTALS + CRYSTALS[:2]:
            ingest.save(crystal)
    for name, field in tables:
        assert _child_rows(bulk, name, field) == expected[name], mode
    search = bulk.searcher()
    crystal = search.variable(Crystal)
    assert sorted((row.crystal for row in search.results(crystal=crystal)), key=lambda c: c.name) == CRYSTALS


def test_fingerprint_distinguishes_layouts() -> None:
    @dataclass(frozen=True)
    class Plain:
        __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="fp", identity_name="fp")
        sites: tuple[Site, ...]

    @dataclass(frozen=True)
    class Inlined:
        __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="fp", identity_name="fp")
        sites: Annotated[tuple[Site, ...], Inline()]

    plain = _table_fingerprint(resolve_schema(Plain))
    inlined = _table_fingerprint(resolve_schema(Inlined))
    assert "inline" not in plain["fields"]["sites"]["child"]  # link-table fingerprints are unchanged
    inline_doc = inlined["fields"]["sites"]["child"]["inline"]
    assert sorted(inline_doc["fields"]) == ["moment", "occupancy", "position", "symbol", "x"]
    assert inline_doc["identity_name"] == "test_inline_child_rows.Site"

    def document(table: dict[str, Any]) -> str:
        return json.dumps({"entry_id_tables": [], "tables": {"fp": table}}, sort_keys=True)

    reason = classify_schema_upgrade(document(plain), document(inlined))
    assert isinstance(reason, str) and "changed field 'sites'" in reason


@dataclass(frozen=True)
class _Ref:
    value: int


@dataclass(frozen=True)
class _Nested:
    inner: Annotated[tuple[Site, ...], Inline()]


@dataclass(frozen=True)
class _WithProperty:
    x: int

    @stored_property
    def doubled(self) -> int:
        return 2 * self.x


def _refused(annotation: Any) -> None:
    namespace: dict[str, Any] = {"__annotations__": {"field": annotation}}
    cls = dataclass(frozen=True)(type("Refused", (), namespace))
    with pytest.raises(SchemaError):
        resolve_schema(cls)


def _element(**annotations: Any) -> type:
    return dataclass(frozen=True)(type("Elem", (), {"__annotations__": annotations}))


@pytest.mark.parametrize(
    "annotation",
    [
        Annotated[int, Inline()],
        Annotated[Site, Inline()],
        Annotated[tuple[int, ...], Inline()],
        Annotated[tuple[Site | None, ...], Inline()],
        Annotated[tuple[Site, Site], Inline()],
        Annotated[FracVector, Shape(0, 3), Inline()],
        Annotated[tuple[_element(ref=_Ref), ...], Inline()],
        Annotated[tuple[_element(items=tuple[int, ...]), ...], Inline()],
        Annotated[tuple[_Nested, ...], Inline()],
        Annotated[tuple[_WithProperty, ...], Inline()],
        Annotated[tuple[_element(key=Annotated[str, Unique()]), ...], Inline()],
        Annotated[tuple[_element(url=Annotated[str | None, IdentitySkip(), KeepStored()]), ...], Inline()],
        Annotated[tuple[_element(refused_sid=int), ...], Inline()],
        Annotated[tuple[_element(field_index=int), ...], Inline()],
        Annotated[tuple[Site, ...], Inline(), Indexed()],
        Annotated[tuple[Site, ...], Inline(), Unique()],
        Annotated[tuple[Site, ...], Inline(), KeepStored()],
    ],
)
def test_schema_refusals(annotation: Any) -> None:
    _refused(annotation)


def test_mutable_element_refused() -> None:
    @dataclass
    class Mutable:
        x: int

    _refused(Annotated[tuple[Mutable, ...], Inline()])


def test_none_element_rejected_on_save() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        with pytest.raises(ValueError, match=r"sites\[1\] cannot be None"):
            store.save(Crystal("bad", (O, None)))  # type: ignore[arg-type]


@dataclass(frozen=True)
class NotedSite:
    v: int
    note: Annotated[str | None, IdentitySkip()] = None


@dataclass(frozen=True)
class Noted:
    sites: Annotated[tuple[NotedSite, ...], Inline()]


def test_identity_skipped_element_fields_are_compared_on_dedup(store: Any) -> None:
    sid = store.save(Noted((NotedSite(1, "a"),)))
    assert store.save(Noted((NotedSite(1, "a"),))) == sid
    with pytest.raises(EntryMetadataConflictError, match=r"for Noted\.sites\[0\]\.note: stored 'a', received 'b'"):
        store.save(Noted((NotedSite(1, "b"),)))


def test_bulk_identity_skipped_element_fields() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        conflict = pytest.raises(EntryMetadataConflictError, match=r"Noted\.sites\[0\]\.note")
        with conflict, store.bulk_ingest(finalize="parity") as ingest:
            ingest.save(Noted((NotedSite(1, "a"),)))
            ingest.save(Noted((NotedSite(1, "b"),)))
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        refused = pytest.raises(ValueError, match="identity-excluded child sequence")
        with refused, store.bulk_ingest(finalize="deferred") as ingest:
            ingest.save(Noted((NotedSite(1, "a"),)))


@dataclass(frozen=True)
class CheckedSite:
    v: int

    @classmethod
    def __httk_validate__(cls, record: "CheckedSite") -> None:
        if record.v < 0:
            raise ValueError("negative site")


@dataclass(frozen=True)
class Checked:
    sites: Annotated[tuple[CheckedSite, ...], Inline()]


def test_validating_mongo_store_validates_inline_elements(request) -> None:
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    client = request.getfixturevalue("mongo_test_client")
    name = f"httk_test_{uuid.uuid4().hex}"
    try:
        store = MongoStore(MongoDatabase(client, name), entry_records={}, validate_records=True)
        with pytest.raises(ValueError, match="negative site"):
            store.save(Checked((CheckedSite(1), CheckedSite(-1))))
    finally:
        client.drop_database(name)


@pytest.mark.parametrize("path", ["save", "parity", "parallel"])
def test_validating_store_validates_inline_elements(path: str) -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={}, validate_records=True)
        bad = Checked((CheckedSite(1), CheckedSite(-1)))
        with pytest.raises(ValueError, match="negative site"):
            if path == "save":
                store.save(bad)
            else:
                options: dict[str, Any] = {"finalize": "parity"} if path == "parity" else {"workers": 2}
                with store.bulk_ingest(**options) as ingest:
                    ingest.save(bad)
        assert SqlStore(database, entry_records={}).save(bad) > 0  # not validated unless asked


@dataclass(frozen=True)
class IndexedSite:
    symbol: Annotated[str, Indexed()]


@dataclass(frozen=True)
class IndexedPair:
    sites_symbol: Annotated[str, Indexed()]
    sites: Annotated[tuple[IndexedSite, ...], Inline()]


@pytest.mark.parametrize("backend", ["sqlite", "duckdb", "mongo"])
def test_parent_and_element_index_names_cannot_collide(request, backend: str) -> None:
    # P.sites_symbol and P.sites[].symbol both read "sites_symbol"; the element index has its own "ixe" prefix.
    record = IndexedPair("Si", (IndexedSite("Si"), IndexedSite("O")))
    names = {"ix_indexed_pair_sites_symbol", "ixe_indexed_pair_sites_symbol"}
    if backend == "mongo":
        from httk.store.backend.mongo import MongoDatabase, MongoStore

        client = request.getfixturevalue("mongo_test_client")
        name = f"httk_test_{uuid.uuid4().hex}"
        try:
            store = MongoStore(MongoDatabase(client, name), entry_records={})
            assert store.fetch(IndexedPair, store.save(record)) == record
            assert names <= set(client[name]["indexed_pair"].index_information())
        finally:
            client.drop_database(name)
        return
    with _sql_database(backend) as database:
        store = SqlStore(database, entry_records={})
        assert store.fetch(IndexedPair, store.save(record)) == record
        built = {
            index.name for table in ("indexed_pair", "indexed_pair_sites") for index in store._table(table).indexes
        }
        assert names <= built


@dataclass(frozen=True)
class SkipSite:
    v: int
    note: Annotated[str | None, IdentitySkip()] = None
    skipped: Annotated[str, Skip()] = "dflt"


@dataclass(frozen=True)
class SkipInline:
    sites: Annotated[tuple[SkipSite, ...], Inline()]


@dataclass(frozen=True)
class SkipLinked:
    sites: tuple[SkipSite, ...]


@pytest.mark.parametrize("parent", [SkipInline, SkipLinked])
def test_unstored_element_fields_never_conflict(store: Any, parent: type) -> None:
    sid = store.save(parent((SkipSite(1, "a", "x"),)))
    assert store.save(parent((SkipSite(1, "a", "y"),))) == sid  # a Skip field is not stored, so not compared
    with pytest.raises(EntryMetadataConflictError, match="note"):
        store.save(parent((SkipSite(1, "b", "x"),)))


@pytest.mark.parametrize("parent", [SkipInline, SkipLinked])
def test_bulk_unstored_element_fields_never_conflict(parent: type) -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        with store.bulk_ingest(finalize="parity") as ingest:
            ingest.save(parent((SkipSite(1, "a", "x"),)))
            ingest.save(parent((SkipSite(1, "a", "y"),)))
        search = store.searcher()
        search.variable(parent)
        assert search.count() == 1


@dataclass(frozen=True)
class TreeNode:
    label: str
    children: Annotated[tuple["TreeNode", ...], Inline()] = ()


@dataclass(frozen=True)
class MutualA:
    bs: Annotated[tuple["MutualB", ...], Inline()]


@dataclass(frozen=True)
class MutualB:
    as_: Annotated[tuple[MutualA, ...], Inline()]


@pytest.mark.parametrize("cls", [TreeNode, MutualA, MutualB])
def test_recursive_inline_is_refused(cls: type) -> None:
    with pytest.raises(SchemaError, match="cannot itself hold an Inline field"):
        resolve_schema(cls)


def test_inline_field_cannot_be_sorted_or_projected(store: Any) -> None:
    store.save(CRYSTALS[0])
    search = store.searcher()
    crystal = search.variable(Crystal)
    with pytest.raises(UnsupportedQueryError, match="variable-length"):
        search.add_sort(crystal.sites)
    with pytest.raises(UnsupportedQueryError, match="variable-length"):
        search.results(sites=crystal.sites)
