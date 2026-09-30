"""Query existing SQL tables in place with ``TableStore``.

The fixture tables are created with plain SQLAlchemy: they stand in for a
user's pre-existing database, not for httk-store's own layout.
"""

import datetime
import decimal
from collections.abc import Iterator, Mapping

import pytest
import sqlalchemy
from postgres_support import POSTGRES_PARAM, IsolatedPostgresDatabase

from httk.store.backend.sql import tables
from httk.store.backend.sql.tables import DelimitedColumn, ListTable, TableSource, TableStore
from httk.store.query import MultipleResultsError, NoResultError, UnsupportedQueryError
from httk.store.query.conformance import check_query_conformance, conformance_rows

pytestmark = pytest.mark.filterwarnings("ignore:duckdb-engine doesn't yet support reflection on indices")


@pytest.fixture(params=["sqlite", "duckdb", POSTGRES_PARAM])
def engine(request: pytest.FixtureRequest, tmp_path) -> Iterator[sqlalchemy.Engine]:
    isolated = None
    if request.param == "sqlite":
        url: str | sqlalchemy.URL = f"sqlite:///{tmp_path / 'existing.sqlite'}"
    elif request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        url = f"duckdb:///{tmp_path / 'existing.duckdb'}"
    else:
        isolated = IsolatedPostgresDatabase()
        url = isolated.uri
    engine = sqlalchemy.create_engine(url)
    try:
        yield engine
    finally:
        engine.dispose()
        if isolated is not None:
            isolated.drop()


@pytest.fixture
def sqlite_engine(tmp_path) -> Iterator[sqlalchemy.Engine]:
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'existing.sqlite'}")
    yield engine
    engine.dispose()


def _key(engine: sqlalchemy.Engine) -> str | None:
    # duckdb_engine does not reflect primary keys, so DuckDB tables name their key.
    return "rid" if engine.dialect.name == "duckdb" else None


def _scalar_columns() -> list[sqlalchemy.Column[object]]:
    return [
        sqlalchemy.Column("rid", sqlalchemy.String(8), primary_key=True),
        sqlalchemy.Column("name", sqlalchemy.String(32)),
        sqlalchemy.Column("n", sqlalchemy.Integer),
        sqlalchemy.Column("x", sqlalchemy.Float),
    ]


def _create(engine: sqlalchemy.Engine, table: sqlalchemy.Table, rows: list[dict[str, object]]) -> None:
    table.metadata.create_all(engine)
    if rows:
        with engine.begin() as connection:
            connection.execute(table.insert(), rows)


def _list_table_store(engine: sqlalchemy.Engine) -> TableStore:
    metadata = sqlalchemy.MetaData()
    parent = sqlalchemy.Table("conf", metadata, *_scalar_columns())
    child = sqlalchemy.Table(
        "conf_tags",
        metadata,
        sqlalchemy.Column("rid", sqlalchemy.String(8)),
        sqlalchemy.Column("tag", sqlalchemy.String(8)),
    )
    rows = conformance_rows()
    _create(engine, parent, [{k: v for k, v in row.items() if k != "tags"} for row in rows])
    _create(engine, child, [{"rid": row["rid"], "tag": tag} for row in rows for tag in row["tags"] or ()])
    lists = {"tags": ListTable("conf_tags", "rid", "tag")}
    return TableStore(engine, {"conf": TableSource("conf", key=_key(engine), lists=lists)})


def test_list_table_layout_conforms(engine):
    check_query_conformance(_list_table_store(engine), "conf")


def test_delimited_column_layout_conforms(engine):
    parent = sqlalchemy.Table(
        "conf_csv", sqlalchemy.MetaData(), *_scalar_columns(), sqlalchemy.Column("tags_csv", sqlalchemy.Text)
    )
    rows = [
        {
            **{k: v for k, v in row.items() if k != "tags"},
            "tags_csv": None if row["tags"] is None else ",".join(row["tags"]),
        }
        for row in conformance_rows()
    ]
    _create(engine, parent, rows)
    source = TableSource("conf_csv", key=_key(engine), lists={"tags": DelimitedColumn("tags_csv")})
    check_query_conformance(TableStore(engine, {"conf": source}), "conf")


def test_rows_are_mappings_hydrated_with_one_child_query_per_page(engine):
    store = _list_table_store(engine)
    statements: list[str] = []
    sqlalchemy.event.listen(engine, "before_cursor_execute", lambda *args: statements.append(args[2]))
    searcher = store.searcher()
    v = searcher.variable("conf")
    searcher.add_sort(v.rid, False)
    rows = [row.row for row in searcher.results(row=v)]
    assert sum("conf_tags" in statement for statement in statements) == 1
    assert all(isinstance(row, Mapping) for row in rows)
    assert dict(rows[0]) == {"rid": "a", "name": "alpha", "n": 1, "x": 0.5, "tags": ("p", "q")}
    assert [row["tags"] for row in rows] == [("p", "q"), ("p",), (), (), ("q",), ("r",)]
    with pytest.raises(TypeError):
        rows[0]["name"] = "changed"  # type: ignore[index]


def test_hydration_batches_parent_keys(sqlite_engine, monkeypatch):
    monkeypatch.setattr(tables, "_HYDRATION_BATCH_SIZE", 2)
    store = _list_table_store(sqlite_engine)
    statements: list[str] = []
    sqlalchemy.event.listen(sqlite_engine, "before_cursor_execute", lambda *args: statements.append(args[2]))
    searcher = store.searcher()
    v = searcher.variable("conf")
    searcher.add_sort(v.rid, False)
    tags = list(searcher.results(tags=v.tags).scalars())
    assert tags == [("p", "q"), ("p",), (), (), ("q",), ("r",)]
    assert sum("conf_tags" in statement for statement in statements) == 3  # ceil(6 / 2)


def test_field_outputs_and_result_set_access(engine):
    store = _list_table_store(engine)
    searcher = store.searcher()
    v = searcher.variable("conf")
    searcher.add(v.rid == "a")
    result = searcher.results(rid=v.rid, tags=v.tags)
    row = result.one()
    assert (row[0], row["tags"], row.rid) == ("a", ("p", "q"), "a")
    assert result.first() == row
    assert list(result.scalars("tags")) == [("p", "q")]
    with pytest.raises(ValueError, match="exactly one output"):
        result.scalars()
    empty = store.searcher()
    empty.add(empty.variable("conf").rid == "zzz")
    with pytest.raises(NoResultError):
        empty.results(row=empty.variable("conf")).one()
    many = store.searcher()
    with pytest.raises(MultipleResultsError):
        many.results(rid=many.variable("conf").rid).one()


@pytest.mark.filterwarnings("ignore:Dialect sqlite\\+pysqlite does \\*not\\* support Decimal")
def test_raw_values_are_exact_and_timestamp_literals_parse(engine):
    created = datetime.datetime(2024, 1, 2, 3, 4, 5)  # noqa: DTZ001  (a naive DATETIME column)
    table = sqlalchemy.Table(
        "events",
        sqlalchemy.MetaData(),
        sqlalchemy.Column("rid", sqlalchemy.String(8), primary_key=True),
        sqlalchemy.Column("created", sqlalchemy.DateTime),
        sqlalchemy.Column("amount", sqlalchemy.Numeric(10, 2)),
    )
    _create(
        engine,
        table,
        [
            {"rid": "a", "created": created, "amount": decimal.Decimal("1.25")},
            {"rid": "b", "created": created + datetime.timedelta(days=1), "amount": None},
        ],
    )
    store = TableStore(engine, {"events": TableSource("events", key=_key(engine))})
    searcher = store.searcher()
    v = searcher.variable("events")
    searcher.add(v.created == "2024-01-02T03:04:05Z")
    row = searcher.results(row=v).one().row
    assert row["created"] == created and isinstance(row["created"], datetime.datetime)
    assert row["amount"] == decimal.Decimal("1.25") and isinstance(row["amount"], decimal.Decimal)
    later = store.searcher()
    later.add(later.variable("events").created > "2024-01-02T06:00:00+02:00")
    assert [r.rid for r in later.results(rid=later.variable("events").rid)] == ["b"]
    with pytest.raises(ValueError, match="'created'"):
        _ = store.searcher().variable("events").created < "yesterday"


def test_integer_columns_accept_string_literals(engine):
    table = sqlalchemy.Table(
        "materials",
        sqlalchemy.MetaData(),
        sqlalchemy.Column("mat_id", sqlalchemy.Integer, primary_key=True, autoincrement=False),
        sqlalchemy.Column("nsites", sqlalchemy.Integer),
    )
    _create(engine, table, [{"mat_id": 1, "nsites": 2}, {"mat_id": 3, "nsites": None}, {"mat_id": 12, "nsites": 5}])
    key = "mat_id" if engine.dialect.name == "duckdb" else None
    store = TableStore(engine, {"m": TableSource("materials", key=key)})

    def ids(build) -> list[int]:
        searcher = store.searcher()
        v = searcher.variable("m")
        searcher.add(build(v))
        searcher.add_sort(v.mat_id, False)
        return list(searcher.results(mat_id=v.mat_id).scalars())

    assert ids(lambda v: v.mat_id == "3") == [3]
    assert ids(lambda v: v.mat_id == "abc") == []
    assert ids(lambda v: v.mat_id != "abc") == [1, 3, 12]
    assert ids(lambda v: v.nsites != "abc") == [1, 12]
    assert ids(lambda v: ~(v.nsites == "abc")) == [1, 12]
    assert ids(lambda v: ~(v.nsites != "abc")) == []
    assert ids(lambda v: ~(v.mat_id == "abc")) == [1, 3, 12]
    assert ids(lambda v: v.mat_id >= "3") == [3, 12]
    assert ids(lambda v: v.mat_id < "abc") == [1, 3, 12]
    assert ids(lambda v: v.mat_id.is_in("1", "x")) == [1]
    assert ids(lambda v: v.mat_id.is_in("x")) == []
    assert ids(lambda v: v.mat_id.startswith("1")) == [1, 12]


def test_delimited_has_only_handles_duplicates(engine):
    table = sqlalchemy.Table(
        "lists",
        sqlalchemy.MetaData(),
        sqlalchemy.Column("rid", sqlalchemy.String(8), primary_key=True),
        sqlalchemy.Column("tags_csv", sqlalchemy.Text),
    )
    values = {"a": "p,p", "b": "p,q", "c": "p,p,p,p", "d": None, "e": "", "f": "q,p,p"}
    _create(engine, table, [{"rid": rid, "tags_csv": value} for rid, value in values.items()])
    source = TableSource("lists", key=_key(engine), lists={"tags": DelimitedColumn("tags_csv")})
    store = TableStore(engine, {"lists": source})

    def ids(build) -> list[str]:
        searcher = store.searcher()
        v = searcher.variable("lists")
        searcher.add(build(v))
        return sorted(searcher.results(rid=v.rid).scalars())

    assert ids(lambda v: v.tags.has_only("p")) == ["a", "c", "d", "e"]
    assert ids(lambda v: v.tags.has_only("p", "q")) == ["a", "b", "c", "d", "e", "f"]
    assert ids(lambda v: v.tags.has_any("q")) == ["b", "f"]
    assert ids(lambda v: ~v.tags.has("p")) == ["d", "e"]
    searcher = store.searcher()
    v = searcher.variable("lists")
    searcher.add(v.rid == "f")
    assert searcher.results(row=v).one().row["tags"] == ("p", "p", "q")
    with pytest.raises(ValueError, match="None is not a valid member"):
        v.tags.has_only(None)
    with pytest.raises(ValueError, match="separator"):
        v.tags.has_any("p,q")


def test_construction_and_query_errors(sqlite_engine):
    engine = sqlite_engine
    metadata = sqlalchemy.MetaData()
    sqlalchemy.Table(
        "items",
        metadata,
        sqlalchemy.Column("rid", sqlalchemy.String(8), primary_key=True),
        sqlalchemy.Column("name", sqlalchemy.String(8)),
    )
    sqlalchemy.Table(
        "tags", metadata, sqlalchemy.Column("rid", sqlalchemy.String(8)), sqlalchemy.Column("tag", sqlalchemy.String(8))
    )
    sqlalchemy.Table("keyless", metadata, sqlalchemy.Column("name", sqlalchemy.String(8)))
    metadata.create_all(engine)
    tag_list = {"tags": ListTable("tags", "rid", "tag")}

    with pytest.raises(ValueError, match="'nope' does not exist"):
        TableStore(engine, {"t": TableSource("nope")})
    with pytest.raises(ValueError, match="'missing' does not exist"):
        TableStore(engine, {"t": TableSource("items", lists={"tags": ListTable("missing", "rid", "tag")})})
    with pytest.raises(ValueError, match="'value' is not in 'tags'"):
        TableStore(engine, {"t": TableSource("items", lists={"tags": ListTable("tags", "rid", "value")})})
    with pytest.raises(ValueError, match="'csv' is not in 'items'"):
        TableStore(engine, {"t": TableSource("items", lists={"tags": DelimitedColumn("csv")})})
    with pytest.raises(ValueError, match="collides"):
        TableStore(engine, {"t": TableSource("items", lists={"name": DelimitedColumn("name")})})
    with pytest.raises(ValueError, match="no single-column primary key"):
        TableStore(engine, {"t": TableSource("keyless", lists={"tags": DelimitedColumn("name")})})
    with pytest.raises(ValueError, match="key column 'id'"):
        TableStore(engine, {"t": TableSource("items", key="id")})
    with pytest.raises(ValueError, match="separator"):
        DelimitedColumn("name", "")

    store = TableStore(engine, {"t": TableSource("items", lists=tag_list), "k": TableSource("keyless")})
    with pytest.raises(ValueError, match="historic"):
        store.searcher(as_of=1)
    searcher = store.searcher()
    with pytest.raises(ValueError, match="unknown TableStore target"):
        searcher.variable("items")
    v = searcher.variable("t")
    with pytest.raises(AttributeError, match="available fields: rid, name, tags"):
        _ = v.bogus
    with pytest.raises(AttributeError):
        _ = v.name.doi
    with pytest.raises(UnsupportedQueryError):
        searcher.add_sort(v.tags, False)
    with pytest.raises(UnsupportedQueryError):
        searcher.variable("k")
    with pytest.raises(TypeError, match="'t.name' is a scalar"):
        v.name.has("x")
    with pytest.raises(TypeError, match="'t.tags' is a list"):
        _ = v.tags == "p"
    with pytest.raises(TypeError, match="'t.tags' is a list"):
        v.tags.contains("p")
    with pytest.raises(ValueError, match="at least one output"):
        searcher.results()
    keyless = store.searcher()
    keyless.set_limit(1)
    assert len(keyless.results(row=keyless.variable("k"))) == 0  # a keyless target without lists is queryable


def test_owned_engine_is_disposed_and_passed_engine_is_not(tmp_path, sqlite_engine):
    path = tmp_path / "existing.sqlite"
    table = sqlalchemy.Table(
        "items", sqlalchemy.MetaData(), sqlalchemy.Column("rid", sqlalchemy.String(8), primary_key=True)
    )
    _create(sqlite_engine, table, [{"rid": "a"}])
    disposed: list[sqlalchemy.Engine] = []

    with TableStore(f"sqlite:///{path}", {"t": TableSource("items")}) as owned:
        sqlalchemy.event.listen(owned._engine, "engine_disposed", disposed.append)
        assert owned.searcher().variable("t") is not None
    assert disposed == [owned._engine]

    sqlalchemy.event.listen(sqlite_engine, "engine_disposed", disposed.append)
    borrowed = TableStore(sqlite_engine, {"t": TableSource("items")})
    borrowed.close()
    assert disposed == [owned._engine]
    searcher = borrowed.searcher()
    searcher.variable("t")
    assert searcher.count() == 1


def test_count_ignores_paging_and_offset_accumulates(engine):
    store = _list_table_store(engine)
    searcher = store.searcher()
    v = searcher.variable("conf")
    searcher.set_limit(2)
    searcher.add_offset(1)
    searcher.add_offset(1)
    assert searcher.offset == 2
    assert searcher.count() == 6
    assert list(searcher.results(rid=v.rid).scalars()) == ["c", "d"]
    searcher.set_limit(-1)
    assert list(searcher.results(rid=v.rid).scalars()) == ["c", "d", "e", "f"]


def test_lazy_root_exports():
    import httk.store

    for name in ("TableStore", "TableSource", "ListTable", "DelimitedColumn"):
        assert getattr(httk.store, name) is getattr(tables, name)
        assert name in httk.store.__all__
    assert "TableSearcher" not in httk.store.__all__
