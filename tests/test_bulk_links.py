"""Weak links through ``bulk_ingest`` (``save(..., links=...)``) in every bulk mode.

Bulk saves each link target as a top-level entry, deduplicated like any save, and
resolves the link rows with ordinary ``SqlStore.link`` semantics: at flush on the
serial parity finalizer, once after finalization on the deferred and parallel ones
(which therefore need ``track_sids=True``). Runs over the SQL arms of the shared
``store_factory`` fixture (Mongo has no ``bulk_ingest``) and, with
``HTTK_TEST_CLICKHOUSE_URI``, the ClickHouse deferred arms.
"""

import contextlib
import datetime
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core import ReferenceRecord, Run, RunEntry
from httk.core.curation import ExternalDatabaseEntry, Note, Tag
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, Unique

from httk.store import EntryIdScheme
from httk.store.backend.schema import SchemaError, resolve_schema
from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql.layout import actual_table_names
from httk.store.store_common import EntryIdConflictError

_DATE = datetime.datetime(2026, 11, 25, 12, 0, tzinfo=datetime.UTC)
_LINKS = {link.name: link.table_name for link in resolve_schema(Run).links}


class Job:
    """A domain object stored through ``as_record=JobRecord``."""

    def __init__(self, label: str) -> None:
        self.label = label


@dataclass(frozen=True)
class JobRecord:
    __httk_canonical_source__: ClassVar[type] = Job

    label: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    @classmethod
    def __httk_project__(cls, source: Job) -> Mapping[str, object]:
        return {"label": source.label, "id": None, "immutable_id": None}


@dataclass(frozen=True)
class Envelope:
    """An entry record nesting a ``Run`` (promoted with ``promote=Run``)."""

    source: str
    run: Run
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class JobFamily:
    type = "test-bulk-links-jobs"


class EnvelopeFamily:
    type = "test-bulk-links-envelopes"


register_entry_family(name="test-bulk-links-jobs", family=f"{__name__}:JobFamily")
register_entry_record(name="test-bulk-links-job", family="test-bulk-links-jobs", record=f"{__name__}:JobRecord")
register_entry_family(name="test-bulk-links-envelopes", family=f"{__name__}:EnvelopeFamily")
register_entry_record(
    name="test-bulk-links-envelope", family="test-bulk-links-envelopes", record=f"{__name__}:Envelope"
)


def _store(store_factory, *, extended: bool = False) -> SqlStore:
    records = {RunEntry: Run, JobFamily: JobRecord, EnvelopeFamily: Envelope} if extended else {RunEntry: Run}
    store = store_factory(entry_records=records, entry_ids=EntryIdScheme("httk.test", "1"))
    if not isinstance(store, SqlStore):
        pytest.skip("MongoStore has no bulk_ingest")
    return store


_MODES = {
    "serial-parity": {"finalize": "parity"},
    "serial-deferred": {"finalize": "deferred"},
    "parallel-parity": {"finalize": "parity", "workers": 2},
    "parallel-deferred": {"finalize": "deferred", "workers": 2},
}


def _bulk(store: SqlStore, mode: str, **options):
    """Open ``store``'s bulk ingest in ``mode``, skipping a combination the backend cannot run."""
    selected = _MODES[mode]
    if selected.get("workers", 1) > 1 and store.backend_facts.parallel_shard_format == "parquet":
        pytest.importorskip("pyarrow")
    return store.bulk_ingest(**selected, **options)


def _link_pairs(store: SqlStore, source_cls: type, name: str) -> list[tuple[str, str, int]]:
    """Every row of one link table as ``(source content id, target content id, retracted)``, sorted."""
    source = resolve_schema(source_cls)
    spec = next(link for link in source.links if link.name == name)
    target = resolve_schema(spec.target).table_name
    with store._database.engine.connect() as connection:
        if spec.table_name not in actual_table_names(connection):
            return []
        rows = connection.execute(
            sqlalchemy.text(
                f'SELECT s.content_id, t.content_id, l.retracted FROM "{spec.table_name}" l '
                f'JOIN "{source.table_name}" s ON s.logical_id = l.source_lid '
                f'JOIN "{target}" t ON t.logical_id = l.target_lid'
            )
        )
        return sorted((str(s), str(t), int(r)) for s, t, r in rows)


def _note(text: str) -> Note:
    return Note(_DATE, "httk-test 1.0", text)


def _link_rows(store: SqlStore, name: str) -> list[tuple[int, int, int, int, int]]:
    """``(sid, logical_id, source_lid, target_lid, retracted)`` of one link table, by sid."""
    table = _LINKS[name]
    with store._database.engine.connect() as connection:
        if table not in actual_table_names(connection):
            return []
        return [
            tuple(int(value) for value in row)  # type: ignore[misc]
            for row in connection.execute(
                sqlalchemy.text(
                    f'SELECT sid, logical_id, source_lid, target_lid, retracted FROM "{table}" ORDER BY sid'
                )
            )
        ]


def _count(store: SqlStore, cls: type) -> int:
    searcher = store.searcher()
    searcher.variable(cls)
    return searcher.count()


def _run(store: SqlStore, source_id: str) -> Run:
    searcher = store.searcher()
    v = searcher.variable(Run)
    searcher.add(v.source_id == source_id)
    rows = [row.values[0] for row in searcher.results(r=v)]
    return max(rows, key=lambda row: store.sid_of(row, as_record=Run))


def _labels(store: SqlStore, condition) -> list[str]:
    searcher = store.searcher()
    v = searcher.variable(Run)
    searcher.add(condition(v))
    return sorted(row.values[0].source_id for row in searcher.results(r=v))


def _assert_fsck_clean(store: SqlStore) -> None:
    exclusive = store._database.engine.dialect.name == "duckdb"
    assert store.fsck(repair=False, collect_garbage=False, exclusive=exclusive).violations == ()


@contextlib.contextmanager
def _statements(store: SqlStore) -> Iterator[list[str]]:
    """Collect every SQL statement the store's engine executes inside the block."""
    seen: list[str] = []

    def record(_connection, _cursor, statement, _parameters, _context, _executemany) -> None:
        seen.append(statement)

    engine = store._database.engine
    sqlalchemy.event.listen(engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        sqlalchemy.event.remove(engine, "before_cursor_execute", record)


def _insert_link_rows(store: SqlStore, name: str, rows: list[dict[str, int]]) -> None:
    """Append raw link rows (sids drawn by the database), copying an existing row's store timestamp."""
    table = store._table(_LINKS[name])
    with store._database.engine.begin() as connection:
        extra = {}
        if "store_timestamp" in table.c:
            stamp = connection.execute(sqlalchemy.select(sqlalchemy.func.max(table.c.store_timestamp))).scalar_one()
            extra["store_timestamp"] = int(stamp)
        for row in rows:
            connection.execute(sqlalchemy.insert(table).values({**row, **extra}))


@pytest.mark.parametrize("chunk_size", [100_000, 1])
@pytest.mark.parametrize(
    ("mode", "track_sids"),
    # Only serial parity links without retained sids; the other modes refuse it (see below).
    [(mode, True) for mode in _MODES] + [("serial-parity", False)],
)
def test_fresh_store_links(store_factory, mode, chunk_size, track_sids) -> None:
    store = _store(store_factory)
    shared = _note("Removed a stray loop.")
    with _bulk(store, mode, chunk_size=chunk_size, track_sids=track_sids) as bulk:
        bulk.save(
            Run(source_id="first"),
            links={"notes": [shared, _note("Second note.")], "tags": Tag("source", "cod")},
        )
        bulk.save(Run(source_id="second"), links={"notes": shared, "tags": (Tag("source", "cod"),)})
        # The same pair again, within and across chunks: no further link row.
        bulk.save(Run(source_id="first"), links={"notes": [shared, shared]})
        bulk.save(Run(source_id="third"), links={})
        timestamp = bulk._store_timestamp
    assert _count(store, Run) == 3
    assert _count(store, Note) == 2
    assert _count(store, Tag) == 1
    assert len(_link_rows(store, "notes")) == 3
    assert len(_link_rows(store, "tags")) == 2
    assert all(row[1] == row[0] and row[4] == 0 for row in _link_rows(store, "notes"))
    assert sorted(note.text for note in store.linked(_run(store, "first"), "notes")) == [
        "Removed a stray loop.",
        "Second note.",
    ]
    assert [note.text for note in store.linked(_run(store, "second"), "notes")] == ["Removed a stray loop."]
    assert store.linked(_run(store, "third"), "notes") == ()
    assert _labels(store, lambda v: v.links.tags.key == "source") == ["first", "second"]
    assert _labels(store, lambda v: v.links.notes.text == "Second note.") == ["first"]
    assert timestamp is not None  # the default store keeps store timestamps
    with store._database.engine.connect() as connection:
        stamps = connection.execute(sqlalchemy.text(f'SELECT DISTINCT store_timestamp FROM "{_LINKS["notes"]}"'))
        assert [int(row[0]) for row in stamps] == [timestamp]
    _assert_fsck_clean(store)


@pytest.mark.parametrize("chunk_size", [100_000, 1])
@pytest.mark.parametrize("index_strategy", ["auto", "keep", "rebuild"])
def test_populated_store_link_semantics(store_factory, chunk_size, index_strategy) -> None:
    store = _store(store_factory)
    kept = _note("kept")
    store.link(store.fetch(Run, store.save(Run(source_id="live"))), "notes", store.fetch(Note, store.save(kept)))
    retracted_run = store.fetch(Run, store.save(Run(source_id="retracted")))
    retracted_note = store.fetch(Note, store.save(_note("retracted")))
    store.link(retracted_run, "notes", retracted_note)
    store.unlink(retracted_run, "notes", retracted_note)
    old = store.fetch(Run, store.save(Run(source_id="old")))
    newest_sid = store.replace(old, Run(source_id="new"))
    before = _link_rows(store, "notes")
    (retraction,) = [row for row in before if row[4] == 1]

    with store.bulk_ingest(finalize="parity", chunk_size=chunk_size, index_strategy=index_strategy) as bulk:
        bulk.save(Run(source_id="live"), links={"notes": _note("kept")})  # live pair: no row
        bulk.save(Run(source_id="retracted"), links={"notes": _note("retracted")})  # revival row
        bulk.save(Run(source_id="new"), links={"notes": _note("fresh")})  # replaced source: lineage id
        bulk.save(Run(source_id="brand-new"), links={"notes": _note("kept")})  # new source, existing target
        bulk.save(Run(source_id="brand-new"), links={"external": ExternalDatabaseEntry("cod", None, "1000000")})
    assert _count(store, Note) == 3
    after = _link_rows(store, "notes")
    added = after[len(before) :]
    assert after[: len(before)] == before
    assert len(added) == 3
    (revival,) = [row for row in added if row[2:4] == retraction[2:4]]
    assert revival[1] == retraction[1] and revival[4] == 0
    (replaced,) = [row for row in added if row[2] == store.sid_of(old, as_record=Run)]
    assert replaced[2] != newest_sid
    newest = store.fetch(Run, newest_sid)
    assert [note.text for note in store.linked(newest, "notes")] == ["fresh"]
    assert [note.text for note in store.linked(retracted_run, "notes")] == ["retracted"]
    assert [note.text for note in store.linked(_run(store, "live"), "notes")] == ["kept"]
    assert [note.text for note in store.linked(_run(store, "brand-new"), "notes")] == ["kept"]
    assert [entry.entry_id for entry in store.linked(_run(store, "brand-new"), "external")] == ["1000000"]
    _assert_fsck_clean(store)


@pytest.mark.parametrize(
    ("links", "error"),
    [({"undeclared": _note("x")}, SchemaError), ({"notes": Tag("source", "cod")}, TypeError)],
)
def test_invalid_links_refuse_before_buffering(store_factory, links, error) -> None:
    store = _store(store_factory)
    store.save(Run(source_id="existing"))
    with store.bulk_ingest(finalize="parity") as bulk:
        with pytest.raises(error):
            bulk.save(Run(source_id="refused"), links=links)
        bulk.save(Run(source_id="accepted"))
    assert _labels(store, lambda v: v.source_id != "") == ["accepted", "existing"]
    assert _count(store, Tag) == 0
    assert _count(store, Note) == 0
    with pytest.raises(error), store.bulk_ingest(finalize="parity") as bulk:
        bulk.save(Run(source_id="before"))
        bulk.save(Run(source_id="refused"), links=links)
    assert _labels(store, lambda v: v.source_id != "") == ["accepted", "existing"]


@pytest.mark.parametrize("mode", [*_MODES, "auto"])
def test_untracked_sids_link_support(store_factory, mode) -> None:
    """Without retained sids every mode validates ``links`` first; only serial parity accepts a target."""
    store = _store(store_factory)
    ingest = store.bulk_ingest(track_sids=False) if mode == "auto" else _bulk(store, mode, track_sids=False)
    with ingest as bulk:
        refuses = bulk._deferred or bulk._parallel
        with pytest.raises(SchemaError):
            bulk.save(Run(source_id="undeclared"), links={"undeclared": _note("x")})
        with pytest.raises(TypeError):
            bulk.save(Run(source_id="mistyped"), links={"notes": Tag("source", "cod")})
        if refuses:
            with pytest.raises(ValueError, match="track_sids=True"):
                bulk.save(Run(source_id="refused"), links={"notes": _note("x")})
        else:
            bulk.save(Run(source_id="linked"), links={"notes": _note("x")})
        bulk.save(Run(source_id="empty-mapping"), links={})
        bulk.save(Run(source_id="empty-list"), links={"notes": [], "tags": ()})
    linked = [] if refuses else ["linked"]
    assert _labels(store, lambda v: v.source_id != "") == sorted(["empty-list", "empty-mapping", *linked])
    assert _count(store, Note) == len(linked)
    assert len(_link_rows(store, "notes")) == len(linked)


def _ingest_corpus(bulk_ingest) -> tuple[dict[str, list[tuple[str, str, int]]], list[int]]:
    """Save the shared link corpus through ``bulk_ingest``; return the link pairs by content and the Run sids."""
    shared, tag = _note("shared"), Tag("source", "cod")
    with bulk_ingest as bulk:
        # Several sources share one target; one source has several targets and link names.
        returned = [
            bulk.save(Run(source_id=f"run-{index}"), links={"notes": shared, "tags": tag}) for index in range(5)
        ]
        returned.append(
            bulk.save(
                Run(source_id="many"),
                links={
                    "notes": [shared, _note("own")],
                    "tags": [tag, Tag("k")],
                    "external": [ExternalDatabaseEntry("cod", None, "1000000")],
                },
            )
        )
        # The same pair twice, and a target equal to an entry saved on its own.
        bulk.save(Run(source_id="many"), links={"notes": [_note("own"), _note("own")]})
        bulk.save(_note("separate"))
        bulk.save(Run(source_id="run-0"), links={"notes": _note("separate")})
        bulk.save(_note("separate"))
    sids = [bulk.resolved_sid(Run, value) for value in returned]
    return {name: _link_pairs(bulk._store, Run, name) for name in ("notes", "tags", "external")}, sids


def _assert_corpus(store: SqlStore, pairs: dict[str, list[tuple[str, str, int]]], sids: list[int]) -> None:
    assert len(pairs["notes"]) == 8 and len(pairs["tags"]) == 7 and len(pairs["external"]) == 1
    assert _count(store, Note) == 3
    assert _count(store, Tag) == 2
    assert [row[0] for row in _link_rows(store, "notes")] == list(range(1, 9))  # dense link sids
    assert all(row[1] == row[0] and row[4] == 0 for row in _link_rows(store, "notes"))
    many = store.fetch(Run, sids[-1])
    assert many.source_id == "many"
    assert sorted(note.text for note in store.linked(many, "notes")) == ["own", "shared"]
    assert sorted(tag.key for tag in store.linked(many, "tags")) == ["k", "source"]
    assert sorted(note.text for note in store.linked(_run(store, "run-0"), "notes")) == ["separate", "shared"]
    assert _labels(store, lambda v: v.links.notes.text == "separate") == ["run-0"]


@pytest.mark.parametrize("mode", [mode for mode in _MODES if mode != "serial-parity"])
def test_links_equal_serial_parity(store_factory, mode) -> None:
    """Deferred and parallel ingests store exactly the links (by content) serial parity stores."""
    reference = _store(store_factory)
    expected, _sids = _ingest_corpus(_bulk(reference, "serial-parity", chunk_size=2))
    store = _store(store_factory)
    actual, sids = _ingest_corpus(_bulk(store, mode, chunk_size=2))
    assert actual == expected
    _assert_corpus(store, actual, sids)
    _assert_fsck_clean(store)


@pytest.mark.parametrize("workers", [1, 2])
def test_clickhouse_deferred_links(workers) -> None:
    """ClickHouse's deferred-only bulk path stores the serial parity links."""
    from conftest import clickhouse_test_uri
    from test_clickhouse_bulk import _clickhouse_bulk_database

    pytest.importorskip("pyarrow")
    records = {RunEntry: Run}
    scheme = EntryIdScheme("httk.test", "1")
    reference = SqlStore(Backend.sqlite(), entry_records=records, entry_ids=scheme)
    try:
        expected, _sids = _ingest_corpus(reference.bulk_ingest(finalize="parity", chunk_size=2))
    finally:
        reference._database.dispose()
    with _clickhouse_bulk_database(clickhouse_test_uri()) as database:
        store = SqlStore(database, entry_records=records, entry_ids=scheme)
        actual, sids = _ingest_corpus(store.bulk_ingest(workers=workers, finalize="deferred", chunk_size=2))
        assert actual == expected
        _assert_corpus(store, actual, sids)


@pytest.mark.parametrize("index_strategy", ["auto", "keep", "rebuild"])
def test_parallel_parity_into_precreated_tables(store_factory, index_strategy) -> None:
    """Parallel ingest refuses stored rows, but appends links into pre-created empty tables, indexes intact."""
    store = _store(store_factory)
    if store._database.engine.dialect.name == "duckdb":
        pytest.skip("parallel DuckDB ingest refuses pre-created tables")
    # Run brings its link tables; content-id repeats are covered by test_db_bulk_parallel.
    store.ensure_tables(Run)
    with _bulk(store, "parallel-parity", index_strategy=index_strategy) as bulk:
        bulk.save(Run(source_id="a"), links={"notes": [_note("n"), _note("m")]})
        bulk.save(Run(source_id="b"), links={"notes": _note("n")})
        assert _LINKS["notes"] in bulk._preexisting
    assert len(_link_rows(store, "notes")) == 3
    assert sorted(note.text for note in store.linked(_run(store, "a"), "notes")) == ["m", "n"]
    link_table = store._table(_LINKS["notes"])
    with store._database.engine.connect() as connection:
        present = {index["name"] for index in sqlalchemy.inspect(connection).get_indexes(link_table.name)}
    assert {index.name for index in link_table.indexes} <= present
    (other,) = [note for note in store.linked(_run(store, "a"), "notes") if note.text == "m"]
    store.link(_run(store, "b"), "notes", other)  # draws its sid past the bulk-assigned ones
    assert [row[0] for row in _link_rows(store, "notes")] == [1, 2, 3, 4]
    _assert_fsck_clean(store)


def test_exception_rolls_back_links_and_targets(store_factory) -> None:
    store = _store(store_factory)
    store.link(store.fetch(Run, store.save(Run(source_id="a"))), "notes", store.fetch(Note, store.save(_note("a"))))
    before = _link_rows(store, "notes")
    with pytest.raises(RuntimeError, match="abort"), store.bulk_ingest(finalize="parity", chunk_size=1) as bulk:
        bulk.save(Run(source_id="a"), links={"notes": [_note("a"), _note("new")]})
        bulk.save(Run(source_id="b"), links={"notes": _note("other"), "tags": Tag("k")})
        raise RuntimeError("abort")
    assert _link_rows(store, "notes") == before
    assert _link_rows(store, "tags") == []
    assert _count(store, Note) == 1
    assert _count(store, Tag) == 0
    assert _labels(store, lambda v: v.source_id != "") == ["a"]
    assert [note.text for note in store.linked(_run(store, "a"), "notes")] == ["a"]


def test_failing_target_records_no_links(store_factory) -> None:
    store = _store(store_factory)
    first, second = (Note(_DATE, "src", text, id="httk.test-1-99") for text in ("first", "second"))
    with store.bulk_ingest(finalize="parity") as bulk:
        # The second target's save fails inside bulk (its explicit id is already taken by the first).
        with pytest.raises(EntryIdConflictError):
            bulk.save(Run(source_id="a"), links={"notes": [first, second]})
        bulk.save(Run(source_id="b"), links={"tags": Tag("k")})
    assert _link_rows(store, "notes") == []
    assert len(_link_rows(store, "tags")) == 1


def test_two_retracted_lineages_revive_the_largest(store_factory) -> None:
    store = _store(store_factory)
    run = store.fetch(Run, store.save(Run(source_id="a")))
    note = store.fetch(Note, store.save(_note("n")))
    store.link(run, "notes", note)
    store.unlink(run, "notes", note)
    (first, retraction) = _link_rows(store, "notes")
    second_lineage = 10**6
    pair = {"source_lid": first[2], "target_lid": first[3]}
    _insert_link_rows(
        store,
        "notes",
        [
            {**pair, "logical_id": second_lineage, "retracted": 0},
            {**pair, "logical_id": second_lineage, "retracted": 1},
        ],
    )
    before = _link_rows(store, "notes")
    with store.bulk_ingest(finalize="parity") as bulk:
        bulk.save(Run(source_id="a"), links={"notes": _note("n")})
    (revival,) = _link_rows(store, "notes")[len(before) :]
    assert (revival[1], revival[4]) == (second_lineage, 0)
    assert [linked.text for linked in store.linked(run, "notes")] == ["n"]
    store.unlink(run, "notes", note)
    (retracted,) = _link_rows(store, "notes")[len(before) + 1 :]
    assert (retracted[1], retracted[4]) == (second_lineage, 1)
    assert store.linked(run, "notes") == ()
    store.link(run, "notes", note)
    assert _link_rows(store, "notes")[-1][1] == second_lineage
    assert [linked.text for linked in store.linked(run, "notes")] == ["n"]
    assert retraction[1] == first[1] != second_lineage


def test_reasserting_across_ingests(store_factory) -> None:
    store = _store(store_factory)
    for _ingest in range(2):
        with store.bulk_ingest(finalize="parity") as bulk:
            bulk.save(Run(source_id="a"), links={"notes": _note("n")})
    (link,) = _link_rows(store, "notes")
    run, (note,) = _run(store, "a"), store.linked(_run(store, "a"), "notes")
    store.unlink(run, "notes", note)
    with store.bulk_ingest(finalize="parity") as bulk:
        bulk.save(Run(source_id="a"), links={"notes": _note("n")})
    rows = _link_rows(store, "notes")
    assert len(rows) == 3
    assert (rows[-1][1], rows[-1][4]) == (link[1], 0)
    assert [linked.text for linked in store.linked(run, "notes")] == ["n"]


def test_shared_target_issues_no_role_update_per_save(store_factory) -> None:
    store = _store(store_factory)
    with _statements(store) as seen, store.bulk_ingest(finalize="parity", chunk_size=2) as bulk:
        for index in range(10):
            bulk.save(Run(source_id=f"run-{index}"), links={"tags": Tag("source", "cod")})
    assert [statement for statement in seen if statement.lstrip().upper().startswith("UPDATE")] == []
    assert len(_link_rows(store, "tags")) == 10


@pytest.mark.parametrize("populated", [False, True])
def test_unused_links_issue_identical_statements(store_factory, populated) -> None:
    """Unused ``links`` (absent, ``None``, ``{}``) take one code path, which does no link work.

    The three traces all come from the current code, so their equality only pins that the
    three spellings share a path; the guard against link work on that path is that the
    ``links=None`` ingest issues no UPDATE and never looks up link rows (by ``source_lid``) in a
    ``_httk_link_`` table.
    """
    traces = []
    for links in ("absent", None, {}):
        store = _store(store_factory)
        if populated:
            store.save(Run(source_id="existing"))  # outside the stream: no dedup role UPDATE
        with _statements(store) as seen, store.bulk_ingest(finalize="parity", chunk_size=2) as bulk:
            for index in range(5):
                if links == "absent":
                    bulk.save(Run(source_id=f"run-{index}"))
                else:
                    bulk.save(Run(source_id=f"run-{index}"), links=links)
        if links is None:
            assert [statement for statement in seen if statement.lstrip().upper().startswith("UPDATE")] == []
            reads = [statement for statement in seen if statement.lstrip().upper().startswith("SELECT")]
            # Bookkeeping scans (count, max sid) of pre-existing link tables predate links; a link lookup reads lids.
            assert [statement for statement in reads if "_httk_link_" in statement and "source_lid" in statement] == []
        # Sorted: the order of a table's CREATE INDEX statements follows a set, not the ingest.
        traces.append(sorted(seen))
    assert len(traces[0]) == len(traces[1]) == len(traces[2])
    assert traces[0] == traces[1] == traces[2]


@pytest.mark.parametrize("mode", _MODES)
def test_references_promote_and_as_record(store_factory, mode) -> None:
    store = _store(store_factory, extended=True)
    reference = ReferenceRecord(title="The external database")  # type: ignore[call-arg]
    with _bulk(store, mode) as bulk:
        bulk.save(Job("job"), as_record=JobRecord, links={"references": reference, "notes": _note("n")})
        bulk.save(Envelope("cif", Run(source_id="nested")), promote=Run, links={"notes": _note("n")})
    searcher = store.searcher()
    v = searcher.variable(JobRecord)
    (job,) = [row.values[0] for row in searcher.results(r=v)]
    assert [linked.title for linked in store.linked(job, "references")] == ["The external database"]
    assert [linked.text for linked in store.linked(job, "notes")] == ["n"]
    searcher = store.searcher()
    v = searcher.variable(Envelope)
    (envelope,) = [row.values[0] for row in searcher.results(r=v)]
    assert [linked.text for linked in store.linked(envelope, "notes")] == ["n"]
    assert store.linked(_run(store, "nested"), "notes") == ()
    assert _count(store, Note) == 1
    _assert_fsck_clean(store)


@pytest.mark.parametrize("mode", [mode for mode in _MODES if mode != "serial-parity"])
def test_unresolved_endpoint_is_an_invariant_breach(store_factory, monkeypatch, mode) -> None:
    """A deferred or parallel link endpoint finalization did not map fails the ingest instead of guessing a sid."""
    from httk.store.backend.sql import bulk_deferred, bulk_parallel

    store = _store(store_factory)

    def drop_notes(ingest) -> None:
        for key in [key for key in ingest._resolved_map if key[0] == "core_note"]:
            del ingest._resolved_map[key]

    merge = bulk_parallel.merge
    populate = bulk_deferred.DeferredFinalizer._populate_returned_sids

    def lossy_merge(ingest, manifests) -> None:
        merge(ingest, manifests)
        drop_notes(ingest)

    def lossy_populate(finalizer) -> None:
        populate(finalizer)
        drop_notes(finalizer.ingest)

    monkeypatch.setattr(bulk_parallel, "merge", lossy_merge)
    monkeypatch.setattr(bulk_deferred.DeferredFinalizer, "_populate_returned_sids", lossy_populate)
    breach = pytest.raises(RuntimeError, match=r"'core_note' save value \d+ was not resolved by finalization")
    with breach, _bulk(store, mode) as bulk:
        bulk.save(Run(source_id="a"), links={"notes": _note("n")})
        # Serial deferred maps its public sids from this table after the finalizer ran.
        bulk._serial_public_stage_sid = {
            key: value for key, value in bulk._serial_public_stage_sid.items() if key[0] != "core_note"
        }
    assert _link_rows(store, "notes") == []
