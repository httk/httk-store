"""``KeepStored`` fields: the first non-null value wins on a deduplicating re-save (SQL single and bulk paths)."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import pytest
import sqlalchemy
from httk.core import FileEntry, FileRecord, content_id
from httk.core.storage import StorageInfo
from sqlalchemy import event

from httk.store import EntryFamilyDeclaration, EntryIdScheme, EntryRecordDeclaration
from httk.store.backend.sql import Backend, SqlStore
from httk.store.store_common import EntryMetadataConflictError


class KeepHolderFamily:
    """Test-only family whose record holds a child ``FileRecord``."""


@dataclass(frozen=True)
class KeepHolder:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="keep_holder", identity_name="keep-holder")

    label: str
    f: FileRecord


_FAMILIES = (
    EntryFamilyDeclaration(
        name="test-keep-stored",
        family=KeepHolderFamily,
        records=(EntryRecordDeclaration(name="test-keep-stored-holder", record=KeepHolder),),
    ),
)
_FIRST = FileRecord("data/ws/jobs/x/POSCAR", "POSCAR", size=10, sha256="ab" * 32)
_SECOND = replace(_FIRST, url="data/ws/jobs/y/POSCAR", size=11, description="other path")


@pytest.fixture(params=["sqlite", "duckdb"])
def store(request: pytest.FixtureRequest) -> Iterator[SqlStore]:
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Backend.duckdb() as database:
            yield SqlStore(database, entry_families=_FAMILIES)
        return
    with Backend.sqlite() as database:
        yield SqlStore(database, entry_families=_FAMILIES)


def _file_rows(store: SqlStore) -> list[Any]:
    with store._database.engine.connect() as connection:
        return list(connection.execute(sqlalchemy.text("SELECT url, size, description FROM core_file")))


def test_top_level_resave_keeps_first_location(store: SqlStore) -> None:
    sid = store.save(_FIRST)
    assert store.save(_SECOND) == sid
    # The stored NULL description is filled; the stored url and size are kept.
    assert _file_rows(store) == [(_FIRST.url, 10, "other path")]
    # A field without KeepStored still conflicts.
    with pytest.raises(EntryMetadataConflictError, match="immutable_id"):
        store.save(replace(_SECOND, immutable_id="different"))


def test_child_resave_keeps_first_location(store: SqlStore) -> None:
    store.save(KeepHolder("a", _FIRST))
    store.save(KeepHolder("b", _SECOND))
    assert _file_rows(store) == [(_FIRST.url, 10, "other path")]


# Deferred finalize and workers>1 accept only a physically empty store, and
# keep the first occurrence's values (no in-batch fill of a NULL there).
@pytest.mark.parametrize(
    ("options", "populated", "description"),
    [
        ({}, False, None),
        ({}, True, "other path"),
        ({"finalize": "parity"}, False, "other path"),
        ({"finalize": "deferred"}, False, None),
        ({"workers": 2}, False, None),
    ],
    ids=["serial-empty", "serial-populated", "parity-empty", "deferred", "parallel"],
)
def test_bulk_resave_keeps_first_location(
    store: SqlStore, options: dict[str, Any], populated: bool, description: str | None
) -> None:
    if populated:
        store.save(_FIRST)
    with store.bulk_ingest(**options) as bulk:
        if not populated:
            bulk.save(_FIRST)
        bulk.save(_SECOND)
        bulk.save(KeepHolder("a", _SECOND))
    assert _file_rows(store) == [(_FIRST.url, 10, description)]


def test_mongo_resave_keeps_first_location(mongo_test_database: Any) -> None:
    from httk.store.backend.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_families=_FAMILIES)
    sid = store.save(_FIRST)
    assert store.save(_SECOND) == sid
    store.save(KeepHolder("b", _SECOND))
    assert [document["f"]["url"] for document in mongo_test_database.database["core_file"].find({})] == [_FIRST.url]
    with pytest.raises(EntryMetadataConflictError, match="immutable_id"):
        store.save(replace(_SECOND, immutable_id="different"))


_LOST = replace(_FIRST, url=None, size=None)
_REAL = replace(_FIRST, url="data/ws/jobs/z/POSCAR")
# (first save, second save, expected stored url): the first non-null url wins.
_URL_CASES = [
    pytest.param(_LOST, _FIRST, _FIRST.url, id="lost-then-real"),
    pytest.param(_FIRST, _LOST, _FIRST.url, id="real-then-lost"),
    pytest.param(_FIRST, _REAL, _FIRST.url, id="real-then-real"),
    pytest.param(_LOST, replace(_LOST, name="other"), None, id="lost-then-lost"),
]


@pytest.fixture(params=["sqlite", "duckdb"])
def file_store(request: pytest.FixtureRequest) -> Iterator[SqlStore]:
    """A store where ``FileRecord`` is a top-level entry, so it mints ids and revisions."""
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
    with getattr(Backend, request.param)() as database:
        yield SqlStore(database, entry_records={FileEntry: FileRecord}, entry_ids=EntryIdScheme("httk.test", "1"))


def test_lost_file_round_trips(store: SqlStore) -> None:
    sid = store.save(_LOST)
    assert store.fetch(FileRecord, sid, eager=True).url is None
    holder = store.save(KeepHolder("a", _LOST))
    assert store.fetch(KeepHolder, holder, eager=True).f.url is None
    assert _file_rows(store) == [(None, None, None)]


@pytest.mark.parametrize(("first", "second", "url"), _URL_CASES)
def test_top_level_first_non_null_url_wins(
    file_store: SqlStore, first: FileRecord, second: FileRecord, url: str | None
) -> None:
    sid = file_store.save(first)
    before = file_store.fetch(FileRecord, sid, eager=True)
    assert file_store.save(second) == sid
    after = file_store.fetch(FileRecord, sid)
    assert (after.url, after.size, after.name) == (url, 10 if _FIRST in (first, second) else None, first.name)
    # An in-place metadata fill: same revision, lineage and content id.
    assert after.immutable_id == before.immutable_id is not None
    assert len(file_store.history(after)) == 1
    assert content_id(after) == content_id(before)
    with file_store._database.engine.connect() as connection:
        assert connection.execute(sqlalchemy.text("SELECT count(*) FROM core_file")).scalar_one() == 1


@pytest.mark.parametrize(("first", "second", "url"), _URL_CASES)
@pytest.mark.parametrize("same_holder", [False, True], ids=["two-holders", "same-holder"])
def test_child_first_non_null_url_wins(
    store: SqlStore, first: FileRecord, second: FileRecord, url: str | None, same_holder: bool
) -> None:
    store.save(KeepHolder("a", first))
    holder = store.save(KeepHolder("a" if same_holder else "b", second))
    assert [row[0] for row in _file_rows(store)] == [url]
    assert store.fetch(KeepHolder, holder, eager=True).f.url == url


@pytest.mark.parametrize(("first", "second", "url"), _URL_CASES)
@pytest.mark.parametrize(
    ("options", "populated"),
    [({"finalize": "parity"}, False), ({"finalize": "parity", "chunk_size": 1}, False), ({}, True)],
    ids=["in-batch", "across-chunks", "against-stored"],
)
def test_bulk_serial_first_non_null_url_wins(
    store: SqlStore,
    first: FileRecord,
    second: FileRecord,
    url: str | None,
    options: dict[str, Any],
    populated: bool,
) -> None:
    if populated:
        store.save(first)
    with store.bulk_ingest(**options) as bulk:
        if not populated:
            bulk.save(KeepHolder("a", first))
        bulk.save(second)
        bulk.save(KeepHolder("b", second))
    assert [row[0] for row in _file_rows(store)] == [url]


def test_mongo_first_non_null_url_wins(mongo_test_database: Any) -> None:
    from httk.store.backend.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_families=_FAMILIES)
    sid = store.save(_LOST)
    assert store.fetch(FileRecord, sid).url is None
    assert store.save(_FIRST) == sid
    store.save(KeepHolder("b", _LOST))
    store.save(replace(_REAL, size=11))
    assert store.fetch(FileRecord, sid).url == _FIRST.url
    assert [
        (document["f"]["url"], document["f"]["size"]) for document in mongo_test_database.database["core_file"].find({})
    ] == [(_FIRST.url, 10)]


def test_fill_never_overwrites_a_concurrently_committed_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer that read NULL must not overwrite another writer's url committed meanwhile (READ COMMITTED race)."""
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_families=_FAMILIES)
        sid = store.save(_LOST)
        original = SqlStore._metadata_parent_row

        def stale_row(self: SqlStore, connection: sqlalchemy.Connection, *args: Any) -> Any:
            row = dict(original(self, connection, *args))  # Reads url NULL ...
            # ... then another writer commits a location before this writer's fill.
            connection.execute(sqlalchemy.text("UPDATE core_file SET url = 'data/a' WHERE sid = :sid"), {"sid": sid})
            return row

        statements: list[str] = []
        event.listen(database.engine, "before_cursor_execute", lambda *args: statements.append(str(args[2])))
        monkeypatch.setattr(SqlStore, "_metadata_parent_row", stale_row)
        assert store.save(_FIRST) == sid
        monkeypatch.undo()
        assert _file_rows(store) == [("data/a", 10, None)]
        assert store.fetch(FileRecord, sid).url == "data/a"
        fills = [statement for statement in statements if statement.startswith("UPDATE core_file")]
        assert any("SET url=" in s and "core_file.url IS NULL" in s for s in fills)
