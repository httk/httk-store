"""``reclaim_space``: in-place maintenance (light) and storage rewrite (full) per backend."""

import os
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
import sqlalchemy
from httk.core.storage import content_id

from httk.store import Backend, DuckdbStore, SqliteStore, SqlStore
from httk.store.backend.schema import resolve_schema


@dataclass(frozen=True)
class Tiny:
    name: str
    number: int


def _tiny(index: int) -> Tiny:
    return Tiny(f"name-{index}-" + "x" * 50, index)


def _delete(store: SqlStore, condition: str) -> None:
    table = resolve_schema(Tiny).table_name
    with store._database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DELETE FROM "{table}" WHERE {condition}'))


def test_invalid_effort_is_refused() -> None:
    store = SqliteStore(entry_records={})
    with pytest.raises(ValueError, match="effort"):
        store.reclaim_space("medium")  # type: ignore[arg-type]


def test_sqlite_full_shrinks_file_and_keeps_data(tmp_path: Path) -> None:
    path = tmp_path / "x.sqlite"
    store = SqliteStore(path, entry_records={})
    with store.transaction():
        sids = [store.save(_tiny(index)) for index in range(2000)]
    _delete(store, "number >= 10")
    assert store.reclaim_space() == 0  # PRAGMA optimize never shrinks the file
    assert store.reclaim_space("full") > 0
    store.close()
    with SqliteStore(path) as reopened:
        assert [reopened.fetch(Tiny, sid, eager=True) for sid in sids[:10]] == [_tiny(index) for index in range(10)]


def test_in_memory_stores_return_none() -> None:
    sqlite = SqliteStore(entry_records={})
    sqlite.save(_tiny(1))
    assert sqlite.reclaim_space() is None
    assert sqlite.reclaim_space("full") is None
    pytest.importorskip("duckdb_engine")
    duck = DuckdbStore(entry_records={})
    sid = duck.save(_tiny(1))
    assert duck.reclaim_space() is None
    assert duck.reclaim_space("full") is None
    assert duck.fetch(Tiny, sid, eager=True) == _tiny(1)


def test_refused_inside_transaction_and_bulk_ingest() -> None:
    store = SqliteStore(entry_records={})
    with store.transaction(), pytest.raises(RuntimeError, match="transaction"):
        store.reclaim_space()
    with store.bulk_ingest() as bulk, pytest.raises(RuntimeError, match="bulk_ingest"):
        bulk.save(_tiny(1))
        store.reclaim_space("full")


def test_duckdb_full_rewrites_fragmented_file(tmp_path: Path) -> None:
    pytest.importorskip("duckdb_engine")
    path = tmp_path / "x.duckdb"
    store = DuckdbStore(path, entry_records={})
    sids = []
    for index in range(200):
        sids.append(store.save(_tiny(index)))
        if index % 10 == 9:
            # Frequent checkpoints, as DuckDB's automatic ones do in a large build, fragment the file.
            assert store.reclaim_space() is not None
    top = max(sids)
    _delete(store, f"sid = {top}")
    path.chmod(0o660)
    before = os.path.getsize(path)
    shrunk = store.reclaim_space("full")
    assert shrunk is not None and shrunk > 0  # measured: 2,633,728 -> 1,585,152 bytes
    assert os.path.getsize(path) < before
    assert sorted(os.listdir(tmp_path)) == ["x.duckdb"]  # no .reclaim temp, no WAL
    assert stat.S_IMODE(path.stat().st_mode) == 0o660
    assert store.fetch(Tiny, sids[0], eager=True) == _tiny(0)
    assert store.fetch_by_content_id(Tiny, content_id(_tiny(7)), eager=True) == _tiny(7)
    # The sequence survived the copy: the deleted top sid is not handed out again.
    new = store.save(Tiny("after", -1))
    assert new > top
    store.close()
    with DuckdbStore(path) as reopened:  # opening validates the stamped layout
        assert reopened.fetch(Tiny, new, eager=True) == Tiny("after", -1)
        assert reopened.fetch(Tiny, sids[-2], eager=True) == _tiny(198)


def test_duckdb_full_refusals_leave_original_untouched(tmp_path: Path) -> None:
    pytest.importorskip("duckdb_engine")
    path = tmp_path / "x.duckdb"
    store = DuckdbStore(path, entry_records={})
    sid = store.save(_tiny(1))
    store.reclaim_space()

    other = DuckdbStore(path)  # a second live handle on the same file
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="still open elsewhere"):
        store.reclaim_space("full")
    assert path.read_bytes() == original
    assert sorted(os.listdir(tmp_path)) == ["x.duckdb"]
    other.close()
    assert store.fetch(Tiny, sid, eager=True) == _tiny(1)  # still usable after the refused swap

    with store.transaction(), pytest.raises(RuntimeError, match="transaction"):
        store.reclaim_space("full")
    store.close()

    database = Backend.duckdb(path)
    try:
        borrowed = SqlStore(database)
        with pytest.raises(RuntimeError, match="owns its backend"):
            borrowed.reclaim_space("full")
        assert borrowed.reclaim_space() is not None
    finally:
        database.dispose()

    with DuckdbStore(path, read_only=True) as read_only, pytest.raises(RuntimeError, match="read-only"):
        read_only.reclaim_space("full")
    assert sorted(os.listdir(tmp_path)) == ["x.duckdb"]


def test_duckdb_full_refused_when_another_process_takes_the_file(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("duckdb_engine")
    path = tmp_path / "x.duckdb"
    store = DuckdbStore(path, entry_records={})
    sid = store.save(_tiny(1))
    store.reclaim_space()
    original = path.read_bytes()
    engine = store._database.engine
    holder: list[subprocess.Popen[str]] = []
    dispose = engine.dispose

    def dispose_then_open_elsewhere() -> None:
        # Another process opens the file read-write in the window after this store lets go of it.
        dispose()
        script = "import duckdb, sys; c = duckdb.connect(sys.argv[1]); print('ready', flush=True); sys.stdin.read()"
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        holder.append(process)
        assert process.stdout is not None and process.stdout.readline().strip() == "ready"

    monkeypatch.setattr(engine, "dispose", dispose_then_open_elsewhere)
    try:
        with pytest.raises(RuntimeError, match="another process") as refused:
            store.reclaim_space("full")
        assert type(refused.value.__cause__).__name__ == "IOException"  # the other process's file lock
    finally:
        for process in holder:
            process.communicate("")
    monkeypatch.undo()
    assert path.read_bytes() == original
    assert sorted(os.listdir(tmp_path)) == ["x.duckdb"]
    assert store.fetch(Tiny, sid, eager=True) == _tiny(1)
    store.close()


def test_duckdb_bulk_ingest_refusal(tmp_path: Path) -> None:
    pytest.importorskip("duckdb_engine")
    with (
        DuckdbStore(tmp_path / "x.duckdb", entry_records={}) as store,
        store.bulk_ingest() as bulk,
        pytest.raises(RuntimeError, match="bulk_ingest"),
    ):
        bulk.save(_tiny(1))
        store.reclaim_space("full")


def test_postgresql_both_efforts() -> None:
    from postgres_support import IsolatedPostgresDatabase

    isolated = IsolatedPostgresDatabase()
    database = Backend.postgresql(isolated.uri)
    try:
        store = SqlStore(database, entry_records={})
        sids = [store.save(_tiny(index)) for index in range(20)]
        _delete(store, "number >= 10")
        assert store.reclaim_space() is None
        assert store.reclaim_space("full") is None
        assert store.fetch(Tiny, sids[3], eager=True) == _tiny(3)
    finally:
        database.dispose()
        isolated.drop()


def test_clickhouse_both_efforts() -> None:
    from conftest import clickhouse_test_uri
    from test_clickhouse_bulk import _clickhouse_bulk_database

    with _clickhouse_bulk_database(clickhouse_test_uri()) as database:
        store = SqlStore(database, entry_records={})
        with store.bulk_ingest(workers=1) as bulk:
            provisional = [bulk.save(_tiny(index)) for index in range(20)]
        sids = [bulk.resolved_sid(Tiny, sid) for sid in provisional]
        assert store.reclaim_space() is None
        assert store.reclaim_space("full") is None
        assert store.fetch(Tiny, sids[3], eager=True) == _tiny(3)


def test_mongo_both_efforts(mongo_test_client) -> None:
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    name = f"httk_test_{uuid.uuid4().hex}"
    database = MongoDatabase(mongo_test_client, name)
    try:
        store = MongoStore(database, entry_records={})
        sid = store.save(_tiny(3))
        with pytest.raises(ValueError, match="effort"):
            store.reclaim_space("medium")  # type: ignore[arg-type]
        assert store.reclaim_space() is None
        assert store.reclaim_space("full") is None
        assert store.fetch(Tiny, sid, eager=True) == _tiny(3)
    finally:
        mongo_test_client.drop_database(name)
