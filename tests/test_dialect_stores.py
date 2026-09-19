"""Per-dialect store classes (``SqliteStore`` and siblings) that own their backend.

Server-backed dialects are only checked for importability/subclassing here; the
sqlite arm exercises the actual save/fetch and the backend-ownership lifecycle.
"""

from dataclasses import dataclass

import pytest

import httk.store
import httk.store.backend.sql.engine
import httk.store.backend.sql.store
from httk.store import Backend, ClickhouseStore, DuckdbStore, PostgresqlStore, SqliteStore, SqlStore


@dataclass(frozen=True)
class Tiny:
    name: str
    number: int


def test_sqlite_store_in_memory_and_file(tmp_path):
    store = SqliteStore(entry_records={})
    assert repr(store).startswith("SqliteStore(")  # subclass reprs under its own name
    sid = store.save(Tiny("a", 1))
    assert store.fetch(Tiny, sid, eager=True) == Tiny("a", 1)

    file_store = SqliteStore(tmp_path / "x.sqlite", entry_records={})
    fsid = file_store.save(Tiny("b", 2))
    assert file_store.fetch(Tiny, fsid, eager=True) == Tiny("b", 2)


def test_context_manager_disposes_owned_backend():
    with SqliteStore(entry_records={}) as store:
        store.save(Tiny("c", 3))
        backend = store._database
    # A disposed Backend refuses to hand out a lifecycle generation.
    with pytest.raises(RuntimeError, match="disposed"):
        _ = backend.lifecycle_generation


def test_close_does_not_dispose_caller_owned_backend():
    backend = Backend.sqlite()
    store = SqlStore(backend, entry_records={})
    store.close()  # store does not own the backend, so this is a no-op
    assert backend.lifecycle_generation == 0  # still usable
    backend.dispose()


def test_first_open_without_declaration_raises_and_does_not_leak():
    # Exercises the _init_owning except-path: the self-built backend is disposed
    # before the TypeError propagates (no leaked pool).
    with pytest.raises(TypeError, match="entry_records or entry_families"):
        SqliteStore()


def test_duckdb_store_owns_backend(tmp_path):
    pytest.importorskip("duckdb_engine")
    with DuckdbStore(tmp_path / "x.duckdb", entry_records={}) as store:
        sid = store.save(Tiny("d", 4))
        assert store.fetch(Tiny, sid, eager=True) == Tiny("d", 4)
        backend = store._database
    with pytest.raises(RuntimeError, match="disposed"):
        _ = backend.lifecycle_generation


def test_server_dialect_stores_are_sqlstore_subclasses():
    assert issubclass(PostgresqlStore, SqlStore)
    assert issubclass(ClickhouseStore, SqlStore)


def test_root_all_surface():
    for name in ("SqliteStore", "DuckdbStore", "PostgresqlStore", "ClickhouseStore"):
        assert name in httk.store.__all__
    assert "Backend" not in httk.store.__all__
    assert "SqlStore" not in httk.store.__all__
    # Both still importable and identical to the SQL-layer classes.
    assert httk.store.SqlStore is httk.store.backend.sql.store.SqlStore
    assert httk.store.Backend is httk.store.backend.sql.engine.Backend
