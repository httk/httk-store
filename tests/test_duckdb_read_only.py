"""``Backend.duckdb(read_only=True)`` opens with no write lock, so reads work and writes fail.

Read-only ``access_mode`` is what lets several processes open the same DuckDB file at once for
concurrent reading (used by build-cod's parallel distinct pass); this checks the single-process
guarantees it rests on -- reads succeed, writes are refused, and two read-only handles coexist.
"""

from pathlib import Path

import pytest
import sqlalchemy


def _require_duckdb() -> None:
    pytest.importorskip("duckdb_engine")


def _seed(path: Path) -> None:
    from httk.store.backend.sql import Backend

    database = Backend.duckdb(path)
    try:
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE t (a INTEGER)"))
            connection.execute(sqlalchemy.text("INSERT INTO t VALUES (1), (2), (3)"))
    finally:
        database.dispose()


def test_read_only_reads_succeed_and_writes_fail(tmp_path: Path) -> None:
    _require_duckdb()
    from httk.store.backend.sql import Backend

    path = tmp_path / "ro.duckdb"
    _seed(path)

    database = Backend.duckdb(path, read_only=True)
    try:
        with database.engine.connect() as connection:
            assert connection.execute(sqlalchemy.text("SELECT count(*) FROM t")).scalar_one() == 3
        # DuckDB refuses any write in READ_ONLY mode.
        with pytest.raises(Exception), database.engine.begin() as connection:  # noqa: B017
            connection.execute(sqlalchemy.text("INSERT INTO t VALUES (4)"))
    finally:
        database.dispose()


def test_two_read_only_handles_open_the_same_file_at_once(tmp_path: Path) -> None:
    _require_duckdb()
    from httk.store.backend.sql import Backend

    path = tmp_path / "shared.duckdb"
    _seed(path)

    first = Backend.duckdb(path, read_only=True)
    second = Backend.duckdb(path, read_only=True)
    try:
        with first.engine.connect() as a, second.engine.connect() as b:
            assert a.execute(sqlalchemy.text("SELECT count(*) FROM t")).scalar_one() == 3
            assert b.execute(sqlalchemy.text("SELECT count(*) FROM t")).scalar_one() == 3
    finally:
        first.dispose()
        second.dispose()
