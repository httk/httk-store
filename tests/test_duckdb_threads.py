"""DuckDB ``threads`` caps: the ``threads=`` parameter and the ``HTTK_DUCKDB_THREADS`` fallback."""

import pytest
import sqlalchemy

pytest.importorskip("duckdb_engine")

from httk.store.backend.sql import Backend
from httk.store.backend.sql.stores import DuckdbStore


def _threads(store: DuckdbStore) -> int:
    with store._database.engine.connect() as connection:  # pyright: ignore[reportPrivateUsage]
        return int(connection.execute(sqlalchemy.text("SELECT current_setting('threads')")).scalar_one())


def test_threads_parameter_caps_duckdb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HTTK_DUCKDB_THREADS", raising=False)
    with DuckdbStore(threads=2, entry_records={}) as store:
        assert _threads(store) == 2


def test_threads_environment_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTK_DUCKDB_THREADS", "2")
    with DuckdbStore(entry_records={}) as store:
        assert _threads(store) == 2
    with DuckdbStore(threads=3, entry_records={}) as store:
        assert _threads(store) == 3


@pytest.mark.parametrize("value", ["0", "-1", "two", "1.5"])
def test_invalid_threads_environment_refused(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HTTK_DUCKDB_THREADS", value)
    with pytest.raises(ValueError, match="HTTK_DUCKDB_THREADS"):
        Backend.duckdb()


def test_invalid_threads_parameter_refused() -> None:
    with pytest.raises(ValueError, match="threads"):
        Backend.duckdb(threads=0)
