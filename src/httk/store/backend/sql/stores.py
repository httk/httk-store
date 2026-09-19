"""Per-dialect stores that build and own their :class:`~httk.store.backend.sql.engine.Backend`.

These are thin :class:`~httk.store.backend.sql.store.SqlStore` subclasses whose
name selects the database engine and whose first argument is only the location:
``SqliteStore("results.sqlite")`` instead of
``SqlStore(Backend.sqlite("results.sqlite"))``.  Each builds the matching
``Backend``, owns it, and disposes it on :meth:`~httk.store.backend.sql.store.SqlStore.close`
or when leaving a ``with`` block.

For a custom SQLAlchemy engine, one ``Backend`` shared across several stores or
a ``with Backend.sqlite(...) as db:`` block, or ``degraded=True`` recovery, use
the two-object form :class:`~httk.store.backend.sql.engine.Backend` +
:class:`~httk.store.backend.sql.store.SqlStore` directly.
"""

import os
from typing import Any

import sqlalchemy

from httk.store.backend.sql.engine import Backend
from httk.store.backend.sql.store import SqlStore

__all__ = [
    "ClickhouseStore",
    "DuckdbStore",
    "PostgresqlStore",
    "SqliteStore",
]


def _init_owning(store: SqlStore, backend: Backend, store_options: dict[str, Any]) -> None:
    """Run ``SqlStore.__init__`` on a self-built backend, disposing it if that fails.

    :param store: The freshly created per-dialect store.
    :param backend: The backend this store built and will own.
    :param store_options: The :class:`~httk.store.backend.sql.store.SqlStore` keyword options to forward.
    :return: None.
    """
    try:
        SqlStore.__init__(store, backend, **store_options)
    except BaseException:
        backend.dispose()
        raise
    store._owns_database = True


class SqliteStore(SqlStore):
    """An SQLite store that builds and owns its :class:`~httk.store.backend.sql.engine.Backend`.

    The store owns the backend it builds and disposes it on :meth:`~httk.store.backend.sql.store.SqlStore.close` or
    when leaving a ``with`` block.  For a custom SQLAlchemy engine or a
    ``Backend`` shared across a ``with`` block, use :class:`~httk.store.backend.sql.engine.Backend` and
    :class:`~httk.store.backend.sql.store.SqlStore` directly.

    :param path: The database file path, or ``None`` for an in-memory database.
    :param degraded: Open with autocommit isolation for degraded-mode access
        (recovery and inspection) instead of the default transactional isolation.
    :param \\**store_options: Keyword options of :class:`~httk.store.backend.sql.store.SqlStore`
        (``entry_records``, ``entry_ids``, ``upgrade``, ...).
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        degraded: bool = False,
        **store_options: Any,
    ) -> None:
        _init_owning(self, Backend.sqlite(path, degraded=degraded), store_options)


class DuckdbStore(SqlStore):
    """A DuckDB store that builds and owns its :class:`~httk.store.backend.sql.engine.Backend`.

    The store owns the backend it builds and disposes it on :meth:`~httk.store.backend.sql.store.SqlStore.close` or
    when leaving a ``with`` block.  For a custom SQLAlchemy engine or a
    ``Backend`` shared across a ``with`` block, use :class:`~httk.store.backend.sql.engine.Backend` and
    :class:`~httk.store.backend.sql.store.SqlStore` directly.

    :param path: The database file path, or ``None`` for an in-memory database.
    :param read_only: Open the file in DuckDB ``READ_ONLY`` access mode, so
        several processes may open it concurrently for reading; writes then fail.
        Ignored for the in-memory database.
    :param memory_limit: An optional DuckDB ``memory_limit`` setting such as
        ``"1GB"``; ``None`` falls back to the ``HTTK_DUCKDB_MEMORY_LIMIT``
        environment variable.
    :param \\**store_options: Keyword options of :class:`~httk.store.backend.sql.store.SqlStore`
        (``entry_records``, ``entry_ids``, ``upgrade``, ...).
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        read_only: bool = False,
        memory_limit: str | None = None,
        **store_options: Any,
    ) -> None:
        _init_owning(self, Backend.duckdb(path, memory_limit=memory_limit, read_only=read_only), store_options)


class PostgresqlStore(SqlStore):
    """A PostgreSQL store that builds and owns its :class:`~httk.store.backend.sql.engine.Backend`.

    The store owns the backend it builds and disposes it on :meth:`~httk.store.backend.sql.store.SqlStore.close` or
    when leaving a ``with`` block.  For a custom SQLAlchemy engine or a
    ``Backend`` shared across a ``with`` block, use :class:`~httk.store.backend.sql.engine.Backend` and
    :class:`~httk.store.backend.sql.store.SqlStore` directly.

    :param url: PostgreSQL SQLAlchemy URL or URL string.
    :param database: The database name overriding the URL path, if supplied.
    :param \\**store_options: Keyword options of :class:`~httk.store.backend.sql.store.SqlStore`
        (``entry_records``, ``entry_ids``, ``upgrade``, ...).
    """

    def __init__(
        self,
        url: str | sqlalchemy.URL,
        *,
        database: str | None = None,
        **store_options: Any,
    ) -> None:
        _init_owning(self, Backend.postgresql(url, database=database), store_options)


class ClickhouseStore(SqlStore):
    """A ClickHouse store that builds and owns its :class:`~httk.store.backend.sql.engine.Backend`.

    The store owns the backend it builds and disposes it on :meth:`~httk.store.backend.sql.store.SqlStore.close` or
    when leaving a ``with`` block.  For a custom SQLAlchemy engine or a
    ``Backend`` shared across a ``with`` block, use :class:`~httk.store.backend.sql.engine.Backend` and
    :class:`~httk.store.backend.sql.store.SqlStore` directly.

    :param url: ClickHouse SQLAlchemy URL or URL string.
    :param database: The database name overriding the URL path, if supplied.
    :param \\**store_options: Keyword options of :class:`~httk.store.backend.sql.store.SqlStore`
        (``entry_records``, ``entry_ids``, ``upgrade``, ...).
    """

    def __init__(
        self,
        url: str | sqlalchemy.URL,
        *,
        database: str | None = None,
        **store_options: Any,
    ) -> None:
        _init_owning(self, Backend.clickhouse(url, database=database), store_options)
