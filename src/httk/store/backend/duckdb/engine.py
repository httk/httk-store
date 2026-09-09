"""DuckDB database construction for :class:`~httk.store.backend.sql.engine.Backend`."""

import importlib
import os
from typing import TYPE_CHECKING, Any

import sqlalchemy

if TYPE_CHECKING:
    from httk.store.backend.sql.engine import Backend


def database(
    cls: "type[Backend]",
    path: str | os.PathLike[str] | None = None,
    *,
    memory_limit: str | None = None,
    read_only: bool = False,
) -> "Backend":
    """Build a DuckDB-backed backend stored in ``path``, or in memory when ``path`` is None.

    :param cls: The backend class to instantiate.
    :param path: The database file path, or ``None`` for an in-memory database.
    :param memory_limit: An optional DuckDB ``memory_limit`` setting such as
        ``"1GB"``. DuckDB's own default allows every instance up to about 80% of
        system RAM, which multiplies dangerously across parallel test or ingest
        processes; when this parameter is ``None`` the ``HTTK_DUCKDB_MEMORY_LIMIT``
        environment variable (if set) supplies the cap instead, so process trees
        can be memory-guarded wholesale.
    :param read_only: Open the file in DuckDB ``READ_ONLY`` access mode. A read-only
        database takes no write lock, so several processes (and this one) may open the
        same file concurrently for reading; write operations on the resulting backend
        will fail. Ignored for the in-memory database.
    :return: The configured backend wrapper.
    :raises ImportError: If the ``duckdb_engine`` SQLAlchemy dialect is not installed;
        install the ``httk-store[duckdb]`` extra to use it.
    """
    try:
        importlib.import_module("duckdb_engine")
    except ImportError as error:
        raise ImportError(
            "the DuckDB backend needs the 'duckdb_engine' SQLAlchemy dialect; "
            "install the 'httk-store[duckdb]' extra to use Backend.duckdb()"
        ) from error
    location = ":memory:" if path is None else os.fspath(path)
    limit = memory_limit if memory_limit is not None else os.environ.get("HTTK_DUCKDB_MEMORY_LIMIT")
    config: dict[str, Any] = {}
    if limit:
        config["memory_limit"] = limit
    if read_only and path is not None:
        config["access_mode"] = "READ_ONLY"
    options: dict[str, Any] = {"connect_args": {"config": config}} if config else {}
    engine = sqlalchemy.create_engine(f"duckdb:///{location}", **options)
    # duckdb_engine derives from the psycopg2 dialect, which doubles
    # backslashes when rendering inline string literals (PostgreSQL's
    # non-standard-conforming-strings legacy). DuckDB always uses
    # standard-conforming string literals, so that doubling corrupts e.g.
    # the LIKE ... ESCAPE '\' clause the search DSL emits; turn it off.
    engine.dialect._backslash_escapes = False  # type: ignore[attr-defined]
    return cls(engine)
