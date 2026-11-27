"""DuckDB database construction for :class:`~httk.store.backend.sql.engine.Backend`."""

import importlib
import os
from typing import TYPE_CHECKING, Any

import sqlalchemy

if TYPE_CHECKING:
    from httk.store.backend.sql.engine import Backend


def _resource_config(memory_limit: str | None = None, threads: int | None = None) -> dict[str, Any]:
    """Return the DuckDB ``memory_limit``/``threads`` connection config, with environment fallbacks.

    :param memory_limit: A DuckDB ``memory_limit`` such as ``"1GB"``; ``None`` falls
        back to ``HTTK_DUCKDB_MEMORY_LIMIT``.
    :param threads: A positive DuckDB ``threads`` cap; ``None`` falls back to
        ``HTTK_DUCKDB_THREADS``.
    :return: The config entries to pass to DuckDB (empty when nothing is capped).
    :raises ValueError: If ``threads`` or ``HTTK_DUCKDB_THREADS`` is not a positive integer.
    """
    config: dict[str, Any] = {}
    limit = memory_limit if memory_limit is not None else os.environ.get("HTTK_DUCKDB_MEMORY_LIMIT")
    if limit:
        config["memory_limit"] = limit
    if threads is None:
        raw = os.environ.get("HTTK_DUCKDB_THREADS", "").strip()
        if raw:
            if not raw.isdecimal() or int(raw) < 1:
                raise ValueError(f"HTTK_DUCKDB_THREADS must be a positive integer, got {raw!r}")
            config["threads"] = int(raw)
    elif isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError(f"threads must be a positive integer, got {threads!r}")
    else:
        config["threads"] = threads
    return config


def database(
    cls: "type[Backend]",
    path: str | os.PathLike[str] | None = None,
    *,
    memory_limit: str | None = None,
    threads: int | None = None,
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
    :param threads: An optional cap on DuckDB's worker ``threads`` (a positive
        integer). DuckDB's own default is one thread per CPU per instance, which
        multiplies across parallel processes; when this parameter is ``None`` the
        ``HTTK_DUCKDB_THREADS`` environment variable (if set) supplies the cap.
    :param read_only: Open the file in DuckDB ``READ_ONLY`` access mode. A read-only
        database takes no write lock, so several processes (and this one) may open the
        same file concurrently for reading; write operations on the resulting backend
        will fail. Ignored for the in-memory database.
    :return: The configured backend wrapper.
    :raises ImportError: If the ``duckdb_engine`` SQLAlchemy dialect is not installed;
        install the ``httk-store[duckdb]`` extra to use it.
    :raises ValueError: If ``threads`` or ``HTTK_DUCKDB_THREADS`` is not a positive integer.
    """
    try:
        importlib.import_module("duckdb_engine")
    except ImportError as error:
        raise ImportError(
            "the DuckDB backend needs the 'duckdb_engine' SQLAlchemy dialect; "
            "install the 'httk-store[duckdb]' extra to use Backend.duckdb()"
        ) from error
    location = ":memory:" if path is None else os.fspath(path)
    config = _resource_config(memory_limit, threads)
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
