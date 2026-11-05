"""PostgreSQL-specific ``bulk_ingest`` regressions the parametrized suites miss.

Two properties that only diverge on PostgreSQL:

* NaN float fidelity -- a SQLite shard flattens NaN to NULL, so deferred and
  parallel bulk (which stage through SQLite shards) would lose a NaN that a
  serial ``save()`` on PostgreSQL's ``double precision`` keeps.  The bulk paths
  must reinstate NaN so the query-visible float column matches serial.
* Sequence recovery -- a failed bulk must leave no orphan ``<table>_sid_seq``
  behind, so a retry starts from a clean sequence and later saves keep counting.
"""

import math
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import IdentitySkip, StorageInfo
from postgres_support import IsolatedPostgresDatabase, postgres_admin_uri

from httk.store.backend.sql import Backend, SqlStore


@dataclass(frozen=True)
class Reading:
    """A record with a required and a nullable plain-``float`` field.

    ``dedup="none"`` because a NaN float that participates in content identity is
    rejected outright (``nonfinite float values cannot have a content identity``);
    a non-deduplicated record is where a NaN float legitimately reaches storage.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="pgnan_reading", dedup="none")

    label: str
    required: float
    optional: float | None = None


@pytest.fixture
def pg_stores():
    """Yield a factory of fresh isolated PostgreSQL stores, dropped on teardown."""
    postgres_admin_uri()  # skip early when no admin URI is configured
    created: list[tuple[Backend, IsolatedPostgresDatabase]] = []

    def make() -> tuple[SqlStore, Backend]:
        isolated = IsolatedPostgresDatabase()
        database = Backend.postgresql(isolated.uri)
        created.append((database, isolated))
        return SqlStore(database, entry_records={}), database

    try:
        yield make
    finally:
        for database, isolated in created:
            database.dispose()
            isolated.drop()


def _stored_floats(database: Backend) -> list[tuple[float | None, float | None]]:
    """Read the query-visible float columns straight from the table (no store cache)."""
    with database.engine.connect() as connection:
        return [
            (row[0], row[1])
            for row in connection.execute(sqlalchemy.text('SELECT required, optional FROM pgnan_reading ORDER BY sid'))
        ]


@pytest.mark.parametrize("mode", ["serial", "deferred", "parallel"])
def test_bulk_preserves_nan_floats_like_serial(pg_stores, mode):
    """A required and a nullable NaN float survive every ingest path as NaN, not NULL."""
    store, database = pg_stores()
    record = Reading(label="probe", required=math.nan, optional=math.nan)
    if mode == "serial":
        store.save(record)
    elif mode == "deferred":
        with store.bulk_ingest(finalize="deferred") as bulk:
            bulk.save(record)
    else:
        with store.bulk_ingest(workers=2) as bulk:
            bulk.save(record)

    ((required, optional),) = _stored_floats(database)
    # Compare query-visible state: NaN, matching a serial PostgreSQL save (not NULL).
    assert required is not None and math.isnan(required), mode
    assert optional is not None and math.isnan(optional), mode


@dataclass(frozen=True)
class Series:
    """A content-deduplicated record whose identity-excluded float sequence may hold NaN."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="pgnan_series")

    label: str
    series: Annotated[tuple[float, ...], IdentitySkip()] = ()


@pytest.mark.parametrize("mode", ["serial", "deferred", "parallel"])
def test_bulk_preserves_nan_float_sequence_elements_like_serial(pg_stores, mode):
    """A NaN child element survives the SQLite-shard staging, restored by (parent sid, index)."""
    store, database = pg_stores()
    record = Series(label="probe", series=(1.5, math.nan, -0.0, math.nan))
    if mode == "serial":
        store.save(record)
    else:
        # Deferred/parallel bulk cannot verify identity-excluded child sequences against duplicates.
        options = {"finalize": "deferred"} if mode == "deferred" else {"workers": 2}
        with store.bulk_ingest(verify_metadata=False, **options) as bulk:
            bulk.save(record)

    with database.engine.connect() as connection:
        stored = [
            row[0]
            for row in connection.execute(
                sqlalchemy.text("SELECT series FROM pgnan_series_series ORDER BY series_index")
            )
        ]
    assert stored[0] == 1.5 and stored[2].hex() == "0x0.0p+0", mode  # signed zero is stored as 0.0
    assert math.isnan(stored[1]) and math.isnan(stored[3]), mode


def test_failed_bulk_leaves_no_orphan_sequence_and_recovers(pg_stores):
    """A failed bulk drops its sequence so a retry plus a later save keep counting sids."""
    store, database = pg_stores()

    with pytest.raises(RuntimeError, match="boom"), store.bulk_ingest(finalize="deferred") as bulk:
        bulk.save(Reading(label="a", required=1.0))
        raise RuntimeError("boom")

    with database.engine.connect() as connection:
        orphans = connection.execute(
            sqlalchemy.text(
                "SELECT sequencename FROM pg_sequences "
                "WHERE schemaname = current_schema() AND sequencename = 'pgnan_reading_sid_seq'"
            )
        ).all()
    assert orphans == []

    # A clean bulk rebuilds the table and its sequence; the following incremental
    # save must draw a fresh, non-colliding sid rather than reusing a stale one.
    with store.bulk_ingest(finalize="deferred") as bulk:
        bulk.save(Reading(label="b", required=2.0))
    sid = store.save(Reading(label="c", required=3.0))
    assert sid == 2
