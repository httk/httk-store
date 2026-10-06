"""Deferred bulk finalize with a StrongLink child table that received no staged rows.

A ``DataRecord`` saved without ``product_of`` next to typed records that carry edges leaves the
shared ``core_run_edge`` table staged while ``core_data_record_product_of`` has no stage relation;
reachability must treat that child table as empty instead of failing with ``KeyError``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from clickhouse_read_support import CLICKHOUSE_PARAM, clickhouse_database
from httk.core import RunEdge
from httk.core.data_records import DataRecord, DataRecordEntry
from httk.core.definition_ids import TOTAL_ENERGY
from httk.core.property_records import TemperatureRecord
from postgres_support import POSTGRES_PARAM, postgres_database

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore

ENTRY_RECORDS = {DataRecordEntry: (DataRecord, TemperatureRecord)}
SHARED = RunEdge("structure", "structures", "ext-shared")
RECORDS = (
    DataRecord.from_value(TOTAL_ENERGY, "e", -1.0),
    TemperatureRecord(300.5, product_of=(SHARED,)),
    TemperatureRecord(1000.25, product_of=(SHARED, RunEdge("run", "runs", "ext-run"))),
)


@contextmanager
def _database(param: str) -> Iterator[Any]:
    if param == "clickhousedb":
        context: Any = clickhouse_database()
    elif param == "postgresql":
        context = postgres_database()
    elif param == "duckdb":
        pytest.importorskip("duckdb_engine")
        context = Backend.duckdb()
    else:
        context = Backend.sqlite()
    with context as database:
        yield database


def _stored(store: SqlStore, backing: type) -> list[Any]:
    search = store.searcher()
    variable = search.variable(backing)
    return [row.record for row in search.results(record=variable)]


@pytest.mark.parametrize("param", ("sqlite", "duckdb", CLICKHOUSE_PARAM, POSTGRES_PARAM))
def test_deferred_ingest_with_unstaged_strong_link_child_round_trips(param: str) -> None:
    with _database(param) as database:
        store = SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
        with store.bulk_ingest(finalize="deferred") as bulk:
            for record in RECORDS:
                bulk.save(record)
        (data,) = _stored(store, DataRecord)
        assert data == RECORDS[0] and data.product_of == ()
        temperatures = _stored(store, TemperatureRecord)
        assert sorted(temperatures, key=lambda record: record.value) == list(RECORDS[1:])
        assert [record.product_of for record in sorted(temperatures, key=lambda record: record.value)] == [
            record.product_of for record in RECORDS[1:]
        ]
