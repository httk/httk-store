"""``DerivedDataRecord`` storage in the ``records`` family and its additive upgrade from a DataRecord-only store.

SQLite always runs; the Mongo arm needs ``HTTK_TEST_MONGODB_URI`` and is skipped without it.
"""

from pathlib import Path

import pytest
from httk.core import DataRecord, DataRecordEntry, DerivedDataRecord

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore
from httk.store.storage_layout import StorageLayoutUpgradeRequiredError

DEFINITION = "https://example.org/defs/total_energy"
STDERR = "https://example.org/derivations/standard_error"
RMSE = "https://example.org/derivations/rmse"
IDS = EntryIdScheme("httk.test", "1")
BOTH = {DataRecordEntry: (DataRecord, DerivedDataRecord)}
ONLY_DATA = {DataRecordEntry: DataRecord}


def derived(derivation: str = STDERR, value: float = 0.5) -> DerivedDataRecord:
    return DerivedDataRecord.from_value(DEFINITION, derivation, "energy", value)


@pytest.fixture(params=["sqlite", "mongo"])
def make_store(request, tmp_path: Path):
    """Return ``open(records, **kw)`` reopening one persistent database on each call."""
    if request.param == "sqlite":

        def open_sql(records, **kwargs):
            return SqlStore(Backend.sqlite(tmp_path / "s.sqlite"), entry_records=records, entry_ids=IDS, **kwargs)

        yield open_sql
        return
    from httk.store.backend.mongo import MongoStore

    database = request.getfixturevalue("mongo_test_database")

    def open_mongo(records, **kwargs):
        return MongoStore(database, entry_records=records, entry_ids=IDS, **kwargs)

    yield open_mongo


def test_save_reopen_fetch(make_store) -> None:
    store = make_store(BOTH)
    record = derived()
    sid = store.save(record)
    store.close()
    fetched = make_store(BOTH).fetch(DerivedDataRecord, sid)
    assert fetched == record
    assert (fetched.derivation, fetched.value) == (STDERR, 0.5)
    assert fetched.id is not None and fetched.immutable_id is not None


def test_derivation_is_identity(make_store) -> None:
    store = make_store(BOTH)
    first = store.save(derived(STDERR))
    second = store.save(derived(RMSE))
    assert first != second
    assert store.fetch(DerivedDataRecord, first).derivation == STDERR
    assert store.fetch(DerivedDataRecord, second).derivation == RMSE
    assert store.save(derived(STDERR)) == first  # identical content dedups


def test_additive_upgrade_from_data_record_only(make_store) -> None:
    store = make_store(ONLY_DATA)
    old = DataRecord.from_value(DEFINITION, "energy", -1.0)
    sid = store.save(old)
    before = store.fetch(DataRecord, sid)
    store.close()

    with pytest.raises(StorageLayoutUpgradeRequiredError) as refused:
        make_store(BOTH)
    assert refused.value.remedy == "upgrade"

    upgraded = make_store(BOTH, upgrade=True)
    after = upgraded.fetch(DataRecord, sid)
    assert (after.id, after.immutable_id, after.value_json) == (before.id, before.immutable_id, before.value_json)
    assert upgraded.save(old) == sid  # same content id: still deduplicated

    upgraded.save(derived(STDERR))
    upgraded.save(derived(RMSE, 2.0))
    search = upgraded.searcher()
    variable = search.variable(DerivedDataRecord)
    search.add(variable.definition_id == DEFINITION)
    search.add(variable.derivation == RMSE)
    rows = [row["derived"] for row in search.results(derived=variable)]
    assert [row.value for row in rows] == [2.0]


def test_value_number_is_queryable(make_store) -> None:
    store = make_store(BOTH)
    for value in (3.0, -1.5, 2.0):
        store.save(derived(STDERR, value))
    store.save(DerivedDataRecord.from_value(DEFINITION, RMSE, "energy", "text"))  # non-numeric: null projection

    search = store.searcher()
    variable = search.variable(DerivedDataRecord)
    search.add(variable.derivation == STDERR)
    search.add(variable.value_number >= 0.0)
    assert sorted(search.results(value=variable.value_number).scalars()) == [2.0, 3.0]

    search = store.searcher()
    variable = search.variable(DerivedDataRecord)
    search.add(variable.value_number == -1.5)
    rows = [row["derived"] for row in search.results(derived=variable)]
    assert [row.value for row in rows] == [-1.5]
