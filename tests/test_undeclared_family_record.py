"""A store that declares entry families refuses to save records it does not declare for them."""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core import DataRecord, DataRecordEntry, DerivedDataRecord
from httk.core.references import ReferenceRecord
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore

IDS = EntryIdScheme("httk.test", "1")
DECLARED = {DataRecordEntry: DerivedDataRecord}


def _derived() -> DerivedDataRecord:
    return DerivedDataRecord.from_value("https://example.org/defs/e", "https://example.org/d/x", "energy", 0.5)


def _data() -> DataRecord:
    return DataRecord.from_value("https://example.org/defs/e", "energy", 0.5)


@dataclass(frozen=True)
class Plain:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="undeclared_family_plain")

    label: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


def _tables(store) -> set[str]:
    return set(sqlalchemy.inspect(store._database.engine).get_table_names())


@pytest.fixture(params=("sqlite", "duckdb"))
def make_store(request):
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
    database = getattr(Backend, request.param)()
    with database:
        yield lambda records: SqlStore(database, entry_records=records, entry_ids=IDS)


def test_undeclared_record_of_declared_family_refused(make_store) -> None:
    store = make_store(DECLARED)
    before = _tables(store)
    with pytest.raises(ValueError, match="declares DerivedDataRecord .*not DataRecord"):
        store.save(_data())
    assert _tables(store) == before
    assert store.save(_derived())


def test_record_of_undeclared_family_refused(make_store) -> None:
    store = make_store(DECLARED)
    before = _tables(store)
    with pytest.raises(ValueError, match="'files' entry family|'references' entry family"):
        store.save(ReferenceRecord(title="t"))
    assert _tables(store) == before


def test_no_families_saves_anything(make_store) -> None:
    store = make_store({})
    assert store.save(_derived()) is not None
    assert store.save(Plain("a")) is not None


def test_plain_dataclass_saves_into_family_store(make_store) -> None:
    assert make_store(DECLARED).save(Plain("a")) is not None


def test_bulk_ingest_refuses_undeclared(make_store) -> None:
    store = make_store(DECLARED)
    if not hasattr(store, "bulk_ingest"):
        pytest.skip("backend does not provide bulk_ingest")
    before = _tables(store)
    with pytest.raises(ValueError, match="not DataRecord"), store.bulk_ingest() as bulk:
        bulk.save(_data())
    assert _tables(store) == before
    with store.bulk_ingest() as bulk:
        bulk.save(_derived())


def test_mongo_refuses_undeclared(mongo_test_database) -> None:
    from httk.store.backend.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_records=DECLARED, entry_ids=IDS)
    before = set(mongo_test_database._database.list_collection_names())
    with pytest.raises(ValueError, match="not DataRecord"):
        store.save(_data())
    assert set(mongo_test_database._database.list_collection_names()) == before
    assert store.save(_derived())


def test_type_field_is_not_an_entry_type(make_store) -> None:
    @dataclass(frozen=True)
    class WithTypeField:
        __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="undeclared_family_type_field")

        type: str
        label: str
        id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
        immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    assert make_store(DECLARED).save(WithTypeField("records", "x")) is not None
