"""Live MongoDB parity coverage for entry-id declaration and save semantics."""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique

from httk.store import EntryIdConflictError, EntryIdScheme
from httk.store.backend.mongo import MongoStore
from httk.store.backend.schema import SchemaError
from httk.store.storage_layout import EntryFamilyDeclaration, EntryRecordDeclaration
from httk.store.store_common import EntryMetadataConflictError


class MongoEntryFamily:
    """A test family with a stable served type."""

    type = "widgets"


@dataclass(frozen=True)
class MongoEntry:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class MissingMongoEntry:
    value: int


class MongoMultiFamily:
    """A test family with two physical backings."""

    type = "multiwidgets"


@dataclass(frozen=True)
class MongoFirst:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class MongoSecond(MongoFirst):
    pass


class MongoContainerFamily:
    """An unserved container family for nested-save coverage."""


@dataclass(frozen=True)
class MongoContainer:
    children: tuple[MongoEntry, ...]


def _declaration(record: type = MongoEntry) -> EntryFamilyDeclaration:
    return EntryFamilyDeclaration(
        name="test-mongo-entry-ids",
        family=MongoEntryFamily,
        records=(EntryRecordDeclaration(name="test-mongo-entry", record=record),),
        definition_id="urn:httk:test:mongo-entry-ids",
    )


def _store(database, *, scheme: EntryIdScheme | None = None) -> MongoStore:
    return MongoStore(
        database,
        entry_families=(_declaration(),),
        entry_ids=EntryIdScheme("httk.test", "1") if scheme is None else scheme,
    )


def _fetch(store: MongoStore, record: type, sid: int):
    return store.fetch(record, sid)


def test_defined_entry_family_requires_id_declarations(mongo_test_database) -> None:
    with pytest.raises(SchemaError, match="id: Annotated"):
        MongoStore(mongo_test_database, entry_families=(_declaration(MissingMongoEntry),))


def test_save_mints_type_base_and_series_override(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    sid = store.save(MongoEntry(1), id_series="2")
    saved = _fetch(store, MongoEntry, sid)
    assert saved.id == f"httk.test-2-{sid}"
    assert saved.immutable_id == f"{saved.id}~1"
    with store.transaction():
        transaction_sid = store.save(MongoEntry(2))
        transaction_record = store.fetch(MongoEntry, transaction_sid)
        assert transaction_record.id == f"httk.test-1-{transaction_sid}"
        assert transaction_record.immutable_id == f"{transaction_record.id}~1"


def test_type_in_base_replace_and_conflicts(mongo_test_database) -> None:
    store = _store(mongo_test_database, scheme=EntryIdScheme("httk.test", "1", type_in_base=True))
    first = MongoEntry(1)
    a = store.save(first)
    b = store.replace(first, MongoEntry(2))
    c = store.replace(_fetch(store, MongoEntry, b), MongoEntry(3))
    saved = [_fetch(store, MongoEntry, sid) for sid in (a, b, c)]
    assert [item.id for item in saved] == [f"httk.test.widgets-1-{a}"] * 3
    assert [item.immutable_id for item in saved] == [f"{saved[0].id}~{n}" for n in (1, 2, 3)]
    with pytest.raises(EntryIdConflictError):
        store.replace(first, MongoEntry(4, id="httk.test.widgets-1-99"))
    with pytest.raises(EntryIdConflictError):
        store.replace(first, MongoEntry(1, id="httk.test.widgets-1-98"))
    with pytest.raises(EntryIdConflictError):
        store.save(MongoEntry(5, id=saved[0].id))
    store.save(MongoEntry(6, id="httk.test-1-6", immutable_id="httk.test-1-6~1"))
    with pytest.raises(EntryIdConflictError):
        store.save(MongoEntry(7, immutable_id="httk.test-1-6~1"))


def test_dedup_validation_and_absent_scheme(mongo_test_database, caplog: pytest.LogCaptureFixture) -> None:
    store = _store(mongo_test_database)
    with caplog.at_level("WARNING", logger="httk.core.entry_ids"):
        store.save(MongoEntry(1, id="anyt:am-1-12"))
    assert caplog.records
    with pytest.raises(ValueError):
        store.save(MongoEntry(2, id="a/b"))
    assert store.save(MongoEntry(1)) == 1
    with pytest.raises(EntryMetadataConflictError):
        store.save(MongoEntry(1, id="httk.test-1-100"))

    absent = MongoStore(mongo_test_database, entry_families=(_declaration(),), entry_ids=None)
    with pytest.raises(ValueError, match="EntryIdScheme"):
        absent.save(MongoEntry(9))


def test_multi_backing_and_nested_series(mongo_test_database) -> None:
    multi = EntryFamilyDeclaration(
        name="test-mongo-entry-ids-multi",
        family=MongoMultiFamily,
        records=(
            EntryRecordDeclaration(name="test-mongo-first", record=MongoFirst),
            EntryRecordDeclaration(name="test-mongo-second", record=MongoSecond),
        ),
        definition_id="urn:httk:test:mongo-entry-ids:multi",
    )
    container = EntryFamilyDeclaration(
        name="test-mongo-entry-ids-container",
        family=MongoContainerFamily,
        records=(EntryRecordDeclaration(name="test-mongo-container", record=MongoContainer),),
    )
    store = MongoStore(
        mongo_test_database,
        entry_families=(multi, _declaration(), container),
        entry_ids=EntryIdScheme("httk.test", "1"),
    )
    first_sid = store.save(MongoFirst(1, id="httk.test-1-700", immutable_id="httk.test-1-700~1"))
    with pytest.raises(EntryIdConflictError):
        store.save(MongoSecond(2, id="httk.test-1-700", immutable_id="httk.test-1-700~2"))
    with pytest.raises(EntryIdConflictError):
        store.save(MongoSecond(2, id="httk.test-1-701", immutable_id="httk.test-1-700~1"))
    second_sid = store.save(MongoSecond(2))
    first = _fetch(store, MongoFirst, first_sid)
    second = _fetch(store, MongoSecond, second_sid)
    assert first.id == "httk.test-1-700"
    assert second.id == f"httk.test-1-{second_sid * 2 + 1}"
    with pytest.raises(EntryIdConflictError):
        store.save(MongoSecond(3, id=first.id))
    store.save(MongoContainer((MongoEntry(10),)), id_series="nested")
    assert _fetch(store, MongoEntry, 1).id == "httk.test-nested-1"
