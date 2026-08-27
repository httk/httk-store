"""Entry-id declaration validation and SQL save/bulk minting."""

import json
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique

from httk.store import EntryIdConflictError, EntryIdScheme
from httk.store.backend.schema import SchemaError, resolve_schema
from httk.store.backend.sql import Backend, SqlStore
from httk.store.storage_layout import (
    EntryFamilyDeclaration,
    EntryRecordDeclaration,
    classify_schema_upgrade,
    normalize_entry_families,
    schema_fingerprint_json,
)
from httk.store.store_common import EntryMetadataConflictError


class EntryIdFamily:
    type = "widgets"


@dataclass(frozen=True)
class EntryIdRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class MissingEntryIdRecord:
    value: int


@dataclass(frozen=True)
class InvalidEntryIdRecord:
    value: int
    id: Annotated[str, Indexed()] = "required"
    immutable_id: Annotated[str, Unique()] = "required~1"


@dataclass(frozen=True)
class UniqueIdEntryIdRecord:
    value: int
    id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class ComparableEntryIdRecord:
    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = None
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = None


class MultiBackingEntryIdFamily:
    type = "multiwidgets"


@dataclass(frozen=True)
class MultiBackingFirst:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class MultiBackingSecond:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class ChildContainerFamily:
    pass


@dataclass(frozen=True)
class ChildContainer:
    children: tuple[EntryIdRecord, ...]


def _declaration(record: type = EntryIdRecord) -> EntryFamilyDeclaration:
    return EntryFamilyDeclaration(
        name="test-entry-ids",
        family=EntryIdFamily,
        records=(EntryRecordDeclaration(name="test-entry-id-record", record=record),),
        definition_id="urn:httk:test:entry-ids",
    )


def _store(*, scheme: EntryIdScheme | None = None) -> tuple[Backend, SqlStore]:
    database = Backend.sqlite()
    scheme = EntryIdScheme("httk.test", "1") if scheme is None else scheme
    return database, SqlStore(database, entry_families=(_declaration(),), entry_ids=scheme)


def _multi_store() -> tuple[Backend, SqlStore]:
    database = Backend.sqlite()
    declaration = EntryFamilyDeclaration(
        name="test-entry-ids-multi",
        family=MultiBackingEntryIdFamily,
        records=(
            EntryRecordDeclaration(name="test-entry-id-multi-first", record=MultiBackingFirst),
            EntryRecordDeclaration(name="test-entry-id-multi-second", record=MultiBackingSecond),
        ),
        definition_id="urn:httk:test:entry-ids:multi",
    )
    return database, SqlStore(database, entry_families=(declaration,), entry_ids=EntryIdScheme("httk.test", "1"))


def _fetched(store: SqlStore, sid: int) -> EntryIdRecord:
    store._clear_identity_caches()
    return store.fetch(EntryIdRecord, sid, eager=True)


def test_defined_entry_family_requires_id_declarations() -> None:
    with Backend.sqlite() as database, pytest.raises(SchemaError, match="id: Annotated"):
        SqlStore(database, entry_families=(_declaration(MissingEntryIdRecord),))


def test_defined_entry_family_requires_optional_identity_skipped_id_declarations() -> None:
    with Backend.sqlite() as database, pytest.raises(SchemaError, match="id: Annotated"):
        SqlStore(database, entry_families=(_declaration(InvalidEntryIdRecord),))


@pytest.mark.parametrize("record", (UniqueIdEntryIdRecord, ComparableEntryIdRecord))
def test_defined_entry_family_requires_non_unique_id_and_compare_false(record: type) -> None:
    with Backend.sqlite() as database, pytest.raises(SchemaError, match="id: Annotated"):
        SqlStore(database, entry_families=(_declaration(record),))


def test_entry_id_fields_are_not_an_additive_upgrade() -> None:
    current = schema_fingerprint_json(normalize_entry_families((_declaration(),)))
    old_document = json.loads(current)
    fields = old_document["tables"][resolve_schema(EntryIdRecord).table_name]["fields"]
    fields.pop("id")
    fields.pop("immutable_id")
    reason = classify_schema_upgrade(json.dumps(old_document), current)
    assert isinstance(reason, str)
    assert "immutable_id" in reason or "id" in reason
    assert "rebuild the store" in reason


def test_save_mints_ids_type_base_and_series_override() -> None:
    database, store = _store()
    try:
        sid = store.save(EntryIdRecord(1), id_series="2")
        record = _fetched(store, sid)
        assert record.id == f"httk.test-2-{sid}"
        assert record.immutable_id == f"{record.id}~1"
    finally:
        database.dispose()
    database, store = _store(scheme=EntryIdScheme("httk.test", "1", type_in_base=True))
    try:
        sid = store.save(EntryIdRecord(1))
        assert _fetched(store, sid).id == f"httk.test.widgets-1-{sid}"
    finally:
        database.dispose()


def test_entry_save_and_dedup_keep_reverse_identity_but_hydrate_minted_ids() -> None:
    database, store = _store()
    try:
        original = EntryIdRecord(1)
        sid = store.save(original)
        fetched = store.fetch(EntryIdRecord, sid, eager=True)
        assert fetched.id == f"httk.test-1-{sid}"
        assert fetched.immutable_id == f"{fetched.id}~1"
        assert store.sid_of(original) == sid

        replacement = store.replace(original, EntryIdRecord(2))
        assert store.fetch(EntryIdRecord, replacement, eager=True).id == fetched.id

        duplicate = EntryIdRecord(1)
        assert store.save(duplicate) == sid
        assert store.sid_of(duplicate) == sid
        assert store.fetch(EntryIdRecord, sid, eager=True).immutable_id == f"{fetched.id}~1"
    finally:
        database.dispose()


def test_replace_keeps_lineage_id_and_increments_immutable_id() -> None:
    database, store = _store()
    try:
        a = EntryIdRecord(1)
        first = store.save(a)
        second = store.replace(a, EntryIdRecord(2))
        third = store.replace(_fetched(store, second), EntryIdRecord(3))
        records = [_fetched(store, sid) for sid in (first, second, third)]
        assert [record.id for record in records] == [records[0].id] * 3
        assert [record.immutable_id for record in records] == [f"{records[0].id}~{n}" for n in (1, 2, 3)]
        assert [record.value for record in store.history(_fetched(store, third))] == [1, 2, 3]
    finally:
        database.dispose()


def test_conflicts_and_content_metadata_are_checked() -> None:
    database, store = _store()
    try:
        first = EntryIdRecord(1)
        store.save(first)
        with pytest.raises(EntryIdConflictError):
            store.replace(first, EntryIdRecord(2, id="httk.test-1-999"))
        with pytest.raises(EntryIdConflictError):
            store.replace(first, EntryIdRecord(1, id="httk.test-1-999"))
        with pytest.raises(EntryIdConflictError):
            store.save(EntryIdRecord(2, id=_fetched(store, 1).id))
        store.save(EntryIdRecord(3, id="httk.test-1-3", immutable_id="httk.test-1-3~1"))
        with pytest.raises(EntryIdConflictError):
            store.save(EntryIdRecord(4, immutable_id="httk.test-1-3~1"))
        assert store.save(EntryIdRecord(1)) == 1
        with pytest.raises(EntryMetadataConflictError):
            store.save(EntryIdRecord(1, id="httk.test-1-100"))
    finally:
        database.dispose()


def test_invalid_ids_are_validated_before_deduplication_or_writes() -> None:
    database, store = _store()
    try:
        with pytest.raises(ValueError):
            store.save(EntryIdRecord(1, immutable_id="source/mixed"))
        assert store.save(EntryIdRecord(2)) == 1
        with pytest.raises(ValueError):
            store.save(EntryIdRecord(2, id="source/mixed"))
    finally:
        database.dispose()


def test_nested_entry_records_honor_id_series() -> None:
    database = Backend.sqlite()
    container = EntryFamilyDeclaration(
        name="test-entry-id-child-container",
        family=ChildContainerFamily,
        records=(EntryRecordDeclaration(name="test-entry-id-child-container-record", record=ChildContainer),),
    )
    store = SqlStore(
        database,
        entry_families=(_declaration(), container),
        entry_ids=EntryIdScheme("httk.test", "1"),
    )
    try:
        store.save(ChildContainer((EntryIdRecord(1),)), id_series="override")
        assert _fetched(store, 1).id == "httk.test-override-1"
    finally:
        database.dispose()


def test_validation_scheme_requirement_and_bulk_final_sids(caplog: pytest.LogCaptureFixture) -> None:
    database, store = _store()
    try:
        with caplog.at_level("WARNING", logger="httk.core.entry_ids"):
            store.save(EntryIdRecord(1, id="anyt:am-1-12"))
        assert len(caplog.records) == 1
        with pytest.raises(ValueError):
            store.save(EntryIdRecord(2, id="a/b"))
    finally:
        database.dispose()
    for mode in ({"finalize": "parity"}, {"finalize": "deferred"}, {"workers": 2}):
        database, bulk_store = _store()
        try:
            with bulk_store.bulk_ingest(**mode) as bulk:
                sid = bulk.save(EntryIdRecord(10 + len(mode)))
            final_sid = bulk.resolved_sid(EntryIdRecord, sid)
            record = _fetched(bulk_store, final_sid)
            assert record.id == f"httk.test-1-{final_sid}"
            assert record.immutable_id == f"{record.id}~1"
        finally:
            database.dispose()
    database = Backend.sqlite()
    absent = SqlStore(database, entry_families=(_declaration(),), entry_ids=None)
    try:
        with pytest.raises(ValueError, match="EntryIdScheme"):
            absent.save(EntryIdRecord(1))
    finally:
        database.dispose()


def test_multi_backing_family_mints_family_unique_numbers_in_save_replace_and_bulk() -> None:
    database, store = _multi_store()
    try:
        first = MultiBackingFirst(1)
        first_sid = store.save(first)
        second_sid = store.save(MultiBackingSecond(2))
        store._clear_identity_caches()
        first_row = store.fetch(MultiBackingFirst, first_sid, eager=True)
        second_row = store.fetch(MultiBackingSecond, second_sid, eager=True)
        assert first_row.id == "httk.test-1-2"
        assert second_row.id == "httk.test-1-3"
        replacement_sid = store.replace(first_row, MultiBackingFirst(3))
        store._clear_identity_caches()
        assert store.fetch(MultiBackingFirst, replacement_sid, eager=True).id == first_row.id
    finally:
        database.dispose()
    for mode in ({"finalize": "parity"}, {"finalize": "deferred"}, {"workers": 2}):
        database, store = _multi_store()
        try:
            with store.bulk_ingest(**mode) as bulk:
                first_sid = bulk.save(MultiBackingFirst(10))
                second_sid = bulk.save(MultiBackingSecond(20))
            first_sid = bulk.resolved_sid(MultiBackingFirst, first_sid)
            second_sid = bulk.resolved_sid(MultiBackingSecond, second_sid)
            store._clear_identity_caches()
            first_row = store.fetch(MultiBackingFirst, first_sid, eager=True)
            second_row = store.fetch(MultiBackingSecond, second_sid, eager=True)
            assert first_row.id == "httk.test-1-2"
            assert second_row.id == "httk.test-1-3"
            assert first_row.id != second_row.id
        finally:
            database.dispose()


def test_explicit_ids_are_unique_across_family_backings() -> None:
    database, store = _multi_store()
    try:
        store.save(MultiBackingFirst(1, id="httk.test-1-700", immutable_id="httk.test-1-700~1"))
        with pytest.raises(EntryIdConflictError):
            store.save(MultiBackingSecond(2, id="httk.test-1-700", immutable_id="httk.test-1-700~1"))
    finally:
        database.dispose()


def test_serial_bulk_explicit_id_conflicts_with_preexisting_family_id() -> None:
    database, store = _multi_store()
    try:
        store.save(MultiBackingFirst(1, id="httk.test-1-701", immutable_id="httk.test-1-701~1"))
        with pytest.raises(EntryIdConflictError), store.bulk_ingest(finalize="parity") as bulk:
            bulk.save(MultiBackingSecond(2, id="httk.test-1-701"))
    finally:
        database.dispose()


@pytest.mark.parametrize(
    ("first", "second"),
    (
        (
            MultiBackingFirst(1, id="httk.test-1-710"),
            MultiBackingSecond(2, id="httk.test-1-710"),
        ),
        (
            MultiBackingFirst(1, id="httk.test-1-711", immutable_id="httk.test-1-711~1"),
            MultiBackingSecond(2, id="httk.test-1-712", immutable_id="httk.test-1-711~1"),
        ),
    ),
)
def test_serial_bulk_duplicate_explicit_ids_across_backings_are_rejected(
    first: MultiBackingFirst, second: MultiBackingSecond
) -> None:
    database, store = _multi_store()
    try:
        with pytest.raises(EntryIdConflictError), store.bulk_ingest(finalize="parity") as bulk:
            bulk.save(first)
            bulk.save(second)
    finally:
        database.dispose()


@pytest.mark.parametrize("mode", ({"finalize": "parity"}, {"workers": 2}))
def test_bulk_duplicate_explicit_ids_are_rejected(mode: dict[str, object]) -> None:
    database, store = _store()
    try:
        with pytest.raises(EntryIdConflictError), store.bulk_ingest(**mode) as bulk:  # type: ignore[arg-type]
            bulk.save(EntryIdRecord(1, id="httk.test-1-702"))
            bulk.save(EntryIdRecord(2, id="httk.test-1-702"))
    finally:
        database.dispose()


@pytest.mark.parametrize("mode", ({"finalize": "parity"}, {"finalize": "deferred"}, {"workers": 2}))
def test_bulk_explicit_ids_do_not_require_a_scheme(mode: dict[str, object]) -> None:
    database = Backend.sqlite()
    store = SqlStore(database, entry_families=(_declaration(),), entry_ids=None)
    try:
        with store.bulk_ingest(**mode) as bulk:  # type: ignore[arg-type]
            sid = bulk.save(EntryIdRecord(1, id="httk.test-1-703"))
        final_sid = bulk.resolved_sid(EntryIdRecord, sid)
        record = _fetched(store, final_sid)
        assert record.id == "httk.test-1-703"
        assert record.immutable_id == "httk.test-1-703~1"
    finally:
        database.dispose()
