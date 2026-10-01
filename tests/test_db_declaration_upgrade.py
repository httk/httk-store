"""Additive declaration upgrades: appending record kinds to a family, or adding a family, to an existing store.

Covers the pure classifier, the crash-convergent SQL application (target-layout
reserved objects, dispatch rebuild, entry-id offsets, compare-and-set restamp),
the stale-writer guard, and the invariant that stores whose record lists never
change mint exactly as before.  SQLite and DuckDB run locally; the PostgreSQL
arm is gated by ``HTTK_TEST_POSTGRES_URI``.
"""

import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, cast

import pytest
import sqlalchemy
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique, content_id
from postgres_support import POSTGRES_PARAM, postgres_database
from sqlalchemy.exc import IntegrityError

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql.layout import actual_table_names
from httk.store.backend.sql.mapping import (
    backing_dispatch_column_name,
    entry_dispatch_table_name,
    identity_owner_tables,
)
from httk.store.storage_layout import (
    ADDITIVE_DECLARATION_UPGRADE_HINT,
    ENTRY_ID_OFFSETS_KEY,
    AdditiveUpgradePlan,
    DeclarationUpgradePlan,
    EntryFamilyDeclaration,
    EntryRecordDeclaration,
    StorageLayoutUpgradeRequiredError,
    classify_declaration_upgrade,
    classify_schema_upgrade,
    declaration_json,
    entry_id_number,
    entry_id_offsets_json,
    next_entry_id_offset,
    normalize_entry_families,
    parse_entry_id_offsets,
    schema_fingerprint_diff,
    schema_fingerprint_json,
)

FAMILY = "decl-up"
DEFINITION = "urn:test:decl-up"


class UpFamily:
    type = "decl_up"


class OtherFamily:
    type = "decl_up_other"


@dataclass(frozen=True)
class UpA:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="decl_up_a", identity_name="decl_up_a")
    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class UpB(UpA):
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="decl_up_b", identity_name="decl_up_b")


@dataclass(frozen=True)
class UpC(UpA):
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="decl_up_c", identity_name="decl_up_c")


@dataclass(frozen=True)
class OtherRec(UpA):
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="decl_up_other", identity_name="decl_up_other")


RECORDS = {
    UpA: EntryRecordDeclaration(name="decl-up-a", record=UpA),
    UpB: EntryRecordDeclaration(name="decl-up-b", record=UpB),
    UpC: EntryRecordDeclaration(name="decl-up-c", record=UpC),
}
OTHER = EntryFamilyDeclaration(
    name="decl-up-other",
    family=OtherFamily,
    records=(EntryRecordDeclaration(name="decl-up-other-rec", record=OtherRec),),
    definition_id="urn:test:decl-up-other",
)


def family(*records: type, definition_id: str | None = DEFINITION) -> EntryFamilyDeclaration:
    return EntryFamilyDeclaration(
        name=FAMILY,
        family=UpFamily,
        records=tuple(RECORDS[record] for record in records),
        definition_id=definition_id,
    )


def opened(
    database: Backend, *records: type, extra: tuple[EntryFamilyDeclaration, ...] = (), **kwargs: Any
) -> SqlStore:
    return SqlStore(
        database,
        entry_families=(family(*records), *extra),
        entry_ids=EntryIdScheme("up", "1"),
        **kwargs,
    )


@pytest.fixture(params=["sqlite", "duckdb", POSTGRES_PARAM])
def database(request: pytest.FixtureRequest) -> Iterator[Backend]:
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Backend.duckdb() as backend:
            yield backend
        return
    if request.param == "postgresql":
        with postgres_database() as backend:
            yield backend
        return
    with Backend.sqlite() as backend:
        yield backend


@pytest.fixture(params=["sqlite", "duckdb", POSTGRES_PARAM])
def file_database(request: pytest.FixtureRequest, tmp_path: Any) -> Iterator[Backend]:
    """A server- or file-backed database whose engine hands out genuinely separate connections."""
    if request.param == "postgresql":
        with postgres_database() as backend:
            yield backend
        return
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Backend.duckdb(tmp_path / "store.duckdb") as backend:
            yield backend
        return
    with Backend.sqlite(tmp_path / "store.sqlite") as backend:
        yield backend


def declaration_aspects(error: StorageLayoutUpgradeRequiredError) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", error.diff["declaration"])


def metadata(database: Backend) -> dict[str, str]:
    with database.engine.connect() as connection:
        rows = connection.execute(sqlalchemy.text("SELECT key, value FROM _httk_store_metadata")).all()
    return {str(key): str(value) for key, value in rows if key != "lease"}


def table_names(database: Backend) -> set[str]:
    with database.engine.connect() as connection:
        return set(actual_table_names(connection))


def dispatch_rows(database: Backend, *records: type) -> list[tuple[Any, ...]]:
    columns = ", ".join(backing_dispatch_column_name(RECORDS[record].name) for record in records)
    with database.engine.connect() as connection:
        return sorted(
            tuple(row)
            for row in connection.execute(
                sqlalchemy.text(f'SELECT content_id, {columns} FROM "{entry_dispatch_table_name(FAMILY)}"')
            )
        )


def main_row_count(database: Backend, *records: type) -> int:
    total = 0
    present = table_names(database)
    with database.engine.connect() as connection:
        for record in records:
            if record.__httk_storage__.storage_name in present:
                total += connection.execute(
                    sqlalchemy.text(
                        f'SELECT count(*) FROM "{record.__httk_storage__.storage_name}" WHERE _httk_role = 1'
                    )
                ).scalar_one()
    return total


def claim_count(database: Backend) -> int:
    entry_owners, _immutable_owners = identity_owner_tables(sqlalchemy.MetaData())
    with database.engine.connect() as connection:
        return connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(entry_owners)).scalar_one()


def assert_fsck_clean(store: SqlStore, known_types: tuple[type, ...] = ()) -> None:
    summary = store.fsck(collect_garbage=False, exclusive=True, known_types=known_types)
    assert summary.violations == ()
    for name, counters in summary.tables.items():
        assert counters.repaired == 0, name
        assert counters.conflicts == 0, name


def fetched_id(store: SqlStore, cls: type, sid: int) -> str:
    value = store.fetch(cls, sid, eager=True).id
    assert isinstance(value, str)
    return value


def ids_by_value(store: SqlStore, cls: type) -> dict[int, list[str]]:
    search = store.searcher()
    variable = search.variable(cls)
    found: dict[int, list[str]] = {}
    for row in search.results(record=variable):
        found.setdefault(row.record.value, []).append(row.record.id)
    return {value: sorted(ids) for value, ids in found.items()}


@dataclass
class Built:
    """The pre-upgrade entries of a scenario, for post-upgrade assertions."""

    sids: dict[tuple[type, int], int]
    ids: dict[tuple[type, int], str]
    max_number: int


def build_one_backing(database: Backend) -> Built:
    """Family (A): three saves, one replace (a revision gap in the minted numbers), one more save."""
    store = opened(database, UpA)
    sids = {(UpA, value): store.save(UpA(value)) for value in (1, 2, 3)}
    sids[(UpA, 10)] = store.replace(UpA(1), UpA(10))
    sids[(UpA, 4)] = store.save(UpA(4))
    ids = {key: fetched_id(store, key[0], sid) for key, sid in sids.items()}
    # One lineage replaced: the fresh sids are 1, 2, 3, (4 is the revision), 5.
    assert [ids[(UpA, value)] for value in (1, 2, 3, 10, 4)] == ["up-1-1", "up-1-2", "up-1-3", "up-1-1", "up-1-5"]
    return Built(sids, ids, 5)


def build_two_backings(database: Backend) -> Built:
    """Family (A, B): interleaved saves plus a replace."""
    store = opened(database, UpA, UpB)
    sids = {
        (UpA, 1): store.save(UpA(1)),
        (UpB, 1): store.save(UpB(1)),
        (UpA, 2): store.save(UpA(2)),
    }
    sids[(UpA, 10)] = store.replace(UpA(1), UpA(10))
    sids[(UpB, 2)] = store.save(UpB(2))
    ids = {key: fetched_id(store, key[0], sid) for key, sid in sids.items()}
    # number = logical_id * 2 + index: A sids 1, 2, (3 = revision); B sids 1, 2.
    assert ids == {
        (UpA, 1): "up-1-2",
        (UpB, 1): "up-1-3",
        (UpA, 2): "up-1-4",
        (UpA, 10): "up-1-2",
        (UpB, 2): "up-1-5",
    }
    return Built(sids, ids, 5)


SCENARIOS: dict[str, tuple[Callable[[Backend], Built], tuple[type, ...], tuple[type, ...]]] = {
    "1to2": (build_one_backing, (UpA,), (UpA, UpB)),
    "2to3": (build_two_backings, (UpA, UpB), (UpA, UpB, UpC)),
}


def assert_old_entries_intact(store: SqlStore, built: Built) -> None:
    for (cls, value), sid in built.sids.items():
        record = store.fetch(cls, sid, eager=True)
        assert record.value == value
        assert record.id == built.ids[(cls, value)]
        # Through the (rebuilt) dispatch table by content id.
        entry = store.fetch_entry(UpFamily, content_id(cls(value)), eager=True)
        assert entry is not None and type(entry) is cls and entry.value == value
    # Revision history of the replaced lineage is intact.
    revised = store.fetch(UpA, built.sids[(UpA, 10)], eager=True)
    history = store.history(revised)
    assert tuple(item.value for item in history) == (1, 10)
    assert tuple(item.immutable_id for item in history) == (f"{revised.id}~1", f"{revised.id}~2")


# --------------------------------------------------------------------------- pure classifier


def _decl(*records: type, definition_id: str | None = DEFINITION) -> str:
    return declaration_json(normalize_entry_families((family(*records, definition_id=definition_id),)))


def test_classify_appended_record_and_new_family_are_additive() -> None:
    target = normalize_entry_families((family(UpA, UpB, UpC), OTHER))
    plan = classify_declaration_upgrade(_decl(UpA), target)
    assert plan == DeclarationUpgradePlan(changed_families=(FAMILY,), new_families=("decl-up-other",))
    plan = classify_declaration_upgrade(_decl(UpA, UpB, UpC), target)
    assert plan == DeclarationUpgradePlan(changed_families=(), new_families=("decl-up-other",))


@pytest.mark.parametrize(
    ("stored", "target", "fragments"),
    [
        ((UpA, UpB), (UpB, UpA), ("'decl-up'", "'decl-up-a'", "appended")),
        ((UpA, UpB), (UpA,), ("'decl-up'", "no longer declares record 'decl-up-b'")),
        ((UpA, UpB), (UpA, UpC), ("'decl-up'", "'decl-up-b'", "found 'decl-up-c'")),
    ],
)
def test_classify_reorder_removal_and_rename_are_rejected(
    stored: tuple[type, ...], target: tuple[type, ...], fragments: tuple[str, ...]
) -> None:
    reason = classify_declaration_upgrade(_decl(*stored), normalize_entry_families((family(*target),)))
    assert isinstance(reason, str)
    for fragment in fragments:
        assert fragment in reason


def test_classify_changed_definition_ids_and_removed_family_are_rejected() -> None:
    reason = classify_declaration_upgrade(
        _decl(UpA), normalize_entry_families((family(UpA, UpB, definition_id="urn:test:other"),))
    )
    assert isinstance(reason, str) and "family 'decl-up' changed its definition_id" in reason
    retyped = EntryFamilyDeclaration(
        name=FAMILY,
        family=UpFamily,
        records=(EntryRecordDeclaration(name="decl-up-a", record=UpA, definition_id="urn:test:record"), RECORDS[UpB]),
        definition_id=DEFINITION,
    )
    reason = classify_declaration_upgrade(_decl(UpA), normalize_entry_families((retyped,)))
    assert isinstance(reason, str) and "record 'decl-up-a' changed its definition_id" in reason
    reason = classify_declaration_upgrade(_decl(UpA), normalize_entry_families((OTHER,)))
    assert isinstance(reason, str) and "family 'decl-up' was removed" in reason
    assert classify_declaration_upgrade("not json", normalize_entry_families((OTHER,))) == (
        "stored entry declaration is not parseable"
    )


def test_offset_helpers() -> None:
    assert entry_id_number("up-1-17") == 17
    assert entry_id_number("up.x-series-3~4") == 3
    assert entry_id_number("up-1-9~alt~2") == 9
    assert entry_id_number("mp-149") is None
    assert entry_id_number(None) is None
    assert next_entry_id_offset([], 0) == 0
    assert next_entry_id_offset([5, 3], 0) == 6
    assert next_entry_id_offset([5], 40) == 41
    assert next_entry_id_offset([], 7) == 8
    layout = normalize_entry_families((family(UpA, UpB),))
    assert entry_id_offsets_json({FAMILY: 6, "zero": 0}) == '{"decl-up":6}'
    assert parse_entry_id_offsets('{"decl-up":6}', layout) == {FAMILY: 6}
    for bad in ('{"decl-up": 6}', '{"decl-up":0}', '{"decl-up":true}', '{"missing":1}', "[]", "x"):
        with pytest.raises(ValueError):
            parse_entry_id_offsets(bad, layout)


# --------------------------------------------------------------------------- end-to-end upgrades


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_append_record_kind_upgrade(database: Backend, scenario: str) -> None:
    build, before, after = SCENARIOS[scenario]
    built = build(database)
    stored_before = metadata(database)
    tables_before = table_names(database)

    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, *after)
    assert error.value.hint == ADDITIVE_DECLARATION_UPGRADE_HINT
    assert error.value.remedy == "upgrade"
    assert "upgrade=True" in str(error.value)
    assert declaration_aspects(error.value)["entry_declaration"]["additive"] is True
    # A refused reopen changes nothing.
    assert metadata(database) == stored_before
    assert table_names(database) == tables_before

    claims_before = claim_count(database)
    store = opened(database, *after, upgrade=True)
    stored = metadata(database)
    assert stored["entry_declaration"] == declaration_json(store.layout)
    assert json.loads(stored[ENTRY_ID_OFFSETS_KEY]) == {FAMILY: built.max_number + 1}
    assert set(stored) == set(stored_before) | {ENTRY_ID_OFFSETS_KEY}
    assert_old_entries_intact(store, built)
    assert dispatch_rows(database, *after) and len(dispatch_rows(database, *after)) == main_row_count(database, *after)
    assert claim_count(database) == claims_before

    # New saves in every backing mint above every pre-upgrade number.
    offset = built.max_number + 1
    backing_count = len(after)
    for index, cls in enumerate(after):
        sid = store.save(cls(1000 + index))
        entry_id = fetched_id(store, cls, sid)
        assert entry_id == f"up-1-{offset + sid * backing_count + index}"
        number = entry_id_number(entry_id)
        assert number is not None and number > built.max_number
    # Explicit collisions with pre-upgrade ids stay loud (ownership untouched).
    assert claim_count(database) == claims_before + backing_count
    assert_fsck_clean(store)

    # Plain reopens now trust the restamped layout; the old layout is refused.
    reopened = opened(database, *after)
    assert_old_entries_intact(reopened, built)
    # An older client (the pre-upgrade declaration) is told to reopen with the
    # newer declaration, never to rebuild a healthy store.
    for upgrade in (False, True):
        with pytest.raises(StorageLayoutUpgradeRequiredError) as refused:
            opened(database, *before, upgrade=upgrade)
        assert refused.value.remedy == "reopen"
        assert refused.value.hint is not None and "newer declaration" in refused.value.hint
        assert declaration_aspects(refused.value)["entry_declaration"]["stored_is_newer"] is True


def test_two_to_three_dispatch_table_has_the_new_shape(database: Backend) -> None:
    build_two_backings(database)
    opened(database, UpA, UpB, UpC, upgrade=True)
    dispatch = entry_dispatch_table_name(FAMILY)
    columns = [backing_dispatch_column_name(RECORDS[record].name) for record in (UpA, UpB, UpC)]
    with database.engine.connect() as connection:
        names = set(connection.execute(sqlalchemy.text(f'SELECT * FROM "{dispatch}" LIMIT 0')).keys())
    assert names == {"content_id", *columns}
    # The exactly-one CHECK spans all three backings.
    with pytest.raises(IntegrityError), database.engine.begin() as connection:
        connection.execute(
            sqlalchemy.text(f'INSERT INTO "{dispatch}" (content_id, {columns[1]}, {columns[2]}) VALUES (\'x\', 91, 92)')
        )
    with pytest.raises(IntegrityError), database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(f"INSERT INTO \"{dispatch}\" (content_id) VALUES ('y')"))
    # No stray reserved-prefix object was left behind: a plain reopen accepts the store.
    opened(database, UpA, UpB, UpC)


def test_new_family_upgrade(database: Backend) -> None:
    built = build_one_backing(database)
    stored_before = metadata(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, extra=(OTHER,))
    assert error.value.hint == ADDITIVE_DECLARATION_UPGRADE_HINT
    assert error.value.remedy == "upgrade"
    store = opened(database, UpA, extra=(OTHER,), upgrade=True)
    # No family's record list changed, so no offsets key is written.
    assert set(metadata(database)) == set(stored_before)
    sid = store.save(OtherRec(1))
    # The new family mints exactly like a fresh store.
    assert fetched_id(store, OtherRec, sid) == f"up-1-{sid}"
    assert fetched_id(store, UpA, store.save(UpA(77))) == "up-1-6"
    for (cls, value), old_sid in built.sids.items():
        assert fetched_id(store, cls, old_sid) == built.ids[(cls, value)]
    assert_fsck_clean(store)
    opened(database, UpA, extra=(OTHER,))


def test_new_multi_backing_family_and_appended_record_together(database: Backend) -> None:
    built = build_one_backing(database)
    pair = EntryFamilyDeclaration(
        name="decl-up-pair",
        family=OtherFamily,
        records=(
            EntryRecordDeclaration(name="decl-up-pair-other", record=OtherRec),
            EntryRecordDeclaration(name="decl-up-pair-c", record=UpC),
        ),
        definition_id="urn:test:decl-up-pair",
    )
    store = opened(database, UpA, UpB, extra=(pair,), upgrade=True)
    assert json.loads(metadata(database)[ENTRY_ID_OFFSETS_KEY]) == {FAMILY: built.max_number + 1}
    sid = store.save(UpC(5))
    assert fetched_id(store, UpC, sid) == f"up-1-{sid * 2 + 1}"
    assert store.fetch_entry(OtherFamily, content_id(UpC(5)), eager=True) == UpC(5)
    assert_fsck_clean(store)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(lambda: (family(UpB, UpA),), id="reorder"),
        pytest.param(lambda: (family(UpA),), id="removal"),
        pytest.param(lambda: (family(UpA, UpC),), id="rename"),
        pytest.param(lambda: (family(UpA, UpB, definition_id="urn:test:changed"),), id="family-definition"),
        pytest.param(
            lambda: (
                EntryFamilyDeclaration(
                    name=FAMILY,
                    family=UpFamily,
                    records=(
                        EntryRecordDeclaration(name="decl-up-a", record=UpA, definition_id="urn:test:record"),
                        RECORDS[UpB],
                        RECORDS[UpC],
                    ),
                    definition_id=DEFINITION,
                ),
            ),
            id="record-definition",
        ),
    ],
)
def test_non_additive_declaration_changes_require_a_rebuild(
    database: Backend, target: Callable[[], tuple[EntryFamilyDeclaration, ...]], request: pytest.FixtureRequest
) -> None:
    build_two_backings(database)
    stored_before = metadata(database)
    tables_before = table_names(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        SqlStore(database, entry_families=target(), entry_ids=EntryIdScheme("up", "1"), upgrade=True)
    assert error.value.hint is not None
    aspect = declaration_aspects(error.value)["entry_declaration"]
    assert "reason" in aspect and aspect["additive"] is False
    if request.node.callspec.id.endswith("removal"):
        # (A) against a store declaring (A, B): the store is newer, not broken.
        assert error.value.remedy == "reopen" and "newer declaration" in error.value.hint
        assert "'decl-up'" in aspect["reason"]
    else:
        assert "rebuild the store" in error.value.hint and "'decl-up'" in error.value.hint
        assert error.value.remedy == "rebuild"
    assert metadata(database) == stored_before
    assert table_names(database) == tables_before


@pytest.mark.parametrize("upgrade", [False, True])
def test_appended_table_holding_foreign_rows_is_refused(database: Backend, upgrade: bool) -> None:
    store = opened(database, UpA)
    store.save(UpA(1))
    # An undeclared (ad-hoc) save of the class a later upgrade appends.
    store.save(UpB(1))
    stored_before = metadata(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB, upgrade=upgrade)
    # Classified at open time (read-only): upgrade=False never advertises an
    # additive upgrade that upgrade=True would refuse.
    assert error.value.hint is not None and "decl_up_b" in error.value.hint and "rebuild" in error.value.hint
    assert error.value.remedy == "rebuild"
    assert declaration_aspects(error.value)["entry_declaration"]["additive"] is False
    assert metadata(database) == stored_before
    assert entry_dispatch_table_name(FAMILY) not in table_names(database)


# --------------------------------------------------------------------------- crash convergence


class Crash(Exception):
    """A simulated process death between upgrade steps."""


STEPS = ("prepared", "claimed", "tables", "dispatch", "offsets", "restamp-schemas", "restamp-offsets", "restamped")


@pytest.fixture(params=["sqlite", "sqlite-degraded", "duckdb", POSTGRES_PARAM])
def crash_database(request: pytest.FixtureRequest) -> Iterator[Backend]:
    """The crash matrix databases, including the degraded (autocommit) SQLite profile."""
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Backend.duckdb() as backend:
            yield backend
        return
    if request.param == "postgresql":
        with postgres_database() as backend:
            yield backend
        return
    with Backend.sqlite(degraded=request.param == "sqlite-degraded") as backend:
        yield backend


def clear_lease(database: Backend) -> None:
    """Release a degraded writer lease the way an operator does after verifying its holder is gone."""
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("DELETE FROM _httk_store_metadata WHERE key = 'lease'"))


def snapshot(database: Backend, records: tuple[type, ...]) -> tuple[Any, ...]:
    return (metadata(database), dispatch_rows(database, *records), claim_count(database))


def next_ids(store: SqlStore, records: tuple[type, ...]) -> list[str]:
    return [fetched_id(store, cls, store.save(cls(5000 + index))) for index, cls in enumerate(records)]


def lineage_numbers(database: Backend, records: tuple[type, ...]) -> list[int]:
    """One entry-id number per stored lineage of the family (revisions share their lineage's id)."""
    present = table_names(database)
    lineages: dict[tuple[str, int], str] = {}
    with database.engine.connect() as connection:
        for cls in records:
            table = cls.__httk_storage__.storage_name
            if table not in present:
                continue
            for logical_id, entry_id in connection.execute(sqlalchemy.text(f'SELECT logical_id, id FROM "{table}"')):
                assert lineages.setdefault((table, int(logical_id)), entry_id) == entry_id
    numbers = [entry_id_number(entry_id) for entry_id in lineages.values()]
    assert None not in numbers
    return [number for number in numbers if number is not None]


def assert_unique_numbers(database: Backend, records: tuple[type, ...]) -> None:
    numbers = lineage_numbers(database, records)
    assert len(numbers) == len(set(numbers)), sorted(numbers)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("durable", [False, True], ids=["rolled-back", "partial-durable"])
@pytest.mark.parametrize("step", STEPS)
def test_crash_after_each_step_converges(
    crash_database: Backend, monkeypatch: pytest.MonkeyPatch, scenario: str, durable: bool, step: str
) -> None:
    database = crash_database
    degraded = database.write_profile == "degraded"
    if degraded and not durable:
        pytest.skip("the degraded profile autocommits every statement: every crash is durable")
    build, _before, after = SCENARIOS[scenario]
    with Backend.sqlite(degraded=degraded) as reference_database:
        build(reference_database)
        if degraded:
            clear_lease(reference_database)
        reference = opened(reference_database, *after, upgrade=True)
        expected = snapshot(reference_database, after)
        expected_ids = next_ids(reference, after)
        clean_offset = reference._entry_id_offsets[FAMILY]

    built = build(database)
    if degraded:
        clear_lease(database)
    metadata_before = metadata(database)
    tables_before = table_names(database)

    def crash(self: SqlStore, name: str, connection: sqlalchemy.Connection) -> None:
        if name == step:
            if durable:
                # Commit what the attempt did so far: the durable residue the
                # degraded autocommit profile leaves at any statement boundary.
                connection.commit()
            raise Crash(name)

    monkeypatch.setattr(SqlStore, "_after_upgrade_step", crash)
    with pytest.raises(Crash):
        opened(database, *after, upgrade=True)
    monkeypatch.undo()
    if degraded:
        clear_lease(database)
    if not durable:
        # Every step ran inside the one upgrade transaction (on SQLite too: the
        # claim's DML opens the DBAPI transaction before any DDL), so a crash
        # leaves no residue at all.
        assert metadata(database) == metadata_before
        assert table_names(database) == tables_before
    elif step != "restamped":
        # The declaration is restamped last: any durable residue keeps the old one.
        assert metadata(database)["entry_declaration"] == metadata_before["entry_declaration"]

    store = opened(database, *after, upgrade=True)
    assert_old_entries_intact(store, built)
    if durable and step == "restamp-offsets":
        # The only inexact case: the crash made the new offset durable under
        # the old declaration, so the retry recomputes it as max(existing
        # numbers, stored offset) + 1 — one higher than a clean run.  Offsets
        # only ever move up; everything else is identical and every number
        # stays unique.
        offset = store._entry_id_offsets[FAMILY]
        assert offset > clean_offset
        stored = metadata(database)
        assert json.loads(stored.pop(ENTRY_ID_OFFSETS_KEY)) == {FAMILY: offset}
        reference_metadata = dict(expected[0])
        reference_metadata.pop(ENTRY_ID_OFFSETS_KEY)
        assert stored == reference_metadata
        assert snapshot(database, after)[1:] == expected[1:]
        minted = next_ids(store, after)
        for index, (cls, entry_id) in enumerate(zip(after, minted, strict=True)):
            sid = store.sid_of(cls(5000 + index))
            assert sid is not None and entry_id == f"up-1-{offset + sid * len(after) + index}"
    else:
        assert snapshot(database, after) == expected
        assert next_ids(store, after) == expected_ids
    assert_unique_numbers(database, after)
    assert_fsck_clean(store)


def _commit_crash_at(monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    def crash(self: SqlStore, name: str, connection: sqlalchemy.Connection) -> None:
        if name == step:
            connection.commit()
            raise Crash(name)

    monkeypatch.setattr(SqlStore, "_after_upgrade_step", crash)


@pytest.mark.parametrize("step", ["restamp-schemas", "restamp-offsets"])
def test_partial_restamp_reopens_with_the_right_remedy(
    database: Backend, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    """New schemas (and offsets) with the old declaration: both plain reopens say how to finish the upgrade."""
    built = build_one_backing(database)
    _commit_crash_at(monkeypatch, step)
    with pytest.raises(Crash):
        opened(database, UpA, UpB, upgrade=True)
    monkeypatch.undo()
    with pytest.raises(StorageLayoutUpgradeRequiredError) as target_reopen:
        opened(database, UpA, UpB)
    assert target_reopen.value.remedy == "upgrade"
    assert target_reopen.value.hint == ADDITIVE_DECLARATION_UPGRADE_HINT
    # Only the upgraded declaration can finish it: the old one is told to
    # reopen with that (remedy "upgrade" would loop forever for a CLI).
    for upgrade in (False, True):
        with pytest.raises(StorageLayoutUpgradeRequiredError) as old_reopen:
            opened(database, UpA, upgrade=upgrade)
        assert old_reopen.value.remedy == "reopen"
        assert old_reopen.value.hint is not None and "interrupted" in old_reopen.value.hint
    store = opened(database, UpA, UpB, upgrade=True)
    assert store._entry_id_offsets[FAMILY] >= built.max_number + 1
    assert_old_entries_intact(store, built)
    sid = store.save(UpB(1))
    number = entry_id_number(fetched_id(store, UpB, sid))
    assert number is not None and number > built.max_number
    assert_fsck_clean(store)


@pytest.mark.parametrize("step", ["tables", "dispatch", "offsets"])
def test_interrupted_one_to_many_upgrade_hint(database: Backend, monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    """A durable dispatch table for a declared single-backing family names the interrupted upgrade."""
    build_one_backing(database)
    _commit_crash_at(monkeypatch, step)
    with pytest.raises(Crash):
        opened(database, UpA, UpB, upgrade=True)
    monkeypatch.undo()
    for upgrade in (False, True):
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            opened(database, UpA, upgrade=upgrade)
        assert error.value.remedy == "reopen"
        assert error.value.hint is not None and "interrupted" in error.value.hint
    # The upgraded declaration is advertised as "upgrade", and upgrade=True converges.
    with pytest.raises(StorageLayoutUpgradeRequiredError) as target_reopen:
        opened(database, UpA, UpB)
    assert target_reopen.value.remedy == "upgrade"
    assert_fsck_clean(opened(database, UpA, UpB, upgrade=True))


# --------------------------------------------------------------------------- compare-and-set and locking


def _run_in_thread(results: dict[str, Any], label: str, action: Callable[[], Any]) -> threading.Thread:
    def run() -> None:
        try:
            results[label] = action()
        except BaseException as error:
            results[label] = error

    thread = threading.Thread(target=run, name=label)
    thread.start()
    return thread


@pytest.mark.parametrize("same_target", [True, False], ids=["same-target", "different-target"])
def test_concurrent_upgraders_serialize(
    file_database: Backend, monkeypatch: pytest.MonkeyPatch, same_target: bool
) -> None:
    """The second upgrader waits for the first; it then no-ops, refuses, or applies its own further upgrade."""
    built = build_one_backing(file_database)
    claimed, gate = threading.Event(), threading.Event()

    def hold(self: SqlStore, name: str, connection: sqlalchemy.Connection) -> None:
        if name == "claimed" and threading.current_thread().name == "first":
            claimed.set()
            gate.wait(5)

    monkeypatch.setattr(SqlStore, "_after_upgrade_step", hold)
    results: dict[str, Any] = {}
    first = _run_in_thread(results, "first", lambda: opened(file_database, UpA, UpB, upgrade=True))
    assert claimed.wait(10)
    second_records = (UpA, UpB) if same_target else (UpA, UpB, UpC)
    second = _run_in_thread(results, "second", lambda: opened(file_database, *second_records, upgrade=True))
    time.sleep(0.5)
    assert second.is_alive()  # waiting on the first upgrader's lock
    gate.set()
    first.join(20)
    second.join(60)
    monkeypatch.undo()
    assert isinstance(results["first"], SqlStore)
    offset = built.max_number + 1
    if same_target:
        # Applied exactly once: the loser found the identical target and opened it.
        assert isinstance(results["second"], SqlStore)
        assert results["second"]._entry_id_offsets == {FAMILY: offset}
        assert json.loads(metadata(file_database)[ENTRY_ID_OFFSETS_KEY]) == {FAMILY: offset}
        store = results["second"]
    elif isinstance(results["second"], StorageLayoutUpgradeRequiredError):
        # SQLite: the claim lost to a different declaration.
        assert results["second"].remedy == "reopen"
        store = opened(file_database, UpA, UpB)
    else:
        # DuckDB: the second open restarted under the lock and applied (A, B) -> (A, B, C).
        assert isinstance(results["second"], SqlStore)
        store = results["second"]
        assert json.loads(metadata(file_database)[ENTRY_ID_OFFSETS_KEY])[FAMILY] > offset
    records = tuple(store.layout.families[0].records)
    assert len(dispatch_rows(file_database, *records)) == main_row_count(file_database, *records)
    assert_old_entries_intact(store, built)
    for cls in records:
        store.save(cls(900))
    assert_unique_numbers(file_database, records)
    assert_fsck_clean(store)


@pytest.mark.parametrize("same_target", [True, False])
def test_final_compare_and_set_loss(database: Backend, monkeypatch: pytest.MonkeyPatch, same_target: bool) -> None:
    """A restamp landing between the claim and the restamp: adopt an identical winner, refuse a different one."""
    build_one_backing(database)
    target = normalize_entry_families((family(UpA, UpB),))
    winner = target if same_target else normalize_entry_families((family(UpA, UpB, UpC),))

    def restamp(self: SqlStore, name: str, connection: sqlalchemy.Connection) -> None:
        if name != "offsets":
            return
        table = sqlalchemy.table("_httk_store_metadata", sqlalchemy.column("key"), sqlalchemy.column("value"))
        for key, value in (
            ("entry_declaration", declaration_json(winner)),
            ("entry_schemas", schema_fingerprint_json(winner)),
        ):
            connection.execute(sqlalchemy.update(table).where(table.c.key == key).values(value=value))
        connection.execute(sqlalchemy.insert(table).values(key=ENTRY_ID_OFFSETS_KEY, value='{"decl-up":42}'))

    monkeypatch.setattr(SqlStore, "_after_upgrade_step", restamp)
    if same_target:
        store = opened(database, UpA, UpB, upgrade=True)
        monkeypatch.undo()
        # The winner's offsets are adopted, never overwritten by a second stamp.
        assert store._entry_id_offsets == {FAMILY: 42}
        assert metadata(database)[ENTRY_ID_OFFSETS_KEY] == '{"decl-up":42}'
        sid = store.save(UpB(9))
        assert fetched_id(store, UpB, sid) == f"up-1-{42 + sid * 2 + 1}"
    else:
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            opened(database, UpA, UpB, upgrade=True)
        monkeypatch.undo()
        assert error.value.remedy == "reopen"
        assert error.value.hint is not None and "different declaration" in error.value.hint


def test_upgrade_blocked_by_own_write_transaction_is_retryable(
    file_database: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upgrade that cannot get the write lock says so (remedy "retry"), and changes nothing."""
    import httk.store.backend.sql.store as store_module

    monkeypatch.setattr(store_module, "_UPGRADE_LOCK_TIMEOUT", 0.5)
    database = file_database
    if database.engine.dialect.name == "sqlite":
        # A short busy timeout keeps the lock wait brief.
        database = Backend(sqlalchemy.create_engine(file_database.engine.url, connect_args={"timeout": 0.2}))
    build_one_backing(database)
    writer = opened(database, UpA)
    before = metadata(database)
    with writer.transaction():
        writer.save(UpA(50))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            opened(database, UpA, UpB, upgrade=True)
    assert error.value.remedy == "retry"
    assert error.value.hint is not None and "retry" in error.value.hint and "concurrently" not in error.value.hint
    if database.engine.dialect.name == "duckdb":
        # Retrying from inside the same scope can never succeed: the message says so.
        assert "inside an open write scope" in error.value.hint
    assert metadata(database) == before
    assert_fsck_clean(opened(database, UpA, UpB, upgrade=True))
    if database is not file_database:
        database.dispose()


# --------------------------------------------------------------------------- stale writers and readers


def test_stale_writer_is_refused_after_another_instance_upgrades(database: Backend) -> None:
    built = build_one_backing(database)
    stale = opened(database, UpA)
    opened(database, UpA, UpB, upgrade=True)
    before = metadata(database)
    rows_before = main_row_count(database, UpA)
    for attempt in (
        lambda: stale.save(UpA(99)),
        lambda: stale.replace(UpA(2), UpA(20)),
        lambda: stale.ensure_tables(UpA),
    ):
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            attempt()
        assert error.value.hint == "the store layout changed since this store was opened; reopen it"
        assert error.value.remedy == "reopen"
    # A read-only transaction() takes no write lock and checks nothing; its
    # first write is refused.
    with stale.transaction():
        ids_by_value(stale, UpA)
    with pytest.raises(StorageLayoutUpgradeRequiredError), stale.transaction():
        stale.save(UpA(97))
    with pytest.raises(StorageLayoutUpgradeRequiredError), stale.bulk_ingest() as bulk:
        bulk.save(UpA(98))
    assert metadata(database) == before
    assert main_row_count(database, UpA) == rows_before
    fresh = opened(database, UpA, UpB)
    assert_old_entries_intact(fresh, built)
    assert fresh.save(UpA(99)) > 0


def assert_consistent_after_race(database: Backend, built: Built, stale: SqlStore) -> None:
    """The upgrade covered every row the stale handle wrote, which is now refused; new saves work."""
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        stale.save(UpA(4242))
    assert error.value.remedy == "reopen"
    store = opened(database, UpA, UpB)
    assert_old_entries_intact(store, built)
    for value in (500, 501, 502):
        store.save(UpB(value))  # never collides with a stale row
        store.save(UpA(value))
    assert_unique_numbers(database, (UpA, UpB))
    assert len(dispatch_rows(database, UpA, UpB)) == main_row_count(database, UpA, UpB)
    assert_fsck_clean(store)


def test_transaction_that_has_written_is_ordered_before_an_upgrade(file_database: Backend) -> None:
    """Once a transaction() has written, an upgrade waits for it to commit, then covers its rows."""
    built = build_one_backing(file_database)
    stale = opened(file_database, UpA)
    wrote, release = threading.Event(), threading.Event()

    def stale_transaction() -> str:
        with stale.transaction():
            stale.save(UpA(100))
            wrote.set()
            release.wait(3)
            stale.save(UpA(101))
        return "committed"

    results: dict[str, Any] = {}
    writer = _run_in_thread(results, "stale", stale_transaction)
    assert wrote.wait(10)
    upgrader = _run_in_thread(results, "upgrade", lambda: opened(file_database, UpA, UpB, upgrade=True))
    time.sleep(0.5)
    assert upgrader.is_alive()  # the upgrade waits for the in-flight write transaction
    release.set()
    writer.join(20)
    upgrader.join(60)
    assert results["stale"] == "committed"
    assert isinstance(results["upgrade"], SqlStore)
    stale_numbers = [
        entry_id_number(ids[0]) for value, ids in ids_by_value(results["upgrade"], UpA).items() if value >= 100
    ]
    assert len(stale_numbers) == 2
    assert all(number is not None and number < results["upgrade"]._entry_id_offsets[FAMILY] for number in stale_numbers)
    assert_consistent_after_race(file_database, built, stale)


def test_read_then_write_transaction_with_an_upgrade_in_between(file_database: Backend) -> None:
    """A read-only start does not block an upgrade (SQLite, PostgreSQL); the first write is then refused.

    DuckDB keeps its in-process layout lock for the whole scope, so there the
    upgrade waits for the transaction instead and covers its rows.
    """
    built = build_one_backing(file_database)
    stale = opened(file_database, UpA)
    has_read, release = threading.Event(), threading.Event()

    def read_then_write() -> str:
        with stale.transaction():
            ids_by_value(stale, UpA)
            has_read.set()
            release.wait(10)
            stale.save(UpA(100))
        return "committed"

    results: dict[str, Any] = {}
    writer = _run_in_thread(results, "stale", read_then_write)
    assert has_read.wait(10)
    upgrader = _run_in_thread(results, "upgrade", lambda: opened(file_database, UpA, UpB, upgrade=True))
    if file_database.engine.dialect.name == "duckdb":
        time.sleep(0.5)
        assert upgrader.is_alive()
        release.set()
        writer.join(20)
        upgrader.join(60)
        assert results["stale"] == "committed"
    else:
        upgrader.join(30)  # not blocked by a transaction that has only read
        release.set()
        writer.join(20)
        assert isinstance(results["stale"], StorageLayoutUpgradeRequiredError)
        assert results["stale"].remedy == "reopen"
        assert 100 not in ids_by_value(results["upgrade"], UpA)
    assert isinstance(results["upgrade"], SqlStore)
    assert_consistent_after_race(file_database, built, stale)


def _hold_read_only_transaction(path: str, seconds: float, ready: Any) -> None:
    with Backend.sqlite(path) as database:
        store = opened(database, UpA)
        with store.transaction():
            ids_by_value(store, UpA)
            ready.set()
            time.sleep(seconds)


def test_read_only_transaction_does_not_block_another_process_writer(tmp_path: Any) -> None:
    import multiprocessing

    path = str(tmp_path / "store.sqlite")
    with Backend.sqlite(path) as database:
        opened(database, UpA).save(UpA(1))
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    holder = context.Process(target=_hold_read_only_transaction, args=(path, 3.0, ready))
    holder.start()
    try:
        assert ready.wait(30)
        with Backend.sqlite(path) as database:
            started = time.monotonic()
            opened(database, UpA).save(UpA(2))
            assert time.monotonic() - started < 1.5  # not held up by the reader's scope
    finally:
        holder.join(30)
    assert holder.exitcode == 0


def test_stale_save_waits_for_an_upgrade_in_flight_and_is_refused(
    file_database: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save arriving during an upgrade blocks on the write lock, then sees the new layout and is refused."""
    built = build_one_backing(file_database)
    stale = opened(file_database, UpA)
    claimed, gate = threading.Event(), threading.Event()

    def hold(self: SqlStore, name: str, connection: sqlalchemy.Connection) -> None:
        if name == "claimed":
            claimed.set()
            gate.wait(5)

    monkeypatch.setattr(SqlStore, "_after_upgrade_step", hold)
    results: dict[str, Any] = {}
    upgrader = _run_in_thread(results, "upgrade", lambda: opened(file_database, UpA, UpB, upgrade=True))
    assert claimed.wait(10)
    writer = _run_in_thread(results, "stale", lambda: stale.save(UpA(100)))
    time.sleep(0.5)
    assert writer.is_alive()  # blocked on the upgrade's lock
    gate.set()
    upgrader.join(20)
    writer.join(20)
    monkeypatch.undo()
    assert isinstance(results["upgrade"], SqlStore)
    assert isinstance(results["stale"], StorageLayoutUpgradeRequiredError)
    assert results["stale"].remedy == "reopen"
    assert 100 not in ids_by_value(results["upgrade"], UpA)
    assert_consistent_after_race(file_database, built, stale)


def test_upgrade_injected_between_guard_and_first_dml_waits(
    file_database: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window the review found: an upgrade started after a save's check cannot commit before its DML."""
    built = build_one_backing(file_database)
    stale = opened(file_database, UpA)
    results: dict[str, Any] = {}
    threads: list[threading.Thread] = []

    def inject(self: SqlStore, connection: sqlalchemy.Connection) -> None:
        if self is stale and not threads:
            threads.append(_run_in_thread(results, "upgrade", lambda: opened(file_database, UpA, UpB, upgrade=True)))
            time.sleep(0.5)
            assert threads[0].is_alive()  # the upgrade waits for this write transaction

    monkeypatch.setattr(SqlStore, "_after_write_guard", inject)
    sid = stale.save(UpA(100))
    threads[0].join(60)
    monkeypatch.undo()
    assert isinstance(results["upgrade"], SqlStore)
    number = entry_id_number(fetched_id(results["upgrade"], UpA, sid))
    assert number is not None and number < results["upgrade"]._entry_id_offsets[FAMILY]
    assert_consistent_after_race(file_database, built, stale)


def test_stale_bulk_parity_context_is_ordered_before_an_upgrade(file_database: Backend) -> None:
    built = build_one_backing(file_database)
    stale = opened(file_database, UpA)
    entered, release = threading.Event(), threading.Event()

    def stale_bulk() -> str:
        with stale.bulk_ingest(finalize="parity") as bulk:
            entered.set()
            release.wait(5)
            bulk.save(UpA(100))
            bulk.save(UpA(101))
        return "committed"

    results: dict[str, Any] = {}
    writer = _run_in_thread(results, "stale", stale_bulk)
    assert entered.wait(10)
    upgrader = _run_in_thread(results, "upgrade", lambda: opened(file_database, UpA, UpB, upgrade=True))
    time.sleep(0.5)
    assert upgrader.is_alive()
    release.set()
    writer.join(20)
    upgrader.join(60)
    assert results["stale"] == "committed"
    assert isinstance(results["upgrade"], SqlStore)
    assert_consistent_after_race(file_database, built, stale)


def test_stale_reader_gets_reopen_instead_of_a_corruption_error(database: Backend) -> None:
    store = opened(database, UpA, UpB)
    store.save(UpA(1))
    store.save(UpB(1))
    stale = opened(database, UpA, UpB)
    upgraded = opened(database, UpA, UpB, UpC, upgrade=True)
    upgraded.save(UpC(7))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        stale.fetch_entry(UpFamily, content_id(UpC(7)))
    assert error.value.remedy == "reopen"
    # Entries the stale handle can still interpret keep resolving.
    assert stale.fetch_entry(UpFamily, content_id(UpB(1)), eager=True) == UpB(1)


def test_stale_single_backing_reader_is_told_to_reopen(database: Backend) -> None:
    built = build_one_backing(database)
    stale = opened(database, UpA)
    upgraded = opened(database, UpA, UpB, upgrade=True)
    upgraded.save(UpB(7))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        stale.fetch_entry(UpFamily, content_id(UpB(7)))
    assert error.value.remedy == "reopen"
    # Entries it can see still resolve; a genuine miss on a current handle is None.
    assert stale.fetch_entry(UpFamily, content_id(UpA(2)), eager=True) == UpA(2)
    assert upgraded.fetch_entry(UpFamily, content_id(UpA(12345))) is None
    del built


# --------------------------------------------------------------------------- remedies, offsets and dispatch


def _alien_table(database: Backend) -> None:
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("CREATE TABLE _httk_alien (x INTEGER)"))


def _dirty_marker(database: Backend) -> None:
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("INSERT INTO _httk_store_metadata (key, value) VALUES ('dirty:ghost', 'x')"))


def _foreign_rows(database: Backend) -> None:
    opened(database, UpA).save(UpB(1))


def _no_damage(database: Backend) -> None:
    del database


@pytest.mark.parametrize(
    ("damage", "extra_options"),
    [
        pytest.param(_no_damage, {}, id="clean"),
        pytest.param(_alien_table, {}, id="alien-reserved-table"),
        pytest.param(_dirty_marker, {}, id="invalid-dirty-marker"),
        pytest.param(_foreign_rows, {}, id="foreign-rows"),
        pytest.param(_no_damage, {"store_timestamps": False}, id="timestamps-mismatch"),
    ],
)
def test_upgrade_remedy_is_only_advertised_when_upgrade_succeeds(
    database: Backend, damage: Callable[[Backend], None], extra_options: dict[str, Any]
) -> None:
    """Invariant: if upgrade=False says remedy "upgrade", upgrade=True with the same declaration succeeds."""
    build_one_backing(database)
    damage(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB, **extra_options)
    before = metadata(database)
    if error.value.remedy == "upgrade":
        opened(database, UpA, UpB, upgrade=True, **extra_options)
    else:
        with pytest.raises(StorageLayoutUpgradeRequiredError) as upgraded:
            opened(database, UpA, UpB, upgrade=True, **extra_options)
        assert upgraded.value.remedy == error.value.remedy
        assert metadata(database) == before
    expected = {
        _no_damage: "reopen" if extra_options else "upgrade",
        _alien_table: "rebuild",
        _dirty_marker: "rebuild",
        _foreign_rows: "rebuild",
    }[damage]
    assert error.value.remedy == expected


def test_upgrade_remedy_after_interrupted_residue_is_honoured(
    database: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With durable residue of an interrupted upgrade, the advertised remedy works for the target declaration."""
    build_one_backing(database)
    _commit_crash_at(monkeypatch, "dispatch")
    with pytest.raises(Crash):
        opened(database, UpA, UpB, upgrade=True)
    monkeypatch.undo()
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB)
    assert error.value.remedy == "upgrade"
    assert_fsck_clean(opened(database, UpA, UpB, upgrade=True))


def test_store_timestamps_mismatch_with_additive_declaration_says_reopen(database: Backend) -> None:
    build_one_backing(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB, store_timestamps=False, upgrade=True)
    assert error.value.remedy == "reopen"
    aspects = declaration_aspects(error.value)
    assert "store_timestamps" in aspects and aspects["entry_declaration"]["additive"] is True


def test_orphan_ownership_claims_raise_the_offset(database: Backend) -> None:
    built = build_one_backing(database)
    entry_owners, immutable_owners = identity_owner_tables(sqlalchemy.MetaData())
    with database.engine.begin() as connection:
        # Residue of a crashed degraded write: claims without rows.
        connection.execute(
            sqlalchemy.insert(entry_owners).values(
                family=FAMILY, entry_id="up-1-40", backing="decl-up-a", logical_id=99
            )
        )
        connection.execute(
            sqlalchemy.insert(immutable_owners).values(
                family=FAMILY, immutable_id="up-1-70~1", backing="decl-up-a", sid=98
            )
        )
    store = opened(database, UpA, UpB, upgrade=True)
    assert built.max_number < 70
    assert store._entry_id_offsets == {FAMILY: 71}


@dataclass(frozen=True)
class HolderA:
    """An ordinary record referencing an entry record, which is stored as a dependency."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="decl_up_holder", identity_name="decl_up_holder")
    tag: int
    a: UpA


@pytest.mark.parametrize("kind", ["single", "fresh-multi", "fresh-multi-dependency-only", "upgraded"])
def test_dependency_only_rows_resolve_alike(database: Backend, kind: str) -> None:
    """``fetch_entry`` resolves a dependency-only row in single-, multi-backing and upgraded families alike."""
    records = (UpA, UpB) if kind.startswith("fresh-multi") else (UpA,)
    store = opened(database, *records)
    holder_sid = store.save(HolderA(1, UpA(50)))
    values = (50,)
    if kind != "fresh-multi-dependency-only":
        store.save(UpA(51))
        values = (50, 51)
    if kind == "upgraded":
        store = opened(database, UpA, UpB, upgrade=True)
    # A family record first stored only by reference still gets its family's
    # sibling tables, so the holder and the record both read back.
    assert store.fetch(HolderA, holder_sid, eager=True) == HolderA(1, UpA(50))
    for value in values:
        assert store.fetch_entry(UpFamily, content_id(UpA(value)), eager=True) == UpA(value)
    assert store.fetch_entry(UpFamily, content_id(UpA(52))) is None
    assert_fsck_clean(store, known_types=(HolderA,))


def test_main_row_without_dispatch_row_is_still_an_integrity_error(database: Backend) -> None:
    from httk.store.store_common import EntryDispatchIntegrityError

    store = opened(database, UpA, UpB)
    store.save(UpA(1))
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DELETE FROM "{entry_dispatch_table_name(FAMILY)}"'))
    with pytest.raises(EntryDispatchIntegrityError):
        store.fetch_entry(UpFamily, content_id(UpA(1)))


# --------------------------------------------------------------------------- minting invariant


def test_fresh_multi_backing_store_mints_exactly_as_before(database: Backend) -> None:
    store = opened(database, UpA, UpB, UpC)
    minted = []
    for cls, value in ((UpA, 1), (UpB, 1), (UpC, 1), (UpA, 2), (UpC, 2)):
        minted.append(fetched_id(store, cls, store.save(cls(value))))
    replacement = store.replace(UpA(1), UpA(10))
    minted.append(fetched_id(store, UpA, replacement))
    minted.append(fetched_id(store, UpB, store.save(UpB(2))))
    minted.append(fetched_id(store, UpA, store.save(UpA(3))))
    # number = logical_id * 3 + backing_index, offset 0.
    assert minted == ["up-1-3", "up-1-4", "up-1-5", "up-1-6", "up-1-8", "up-1-3", "up-1-7", "up-1-12"]
    assert ENTRY_ID_OFFSETS_KEY not in metadata(database)
    assert store._entry_id_offsets == {}


# --------------------------------------------------------------------------- bulk ingest


def test_parity_bulk_ingest_into_upgraded_family_mints_above_offset(database: Backend) -> None:
    built = build_one_backing(database)
    store = opened(database, UpA, UpB, upgrade=True)
    offset = built.max_number + 1
    with store.bulk_ingest(finalize="parity") as bulk:
        bulk.save(UpA(300))
        bulk.save(UpB(300))
    found = {cls: ids_by_value(store, cls)[300] for cls in (UpA, UpB)}
    for index, cls in enumerate((UpA, UpB)):
        (entry_id,) = found[cls]
        number = entry_id_number(entry_id)
        assert number is not None and number > built.max_number
        sid = store.sid_of(cls(300))
        assert sid is not None and entry_id == f"up-1-{offset + sid * 2 + index}"
    assert_fsck_clean(store)


def _empty_store_with_offset(database: Backend, offset: int) -> SqlStore:
    """A physically empty (A, B) store carrying a persisted offset, as after an upgrade of since-emptied data."""
    opened(database, UpA, UpB)
    with database.engine.begin() as connection:
        connection.execute(
            sqlalchemy.text("INSERT INTO _httk_store_metadata (key, value) VALUES (:key, :value)"),
            {"key": ENTRY_ID_OFFSETS_KEY, "value": entry_id_offsets_json({FAMILY: offset})},
        )
    return opened(database, UpA, UpB)


@pytest.mark.parametrize(("finalize", "workers"), [("deferred", 1), ("parity", 1), ("auto", 2)])
def test_bulk_minting_expressions_apply_the_offset(database: Backend, finalize: str, workers: int) -> None:
    if database.engine.dialect.name == "postgresql" and finalize == "deferred":
        pytest.skip("the deferred profile is exercised on SQLite and DuckDB")
    store = _empty_store_with_offset(database, 100)
    with store.bulk_ingest(finalize=finalize, workers=workers) as bulk:  # type: ignore[arg-type]
        for value in (1, 2):
            bulk.save(UpA(value))
            bulk.save(UpB(value))
    for index, cls in enumerate((UpA, UpB)):
        for value in (1, 2):
            sid = store.sid_of(cls(value))
            assert sid is not None
            assert fetched_id(store, cls, sid) == f"up-1-{100 + sid * 2 + index}"


def test_malformed_offsets_key_is_refused(database: Backend) -> None:
    opened(database, UpA, UpB)
    with database.engine.begin() as connection:
        connection.execute(
            sqlalchemy.text("INSERT INTO _httk_store_metadata (key, value) VALUES (:key, '{\"decl-up\":-1}')"),
            {"key": ENTRY_ID_OFFSETS_KEY},
        )
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB)
    assert ENTRY_ID_OFFSETS_KEY in declaration_aspects(error.value)


def test_duckdb_layout_lock_prefers_a_waiting_upgrader_without_deadlocking_reentrant_writers() -> None:
    from httk.store.backend.sql.store import _LayoutLock

    lock = _LayoutLock()
    stop = threading.Event()

    def overlapping_writer() -> None:
        while not stop.is_set():
            lock.acquire_shared()
            time.sleep(0.02)
            lock.release_shared()

    writers = [threading.Thread(target=overlapping_writer) for _ in range(4)]
    for writer in writers:
        writer.start()
    try:
        time.sleep(0.1)
        assert lock.acquire_exclusive(5) == "acquired"  # not starved by overlapping scopes
        lock.release_exclusive()
    finally:
        stop.set()
        for writer in writers:
            writer.join(5)
    # A thread already holding the lock shared re-enters while an upgrader waits.
    lock.acquire_shared()
    outcome: list[str] = []
    waiter = threading.Thread(target=lambda: outcome.append(lock.acquire_exclusive(5)))
    waiter.start()
    time.sleep(0.1)
    lock.acquire_shared()
    lock.release_shared()
    assert lock.acquire_exclusive(0.1) == "own-scope"
    lock.release_shared()
    waiter.join(5)
    assert outcome == ["acquired"]
    lock.release_exclusive()


class HolderFamily:
    type = "decl_up_holders"


@dataclass(frozen=True)
class HolderC:
    """A declared (definition-free) family record that makes ``UpC``'s table reachable by reference."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="decl_up_holder_c", identity_name="decl_up_holder_c"
    )
    tag: int
    c: UpC


HOLDERS = EntryFamilyDeclaration(
    name="decl-up-holders",
    family=HolderFamily,
    records=(EntryRecordDeclaration(name="decl-up-holder-c", record=HolderC),),
)


@pytest.mark.parametrize("step", ["restamp-schemas", "restamp-offsets"])
def test_partial_restamp_is_visible_when_the_appended_table_was_already_reachable(
    database: Backend, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    """Old and new fingerprints differ only in ``entry_id_tables``; the old declaration must still be told to reopen.

    N→N+1 (2→3), so the dispatch table is legitimately present for the old
    declaration too: only the fingerprint can reveal the interrupted upgrade.
    """
    old_layout = normalize_entry_families((family(UpA, UpB), HOLDERS))
    new_layout = normalize_entry_families((family(UpA, UpB, UpC), HOLDERS))
    old_fingerprint = json.loads(schema_fingerprint_json(old_layout))
    new_fingerprint = json.loads(schema_fingerprint_json(new_layout))
    assert old_fingerprint["tables"] == new_fingerprint["tables"]
    assert old_fingerprint["entry_id_tables"] != new_fingerprint["entry_id_tables"]

    store = opened(database, UpA, UpB, extra=(HOLDERS,))
    for value in (1, 2):
        store.save(UpA(value))
        store.save(UpB(value))
    _commit_crash_at(monkeypatch, step)
    with pytest.raises(Crash):
        opened(database, UpA, UpB, UpC, extra=(HOLDERS,), upgrade=True)
    monkeypatch.undo()
    for upgrade in (False, True):
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            opened(database, UpA, UpB, extra=(HOLDERS,), upgrade=upgrade)
        assert error.value.remedy == "reopen"
        assert error.value.hint is not None and "interrupted" in error.value.hint
    with pytest.raises(StorageLayoutUpgradeRequiredError) as target_reopen:
        opened(database, UpA, UpB, UpC, extra=(HOLDERS,))
    assert target_reopen.value.remedy == "upgrade"
    upgraded = opened(database, UpA, UpB, UpC, extra=(HOLDERS,), upgrade=True)
    for value in (10, 11):
        for cls in (UpA, UpB, UpC):
            upgraded.save(cls(value))
    assert_unique_numbers(database, (UpA, UpB, UpC))
    assert_fsck_clean(upgraded)


def test_entry_id_tables_may_grow_but_never_shrink() -> None:
    old = schema_fingerprint_json(normalize_entry_families((family(UpA, UpB), HOLDERS)))
    new = schema_fingerprint_json(normalize_entry_families((family(UpA, UpB, UpC), HOLDERS)))
    assert set(schema_fingerprint_diff(old, new)) == {"<entry_id_tables>"}
    assert classify_schema_upgrade(old, new) == AdditiveUpgradePlan({})
    reason = classify_schema_upgrade(new, old)
    assert isinstance(reason, str) and "decl_up_c" in reason
