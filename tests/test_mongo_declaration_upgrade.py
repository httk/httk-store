"""MongoStore parity for additive declaration upgrades (appended record kinds, new families).

Mirrors ``test_db_declaration_upgrade``: classification and typed remedies,
dispatch validator widening and back-fill, entry-id offsets, the restamp order
and its crash convergence, the lease-ordered stale-writer guard, stale
readers, and dependency-only ``fetch_entry``.  Every test needs a live MongoDB
replica set (``HTTK_TEST_MONGODB_URI``) and is skipped without one.
"""

import datetime
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast

import pytest
from httk.core.storage import content_id
from pymongo.errors import BulkWriteError
from test_db_declaration_upgrade import (
    DEFINITION,
    FAMILY,
    HOLDERS,
    OTHER,
    RECORDS,
    HolderA,
    OtherFamily,
    OtherRec,
    UpA,
    UpB,
    UpC,
    UpFamily,
    family,
)

import httk.store.backend.mongo.store as mongo_store_module
from httk.store import EntryIdScheme
from httk.store.backend.mongo import MongoDatabase, MongoStore, clear_stale_lock
from httk.store.backend.mongo.leases import acquire_writer
from httk.store.backend.mongo.mapping import METADATA_COLLECTION, entry_dispatch_table_name
from httk.store.storage_layout import (
    ADDITIVE_DECLARATION_UPGRADE_HINT,
    ENTRY_ID_OFFSETS_KEY,
    EntryFamilyDeclaration,
    EntryFamilyLayout,
    EntryRecordDeclaration,
    StorageLayoutUpgradeRequiredError,
    declaration_json,
    entry_id_number,
    normalize_entry_families,
    schema_fingerprint_json,
)
from httk.store.store_common import EntryDispatchIntegrityError

_IDENTITY_OWNERS = "_httk_identity_owners"


def opened(
    database: MongoDatabase, *records: type, extra: tuple[EntryFamilyDeclaration, ...] = (), **kwargs: Any
) -> MongoStore:
    return MongoStore(
        database,
        entry_families=(family(*records), *extra),
        entry_ids=EntryIdScheme("up", "1"),
        **kwargs,
    )


@pytest.fixture
def reference_database(mongo_test_client: Any) -> Iterator[MongoDatabase]:
    """A second fresh database for clean reference runs."""
    name = f"httk_test_{uuid.uuid4().hex}"
    database = MongoDatabase(mongo_test_client, name)
    try:
        yield database
    finally:
        database.client.drop_database(name)


def declaration_aspects(error: StorageLayoutUpgradeRequiredError) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", error.diff["declaration"])


def layout_document(database: MongoDatabase) -> dict[str, Any]:
    """The layout document without the fsck/upgrade ``generation`` counter."""
    document = database.database[METADATA_COLLECTION].find_one({"_id": "layout"})
    assert document is not None
    return {key: value for key, value in document.items() if key != "generation"}


def dispatch_documents(database: MongoDatabase) -> list[tuple[Any, ...]]:
    collection = database.database[entry_dispatch_table_name(FAMILY)]
    return sorted((row["_id"], row["record"], row["sid"]) for row in collection.find())


def storage_name(cls: type) -> str:
    return cast(Any, cls).__httk_storage__.storage_name


def main_document_count(database: MongoDatabase, *records: type) -> int:
    return sum(database.database[storage_name(record)].count_documents({"_httk_role": "main"}) for record in records)


def claim_count(database: MongoDatabase) -> int:
    return database.database[_IDENTITY_OWNERS].count_documents({"kind": "id"})


def assert_fsck_clean(store: MongoStore, known_types: tuple[type, ...] = ()) -> None:
    summary = store.fsck(collect_garbage=False, known_types=known_types)
    assert summary.violations == ()
    for name, counters in summary.collections.items():
        assert counters.repaired == 0, name
        assert counters.conflicts == 0, name


def fetched_id(store: MongoStore, cls: type, sid: int) -> str:
    value: Any = store.fetch(cls, sid).id
    assert isinstance(value, str)
    return value


def lineage_numbers(database: MongoDatabase, *records: type) -> list[int]:
    """One entry-id number per stored lineage of the family (revisions share their lineage's id)."""
    lineages: dict[tuple[str, int], str] = {}
    for cls in records:
        collection = database.database[storage_name(cls)]
        for document in collection.find({}, {"logical_id": 1, "f.id": 1}):
            entry_id = document["f"]["id"]
            key = (storage_name(cls), int(document["logical_id"]))
            assert lineages.setdefault(key, entry_id) == entry_id
    numbers = [entry_id_number(entry_id) for entry_id in lineages.values()]
    assert None not in numbers
    return [number for number in numbers if number is not None]


def assert_unique_numbers(database: MongoDatabase, *records: type) -> None:
    numbers = lineage_numbers(database, *records)
    assert len(numbers) == len(set(numbers)), sorted(numbers)


class Built:
    """The pre-upgrade entries of a scenario."""

    def __init__(self, sids: dict[tuple[type, int], int], ids: dict[tuple[type, int], str], max_number: int) -> None:
        self.sids = sids
        self.ids = ids
        self.max_number = max_number


def build_one_backing(database: MongoDatabase) -> Built:
    store = opened(database, UpA)
    sids: dict[tuple[type, int], int] = {(UpA, value): store.save(UpA(value)) for value in (1, 2, 3)}
    sids[(UpA, 10)] = store.replace(UpA(1), UpA(10))
    sids[(UpA, 4)] = store.save(UpA(4))
    ids: dict[tuple[type, int], str] = {key: fetched_id(store, key[0], sid) for key, sid in sids.items()}
    numbers = [entry_id_number(entry_id) for entry_id in ids.values()]
    return Built(sids, ids, max(number for number in numbers if number is not None))


def build_two_backings(database: MongoDatabase) -> Built:
    store = opened(database, UpA, UpB)
    sids: dict[tuple[type, int], int] = {
        (UpA, 1): store.save(UpA(1)),
        (UpB, 1): store.save(UpB(1)),
        (UpA, 2): store.save(UpA(2)),
    }
    sids[(UpA, 10)] = store.replace(UpA(1), UpA(10))
    sids[(UpB, 2)] = store.save(UpB(2))
    ids: dict[tuple[type, int], str] = {key: fetched_id(store, key[0], sid) for key, sid in sids.items()}
    numbers = [entry_id_number(entry_id) for entry_id in ids.values()]
    return Built(sids, ids, max(number for number in numbers if number is not None))


SCENARIOS: dict[str, tuple[Callable[[MongoDatabase], Built], tuple[type, ...], tuple[type, ...]]] = {
    "1to2": (build_one_backing, (UpA,), (UpA, UpB)),
    "2to3": (build_two_backings, (UpA, UpB), (UpA, UpB, UpC)),
}


def assert_old_entries_intact(store: MongoStore, built: Built) -> None:
    for (cls, value), sid in built.sids.items():
        record: Any = store.fetch(cls, sid)
        assert record.value == value
        assert record.id == built.ids[(cls, value)]
        entry: Any = store.fetch_entry(UpFamily, content_id(cls(value)))
        assert entry is not None
        assert entry.value == value
        assert type(entry) is cls
    revised = store.fetch(UpA, built.sids[(UpA, 10)])
    history = store.history(revised)
    assert tuple(item.value for item in history) == (1, 10)


# --------------------------------------------------------------------------- end-to-end upgrades


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_append_record_kind_upgrade(mongo_test_database: MongoDatabase, scenario: str) -> None:
    database = mongo_test_database
    build, before, after = SCENARIOS[scenario]
    built = build(database)
    stored_before = layout_document(database)

    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, *after)
    assert error.value.hint == ADDITIVE_DECLARATION_UPGRADE_HINT
    assert error.value.remedy == "upgrade"
    assert declaration_aspects(error.value)["entry_declaration"]["additive"] is True
    assert layout_document(database) == stored_before

    claims_before = claim_count(database)
    store = opened(database, *after, upgrade=True)
    stored = layout_document(database)
    assert stored["entry_declaration"] == declaration_json(store.layout)
    assert json.loads(stored[ENTRY_ID_OFFSETS_KEY]) == {FAMILY: built.max_number + 1}
    assert set(stored) == set(stored_before) | {ENTRY_ID_OFFSETS_KEY}
    assert_old_entries_intact(store, built)
    assert len(dispatch_documents(database)) == main_document_count(database, *after)
    assert claim_count(database) == claims_before

    offset = built.max_number + 1
    for index, cls in enumerate(after):
        sid = store.save(cls(1000 + index))
        assert fetched_id(store, cls, sid) == f"up-1-{offset + sid * len(after) + index}"
    assert claim_count(database) == claims_before + len(after)
    assert_unique_numbers(database, *after)
    assert_fsck_clean(store)

    assert_old_entries_intact(opened(database, *after), built)
    for upgrade in (False, True):
        with pytest.raises(StorageLayoutUpgradeRequiredError) as refused:
            opened(database, *before, upgrade=upgrade)
        assert refused.value.remedy == "reopen"
        assert declaration_aspects(refused.value)["entry_declaration"]["stored_is_newer"] is True


def test_new_family_upgrade(mongo_test_database: MongoDatabase) -> None:
    database = mongo_test_database
    built = build_one_backing(database)
    stored_before = layout_document(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, extra=(OTHER,))
    assert error.value.remedy == "upgrade"
    store = opened(database, UpA, extra=(OTHER,), upgrade=True)
    assert set(layout_document(database)) == set(stored_before)
    sid = store.save(OtherRec(1))
    assert fetched_id(store, OtherRec, sid) == f"up-1-{sid}"
    for (cls, value), old_sid in built.sids.items():
        assert fetched_id(store, cls, old_sid) == built.ids[(cls, value)]
    assert_fsck_clean(store)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        pytest.param(lambda: (family(UpB, UpA),), "rebuild", id="reorder"),
        pytest.param(lambda: (family(UpA),), "reopen", id="removal-older-client"),
        pytest.param(lambda: (family(UpA, UpC),), "rebuild", id="rename"),
        pytest.param(lambda: (family(UpA, UpB, definition_id="urn:test:changed"),), "rebuild", id="family-definition"),
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
            "rebuild",
            id="record-definition",
        ),
    ],
)
def test_non_additive_declaration_changes(
    mongo_test_database: MongoDatabase, target: Callable[[], tuple[EntryFamilyDeclaration, ...]], expected: str
) -> None:
    database = mongo_test_database
    build_two_backings(database)
    stored_before = layout_document(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        MongoStore(database, entry_families=target(), entry_ids=EntryIdScheme("up", "1"), upgrade=True)
    assert error.value.remedy == expected
    aspect = declaration_aspects(error.value)["entry_declaration"]
    assert aspect["additive"] is False and "'decl-up'" in aspect["reason"]
    assert layout_document(database) == stored_before


@pytest.mark.parametrize("upgrade", [False, True])
def test_attached_collection_holding_foreign_documents_is_refused(
    mongo_test_database: MongoDatabase, upgrade: bool
) -> None:
    database = mongo_test_database
    store = opened(database, UpA)
    store.save(UpA(1))
    store.save(UpB(1))  # an ad-hoc save of the class a later upgrade appends
    stored_before = layout_document(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB, upgrade=upgrade)
    assert error.value.remedy == "rebuild"
    assert declaration_aspects(error.value)["entry_declaration"]["additive"] is False
    assert layout_document(database) == stored_before


# --------------------------------------------------------------------------- crash convergence


class Crash(Exception):
    """A simulated process death between upgrade steps."""


STEPS = ("prepared", "claimed", "dispatch", "offsets", "restamp-schemas", "restamp-offsets", "restamped")


def _crash_at(monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    def crash(self: MongoStore, name: str) -> None:
        if name == step:
            raise Crash(name)

    monkeypatch.setattr(MongoStore, "_after_upgrade_step", crash)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("step", STEPS)
def test_crash_after_each_step_converges(
    mongo_test_database: MongoDatabase,
    reference_database: MongoDatabase,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    step: str,
) -> None:
    """Every Mongo step is durable as it happens, so every seam is a real crash point.

    The simulated ``Crash`` unwinds through the upgrade's ``finally``, which
    releases the fsck lease; a real process death leaves that lease behind —
    covered by ``test_dead_upgraders_lease_is_cleared_with_clear_stale_lock``.
    """
    database = mongo_test_database
    build, before, after = SCENARIOS[scenario]
    build(reference_database)
    reference = opened(reference_database, *after, upgrade=True)
    clean_offset = reference._entry_id_offsets[FAMILY]
    expected_layout = layout_document(reference_database)
    expected_dispatch = dispatch_documents(reference_database)

    built = build(database)
    declaration_before = layout_document(database)["entry_declaration"]
    _crash_at(monkeypatch, step)
    with pytest.raises(Crash):
        opened(database, *after, upgrade=True)
    monkeypatch.undo()
    if step != "restamped":
        # The declaration is restamped last: any durable residue keeps the old one.
        assert layout_document(database)["entry_declaration"] == declaration_before
        with pytest.raises(StorageLayoutUpgradeRequiredError) as target_reopen:
            opened(database, *after)
        assert target_reopen.value.remedy == "upgrade"
        if step in {"dispatch", "offsets", "restamp-schemas", "restamp-offsets"} and scenario == "1to2":
            # Interrupted residue opened with the old declaration: only the
            # upgraded declaration can finish it.
            with pytest.raises(StorageLayoutUpgradeRequiredError) as old_reopen:
                opened(database, *before)
            assert old_reopen.value.remedy == "reopen"

    store = opened(database, *after, upgrade=True)
    assert_old_entries_intact(store, built)
    assert dispatch_documents(database) == expected_dispatch
    stored = layout_document(database)
    if step == "restamp-offsets":
        # The only inexact case: the offset was durable under the old
        # declaration, so the retry recomputes it one higher.
        assert json.loads(stored.pop(ENTRY_ID_OFFSETS_KEY))[FAMILY] > clean_offset
        expected = dict(expected_layout)
        expected.pop(ENTRY_ID_OFFSETS_KEY)
        assert stored == expected
    else:
        assert stored == expected_layout
    for index, cls in enumerate(after):
        store.save(cls(5000 + index))
    assert_unique_numbers(database, *after)
    assert_fsck_clean(store)


# --------------------------------------------------------------------------- compare-and-set and locking


@pytest.mark.parametrize("same_target", [True, False])
def test_restamp_finding_a_moved_declaration(
    mongo_test_database: MongoDatabase, monkeypatch: pytest.MonkeyPatch, same_target: bool
) -> None:
    """A declaration that moved under the upgrade: adopt an identical winner, refuse a different one."""
    database = mongo_test_database
    build_one_backing(database)
    winner = normalize_entry_families((family(UpA, UpB),) if same_target else (family(UpA, UpB, UpC),))

    def restamp(self: MongoStore, name: str) -> None:
        if name == "offsets":
            database.database[METADATA_COLLECTION].update_one(
                {"_id": "layout"},
                {
                    "$set": {
                        "entry_declaration": declaration_json(winner),
                        "entry_schemas": schema_fingerprint_json(winner),
                        ENTRY_ID_OFFSETS_KEY: '{"decl-up":42}',
                    }
                },
            )

    monkeypatch.setattr(MongoStore, "_after_upgrade_step", restamp)
    if same_target:
        store = opened(database, UpA, UpB, upgrade=True)
        monkeypatch.undo()
        assert store._entry_id_offsets == {FAMILY: 42}
        sid = store.save(UpB(9))
        assert fetched_id(store, UpB, sid) == f"up-1-{42 + sid * 2 + 1}"
    else:
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            opened(database, UpA, UpB, upgrade=True)
        monkeypatch.undo()
        assert error.value.remedy == "reopen"


def test_second_upgrader_is_refused_while_the_first_holds_the_fsck_lease(
    mongo_test_database: MongoDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = mongo_test_database
    built = build_one_backing(database)
    claimed, gate = threading.Event(), threading.Event()

    def hold(self: MongoStore, name: str) -> None:
        if name == "claimed" and threading.current_thread().name == "first":
            claimed.set()
            gate.wait(5)

    monkeypatch.setattr(MongoStore, "_after_upgrade_step", hold)
    results: dict[str, Any] = {}

    def first() -> None:
        results["first"] = opened(database, UpA, UpB, upgrade=True)

    thread = threading.Thread(target=first, name="first")
    thread.start()
    assert claimed.wait(10)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB, upgrade=True)
    assert error.value.remedy == "retry"
    gate.set()
    thread.join(30)
    monkeypatch.undo()
    assert isinstance(results["first"], MongoStore)
    assert results["first"]._entry_id_offsets == {FAMILY: built.max_number + 1}
    assert len(dispatch_documents(database)) == main_document_count(database, UpA, UpB)


# --------------------------------------------------------------------------- stale writers and readers


def test_stale_writer_is_refused_after_another_instance_upgrades(mongo_test_database: MongoDatabase) -> None:
    database = mongo_test_database
    built = build_one_backing(database)
    stale = opened(database, UpA)
    opened(database, UpA, UpB, upgrade=True)
    before = layout_document(database)
    mains_before = main_document_count(database, UpA)
    for attempt in (
        lambda: stale.save(UpA(99)),
        lambda: stale.replace(UpA(2), UpA(20)),
    ):
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            attempt()
        assert error.value.remedy == "reopen"
    # Mongo registers the writer lease at transaction() entry, so a stale
    # handle is refused there already.
    with pytest.raises(StorageLayoutUpgradeRequiredError), stale.transaction():
        pass
    assert layout_document(database) == before
    assert main_document_count(database, UpA) == mains_before
    assert_old_entries_intact(opened(database, UpA, UpB), built)


def test_stale_handle_never_narrows_a_grown_dispatch_validator(mongo_test_database: MongoDatabase) -> None:
    database = mongo_test_database
    build_two_backings(database)
    stale = opened(database, UpA, UpB)  # has not prepared any collection yet
    upgraded = opened(database, UpA, UpB, UpC, upgrade=True)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        stale.ensure_collections(UpA)
    assert error.value.remedy == "reopen"
    sid = upgraded.save(UpC(3))
    assert upgraded.fetch_entry(UpFamily, content_id(UpC(3))) == upgraded.fetch(UpC, sid)


def test_transaction_in_flight_is_ordered_before_an_upgrade(mongo_test_database: MongoDatabase) -> None:
    """The upgrade drains the writer lease of a transaction already in flight, then covers its rows."""
    database = mongo_test_database
    built = build_one_backing(database)
    stale = opened(database, UpA)
    entered, release = threading.Event(), threading.Event()
    results: dict[str, Any] = {}

    def stale_transaction() -> None:
        with stale.transaction():
            stale.save(UpA(100))
            entered.set()
            release.wait(3)
            stale.save(UpA(101))
        results["stale"] = "committed"

    def upgrade() -> None:
        results["upgrade"] = opened(database, UpA, UpB, upgrade=True)

    writer = threading.Thread(target=stale_transaction)
    writer.start()
    assert entered.wait(10)
    upgrader = threading.Thread(target=upgrade)
    upgrader.start()
    time.sleep(0.5)
    assert upgrader.is_alive()  # draining the writer's lease
    release.set()
    writer.join(20)
    upgrader.join(60)
    assert results["stale"] == "committed"
    store = results["upgrade"]
    assert isinstance(store, MongoStore)
    for value in (100, 101):
        sid = store.sid_of(UpA(value))
        assert sid is not None
        number = entry_id_number(fetched_id(store, UpA, sid))
        assert number is not None and number < store._entry_id_offsets[FAMILY]
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        stale.save(UpA(4242))
    assert error.value.remedy == "reopen"
    for value in (500, 501):
        store.save(UpB(value))
    assert_unique_numbers(database, UpA, UpB)
    assert_old_entries_intact(store, built)
    assert_fsck_clean(store)


def test_upgrade_from_inside_an_own_write_transaction_is_retryable(mongo_test_database: MongoDatabase) -> None:
    """The fsck lease cannot drain this thread's own writer lease: the upgrade gives up with "retry".

    It waits until the idle writer lease turns stale (the lease protocol's
    stale interval), so this test is deliberately slow.
    """
    database = mongo_test_database
    build_one_backing(database)
    writer = opened(database, UpA)
    with writer.transaction():
        writer.save(UpA(50))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            opened(database, UpA, UpB, upgrade=True)
    assert error.value.remedy == "retry"


def test_stale_readers_are_told_to_reopen(mongo_test_database: MongoDatabase) -> None:
    database = mongo_test_database
    build_one_backing(database)
    single = opened(database, UpA)
    two = opened(database, UpA, UpB, upgrade=True)
    two.save(UpB(7))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        single.fetch_entry(UpFamily, content_id(UpB(7)))
    assert error.value.remedy == "reopen"
    three = opened(database, UpA, UpB, UpC, upgrade=True)
    three.save(UpC(8))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        two.fetch_entry(UpFamily, content_id(UpC(8)))
    assert error.value.remedy == "reopen"
    assert three.fetch_entry(UpFamily, content_id(UpA(12345))) is None


# --------------------------------------------------------------------------- remedies, offsets and dispatch


def test_fresh_multi_backing_store_mints_exactly_as_before(mongo_test_database: MongoDatabase) -> None:
    store = opened(mongo_test_database, UpA, UpB, UpC)
    minted = [fetched_id(store, cls, store.save(cls(value))) for cls, value in ((UpA, 1), (UpB, 1), (UpC, 1))]
    assert all(entry_id_number(entry_id) is not None for entry_id in minted)
    for index, (cls, entry_id) in enumerate(zip((UpA, UpB, UpC), minted, strict=True)):
        sid = store.sid_of(cls(1))
        assert sid is not None and entry_id == f"up-1-{sid * 3 + index}"
    assert ENTRY_ID_OFFSETS_KEY not in layout_document(mongo_test_database)
    assert store._entry_id_offsets == {}


def test_orphan_ownership_claims_raise_the_offset(mongo_test_database: MongoDatabase) -> None:
    database = mongo_test_database
    built = build_one_backing(database)
    owners = database.database[_IDENTITY_OWNERS]
    owners.insert_one({"family": FAMILY, "kind": "id", "value": "up-1-40", "backing": "decl-up-a", "owner": 990})
    owners.insert_one(
        {"family": FAMILY, "kind": "immutable_id", "value": "up-1-70~1", "backing": "decl-up-a", "owner": 991}
    )
    store = opened(database, UpA, UpB, upgrade=True)
    assert built.max_number < 70
    assert store._entry_id_offsets == {FAMILY: 71}


@pytest.mark.parametrize("kind", ["single", "multi", "upgraded"])
def test_dependency_only_documents_resolve_alike(mongo_test_database: MongoDatabase, kind: str) -> None:
    database = mongo_test_database
    store = opened(database, *((UpA, UpB) if kind == "multi" else (UpA,)))
    holder_sid = store.save(HolderA(1, UpA(50)))
    store.save(UpA(51))
    if kind == "upgraded":
        store = opened(database, UpA, UpB, upgrade=True)
    assert store.fetch(HolderA, holder_sid) == HolderA(1, UpA(50))
    for value in (50, 51):
        assert store.fetch_entry(UpFamily, content_id(UpA(value))) == UpA(value)
    assert_fsck_clean(store, known_types=(HolderA,))


def _alien_collection(database: MongoDatabase) -> None:
    database.database.create_collection("_httk_alien")


def _foreign_documents(database: MongoDatabase) -> None:
    opened(database, UpA).save(UpB(1))


def _no_damage(database: MongoDatabase) -> None:
    del database


@pytest.mark.parametrize(
    ("damage", "extra_options", "expected"),
    [
        pytest.param(_no_damage, {}, "upgrade", id="clean"),
        pytest.param(_alien_collection, {}, "rebuild", id="alien-reserved-collection"),
        pytest.param(_foreign_documents, {}, "rebuild", id="foreign-documents"),
        pytest.param(_no_damage, {"store_timestamps": False}, "reopen", id="timestamps-mismatch"),
    ],
)
def test_upgrade_remedy_is_only_advertised_when_upgrade_succeeds(
    mongo_test_database: MongoDatabase,
    damage: Callable[[MongoDatabase], None],
    extra_options: dict[str, Any],
    expected: str,
) -> None:
    """Invariant: if upgrade=False says remedy "upgrade", upgrade=True with the same declaration succeeds."""
    database = mongo_test_database
    build_one_backing(database)
    damage(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB, **extra_options)
    assert error.value.remedy == expected
    before = layout_document(database)
    if error.value.remedy == "upgrade":
        opened(database, UpA, UpB, upgrade=True, **extra_options)
    else:
        with pytest.raises(StorageLayoutUpgradeRequiredError) as upgraded:
            opened(database, UpA, UpB, upgrade=True, **extra_options)
        assert upgraded.value.remedy == expected
        assert layout_document(database) == before


def test_malformed_offsets_key_is_refused(mongo_test_database: MongoDatabase) -> None:
    database = mongo_test_database
    opened(database, UpA, UpB)
    database.database[METADATA_COLLECTION].update_one(
        {"_id": "layout"}, {"$set": {ENTRY_ID_OFFSETS_KEY: '{"decl-up":-1}'}}
    )
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB)
    assert error.value.remedy == "rebuild"
    assert ENTRY_ID_OFFSETS_KEY in declaration_aspects(error.value)


def test_new_multi_backing_family_needs_no_offset(mongo_test_database: MongoDatabase) -> None:
    database = mongo_test_database
    build_one_backing(database)
    pair = EntryFamilyDeclaration(
        name="decl-up-pair",
        family=OtherFamily,
        records=(
            EntryRecordDeclaration(name="decl-up-pair-other", record=OtherRec),
            EntryRecordDeclaration(name="decl-up-pair-c", record=UpC),
        ),
        definition_id="urn:test:decl-up-pair",
    )
    store = opened(database, UpA, extra=(pair,), upgrade=True)
    assert ENTRY_ID_OFFSETS_KEY not in layout_document(database)
    sid = store.save(UpC(5))
    assert fetched_id(store, UpC, sid) == f"up-1-{sid * 2 + 1}"
    assert store.fetch_entry(OtherFamily, content_id(UpC(5))) == UpC(5)


# --------------------------------------------------------------------------- review corrections


class _FakeDispatch:
    """A server-free stand-in for a dispatch collection (``insert_many`` raises a canned error)."""

    def __init__(self, error: BaseException, existing: Mapping[str, Mapping[str, Any]]) -> None:
        self.error = error
        self.existing = existing

    def insert_many(self, batch: list[dict[str, Any]], ordered: bool) -> None:
        assert ordered is False
        raise self.error

    def find_one(self, query: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return self.existing.get(query["_id"])


def _grown_family() -> EntryFamilyLayout:
    return normalize_entry_families((family(UpA, UpB),)).families[0]


def test_dispatch_back_fill_reraises_write_concern_errors() -> None:
    """A BulkWriteError carrying only writeConcernErrors must not be mistaken for tolerated duplicates."""
    batch = [{"_id": "c1", "record": "decl-up-a", "sid": 1}]
    error = BulkWriteError(
        {"writeErrors": [], "writeConcernErrors": [{"code": 64, "errmsg": "waiting for replication"}]}
    )
    with pytest.raises(BulkWriteError):
        MongoStore._insert_dispatch_batch(_grown_family(), _FakeDispatch(error, {}), batch)


def test_dispatch_back_fill_accepts_only_identical_duplicates() -> None:
    batch = [{"_id": "c1", "record": "decl-up-a", "sid": 1}, {"_id": "c2", "record": "decl-up-a", "sid": 2}]
    duplicates = BulkWriteError(
        {"writeErrors": [{"index": 0, "code": 11000}, {"index": 1, "code": 11000}], "writeConcernErrors": []}
    )
    identical = {"c1": batch[0], "c2": batch[1]}
    MongoStore._insert_dispatch_batch(_grown_family(), _FakeDispatch(duplicates, identical), batch)
    conflicting = {"c1": batch[0], "c2": {"_id": "c2", "record": "decl-up-b", "sid": 2}}
    with pytest.raises(EntryDispatchIntegrityError):
        MongoStore._insert_dispatch_batch(_grown_family(), _FakeDispatch(duplicates, conflicting), batch)
    other = BulkWriteError({"writeErrors": [{"index": 0, "code": 121}], "writeConcernErrors": []})
    with pytest.raises(BulkWriteError):
        MongoStore._insert_dispatch_batch(_grown_family(), _FakeDispatch(other, identical), batch)


def test_dead_upgraders_lease_is_cleared_with_clear_stale_lock(
    mongo_test_database: MongoDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real death mid-upgrade leaves ``lease/fsck``: retry says how to clear it, and then converges."""
    database = mongo_test_database
    built = build_one_backing(database)
    _crash_at(monkeypatch, "dispatch")
    with pytest.raises(Crash):
        opened(database, UpA, UpB, upgrade=True)
    monkeypatch.undo()
    # The simulated crash released the lease in ``finally``; a dead process would not have.
    database.database[METADATA_COLLECTION].insert_one(
        {"_id": "lease/fsck", "owner": "dead", "heartbeat": datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)}
    )
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        opened(database, UpA, UpB, upgrade=True)
    assert error.value.remedy == "retry"
    assert error.value.hint is not None and "clear_stale_lock" in error.value.hint
    clear_stale_lock(database.database)
    store = opened(database, UpA, UpB, upgrade=True)
    assert_old_entries_intact(store, built)
    assert_fsck_clean(store)


def test_upgrade_gives_up_when_writers_do_not_drain(
    mongo_test_database: MongoDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A continuously refreshed writer lease makes the upgrade retry after its drain budget, changing nothing."""
    database = mongo_test_database
    build_one_backing(database)
    before = layout_document(database)
    writer = acquire_writer(database.database)
    stop = threading.Event()

    def keep_alive() -> None:
        while not stop.is_set():
            writer.refresh_heartbeat(force=True)
            time.sleep(0.2)

    refresher = threading.Thread(target=keep_alive)
    refresher.start()
    monkeypatch.setattr(mongo_store_module, "_UPGRADE_DRAIN_BUDGET", 1.0)
    try:
        started = time.monotonic()
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            opened(database, UpA, UpB, upgrade=True)
        assert time.monotonic() - started < 10
        assert error.value.remedy == "retry"
        assert error.value.hint is not None and "did not finish" in error.value.hint
        assert database.database[METADATA_COLLECTION].find_one({"_id": "lease/fsck"}) is None
        assert layout_document(database) == before
    finally:
        stop.set()
        refresher.join(5)
        writer.release()
    opened(database, UpA, UpB, upgrade=True)


@pytest.mark.parametrize("step", ["restamp-schemas", "restamp-offsets"])
def test_partial_restamp_is_visible_when_the_appended_collection_was_already_reachable(
    mongo_test_database: MongoDatabase, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    """2→3 with ``UpC`` reachable by reference: only ``entry_id_tables`` reveals the interrupted upgrade."""
    database = mongo_test_database
    store = opened(database, UpA, UpB, extra=(HOLDERS,))
    for value in (1, 2):
        store.save(UpA(value))
        store.save(UpB(value))
    _crash_at(monkeypatch, step)
    with pytest.raises(Crash):
        opened(database, UpA, UpB, UpC, extra=(HOLDERS,), upgrade=True)
    monkeypatch.undo()
    for upgrade in (False, True):
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            opened(database, UpA, UpB, extra=(HOLDERS,), upgrade=upgrade)
        assert error.value.remedy == "reopen"
        assert error.value.hint is not None and "interrupted" in error.value.hint
    upgraded = opened(database, UpA, UpB, UpC, extra=(HOLDERS,), upgrade=True)
    for cls in (UpA, UpB, UpC):
        upgraded.save(cls(10))
    assert_unique_numbers(database, UpA, UpB, UpC)
    assert_fsck_clean(upgraded)
