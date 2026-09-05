"""Durable family-wide identity ownership, upgrades, and transaction boundaries."""

import multiprocessing
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique

from httk.store import EntryIdConflictError, EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql.mapping import identity_owner_tables
from httk.store.storage_layout import (
    EntryFamilyDeclaration,
    EntryRecordDeclaration,
    StorageLayoutUpgradeRequiredError,
)


class Owners:
    type = "owners"


@dataclass(frozen=True)
class First:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="ownership_first", identity_name="ownership_first"
    )
    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class Second(First):
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="ownership_second", identity_name="ownership_second"
    )


DECLARATION = EntryFamilyDeclaration(
    name="ownership-test",
    family=Owners,
    records=(
        EntryRecordDeclaration(name="first", record=First),
        EntryRecordDeclaration(name="second", record=Second),
    ),
    definition_id="urn:test:ownership",
)


def opened(database, **kwargs):
    return SqlStore(
        database,
        entry_families=(DECLARATION,),
        entry_ids=EntryIdScheme("test", "1"),
        **kwargs,
    )


def claim_counts(database):
    with database.engine.connect() as connection:
        return tuple(
            connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(table)).scalar_one()
            for table in identity_owner_tables(sqlalchemy.MetaData())
        )


def legacy(database, *, drop=True):
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("DELETE FROM _httk_store_metadata WHERE key='identity_ownership'"))
        if drop:
            for table in identity_owner_tables(sqlalchemy.MetaData()):
                table.drop(connection)


def race_writer(path, backing, immutable, barrier, outcomes):
    try:
        with Backend.sqlite(path) as database:
            store = opened(database)
            # Bypass the advisory family SELECT, so only durable ownership can
            # arbitrate writers which both checked before either committed.
            store._entry_family_tables = lambda record_type: ()
            barrier.wait(timeout=15)
            index = 1 if backing is First else 2
            store.save(
                backing(
                    1,
                    f"test-1-{index if immutable else 1}",
                    f"test-shared-1~{1 if immutable else index}",
                )
            )
    except EntryIdConflictError:
        outcomes.put("conflict")
    except BaseException as error:
        outcomes.put(repr(error))
    else:
        outcomes.put("saved")


@pytest.mark.parametrize("immutable", (False, True))
def test_process_race_is_arbitrated_across_backings(tmp_path, immutable):
    path = str(tmp_path / "owners.sqlite")
    with Backend.sqlite(path) as database:
        opened(database).ensure_tables(First, Second)
    context = multiprocessing.get_context("spawn")
    barrier, outcomes = context.Barrier(2), context.Queue()
    processes = [
        context.Process(target=race_writer, args=(path, cls, immutable, barrier, outcomes)) for cls in (First, Second)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(25)
        assert not process.is_alive()
        assert process.exitcode == 0
    assert sorted(outcomes.get(timeout=2) for _ in processes) == ["conflict", "saved"]
    with Backend.sqlite(path) as database:
        assert claim_counts(database) == (1, 1)


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_revision_alternative_rollback_and_retry(dialect):
    with getattr(Backend, dialect)() as database:
        store = opened(database)
        original = First(1)
        sid = store.save(original)
        entry = store.fetch(First, sid, eager=True)
        store.replace(original, First(2))
        store.save(First(3), alternative_of=entry.id, alternative_kind="test")
        assert claim_counts(database) == (1, 3)
        with pytest.raises(RuntimeError, match="rollback"), store.transaction():
            store.save(Second(4, "test-1-20"))
            raise RuntimeError("rollback")
        assert claim_counts(database) == (1, 3)
        store.save(Second(4, "test-1-20"))
        assert claim_counts(database) == (2, 4)


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
@pytest.mark.parametrize("finalize", ("parity", "deferred"))
@pytest.mark.parametrize("workers", (1, 2))
def test_bulk_claims_protect_later_writers(dialect, finalize, workers):
    with getattr(Backend, dialect)() as database:
        store = opened(database)
        with store.bulk_ingest(finalize=finalize, workers=workers) as bulk:
            bulk.save(First(1, "test-1-10", "test-1-10~1"))
        assert claim_counts(database) == (1, 1)
        store._entry_family_tables = lambda record_type: ()
        with pytest.raises(EntryIdConflictError):
            store.save(Second(2, "test-1-10", "test-1-10~2"))
        assert claim_counts(database) == (1, 1)


@pytest.mark.parametrize("interrupted", (False, True))
def test_legacy_readonly_explicit_upgrade_is_repeatable(interrupted):
    with Backend.sqlite() as database:
        store = opened(database)
        sid = store.save(First(1))
        entry = store.fetch(First, sid, eager=True)
        store.replace(entry, First(2))
        legacy(database, drop=not interrupted)
        if interrupted:
            with database.engine.begin() as connection:
                connection.execute(
                    sqlalchemy.text("DELETE FROM _httk_immutable_id_owners WHERE immutable_id LIKE '%~2'")
                )
        old = opened(database)
        assert old.fetch(First, sid, eager=True).id == entry.id
        with pytest.raises(StorageLayoutUpgradeRequiredError, match="upgrade=True"):
            old.save(First(3))
        upgraded = opened(database, upgrade=True)
        assert upgraded.fetch(First, sid, eager=True) == entry
        assert claim_counts(database) == (1, 2)
        opened(database, upgrade=True)
        assert claim_counts(database) == (1, 2)


def test_conflicting_legacy_upgrade_does_not_rewrite_records():
    with Backend.sqlite() as database:
        store = opened(database)
        store.save(First(1, "test-1-1"))
        store.save(Second(2, "test-1-2"))
        legacy(database)
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("UPDATE ownership_second SET id='test-1-1'"))
        for _ in range(2):
            with pytest.raises(EntryIdConflictError):
                opened(database, upgrade=True)
        with database.engine.connect() as connection:
            assert connection.execute(sqlalchemy.text("SELECT id FROM ownership_second")).scalar_one() == "test-1-1"
            assert (
                connection.execute(
                    sqlalchemy.text("SELECT value FROM _httk_store_metadata WHERE key='identity_ownership'")
                ).first()
                is None
            )


@pytest.mark.parametrize("point", ("identity-claim:ownership_first", "parent-row-write:ownership_first"))
def test_degraded_crash_reconciles_only_absent_owners(point):
    from httk.store.backend.sql.store import _DegradedWriteCrash

    with Backend.sqlite(degraded=True) as database:
        store = opened(database)
        store._degraded_fault_hook = lambda current: current == point
        with pytest.raises(_DegradedWriteCrash):
            store.save(First(1, "test-1-10"))
        store._degraded_fault_hook = None
        if point.startswith("identity-claim"):
            # The failed parent is absent: a different backing may reuse the id.
            store.save(Second(2, "test-1-10"))
        else:
            # The parent survived: its claim must survive recovery as well.
            with pytest.raises(EntryIdConflictError):
                store.save(Second(2, "test-1-10"))
            store.save(First(1, "test-1-10"))
        assert claim_counts(database) == (1, 1)


def test_missing_unique_constraints_refuse_stamped_reopen():
    with Backend.sqlite() as database:
        opened(database)
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("DROP TABLE _httk_entry_id_owners"))
            connection.execute(
                sqlalchemy.text(
                    "CREATE TABLE _httk_entry_id_owners (family TEXT, entry_id TEXT, backing TEXT, logical_id BIGINT)"
                )
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError):
            opened(database)


def test_caught_identity_conflict_cannot_commit_a_partial_parent():
    with Backend.sqlite() as database:
        store = opened(database)
        store.save(First(1, "test-1-10"))
        store.ensure_tables(Second)
        store._entry_family_tables = lambda record_type: ()
        with pytest.raises(RuntimeError, match="rolled back"), store.transaction():
            with pytest.raises(EntryIdConflictError):
                store.save(Second(2, "test-1-10", "test-1-10~2"))
        with database.engine.connect() as connection:
            assert connection.execute(sqlalchemy.text("SELECT count(*) FROM ownership_second")).scalar_one() == 0
        assert claim_counts(database) == (1, 1)


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_caught_conflict_aborts_prior_writes_and_expires_lazy_rows(dialect):
    from httk.store.backend.sql.rows import ExpiredLazyRecordError

    with getattr(Backend, dialect)() as database:
        store = opened(database)
        store.save(First(1, "test-1-10"))
        store.ensure_tables(Second)
        with pytest.raises(RuntimeError, match="rolled back"), store.transaction():
            sid = store.save(Second(2, "test-1-20"))
            lazy = store.fetch(Second, sid)
            with pytest.raises(EntryIdConflictError):
                store.save(Second(3, "test-1-10"))
            with pytest.raises(ExpiredLazyRecordError):
                _ = lazy.value
        with pytest.raises(ExpiredLazyRecordError):
            _ = lazy.value
        with database.engine.connect() as connection:
            assert connection.execute(sqlalchemy.text("SELECT count(*) FROM ownership_second")).scalar_one() == 0
        assert claim_counts(database) == (1, 1)


def test_legacy_duckdb_can_open_on_a_readonly_connection(tmp_path):
    path = tmp_path / "old.duckdb"
    with Backend.duckdb(path) as database:
        store = opened(database)
        sid = store.save(First(1))
        legacy(database)
    before = path.read_bytes()
    with Backend.duckdb(path, read_only=True) as database:
        store = opened(database)
        assert store.fetch(First, sid, eager=True).value == 1
        with pytest.raises(StorageLayoutUpgradeRequiredError):
            store.save(First(2))
    assert path.read_bytes() == before


def test_mongo_claims_transaction_and_upgrade(mongo_test_database):
    from httk.store.backend.mongo import MongoStore

    def mongo(**kwargs):
        return MongoStore(
            mongo_test_database,
            entry_families=(DECLARATION,),
            entry_ids=EntryIdScheme("test", "1"),
            **kwargs,
        )

    store = mongo()
    first = First(1, "test-1-10")
    sid = store.save(first)
    store.replace(first, First(2))
    owners = mongo_test_database.database["_httk_identity_owners"]
    assert owners.count_documents({}) == 3
    with pytest.raises(RuntimeError, match="rollback"), store.transaction():
        store.save(Second(3, "test-1-30"))
        raise RuntimeError("rollback")
    assert owners.count_documents({}) == 3
    with pytest.raises(EntryIdConflictError):
        store.save(Second(4, "test-1-40", "test-1-10~1"))
    metadata = mongo_test_database.database["_httk_store_metadata"]
    metadata.update_one({"_id": "layout"}, {"$unset": {"identity_ownership": ""}})
    old = mongo()
    assert old.fetch(First, sid).id == first.id
    with pytest.raises(StorageLayoutUpgradeRequiredError):
        old.save(Second(3))
    mongo(upgrade=True)
    mongo(upgrade=True)
    assert owners.count_documents({}) == 3


def test_mongo_reopen_rejects_partial_identity_uniqueness(mongo_test_database):
    from httk.store.backend.mongo import MongoStore

    MongoStore(mongo_test_database, entry_families=(DECLARATION,))
    owners = mongo_test_database.database["_httk_identity_owners"]
    owners.drop_index("identity_value")
    owners.create_index(
        [("family", 1), ("kind", 1), ("value", 1)],
        name="identity_value",
        unique=True,
        partialFilterExpression={"kind": "never-claimed"},
    )
    with pytest.raises(StorageLayoutUpgradeRequiredError):
        MongoStore(mongo_test_database, entry_families=(DECLARATION,))


def test_mongo_degraded_orphan_claims_are_reconciled(mongo_test_database, monkeypatch):
    from pymongo.collection import Collection

    from httk.store.backend.mongo import MongoDatabase, MongoStore

    database = MongoDatabase(mongo_test_database.client, mongo_test_database.database.name, transactions="never")
    store = MongoStore(database, entry_families=(DECLARATION,), entry_ids=EntryIdScheme("test", "1"))
    original_insert = Collection.insert_one

    def fail_parent(collection, *args, **kwargs):
        if collection.name == "ownership_first":
            raise RuntimeError("parent interrupted")
        return original_insert(collection, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Collection, "insert_one", fail_parent)
        with pytest.raises(RuntimeError, match="parent interrupted"):
            store.save(First(1, "test-1-10"))
    owners = database.database["_httk_identity_owners"]
    assert owners.count_documents({}) == 2
    store.save(Second(2, "test-1-10"))
    assert owners.count_documents({}) == 2
