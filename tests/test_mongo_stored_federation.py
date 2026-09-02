"""Live MongoDB coverage for stored-entry federation plans."""

import os
import uuid

import pytest
from test_db_stored_federation import FederatedCalculation, FederationFirst, FederationSecond, _record

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, DuplicateEntryIdError, SqlStore, StoredEntryFederation, StoredEntrySource
from httk.store.backend.mongo import MongoDatabase, MongoStore


def _mongo_store(database):
    return MongoStore(
        database,
        entry_records={FederatedCalculation: (FederationFirst, FederationSecond)},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


@pytest.fixture
def mongo_store_pair(mongo_test_database):
    name = f"httk_test_federation_{uuid.uuid4().hex}"
    second_database = MongoDatabase(os.environ["HTTK_TEST_MONGODB_URI"], database=name, transactions="never")
    try:
        yield _mongo_store(mongo_test_database), _mongo_store(second_database)
    finally:
        second_database.client.drop_database(name)
        second_database.dispose()


def test_mongo_only_federation_pages_filters_and_audits(mongo_store_pair):
    first_store, second_store = mongo_store_pair
    first_records = (_record("alpha-one"), _record("alpha-two", second=True))
    second_records = (_record("beta-one"), _record("beta-two", second=True))
    for record in first_records:
        first_store.save(record)
    for record in second_records:
        second_store.save(record)

    federation = StoredEntryFederation(
        (
            StoredEntrySource(first_store, FederatedCalculation, "alpha", "alpha:"),
            StoredEntrySource(second_store, FederatedCalculation, "beta", "beta:"),
        )
    )
    complete = federation.query(sort=(("immutable_id", False),), limit=10)
    paged = tuple(
        row
        for offset in range(0, complete.total_count, 2)
        for row in federation.query(sort=(("immutable_id", False),), offset=offset, limit=2).rows
    )

    assert complete.total_count == 4
    assert tuple(row["id"] for row in paged) == tuple(row["id"] for row in complete.rows)
    federation.audit_duplicate_ids(batch_size=1)

    target = next(row["id"] for row in complete.rows if row["id"].startswith("beta:"))
    page = federation.query(f'id = "{target}"', sort=(("id", False),), limit=10)
    assert [row["id"] for row in page.rows] == [target]


def test_mongo_only_federation_probes_cross_source_prefix_collisions(mongo_store_pair):
    first_store, second_store = mongo_store_pair
    duplicate = _record("duplicate")
    first_store.save(duplicate)
    second_store.save(duplicate)
    federation = StoredEntryFederation(
        (
            StoredEntrySource(first_store, FederatedCalculation, "first", "shared:"),
            StoredEntrySource(second_store, FederatedCalculation, "second", "shared:"),
        )
    )
    public_id = "shared:httk.test-1-2"

    with pytest.raises(DuplicateEntryIdError) as page_error:
        federation.query(f'id = "{public_id}"', limit=1)
    with pytest.raises(DuplicateEntryIdError) as fetch_error:
        federation.fetch(public_id)
    with pytest.raises(DuplicateEntryIdError) as audit_error:
        federation.audit_duplicate_ids(batch_size=1)

    assert page_error.value.public_id == public_id
    assert fetch_error.value.public_id == public_id
    assert audit_error.value.public_id == public_id


def test_mixed_sql_mongo_federation_pages_audits_and_probes_prefix_collisions(mongo_test_database):
    with Backend.sqlite() as database:
        sql_store = SqlStore(
            database,
            entry_records={FederatedCalculation: (FederationFirst, FederationSecond)},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        mongo_store = _mongo_store(mongo_test_database)
        shared = _record("shared")
        sql_only = _record("sql-only", second=True)
        mongo_only = _record("mongo-only", second=True)
        sql_store.save(shared)
        sql_store.save(sql_only)
        sql_store._clear_identity_caches()
        mongo_store.save(shared)
        mongo_store.save(mongo_only)

        federation = StoredEntryFederation(
            (
                StoredEntrySource(sql_store, FederatedCalculation, "sql", "sql:"),
                StoredEntrySource(mongo_store, FederatedCalculation, "mongo", "mongo:"),
            )
        )
        complete = federation.query(sort=(("immutable_id", False),), limit=10)
        paged = tuple(
            row
            for offset in range(0, complete.total_count, 2)
            for row in federation.query(sort=(("immutable_id", False),), offset=offset, limit=2).rows
        )
        federation.audit_duplicate_ids(batch_size=1)

        assert complete.total_count == 4
        assert tuple(row["id"] for row in paged) == tuple(row["id"] for row in complete.rows)
        immutable_ids = [row["immutable_id"] for row in complete.rows]
        assert immutable_ids[0] == immutable_ids[1]
        assert immutable_ids[2] == immutable_ids[3]
        assert immutable_ids[0] != immutable_ids[2]
        typed = federation.query(sort=(("type", True), ("immutable_id", False)), limit=10)
        assert [row["immutable_id"] for row in typed.rows] == immutable_ids
        assert {row["id"].split(":", 1)[0] for row in complete.rows} == {"mongo", "sql"}
        target = next(row["id"] for row in complete.rows if row["id"].startswith("mongo:"))
        assert [row["id"] for row in federation.query(f'id = "{target}"', limit=10).rows] == [target]

        collision = StoredEntryFederation(
            (
                StoredEntrySource(sql_store, FederatedCalculation, "sql-collision", "shared:"),
                StoredEntrySource(mongo_store, FederatedCalculation, "mongo-collision", "shared:"),
            )
        )
        public_id = "shared:" + next(
            row["id"].split(":", 1)[1] for row in complete.rows if row["_httk_label"] == "shared"
        )
        with pytest.raises(DuplicateEntryIdError) as page_error:
            collision.query(f'id = "{public_id}"', limit=1)
        with pytest.raises(DuplicateEntryIdError) as audit_error:
            collision.audit_duplicate_ids(batch_size=1)

        assert page_error.value.public_id == public_id
        assert audit_error.value.public_id == public_id


def test_mongo_backed_source_serves_no_weak_link_relationships(mongo_test_database):
    """Documented current behavior: the federation's relationships collector is
    SQL-only, so a Mongo-backed source's exposed weak links serve nothing (the
    channel is present but empty for every row)."""
    from httk.core import Run, RunEntry
    from test_stored_federation_relationships import ArtifactCalculation, ArtifactRecord

    store = MongoStore(
        mongo_test_database,
        entry_records={ArtifactCalculation: (ArtifactRecord,), RunEntry: Run},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )
    target = store.fetch(Run, store.save(Run(source_id="ws:job")), eager=True)
    artifact = store.fetch(ArtifactRecord, store.save(ArtifactRecord("a1")), eager=True)
    store.link(artifact, "produced_by", target)

    federation = StoredEntryFederation((StoredEntrySource(store, ArtifactCalculation, "art"),))
    page = federation.query()
    assert len(page.rows) == 1
    assert all(rel == {} for rel in page.relationships)
