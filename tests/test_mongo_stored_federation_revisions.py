"""Mongo revision-serving parity with the SQL stored federation tests."""

from dataclasses import replace

import pytest
from test_stored_federation_revisions import RevisionCalculation, RevisionRecord

from httk.store import EntryIdScheme
from httk.store.backend.mongo import MongoStore
from httk.store.backend.mongo.mapping import collection_name_for
from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql import DuplicateEntryIdError, StoredEntryFederation, StoredEntrySource


def _federation(database):
    store = MongoStore(
        database,
        entry_records={RevisionCalculation: (RevisionRecord,)},
        entry_ids=EntryIdScheme("httk.test", "1"),
        store_timestamp_resolution=1,
    )
    store._clock = iter((1_000, 2_000, 3_000)).__next__
    first = RevisionRecord("A")
    first_sid = store.save(first)
    first_stored = store.fetch(RevisionRecord, first_sid)
    replacement_sid = store.replace(first, replace(first, label="A2"))
    latest_a = store.fetch(RevisionRecord, replacement_sid)
    second_sid = store.save(RevisionRecord("B"))
    latest_b = store.fetch(RevisionRecord, second_sid)
    return store, StoredEntryFederation((StoredEntrySource(store, RevisionCalculation, "test"),)), {
        first_stored.label: first_stored.immutable_id,
        latest_a.label: latest_a.immutable_id,
        latest_b.label: latest_b.immutable_id,
    }


def test_mongo_revision_query_fetch_and_as_of(mongo_test_database):
    """Default queries are latest-only; revisions expose every immutable row."""
    _store, federation, stored_immutable_ids = _federation(mongo_test_database)

    latest = federation.query(sort=(("_httk_label", False),))
    assert latest.total_count == 2
    assert [row["_httk_label"] for row in latest.rows] == ["A2", "B"]
    assert {row["_httk_label"]: row["immutable_id"] for row in latest.rows} == {
        "A2": stored_immutable_ids["A2"],
        "B": stored_immutable_ids["B"],
    }
    assert all("_httk_id" not in row for row in latest.rows)

    revisions = federation.query(sort=(("immutable_id", False),), revisions=True)
    assert revisions.total_count == 3
    assert [row["id"] for row in revisions.rows] == [
        "httk.test-1-1~1",
        "httk.test-1-1~2",
        "httk.test-1-3~1",
    ]
    assert all(row["immutable_id"] == row["id"] for row in revisions.rows)
    assert [row["_httk_id"] for row in revisions.rows] == [
        "httk.test-1-1",
        "httk.test-1-1",
        "httk.test-1-3",
    ]
    assert federation.query('_httk_id = "httk.test-1-1"', revisions=True).total_count == 2
    assert federation.fetch("httk.test-1-1")[0]["_httk_label"] == "A2"
    assert federation.fetch_revision("httk.test-1-1", "httk.test-1-1~1")[0]["_httk_label"] == "A"
    assert federation.fetch_revision("httk.test-1-3", "httk.test-1-1~1") is None
    assert federation.query(revisions=True, as_of=2_500).total_count == 2


def test_mongo_revision_audit_detects_injected_lineage_duplicate(mongo_test_database):
    """An out-of-band duplicate ``f.id`` is reported by the explicit audit."""
    _store, federation, _stored_immutable_ids = _federation(mongo_test_database)
    collection = mongo_test_database.database[collection_name_for(resolve_schema(RevisionRecord))]
    source = collection.find_one({"f.id": "httk.test-1-3"})
    assert source is not None
    collection.update_one({"_id": source["_id"]}, {"$set": {"f.id": "httk.test-1-1"}})
    with pytest.raises(DuplicateEntryIdError, match="httk.test-1-1"):
        federation.audit_duplicate_ids()
