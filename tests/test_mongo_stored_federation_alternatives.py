"""Mongo alternative-serving parity with :mod:`test_stored_federation_alternatives`.

Every test needs a real replica-set MongoDB and is env-gated through the
``mongo_test_client`` fixture (skips when ``HTTK_TEST_MONGODB_URI`` is unset).
The family/record declarations and definition are reused from the SQL battery
so both backends share one alternative-serving specification.
"""

import uuid

import pytest
from test_stored_federation_alternatives import AltCalculation, AltRecord

from httk.store import EntryIdScheme
from httk.store.backend.mongo import MongoDatabase, MongoStore
from httk.store.backend.sql import (
    DuplicateEntryIdError,
    StoredEntryFederation,
    StoredEntrySource,
)
from httk.store.query.optimade_filters import FilterTranslationError


@pytest.fixture
def mongo_database_factory(mongo_test_client):
    """Yield a factory minting fresh, independently dropped live databases."""
    created: list[tuple[MongoDatabase, str]] = []

    def make() -> MongoDatabase:
        name = f"httk_test_{uuid.uuid4().hex}"
        database = MongoDatabase(mongo_test_client, name)
        created.append((database, name))
        return database

    try:
        yield make
    finally:
        for database, name in created:
            database.client.drop_database(name)


def _store(database: MongoDatabase) -> MongoStore:
    return MongoStore(
        database,
        entry_records={AltCalculation: (AltRecord,)},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


def _populate_one(store: MongoStore) -> str:
    """Save a main, its conventional and primitive alternatives, then replace the conventional one."""
    main = store.fetch(AltRecord, store.save(AltRecord("m1")), eager=True)
    conv = store.fetch(
        AltRecord,
        store.save(AltRecord("conv1"), alternative_of=main.id, alternative_kind="conventional"),
        eager=True,
    )
    store.save(AltRecord("prim1"), alternative_of=main.id, alternative_kind="primitive")
    store.replace(conv, AltRecord("conv1b"))
    return main.id


def _populate_two(store: MongoStore) -> str:
    main = store.fetch(AltRecord, store.save(AltRecord("m2")), eager=True)
    store.save(AltRecord("conv2"), alternative_of=main.id, alternative_kind="conventional")
    return main.id


def test_default_serving_excludes_alternatives(mongo_database_factory) -> None:
    """Ordinary query/fetch expose only mains; alternatives are invisible without the flag."""
    store = _store(mongo_database_factory())
    main_id = _populate_one(store)
    federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))

    page = federation.query()
    assert page.total_count == 1
    assert [row["id"] for row in page.rows] == [f"one/{main_id}"]
    assert all("_httk_kind" not in row for row in page.rows)
    assert federation.fetch(f"one/{main_id}")["_httk_label"] == "m1"
    # A composite alternative id is not a main; default fetch misses it.
    assert federation.fetch(f"one/{main_id}~conventional") is None


def test_alternatives_listing_is_composite_and_latest_only(
    mongo_database_factory,
) -> None:
    """Alternatives mode lists latest-of-kind rows with composite ids, group ``_httk_id`` and ``_httk_kind``."""
    store = _store(mongo_database_factory())
    main_id = _populate_one(store)
    federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))

    page = federation.query(sort=(("_httk_kind", False),), alternatives=True)
    assert page.total_count == 2
    assert [row["id"] for row in page.rows] == [
        f"one/{main_id}~conventional",
        f"one/{main_id}~primitive",
    ]
    assert [row["_httk_id"] for row in page.rows] == [
        f"one/{main_id}",
        f"one/{main_id}",
    ]
    assert [row["_httk_kind"] for row in page.rows] == ["conventional", "primitive"]
    # Latest revision of the conventional alternative wins after the replace.
    assert [row["_httk_label"] for row in page.rows] == ["conv1b", "prim1"]


def test_alternatives_filter_and_sort(mongo_database_factory) -> None:
    """``_httk_kind`` and the composite ``id`` are filterable and sortable in alternatives mode."""
    store = _store(mongo_database_factory())
    main_id = _populate_one(store)
    federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))

    by_kind = federation.query('_httk_kind = "primitive"', alternatives=True)
    assert by_kind.total_count == 1
    assert by_kind.rows[0]["id"] == f"one/{main_id}~primitive"
    assert by_kind.rows[0]["_httk_label"] == "prim1"

    by_id = federation.query(f'id = "one/{main_id}~conventional"', alternatives=True)
    assert by_id.total_count == 1
    assert by_id.rows[0]["_httk_label"] == "conv1b"

    descending = federation.query(sort=(("_httk_kind", True),), alternatives=True)
    assert [row["_httk_kind"] for row in descending.rows] == [
        "primitive",
        "conventional",
    ]

    # ``_httk_kind`` is not a recognized property in ordinary (mains) mode.
    with pytest.raises(FilterTranslationError, match="unrecognized property"):
        federation.query('_httk_kind = "conventional"')


def test_fetch_alternative_hit_and_misses(mongo_database_factory) -> None:
    """``fetch_alternative`` addresses one latest alternative; kind/entry/format misses return ``None``."""
    store = _store(mongo_database_factory())
    main_id = _populate_one(store)
    federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))

    hit = federation.fetch_alternative(f"one/{main_id}", "conventional")
    assert hit is not None
    assert hit["id"] == f"one/{main_id}~conventional"
    assert hit["_httk_id"] == f"one/{main_id}"
    assert hit["_httk_kind"] == "conventional"
    assert hit["_httk_label"] == "conv1b"  # replaced alternative returns its latest revision

    assert federation.fetch_alternative(f"one/{main_id}", "tetragonal") is None  # kind miss
    assert federation.fetch_alternative("one/httk.test-does-not-exist-1", "conventional") is None  # entry miss
    assert federation.fetch_alternative(f"one/{main_id}", "Bad Kind") is None  # malformed kind


def test_revisions_are_mains_only(mongo_database_factory) -> None:
    """An alternative's lineage never enters the revision stream, and its revision id never fetches."""
    store = _store(mongo_database_factory())
    main_id = _populate_one(store)
    federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))

    revisions = federation.query(sort=(("immutable_id", False),), revisions=True)
    assert revisions.total_count == 1  # only the main's single revision
    assert [row["id"] for row in revisions.rows] == [f"one/{main_id}~1"]
    assert all("conventional" not in row["id"] and "primitive" not in row["id"] for row in revisions.rows)

    # The alternative's revision id is invisible to the mains-only revision path.
    assert federation.fetch_revision(f"one/{main_id}", f"one/{main_id}~conventional~1") is None
    assert federation.fetch_revision(f"one/{main_id}", f"one/{main_id}~1")["_httk_label"] == "m1"


def test_two_sources_prefix_disambiguate_alternatives(mongo_database_factory) -> None:
    """Two prefixed sources with identical raw ids merge without collision."""
    store_one = _store(mongo_database_factory())
    store_two = _store(mongo_database_factory())
    id_one = _populate_one(store_one)
    id_two = _populate_two(store_two)
    assert id_one == id_two  # each fresh store restarts numbering
    federation = StoredEntryFederation(
        (
            StoredEntrySource(store_one, AltCalculation, "one", public_id_prefix="one/"),
            StoredEntrySource(store_two, AltCalculation, "two", public_id_prefix="two/"),
        )
    )

    page = federation.query(sort=(("_httk_kind", False),), alternatives=True)
    assert page.total_count == 3
    assert [row["id"] for row in page.rows] == [
        f"one/{id_one}~conventional",
        f"two/{id_two}~conventional",
        f"one/{id_one}~primitive",
    ]
    assert federation.fetch_alternative(f"two/{id_two}", "conventional")["_httk_label"] == "conv2"


def test_audit_duplicate_ids_unaffected_by_alternatives(mongo_database_factory) -> None:
    """Auditing scans mains only, so an alternative sharing its main's id is not a duplicate."""
    store = _store(mongo_database_factory())
    _populate_one(store)  # a main plus alternatives that copy the main's id
    store.save(AltRecord("m3"))  # a second, distinct main
    federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))
    federation.audit_duplicate_ids()  # no raise: alternatives are excluded from the main-only audit


def test_duplicate_alternative_across_sources(mongo_database_factory) -> None:
    """A composite alternative id claimed by two shared-prefix sources is a collision."""
    store_one = _store(mongo_database_factory())
    store_two = _store(mongo_database_factory())
    id_one = _populate_one(store_one)
    _populate_one(store_two)  # identical raw ids under the same shared prefix
    federation = StoredEntryFederation(
        (
            StoredEntrySource(store_one, AltCalculation, "one", public_id_prefix="shared/"),
            StoredEntrySource(store_two, AltCalculation, "two", public_id_prefix="shared/"),
        )
    )
    with pytest.raises(DuplicateEntryIdError, match="duplicate public entry id"):
        federation.fetch_alternative(f"shared/{id_one}", "conventional")


def test_alternatives_id_sort_paging_is_stable_across_a_tied_group(mongo_database_factory) -> None:
    """Paging an id-sorted 3-kind group at limit=1 yields each composite exactly once.

    All three alternatives share their main's ``f.id``; the composite id key
    sorts server-side on ``f.id`` alone, so without the unique-sid final tiebreak
    MongoDB's unstable ``$sort`` could duplicate or drop a row across a page
    boundary within the tie.
    """
    store = _store(mongo_database_factory())
    main = store.fetch(AltRecord, store.save(AltRecord("m")), eager=True)
    for kind in ("conventional", "primitive", "tetragonal"):
        store.save(AltRecord(kind), alternative_of=main.id, alternative_kind=kind)
    federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))

    expected = [f"one/{main.id}~{kind}" for kind in ("conventional", "primitive", "tetragonal")]
    first = federation.query(sort=(("id", False),), offset=0, limit=1, alternatives=True)
    assert first.total_count == 3
    paged = [
        row["id"]
        for offset in range(first.total_count)
        for row in federation.query(sort=(("id", False),), offset=offset, limit=1, alternatives=True).rows
    ]
    assert paged == expected  # ascending order, each exactly once: no dupes, no drops

    # Deterministic guard: the composite id key must be followed by the unique
    # ``_id`` server sort key (Mongo's tie-break is otherwise unspecified).
    plan = store.stored_property_plan(AltCalculation)
    stream = plan.candidate_searchers(sort=(("id", False),), public_id_prefix="one/", alternatives=True)[0]
    assert stream.searcher._sorts[-1][0]._path == "_id"
