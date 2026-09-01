"""Stored federation alternative serving over composite ``<id>~<kind>`` identifiers."""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
from httk.core import PropertyDefinition, load_entry_type_definition
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, StoredPropertyProjection, Unique

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, DuplicateEntryIdError, SqlStore, StoredEntryFederation, StoredEntrySource
from httk.store.query.optimade_filters import FilterTranslationError

_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations"


class AltCalculation:
    """A minimal calculation family exposing one queryable label."""

    type = "calculations"
    definition_id = _DEFINITION

    @staticmethod
    def entry_type_definition():
        return load_entry_type_definition(_DEFINITION).extended(
            {
                "_httk_label": PropertyDefinition.from_simple("_httk_label", description="Test label."),
            }
        )


@dataclass(frozen=True)
class AltRecord:
    """One alternative-capable backing (default ``content_id`` dedup)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="stored_federation_alternatives")

    label: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = {
        "_httk_label": StoredPropertyProjection(
            response=lambda record: record.label,
            query=lambda context, operator, literal: context.compare(
                context.field("label"), operator, context.constant(literal)
            ),
            sort=lambda context: context.field("label"),
        )
    }


register_entry_family(
    name="test-stored-federation-alternatives",
    family=f"{__name__}:AltCalculation",
    definition_id=_DEFINITION,
)
register_entry_record(
    name="test-stored-federation-alternatives-record",
    family="test-stored-federation-alternatives",
    record=f"{__name__}:AltRecord",
)


def _store(database: Backend) -> SqlStore:
    return SqlStore(
        database,
        entry_records={AltCalculation: (AltRecord,)},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


def _populate_one(store: SqlStore) -> str:
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


def _populate_two(store: SqlStore) -> str:
    main = store.fetch(AltRecord, store.save(AltRecord("m2")), eager=True)
    store.save(AltRecord("conv2"), alternative_of=main.id, alternative_kind="conventional")
    return main.id


def test_default_serving_excludes_alternatives() -> None:
    """Ordinary query/fetch expose only mains; alternatives are invisible without the flag."""
    with Backend.sqlite() as database:
        store = _store(database)
        main_id = _populate_one(store)
        federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))

        page = federation.query()
        assert page.total_count == 1
        assert [row["id"] for row in page.rows] == [f"one/{main_id}"]
        assert all("_httk_kind" not in row for row in page.rows)
        assert federation.fetch(f"one/{main_id}")["_httk_label"] == "m1"
        # A composite alternative id is not a main; default fetch misses it.
        assert federation.fetch(f"one/{main_id}~conventional") is None


def test_alternatives_listing_is_composite_and_latest_only() -> None:
    """Alternatives mode lists latest-of-kind rows with composite ids, group ``_httk_id`` and ``_httk_kind``."""
    with Backend.sqlite() as database:
        store = _store(database)
        main_id = _populate_one(store)
        federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))

        page = federation.query(sort=(("_httk_kind", False),), alternatives=True)
        assert page.total_count == 2
        assert [row["id"] for row in page.rows] == [
            f"one/{main_id}~conventional",
            f"one/{main_id}~primitive",
        ]
        assert [row["_httk_id"] for row in page.rows] == [f"one/{main_id}", f"one/{main_id}"]
        assert [row["_httk_kind"] for row in page.rows] == ["conventional", "primitive"]
        # Latest revision of the conventional alternative wins after the replace.
        assert [row["_httk_label"] for row in page.rows] == ["conv1b", "prim1"]


def test_alternatives_filter_and_sort() -> None:
    """``_httk_kind`` and the composite ``id`` are filterable and sortable in alternatives mode."""
    with Backend.sqlite() as database:
        store = _store(database)
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
        assert [row["_httk_kind"] for row in descending.rows] == ["primitive", "conventional"]

        # ``_httk_kind`` is not a recognized property in ordinary (mains) mode.
        with pytest.raises(FilterTranslationError, match="unrecognized property"):
            federation.query('_httk_kind = "conventional"')


def test_fetch_alternative_hit_and_misses() -> None:
    """``fetch_alternative`` addresses one latest alternative; kind/entry/format misses return ``None``."""
    with Backend.sqlite() as database:
        store = _store(database)
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


def test_revisions_are_mains_only() -> None:
    """An alternative's lineage never enters the revision stream, and its revision id never fetches."""
    with Backend.sqlite() as database:
        store = _store(database)
        main_id = _populate_one(store)
        federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))

        revisions = federation.query(sort=(("immutable_id", False),), revisions=True)
        assert revisions.total_count == 1  # only the main's single revision
        assert [row["id"] for row in revisions.rows] == [f"one/{main_id}~1"]
        assert all("conventional" not in row["id"] and "primitive" not in row["id"] for row in revisions.rows)

        # The alternative's revision id is invisible to the mains-only revision path.
        assert federation.fetch_revision(f"one/{main_id}", f"one/{main_id}~conventional~1") is None
        assert federation.fetch_revision(f"one/{main_id}", f"one/{main_id}~1")["_httk_label"] == "m1"


def test_two_sources_prefix_disambiguate_alternatives() -> None:
    """Two prefixed sources with identical raw ids merge without collision."""
    with Backend.sqlite() as database_one, Backend.sqlite() as database_two:
        store_one = _store(database_one)
        store_two = _store(database_two)
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


def test_audit_duplicate_ids_unaffected_by_alternatives() -> None:
    """Auditing scans mains only, so an alternative sharing its main's id is not a duplicate."""
    with Backend.sqlite() as database:
        store = _store(database)
        _populate_one(store)  # a main plus alternatives that copy the main's id
        store.save(AltRecord("m3"))  # a second, distinct main
        federation = StoredEntryFederation((StoredEntrySource(store, AltCalculation, "one", public_id_prefix="one/"),))
        federation.audit_duplicate_ids()  # no raise: alternatives are excluded from the main-only audit


def test_duplicate_alternative_across_sources() -> None:
    """A composite alternative id claimed by two shared-prefix sources is a collision."""
    with Backend.sqlite() as database_one, Backend.sqlite() as database_two:
        store_one = _store(database_one)
        store_two = _store(database_two)
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
