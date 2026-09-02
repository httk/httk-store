"""Stored federation serving of exposed weak-link relationships.

Mirrors the in-memory provider path (StoreEntryProvider._collect_weak_relationships):
links bind lineages, retracted links vanish, and each target resolves at its
lineage's latest revision. The related id is the target row's raw stored ``id``
column (F9: it matches a mounted endpoint only when the target source's
``public_id_prefix`` is empty).

The weak-link target is the prefixed core ``RunEntry``/``Run`` family, so the
relationship type is exercised as a real wire name (``_httk_runs``): with no
``served_type_names`` mapping the collector falls back to the target family's
served name, never its internal ``runs`` vocabulary.
"""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

from httk.core import PropertyDefinition, RelatedEntry, Run, RunEntry, load_entry_type_definition
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import (
    IdentitySkip,
    Indexed,
    StorageInfo,
    StoredPropertyProjection,
    Unique,
    WeakLink,
)

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore, StoredEntryFederation, StoredEntrySource

_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations"

_LABEL_PROJECTION = {
    "_httk_label": StoredPropertyProjection(
        response=lambda record: record.label,
        query=lambda context, operator, literal: context.compare(
            context.field("label"), operator, context.constant(literal)
        ),
        sort=lambda context: context.field("label"),
    )
}


def _calculations_definition():
    return load_entry_type_definition(_DEFINITION).extended(
        {"_httk_label": PropertyDefinition.from_simple("_httk_label", description="Test label.")}
    )


@dataclass(frozen=True)
class ArtifactRecord:
    """The declaring family's backing, exposing one weak link to the prefixed ``Run`` family."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="stored_federation_rel_artifact",
        links=(
            WeakLink(
                "produced_by",
                Run,
                exposed_relationship=True,
                role="artifact+output",
                description="Produced by",
            ),
        ),
    )

    label: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = _LABEL_PROJECTION


@dataclass(frozen=True)
class PlainRecord:
    """A backing declaring no exposed weak links at all."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="stored_federation_rel_plain")

    label: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = _LABEL_PROJECTION


class ArtifactCalculation:
    """The declaring, federation-served family."""

    type = "calculations"
    definition_id = _DEFINITION

    @staticmethod
    def entry_type_definition():
        return _calculations_definition()


class PlainCalculation:
    """A federation-served family whose backing declares no exposed weak links."""

    type = "calculations"
    definition_id = _DEFINITION

    @staticmethod
    def entry_type_definition():
        return _calculations_definition()


register_entry_family(
    name="test-stored-federation-rel-artifact",
    family=f"{__name__}:ArtifactCalculation",
    definition_id=_DEFINITION,
)
register_entry_record(
    name="test-stored-federation-rel-artifact-record",
    family="test-stored-federation-rel-artifact",
    record=f"{__name__}:ArtifactRecord",
)
register_entry_family(
    name="test-stored-federation-rel-plain",
    family=f"{__name__}:PlainCalculation",
    definition_id=_DEFINITION,
)
register_entry_record(
    name="test-stored-federation-rel-plain-record",
    family="test-stored-federation-rel-plain",
    record=f"{__name__}:PlainRecord",
)


def _store(database: Backend) -> SqlStore:
    return SqlStore(
        database,
        entry_records={ArtifactCalculation: (ArtifactRecord,), RunEntry: Run},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


def _saved(store: SqlStore, record):
    """Save a record and return the reloaded copy carrying its minted ids."""
    return store.fetch(type(record), store.save(record), eager=True)


def _produced_by(target_id: str, related_type: str = "_httk_runs") -> RelatedEntry:
    return RelatedEntry(
        related_type,
        target_id,
        description="Produced by",
        role="artifact+output",
        label="produced_by",
    )


def test_query_serves_weak_link_relationships_with_role_and_label() -> None:
    """Linked rows carry the relationship (role + label); unlinked rows carry an empty mapping.

    No ``served_type_names`` is supplied, so the unmapped prefixed target falls
    back to its served wire name ``_httk_runs`` (never the internal ``runs``).
    """
    with Backend.sqlite() as database:
        store = _store(database)
        t1 = _saved(store, Run(source_id="t1"))
        t2 = _saved(store, Run(source_id="t2"))
        a1 = _saved(store, ArtifactRecord("a1"))
        _saved(store, ArtifactRecord("a2"))  # no links
        store.link(a1, "produced_by", t1)
        store.link(a1, "produced_by", t2)
        federation = StoredEntryFederation((StoredEntrySource(store, ArtifactCalculation, "art"),))

        page = federation.query(sort=(("_httk_label", False),))
        by_id = {row["id"]: rel for row, rel in zip(page.rows, page.relationships, strict=True)}
        assert by_id[a1.id] == {"_httk_runs": (_produced_by(t1.id), _produced_by(t2.id))}
        assert all(rel == {} for entry_id, rel in by_id.items() if entry_id != a1.id)


def test_retracted_weak_link_disappears() -> None:
    """Unlinking a live pair removes the relationship from the page channel."""
    with Backend.sqlite() as database:
        store = _store(database)
        t1 = _saved(store, Run(source_id="t1"))
        a1 = _saved(store, ArtifactRecord("a1"))
        store.link(a1, "produced_by", t1)
        store.unlink(a1, "produced_by", t1)
        federation = StoredEntryFederation((StoredEntrySource(store, ArtifactCalculation, "art"),))
        page = federation.query()
        assert [rel for rel in page.relationships] == [{}]


def test_as_of_page_pairs_with_live_link_state() -> None:
    """A historic page reflects the CURRENT link state: a post-cutoff unlink hides the relationship."""
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={ArtifactCalculation: (ArtifactRecord,), RunEntry: Run},
            entry_ids=EntryIdScheme("httk.test", "1"),
            store_timestamp_resolution=1,
        )
        store._clock = iter((1_000, 2_000, 3_000, 4_000, 5_000)).__next__
        t1 = _saved(store, Run(source_id="t1"))  # ts 1000
        a1 = _saved(store, ArtifactRecord("a1"))  # ts 2000
        store.link(a1, "produced_by", t1)  # ts 3000
        store.unlink(a1, "produced_by", t1)  # ts 4000 (after the cutoff below)
        federation = StoredEntryFederation((StoredEntrySource(store, ArtifactCalculation, "art"),))

        page = federation.query(as_of=3_500)
        assert [row["id"] for row in page.rows] == [a1.id]  # the row is historic (as-of 3500)
        assert list(page.relationships) == [{}]  # but relationships use the live state: unlinked


def test_relationship_attaches_to_every_source_revision() -> None:
    """A replaced source lineage carries the link on every immutable revision row."""
    with Backend.sqlite() as database:
        store = _store(database)
        t1 = _saved(store, Run(source_id="t1"))
        a1 = _saved(store, ArtifactRecord("a1"))
        store.link(a1, "produced_by", t1)
        store.replace(a1, ArtifactRecord("a1b"))
        federation = StoredEntryFederation((StoredEntrySource(store, ArtifactCalculation, "art"),))

        revisions = federation.query(sort=(("immutable_id", False),), revisions=True)
        assert len(revisions.rows) == 2
        assert all(row["_httk_id"] == a1.id for row in revisions.rows)
        assert all(rel == {"_httk_runs": (_produced_by(t1.id),)} for rel in revisions.relationships)


def test_related_id_follows_target_lineage_latest() -> None:
    """The related id resolves at the target lineage's latest row (id is lineage-level)."""
    with Backend.sqlite() as database:
        store = _store(database)
        t1 = _saved(store, Run(source_id="t1"))
        a1 = _saved(store, ArtifactRecord("a1"))
        store.link(a1, "produced_by", t1)
        store.replace(t1, Run(source_id="t1b"))
        federation = StoredEntryFederation((StoredEntrySource(store, ArtifactCalculation, "art"),))
        page = federation.query()
        assert list(page.relationships) == [{"_httk_runs": (_produced_by(t1.id),)}]


def test_served_type_names_translates_relationship_type() -> None:
    """An explicit served_type_names mapping overrides the wire name on both the key and the entry."""
    with Backend.sqlite() as database:
        store = _store(database)
        t1 = _saved(store, Run(source_id="t1"))
        a1 = _saved(store, ArtifactRecord("a1"))
        store.link(a1, "produced_by", t1)
        federation = StoredEntryFederation(
            (StoredEntrySource(store, ArtifactCalculation, "art"),),
            served_type_names={"runs": "_httk_target"},
        )
        page = federation.query()
        assert list(page.relationships) == [{"_httk_target": (_produced_by(t1.id, "_httk_target"),)}]


def test_single_paths_return_the_relationship_pair() -> None:
    """fetch, fetch_revision, and fetch_alternative return ``(row, relationships)``."""
    with Backend.sqlite() as database:
        store = _store(database)
        t1 = _saved(store, Run(source_id="t1"))
        a1 = _saved(store, ArtifactRecord("a1"))
        store.link(a1, "produced_by", t1)
        alt = store.fetch(
            ArtifactRecord,
            store.save(ArtifactRecord("a1-conv"), alternative_of=a1.id, alternative_kind="conventional"),
            eager=True,
        )
        store.link(alt, "produced_by", t1)
        federation = StoredEntryFederation((StoredEntrySource(store, ArtifactCalculation, "art"),))

        row, related = federation.fetch(a1.id)
        assert row["id"] == a1.id
        assert related == {"_httk_runs": (_produced_by(t1.id),)}

        rev_row, rev_related = federation.fetch_revision(a1.id, a1.immutable_id)
        assert rev_row["_httk_id"] == a1.id
        assert rev_related == {"_httk_runs": (_produced_by(t1.id),)}

        alt_row, alt_related = federation.fetch_alternative(a1.id, "conventional")
        assert alt_row["_httk_kind"] == "conventional"
        assert alt_related == {"_httk_runs": (_produced_by(t1.id),)}


def test_family_without_exposed_links_has_an_empty_channel() -> None:
    """A backing declaring no exposed weak links yields an all-empty, query-free channel."""
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={PlainCalculation: (PlainRecord,)},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        store.save(PlainRecord("p1"))
        store.save(PlainRecord("p2"))
        federation = StoredEntryFederation((StoredEntrySource(store, PlainCalculation, "plain"),))
        page = federation.query()
        assert len(page.rows) == 2
        assert all(rel == {} for rel in page.relationships)


def test_federation_serves_a_prefixed_run_family() -> None:
    """A prefixed family (RunEntry/Run) federates in wire form: served type, non-null values, filter, and sort."""
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={RunEntry: Run},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        store.save(Run(source_id="ws:a", workflow_declaration_uri="https://wf.example/a"))
        store.save(Run(source_id="ws:b", workflow_declaration_uri="https://wf.example/b"))
        federation = StoredEntryFederation((StoredEntrySource(store, RunEntry, "runs"),))

        page = federation.query(sort=(("_httk_source_id", False),))
        assert [row["type"] for row in page.rows] == ["_httk_runs", "_httk_runs"]
        assert [row["_httk_source_id"] for row in page.rows] == ["ws:a", "ws:b"]
        assert all(row["_httk_workflow_declaration_uri"] is not None for row in page.rows)

        filtered = federation.query('_httk_source_id = "ws:a"')
        assert [row["_httk_source_id"] for row in filtered.rows] == ["ws:a"]
