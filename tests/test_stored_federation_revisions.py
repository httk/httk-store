"""Stored federation revision serving over physical entry identifiers."""

from dataclasses import dataclass, field, replace
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core import PropertyDefinition, load_entry_type_definition
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, StoredPropertyProjection, Unique

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, DuplicateEntryIdError, SqlStore, StoredEntryFederation, StoredEntrySource
from httk.store.query.optimade_filters import FilterTranslationError

_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations"


class RevisionCalculation:
    """A minimal calculation family exposing the revision identity fields."""

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
class RevisionRecord:
    """One revision-capable backing."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="stored_federation_revisions")

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
    name="test-stored-federation-revisions",
    family=f"{__name__}:RevisionCalculation",
    definition_id=_DEFINITION,
)
register_entry_record(
    name="test-stored-federation-revisions-record",
    family="test-stored-federation-revisions",
    record=f"{__name__}:RevisionRecord",
)


def test_query_and_fetch_revisions_use_rendered_revision_identity() -> None:
    """Default pages select lineages while revision pages expose every immutable row."""
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={RevisionCalculation: (RevisionRecord,)},
            entry_ids=EntryIdScheme("httk.test", "1"),
            store_timestamp_resolution=1,
        )
        store._clock = iter((1_000, 2_000, 3_000)).__next__
        a = RevisionRecord("A")
        store.save(a)
        store.replace(a, replace(a, label="A2"))
        store.save(RevisionRecord("B"))
        federation = StoredEntryFederation((StoredEntrySource(store, RevisionCalculation, "test"),))

        latest = federation.query(sort=(("_httk_label", False),))
        assert latest.total_count == 2
        assert [(row["id"], row["_httk_label"]) for row in latest.rows] == [
            ("httk.test-1-1", "A2"),
            ("httk.test-1-3", "B"),
        ]

        revisions = federation.query(sort=(("immutable_id", False),), revisions=True)
        assert revisions.total_count == 3
        assert [row["id"] for row in revisions.rows] == [
            "httk.test-1-1~1",
            "httk.test-1-1~2",
            "httk.test-1-3~1",
        ]
        assert [row["_httk_id"] for row in revisions.rows] == [
            "httk.test-1-1",
            "httk.test-1-1",
            "httk.test-1-3",
        ]
        assert [row["_httk_label"] for row in revisions.rows] == ["A", "A2", "B"]
        assert federation.query('id = "httk.test-1-1~1"', revisions=True).total_count == 1
        assert federation.query('_httk_id = "httk.test-1-1"', revisions=True).total_count == 2
        assert [row["_httk_id"] for row in federation.query(sort=(("_httk_id", False),), revisions=True).rows] == [
            "httk.test-1-1",
            "httk.test-1-1",
            "httk.test-1-3",
        ]
        with pytest.raises(FilterTranslationError, match="unrecognized property"):
            federation.query('_httk_id = "httk.test-1-1"')
        assert all("_httk_id" not in row for row in latest.rows)

        assert federation.fetch("httk.test-1-1")[0]["_httk_label"] == "A2"
        assert federation.fetch_revision("httk.test-1-1", "httk.test-1-1~1")[0]["_httk_label"] == "A"
        assert federation.fetch_revision("httk.test-1-3", "httk.test-1-1~1") is None
        assert federation.query(revisions=True, as_of=2_500).total_count == 2


def test_audit_duplicate_ids_inspects_a_single_backing_stream() -> None:
    """The explicit audit detects duplicate lineage ids within one backing."""
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={RevisionCalculation: (RevisionRecord,)},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        first_sid = store.save(RevisionRecord("A"))
        second_sid = store.save(RevisionRecord("B"))
        table = store._table("stored_federation_revisions")
        with database.engine.begin() as connection:
            entry_id = connection.execute(
                sqlalchemy.select(table.c["id"]).where(table.c["sid"] == first_sid)
            ).scalar_one()
            connection.execute(sqlalchemy.update(table).where(table.c["sid"] == second_sid).values(id=entry_id))
        federation = StoredEntryFederation((StoredEntrySource(store, RevisionCalculation, "test"),))
        with pytest.raises(DuplicateEntryIdError, match="duplicate public entry id"):
            federation.audit_duplicate_ids()
