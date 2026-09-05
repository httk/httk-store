"""Family identity survives internal-type collisions in relationship mounts."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from httk.core import Run, RunEdge, RunEntry
from httk.core.optimade import parse_optimade_filter
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import StorageInfo
from test_stored_relationship_filters import (
    CitingFamily,
    CitingRow,
    RecordFamily,
    RecordRow,
    RefBackA,
    ReferenceFamily,
    ReferenceRow,
    RunBackingA,
    RunBackingB,
    TwoBackingRunFamily,
    TwoBackRefFamily,
    _store,
)

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore, StoredEntryFederation, StoredEntrySource
from httk.store.backend.sql.stored_federation import related_property_resolver_factory
from httk.store.backend.sql.stored_properties import stored_property_sql_plan


@dataclass(frozen=True)
class CrossFamilyCitingRow(CitingRow):
    """A row citing two distinct families sharing the references wire type."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_cross_family")

    cite_b: RefBackA | None = None


class CrossFamilyCitingFamily(CitingFamily):
    """The records family containing cross-family citations."""


register_entry_family(
    name="rel-filter-cross-family",
    family=f"{__name__}:CrossFamilyCitingFamily",
    definition_id=CitingFamily.definition_id,
)
register_entry_record(
    name="rel-filter-cross-family-rec", family="rel-filter-cross-family", record=f"{__name__}:CrossFamilyCitingRow"
)


def test_unmounted_run_family_keeps_each_backings_raw_reverse_id() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={
                RunEntry: Run,
                TwoBackingRunFamily: (RunBackingA, RunBackingB),
                RecordFamily: RecordRow,
            },
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        target = store.fetch(RecordRow, store.save(RecordRow("target")), eager=True)
        edge = RunEdge("in", "records", target.id)
        mounted = store.fetch(Run, store.save(Run(inputs=(edge,))), eager=True)
        unmounted = [
            store.fetch(backing, store.save(backing(inputs=(edge,))), eager=True)
            for backing in (RunBackingA, RunBackingB)
        ]
        source = StoredEntrySource(store, RecordFamily, "records", "D:")
        federation = StoredEntryFederation(
            (source,), source_inventory=(source, StoredEntrySource(store, RunEntry, "runs", "R:"))
        )
        expected = {"R:" + mounted.id, *(record.id for record in unmounted)}
        assert {entry.id for entry in federation.query().relationships[0]["_httk_is_input"]} == expected
        for value in expected:
            page = federation.query(f'_httk_relationships._httk_is_input.id HAS "{value}"')
            assert [row["id"] for row in page.rows] == ["D:" + target.id]
        for record in unmounted:
            page = federation.query(f'_httk_relationships._httk_is_input.id HAS "R:{record.id}"')
            assert page.rows == ()


@pytest.mark.parametrize("sibling_prefix", ("D:", "other:"))
def test_loose_types_require_consistent_prefixes_across_mounted_families(sibling_prefix: str) -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        target = store.fetch(RecordRow, store.save(RecordRow("target")), eager=True)
        run = store.fetch(Run, store.save(Run(inputs=(RunEdge("in", "records", target.id),))), eager=True)
        source = StoredEntrySource(store, RunEntry, "runs", "R:")
        inventory = (
            source,
            StoredEntrySource(store, RecordFamily, "records", "D:"),
            StoredEntrySource(store, CitingFamily, "citing", sibling_prefix),
        )
        if sibling_prefix != "D:":
            with pytest.raises(ValueError, match="Ambiguous relationship target type 'records'"):
                StoredEntryFederation((source,), source_inventory=inventory)
            return
        federation = StoredEntryFederation((source,), source_inventory=inventory)
        page = federation.query(f'_httk_relationships._httk_has_input.id HAS "D:{target.id}"')
        assert [row["id"] for row in page.rows] == ["R:" + run.id]
        assert page.relationships[0]["_httk_has_input"][0].id == "D:" + target.id


@pytest.mark.parametrize("wire_type", ("references", "_test_references"))
@pytest.mark.parametrize("sibling_prefix", (None, "Q:"))
def test_related_properties_prefix_matching_family_before_combining_ids(
    wire_type: str, sibling_prefix: str | None
) -> None:
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={
                CrossFamilyCitingFamily: CrossFamilyCitingRow,
                ReferenceFamily: ReferenceRow,
                TwoBackRefFamily: RefBackA,
            },
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        first = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/first", id="same")), eager=True)
        second = store.fetch(RefBackA, store.save(RefBackA("10.2/second", id="same")), eager=True)
        expected = {}
        for name, kwargs in (
            ("first", {"cite": first}),
            ("second", {"cite_b": second}),
            ("both", {"cite": first, "cite_b": second}),
        ):
            row = store.fetch(CrossFamilyCitingRow, store.save(CrossFamilyCitingRow(name, **kwargs)), eager=True)
            expected[name] = "W:" + row.id
        source = StoredEntrySource(store, CrossFamilyCitingFamily, "works", "W:")
        inventory = [source, StoredEntrySource(store, ReferenceFamily, "first", "P:")]
        if sibling_prefix is not None:
            inventory.append(StoredEntrySource(store, TwoBackRefFamily, "second", sibling_prefix))
        factory = related_property_resolver_factory(
            [stored_property_sql_plan(store, family) for family in (ReferenceFamily, TwoBackRefFamily)]
        )
        federation = StoredEntryFederation(
            (source,),
            source_inventory=inventory,
            served_type_names={"references": wire_type},
            related_resolver_factory=factory,
        )
        first_id, second_id = "P:same", (sibling_prefix or "") + "same"
        page = federation.query()
        relationships = dict(zip((row["id"] for row in page.rows), page.relationships, strict=True))
        assert {item.id for item in relationships[expected["both"]][wire_type]} == {first_id, second_id}
        for name, value, public_id in (("first", "10.1", first_id), ("second", "10.2", second_id)):
            ids = {expected[name], expected["both"]}
            for expression in (
                f'{wire_type}.id HAS "{public_id}"',
                f'{wire_type}.id = "{public_id}"',
                f'{wire_type}.id STARTS WITH "{public_id}"',
                f'{wire_type}.doi CONTAINS "{value}"',
            ):
                assert {row["id"] for row in federation.query(expression).rows} == ids
        raw_matches = {expected["second"], expected["both"]} if sibling_prefix is None else set()
        assert {row["id"] for row in federation.query(f'{wire_type}.id = "same"').rows} == raw_matches
        assert {row["id"] for row in federation.query(f'{wire_type}.id CONTAINS "same"').rows} == set(expected.values())
        assert first.immutable_id == second.immutable_id
        assert {
            row["id"] for row in federation.query(f'{wire_type}.immutable_id = "{first.immutable_id}"').rows
        } == set(expected.values())
        assert federation.query(f'{wire_type}.immutable_id = "P:{first.immutable_id}"').rows == ()
        assert [row["id"] for row in federation.query(f'{wire_type}.id HAS ALL "{first_id}","{second_id}"').rows] == [
            expected["both"]
        ]
        assert {row["id"] for row in federation.query(f'{wire_type}.doi CONTAINS "10."').rows} == set(expected.values())


def test_related_properties_exclude_unrelated_family_with_same_raw_id() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={CitingFamily: CitingRow, ReferenceFamily: ReferenceRow, TwoBackRefFamily: RefBackA},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        target = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/target", id="same")), eager=True)
        store.save(RefBackA("10.2/unrelated", id="same"))
        row = store.fetch(CitingRow, store.save(CitingRow("work", cite=target)), eager=True)
        source = StoredEntrySource(store, CitingFamily, "works", "W:")
        factory = related_property_resolver_factory(
            [stored_property_sql_plan(store, family) for family in (ReferenceFamily, TwoBackRefFamily)]
        )
        federation = StoredEntryFederation(
            (source,),
            source_inventory=(source, StoredEntrySource(store, ReferenceFamily, "target", "P:")),
            related_resolver_factory=factory,
        )
        assert [item["id"] for item in federation.query('references.doi CONTAINS "10.1"').rows] == ["W:" + row.id]
        assert federation.query('references.doi CONTAINS "10.2"').rows == ()
        resolver = factory(store)
        assert resolver is not None
        assert resolver("references", parse_optimade_filter('doi CONTAINS "10.2"')) == ("same",)
        assert resolver("references", parse_optimade_filter('id = "same"')) == ("same",)
        assert resolver("references", parse_optimade_filter('id = "P:same"')) == ()
