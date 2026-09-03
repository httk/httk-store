"""Stored forward/reverse serving of Run provenance (StrongLink) edges.

Covers the SQL entry provider and the durable federation: a run's own edges
served as forward semantic relationships (``_httk_has_*``) and the derived
reverse relationships (``_httk_is_*``) on the targeted data entries, plus the
review-pinned guards (alternatives excluded, lineage-level, store-scoped,
wire-named, latest-revision-only).
"""

from dataclasses import dataclass, field, replace
from typing import Annotated, ClassVar

import pytest
from httk.core import Run, RunEdge, RunEntry
from httk.core.data_records import RECORDS_DEFINITION_ID
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore, StoredEntrySource
from httk.store.backend.sql.entry_provider import StoreEntryProvider
from httk.store.backend.sql.stored_federation import StoredEntryFederation

_STRUCTURES = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"


@dataclass(frozen=True)
class RecordRow:
    """A minimal ``records`` backing (wire type ``_httk_records``)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="prov_edge_record")

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class StructureRow:
    """A minimal standard ``structures`` backing (wire type ``structures``)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="prov_edge_structure")

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class RecordFamily:
    """The records family with the httk records definition (prefixed wire name)."""

    type = "records"
    definition_id = RECORDS_DEFINITION_ID


class StructureFamily:
    """The standard structures family (unprefixed wire name)."""

    type = "structures"
    definition_id = _STRUCTURES


register_entry_family(name="prov-edge-records", family=f"{__name__}:RecordFamily", definition_id=RECORDS_DEFINITION_ID)
register_entry_record(name="prov-edge-records-rec", family="prov-edge-records", record=f"{__name__}:RecordRow")
register_entry_family(name="prov-edge-structures", family=f"{__name__}:StructureFamily", definition_id=_STRUCTURES)
register_entry_record(name="prov-edge-structures-rec", family="prov-edge-structures", record=f"{__name__}:StructureRow")


def _store(database: Backend) -> SqlStore:
    return SqlStore(
        database,
        entry_records={RunEntry: Run, RecordFamily: RecordRow, StructureFamily: StructureRow},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


def _save(store: SqlStore, record: object) -> str:
    return store.fetch(type(record), store.save(record), eager=True).id  # type: ignore[union-attr,attr-defined]


def _provider(store: SqlStore) -> StoreEntryProvider:
    return StoreEntryProvider(store, {"_httk_runs": Run, "_httk_records": RecordRow, "structures": StructureRow})


def _federation(store: SqlStore, family: type, name: str) -> StoredEntryFederation:
    return StoredEntryFederation((StoredEntrySource(store, family, name),))


def _block(relationships: dict[str, tuple], key: str) -> list[tuple[str, str, str | None, str | None]]:
    """The ``(type, id, role, label)`` tuples of one served relationship key, sorted."""
    return sorted((e.entry_type, e.id, e.role, e.label) for e in relationships.get(key, ()))


def test_sql_provider_forward_three_groups_and_reverse() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        rec = _save(store, RecordRow("r"))
        struct = _save(store, StructureRow("s"))
        run = Run(
            inputs=(RunEdge("in-rec", "records", rec), RunEdge("in-str", "structures", struct)),
            artifacts=(RunEdge("art-rec", "records", rec),),
            outputs=(RunEdge("out-str", "structures", struct),),
            source_id="ws:job",
        )
        run_id = _save(store, run)
        provider = _provider(store)

        forward = dict(provider.relationships("_httk_runs"))[run_id]
        by_key: dict[str, list] = {}
        for entry in forward:
            by_key.setdefault(entry.relationship, []).append((entry.entry_type, entry.id, entry.role, entry.label))
        assert sorted(by_key["_httk_has_input"]) == sorted(
            [("_httk_records", rec, "input", "in-rec"), ("structures", struct, "input", "in-str")]
        )
        assert by_key["_httk_has_artifact"] == [("_httk_records", rec, "artifact", "art-rec")]
        assert by_key["_httk_has_output"] == [("structures", struct, "output", "out-str")]

        rec_rel = dict(provider.relationships("_httk_records"))[rec]
        struct_rel = dict(provider.relationships("structures"))[struct]
        rec_by_key = {e.relationship: e for e in rec_rel}
        struct_by_key = {e.relationship: e for e in struct_rel}

        # (c) reverse on the prefixed _httk_records target is wire-named both the
        # relationship key and the identifier type; role/label match the forward.
        assert rec_by_key["_httk_is_input"] == replace(
            rec_by_key["_httk_is_input"],
            entry_type="_httk_runs",
            id=run_id,
            role="input",
            label="in-rec",
            relationship="_httk_is_input",
        )
        assert rec_by_key["_httk_is_artifact"].entry_type == "_httk_runs"
        assert rec_by_key["_httk_is_artifact"].id == run_id
        assert rec_by_key["_httk_is_artifact"].label == "art-rec"
        assert struct_by_key["_httk_is_input"].entry_type == "_httk_runs"
        assert struct_by_key["_httk_is_input"].label == "in-str"
        assert struct_by_key["_httk_is_output"].label == "out-str"


def test_sql_federation_forward_and_reverse_pages() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        rec_in = _save(store, RecordRow("in"))
        rec_art = _save(store, RecordRow("art"))
        run_id = _save(
            store,
            Run(
                inputs=(RunEdge("in-rec", "records", rec_in),),
                artifacts=(RunEdge("art-rec", "records", rec_art),),
                source_id="ws:job",
            ),
        )

        runs = _federation(store, RunEntry, "runs").query()
        (run_rel,) = runs.relationships
        assert _block(dict(run_rel), "_httk_has_input") == [("_httk_records", rec_in, "input", "in-rec")]
        assert _block(dict(run_rel), "_httk_has_artifact") == [("_httk_records", rec_art, "artifact", "art-rec")]

        records = _federation(store, RecordFamily, "recs").query(sort=(("id", False),))
        rel_by_id = dict(zip((r["id"] for r in records.rows), records.relationships, strict=True))
        assert _block(dict(rel_by_id[rec_in]), "_httk_is_input") == [("_httk_runs", run_id, "input", "in-rec")]
        assert _block(dict(rel_by_id[rec_art]), "_httk_is_artifact") == [("_httk_runs", run_id, "artifact", "art-rec")]


def test_reverse_excludes_run_alternative() -> None:
    """(a) A target referenced only by a run ALTERNATIVE gets no reverse edge."""
    with Backend.sqlite() as database:
        store = _store(database)
        main_target = _save(store, RecordRow("main-target"))
        alt_target = _save(store, RecordRow("alt-target"))
        main_run = store.fetch(
            Run, store.save(Run(inputs=(RunEdge("main-in", "records", main_target),), source_id="m")), eager=True
        )
        store.save(
            Run(inputs=(RunEdge("alt-in", "records", alt_target),), source_id="a"),
            alternative_of=main_run.id,
            alternative_kind="variant",
        )
        relationships = dict(_provider(store).relationships("_httk_records"))
        assert "_httk_is_input" in {e.relationship for e in relationships.get(main_target, ())}
        assert alt_target not in relationships  # only the alt run references it


def test_reverse_on_revisions_not_alternatives() -> None:
    """(b) EVERY ~revs row of a lineage carries the reverse block; ~alts carries none."""
    with Backend.sqlite() as database:
        store = _store(database)
        target = store.fetch(RecordRow, store.save(RecordRow("t")), eager=True)
        # A second revision of the SAME lineage: both revisions share the raw
        # entry_id the reverse lookup matches, so both ~revs rows must carry it.
        store.replace(target, RecordRow("t2"))
        store.save(RecordRow("t-conv"), alternative_of=target.id, alternative_kind="conventional")
        run_id = _save(store, Run(inputs=(RunEdge("in", "records", target.id),), source_id="j"))
        federation = _federation(store, RecordFamily, "recs")

        revisions = federation.query(revisions=True)
        # Both immutable revisions of the target lineage are on the page; each
        # one carries the exact same reverse block naming the referencing run.
        rev_rows = [
            (row, dict(rel))
            for row, rel in zip(revisions.rows, revisions.relationships, strict=True)
            if row["_httk_id"] == target.id
        ]
        assert len(rev_rows) == 2
        for _row, rel in rev_rows:
            assert _block(rel, "_httk_is_input") == [("_httk_runs", run_id, "input", "in")]

        alternatives = federation.query(alternatives=True)
        assert alternatives.rows  # the conventional alternative is on this page
        assert all(dict(rel) == {} for rel in alternatives.relationships)


def test_reverse_is_store_scoped() -> None:
    """(d) With runs only in store A, entries served from store B get no reverse."""
    with Backend.sqlite() as database_a, Backend.sqlite() as database_b:
        store_a = _store(database_a)
        store_b = _store(database_b)
        rec_a = _save(store_a, RecordRow("in-a"))
        rec_b = _save(store_b, RecordRow("in-b"))
        run_id = _save(store_a, Run(inputs=(RunEdge("in", "records", rec_a),), source_id="j"))
        federation = StoredEntryFederation(
            (
                StoredEntrySource(store_a, RecordFamily, "a", "a-"),
                StoredEntrySource(store_b, RecordFamily, "b", "b-"),
            )
        )
        page = federation.query(sort=(("id", False),))
        by_id = {row["id"]: dict(rel) for row, rel in zip(page.rows, page.relationships, strict=True)}
        assert _block(by_id["a-" + rec_a], "_httk_is_input") == [("_httk_runs", run_id, "input", "in")]
        assert by_id["b-" + rec_b] == {}  # store B has no run family to scan


def test_reverse_reflects_latest_run_revision_only() -> None:
    """(e) Superseding a run drops the superseded edge's reverse identifier."""
    with Backend.sqlite() as database:
        store = _store(database)
        old_target = _save(store, RecordRow("old"))
        new_target = _save(store, RecordRow("new"))
        run_v1 = store.fetch(
            Run, store.save(Run(inputs=(RunEdge("in", "records", old_target),), source_id="j")), eager=True
        )
        store.replace(run_v1, Run(inputs=(RunEdge("in", "records", new_target),), source_id="j"))
        relationships = dict(_provider(store).relationships("_httk_records"))
        assert old_target not in relationships  # the superseded edge no longer contributes
        assert "_httk_is_input" in {e.relationship for e in relationships[new_target]}


def test_reverse_order_is_deterministic_run_then_marker_then_edge() -> None:
    """Reverse hits order by (run raw id, field/marker order, edge row order)."""
    with Backend.sqlite() as database:
        store = _store(database)
        rec = _save(store, RecordRow("shared"))
        # Run A points at the record from both its input and its artifact field;
        # Run B (saved later, higher id) points at it from its input field.
        run_a = _save(
            store,
            Run(
                inputs=(RunEdge("a-in", "records", rec),),
                artifacts=(RunEdge("a-art", "records", rec),),
                source_id="a",
            ),
        )
        run_b = _save(store, Run(inputs=(RunEdge("b-in", "records", rec),), source_id="b"))
        assert run_a < run_b  # ascending run-id order is the outer sort key

        reverse = dict(_provider(store).relationships("_httk_records"))[rec]
        assert [(e.relationship, e.id, e.label) for e in reverse] == [
            ("_httk_is_input", run_a, "a-in"),
            ("_httk_is_artifact", run_a, "a-art"),
            ("_httk_is_input", run_b, "b-in"),
        ]


def test_forward_only_run_has_no_hydrated_target_dependency() -> None:
    """A run whose edges point at unserved ids still serves its forward edges."""
    with Backend.sqlite() as database:
        store = _store(database)
        run_id = _save(store, Run(inputs=(RunEdge("dangling", "structures", "no-such-id"),), source_id="j"))
        forward = dict(_provider(store).relationships("_httk_runs"))[run_id]
        assert [(e.relationship, e.entry_type, e.id) for e in forward] == [
            ("_httk_has_input", "structures", "no-such-id")
        ]


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    pytest.main([__file__, "-q"])
