"""Stored forward/reverse serving of Run provenance (StrongLink) edges.

Covers the SQL entry provider and the durable federation: a run's own edges
served as forward semantic relationships (``_httk_has_*``) and the derived
reverse relationships (``_httk_is_*``) on the targeted data entries, plus the
review-pinned guards (alternatives excluded, lineage-level, store-scoped,
wire-named, latest-revision-only).
"""

import itertools
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
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

_REFERENCES = "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/references"


@dataclass(frozen=True)
class RecordRow:
    """A minimal ``records`` backing (wire type ``_httk_records``)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="prov_edge_record")

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class ReferenceRow:
    """A minimal standard ``references`` backing (wire type ``references``)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="prov_edge_reference")

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class RecordFamily:
    """The records family with the httk records definition (prefixed wire name)."""

    type = "records"
    definition_id = RECORDS_DEFINITION_ID


class ReferenceFamily:
    """The standard references family (unprefixed wire name)."""

    type = "references"
    definition_id = _REFERENCES


register_entry_family(name="prov-edge-records", family=f"{__name__}:RecordFamily", definition_id=RECORDS_DEFINITION_ID)
register_entry_record(name="prov-edge-records-rec", family="prov-edge-records", record=f"{__name__}:RecordRow")
register_entry_family(name="prov-edge-references", family=f"{__name__}:ReferenceFamily", definition_id=_REFERENCES)
register_entry_record(name="prov-edge-references-rec", family="prov-edge-references", record=f"{__name__}:ReferenceRow")


def _store(database: Backend) -> SqlStore:
    return SqlStore(
        database,
        entry_records={RunEntry: Run, RecordFamily: RecordRow, ReferenceFamily: ReferenceRow},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


def _save(store: SqlStore, record: object) -> str:
    return store.fetch(type(record), store.save(record), eager=True).id  # type: ignore[union-attr,attr-defined]


def _provider(store: SqlStore) -> StoreEntryProvider:
    return StoreEntryProvider(store, {"_httk_runs": Run, "_httk_records": RecordRow, "references": ReferenceRow})


def _federation(store: SqlStore, family: type, name: str) -> StoredEntryFederation:
    return StoredEntryFederation((StoredEntrySource(store, family, name),))


def _block(relationships: dict[str, tuple], key: str) -> list[tuple[str, str, str | None, str | None]]:
    """The ``(type, id, role, label)`` tuples of one served relationship key, sorted."""
    return sorted((e.entry_type, e.id, e.role, e.label) for e in relationships.get(key, ()))


def test_sql_provider_forward_three_groups_and_reverse() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        rec = _save(store, RecordRow("r"))
        ref = _save(store, ReferenceRow("s"))
        run = Run(
            inputs=(RunEdge("in-rec", "records", rec), RunEdge("in-ref", "references", ref)),
            artifacts=(RunEdge("art-rec", "records", rec),),
            outputs=(RunEdge("out-ref", "references", ref),),
            source_id="ws:job",
        )
        run_id = _save(store, run)
        provider = _provider(store)

        forward = dict(provider.relationships("_httk_runs"))[run_id]
        by_key: dict[str, list] = {}
        for entry in forward:
            by_key.setdefault(entry.relationship, []).append((entry.entry_type, entry.id, entry.role, entry.label))
        assert sorted(by_key["_httk_has_input"]) == sorted(
            [("_httk_records", rec, "input", "in-rec"), ("references", ref, "input", "in-ref")]
        )
        assert by_key["_httk_has_artifact"] == [("_httk_records", rec, "artifact", "art-rec")]
        assert by_key["_httk_has_output"] == [("references", ref, "output", "out-ref")]

        rec_rel = dict(provider.relationships("_httk_records"))[rec]
        ref_rel = dict(provider.relationships("references"))[ref]
        rec_by_key = {e.relationship: e for e in rec_rel}
        ref_by_key = {e.relationship: e for e in ref_rel}

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
        assert ref_by_key["_httk_is_input"].entry_type == "_httk_runs"
        assert ref_by_key["_httk_is_input"].label == "in-ref"
        assert ref_by_key["_httk_is_output"].label == "out-ref"


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
        run_id = _save(store, Run(inputs=(RunEdge("dangling", "references", "no-such-id"),), source_id="j"))
        forward = dict(_provider(store).relationships("_httk_runs"))[run_id]
        assert [(e.relationship, e.entry_type, e.id) for e in forward] == [
            ("_httk_has_input", "references", "no-such-id")
        ]


def test_prefixed_mounts_preserve_revision_and_alternative_edges() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        target = store.fetch(RecordRow, store.save(RecordRow("target")), eager=True)
        other = _save(store, RecordRow("other"))
        store.replace(target, RecordRow("target-v2"))
        store.save(RecordRow("target-alt"), alternative_of=target.id, alternative_kind="variant")
        run = store.fetch(Run, store.save(Run(inputs=(RunEdge("v1", "records", target.id),))), eager=True)
        store.replace(run, Run(inputs=(RunEdge("v2", "records", other),)))
        store.save(
            Run(inputs=(RunEdge("alt", "records", target.id),)), alternative_of=run.id, alternative_kind="variant"
        )
        sources = (
            StoredEntrySource(store, RunEntry, "runs", "R:"),
            StoredEntrySource(store, RecordFamily, "records", "D:"),
        )
        runs = StoredEntryFederation((sources[0],), source_inventory=sources)
        records = StoredEntryFederation((sources[1],), source_inventory=sources)
        forward = f'_httk_relationships._httk_has_input.id HAS "D:{target.id}"'
        assert not runs.query(forward).rows
        revisions = runs.query(forward, revisions=True)
        assert [row["id"] for row in revisions.rows] == ["R:" + run.immutable_id]
        assert revisions.relationships[0]["_httk_has_input"][0].id == "D:" + target.id
        alternatives = runs.query(forward, alternatives=True)
        assert [row["id"] for row in alternatives.rows] == ["R:" + run.id + "~variant"]
        assert alternatives.relationships[0]["_httk_has_input"][0].label == "alt"
        reverse = f'_httk_relationships._httk_is_input.id HAS "R:{run.id}"'
        assert [row["id"] for row in records.query(reverse).rows] == ["D:" + other]
        assert not records.query(reverse, alternatives=True).rows
        assert all(not rel for rel in records.query(alternatives=True).relationships)
        store.replace(
            store.fetch(Run, store.save(Run(inputs=(RunEdge("v2", "records", other),))), eager=True),
            Run(inputs=(RunEdge("v3", "records", target.id),)),
        )
        assert len(records.query(reverse, revisions=True).rows) == 2
        assert all(
            rel["_httk_is_input"][0].id == "R:" + run.id for rel in records.query(reverse, revisions=True).relationships
        )


def test_same_prefix_and_self_mount_resolution_with_custom_wire_names() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        target = _save(store, RecordRow("r"))
        run = _save(store, Run(inputs=(RunEdge("record", "records", target),)))
        sources = (
            StoredEntrySource(store, RunEntry, "run-a", "A:"),
            StoredEntrySource(store, RunEntry, "run-b", "B:"),
            StoredEntrySource(store, RecordFamily, "rec-a", "A:"),
            StoredEntrySource(store, RecordFamily, "rec-b", "B:"),
        )
        wire_names = {"records": "_custom_records", "runs": "_custom_runs"}
        runs = StoredEntryFederation(sources[:2], source_inventory=sources, served_type_names=wire_names)
        records = StoredEntryFederation(sources[2:], source_inventory=sources, served_type_names=wire_names)
        for row, rel in zip(runs.query().rows, runs.query().relationships, strict=True):
            edge = rel["_httk_has_input"][0]
            assert (edge.entry_type, edge.id) == ("_custom_records", row["id"][:2] + target)
        for row, rel in zip(records.query().rows, records.query().relationships, strict=True):
            edge = rel["_httk_is_input"][0]
            assert (edge.entry_type, edge.id) == ("_custom_runs", row["id"][:2] + run)
        assert [row["id"] for row in records.query(f'_httk_relationships._httk_is_input.id HAS "A:{run}"').rows] == [
            "A:" + target
        ]
        # Same-family selection is the source itself even when another mount exists.
        self_run = _save(store, Run(inputs=(RunEdge("prior", "runs", run),)))
        self_sources = sources[:2]
        for source in self_sources:
            federation = StoredEntryFederation((source,), source_inventory=self_sources)
            page = federation.query(f'id = "{source.public_id_prefix}{self_run}"')
            assert page.relationships[0]["_httk_has_input"][0].id == source.public_id_prefix + run


def test_reverse_relationship_union_of_run_mounts_needs_no_forward_target_override() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        target = _save(store, RecordRow("r"))
        run = _save(store, Run(inputs=(RunEdge("in", "records", target),)))
        sources = (
            StoredEntrySource(store, RunEntry, "run-a", "A:"),
            StoredEntrySource(store, RunEntry, "run-b", "B:"),
            StoredEntrySource(store, RecordFamily, "record", "D:"),
        )
        records = StoredEntryFederation((sources[2],), source_inventory=sources)
        edges = records.query().relationships[0]["_httk_is_input"]
        assert {edge.id for edge in edges} == {"A:" + run, "B:" + run}
        key = "_httk_relationships._httk_is_input.id"
        for operator in ("HAS ALL", "HAS ONLY"):
            assert [row["id"] for row in records.query(f'{key} {operator} "A:{run}", "B:{run}"').rows] == [
                "D:" + target
            ]
        assert not records.query(f'{key} HAS ONLY "A:{run}"').rows


_FILE_KIND = "file_0123456789abcdef"


@pytest.fixture(params=["sqlite", "duckdb"])
def backend(request: pytest.FixtureRequest):
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        database = Backend.duckdb()
    else:
        database = Backend.sqlite()
    with database:
        yield database


def _mounted(store: SqlStore) -> tuple[StoredEntryFederation, StoredEntryFederation]:
    sources = (
        StoredEntrySource(store, RunEntry, "runs", "R:"),
        StoredEntrySource(store, RecordFamily, "records", "D:"),
    )
    return (
        StoredEntryFederation((sources[0],), source_inventory=sources),
        StoredEntryFederation((sources[1],), source_inventory=sources),
    )


def test_edge_naming_alternative_serves_on_that_alternative_only(backend: Backend) -> None:
    """An ``<id>~<kind>`` edge: prefixed forward id, reverse on the alternative only, filters both ways."""
    store = _store(backend)
    target = _save(store, RecordRow("main"))
    store.save(RecordRow("as-read"), alternative_of=target, alternative_kind=_FILE_KIND)
    named = f"{target}~{_FILE_KIND}"
    run = _save(store, Run(inputs=(RunEdge("initial_structure", "records", named),), source_id="j"))
    runs, records = _mounted(store)

    forward = runs.query()
    assert _block(forward.relationships[0], "_httk_has_input") == [
        ("_httk_records", "D:" + named, "input", "initial_structure")
    ]
    has = '_httk_relationships._httk_has_input.id HAS "{}"'
    assert [row["id"] for row in runs.query(has.format("D:" + named)).rows] == ["R:" + run]
    assert not runs.query(has.format("D:" + target)).rows
    # Unmounted (no prefix): the raw composite id is served and filterable.
    plain = _federation(store, RunEntry, "runs")
    assert plain.query().relationships[0]["_httk_has_input"][0].id == named
    assert len(plain.query(has.format(named)).rows) == 1

    expected = [("_httk_runs", "R:" + run, "input", "initial_structure")]
    assert "_httk_is_input" not in records.query().relationships[0]
    alternatives = records.query(alternatives=True)
    assert [row["id"] for row in alternatives.rows] == ["D:" + named]
    assert _block(alternatives.relationships[0], "_httk_is_input") == expected
    # The served composite id resolves to the same alternative with the same block.
    found = records.fetch_alternative(*("D:" + named).rsplit("~", 1))
    assert found is not None and found[0]["id"] == "D:" + named
    assert _block(found[1], "_httk_is_input") == expected

    reverse = f'_httk_relationships._httk_is_input.id HAS "R:{run}"'
    assert [row["id"] for row in records.query(reverse, alternatives=True).rows] == ["D:" + named]
    assert not records.query(reverse).rows


def test_edge_naming_main_gives_cell_alternatives_no_reverse(backend: Backend) -> None:
    """The altermagnets shape: ``conventional``/``primitive`` cells are never named, so carry no reverse."""
    store = _store(backend)
    target = _save(store, RecordRow("main"))
    for kind in ("conventional", "primitive"):
        store.save(RecordRow(kind), alternative_of=target, alternative_kind=kind)
    run = _save(store, Run(inputs=(RunEdge("in", "records", target),), source_id="j"))
    runs, records = _mounted(store)

    assert runs.query().relationships[0]["_httk_has_input"][0].id == "D:" + target
    assert _block(records.query().relationships[0], "_httk_is_input") == [("_httk_runs", "R:" + run, "input", "in")]
    alternatives = records.query(alternatives=True)
    assert len(alternatives.rows) == 2
    assert all(not rel for rel in alternatives.relationships)
    reverse = f'_httk_relationships._httk_is_input.id HAS "R:{run}"'
    assert [row["id"] for row in records.query(reverse).rows] == ["D:" + target]
    assert not records.query(reverse, alternatives=True).rows


def test_forward_has_filters_agree_with_served_edges(backend: Backend) -> None:
    """Index-driven forward HAS/HAS ANY/HAS ALL (and per-run HAS ONLY) select exactly the served edges.

    Every edge family and value kind -- prefixed or bare main and alternative
    ids, a loose edge whose raw id carries the prefix, and unknown ids -- is
    checked against the relationships the same federation serves.
    """
    store = _store(backend)
    first, second = _save(store, RecordRow("first")), _save(store, RecordRow("second"))
    store.save(RecordRow("as-read"), alternative_of=first, alternative_kind=_FILE_KIND)
    named = f"{first}~{_FILE_KIND}"
    edges = (first, second, named, "D:loose", "nosuch")
    for index, (one, two) in enumerate(zip(edges, edges[1:] + edges[:1], strict=True)):
        store.save(
            Run(
                inputs=(RunEdge("i", "records", one),),
                outputs=(RunEdge("o1", "records", two), RunEdge("o2", "records", one)),
                artifacts=(RunEdge("a", "records", two),),
                source_id=f"run-{index}",
            )
        )
    store.save(Run(inputs=(RunEdge("i", "records", first),), source_id="dup"))
    store.save(Run(source_id="edgeless"))
    runs, _records = _mounted(store)
    values = [first, named, "D:" + first, "D:" + second, "D:" + named, "D:loose", "nosuch", "D:nosuch"]
    for federation in (runs, _federation(store, RunEntry, "runs")):
        page = federation.query()
        assert len(page.rows) == 7
        for key in ("_httk_has_input", "_httk_has_output", "_httk_has_artifact"):
            served = {
                row["id"]: {edge.id for edge in rel.get(key, ())}
                for row, rel in zip(page.rows, page.relationships, strict=True)
            }

            def selected(
                operator: str, chosen: list[str], key: str = key, federation: StoredEntryFederation = federation
            ) -> set[str]:
                listed = ", ".join(f'"{value}"' for value in chosen)
                return {row["id"] for row in federation.query(f"_httk_relationships.{key}.id {operator} {listed}").rows}

            for value in values:
                assert selected("HAS", [value]) == {run for run, ids in served.items() if value in ids}
            for pair in itertools.pairwise(values):
                assert selected("HAS ANY", list(pair)) == {run for run, ids in served.items() if ids & set(pair)}
                assert selected("HAS ALL", list(pair)) == {run for run, ids in served.items() if ids >= set(pair)}
                assert selected("HAS ONLY", list(pair)) == {run for run, ids in served.items() if ids <= set(pair)}


def test_upgrade_creates_edge_indexes_missing_from_an_older_store(tmp_path: Path) -> None:
    """Stores built before the StrongLink edge indexes open unchanged; ``upgrade=True`` adds them."""
    path = tmp_path / "store.sqlite"
    with Backend.sqlite(path) as database:
        target = _save(_store(database), RecordRow("r"))
        _save(_store(database), Run(inputs=(RunEdge("in", "records", target),)))
    indexes = {"ix_core_run_inputs_inputs_sid", "ix_core_run_edge_entry_id"}
    with sqlite3.connect(path) as connection:
        for name in indexes:
            connection.execute(f'DROP INDEX "{name}"')

    def present() -> set[str]:
        with sqlite3.connect(path) as connection:
            return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}

    with Backend.sqlite(path) as database:
        plain = _federation(_store(database), RunEntry, "runs")
        assert len(plain.query(f'_httk_relationships._httk_has_input.id HAS "{target}"').rows) == 1
    assert not indexes & present()
    with Backend.sqlite(path) as database:
        SqlStore(
            database,
            entry_records={RunEntry: Run, RecordFamily: RecordRow, ReferenceFamily: ReferenceRow},
            upgrade=True,
        )
    assert indexes <= present()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    pytest.main([__file__, "-q"])
