"""Stored-route relationship filtering (the ``_httk_relationships.<key>.id`` extension).

Exercises the durable federation's SQL filter path built in
``backend/sql/stored_properties.py``: StrongLink forward/reverse semantic keys,
typed reference and weak-link keys under both the bare and the
``_httk_relationships.<type>.id`` alias spellings, the HAS family (incl. the S4
HAS ALL fix and vacuous HAS ONLY), the ``~alts`` reverse suppression, and the
per-revision ``~revs`` forward pin. Every filter asserts the exact expected id
set, and every 400 is paired with a positive same-route filter.
"""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
from httk.core import Run, RunEdge, RunEntry
from httk.core.data_records import RECORDS_DEFINITION_ID
from httk.core.provenance import RUNS_DEFINITION_ID
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import (
    IdentitySkip,
    Indexed,
    StorageInfo,
    StoredPropertyProjection,
    StrongLink,
    Unique,
    WeakLink,
)

from httk.store import EntryIdScheme, FilterTranslationError
from httk.store.backend.sql import Backend, SqlStore, StoredEntrySource, stored_property_sql_plan
from httk.store.backend.sql.stored_federation import StoredEntryFederation, related_property_resolver_factory

_STRUCTURES = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
_REFERENCES = "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/references"


@dataclass(frozen=True)
class RecordRow:
    """A minimal ``records`` backing (wire type ``_httk_records``)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_record")

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class ReferenceRow:
    """A minimal ``references`` backing (the target of a typed reference field).

    ``doi`` is projected as a queryable/served property so the depth-1
    related-property resolver has a real reference-side field to filter on.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_reference")

    doi: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = {
        "doi": StoredPropertyProjection(
            response=lambda record: record.doi,
            query=lambda context, operator, literal: context.compare(
                context.field("doi"), operator, context.constant(literal)
            ),
            sort=lambda context: context.field("doi"),
        )
    }


@dataclass(frozen=True)
class CitingRow:
    """A ``records`` backing carrying a typed reference field to a ``references`` row."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_citing")

    name: str
    cite: ReferenceRow | None = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class WeakRow:
    """A ``records`` backing exposing one weak link to the prefixed ``Run`` family."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="rel_filter_weak",
        links=(WeakLink("produced_by", Run, exposed_relationship=True, role="artifact+output"),),
    )

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class RecordFamily:
    type = "records"
    definition_id = RECORDS_DEFINITION_ID


class ReferenceFamily:
    type = "references"
    definition_id = _REFERENCES


class CitingFamily:
    type = "records"
    definition_id = RECORDS_DEFINITION_ID


class WeakFamily:
    type = "records"
    definition_id = RECORDS_DEFINITION_ID


register_entry_family(name="rel-filter-records", family=f"{__name__}:RecordFamily", definition_id=RECORDS_DEFINITION_ID)
register_entry_record(name="rel-filter-records-rec", family="rel-filter-records", record=f"{__name__}:RecordRow")
register_entry_family(name="rel-filter-references", family=f"{__name__}:ReferenceFamily", definition_id=_REFERENCES)
register_entry_record(
    name="rel-filter-references-rec", family="rel-filter-references", record=f"{__name__}:ReferenceRow"
)
register_entry_family(name="rel-filter-citing", family=f"{__name__}:CitingFamily", definition_id=RECORDS_DEFINITION_ID)
register_entry_record(name="rel-filter-citing-rec", family="rel-filter-citing", record=f"{__name__}:CitingRow")
register_entry_family(name="rel-filter-weak", family=f"{__name__}:WeakFamily", definition_id=RECORDS_DEFINITION_ID)
register_entry_record(name="rel-filter-weak-rec", family="rel-filter-weak", record=f"{__name__}:WeakRow")


def _store(database: Backend) -> SqlStore:
    return SqlStore(
        database,
        entry_records={
            RunEntry: Run,
            RecordFamily: RecordRow,
            ReferenceFamily: ReferenceRow,
            CitingFamily: CitingRow,
            WeakFamily: WeakRow,
        },
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


def _save(store: SqlStore, record: object) -> str:
    return store.fetch(type(record), store.save(record), eager=True).id  # type: ignore[union-attr,attr-defined]


def _ids(page: object) -> list[str]:
    return sorted(row["id"] for row in page.rows)  # type: ignore[attr-defined]


def _query(store: SqlStore, family: type, filter_string: str, **kwargs: object) -> object:
    federation = StoredEntryFederation((StoredEntrySource(store, family, "src"),))
    return federation.query(filter_string, **kwargs)


def _resolved_query(store: SqlStore, family: type, filter_string: str, **kwargs: object) -> object:
    """Query with a real depth-1 related-property resolver over the reference target."""
    factory = related_property_resolver_factory([stored_property_sql_plan(store, ReferenceFamily)])
    federation = StoredEntryFederation((StoredEntrySource(store, family, "src"),), related_resolver_factory=factory)
    return federation.query(filter_string, **kwargs)


# ---------------------------------------------------------------------- StrongLink forward


def test_forward_key_filters_runs_has_family() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        rec_a = _save(store, RecordRow("a"))
        rec_b = _save(store, RecordRow("b"))
        run_ab = _save(
            store, Run(inputs=(RunEdge("ia", "records", rec_a), RunEdge("ib", "records", rec_b)), source_id="ab")
        )
        run_a = _save(store, Run(inputs=(RunEdge("ia", "records", rec_a),), source_id="a"))
        run_b = _save(store, Run(inputs=(RunEdge("ib", "records", rec_b),), source_id="b"))

        # HAS: every run whose inputs reference rec_a.
        assert _ids(_query(store, RunEntry, f'_httk_relationships._httk_has_input.id HAS "{rec_a}"')) == sorted(
            [run_ab, run_a]
        )
        # HAS ANY over both targets: any run referencing either.
        assert _ids(
            _query(store, RunEntry, f'_httk_relationships._httk_has_input.id HAS ANY "{rec_a}","{rec_b}"')
        ) == sorted([run_ab, run_a, run_b])
        # HAS ALL over both targets: only the run referencing both.
        assert _ids(_query(store, RunEntry, f'_httk_relationships._httk_has_input.id HAS ALL "{rec_a}","{rec_b}"')) == [
            run_ab
        ]


def test_forward_has_only_vacuous_for_edgeless_run() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        rec_a = _save(store, RecordRow("a"))
        rec_b = _save(store, RecordRow("b"))
        run_a = _save(store, Run(inputs=(RunEdge("ia", "records", rec_a),), source_id="a"))
        _save(store, Run(inputs=(RunEdge("ib", "records", rec_b),), source_id="b"))
        edgeless = _save(store, Run(source_id="none"))
        # HAS ONLY rec_a: the run whose only input is rec_a, plus the edge-less
        # run (vacuously true), never the run referencing rec_b.
        assert _ids(_query(store, RunEntry, f'_httk_relationships._httk_has_input.id HAS ONLY "{rec_a}"')) == sorted(
            [run_a, edgeless]
        )


def test_two_revision_revs_forward_pins_each_revision() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        rec_a = _save(store, RecordRow("a"))
        rec_b = _save(store, RecordRow("b"))
        run1 = store.fetch(Run, store.save(Run(inputs=(RunEdge("i", "records", rec_a),), source_id="r1")), eager=True)
        rev1 = run1.immutable_id
        run2 = store.fetch(
            Run, store.replace(run1, Run(inputs=(RunEdge("i", "records", rec_b),), source_id="r2")), eager=True
        )
        rev2 = run2.immutable_id
        assert rev1 != rev2

        # Each revision evaluates its OWN edges: rev1 references rec_a, rev2 rec_b.
        rev_a = _query(store, RunEntry, f'_httk_relationships._httk_has_input.id HAS "{rec_a}"', revisions=True)
        assert _ids(rev_a) == [rev1]
        rev_b = _query(store, RunEntry, f'_httk_relationships._httk_has_input.id HAS "{rec_b}"', revisions=True)
        assert _ids(rev_b) == [rev2]


# ---------------------------------------------------------------------- StrongLink reverse


def test_reverse_key_filters_data_entries() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        rec_in = _save(store, RecordRow("in"))
        rec_art = _save(store, RecordRow("art"))
        rec_none = _save(store, RecordRow("none"))
        run_id = _save(
            store,
            Run(
                inputs=(RunEdge("in", "records", rec_in),),
                artifacts=(RunEdge("art", "records", rec_art),),
                source_id="j",
            ),
        )
        # Reverse input key: the record the run took as input.
        assert _ids(_query(store, RecordFamily, f'_httk_relationships._httk_is_input.id HAS "{run_id}"')) == [rec_in]
        # Reverse artifact key: the record the run produced as artifact.
        assert _ids(_query(store, RecordFamily, f'_httk_relationships._httk_is_artifact.id HAS "{run_id}"')) == [
            rec_art
        ]
        # An unreferenced record matches neither; HAS ONLY is vacuously true for it.
        only = _query(store, RecordFamily, f'_httk_relationships._httk_is_input.id HAS ONLY "{run_id}"')
        assert rec_none in _ids(only) and rec_art in _ids(only) and rec_in in _ids(only)


def test_reverse_excludes_non_latest_and_alternative_runs() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        target = _save(store, RecordRow("t"))
        main_run = store.fetch(
            Run, store.save(Run(inputs=(RunEdge("in", "records", target),), source_id="m")), eager=True
        )
        run_id = main_run.id
        # A newer main revision of the run that DROPS the edge: the reverse must vanish.
        store.replace(main_run, Run(source_id="m2"))
        assert _ids(_query(store, RecordFamily, f'_httk_relationships._httk_is_input.id HAS "{run_id}"')) == []


def test_reverse_suppressed_on_alternatives_with_group_targeted_edge() -> None:
    """~alts reverse no-match even when a run edge targets the alternative's group id."""
    with Backend.sqlite() as database:
        store = _store(database)
        target = store.fetch(RecordRow, store.save(RecordRow("t")), eager=True)
        # An alternative of the SAME lineage: its raw id column carries the group
        # id, which the run below targets -- the suppression, not chance, is what
        # keeps it out of the ~alts result.
        store.save(RecordRow("t-conv"), alternative_of=target.id, alternative_kind="conventional")
        run_id = _save(store, Run(inputs=(RunEdge("in", "records", target.id),), source_id="j"))

        # Positive control on the mains page: the reverse filter matches the main.
        assert _ids(_query(store, RecordFamily, f'_httk_relationships._httk_is_input.id HAS "{run_id}"')) == [target.id]
        # The alternatives page suppresses reverse keys: no alt row matches.
        alts = _query(store, RecordFamily, f'_httk_relationships._httk_is_input.id HAS "{run_id}"', alternatives=True)
        assert _ids(alts) == []


# ---------------------------------------------------------------------- typed reference (conformance fix)


def test_bare_reference_id_filters_and_equals_alias() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ref_a = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/a")), eager=True)
        ref_b = store.fetch(ReferenceRow, store.save(ReferenceRow("10.2/b")), eager=True)
        cite_a = _save(store, CitingRow("cites-a", cite=ref_a))
        _save(store, CitingRow("cites-b", cite=ref_b))
        _save(store, CitingRow("cites-none"))

        # The stored route's bare `references.id HAS` now really filters
        # (previously matched nothing) ...
        bare = _query(store, CitingFamily, f'references.id HAS "{ref_a.id}"')
        assert _ids(bare) == [cite_a]
        # ... and the `_httk_relationships.references.id` alias gives the same set.
        alias = _query(store, CitingFamily, f'_httk_relationships.references.id HAS "{ref_a.id}"')
        assert _ids(alias) == [cite_a]


def test_reference_has_only_vacuous_for_unmatched_id() -> None:
    """HAS ONLY ids matching no target row still keeps the referent-less rows (vacuous truth)."""
    with Backend.sqlite() as database:
        store = _store(database)
        ref_a = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/a")), eager=True)
        cite_a = _save(store, CitingRow("cites-a", cite=ref_a))
        cite_none = _save(store, CitingRow("cites-none"))

        assert _ids(_query(store, CitingFamily, f'references.id HAS ONLY "{ref_a.id}"')) == sorted([cite_a, cite_none])
        # An id no reference row carries: the empty allowed set must not turn the
        # referent-less row into an outsider (``NULL NOT IN ()`` is SQL TRUE).
        assert _ids(_query(store, CitingFamily, 'references.id HAS ONLY "nonesuch"')) == [cite_none]


# ---------------------------------------------------------------------- weak link both spellings + HAS ALL


def test_weak_link_both_spellings_and_has_all() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        run1 = store.fetch(Run, store.save(Run(source_id="t1")), eager=True)
        run2 = store.fetch(Run, store.save(Run(source_id="t2")), eager=True)
        both = store.fetch(WeakRow, store.save(WeakRow("both")), eager=True)
        one = store.fetch(WeakRow, store.save(WeakRow("one")), eager=True)
        store.link(both, "produced_by", run1)
        store.link(both, "produced_by", run2)
        store.link(one, "produced_by", run1)

        # Bare weak-link spelling and the alias agree.
        bare = _query(store, WeakFamily, f'_httk_runs.id HAS "{run1.id}"')
        alias = _query(store, WeakFamily, f'_httk_relationships._httk_runs.id HAS "{run1.id}"')
        assert _ids(bare) == sorted([both.id, one.id])
        assert _ids(alias) == _ids(bare)

        # HAS ALL through BOTH spellings: only the row linked to every run (guards S4).
        bare_all = _query(store, WeakFamily, f'_httk_runs.id HAS ALL "{run1.id}","{run2.id}"')
        alias_all = _query(store, WeakFamily, f'_httk_relationships._httk_runs.id HAS ALL "{run1.id}","{run2.id}"')
        assert _ids(bare_all) == [both.id]
        assert _ids(alias_all) == [both.id]


# ---------------------------------------------------------------------- dispatch: unknown key 400 (+ positive)


def test_unknown_relationship_key_400_paired_with_positive() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        rec = _save(store, RecordRow("a"))
        run_id = _save(store, Run(inputs=(RunEdge("i", "records", rec),), source_id="j"))

        # Positive same-route filter (a declared key with data).
        assert _ids(_query(store, RunEntry, f'_httk_relationships._httk_has_input.id HAS "{rec}"')) == [run_id]

        # Unknown key under our own prefix: 400 naming the full dotted path.
        with pytest.raises(FilterTranslationError) as excinfo:
            _query(store, RunEntry, '_httk_relationships.bogus.id HAS "x"')
        assert excinfo.value.category == "unrecognized-property"
        assert "_httk_relationships.bogus.id" in str(excinfo.value)


def test_declared_key_without_data_filters_to_empty_not_400() -> None:
    """Schema-derived inventory: a declared key with no matching data is empty, never 400."""
    with Backend.sqlite() as database:
        store = _store(database)
        _save(store, RecordRow("a"))
        _save(store, Run(source_id="edgeless"))
        # No run references this id: an empty result, not an error.
        assert _ids(_query(store, RecordFamily, '_httk_relationships._httk_is_input.id HAS "nonesuch"')) == []


# ---------------------------------------------------------------------- reverse across two run backings


@dataclass(frozen=True)
class RunBackingA:
    """One backing of a two-backing run family, declaring the StrongLink input edge."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_run_a")

    inputs: Annotated[tuple[RunEdge, ...], StrongLink("has_input", reverse="is_input", role="input")] = ()
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class RunBackingB:
    """A second backing of the same run family (a separate table, same edge field)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_run_b")

    inputs: Annotated[tuple[RunEdge, ...], StrongLink("has_input", reverse="is_input", role="input")] = ()
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class TwoBackingRunFamily:
    type = "runs"
    definition_id = RUNS_DEFINITION_ID


register_entry_family(
    name="rel-filter-two-run", family=f"{__name__}:TwoBackingRunFamily", definition_id=RUNS_DEFINITION_ID
)
register_entry_record(name="rel-filter-two-run-a", family="rel-filter-two-run", record=f"{__name__}:RunBackingA")
register_entry_record(name="rel-filter-two-run-b", family="rel-filter-two-run", record=f"{__name__}:RunBackingB")


def test_reverse_key_ors_across_run_backings() -> None:
    """A multi-backing run family: the reverse filter matches runs from EVERY backing.

    ``strong_link_families`` yields one entry per backing; serving ORs a
    target's reverse edges across all of them, so the filter must too (else runs
    in the second backing are served but not filterable).
    """
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={TwoBackingRunFamily: (RunBackingA, RunBackingB), RecordFamily: RecordRow},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        target = _save(store, RecordRow("t"))
        run_a = store.fetch(
            RunBackingA, store.save(RunBackingA(inputs=(RunEdge("ia", "records", target),))), eager=True
        ).id
        run_b = store.fetch(
            RunBackingB, store.save(RunBackingB(inputs=(RunEdge("ib", "records", target),))), eager=True
        ).id

        federation = StoredEntryFederation((StoredEntrySource(store, RecordFamily, "recs"),))

        # Serving names the runs from BOTH backings on the target's reverse block.
        page = federation.query()
        (rel,) = page.relationships
        served_run_ids = {entry.id for entry in dict(rel).get("_httk_is_input", ())}
        assert served_run_ids == {run_a, run_b}

        # Filtering agrees: a match on EITHER backing's run id, and on either alone.
        assert _ids(federation.query(f'_httk_relationships._httk_is_input.id HAS ANY "{run_a}","{run_b}"')) == [target]
        assert _ids(federation.query(f'_httk_relationships._httk_is_input.id HAS "{run_a}"')) == [target]
        assert _ids(federation.query(f'_httk_relationships._httk_is_input.id HAS "{run_b}"')) == [target]


# ------------------------------------------------- two typed reference fields to one served family


_DOI_PROJECTION = {
    "doi": StoredPropertyProjection(
        response=lambda record: record.doi,
        query=lambda context, operator, literal: context.compare(
            context.field("doi"), operator, context.constant(literal)
        ),
        sort=lambda context: context.field("doi"),
    )
}


@dataclass(frozen=True)
class RefBackA:
    """One backing of a two-backing ``references`` family."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_ref_a")

    doi: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = _DOI_PROJECTION


@dataclass(frozen=True)
class RefBackB:
    """The second backing of the same ``references`` family (a separate table)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_ref_b")

    doi: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = _DOI_PROJECTION


@dataclass(frozen=True)
class DoubleCitingRow:
    """A ``records`` backing referencing BOTH backings of the two-backing family."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_double")

    name: str
    cite_a: RefBackA | None = None
    cite_b: RefBackB | None = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class SwappedCitingRow:
    """The same two reference fields declared in the opposite order (order-insensitivity)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="rel_filter_swapped")

    name: str
    cite_b: RefBackB | None = None
    cite_a: RefBackA | None = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class TwoBackRefFamily:
    type = "references"
    definition_id = _REFERENCES


class DoubleCitingFamily:
    type = "records"
    definition_id = RECORDS_DEFINITION_ID


class SwappedCitingFamily:
    type = "records"
    definition_id = RECORDS_DEFINITION_ID


register_entry_family(name="rel-filter-two-ref", family=f"{__name__}:TwoBackRefFamily", definition_id=_REFERENCES)
register_entry_record(name="rel-filter-two-ref-a", family="rel-filter-two-ref", record=f"{__name__}:RefBackA")
register_entry_record(name="rel-filter-two-ref-b", family="rel-filter-two-ref", record=f"{__name__}:RefBackB")
register_entry_family(
    name="rel-filter-double", family=f"{__name__}:DoubleCitingFamily", definition_id=RECORDS_DEFINITION_ID
)
register_entry_record(name="rel-filter-double-rec", family="rel-filter-double", record=f"{__name__}:DoubleCitingRow")
register_entry_family(
    name="rel-filter-swapped", family=f"{__name__}:SwappedCitingFamily", definition_id=RECORDS_DEFINITION_ID
)
register_entry_record(name="rel-filter-swapped-rec", family="rel-filter-swapped", record=f"{__name__}:SwappedCitingRow")

_DOUBLE_CASES = [(DoubleCitingRow, DoubleCitingFamily), (SwappedCitingRow, SwappedCitingFamily)]


def _double_store(database: Backend, row_cls: type, family: type) -> SqlStore:
    return SqlStore(
        database,
        entry_records={family: row_cls, TwoBackRefFamily: (RefBackA, RefBackB)},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


@pytest.mark.parametrize("row_cls,family", _DOUBLE_CASES)
def test_two_reference_fields_union_id_has_family(row_cls: type, family: type) -> None:
    """Two reference fields serving under ONE wire type filter over the UNION of their targets.

    Registering only the first-declared field's handler made the second field's
    targets silently unmatchable; the results must not depend on declaration
    order (both field orders are parameterized).
    """
    with Backend.sqlite() as database:
        store = _double_store(database, row_cls, family)
        ref_a = store.fetch(RefBackA, store.save(RefBackA("10.1/a")), eager=True)
        ref_b = store.fetch(RefBackB, store.save(RefBackB("10.2/b")), eager=True)
        both = _save(store, row_cls("both", cite_a=ref_a, cite_b=ref_b))
        only_a = _save(store, row_cls("only-a", cite_a=ref_a))
        only_b = _save(store, row_cls("only-b", cite_b=ref_b))
        neither = _save(store, row_cls("neither"))

        # HAS through the first field and through the second one alike ...
        assert _ids(_query(store, family, f'references.id HAS "{ref_a.id}"')) == sorted([both, only_a])
        assert _ids(_query(store, family, f'references.id HAS "{ref_b.id}"')) == sorted([both, only_b])
        # ... under both spellings.
        assert _ids(_query(store, family, f'_httk_relationships.references.id HAS "{ref_a.id}"')) == sorted(
            [both, only_a]
        )
        assert _ids(_query(store, family, f'_httk_relationships.references.id HAS "{ref_b.id}"')) == sorted(
            [both, only_b]
        )

        # HAS ANY spans both fields.
        assert _ids(_query(store, family, f'references.id HAS ANY "{ref_a.id}","{ref_b.id}"')) == sorted(
            [both, only_a, only_b]
        )
        # HAS ALL with one id from each field: only the row carrying both edges.
        assert _ids(_query(store, family, f'references.id HAS ALL "{ref_a.id}","{ref_b.id}"')) == [both]
        # HAS ONLY the exact union holds for every row (the edge-less one vacuously) ...
        assert _ids(_query(store, family, f'references.id HAS ONLY "{ref_a.id}","{ref_b.id}"')) == sorted(
            [both, only_a, only_b, neither]
        )
        # ... a strict subset excludes every row reaching the omitted target.
        assert _ids(_query(store, family, f'references.id HAS ONLY "{ref_a.id}"')) == sorted([only_a, neither])


@pytest.mark.parametrize("row_cls,family", _DOUBLE_CASES)
def test_two_reference_fields_union_depth1_property(row_cls: type, family: type) -> None:
    """Depth-1 ``<type>.<prop>`` resolves through EITHER contributing field."""
    with Backend.sqlite() as database:
        store = _double_store(database, row_cls, family)
        ref_a = store.fetch(RefBackA, store.save(RefBackA("10.1/a")), eager=True)
        ref_b = store.fetch(RefBackB, store.save(RefBackB("10.2/b")), eager=True)
        both = _save(store, row_cls("both", cite_a=ref_a, cite_b=ref_b))
        only_a = _save(store, row_cls("only-a", cite_a=ref_a))
        only_b = _save(store, row_cls("only-b", cite_b=ref_b))
        _save(store, row_cls("neither"))

        factory = related_property_resolver_factory([stored_property_sql_plan(store, TwoBackRefFamily)])
        federation = StoredEntryFederation((StoredEntrySource(store, family, "src"),), related_resolver_factory=factory)

        assert _ids(federation.query('references.doi = "10.1/a"')) == sorted([both, only_a])
        assert _ids(federation.query('references.doi = "10.2/b"')) == sorted([both, only_b])
        assert _ids(federation.query('references.doi CONTAINS "10."')) == sorted([both, only_a, only_b])


# ---------------------------------------------------------------------- depth-1 related-property resolver


def test_depth1_reference_property_stringmatching_and_comparison() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ref_a = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/a")), eager=True)
        ref_b = store.fetch(ReferenceRow, store.save(ReferenceRow("10.2/b")), eager=True)
        cite_a = _save(store, CitingRow("cites-a", cite=ref_a))
        cite_b = _save(store, CitingRow("cites-b", cite=ref_b))
        _save(store, CitingRow("cites-none"))

        # The dotted related-property filter now really resolves (was matches-nothing).
        assert _ids(_resolved_query(store, CitingFamily, 'references.doi CONTAINS "10.1"')) == [cite_a]
        assert _ids(_resolved_query(store, CitingFamily, 'references.doi = "10.2/b"')) == [cite_b]
        # Non-HAS comparison on <type>.id routes through the same semi-join.
        assert _ids(_resolved_query(store, CitingFamily, f'references.id = "{ref_a.id}"')) == [cite_a]


def test_depth1_empty_match_and_not_composition() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ref_a = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/a")), eager=True)
        ref_b = store.fetch(ReferenceRow, store.save(ReferenceRow("10.2/b")), eager=True)
        _save(store, CitingRow("cites-a", cite=ref_a))
        cite_b = _save(store, CitingRow("cites-b", cite=ref_b))
        cite_none = _save(store, CitingRow("cites-none"))

        # An empty resolver result is a constant-false expression (matches nothing).
        assert _ids(_resolved_query(store, CitingFamily, 'references.doi CONTAINS "nomatch"')) == []
        # NOT strips into the semi-join (identical to the in-memory route): every
        # citing row whose reference does NOT match, INCLUDING the reference-less row.
        assert _ids(_resolved_query(store, CitingFamily, 'NOT (references.doi CONTAINS "10.1")')) == sorted(
            [cite_b, cite_none]
        )


def test_depth1_excludes_stale_revision_value() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ref = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/old")), eager=True)
        cite = _save(store, CitingRow("cites", cite=ref))
        # Supersede the referenced lineage with a new doi. The reference FK still
        # points at the old revision, but resolution runs only_latest over mains.
        store.replace(ref, ReferenceRow("10.2/new"))

        # The stale value no longer satisfies the filter ...
        assert _ids(_resolved_query(store, CitingFamily, 'references.doi CONTAINS "10.1"')) == []
        # ... while the current (latest main) value does.
        assert _ids(_resolved_query(store, CitingFamily, 'references.doi CONTAINS "10.2"')) == [cite]


def test_depth1_excludes_alternative_value() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ref = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/main")), eager=True)
        # A named alternative of the reference lineage carrying a distinctive doi.
        store.save(ReferenceRow("99.9/alt"), alternative_of=ref.id, alternative_kind="conventional")
        cite = _save(store, CitingRow("cites", cite=ref))

        # The alternative's doi must never satisfy the filter (mains-only resolution).
        assert _ids(_resolved_query(store, CitingFamily, 'references.doi CONTAINS "99.9"')) == []
        # The main's doi does.
        assert _ids(_resolved_query(store, CitingFamily, 'references.doi CONTAINS "10.1"')) == [cite]


def test_depth2_related_property_not_implemented() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ref = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/a")), eager=True)
        _save(store, CitingRow("cites", cite=ref))
        # Positive same-route depth-1 control.
        assert _ids(_resolved_query(store, CitingFamily, 'references.doi CONTAINS "10.1"')) != []
        # A depth>=2 dotted path stays not-implemented (never reaches the resolver).
        with pytest.raises(FilterTranslationError) as excinfo:
            _resolved_query(store, CitingFamily, 'references.doi.deep CONTAINS "x"')
        assert excinfo.value.category == "not-implemented"


def test_depth1_without_resolver_matches_nothing() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ref = store.fetch(ReferenceRow, store.save(ReferenceRow("10.1/a")), eager=True)
        _save(store, CitingRow("cites", cite=ref))
        # A federation without a resolver factory keeps the matches-nothing fallback
        # for dotted related-property filters (only <type>.id HAS is served directly).
        assert _ids(_query(store, CitingFamily, 'references.doi CONTAINS "10.1"')) == []
