"""Searching StrongLink (provenance edge) relationships through the ``links`` namespace.

Forward traversals follow the edges a class declares (``run.links.has_output``,
``record.links.product_of``); reverse traversals follow the configured owner's edges that
point at the variable (``structure.links.is_output``, ``structure.links.has_product``).
SQL arms only: MongoDB raises ``UnsupportedQueryError`` for strong-link names.
"""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
from httk.core import DataRecord, DataRecordEntry, Run, RunEdge, RunEntry
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique

from httk.store import EntryIdScheme
from httk.store.backend.schema import SchemaError
from httk.store.backend.sql import SqlStore
from httk.store.query import UnsupportedQueryError

_STRUCTURES = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/total-energy"


@dataclass(frozen=True)
class StructureRow:
    """A minimal ``structures`` backing."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="strong_search_structure")

    formula: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class StructureFamily:
    """The structures family."""

    type = "structures"
    definition_id = _STRUCTURES


register_entry_family(name="strong-search-structures", family=f"{__name__}:StructureFamily", definition_id=_STRUCTURES)
register_entry_record(
    name="strong-search-structures-rec", family="strong-search-structures", record=f"{__name__}:StructureRow"
)


def _store(store_factory):
    store = store_factory(
        entry_records={RunEntry: Run, StructureFamily: StructureRow, DataRecordEntry: DataRecord},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )
    return store


def _save(store, record):
    return store.fetch(type(record), store.save(record), eager=True)


def _populate(store):
    """Two structures; each has a total energy record that is a product of it, and a run that output both."""
    ca = _save(store, StructureRow("Ca"))
    ti = _save(store, StructureRow("Ti"))
    e_ca = _save(
        store, DataRecord.from_value(_ENERGY, "e", -1.0, product_of=[RunEdge("structure", "structures", ca.id)])
    )
    e_ti = _save(
        store, DataRecord.from_value(_ENERGY, "e", -2.0, product_of=[RunEdge("structure", "structures", ti.id)])
    )
    orphan = _save(store, DataRecord.from_value(_ENERGY, "e", -3.0))
    run_ca = _save(
        store,
        Run(
            inputs=(RunEdge("initial", "structures", ca.id),),
            outputs=(RunEdge("relaxed", "structures", ca.id), RunEdge("energy", "records", e_ca.id)),
            source_id="ws:ca",
        ),
    )
    run_ti = _save(
        store,
        Run(
            inputs=(RunEdge("initial", "structures", ti.id),),
            outputs=(RunEdge("relaxed", "structures", ti.id), RunEdge("energy", "records", e_ti.id)),
            source_id="ws:ti",
        ),
    )
    return ca, ti, e_ca, e_ti, orphan, run_ca, run_ti


def _sql(store_factory):
    store = _store(store_factory)
    if not isinstance(store, SqlStore):
        pytest.skip("strong-link search is SQL-only")
    return store


def test_join_records_to_their_subject_structures(store_factory):
    store = _sql(store_factory)
    _ca, _ti, _e_ca, _e_ti, _orphan, _, _ = _populate(store)

    search = store.searcher()
    structure = search.variable(StructureRow)
    record = search.variable(DataRecord)
    search.add(record.links.product_of == structure)
    rows = sorted(
        (row.structure.formula, row.record.value) for row in search.results(structure=structure, record=record)
    )
    assert rows == [("Ca", -1.0), ("Ti", -2.0)]

    # The reverse direction from the structure side gives the same pairs.
    search = store.searcher()
    structure = search.variable(StructureRow)
    record = search.variable(DataRecord)
    search.add(structure.links.has_product == record)
    rows = sorted(
        (row.structure.formula, row.record.value) for row in search.results(structure=structure, record=record)
    )
    assert rows == [("Ca", -1.0), ("Ti", -2.0)]


def test_stored_object_operands_label_chaining_and_set_forms(store_factory):
    store = _sql(store_factory)
    ca, ti, _e_ca, _e_ti, _orphan, _, _ = _populate(store)

    def energies(add):
        search = store.searcher()
        record = search.variable(DataRecord)
        add(search, record)
        return sorted(search.results(value=record.value_number).scalars())

    assert energies(lambda s, r: s.add(r.links.product_of == ca)) == [-1.0]
    assert energies(lambda s, r: s.add(r.links.product_of != ca)) == [-3.0, -2.0]
    assert energies(lambda s, r: s.add(r.links.product_of.has_any(ca, ti))) == [-2.0, -1.0]
    # has_only: every edge among the allowed set; the orphan matches vacuously.
    assert energies(lambda s, r: s.add(r.links.product_of.has_only(ca))) == [-3.0, -1.0]
    assert energies(lambda s, r: s.add(r.links.product_of.label == "structure")) == [-2.0, -1.0]
    assert energies(lambda s, r: s.add(r.links.product_of.entry_type == "structures")) == [-2.0, -1.0]


def test_run_edges_are_searchable_both_ways(store_factory):
    store = _sql(store_factory)
    _ca, _ti, _e_ca, _e_ti, _orphan, run_ca, run_ti = _populate(store)

    search = store.searcher()
    run = search.variable(Run)
    structure = search.variable(StructureRow)
    search.add(run.links.has_output == structure)
    search.add(structure.formula == "Ti")
    assert [row.run.source_id for row in search.results(run=run)] == ["ws:ti"]

    search = store.searcher()
    run = search.variable(Run)
    record = search.variable(DataRecord)
    search.add(record.links.is_output == run)
    search.add(run.source_id == "ws:ca")
    assert list(search.results(value=record.value_number).scalars()) == [-1.0]

    search = store.searcher()
    structure = search.variable(StructureRow)
    search.add(structure.links.is_input == run_ca)
    assert list(search.results(formula=structure.formula).scalars()) == ["Ca"]

    # A structure that is nobody's input.
    free = _save(store, StructureRow("O"))
    search = store.searcher()
    structure = search.variable(StructureRow)
    search.add(~structure.links.is_input.has_any(run_ca, run_ti))
    assert list(search.results(formula=structure.formula).scalars()) == ["O"]
    assert free.formula == "O"


def test_unknown_names_and_unsupported_forms(store_factory):
    store = _sql(store_factory)
    _populate(store)
    search = store.searcher()
    record = search.variable(DataRecord)
    structure = search.variable(StructureRow)
    with pytest.raises(SchemaError, match="declares no link named 'nonsense'.*product_of.*is_input"):
        _ = record.links.nonsense
    with pytest.raises(UnsupportedQueryError, match="typed per edge"):
        _ = record.links.product_of.formula
    with pytest.raises(UnsupportedQueryError, match="forward strong link"):
        _ = structure.links.has_product.label  # reverse chaining would read every owner's edges
    with pytest.raises(UnsupportedQueryError, match="projects a strong-link traversal"):
        search.results(p=record.links.product_of)
    with pytest.raises(TypeError, match="expects a DataRecord variable"):
        _ = structure.links.has_product == search.variable(Run)
    with pytest.raises(TypeError, match="compare against a stored entry"):
        _ = record.links.product_of == "httk.test:1:whatever"


def test_mongo_refuses_strong_link_names(store_factory):
    store = _store(store_factory)
    if isinstance(store, SqlStore):
        pytest.skip("mongo arm only")
    search = store.searcher()
    record = search.variable(DataRecord)
    with pytest.raises(UnsupportedQueryError, match="not supported on MongoDB"):
        _ = record.links.product_of


def test_target_revisions_follow_only_latest(store_factory):
    store = _sql(store_factory)
    ca, _ti, _e_ca, _e_ti, _orphan, _run_ca, _run_ti = _populate(store)
    # A second revision of the Ca structure keeps the public id the edge names.
    store.replace(ca, StructureRow("Ca-v2"))

    def pairs(**searcher_options):
        search = store.searcher(**searcher_options)
        structure = search.variable(StructureRow)
        record = search.variable(DataRecord)
        search.add(record.links.product_of == structure)
        return sorted(
            (row.structure.formula, row.record.value) for row in search.results(structure=structure, record=record)
        )

    assert pairs() == [("Ca", -1.0), ("Ca-v2", -1.0), ("Ti", -2.0)]  # every revision, documented
    assert pairs(only_latest=True) == [("Ca-v2", -1.0), ("Ti", -2.0)]
