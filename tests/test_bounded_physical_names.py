"""End-to-end storage of records whose derived physical names exceed 63 characters.

Every derived identifier is hash-shortened where it is resolved, so these
records store on every arm (sqlite/duckdb, and postgresql/mongo when
configured) instead of failing on DuckDB or being truncated by PostgreSQL.
The 58-character field names push every bookkeeping column (``_present``,
``_index``, ``_exact``, ``<field>_sid``, ``_<i>``) over the limit.
"""

import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Any, ClassVar

import pytest
from httk.core import FracVector
from httk.core.storage import Shape, StorageInfo, WeakLink

from httk.store import EntryIdScheme
from httk.store.backend.schema import TableSchema, _sid_sequence_name, resolve_schema
from httk.store.backend.sql import SqlStore

_LONG_TABLE = "long_physical_names_" + "n" * 40  # 60 characters
_CHILD = "c" + "m" * 56 + "x"  # optional tuple child: _present, _index
_ROWS = "g" + "m" * 56 + "x"  # FracVector child: _index, _0.._2, _exact
_EXACT = "h" + "m" * 56 + "x"  # Fraction: _exact
_REF = "r" + "m" * 56 + "x"  # reference: _sid
_TENSOR = "k" + "m" * 56 + "x"  # fixed tensor: _0.._8, _exact
_LINK = "targets_under_a_long_link_name"


@dataclass(frozen=True)
class LongNameTarget:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="long_name_target")

    label: str


LongNames: Any = dataclasses.make_dataclass(
    "LongNames",
    [
        ("label", str),
        (_ROWS, Annotated[FracVector, Shape(0, 3)]),
        (_EXACT, Fraction),
        (_REF, LongNameTarget | None, dataclasses.field(default=None)),
        (_CHILD, tuple[int, ...] | None, dataclasses.field(default=None)),
        (_TENSOR, Annotated[FracVector, Shape(3, 3)] | None, dataclasses.field(default=None)),
    ],
    namespace={"__httk_storage__": StorageInfo(storage_name=_LONG_TABLE, links=(WeakLink(_LINK, LongNameTarget),))},
    frozen=True,
    module=__name__,
)

_COORDS = FracVector([[0, 0, 0], [Fraction(1, 2), Fraction(1, 3), 1]])
_TENSOR_VALUE = FracVector([[1, 0, 0], [0, Fraction(1, 2), 0], [0, 0, 1]])


def _record(label: str, exact: Fraction, child: tuple[int, ...] | None, ref: object = None, tensor: object = None):
    return LongNames(label, _COORDS, exact, **{_REF: ref, _CHILD: child, _TENSOR: tensor})


def _physical_names(schema: TableSchema) -> list[str]:
    names = [schema.table_name, _sid_sequence_name(schema.table_name), *(link.table_name for link in schema.links)]
    for spec in schema.fields:
        names.extend(column.name for column in spec.columns)
        if spec.child is not None:
            names.append(spec.child.table_name)
            names.extend(column.name for column in spec.child.element_columns)
    return names


def _labels(store: Any, where: Any = None, sort: bool = False) -> list[str]:
    searcher = store.searcher()
    variable = searcher.variable(LongNames)
    if where is not None:
        searcher.add(where(variable))
    if sort:
        searcher.add_sort(getattr(variable, _EXACT), descending=True)
    labels = [row[0].label for row in searcher.results(r=variable)]
    return labels if sort else sorted(labels)


def test_long_physical_names_store_query_link_fsck_reopen(store_factory) -> None:
    names = _physical_names(resolve_schema(LongNames))
    assert all(len(name) <= 63 for name in names), names
    assert sum(len(name) == 63 for name in names) > 5  # many columns really were shortened

    store = store_factory(transactions="require")
    target = LongNameTarget("t")
    store.save(target)
    records = [
        _record("a", Fraction(1, 3), (1, 2, 3), target, _TENSOR_VALUE),
        _record("b", Fraction(2, 3), None),
        _record("c", Fraction(5, 3), (), None, _TENSOR_VALUE),
        _record("d", Fraction(1, 7), (7,)),
    ]
    sids = [store.save(record) for record in records]
    assert store.save(_record("a", Fraction(1, 3), (1, 2, 3), target, _TENSOR_VALUE)) == sids[0]
    for sid, record in zip(sids, records, strict=True):
        assert store.fetch(LongNames, sid, eager=True) == record

    assert _labels(store, lambda v: getattr(v, _EXACT) > Fraction(1, 2)) == ["b", "c"]
    assert _labels(store, lambda v: getattr(v, _CHILD).has_any(2, 7)) == ["a", "d"]
    assert _labels(store, sort=True) == ["c", "b", "a", "d"]

    stored = store.fetch(LongNames, sids[0])
    store.link(stored, _LINK, target)
    assert [linked.label for linked in store.linked(stored, _LINK)] == ["t"]

    exclusive = {"exclusive": True} if isinstance(store, SqlStore) else {}
    assert store.fsck(known_types=(LongNames, LongNameTarget), **exclusive).violations == ()

    reopened = store_factory.reopen(store)
    again = reopened.fetch(LongNames, sids[0], eager=True)
    assert again == records[0]
    assert [linked.label for linked in reopened.linked(again, _LINK)] == ["t"]
    replacement = _record("a2", Fraction(9), (9,))
    assert reopened.fetch(LongNames, reopened.replace(again, replacement), eager=True) == replacement


@pytest.mark.parametrize("workers", [1, 2])
def test_long_physical_names_deferred_bulk_ingest(store_factory, workers: int) -> None:
    store = store_factory()
    if not isinstance(store, SqlStore):
        pytest.skip("bulk_ingest is SQL-only")
    records = [_record(str(i), Fraction(i, 7), (i,) if i % 2 else None, None, _TENSOR_VALUE) for i in range(6)]
    with store.bulk_ingest(finalize="deferred", workers=workers, track_sids=False) as bulk:
        for record in (*records, records[0]):
            bulk.save(record)
    assert _labels(store) == [str(i) for i in range(6)]
    assert _labels(store, lambda v: getattr(v, _CHILD).has_any(3)) == ["3"]
    assert store.fsck(known_types=(LongNames,), exclusive=True).violations == ()


def test_generated_analyse_records_with_long_child_tables(store_factory) -> None:
    """The records ``httk collect --into`` stores: generated child tables up to 93 characters before bounding."""
    analyse = pytest.importorskip("httk.analyse.property_records")
    from httk.core import DataRecordEntry, RunEdge, TotalEnergyRecord

    heat = analyse.HeatCapacityConstantPressureStandardErrorRecord
    conductivity = analyse.ThermalConductivityTensorStandardErrorRecord
    declaration = {DataRecordEntry: (TotalEnergyRecord, heat, conductivity)}
    store = store_factory(entry_records=declaration, entry_ids=EntryIdScheme("httk.test", "1"))
    source = store.fetch(TotalEnergyRecord, store.save(TotalEnergyRecord(-1.0)), eager=True)
    edge = (RunEdge("source", "_httk_records", source.id),)
    records = (heat(0.25, product_of=edge), conductivity(tuple(float(i) for i in range(9)), product_of=edge))
    for record in records:
        assert store.fetch(type(record), store.save(record), eager=True) == record

    # Deferred bulk ingest into a fresh store: its temporary stage/map relations
    # collided after PostgreSQL's truncation before they were bounded.
    fresh = store_factory(entry_records=declaration, entry_ids=EntryIdScheme("httk.test", "1"))
    if not isinstance(fresh, SqlStore):
        pytest.skip("bulk_ingest is SQL-only")
    tensors = [conductivity(tuple(float(i + j) for i in range(9))) for j in range(3)]
    with fresh.bulk_ingest(finalize="deferred") as bulk:
        for record in (*tensors, tensors[0]):
            bulk.save(record)
    searcher = fresh.searcher()
    variable = searcher.variable(conductivity)
    stored = sorted(row[0].thermal_conductivity_tensor_standard_error for row in searcher.results(r=variable))
    assert stored == sorted(record.thermal_conductivity_tensor_standard_error for record in tensors)
    assert fresh.fsck(known_types=(conductivity,), exclusive=True).violations == ()
