"""``bulk_ingest(workers>1)`` accepts view objects, which cross into forked workers by pickle."""

from pathlib import Path

import httk.core
import pytest
from httk.core.storage import content_id
from test_db_bulk import _table_stats

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore

pytest.importorskip("httk.atomistic")

pytestmark = pytest.mark.xdist_group("bulk-heavy")

COD = Path(__file__).parent / "fixtures" / "cod" / "1000008.cif"


def _views() -> list[object]:
    from httk.atomistic import Cell, Sites, Species, StructureSymmetry, UnitcellStructure, UnitcellStructureView

    def rocksalt(a: int) -> UnitcellStructure:
        return UnitcellStructure(
            Cell([[a, 0, 0], [0, a, 0], [0, 0, a]]),
            Sites([[0, 0, 0], ["1/2", "1/2", "1/2"]]),
            (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
            ("Na", "Cl"),
        )

    def stated_rocksalt(a: int) -> UnitcellStructure:
        # States a space group without operations; its symmetry holds read-only settings.
        return UnitcellStructure(
            Cell([[a, 0, 0], [0, a, 0], [0, 0, a]]),
            Sites([[0, 0, 0], ["1/2", "1/2", "1/2"]]),
            (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
            ("Na", "Cl"),
            symmetry=StructureSymmetry(space_group_it_number=225),
        )

    cod = UnitcellStructureView(httk.core.load(str(COD)))
    # Repeated objects exercise dedupe across the workers' shards.
    return [
        cod,
        UnitcellStructureView(rocksalt(4)),
        UnitcellStructureView(rocksalt(5)),
        UnitcellStructureView(stated_rocksalt(6)),
        cod,
    ]


def _ingest(views: list[object], workers: int) -> dict[str, tuple[int, frozenset[str] | None]]:
    from httk.atomistic import StructureEntry, UnitcellStructureRecord

    database = Backend.sqlite()
    try:
        store = SqlStore(
            database,
            entry_records={StructureEntry: UnitcellStructureRecord},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        with store.bulk_ingest(workers=workers, chunk_size=1) as bulk:
            for view in views:
                bulk.save(view)
        return _table_stats(store, database)
    finally:
        database.dispose()


def test_parallel_ingest_of_views_matches_serial() -> None:
    views = _views()
    parallel = _ingest(views, workers=2)
    assert parallel == _ingest(views, workers=1)
    content_ids = frozenset().union(*(ids for _count, ids in parallel.values() if ids is not None))
    assert {content_id(view) for view in views} <= content_ids
