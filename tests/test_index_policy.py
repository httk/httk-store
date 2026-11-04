"""Per-store index policies: ``indexes=`` / ``skip_indexes=`` on SQL stores."""

from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import pytest
import sqlalchemy

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql.layout import actual_index_names, expected_metadata, read_store_metadata

atomistic = pytest.importorskip("httk.atomistic")
pytest.importorskip("duckdb_engine")

_RECORDS = {atomistic.StructureEntry: atomistic.UnitcellStructureRecord}
_SITES_CHILD = "ix_atomistic_sites_reduced_coords_atomistic_sites_sid"
_SUBRECORD_LOGICAL_ID = "ix_atomistic_cell_logical_id"
_ENTRY_BOOKKEEPING = {
    "ix_atomistic_unitcell_structure_logical_id",
    "ix_atomistic_unitcell_structure_alt_id",
    "ix_atomistic_unitcell_structure_alt_id_alt_kind",
    "ix_atomistic_unitcell_structure_store_timestamp",
}


def _structure(a: int) -> Any:
    return atomistic.UnitcellStructure(
        atomistic.Cell([[a, 0, 0], [0, a, 0], [0, 0, a]]),
        atomistic.Sites([[0, 0, 0], [Fraction(1, 2)] * 3]),
        (atomistic.Species("Na", ("Na",), (1,)), atomistic.Species("Cl", ("Cl",), (1,))),
        ("Na", "Cl"),
    )


def _open(kind: str, path: Path, **options: Any) -> SqlStore:
    database = Backend.duckdb(path) if kind == "duckdb" else Backend.sqlite(path)
    try:
        return SqlStore(database, entry_records=_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"), **options)
    except BaseException:
        database.dispose()
        raise


def _indexes(store: SqlStore) -> frozenset[str]:
    with store._database.engine.connect() as connection:
        return actual_index_names(connection)


def _stamp(store: SqlStore) -> str | None:
    with store._database.engine.connect() as connection:
        metadata = read_store_metadata(connection)
    assert metadata is not None
    return metadata.get("index_policy")


def _close(store: SqlStore) -> None:
    store._database.dispose()


def _parent_sid_indexes(store: SqlStore) -> set[str]:
    """Child-table parent-sid index names, found structurally (a single ``<parent>_sid`` column)."""
    return {
        str(index.name)
        for table in expected_metadata(store.layout).tables.values()
        for index in table.indexes
        if len(index.columns) == 1 and next(iter(index.columns)).name.endswith("_sid")
    }


def _all_indexes(tmp_path: Path) -> frozenset[str]:
    store = _open("sqlite", tmp_path / "reference.sqlite")
    store.save(_structure(4))
    names = frozenset(name for name in _indexes(store) if not name.startswith("sqlite_autoindex_"))
    _close(store)
    return names


def test_new_duckdb_store_is_minimal(tmp_path: Path) -> None:
    every = _all_indexes(tmp_path)
    store = _open("duckdb", tmp_path / "s.duckdb")
    store.save(_structure(4))
    present = _indexes(store)
    child = _parent_sid_indexes(store)
    assert _SITES_CHILD in child
    assert not present & child
    assert _SUBRECORD_LOGICAL_ID not in present
    assert _ENTRY_BOOKKEEPING <= present
    assert {name for name in every if name.startswith("uq_")} <= present
    assert _stamp(store) == '{"indexes": "minimal", "skip_indexes": []}'
    _close(store)


def test_new_sqlite_store_keeps_all_indexes(tmp_path: Path) -> None:
    store = _open("sqlite", tmp_path / "s.sqlite")
    store.save(_structure(4))
    present = _indexes(store)
    assert _parent_sid_indexes(store) <= present
    assert {_SUBRECORD_LOGICAL_ID, *_ENTRY_BOOKKEEPING} <= present
    assert _stamp(store) is None
    _close(store)


def test_skip_indexes_drops_exactly_the_named_index_and_survives_upgrade(tmp_path: Path) -> None:
    every = _all_indexes(tmp_path)
    path = tmp_path / "s.sqlite"
    store = _open("sqlite", path, skip_indexes=[_SITES_CHILD])
    store.save(_structure(4))
    assert _indexes(store) - {n for n in _indexes(store) if n.startswith("sqlite_autoindex_")} == every - {_SITES_CHILD}
    assert _stamp(store) == f'{{"indexes": "all", "skip_indexes": ["{_SITES_CHILD}"]}}'
    with store._database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DROP INDEX "{_SUBRECORD_LOGICAL_ID}"'))
    _close(store)
    upgraded = _open("sqlite", path, upgrade=True)
    present = _indexes(upgraded)
    assert _SUBRECORD_LOGICAL_ID in present  # upgrade=True heals a missing kept index ...
    assert _SITES_CHILD not in present  # ... but never creates a skipped one
    _close(upgraded)


@pytest.mark.parametrize("finalize", ["auto", "parity"])
@pytest.mark.parametrize(
    ("kind", "options", "absent"),
    [
        ("sqlite", {"skip_indexes": [_SITES_CHILD]}, {_SITES_CHILD}),
        ("duckdb", {}, {_SITES_CHILD, _SUBRECORD_LOGICAL_ID}),
    ],
)
def test_bulk_ingest_honours_the_policy(
    tmp_path: Path, kind: str, options: dict[str, Any], absent: set[str], finalize: Literal["auto", "parity"]
) -> None:
    store = _open(kind, tmp_path / f"s.{kind}", **options)
    with store.bulk_ingest(finalize=finalize) as bulk:
        first = bulk.save(_structure(4))
    after_build = _indexes(store)
    assert not after_build & absent
    assert _ENTRY_BOOKKEEPING <= after_build
    # Appending to the populated store (rebuild drops and recreates the table indexes).
    with store.bulk_ingest(index_strategy="rebuild") as bulk:
        second = bulk.save(_structure(5))
    assert _indexes(store) == after_build
    assert store.fetch(atomistic.UnitcellStructureRecord, first).cell.basis[0] == 4
    assert store.fetch(atomistic.UnitcellStructureRecord, second).cell.basis[0] == 5
    _close(store)


def test_invalid_skip_indexes_are_refused(tmp_path: Path) -> None:
    for name in ("uq_atomistic_cell_content_id", "sqlite_autoindex__httk_store_metadata_1", "atomistic_cell_pkey"):
        with pytest.raises(ValueError, match="integrity constraints and cannot be skipped"):
            _open("sqlite", tmp_path / "s.sqlite", skip_indexes=[name])
    with pytest.raises(ValueError, match="never create it; close matches: ix_atomistic_cell_logical_id"):
        _open("sqlite", tmp_path / "s.sqlite", skip_indexes=["ix_atomistic_cell_logicalid"])
    with pytest.raises(ValueError, match="indexes must be"):
        _open("sqlite", tmp_path / "s.sqlite", indexes="none")
    with pytest.raises(TypeError, match="skip_indexes"):
        _open("sqlite", tmp_path / "s.sqlite", skip_indexes=_SITES_CHILD)
    # Every refusal happens before the store is created.
    store = _open("sqlite", tmp_path / "s.sqlite")
    assert _stamp(store) is None
    _close(store)


def test_reopen_adopts_the_stamp_and_refuses_a_conflicting_policy(tmp_path: Path) -> None:
    duck, lite = tmp_path / "s.duckdb", tmp_path / "s.sqlite"
    _close(_open("duckdb", duck))
    _close(_open("sqlite", lite, skip_indexes=[_SITES_CHILD]))
    # Tables are created after the reopen, so they show the adopted policy.
    for kind, path, absent in (("duckdb", duck, _SUBRECORD_LOGICAL_ID), ("sqlite", lite, _SITES_CHILD)):
        store = _open(kind, path)
        store.save(_structure(4))
        assert absent not in _indexes(store)
        _close(store)
    _close(_open("duckdb", duck, indexes="minimal"))
    _close(_open("sqlite", lite, indexes="all", skip_indexes=[_SITES_CHILD]))
    with pytest.raises(ValueError, match=r"indexes='all'.*differs from the store's \(indexes='minimal'"):
        _open("duckdb", duck, indexes="all")
    with pytest.raises(ValueError, match="rebuild the store to change it"):
        _open("sqlite", lite, skip_indexes=[_SUBRECORD_LOGICAL_ID])


def test_unstamped_store_opens_as_all(tmp_path: Path) -> None:
    # A store from before index policies: simulate it by removing the stamp of
    # a still-empty DuckDB store (an explicit "all" store is never stamped).
    path = tmp_path / "s.duckdb"
    store = _open("duckdb", path)
    with store._database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("DELETE FROM _httk_store_metadata WHERE key = 'index_policy'"))
    _close(store)
    with pytest.raises(ValueError, match="differs"):
        _open("duckdb", path, indexes="minimal")
    store = _open("duckdb", path)
    store.save(_structure(4))
    assert {_SITES_CHILD, _SUBRECORD_LOGICAL_ID} <= _indexes(store)
    assert _stamp(store) is None
    _close(store)
    explicit = _open("duckdb", tmp_path / "all.duckdb", indexes="all")
    assert _stamp(explicit) is None
    _close(explicit)


def test_data_round_trips_identically_under_both_policies(tmp_path: Path) -> None:
    fetched = []
    for name, options in (("minimal", {}), ("all", {"indexes": "all"})):
        store = _open("duckdb", tmp_path / f"{name}.duckdb", **options)
        sid = store.save(_structure(4))
        fetched.append(store.fetch(atomistic.UnitcellStructureRecord, sid))
        _close(store)
    assert fetched[0] == fetched[1]
    assert fetched[0].sites.reduced_coords == _structure(4).sites.reduced_coords
