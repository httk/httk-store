"""Stored OPTIMADE ``references`` weak links: bounded link table names and end-to-end serving.

Runs over the parametrized ``store_factory`` arms (sqlite/duckdb, and
postgresql/mongo when configured); the mongo arm covers save dedup, linking and
the bounded collection name, then skips the SQL-only serving checks.
``AverageTotalEnergyRecord``'s link table name would be 64 characters unbounded.
"""

import hashlib

import pytest
from httk.core import (
    AverageTotalEnergyRecord,
    DataRecordEntry,
    DerivedDataRecord,
    FileEntry,
    FileRecord,
    ReferenceEntry,
    ReferenceRecord,
    Run,
    RunEntry,
    TotalEnergyRecord,
)

from httk.store import EntryIdScheme
from httk.store.backend.schema import _bounded_identifier, resolve_schema
from httk.store.backend.sql import SqlStore, StoredEntrySource
from httk.store.backend.sql.stored_federation import StoredEntryFederation

_PAW = {
    "doi": "10.1103/PhysRevB.59.1758",
    "title": "From ultrasoft pseudopotentials to the projector augmented-wave method",
    "authors": ({"name": "G. Kresse"}, {"name": "D. Joubert"}),
}


def test_bounded_identifier_keeps_short_names_and_hashes_long_ones() -> None:
    short = "_httk_link_core_derived_data_record__core_reference__references"
    assert len(short) == 63
    assert _bounded_identifier(short) == short
    first = "_httk_link_atomistic_unitcell_structure__core_reference__references"
    second = "_httk_link_atomistic_unitcell_structure__core_reference__citations"
    bounded = _bounded_identifier(first)
    assert len(bounded) == 63
    assert bounded == _bounded_identifier(first)
    assert bounded == f"{first[:54]}_{hashlib.sha256(first.encode()).hexdigest()[:8]}"
    assert _bounded_identifier(second) != bounded
    assert bounded.startswith("_httk_link_")


def test_long_link_table_name_is_bounded() -> None:
    (link,) = resolve_schema(AverageTotalEnergyRecord).links
    assert link.table_name == _bounded_identifier("_httk_link_core_average_total_energy__core_reference__references")
    assert len(link.table_name) == 63
    (short,) = resolve_schema(TotalEnergyRecord).links
    assert short.table_name == "_httk_link_core_total_energy__core_reference__references"


def _query(store: SqlStore, family: type, filter_string: str | None = None) -> object:
    return StoredEntryFederation((StoredEntrySource(store, family, "src"),)).query(filter_string)


def _ids(page: object) -> list[str]:
    return sorted(row["id"] for row in page.rows)  # type: ignore[attr-defined]


def test_references_link_serves_and_filters(store_factory) -> None:
    store = store_factory(
        entry_records={
            # DerivedDataRecord's link table is exactly 63 characters (PostgreSQL sequence case).
            DataRecordEntry: (TotalEnergyRecord, AverageTotalEnergyRecord, DerivedDataRecord),
            RunEntry: Run,
            FileEntry: FileRecord,
            ReferenceEntry: ReferenceRecord,
        },
        entry_ids=EntryIdScheme("httk.test", "1"),
    )
    # Content-deduplicated: the same reference saved twice is one entry.
    ref_sid = store.save(ReferenceRecord(**_PAW))  # type: ignore[arg-type]
    assert store.save(ReferenceRecord(**_PAW)) == ref_sid  # type: ignore[arg-type]
    ref = store.fetch(ReferenceRecord, ref_sid, eager=True)
    assert ref.id is not None

    run = store.fetch(Run, store.save(Run(source_id="ws:job")), eager=True)
    file = store.fetch(FileRecord, store.save(FileRecord(url=None, name="OUTCAR", sha256="a" * 64)), eager=True)
    energy_sid = store.save(AverageTotalEnergyRecord(-1.5))
    energy = store.fetch(AverageTotalEnergyRecord, energy_sid, eager=True)
    unlinked = store.fetch(TotalEnergyRecord, store.save(TotalEnergyRecord(-2.5)), eager=True)
    for citing in (run, file, energy):
        store.link(citing, "references", ref)
    assert [target.id for target in store.linked(energy, "references")] == [ref.id]
    if not isinstance(store, SqlStore):
        # The bounded link name is the Mongo collection name.
        (link,) = resolve_schema(AverageTotalEnergyRecord).links
        assert link.table_name in store._database.database.list_collection_names()  # type: ignore[attr-defined]
        pytest.skip("stored-federation serving, filtering and the sequence/reopen checks are SQL-only")

    # Served relationships (the `httk serve optimade` stored federation path).
    for family, citing in ((RunEntry, run), (FileEntry, file), (DataRecordEntry, energy)):
        page = _query(store, family, f'id = "{citing.id}"')
        (relationships,) = page.relationships  # type: ignore[attr-defined]
        assert [(e.entry_type, e.id) for e in relationships["references"]] == [("references", ref.id)]

    # `references.id HAS` filters (bare and alias spellings).
    assert _ids(_query(store, FileEntry, f'references.id HAS "{ref.id}"')) == [file.id]
    # The unlinked total-energy record is in the family but not matched.
    assert _ids(_query(store, DataRecordEntry)) == sorted([energy.id, unlinked.id])
    assert _ids(_query(store, DataRecordEntry, f'references.id HAS "{ref.id}"')) == [energy.id]
    alias = _query(store, DataRecordEntry, f'_httk_relationships.references.id HAS "{ref.id}"')
    assert _ids(alias) == [energy.id]

    # The references entry serves its fields in OPTIMADE shape.
    (row,) = _query(store, ReferenceEntry).rows  # type: ignore[attr-defined]
    assert row["id"] == ref.id
    assert row["doi"] == _PAW["doi"]
    assert row["title"] == _PAW["title"]
    assert row["authors"] == [{"name": "G. Kresse"}, {"name": "D. Joubert"}]

    # The hashed link table and its sequence are recognized as store-owned on reopen.
    reopened = store_factory.reopen(store)
    cited = reopened.fetch(AverageTotalEnergyRecord, energy_sid)
    assert [target.id for target in reopened.linked(cited, "references")] == [ref.id]
