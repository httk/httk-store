"""OPTIMADE serving of the ``imports`` entry type: every property projected, filters and relationships.

Runs over the parametrized ``store_factory`` arms; serving and filtering are SQL-only, as in
``test_curation_links.py``.
"""

import datetime
from fractions import Fraction as F
from typing import Any

import pytest
import sqlalchemy
from httk.atomistic import (
    ASUStructure,
    ASUStructureImportRecord,
    ASUStructureRecord,
    CIFStructure,
    CIFStructureImportRecord,
    CIFStructureRecord,
    Species,
    StructureEntry,
    UnitcellStructureImportRecord,
    UnitcellStructureRecord,
    WyckoffSite,
)
from httk.atomistic.io.cif.cif_items import CIFDataItem
from httk.core import FileEntry, FileRecord, ImportEntry, ImportRecord, ImportReport
from httk.core.register.schemas import load_entry_type_definition
from httk.core.storage import project_storage_record, stored_property_projections

from httk.store import EntryIdScheme
from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql import SqlStore, StoredEntrySource
from httk.store.backend.sql.stored_federation import StoredEntryFederation
from httk.store.query.optimade_filters import FilterTranslationError

_KINDS = (ImportRecord, ASUStructureImportRecord, UnitcellStructureImportRecord, CIFStructureImportRecord)
_RECORDS: dict[type, Any] = {
    ImportEntry: _KINDS,
    StructureEntry: (ASUStructureRecord, UnitcellStructureRecord, CIFStructureRecord),
    FileEntry: FileRecord,
}
_AT = datetime.datetime(2026, 11, 25, 12, 0, tzinfo=datetime.UTC)
_FILE = FileRecord("https://example.org/1000000.cif", "1000000.cif", sha256="0" * 64)
_REPORTS = (
    ImportReport("warning", "Dropped an auxiliary loop.", "cif", "httk.atomistic.io.cif"),
    ImportReport("info", "Inferred symbols from labels."),
)
_SERVED = {
    "_httk_source_uri": "source_uri",
    "_httk_source_item": "source_item",
    "_httk_importer": "importer",
    "_httk_status": "status",
    "_httk_error": "error",
    "_httk_repaired": "repaired",
    "_httk_imported_at": "imported_at",
    "last_modified": "last_modified",
}


def _asu() -> ASUStructureRecord:
    a = F(564, 100)
    na, cl = Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))
    return ASUStructureRecord.from_structure(
        ASUStructure(
            [[a, 0, 0], [0, a, 0], [0, 0, a]], 225, [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")], [na, cl]
        )
    )


def _imported(source_uri: str = "cif/1/00/00/1000000.cif") -> ASUStructureImportRecord:
    return ASUStructureImportRecord(
        source_uri,
        "build-cod 1.0",
        "imported",
        source_item="1000000",
        source_file=_FILE,
        reports=_REPORTS,
        repaired=True,
        imported_at=_AT,
        structure=_asu(),
    )


_FAILED = ImportRecord(
    "cif/2.cif", "other-importer 2", "failed", error="ValueError: no cell", imported_at=_AT, last_modified=_AT
)
_CIF = CIFStructureImportRecord(
    "cif/4.cif",
    "cif-importer 3",
    "imported",
    structure=CIFStructureRecord(
        **project_storage_record(CIFStructureRecord, CIFStructure("a", (CIFDataItem("_a", "x"),)))  # type: ignore[arg-type]
    ),
)
_EXCLUDED = UnitcellStructureImportRecord(
    "cif/3.cif", "build-cod 1.0", "excluded", error="Disordered.", imported_at=_AT - datetime.timedelta(days=1)
)


def _store(store_factory) -> SqlStore:
    store = store_factory(entry_records=_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
    if not isinstance(store, SqlStore):
        pytest.skip("stored-federation serving and filtering are SQL-only")
    return store


def _query(store: SqlStore, family: type, filter_string: str | None = None):
    return StoredEntryFederation((StoredEntrySource(store, family, "src"),)).query(filter_string)


def _ids(page) -> list[str]:
    return sorted(row["id"] for row in page.rows)


def _wire(value: object) -> object:
    return value.isoformat() if isinstance(value, datetime.datetime) else value


def _relationships(page, entry_id: str) -> dict[str, list[str]]:
    (index,) = [index for index, row in enumerate(page.rows) if row["id"] == entry_id]
    return {key: [entry.id for entry in entries] for key, entries in page.relationships[index].items()}


def test_every_served_property_has_a_projection_on_each_kind() -> None:
    served = load_entry_type_definition(ImportEntry.definition_id).served_form().properties
    expected = set(served) - {"id", "type", "immutable_id"}
    for kind in _KINDS:
        assert set(stored_property_projections(kind)) == expected, kind.__name__


def test_import_records_serve_filter_and_relate(store_factory) -> None:
    store = _store(store_factory)
    records = (_imported(), _FAILED, _EXCLUDED, _CIF)
    saved = [store.fetch(type(record), store.save(record), eager=True) for record in records]
    imported, failed, excluded, cif = saved

    rows = {row["id"]: row for row in _query(store, ImportEntry).rows}
    assert set(rows) == {record.id for record in saved}
    for record in saved:
        row = rows[record.id]
        # No silent nulls: every served property is the record's own value.
        assert {name: row[name] for name in _SERVED} == {
            name: _wire(getattr(record, field)) for name, field in _SERVED.items()
        }
    assert rows[imported.id]["_httk_reports"] == [
        {
            "level": "warning",
            "message": "Dropped an auxiliary loop.",
            "context": "cif",
            "logger": "httk.atomistic.io.cif",
        },
        {"level": "info", "message": "Inferred symbols from labels.", "context": None, "logger": None},
    ]
    assert rows[failed.id]["_httk_reports"] == []

    assert _ids(_query(store, ImportEntry, '_httk_status = "failed"')) == [failed.id]
    assert _ids(_query(store, ImportEntry, "_httk_repaired = TRUE")) == [imported.id]
    assert _ids(_query(store, ImportEntry, '_httk_importer STARTS "build-cod"')) == sorted([imported.id, excluded.id])
    assert _ids(_query(store, ImportEntry, '_httk_imported_at > "2026-11-25T00:00:00Z"')) == sorted(
        [imported.id, failed.id]
    )
    assert _ids(_query(store, ImportEntry, "_httk_error IS UNKNOWN")) == sorted([imported.id, cif.id])
    assert _ids(_query(store, ImportEntry, 'last_modified > "2026-11-25T00:00:00Z"')) == [failed.id]
    assert _ids(_query(store, ImportEntry, "last_modified IS UNKNOWN")) == sorted([imported.id, excluded.id, cif.id])

    page = _query(store, ImportEntry, f'id = "{imported.id}"')
    structure = imported.structure
    assert structure is not None and structure.id is not None
    assert imported.source_file is not None and imported.source_file.id is not None
    assert _relationships(page, imported.id) == {"structures": [structure.id], "files": [imported.source_file.id]}
    assert _ids(_query(store, ImportEntry, f'files.id HAS "{imported.source_file.id}"')) == [imported.id]
    assert cif.structure is not None and cif.structure.id is not None
    assert _relationships(_query(store, ImportEntry, f'id = "{cif.id}"'), cif.id) == {"structures": [cif.structure.id]}


def test_structure_to_import_record_by_relationship_filter(store_factory) -> None:
    store = _store(store_factory)
    imported = store.fetch(ASUStructureImportRecord, store.save(_imported()), eager=True)
    base = store.fetch(ImportRecord, store.save(_FAILED), eager=True)  # no structure field at all
    null = store.fetch(UnitcellStructureImportRecord, store.save(_EXCLUDED), eager=True)  # NULL structure
    assert imported.structure is not None
    structure = imported.structure.id

    def ids(filter_string: str) -> list[str]:
        return _ids(_query(store, ImportEntry, filter_string))

    # From a structure to its import record: a relationship filter on imports.
    assert ids(f'structures.id HAS "{structure}"') == [imported.id]
    # The base backing has no ``structures`` relationship and answers as an empty one, like a NULL reference.
    assert ids(f'NOT structures.id HAS "{structure}"') == sorted([base.id, null.id])
    assert ids(f'structures.id HAS ONLY "{structure}"') == sorted([imported.id, base.id, null.id])
    assert ids('structures.id HAS ONLY "nonesuch"') == sorted([base.id, null.id])
    assert ids('NOT structures.id HAS ONLY "nonesuch"') == [imported.id]
    assert ids(f'_httk_relationships.structures.id HAS ANY "{structure}","nonesuch"') == [imported.id]
    # A relationship no backing of the family serves stays an unrecognized property.
    with pytest.raises(FilterTranslationError, match="unrecognized property name: calculations.id"):
        ids('calculations.id HAS "x"')


@pytest.mark.parametrize("promote", [True, False])
def test_bulk_ingested_import_records_serve(store_factory, promote: bool) -> None:
    store = _store(store_factory)
    with store.bulk_ingest() as bulk:
        bulk.save(_imported(), promote=ASUStructureRecord if promote else None)

    (row,) = _query(store, ImportEntry).rows
    assert (row["_httk_status"], row["_httk_imported_at"]) == ("imported", _AT.isoformat())
    (structure_id,) = _relationships(_query(store, ImportEntry), row["id"])["structures"]
    # Served listings include the embedded structure whether or not it was promoted ...
    assert _ids(_query(store, StructureEntry)) == [structure_id]
    # ... promotion only marks it a main row (an fsck root with a dispatch row) instead of a dependency.
    table = store._table(resolve_schema(ASUStructureRecord).table_name)
    with store._read_connection() as connection:
        roles = connection.execute(sqlalchemy.select(table.c["_httk_role"])).scalars().all()
    assert roles == [1 if promote else 0]
