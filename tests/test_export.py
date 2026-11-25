"""Tests for definitions-bundled dataset exports."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from httk.core import Run, RunEntry, load_entry_type_definition
from httk.core.files import FileEntry, FileRecord
from httk.core.project import initialize_project

from httk.store import EntryIdScheme, export_dataset
from httk.store.backend.sql import Backend, SqlStore
from httk.store.export import _add_definition_candidate


def _store(path: Path) -> None:
    with Backend.sqlite(path) as database:
        store = SqlStore(
            database,
            entry_records={FileEntry: FileRecord},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        store.save(FileRecord("file:///presentation", "presentation", size=3, sha256="abc" + "0" * 61))


def test_export_contains_store_manifest_and_every_definition(tmp_path: Path) -> None:
    source = tmp_path / "presentation.sqlite"
    output = tmp_path / "presentation.zip"
    _store(source)
    export_dataset(source, output)

    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        snapshot = archive.read("store/presentation.sqlite")
        assert hashlib.sha256(snapshot).hexdigest() == manifest["store"]["sha256"]
        assert manifest["store"]["snapshot_consistent"] is True
        assert manifest["store"]["snapshot_method"] == "sqlite-backup"
        names = set(archive.namelist())
        expected_ids = {
            "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/files",
            "https://schemas.optimade.org/defs/v1.2/properties/core/id",
            "https://schemas.optimade.org/defs/v1.2/properties/core/immutable_id",
            "https://schemas.optimade.org/defs/v1.2/properties/core/last_modified",
            "https://schemas.optimade.org/defs/v1.2/properties/core/type",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/atime",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/checksums",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/ctime",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/description",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/media_type",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/modification_timestamp",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/mtime",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/name",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/size",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/url",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/url_stable_until",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/version",
        }
        # The store also declares the four standard-link target families: their entry types and
        # embedded properties are exported too.
        for entry_type in (
            "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/references",
            "https://schemas.httk.org/defs/v0.1/entrytypes/notes",
            "https://schemas.httk.org/defs/v0.1/entrytypes/tags",
            "https://schemas.httk.org/defs/v0.1/entrytypes/external_database_entries",
        ):
            expected_ids.add(entry_type)
            expected_ids.update(
                prop.definition_id for prop in load_entry_type_definition(entry_type).properties.values()
            )
        assert {definition["id"] for definition in manifest["definitions"]} == expected_ids
        for definition in manifest["definitions"]:
            assert definition["path"] in names
            assert hashlib.sha256(archive.read(definition["path"])).hexdigest() == definition["sha256"]
        families = {item["family"]: item["records"] for item in manifest["entry_record_declaration"]}
        assert families["files"] == ["core-file"]
        assert set(families) == {"external_database_entries", "files", "notes", "references", "tags"}


def test_export_rejects_aliases_and_protected_project_files(tmp_path: Path) -> None:
    source = tmp_path / "presentation.sqlite"
    _store(source)
    with pytest.raises(ValueError, match="overwrite the source"):
        export_dataset(source, source)

    project = tmp_path / "project"
    initialize_project(project, name="export")
    project_store = project / "presentation.sqlite"
    _store(project_store)
    with pytest.raises(ValueError, match="protected project data"):
        export_dataset(project_store, project / "httk_project" / "project.json")


def test_export_refuses_duckdb_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "store.duckdb"
    source.write_bytes(b"not a database")
    with pytest.raises(ValueError, match="DuckDB dataset export is refused"):
        export_dataset(source, tmp_path / "store.zip")


def test_export_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "presentation.sqlite"
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _store(source)
    export_dataset(source, first)
    export_dataset(source, second)
    assert first.read_bytes() == second.read_bytes()


def test_structure_export_contains_extended_family_definition(tmp_path: Path) -> None:
    atomistic = pytest.importorskip("httk.atomistic")
    source = tmp_path / "structures.sqlite"
    output = tmp_path / "structures.zip"
    from httk.atomistic import Cell, Sites, Species, StructureEntry, UnitcellStructure, UnitcellStructureRecord

    structure = UnitcellStructure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
        Sites([[0, 0, 0]]),
        (Species("Na", ("Na",), (1,)),),
        ("Na",),
    )
    with Backend.sqlite(source) as database:
        store = SqlStore(
            database,
            entry_records={StructureEntry: UnitcellStructureRecord},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        store.save(structure)

    export_dataset(source, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        definitions = {item["id"]: json.loads(archive.read(item["path"])) for item in manifest["definitions"]}
    entry = definitions[StructureEntry.definition_id]
    property_names = set(entry["properties"])
    assert len(property_names) == 43  # includes _httk_natoms, _httk_nformula_units and _httk_bonds
    assert {"_httk_basis_precision", "_httk_charge", "_httk_site_moments"} <= property_names
    assert (
        atomistic.StructureEntry.entry_type_definition().as_optimade()["properties"].keys()
        == entry["properties"].keys()
    )


def test_records_export_contains_typed_backing_property_definitions(tmp_path: Path) -> None:
    from httk.core.data_records import (
        RECORDS_DEFINITION_ID,
        TOTAL_ENERGY_DEFINITION_ID,
        DataRecord,
        DataRecordEntry,
        TotalEnergyRecord,
    )

    source = tmp_path / "records.sqlite"
    output = tmp_path / "records.zip"
    with Backend.sqlite(source) as database:
        store = SqlStore(
            database,
            entry_records={DataRecordEntry: (DataRecord, TotalEnergyRecord)},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        store.save(TotalEnergyRecord(-1.5))

    export_dataset(source, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        definitions = {item["id"]: json.loads(archive.read(item["path"])) for item in manifest["definitions"]}
    assert "total_energy" not in definitions[RECORDS_DEFINITION_ID]["properties"]
    assert definitions[RECORDS_DEFINITION_ID]["properties"]["_httk_total_energy"]["$id"] == TOTAL_ENERGY_DEFINITION_ID
    assert definitions[TOTAL_ENERGY_DEFINITION_ID]["x-optimade-unit"] == "eV"


_ID = "https://schemas.optimade.org/defs/v1.2/properties/core/id"


def test_export_merges_property_copies_that_differ_only_in_requirements(tmp_path: Path) -> None:
    source = tmp_path / "runs.sqlite"
    output = tmp_path / "runs.zip"
    with Backend.sqlite(source) as database:
        # The runs entry type embeds `id` as sortable, the OPTIMADE files entry type as not sortable.
        store = SqlStore(
            database, entry_records={RunEntry: Run, FileEntry: FileRecord}, entry_ids=EntryIdScheme("httk.test", "1")
        )
        store.save(Run(source_id="ws:job"))

    export_dataset(source, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        definitions = {item["id"]: json.loads(archive.read(item["path"])) for item in manifest["definitions"]}
    assert definitions[_ID]["$id"] == _ID
    assert "x-optimade-requirements" not in definitions[_ID]
    # Entry types keep their per-property requirements.
    runs = definitions["https://schemas.httk.org/defs/v0.1/entrytypes/runs"]
    files = definitions["https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/files"]
    assert runs["properties"]["id"]["x-optimade-requirements"]["sortable"] is True
    assert files["properties"]["id"]["x-optimade-requirements"]["sortable"] is False


def test_property_copies_differing_beyond_requirements_conflict() -> None:
    candidates: dict[tuple[str, str], dict[bytes, str]] = {}
    document = {"$id": _ID, "description": "An id.", "x-optimade-requirements": {"sortable": True}}
    _add_definition_candidate(candidates, "property", _ID, document, "first")
    _add_definition_candidate(
        candidates, "property", _ID, {**document, "x-optimade-requirements": {"sortable": False}}, "second"
    )
    assert document["x-optimade-requirements"] == {"sortable": True}  # the caller's document is not mutated
    with pytest.raises(ValueError, match="conflicting property definitions .*third"):
        _add_definition_candidate(candidates, "property", _ID, {**document, "description": "Another id."}, "third")
    # Entry-type documents are compared whole, requirements included.
    _add_definition_candidate(candidates, "entry-type", "urn:e", {"properties": {"id": {"sortable": True}}}, "a")
    with pytest.raises(ValueError, match="conflicting entry-type definitions"):
        _add_definition_candidate(candidates, "entry-type", "urn:e", {"properties": {"id": {"sortable": False}}}, "b")
