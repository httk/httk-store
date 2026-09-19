"""Coverage for the compact decorated-record SQL store declaration."""

from dataclasses import dataclass
from typing import Annotated

import pytest
from httk.atomistic import UnitcellStructureRecord
from httk.core import DataEntryRecord, Property, entry_record, register_entry_record

from httk.store import EntryIdScheme, SqliteStore
from httk.store.backend.sql import optimade_filter_searcher


@entry_record("test.result")
class Result(DataEntryRecord):
    formation_energy: Annotated[float, Property(description="Formation energy", unit="eV")]


@entry_record("test.result_with_structure")
class ResultWithStructure(DataEntryRecord):
    formation_energy: Annotated[float, Property(description="Formation energy", unit="eV")]
    structure: UnitcellStructureRecord


@entry_record("test.alpha")
class Alpha(DataEntryRecord):
    value: Annotated[float, Property(description="Alpha value")]


@entry_record("test.beta")
class Beta(DataEntryRecord):
    score: Annotated[int, Property(description="Beta score")]


@dataclass(frozen=True)
class PrivateEnvelope:
    structure: UnitcellStructureRecord


@dataclass(frozen=True)
class UnservedRegistered:
    payload: UnitcellStructureRecord


register_entry_record(
    name="test-unserved-registered",
    record=f"{__name__}:UnservedRegistered",
    definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations",
)


@entry_record("test.enveloped")
class Enveloped(DataEntryRecord):
    payload: PrivateEnvelope


@entry_record("test.unserved_envelope")
class UnservedEnvelope(DataEntryRecord):
    payload: UnservedRegistered


@entry_record("test.alpha")
class DuplicateName(DataEntryRecord):
    other: Annotated[str, Property(description="Duplicate name")]


def test_records_declaration_saves_reopens_and_discovers_properties(tmp_path) -> None:
    path = tmp_path / "results.sqlite"
    store = SqliteStore(path, records=[Result], entry_ids=EntryIdScheme("test", "1"))
    sid = store.save(Result(formation_energy=-1.25))
    store.close()

    reopened = SqliteStore(path, records=[Result])
    result = reopened.fetch(Result, sid)
    assert result.formation_energy == -1.25
    assert reopened.layout.declaration == {"__httk_records": ("test.result",)}
    matches = optimade_filter_searcher(reopened, Result, "_httk_custom_formation_energy < 0").results()
    assert [row[0].formation_energy for row in matches] == [-1.25]
    reopened.close()


def test_records_declaration_includes_registered_references() -> None:
    store = SqliteStore(None, records=[ResultWithStructure], entry_ids=EntryIdScheme("test", "1"))
    assert store.layout.declaration == {
        "__httk_records": ("test.result_with_structure",),
        "structures": ("atomistic-unitcell-structure",),
    }
    store.close()


def test_records_declaration_traverses_private_nested_records() -> None:
    store = SqliteStore(None, records=[Enveloped], entry_ids=EntryIdScheme("test", "1"))
    assert "structures" in store.layout.declaration
    store.close()


def test_records_declaration_traverses_registered_records_without_families() -> None:
    store = SqliteStore(None, records=[UnservedEnvelope], entry_ids=EntryIdScheme("test", "1"))
    assert "structures" in store.layout.declaration
    assert "test-unserved-registered" not in store.layout.declaration
    store.close()


def test_multiple_decorated_records_keep_one_merged_definition(tmp_path) -> None:
    store = SqliteStore(tmp_path / "multiple.sqlite", records=[Alpha, Beta], entry_ids=EntryIdScheme("test", "1"))
    family = next(layout.family for layout in store.entry_layout if layout.name == "__httk_records")
    definition = family.entry_type_definition()
    assert {"_httk_custom_value", "_httk_custom_score"} <= set(definition.properties)
    layout = next(layout for layout in store.entry_layout if layout.name == "__httk_records")
    assert {record.__httk_entry_name__ for record in layout.records} == {"test.alpha", "test.beta"}
    alpha_sid = store.save(Alpha(value=-1.0))
    beta_sid = store.save(Beta(score=7))
    store.close()
    reopened = SqliteStore(tmp_path / "multiple.sqlite", records=[Alpha, Beta])
    assert reopened.fetch(Alpha, alpha_sid).value == -1.0
    assert reopened.fetch(Beta, beta_sid).score == 7
    reopened.close()


def test_multiple_decorated_records_reject_duplicate_stable_names() -> None:
    with pytest.raises(ValueError, match="record name"):
        SqliteStore(None, records=[Alpha, DuplicateName])


def test_records_declaration_is_mutually_exclusive_with_explicit_layout() -> None:
    with pytest.raises(TypeError, match="mutually exclusive"):
        SqliteStore(None, records=[Result], entry_records={})
