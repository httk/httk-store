"""Physical names are bounded to 63 characters where derived; exact duplicates are rejected; reserved field names."""

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

import pytest
from httk.core.storage import StorageInfo

from httk.store import EntryFamilyDeclaration, EntryRecordDeclaration
from httk.store.backend.schema import _RESERVED_FIELD_NAMES, SchemaError, _bounded_identifier, resolve_schema
from httk.store.backend.sql.layout import normalize_entry_families, normalize_entry_records
from httk.store.backend.sql.mapping import _stable_identifier, backing_dispatch_column_name

_TABLE = "t" * 50
_SHARED = "x" * 20  # f"{_TABLE}_{_SHARED}" already spans 71 characters.
_LONG_FIELD = "y" * 60  # its ``_exact`` column spans 66 characters.


class LongNameFamily:
    """An application-owned family used only for layout validation."""


@dataclass(frozen=True)
class LongChildNames:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name=_TABLE)

    xxxxxxxxxxxxxxxxxxxxa: tuple[int, ...]
    xxxxxxxxxxxxxxxxxxxxb: tuple[int, ...]


LongColumnNames = dataclasses.make_dataclass(
    "LongColumnNames", [(_LONG_FIELD + "a", float), (_LONG_FIELD + "b", float)], frozen=True
)


def _declare(record: type) -> tuple[EntryFamilyDeclaration, ...]:
    return (
        EntryFamilyDeclaration(
            name="test-long-name-family",
            family=LongNameFamily,
            records=(EntryRecordDeclaration(name="test-long-name-record", record=record),),
        ),
    )


def test_long_child_table_names_are_bounded_and_distinct():
    normalize_entry_families(_declare(LongChildNames))
    names = [spec.child.table_name for spec in resolve_schema(LongChildNames).fields if spec.child is not None]
    assert names == [_bounded_identifier(f"{_TABLE}_{_SHARED}a"), _bounded_identifier(f"{_TABLE}_{_SHARED}b")]
    assert len(set(names)) == 2
    assert all(len(name) == 63 for name in names)


def test_long_column_names_are_bounded_distinct_and_keep_their_suffix():
    normalize_entry_families(_declare(LongColumnNames))
    columns = [column.name for spec in resolve_schema(LongColumnNames).fields for column in spec.columns]
    assert columns[0] == _LONG_FIELD + "a"
    assert columns[1] == _bounded_identifier(_LONG_FIELD + "a", "_exact")
    assert len(set(columns)) == 4
    assert all(len(name) <= 63 for name in columns)
    assert all(name.endswith("_exact") for name in columns[1::2])


def test_field_name_longer_than_the_limit_is_rejected():
    record = dataclasses.make_dataclass("TooLong", [("z" * 64, int)], frozen=True)
    with pytest.raises(SchemaError, match="longer than 63 UTF-8 bytes"):
        resolve_schema(record)


def test_non_ascii_names_are_bounded_on_utf8_bytes():
    fits = "å" * 31 + "a"  # 63 bytes, 32 characters
    assert _bounded_identifier(fits) == fits
    record = dataclasses.make_dataclass(
        "NonAscii", [("label", str)], namespace={"__httk_storage__": StorageInfo(storage_name="å" * 40)}, frozen=True
    )
    name = resolve_schema(record).table_name  # 40 characters, 80 bytes
    assert len(name.encode("utf-8")) <= 63
    assert name == _bounded_identifier("å" * 40) != "å" * 40
    assert name.startswith("å" * 27) and len(name.removesuffix(name[-9:]).encode("utf-8")) == 54
    assert _bounded_identifier("å" * 40, "_sid").endswith("_sid")


def test_exact_duplicate_after_bounding_is_rejected():
    """A declared column equal to another column's hash-shortened name is refused, naming both sources."""
    shortened = _bounded_identifier(_LONG_FIELD, "_exact")
    record = dataclasses.make_dataclass("Clash", [(_LONG_FIELD, float), (shortened, int)], frozen=True)
    with pytest.raises(ValueError, match=f"Clash.{_LONG_FIELD} column 1 and .*Clash.{shortened} column 0"):
        normalize_entry_families(_declare(record))


def test_generated_core_and_analyse_kinds_are_bounded():
    analyse = pytest.importorskip("httk.analyse.property_records")
    from httk.core import property_records as core
    from httk.core.data_records import (
        AverageTotalEnergyRecord,
        DataRecord,
        DataRecordEntry,
        DerivedDataRecord,
        TotalEnergyRecord,
    )

    kinds = (
        DataRecord,
        DerivedDataRecord,
        TotalEnergyRecord,
        AverageTotalEnergyRecord,
        *core.RECORD_KINDS.values(),
        *analyse.RECORD_KINDS.values(),
        *analyse.DERIVED_RECORD_KINDS.values(),
    )
    layout = normalize_entry_records({DataRecordEntry: kinds})
    (family,) = layout.families
    stable = [_stable_identifier("backing", name) for name in family.record_names]
    dispatch_columns = [backing_dispatch_column_name(name) for name in family.record_names]
    assert all(len(name) <= 63 and name.endswith("_sid") for name in dispatch_columns)
    assert len(set(dispatch_columns)) == len(dispatch_columns)
    # The dispatch bound is exercised: some backing columns were over-long before it.
    assert any(len(body) + len("_sid") > 63 for body in stable)
    shortened = 0
    for kind in kinds:
        schema = resolve_schema(kind)
        names = [schema.table_name, *(link.table_name for link in schema.links)]
        for spec in schema.fields:
            names.extend(column.name for column in spec.columns)
            if spec.child is not None:
                names.append(spec.child.table_name)
                names.extend(column.name for column in spec.child.element_columns)
                composed = f"{schema.table_name}_{spec.field}"
                if len(composed) > 63:
                    assert spec.child.table_name == _bounded_identifier(composed) != composed
                    shortened += 1
        assert all(len(name) <= 63 for name in names), kind
    # The bound is exercised: several generated child tables were over-long before it.
    assert shortened > 0


def test_store_reserved_field_names_are_reserved_by_typed_record_layouts():
    """A typed-record definition must be rejected at layout time, not at store declaration."""
    from httk.core.typed_records import _RESERVED_FIELDS

    assert _RESERVED_FIELD_NAMES - _RESERVED_FIELDS == set()
