"""SQL physical names that collide only after PostgreSQL's 63-byte truncation, and reserved field names."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from httk.core.storage import StorageInfo

from httk.store import EntryFamilyDeclaration, EntryRecordDeclaration
from httk.store.backend.schema import _RESERVED_FIELD_NAMES
from httk.store.backend.sql.layout import normalize_entry_families, normalize_entry_records

_TABLE = "t" * 50
_SHARED = "x" * 20  # f"{_TABLE}_{_SHARED}" already spans 71 characters.


class LongNameFamily:
    """An application-owned family used only for layout validation."""


@dataclass(frozen=True)
class LongChildNames:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name=_TABLE)

    xxxxxxxxxxxxxxxxxxxxa: tuple[int, ...]
    xxxxxxxxxxxxxxxxxxxxb: tuple[int, ...]


@dataclass(frozen=True)
class LongColumnNames:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="long_column_names")

    yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyya: int
    yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyb: int


def _declare(record: type) -> tuple[EntryFamilyDeclaration, ...]:
    return (
        EntryFamilyDeclaration(
            name="test-long-name-family",
            family=LongNameFamily,
            records=(EntryRecordDeclaration(name="test-long-name-record", record=record),),
        ),
    )


def test_child_tables_sharing_a_63_byte_prefix_are_rejected():
    with pytest.raises(ValueError, match="63-byte identifier truncation") as excinfo:
        normalize_entry_families(_declare(LongChildNames))
    assert f"{_TABLE}_{_SHARED}a" in str(excinfo.value)
    assert f"{_TABLE}_{_SHARED}b" in str(excinfo.value)


def test_columns_sharing_a_63_byte_prefix_are_rejected():
    with pytest.raises(ValueError, match="columns of table 'long_column_names'"):
        normalize_entry_families(_declare(LongColumnNames))


def test_generated_core_and_analyse_kinds_validate_despite_long_names():
    analyse = pytest.importorskip("httk.analyse.property_records")
    from httk.core import property_records as core
    from httk.core.data_records import (
        AverageTotalEnergyRecord,
        DataRecord,
        DataRecordEntry,
        DerivedDataRecord,
        TotalEnergyRecord,
    )

    from httk.store.backend.schema import resolve_schema

    kinds = (
        DataRecord,
        DerivedDataRecord,
        TotalEnergyRecord,
        AverageTotalEnergyRecord,
        *core.RECORD_KINDS.values(),
        *analyse.RECORD_KINDS.values(),
        *analyse.DERIVED_RECORD_KINDS.values(),
    )
    normalize_entry_records({DataRecordEntry: kinds})
    child_names = [
        spec.child.table_name for kind in kinds for spec in resolve_schema(kind).fields if spec.child is not None
    ]
    # The check is exercised: some generated child tables exceed PostgreSQL's limit.
    assert any(len(name) > 63 for name in child_names)


def test_store_reserved_field_names_are_reserved_by_typed_record_layouts():
    """A typed-record definition must be rejected at layout time, not at store declaration."""
    from httk.core.typed_records import _RESERVED_FIELDS

    assert _RESERVED_FIELD_NAMES - _RESERVED_FIELDS == set()
