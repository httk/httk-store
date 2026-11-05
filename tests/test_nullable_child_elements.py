"""Nullable scalar/codec child elements: ``tuple[T | None, ...]`` layouts and round trips."""

import hashlib
import json
import math
import typing
from dataclasses import dataclass
from typing import Annotated

import pytest
from httk.core.storage import IdentitySkip, content_id

from httk.store.backend.mongo import entry_provider as mongo_entry_provider
from httk.store.backend.mongo.documents import decode_record, encode_record
from httk.store.backend.mongo.mapping import validator_for
from httk.store.backend.schema import ChildTableSpec, ColumnSpec, SchemaError, resolve_schema
from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql import entry_provider as sql_entry_provider
from httk.store.storage_layout import _table_fingerprint


@dataclass(frozen=True)
class NullableElements:
    values: tuple[float | None, ...]
    counts: tuple[int | None, ...]
    names: tuple[str | None, ...]
    maybe: tuple[float | None, ...] | None = None


@dataclass(frozen=True)
class NullableNotes(NullableElements):
    """Adds an identity-excluded nullable child, which deferred bulk finalization rejects by shape."""

    notes: Annotated[tuple[float | None, ...], IdentitySkip()] = ()


@dataclass(frozen=True)
class PlainElements:
    values: tuple[float, ...]
    counts: tuple[int, ...]


@dataclass(frozen=True)
class Target:
    name: str


@dataclass(frozen=True)
class NullableTargets:
    targets: tuple[Target | None, ...]


RECORD = NullableElements(values=(1.5, None, -0.0, 0.1, None), counts=(None, 0, -3), names=("a", None, ""), maybe=())
NOTED = NullableNotes(RECORD.values, RECORD.counts, RECORD.names, RECORD.maybe, notes=(None, -0.0))


def _assert_same(found: typing.Any, expected: NullableElements) -> None:
    for field in expected.__dataclass_fields__:
        got, want = getattr(found, field), getattr(expected, field)
        assert type(got) is type(want), field
        if want is None:
            continue
        assert len(got) == len(want), field
        for left, right in zip(got, want, strict=True):
            assert type(left) is type(right), (field, left, right)
            if isinstance(right, float):
                # Exact, except that signed zero is not content: a reopened store reads -0.0 back as 0.0.
                assert (left + 0.0).hex() == (right + 0.0).hex(), (field, left, right)
            else:
                assert left == right, field


def test_nullable_element_columns_resolve_nullable() -> None:
    schema = resolve_schema(NullableElements)
    assert schema.field("values").child == ChildTableSpec(
        "nullable_elements_values",
        (ColumnSpec("values", "float", nullable=True),),
    )
    assert schema.field("counts").child == ChildTableSpec(
        "nullable_elements_counts", (ColumnSpec("counts", "int", nullable=True),)
    )
    assert schema.field("names").child == ChildTableSpec(
        "nullable_elements_names", (ColumnSpec("names", "str", nullable=True),)
    )
    maybe = schema.field("maybe")
    assert maybe.optional and maybe.codec_name == "float" and maybe.child is not None
    assert all(column.nullable for column in maybe.child.element_columns)


# The fields fingerprint of PlainElements with the single-column float layout (storage protocol "3").
_PLAIN_FIELDS_FINGERPRINT = "bdaec75cc625fb8133871c7924ab1111c408d87023a0cb828ad4694f123f4eac"


def test_non_nullable_element_layout_unchanged() -> None:
    schema = resolve_schema(PlainElements)
    assert schema.field("values").child == ChildTableSpec(
        "plain_elements_values",
        (ColumnSpec("values", "float", nullable=False),),
    )
    assert schema.field("counts").child == ChildTableSpec(
        "plain_elements_counts", (ColumnSpec("counts", "int", nullable=False),)
    )
    fields = json.dumps(_table_fingerprint(schema)["fields"], sort_keys=True)
    assert hashlib.sha256(fields.encode()).hexdigest() == _PLAIN_FIELDS_FINGERPRINT


def test_storable_element_union_with_none_rejected() -> None:
    with pytest.raises(SchemaError, match="storable-class elements cannot be None"):
        resolve_schema(NullableTargets)

    @dataclass(frozen=True)
    class MixedElements:
        values: tuple[int | str, ...]

    with pytest.raises(SchemaError, match="element type"):
        resolve_schema(MixedElements)


def test_content_id_distinguishes_none_elements() -> None:
    twin = NullableElements(**{field: getattr(RECORD, field) for field in RECORD.__dataclass_fields__})
    assert content_id(twin) == content_id(RECORD)
    assert content_id(NullableElements((None,), (), ())) != content_id(NullableElements((0.0,), (), ()))
    assert content_id(NullableElements((), (), (), maybe=None)) != content_id(NullableElements((), (), (), maybe=()))


def test_round_trip(store_factory) -> None:
    store = store_factory()
    sid = store.save(RECORD)
    absent = NullableElements(values=(None,), counts=(), names=(None,), maybe=None)
    absent_sid = store.save(absent)
    noted_sid = store.save(NOTED)
    # Saving the same content again reuses the row and runs the metadata
    # (IdentitySkip) comparison over the nullable ``notes`` elements.
    assert store.save(NOTED) == noted_sid
    assert store.save(RECORD) == sid
    for current in (store, store_factory.reopen(store)):
        _assert_same(current.fetch(NullableElements, sid), RECORD)
        _assert_same(current.fetch(NullableElements, sid, eager=True), RECORD)
        _assert_same(current.fetch(NullableElements, absent_sid), absent)
        _assert_same(current.fetch(NullableNotes, noted_sid), NOTED)


def test_bulk_round_trip(store_factory) -> None:
    store = store_factory()
    if not hasattr(store, "bulk_ingest"):
        pytest.skip("backend does not provide bulk_ingest")
    with store.bulk_ingest() as bulk:
        sid = bulk.save(RECORD)
    with store.bulk_ingest(finalize="parity") as bulk:
        noted_sid = bulk.save(NOTED)
        assert bulk.save(NOTED) == noted_sid
    reopened = store_factory.reopen(store)
    _assert_same(reopened.fetch(NullableElements, sid), RECORD)
    _assert_same(reopened.fetch(NullableNotes, noted_sid), NOTED)


@pytest.mark.xdist_group("clickhouse_read")
def test_clickhouse_round_trip() -> None:
    from clickhouse_read_support import bulk_store

    with bulk_store([RECORD]) as store:
        found = store.fetch_by_content_id(NullableElements, content_id(RECORD))
        assert found is not None
        _assert_same(found, RECORD)


def test_mongo_validator_allows_null_elements() -> None:
    fields = validator_for(resolve_schema(NullableElements))["$jsonSchema"]["properties"]["f"]["properties"]
    values = fields["values"]["items"]
    assert values["properties"] == {"values": {"bsonType": ["double", "null"]}}
    assert values["required"] == ["values"]
    assert fields["counts"]["items"]["properties"] == {"counts": {"bsonType": ["int", "long", "null"]}}
    plain = validator_for(resolve_schema(PlainElements))["$jsonSchema"]["properties"]["f"]["properties"]
    assert plain["values"]["items"]["properties"]["values"] == {"bsonType": "double"}
    assert plain["counts"]["items"]["properties"]["counts"] == {"bsonType": ["int", "long"]}


@pytest.mark.parametrize("module", [sql_entry_provider, mongo_entry_provider])
def test_served_json_keeps_none_elements(module: typing.Any) -> None:
    schema = resolve_schema(NullableElements)
    served = module._json_value(schema, schema.field("values"), RECORD.values)
    assert served == [1.5, None, 0.0, 0.1, None]
    assert math.copysign(1.0, served[2]) == 1.0  # signed zero is not content: served as stored


def test_mongo_document_round_trip_without_server() -> None:
    schema = resolve_schema(NullableNotes)
    projected = {field: getattr(NOTED, field) for field in NOTED.__dataclass_fields__}
    embedded = encode_record(schema, projected, NOTED, NullableNotes, lambda target, value: 0)
    assert embedded["values"][1] == {"values": None}
    assert embedded["counts"][0] == {"counts": None}
    _assert_same(decode_record(schema, {"f": embedded}, lambda target, sid: None), NOTED)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (PlainElements(values=(1.0, None), counts=()), r"PlainElements\.values\[1\] cannot be None"),  # type: ignore[arg-type]
        (PlainElements(values=(), counts=(1, None)), r"PlainElements\.counts\[1\] cannot be None"),  # type: ignore[arg-type]
    ],
)
def test_none_element_in_non_nullable_layout_rejected(record: PlainElements, message: str) -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        with pytest.raises(ValueError, match=message):
            store.save(record)
        sid = store.save(PlainElements(values=(1.0,), counts=(1,)))
        assert store.fetch(PlainElements, sid) == PlainElements(values=(1.0,), counts=(1,))
    schema = resolve_schema(PlainElements)
    projected = {field: getattr(record, field) for field in record.__dataclass_fields__}
    with pytest.raises(ValueError, match=message):
        encode_record(schema, projected, record, PlainElements, lambda target, value: 0)
