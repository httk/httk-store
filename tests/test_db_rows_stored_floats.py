"""Behavioral tests for the lazy row `_httk_stored_floats` companion-float hook."""

import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated

import pytest
from httk.core import FracVector, SurdScalar
from httk.core.storage import Shape

from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql import rows as rows_module
from httk.store.backend.sql.rows import ExpiredLazyRecordError


@dataclass(frozen=True)
class ShapeChildRecord:
    name: str
    reduced_coords: Annotated[FracVector, Shape(0, 3)]


@dataclass(frozen=True)
class SurdChildRecord:
    name: str
    basis: tuple[SurdScalar, ...]


@dataclass(frozen=True)
class OptionalSurdChildRecord:
    name: str
    site_moments: tuple[SurdScalar, ...] | None = None


@pytest.fixture
def store():
    with Backend.sqlite() as db:
        yield SqlStore(db, entry_records={})


def _shape_fixture() -> FracVector:
    rows = [
        [Fraction(1, 4), Fraction(-3, 8), Fraction(1, 2)],
        [Fraction(1, 3), Fraction(2, 5), Fraction(-1, 6)],
        [Fraction(0, 1), Fraction(5, 8), Fraction(-7, 9)],
        [Fraction(3, 7), Fraction(-1, 2), Fraction(1, 9)],
    ]
    return FracVector(rows)


def test_shape_child_returns_stored_floats_in_row_order(store):
    exact = _shape_fixture()
    sid = store.save(ShapeChildRecord("A", exact))
    store._clear_identity_caches()

    row = store.fetch_many(ShapeChildRecord, [sid], eager=False)[0]
    result = row._httk_stored_floats("reduced_coords")

    expected = [[float(v) for v in tensor_row] for tensor_row in exact.to_fractions()]
    assert result == expected


def test_six_row_shape_child_round_trips_in_order(store):
    rows = [[Fraction(i, 11), Fraction(-i, 13), Fraction(i, 17)] for i in range(1, 7)]
    exact = FracVector(rows)
    sid = store.save(ShapeChildRecord("A", exact))
    store._clear_identity_caches()

    row = store.fetch(ShapeChildRecord, sid)
    result = row._httk_stored_floats("reduced_coords")

    expected = [[float(v) for v in tensor_row] for tensor_row in exact.to_fractions()]
    assert result == expected


def test_surdscalar_child_returns_stored_floats(store):
    components = (SurdScalar(Fraction(1, 4)), SurdScalar.sqrt_of(2) / 2)
    sid = store.save(SurdChildRecord("A", components))
    store._clear_identity_caches()

    row = store.fetch(SurdChildRecord, sid)
    result = row._httk_stored_floats("basis")

    assert result == [[component.to_float()] for component in components]


def test_optional_child_present_and_absent(store):
    present_sid = store.save(OptionalSurdChildRecord("A", (SurdScalar(Fraction(1, 4)),)))
    absent_sid = store.save(OptionalSurdChildRecord("B", None))
    store._clear_identity_caches()

    present_row = store.fetch(OptionalSurdChildRecord, present_sid)
    absent_row = store.fetch(OptionalSurdChildRecord, absent_sid)

    assert present_row._httk_stored_floats("site_moments") == [[0.25]]
    assert absent_row._httk_stored_floats("site_moments") is None


def test_scalar_field_returns_none(store):
    exact = _shape_fixture()
    sid = store.save(ShapeChildRecord("A", exact))
    store._clear_identity_caches()

    row = store.fetch(ShapeChildRecord, sid)
    assert row._httk_stored_floats("name") is None


def test_unknown_field_returns_none(store):
    exact = _shape_fixture()
    sid = store.save(ShapeChildRecord("A", exact))
    store._clear_identity_caches()

    row = store.fetch(ShapeChildRecord, sid)
    assert row._httk_stored_floats("does_not_exist") is None


def test_replace_created_instance_has_no_chunk(store):
    exact = _shape_fixture()
    sid = store.save(ShapeChildRecord("A", exact))
    store._clear_identity_caches()

    row = store.fetch(ShapeChildRecord, sid)
    replaced = dataclasses.replace(row, name="B")
    assert replaced._httk_stored_floats("reduced_coords") is None


def test_stored_floats_bypasses_exact_decoder_for_shape_child(store, monkeypatch):
    exact = _shape_fixture()
    sid = store.save(ShapeChildRecord("A", exact))
    store._clear_identity_caches()

    def _boom(*args, **kwargs):
        raise AssertionError("exact decoder invoked")

    monkeypatch.setattr(rows_module, "decode_fracvector_exact", _boom)

    row = store.fetch(ShapeChildRecord, sid)
    result = row._httk_stored_floats("reduced_coords")
    expected = [[float(v) for v in tensor_row] for tensor_row in exact.to_fractions()]
    assert result == expected

    # The patch is live: reading the exact field now raises through the same call.
    with pytest.raises(AssertionError):
        _ = row.reduced_coords


def test_stored_floats_bypasses_exact_decoder_for_surdscalar_child(store, monkeypatch):
    components = (SurdScalar(Fraction(1, 4)), SurdScalar.sqrt_of(2) / 2)
    sid = store.save(SurdChildRecord("A", components))
    store._clear_identity_caches()

    original_codec_named = rows_module.codec_named

    def _boom(values):
        raise AssertionError("exact decoder invoked")

    def guarded_codec_named(name):
        codec = original_codec_named(name)
        if name == "surdscalar":
            return dataclasses.replace(codec, decode=_boom)
        return codec

    monkeypatch.setattr(rows_module, "codec_named", guarded_codec_named)

    row = store.fetch(SurdChildRecord, sid)
    result = row._httk_stored_floats("basis")
    assert result == [[component.to_float()] for component in components]

    # The patch is live: reading the exact field now raises through the same call.
    with pytest.raises(AssertionError):
        _ = row.basis


def test_stored_floats_raises_after_rollback(store):
    sid = store.save(SurdChildRecord("A", (SurdScalar(Fraction(1, 4)),)))
    store._clear_identity_caches()
    row = store.fetch(SurdChildRecord, sid)
    assert row.name == "A"
    with pytest.raises(RuntimeError, match="boom"), store.transaction():
        # The deferred child read executes on the rolled-back transaction's
        # connection, capturing that transaction's token for this field.
        assert row._httk_stored_floats("basis") == [[0.25]]
        raise RuntimeError("boom")
    with pytest.raises(ExpiredLazyRecordError):
        row._httk_stored_floats("basis")
