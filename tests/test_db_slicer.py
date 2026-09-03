"""Tests for the pandas-style slicer indexing layer (httk.store.query.slicer).

Each assertion compares the slicer's output against the equivalent hand-built
searcher query, so the slicer is verified to be exactly a compiler over the
existing variable/field/expression/results machinery and nothing more.
"""

from dataclasses import dataclass

import pytest
from clickhouse_read_support import CLICKHOUSE_PARAM, bulk_store
from postgres_support import POSTGRES_PARAM, postgres_database

from httk.store import FederatedStore
from httk.store.backend.sql import Backend, SqlStore
from httk.store.query.slicer import SlicerMask

pytestmark = pytest.mark.xdist_group("clickhouse_read_corpus")


@dataclass(frozen=True)
class Note:
    """A small record with two comparable ints, a title, and a nullable int."""

    title: str
    value: int
    other: int
    weight: int | None = None


NOTES = [
    Note("alpha", 5, 1, 10),
    Note("beta", 15, 2, None),
    Note("gamma", 20, 5, 30),
    Note("dummy", 25, 0, None),
    Note("delta", 12, 4, 40),
]


@pytest.fixture(scope="module", params=["sqlite", "duckdb", CLICKHOUSE_PARAM, POSTGRES_PARAM])
def store(request):
    """A populated store per supported dialect (duckdb skips where not installed)."""
    if request.param == "clickhousedb":
        with bulk_store(tuple(NOTES)) as sql_store:
            yield sql_store
        return
    if request.param == "postgresql":
        with postgres_database() as database:
            sql_store = SqlStore(database, entry_records={})
            with sql_store.transaction():
                for note in NOTES:
                    sql_store.save(note)
            yield sql_store
        return
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        database_manager = Backend.duckdb()
    else:
        database_manager = Backend.sqlite()
    with database_manager as database:
        sql_store = SqlStore(database, entry_records={})
        with sql_store.transaction():
            for note in NOTES:
                sql_store.save(note)
        yield sql_store


def hand_records(store, build) -> set[Note]:
    """The Note instances a hand-built searcher matches for ``build(variable)``."""
    searcher = store.searcher()
    variable = searcher.variable(Note)
    if build is not None:
        searcher.add(build(variable))
    return set(searcher.results(record=variable).scalars("record"))


def hand_scalars(store, field_name: str) -> list:
    """The decoded values of one field across every record, hand-built."""
    searcher = store.searcher()
    variable = searcher.variable(Note)
    field = getattr(variable, field_name)
    return list(searcher.results(col=field).scalars("col"))


def slicer(store):
    """The slicer over Note for this store."""
    return store.searcher().slicer(Note)


# --------------------------------------------------------------- column projection


def test_column_yields_decoded_scalars(store):
    note = slicer(store)
    titles = list(note["title"])
    assert sorted(titles) == sorted(hand_scalars(store, "title"))
    assert sorted(titles) == sorted(n.title for n in NOTES)
    assert all(isinstance(title, str) for title in titles)  # plain scalars, not rows


def test_iterating_slicer_yields_all_records(store):
    note = slicer(store)
    assert set(note) == set(NOTES)
    assert len(note) == len(NOTES)


# --------------------------------------------------------------- boolean masks


def test_single_comparison_mask(store):
    note = slicer(store)
    got = set(note[note["value"] > 10])
    assert got == hand_records(store, lambda v: v.value > 10)
    assert all(isinstance(record, Note) for record in got)


def test_len_of_mask_matches_count(store):
    note = slicer(store)
    mask = note["value"] > 10
    searcher = store.searcher()
    variable = searcher.variable(Note)
    searcher.add(variable.value > 10)
    assert len(note[mask]) == searcher.count()


def test_compound_and_or_not(store):
    note = slicer(store)
    mask = (note["value"] > 10) & (note["title"] != "dummy") | (note["other"] < 3)
    got = set(note[mask])
    expected = hand_records(
        store, lambda v: (v.value > 10) & (v.title != "dummy") | (v.other < 3)
    )
    assert got == expected


def test_invert(store):
    note = slicer(store)
    got = set(note[~(note["value"] > 10)])
    assert got == hand_records(store, lambda v: ~(v.value > 10))


def test_xor(store):
    note = slicer(store)
    left = note["value"] > 10
    right = note["other"] < 3
    got = set(note[left ^ right])
    expected = hand_records(
        store, lambda v: ((v.value > 10) | (v.other < 3)) & ~((v.value > 10) & (v.other < 3))
    )
    assert got == expected


# --------------------------------------------------------------- helper predicates


def test_isin(store):
    note = slicer(store)
    got = set(note[note["title"].isin(["alpha", "gamma", "absent"])])
    assert got == hand_records(store, lambda v: v.title.is_in("alpha", "gamma", "absent"))


def test_isna_and_notna(store):
    note = slicer(store)
    assert set(note[note["weight"].isna()]) == hand_records(store, lambda v: v.weight == None)  # noqa: E711
    assert set(note[note["weight"].notna()]) == hand_records(store, lambda v: v.weight != None)  # noqa: E711


def test_between(store):
    note = slicer(store)
    got = set(note[note["value"].between(12, 20)])
    assert got == hand_records(store, lambda v: (v.value >= 12) & (v.value <= 20))


def test_str_contains(store):
    note = slicer(store)
    got = set(note[note["title"].str.contains("a")])
    assert got == hand_records(store, lambda v: v.title.contains("a"))
    assert set(note[note["title"].str.startswith("de")]) == hand_records(
        store, lambda v: v.title.startswith("de")
    )
    assert set(note[note["title"].str.endswith("a")]) == hand_records(store, lambda v: v.title.endswith("a"))


# --------------------------------------------------------------- independence / no leakage


def test_operations_do_not_accumulate_filters(store):
    note = slicer(store)
    _ = list(note[note["value"] > 10])  # a filtered op first
    assert sorted(note["title"]) == sorted(n.title for n in NOTES)  # then unfiltered, still all
    assert set(note) == set(NOTES)
    assert len(note) == len(NOTES)


def test_reusing_a_selection_does_not_mutate(store):
    note = slicer(store)
    selection = note[note["value"] > 10]
    first = set(selection)
    second = set(selection)  # re-iterating the same selection is stable
    assert first == second == hand_records(store, lambda v: v.value > 10)


# --------------------------------------------------------------- errors


def test_unknown_field_raises_attribute_error(store):
    note = slicer(store)
    with pytest.raises(AttributeError):
        list(note["nope"])


def test_unsupported_key_types_raise_type_error(store):
    note = slicer(store)
    with pytest.raises(TypeError):
        _ = note[123]
    with pytest.raises(TypeError):
        _ = note[["title", "value"]]


def test_mask_operators_require_sibling_mask(store):
    note = slicer(store)
    mask = note["value"] > 10
    with pytest.raises(TypeError):
        _ = mask & "not a mask"
    other_note = slicer(store)  # a mask from a different slicer
    with pytest.raises(TypeError):
        _ = mask | (other_note["value"] > 10)


def test_mask_is_not_iterable(store):
    note = slicer(store)
    mask = note["value"] > 10
    assert isinstance(mask, SlicerMask)
    assert not hasattr(mask, "__iter__")


# --------------------------------------------------------------- federated smoke


def test_federated_slicer_over_in_memory_sqlite():
    """The federated searcher exposes the same slicer surface over child stores."""

    def _store(database, records):
        store = SqlStore(database, entry_records={})
        with store.transaction():
            for record in records:
                store.save(record)
        return store

    with Backend.sqlite() as first_db, Backend.sqlite() as second_db:
        federation = FederatedStore(
            {"first": _store(first_db, NOTES[:3]), "second": _store(second_db, NOTES[3:])}
        )
        note = federation.searcher().slicer(Note)
        assert set(note) == set(NOTES)
        assert len(note) == len(NOTES)
        got = set(note[note["value"] > 10])
        hand = federation.searcher()
        variable = hand.variable(Note)
        hand.add(variable.value > 10)
        assert got == set(hand.results(record=variable).scalars("record"))
