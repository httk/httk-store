"""Weak-link (lineage-level link collection) coverage for the MongoDB store backend.

Mirrors the SQL ``test_weak_links.py`` battery against a live replica-set server
(gated on ``HTTK_TEST_MONGODB_URI`` through the ``mongo_test_database`` fixture).
Two deliberate parity gaps versus SQL: Mongo has no degraded/bulk write profile
(so those refusal tests do not apply), and identity comparison against a *target
search variable* is not supported on this backend (a clear
``UnsupportedQueryError`` instead of the SQL join).
"""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from httk.core.storage import StorageInfo, WeakLink

from httk.store.backend.schema import SchemaError
from httk.store.query import UnsupportedQueryError


@dataclass(frozen=True)
class Project:
    name: str
    note: str = ""


@dataclass(frozen=True)
class Other:
    tag: str


@dataclass(frozen=True)
class Result:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(WeakLink("projects", Project),))

    label: str


@dataclass(frozen=True)
class Team:
    lead: str
    members: tuple[str, ...] = ()  # a variable-length child field (not chainable)


@dataclass(frozen=True)
class Owned:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(WeakLink("teams", Team),))

    label: str


_LINK_COLLECTION = "_httk_link_result__project__projects"


def _store(database):
    from httk.store.backend.mongo import MongoStore

    return MongoStore(database, entry_records={})


def _names(objects):
    return sorted(obj.name for obj in objects)


def _query_labels(searcher, variable):
    searcher.output(variable, "r")
    return sorted(row.values[0].label for row in searcher)


def _link_count(store):
    return store._database.database[_LINK_COLLECTION].count_documents({})


def _distinct_lineages(store):
    return store._database.database[_LINK_COLLECTION].distinct("logical_id")


# --------------------------------------------------------------------------- link/unlink idempotency


def test_link_is_idempotent_and_linked_dedups(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    result = Result("R1")
    store.save(result)

    store.link(result, "projects", p1)
    store.link(result, "projects", p1)  # idempotent: no second lineage
    store.link(result, "projects", p2)

    assert _names(store.linked(result, "projects")) == ["P1", "P2"]
    assert _link_count(store) == 2  # one lineage per pair


def test_unlink_then_relink_toggles_within_one_lineage(mongo_test_database):
    store = _store(mongo_test_database)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)

    store.link(result, "projects", p1)
    assert _names(store.linked(result, "projects")) == ["P1"]

    store.unlink(result, "projects", p1)
    assert store.linked(result, "projects") == ()
    store.unlink(result, "projects", p1)  # already retracted: no-op

    store.link(result, "projects", p1)  # revives the same lineage
    assert _names(store.linked(result, "projects")) == ["P1"]

    assert len(_distinct_lineages(store)) == 1  # toggling appended revisions, not a fresh lineage


def test_unlink_absent_pair_is_a_noop(mongo_test_database):
    store = _store(mongo_test_database)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    store.unlink(result, "projects", p1)  # never linked: no error, no rows
    assert store.linked(result, "projects") == ()


# --------------------------------------------------------------------------- weak = latest revision


def test_linked_returns_latest_target_revision(mongo_test_database):
    store = _store(mongo_test_database)
    p1 = Project("P1", "v1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)

    store.replace(p1, Project("P1", "v2"))
    (linked,) = store.linked(result, "projects")
    assert linked.note == "v2"


# --------------------------------------------------------------------------- dedup-save accumulates


def test_content_dedup_save_accumulates_links_without_new_rows(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)

    first_sid = store.save(Result("R1"), links={"projects": p1})
    second_sid = store.save(Result("R1"), links={"projects": p2})
    assert first_sid == second_sid  # deduplicated onto the same document

    result = store.fetch(Result, first_sid)
    assert _names(store.linked(result, "projects")) == ["P1", "P2"]


# --------------------------------------------------------------------------- linked() ordering


def test_linked_orders_by_first_link_and_is_stable_across_retract_relink(
    mongo_test_database,
):
    store = _store(mongo_test_database)
    p1, p2, p3 = Project("P1"), Project("P2"), Project("P3")
    for project in (p1, p2, p3):
        store.save(project)
    result = Result("R1")
    store.save(result)

    store.link(result, "projects", p2)
    store.link(result, "projects", p1)
    store.link(result, "projects", p3)
    assert [obj.name for obj in store.linked(result, "projects")] == ["P2", "P1", "P3"]

    store.unlink(result, "projects", p2)
    store.link(result, "projects", p2)  # lineage root unchanged: keeps its front position
    assert [obj.name for obj in store.linked(result, "projects")] == ["P2", "P1", "P3"]


# --------------------------------------------------------------------------- save/replace sugar + atomicity


def test_save_links_accepts_single_and_iterable_targets(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)

    single = store.fetch(Result, store.save(Result("single"), links={"projects": p1}))
    assert _names(store.linked(single, "projects")) == ["P1"]

    many = store.fetch(Result, store.save(Result("many"), links={"projects": [p1, p2]}))
    assert _names(store.linked(many, "projects")) == ["P1", "P2"]


def test_replace_links_adds_to_replacement_lineage(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    original = Result("R1")
    store.save(original)
    store.link(original, "projects", p1)

    store.replace(original, Result("R1b"), links={"projects": p2})
    result = store.fetch(Result, store.sid_of(original))
    assert _names(store.linked(result, "projects")) == ["P1", "P2"]


def test_save_links_is_atomic_on_a_failing_link(mongo_test_database):
    store = _store(mongo_test_database)
    p1 = Project("P1")
    store.save(p1)

    with pytest.raises(SchemaError):
        store.save(Result("R1"), links={"nonesuch": p1})
    # The save rolled back with the failing link: no Result document leaked.
    assert store._database.database["result"].count_documents({}) == 0


# --------------------------------------------------------------------------- error paths


def test_unknown_link_name_raises_schema_error(mongo_test_database):
    store = _store(mongo_test_database)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    with pytest.raises(SchemaError, match="no weak link named 'ghost'"):
        store.link(result, "ghost", p1)
    with pytest.raises(SchemaError, match="no weak link named 'ghost'"):
        store.linked(result, "ghost")


def test_wrong_target_type_raises_type_error(mongo_test_database):
    store = _store(mongo_test_database)
    result = Result("R1")
    store.save(result)
    other = Other("x")
    store.save(other)
    with pytest.raises(TypeError, match="expects a Project target"):
        store.link(result, "projects", other)


def test_link_of_unsaved_object_raises(mongo_test_database):
    store = _store(mongo_test_database)
    p1 = Project("P1")
    store.save(p1)
    with pytest.raises(ValueError, match="has not been stored"):
        store.link(Result("unsaved"), "projects", p1)
    result = Result("R1")
    store.save(result)
    with pytest.raises(ValueError, match="has not been stored"):
        store.link(result, "projects", Project("unsaved"))


# --------------------------------------------------------------------------- reopen / plain instances


def test_links_survive_a_store_reopen(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)
    store.link(result, "projects", p2)
    store.unlink(result, "projects", p1)

    reopened = _store(mongo_test_database)  # fresh handle over the same database
    fetched = reopened.fetch(Result, reopened.sid_of(Result("R1")))
    assert _names(reopened.linked(fetched, "projects")) == ["P2"]
    reopened.link(fetched, "projects", reopened.fetch(Project, reopened.sid_of(Project("P1"))))
    assert _names(reopened.linked(fetched, "projects")) == ["P1", "P2"]


def test_plain_instance_has_no_links_attribute_but_store_linked_works(
    mongo_test_database,
):
    store = _store(mongo_test_database)
    p1 = Project("P1")
    store.save(p1)
    store.save(Result("R1"))

    # A hand-constructed instance is not store-bound: no `.links` accessor.
    assert not hasattr(Result("R1"), "links")

    reopened = _store(mongo_test_database)  # dodge the save identity cache
    fetched = reopened.fetch(Result, reopened.sid_of(Result("R1")))
    reopened.link(fetched, "projects", reopened.fetch(Project, reopened.sid_of(Project("P1"))))
    assert _names(reopened.linked(fetched, "projects")) == ["P1"]


# =========================================================================== #
# Searcher DSL: no-unwind array predicates, field chaining, identity, sets,
# negations, multiplicity, as_of, only_latest, rejections.
# =========================================================================== #


def test_field_chaining_reflects_latest_target_revision(mongo_test_database):
    store = _store(mongo_test_database)
    p = Project("P", "v1")
    store.save(p)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p)

    def found(note):
        searcher = store.searcher()
        v = searcher.variable(Result)
        searcher.add(v.links.projects.note == note)
        return _query_labels(searcher, v)

    assert found("v1") == ["R"]
    store.replace(p, Project("P", "v2"))
    assert found("v1") == []  # weak: the superseded target revision no longer matches
    assert found("v2") == ["R"]


def test_field_chaining_ignores_retracted_links(mongo_test_database):
    store = _store(mongo_test_database)
    p = Project("P")
    store.save(p)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p)
    store.unlink(r, "projects", p)

    searcher = store.searcher()
    v = searcher.variable(Result)
    searcher.add(v.links.projects.name == "P")
    assert _query_labels(searcher, v) == []


def test_identity_equals_stored_object(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    r1 = Result("R1")
    store.save(r1)
    store.link(r1, "projects", p1)
    r2 = Result("R2")
    store.save(r2)
    store.link(r2, "projects", p2)

    searcher = store.searcher()
    v = searcher.variable(Result)
    searcher.add(v.links.projects == p1)
    assert _query_labels(searcher, v) == ["R1"]


def test_identity_against_target_variable_is_unsupported(mongo_test_database):
    # Parity gap versus SQL: a target search variable RHS is not supported here.
    store = _store(mongo_test_database)
    searcher = store.searcher()
    v = searcher.variable(Result)
    pv = searcher.variable(Project)
    with pytest.raises(UnsupportedQueryError, match="target search variable"):
        _ = v.links.projects == pv


def test_has_any_and_has_only_including_vacuous_no_links(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    r1 = Result("R1")
    store.save(r1)
    store.link(r1, "projects", p1)
    r_both = Result("Rboth")
    store.save(r_both)
    store.link(r_both, "projects", p1)
    store.link(r_both, "projects", p2)
    r_none = Result("Rnone")
    store.save(r_none)

    def found(add):
        searcher = store.searcher()
        v = searcher.variable(Result)
        add(searcher, v)
        return _query_labels(searcher, v)

    assert found(lambda s, v: s.add(v.links.projects.has_any(p1))) == ["R1", "Rboth"]
    # has_only(p1): every live link is p1 — R1, plus vacuous Rnone; Rboth excluded.
    assert found(lambda s, v: s.add(v.links.projects.has_only(p1))) == ["R1", "Rnone"]
    assert found(lambda s, v: s.add(v.links.projects.has_only(p1, p2))) == [
        "R1",
        "Rboth",
        "Rnone",
    ]


def test_has_all_pattern_over_fresh_aliases(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    r_both = Result("Rboth")
    store.save(r_both)
    store.link(r_both, "projects", p1)
    store.link(r_both, "projects", p2)
    r_one = Result("Rone")
    store.save(r_one)
    store.link(r_one, "projects", p1)

    searcher = store.searcher()
    v = searcher.variable(Result)
    # Fresh link array per access: the two ANDed predicates constrain independent
    # link elements, so only a source linked to BOTH matches (HAS ALL).
    searcher.add((v.links.projects.name == "P1") & (v.links.projects.name == "P2"))
    assert _query_labels(searcher, v) == ["Rboth"]


def test_negations_are_set_wise_and_match_the_zero_links_row(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    r1 = Result("R1")
    store.save(r1)
    store.link(r1, "projects", p1)
    r2 = Result("R2")
    store.save(r2)
    store.link(r2, "projects", p2)
    r_none = Result("Rnone")
    store.save(r_none)

    def found(add):
        searcher = store.searcher()
        v = searcher.variable(Result)
        add(searcher, v)
        return _query_labels(searcher, v)

    assert found(lambda s, v: s.add(~v.links.projects.has_any(p1))) == ["R2", "Rnone"]
    assert found(lambda s, v: s.add(~(v.links.projects.name == "P1"))) == [
        "R2",
        "Rnone",
    ]


def test_string_rhs_raises_type_error(mongo_test_database):
    store = _store(mongo_test_database)
    r = Result("R")
    store.save(r)
    searcher = store.searcher()
    v = searcher.variable(Result)
    with pytest.raises(TypeError, match="chain a target field|stored Project"):
        _ = v.links.projects == "P1"


def test_source_with_two_live_links_appears_exactly_once(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p1)
    store.link(r, "projects", p2)

    searcher = store.searcher()
    v = searcher.variable(Result)
    searcher.add(v.links.projects.has_any(p1, p2))
    assert _query_labels(searcher, v) == ["R"]  # grouped: a single document, not one per link

    counting = store.searcher()
    cv = counting.variable(Result)
    counting.add(cv.links.projects.has_any(p1, p2))
    assert counting.count() == 1


def test_multi_revision_links_and_targets_resolve_to_latest(mongo_test_database):
    store = _store(mongo_test_database)
    p = Project("P", "v1")
    store.save(p)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p)
    store.unlink(r, "projects", p)
    store.link(r, "projects", p)  # link lineage: 3 revisions, latest live
    store.replace(p, Project("P", "v2"))  # target lineage: 2 revisions

    searcher = store.searcher()
    v = searcher.variable(Result)
    searcher.add(v.links.projects.note == "v2")
    assert _query_labels(searcher, v) == ["R"]

    counting = store.searcher()
    cv = counting.variable(Result)
    counting.add(cv.links.projects.note == "v2")
    assert counting.count() == 1  # one document despite the multi-revision link lineage


def test_searcher_as_of_time_travels_links_and_targets(mongo_test_database):
    store = _store(mongo_test_database)
    store._clock = lambda: 1_000_000
    p = Project("P", "v1")
    store.save(p)
    q = Project("Q", "q1")
    store.save(q)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p)  # link live as of 1M

    store._clock = lambda: 3_000_000
    store.replace(p, Project("P", "v2"))  # target revision after the cutoff
    store.link(r, "projects", q)  # link created after the cutoff

    def labels_asof(cutoff, add):
        searcher = store.searcher(as_of=cutoff)
        v = searcher.variable(Result)
        add(searcher, v)
        return _query_labels(searcher, v)

    # As of 2M: only P's v1 revision is linked; Q's link and P's v2 are future.
    assert labels_asof(2_000_000, lambda s, v: s.add(v.links.projects.note == "v1")) == ["R"]
    assert labels_asof(2_000_000, lambda s, v: s.add(v.links.projects.note == "v2")) == []
    assert labels_asof(2_000_000, lambda s, v: s.add(v.links.projects.name == "Q")) == []
    # As of 3M: Q's link is visible and P resolves to v2.
    assert labels_asof(3_000_000, lambda s, v: s.add(v.links.projects.name == "Q")) == ["R"]
    assert labels_asof(3_000_000, lambda s, v: s.add(v.links.projects.note == "v2")) == ["R"]


def test_searcher_as_of_sees_link_live_before_a_later_retraction(mongo_test_database):
    store = _store(mongo_test_database)
    store._clock = lambda: 1_000_000
    p = Project("P")
    store.save(p)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p)  # live at 1M
    store._clock = lambda: 3_000_000
    store.unlink(r, "projects", p)  # retracted at 3M

    past = store.searcher(as_of=2_000_000)
    pv = past.variable(Result)
    past.add(pv.links.projects.has_any(p))
    assert _query_labels(past, pv) == ["R"]  # retraction straddles the cutoff: still live

    now = store.searcher()
    nv = now.variable(Result)
    now.add(nv.links.projects.has_any(p))
    assert _query_labels(now, nv) == []  # now retracted


def test_only_latest_is_orthogonal_to_weak_link_traversal(mongo_test_database):
    store = _store(mongo_test_database)
    p = Project("P1")
    store.save(p)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p)
    store.replace(r, Result("R2"))  # supersede the source; the link binds the lineage

    both = store.searcher()
    bv = both.variable(Result)
    both.add(bv.links.projects.name == "P1")
    assert _query_labels(both, bv) == ["R", "R2"]
    counting = store.searcher()
    cv = counting.variable(Result)
    counting.add(cv.links.projects.name == "P1")
    assert counting.count() == 2

    latest = store.searcher(only_latest=True)
    lv = latest.variable(Result)
    latest.add(lv.links.projects.name == "P1")
    assert _query_labels(latest, lv) == ["R2"]
    latest_count = store.searcher(only_latest=True)
    lcv = latest_count.variable(Result)
    latest_count.add(lcv.links.projects.name == "P1")
    assert latest_count.count() == 1


def test_deep_chaining_into_target_non_scalar_is_rejected(mongo_test_database):
    store = _store(mongo_test_database)
    store.save(Owned("O"))  # creates Owned's parent + link collections
    searcher = store.searcher()
    v = searcher.variable(Owned)
    with pytest.raises(UnsupportedQueryError, match="scalar and encoded"):
        _ = v.links.teams.members  # a child field of the target
    with pytest.raises(UnsupportedQueryError, match="not supported"):
        _ = v.links.teams.links  # nested weak links of the target


def test_link_path_projection_is_rejected(mongo_test_database):
    store = _store(mongo_test_database)
    p = Project("P")
    store.save(p)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p)

    searcher = store.searcher()
    v = searcher.variable(Result)
    with pytest.raises(UnsupportedQueryError, match="weak-link path"):
        searcher.output(v.links.projects.name, "pname")
    with pytest.raises(UnsupportedQueryError, match="weak-link path"):
        searcher.results(pname=v.links.projects.name)


def test_link_path_sort_is_rejected(mongo_test_database):
    store = _store(mongo_test_database)
    p = Project("P")
    store.save(p)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p)

    searcher = store.searcher()
    v = searcher.variable(Result)
    with pytest.raises(UnsupportedQueryError, match="weak-link path"):
        searcher.add_sort(v.links.projects.name)


# --------------------------------------------------------------------------- .links accessor on fetched records


def test_links_accessor_on_a_freshly_fetched_record(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    r = Result("R")
    sid = store.save(r)
    store.link(r, "projects", p1)
    store.link(r, "projects", p2)

    reopened = _store(mongo_test_database)  # fresh handle: dodge the identity cache
    fetched = reopened.fetch(Result, sid)
    assert type(fetched) is not Result  # a store-bound thin subclass

    assert _names(fetched.links.projects) == ["P1", "P2"]
    # Memoized like reference fields: the same tuple object, no re-query.
    assert fetched.links.projects is fetched.links.projects
    assert "projects" in dir(fetched.links)
    with pytest.raises(AttributeError, match="no weak link named 'ghost'"):
        _ = fetched.links.ghost
    # A plain, unbound instance has no accessor at all.
    with pytest.raises(AttributeError):
        _ = Result("plain").links


def test_bound_instance_equals_plain_instance(mongo_test_database):
    store = _store(mongo_test_database)
    r = Result("R")
    sid = store.save(r)
    reopened = _store(mongo_test_database)
    fetched = reopened.fetch(Result, sid)
    assert type(fetched) is not Result
    # Dataclass equality/hash treat a bound record and its plain twin as equal.
    assert fetched == Result("R")
    assert Result("R") == fetched
    assert hash(fetched) == hash(Result("R"))
    assert fetched != Result("other")


# --------------------------------------------------------------------------- fsck


def _violations_only_dup(summary):
    for violation in summary.violations:
        assert "non-corrupting state that fsck repair does not deduplicate" in violation, violation


def test_fsck_is_clean_for_healthy_links(mongo_test_database):
    store = _store(mongo_test_database)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)
    store.link(result, "projects", p2)
    store.unlink(result, "projects", p2)
    summary = store.fsck(known_types=(Result, Project))
    assert summary.violations == ()


def test_fsck_reports_dangling_endpoint_and_lineage_integrity(mongo_test_database):
    store = _store(mongo_test_database)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)

    # A dangling target_lid, deliberately with logical_id != its own sid so both
    # the dangling and the lineage-integrity checks fire.
    store._database.database[_LINK_COLLECTION].insert_one(
        {
            "_id": 424242,
            "logical_id": 424243,
            "source_lid": store.sid_of(result),
            "target_lid": 99999,
            "retracted": 0,
            "store_timestamp": 1,
        }
    )
    summary = store.fsck(known_types=(Result, Project), repair=False, collect_garbage=False)
    assert any("target_lid 99999 matches no" in violation for violation in summary.violations)
    assert any("does not equal its founder sid" in violation for violation in summary.violations)


def test_fsck_reports_duplicate_pair_as_repairable_note(mongo_test_database):
    store = _store(mongo_test_database)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)

    source_lid = store.sid_of(result)
    target_lid = store.sid_of(p1)
    # A second live lineage for the same pair (a tolerated concurrency outcome),
    # founded correctly: its logical_id equals its own sid.
    store._database.database[_LINK_COLLECTION].insert_one(
        {
            "_id": 777777,
            "logical_id": 777777,
            "source_lid": source_lid,
            "target_lid": target_lid,
            "retracted": 0,
            "store_timestamp": 1,
        }
    )
    summary = store.fsck(known_types=(Result, Project), repair=False, collect_garbage=False)
    notes = [
        violation
        for violation in summary.violations
        if "non-corrupting state that fsck repair does not deduplicate" in violation
    ]
    assert len(notes) == 1
    link_counter = summary.collections.get(_LINK_COLLECTION)
    assert link_counter is None or link_counter.conflicts == 0  # a tolerated duplicate is not corruption
    assert _names(store.linked(result, "projects")) == ["P1"]


def test_reopen_mark_reflects_a_link_as_the_newest_event(mongo_test_database):
    store = _store(mongo_test_database)
    store._clock = lambda: 1_000_000
    p = Project("P")
    store.save(p)
    result = Result("R1")
    store.save(result)
    store._clock = lambda: 5_000_000
    store.link(result, "projects", p)  # the newest store event is a link revision

    link_ts = store._database.database[_LINK_COLLECTION].find_one({}, {"store_timestamp": 1})["store_timestamp"]
    reopened = _store(mongo_test_database)  # fresh handle recomputes the mark from present collections
    # The link's timestamp (newer than any record) advances the reopen clock-regression mark.
    assert reopened._store_timestamp_mark == link_ts
