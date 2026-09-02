"""Weak-link (lineage-level link table) coverage for the SQL store backend.

Runs over the parametrized ``store_factory`` fixture's SQL arms (sqlite/duckdb,
and postgresql when configured); the mongo arm is skipped until P3 adds Mongo
parity, mirroring how the SQL-only bulk suite excludes it.
"""

import threading
from dataclasses import dataclass
from typing import ClassVar

import pytest
import sqlalchemy
from httk.core.storage import StorageInfo, WeakLink

from httk.store.backend.schema import SchemaError
from httk.store.backend.sql import Backend, SqlStore


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


def _sql_store(store_factory, **kwargs):
    """Build a store, skipping the mongo arm (the weak-link store API is SQL-only until P3)."""
    store = store_factory(**kwargs)
    if not isinstance(store, SqlStore):
        pytest.skip("weak-link store API is SQL-only until P3 (Mongo parity)")
    return store


def _names(objects):
    return sorted(obj.name for obj in objects)


def _fsck_clean_or_dup(summary):
    """Assert an fsck summary carries no violations except tolerated duplicate-pair notes."""
    for violation in summary.violations:
        assert "repairable, non-corrupting note" in violation, violation


# --------------------------------------------------------------------------- link/unlink idempotency


def test_link_is_idempotent_and_linked_dedups(store_factory):
    store = _sql_store(store_factory)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    result = Result("R1")
    store.save(result)

    store.link(result, "projects", p1)
    store.link(result, "projects", p1)  # idempotent: no second lineage
    store.link(result, "projects", p2)

    assert _names(store.linked(result, "projects")) == ["P1", "P2"]
    # Idempotency means one lineage per pair: exactly two live link rows.
    table = store._table("_httk_link_result__project__projects")
    with store._read_connection() as connection:
        count = connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(table)).scalar_one()
    assert count == 2


def test_unlink_then_relink_toggles_within_one_lineage(store_factory):
    store = _sql_store(store_factory)
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

    # Toggling appended revisions, never a fresh lineage: still one lineage.
    table = store._table("_httk_link_result__project__projects")
    with store._read_connection() as connection:
        lineages = connection.execute(sqlalchemy.select(table.c.logical_id).distinct()).scalars().all()
    assert len(lineages) == 1


def test_unlink_absent_pair_is_a_noop(store_factory):
    store = _sql_store(store_factory)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    store.unlink(result, "projects", p1)  # never linked: no error, no rows
    assert store.linked(result, "projects") == ()


# --------------------------------------------------------------------------- weak = latest revision


def test_linked_returns_latest_target_revision(store_factory):
    store = _sql_store(store_factory)
    p1 = Project("P1", "v1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)

    store.replace(p1, Project("P1", "v2"))
    (linked,) = store.linked(result, "projects")
    assert linked.note == "v2"


# --------------------------------------------------------------------------- dedup-save accumulates


def test_content_dedup_save_accumulates_links_without_new_rows(store_factory):
    store = _sql_store(store_factory)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)

    # Two "projects" contexts save identical Result content; content-id dedup
    # collapses onto one row, but each link add accumulates an association.
    first_sid = store.save(Result("R1"), links={"projects": p1})
    second_sid = store.save(Result("R1"), links={"projects": p2})
    assert first_sid == second_sid  # deduplicated onto the same row

    result = store.fetch(Result, first_sid)
    assert _names(store.linked(result, "projects")) == ["P1", "P2"]


# --------------------------------------------------------------------------- linked() ordering + laziness


def test_linked_orders_by_first_link_and_is_stable_across_retract_relink(store_factory):
    store = _sql_store(store_factory)
    p1, p2, p3 = Project("P1"), Project("P2"), Project("P3")
    for project in (p1, p2, p3):
        store.save(project)
    result = Result("R1")
    store.save(result)

    store.link(result, "projects", p2)
    store.link(result, "projects", p1)
    store.link(result, "projects", p3)
    assert [obj.name for obj in store.linked(result, "projects")] == ["P2", "P1", "P3"]

    # Retract then relink P2: its lineage root logical_id is unchanged, so it
    # keeps its first-link position at the front.
    store.unlink(result, "projects", p2)
    store.link(result, "projects", p2)
    assert [obj.name for obj in store.linked(result, "projects")] == ["P2", "P1", "P3"]


def test_linked_is_lazy_by_default_and_eager_on_request(store_factory):
    store = _sql_store(store_factory)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)

    # Drop the identity cache so the target is not returned as the already
    # materialized saved instance; linked()'s default then hands back a lazy row.
    store._clear_identity_caches()
    (lazy,) = store.linked(result, "projects")
    assert type(lazy) is not Project  # a lazy row proxy, not the materialized dataclass
    assert lazy.name == "P1"

    store._clear_identity_caches()
    (eager,) = store.linked(result, "projects", eager=True)
    assert type(eager) is Project


# --------------------------------------------------------------------------- save/replace sugar + atomicity


def test_save_links_accepts_single_and_iterable_targets(store_factory):
    store = _sql_store(store_factory)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)

    single = store.fetch(Result, store.save(Result("single"), links={"projects": p1}))
    assert _names(store.linked(single, "projects")) == ["P1"]

    many = store.fetch(Result, store.save(Result("many"), links={"projects": [p1, p2]}))
    assert _names(store.linked(many, "projects")) == ["P1", "P2"]


def test_replace_links_adds_to_replacement_lineage(store_factory):
    store = _sql_store(store_factory)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    original = Result("R1")
    store.save(original)
    store.link(original, "projects", p1)

    store.replace(original, Result("R1b"), links={"projects": p2})
    result = store.fetch(Result, store.sid_of(original))
    # The link binds the lineage, so both endpoints resolve latest: P1 (from the
    # original revision) and P2 (added on replace) are both live.
    assert _names(store.linked(result, "projects")) == ["P1", "P2"]


def test_save_links_is_atomic_on_a_failing_link(store_factory):
    store = _sql_store(store_factory)
    p1 = Project("P1")
    store.save(p1)

    before = _row_count(store, "result")
    with pytest.raises(SchemaError):
        store.save(Result("R1"), links={"nonesuch": p1})
    # The save rolled back with the failing link: no Result row leaked.
    assert _row_count(store, "result") == before


def _row_count(store, table_name):
    from httk.store.backend.sql.layout import actual_table_names

    with store._read_connection() as connection:
        if table_name not in actual_table_names(connection):
            return 0
        table = store._table(table_name)
        return connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(table)).scalar_one()


# --------------------------------------------------------------------------- error paths


def test_unknown_link_name_raises_schema_error(store_factory):
    store = _sql_store(store_factory)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    with pytest.raises(SchemaError, match="no weak link named 'ghost'"):
        store.link(result, "ghost", p1)
    with pytest.raises(SchemaError, match="no weak link named 'ghost'"):
        store.linked(result, "ghost")


def test_wrong_target_type_raises_type_error(store_factory):
    store = _sql_store(store_factory)
    result = Result("R1")
    store.save(result)
    other = Other("x")
    store.save(other)
    with pytest.raises(TypeError, match="expects a Project target"):
        store.link(result, "projects", other)


def test_link_of_unsaved_object_raises(store_factory):
    store = _sql_store(store_factory)
    p1 = Project("P1")
    store.save(p1)
    with pytest.raises(ValueError, match="has not been stored"):
        store.link(Result("unsaved"), "projects", p1)
    result = Result("R1")
    store.save(result)
    with pytest.raises(ValueError, match="has not been stored"):
        store.link(result, "projects", Project("unsaved"))


# --------------------------------------------------------------------------- profile / fence refusals


def test_degraded_profile_refuses_link():
    with Backend.sqlite(degraded=True) as database:
        store = SqlStore(database, entry_records={})
        p1 = Project("P1")
        store.save(p1)
        result = Result("R1")
        store.save(result)
        with pytest.raises(RuntimeError, match="degraded"):
            store.link(result, "projects", p1)
        with pytest.raises(RuntimeError, match="degraded"):
            store.unlink(result, "projects", p1)
        with pytest.raises(RuntimeError, match="degraded"):
            store.save(Result("R2"), links={"projects": p1})


def test_bulk_context_fences_link(store_factory):
    store = _sql_store(store_factory)
    if not hasattr(store, "bulk_ingest"):
        pytest.skip("backend does not provide bulk_ingest")
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    with store.bulk_ingest() as _bulk, pytest.raises(RuntimeError, match="bulk_ingest"):
        store.link(result, "projects", p1)


# --------------------------------------------------------------------------- concurrency (SQLite)


def test_two_thread_same_pair_link_race_on_sqlite():
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        p1 = Project("P1")
        store.save(p1)
        result = Result("R1")
        store.save(result)

        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def worker():
            try:
                barrier.wait()
                store.link(result, "projects", p1)
            except BaseException as error:  # noqa: BLE001 - recorded and re-asserted
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors
        # Store ends live, and fsck is clean or carries only the tolerated
        # duplicate-pair note (never corruption).
        assert _names(store.linked(result, "projects")) == ["P1"]
        _fsck_clean_or_dup(store.fsck(known_types=(Result, Project), exclusive=True))


# --------------------------------------------------------------------------- save-then-fetch / plain instances


def test_plain_instance_has_no_links_attribute_but_store_linked_works(store_factory):
    store = _sql_store(store_factory)
    p1 = Project("P1")
    store.save(p1)
    store.save(Result("R1"))

    # A hand-constructed instance is not store-bound: no `.links` accessor
    # (the instance accessor is P2b; P2a exposes only store.linked()).
    assert not hasattr(Result("R1"), "links")

    # After save-then-fetch, store.linked() works on the fetched (store-bound) row.
    fetched = store.fetch(Result, store.sid_of(Result("R1")))
    store.link(fetched, "projects", p1)
    assert _names(store.linked(fetched, "projects")) == ["P1"]


def test_links_survive_a_store_reopen(store_factory):
    store = _sql_store(store_factory)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)
    store.link(result, "projects", p2)
    store.unlink(result, "projects", p1)

    reopened = store_factory.reopen(store)  # would raise if link tables were flagged as intruders
    fetched = reopened.fetch(Result, reopened.sid_of(Result("R1")))
    assert _names(reopened.linked(fetched, "projects")) == ["P2"]
    # A relink after reopen still works and advanced the clock without regression complaints.
    reopened.link(fetched, "projects", reopened.fetch(Project, reopened.sid_of(Project("P1"))))
    assert _names(reopened.linked(fetched, "projects")) == ["P1", "P2"]


# --------------------------------------------------------------------------- fsck


def test_fsck_is_clean_for_healthy_links(store_factory):
    store = _sql_store(store_factory)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)
    store.link(result, "projects", p2)
    store.unlink(result, "projects", p2)
    summary = store.fsck(known_types=(Result, Project), exclusive=True)
    assert summary.violations == ()


def test_fsck_reports_dangling_endpoint_and_lineage_integrity(store_factory):
    store = _sql_store(store_factory)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)

    link_table = store._table("_httk_link_result__project__projects")
    with store._write_connection() as connection:
        # A dangling target_lid, deliberately with logical_id != its own sid so
        # both the dangling and the lineage-integrity checks fire.
        connection.execute(
            sqlalchemy.insert(link_table).values(
                {
                    "logical_id": 424242,
                    "source_lid": store.sid_of(result),
                    "target_lid": 99999,
                    "retracted": 0,
                    "store_timestamp": 1,
                }
            )
        )
    summary = store.fsck(known_types=(Result, Project), repair=False, collect_garbage=False, exclusive=True)
    assert any("target_lid 99999 matches no" in violation for violation in summary.violations)
    assert any("does not equal its founder sid" in violation for violation in summary.violations)


def test_fsck_reports_duplicate_pair_as_repairable_note(store_factory):
    store = _sql_store(store_factory)
    p1 = Project("P1")
    store.save(p1)
    result = Result("R1")
    store.save(result)
    store.link(result, "projects", p1)

    link_table = store._table("_httk_link_result__project__projects")
    source_lid = store.sid_of(result)
    target_lid = store.sid_of(p1)
    with store._write_connection() as connection:
        # A second live lineage for the same pair (a tolerated concurrency
        # outcome). Founded correctly: its logical_id equals its own sid.
        result_insert = connection.execute(
            sqlalchemy.insert(link_table).values(
                {"logical_id": 0, "source_lid": source_lid, "target_lid": target_lid, "retracted": 0, "store_timestamp": 1}
            )
        )
        new_sid = int(result_insert.inserted_primary_key[0])
        connection.execute(
            sqlalchemy.update(link_table).where(link_table.c.sid == new_sid).values({"logical_id": new_sid})
        )
    summary = store.fsck(known_types=(Result, Project), repair=False, collect_garbage=False, exclusive=True)
    notes = [violation for violation in summary.violations if "repairable, non-corrupting note" in violation]
    assert len(notes) == 1
    # A tolerated duplicate is not counted as corruption.
    link_counter = summary.tables.get("_httk_link_result__project__projects")
    assert link_counter is None or link_counter.conflicts == 0
    # linked() still dedups the pair to a single target.
    assert _names(store.linked(result, "projects")) == ["P1"]


# =========================================================================== #
# Sub-package A (P2b) — SPIKE: correlated NOT EXISTS inside a LEFT JOIN ON.
#
# The weak-link searcher DSL joins the link table with a latest-of-lineage
# NOT EXISTS in the JOIN onclause (never in WHERE, which would null-kill the
# LEFT JOIN rows). This spike proves that construct renders and evaluates on
# every SQL arm (sqlite/duckdb, and postgresql when configured) BEFORE the DSL
# is built on it. If it fails on any backend the onclause form is unsound and
# the fallback is a design change.
# =========================================================================== #


def _spike_latest_not_exists(table, alias):
    """Replicate the searcher's latest-of-lineage NOT EXISTS over ``alias``."""
    newer = table.alias()
    subquery = (
        sqlalchemy.select(sqlalchemy.literal(1))
        .select_from(newer)
        .where(newer.c.logical_id == alias.c.logical_id, newer.c.sid > alias.c.sid)
        .correlate(alias)
    )
    return ~subquery.exists()


def test_spike_not_exists_in_left_join_onclause(store_factory):
    store = _sql_store(store_factory)
    p = Project("P")
    store.save(p)

    r_live = Result("live")
    store.save(r_live)
    store.link(r_live, "projects", p)

    r_retracted = Result("retr")  # link then unlink: latest revision retracted, older one live
    store.save(r_retracted)
    store.link(r_retracted, "projects", p)
    store.unlink(r_retracted, "projects", p)

    r_superseded = Result("sup")  # link/unlink/link: 3 revisions, latest live, an old live one superseded
    store.save(r_superseded)
    store.link(r_superseded, "projects", p)
    store.unlink(r_superseded, "projects", p)
    store.link(r_superseded, "projects", p)

    r_none = Result("none")  # never linked
    store.save(r_none)

    result = store._table("result")
    link = store._table("_httk_link_result__project__projects")
    link_alias = link.alias()

    onclause = sqlalchemy.and_(
        link_alias.c.source_lid == result.c.logical_id,
        link_alias.c.retracted == 0,
        _spike_latest_not_exists(link, link_alias),
    )
    statement = (
        sqlalchemy.select(result.c.label, sqlalchemy.func.count(link_alias.c.sid))
        .select_from(result.outerjoin(link_alias, onclause))
        .group_by(result.c.label)
    )
    # The same join WITHOUT the NOT EXISTS, to prove the latest-of-lineage
    # clause is load-bearing (not incidentally true on this data).
    onclause_no_latest = sqlalchemy.and_(
        link_alias.c.source_lid == result.c.logical_id, link_alias.c.retracted == 0
    )
    statement_no_latest = (
        sqlalchemy.select(result.c.label, sqlalchemy.func.count(link_alias.c.sid))
        .select_from(result.outerjoin(link_alias, onclause_no_latest))
        .group_by(result.c.label)
    )
    with store._read_connection() as connection:
        matched = {label: int(count) for label, count in connection.execute(statement).all()}
        without_latest = {label: int(count) for label, count in connection.execute(statement_no_latest).all()}

    # Full onclause: latest-live links join exactly once; retracted-latest and
    # unlinked sources join no live row.
    assert matched == {"live": 1, "retr": 0, "sup": 1, "none": 0}
    # Dropping the NOT EXISTS lets superseded/older-live revisions match, which
    # is precisely what the latest-of-lineage clause must (and does) exclude.
    assert without_latest["retr"] == 1
    assert without_latest["sup"] == 2


# =========================================================================== #
# Sub-package B/D (P2b) — searcher DSL: field chaining, identity, sets,
# negations, multiplicity, as_of, only_latest, rejections.
# =========================================================================== #


@dataclass(frozen=True)
class Team:
    lead: str
    members: tuple[str, ...] = ()  # a variable-length child field (not chainable)


@dataclass(frozen=True)
class Owned:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(WeakLink("teams", Team),))

    label: str


def _query_labels(searcher, variable):
    """Run a Result search and return the matched labels, sorted."""
    searcher.output(variable, "r")
    return sorted(row.values[0].label for row in searcher)


def test_field_chaining_reflects_latest_target_revision(store_factory):
    store = _sql_store(store_factory)
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
    assert found("v2") == ["R"]  # the latest revision does


def test_field_chaining_ignores_retracted_links(store_factory):
    store = _sql_store(store_factory)
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


def test_identity_equals_stored_object_and_target_variable(store_factory):
    store = _sql_store(store_factory)
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

    searcher = store.searcher()
    v = searcher.variable(Result)
    pv = searcher.variable(Project)
    searcher.add(pv.name == "P2")
    searcher.add(v.links.projects == pv)
    assert _query_labels(searcher, v) == ["R2"]


def test_has_any_and_has_only_including_vacuous_no_links(store_factory):
    store = _sql_store(store_factory)
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
    # has_only(p1): sources whose every live link is p1 — R1, and the no-links
    # Rnone by vacuous truth; Rboth (which also links p2) is excluded.
    assert found(lambda s, v: s.add(v.links.projects.has_only(p1))) == ["R1", "Rnone"]
    assert found(lambda s, v: s.add(v.links.projects.has_only(p1, p2))) == ["R1", "Rboth", "Rnone"]


def test_has_all_pattern_over_fresh_aliases(store_factory):
    store = _sql_store(store_factory)
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
    # Fresh alias per access: the two ANDed predicates constrain independent
    # link rows, so only a source linked to BOTH matches (HAS ALL).
    searcher.add((v.links.projects.name == "P1") & (v.links.projects.name == "P2"))
    assert _query_labels(searcher, v) == ["Rboth"]


def test_negations_are_set_wise_and_match_the_zero_links_row(store_factory):
    store = _sql_store(store_factory)
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

    # Set-wise negation: "no live link is p1" — includes the no-links row.
    assert found(lambda s, v: s.add(~v.links.projects.has_any(p1))) == ["R2", "Rnone"]
    assert found(lambda s, v: s.add(~(v.links.projects.name == "P1"))) == ["R2", "Rnone"]


def test_string_rhs_raises_type_error(store_factory):
    store = _sql_store(store_factory)
    r = Result("R")
    store.save(r)
    searcher = store.searcher()
    v = searcher.variable(Result)
    with pytest.raises(TypeError, match="chain a target field|stored Project"):
        _ = v.links.projects == "P1"


def test_source_with_two_live_links_appears_exactly_once(store_factory):
    store = _sql_store(store_factory)
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
    assert _query_labels(searcher, v) == ["R"]  # grouped: a single row, not one per link

    counting = store.searcher()
    cv = counting.variable(Result)
    counting.add(cv.links.projects.has_any(p1, p2))
    assert counting.count() == 1


def test_multi_revision_links_and_targets_resolve_to_latest(store_factory):
    store = _sql_store(store_factory)
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
    assert counting.count() == 1  # one row despite the multi-revision link lineage


def test_searcher_as_of_time_travels_links_and_targets(store_factory):
    store = _sql_store(store_factory)
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


def test_searcher_as_of_sees_link_live_before_a_later_retraction(store_factory):
    store = _sql_store(store_factory)
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


def test_only_latest_is_orthogonal_to_weak_link_traversal(store_factory):
    store = _sql_store(store_factory)
    p = Project("P1")
    store.save(p)
    r = Result("R")
    store.save(r)
    store.link(r, "projects", p)
    store.replace(r, Result("R2"))  # supersede the source; the link binds the lineage

    # Default (all revisions): both the superseded and the latest source
    # revision traverse the link and match.
    both = store.searcher()
    bv = both.variable(Result)
    both.add(bv.links.projects.name == "P1")
    assert _query_labels(both, bv) == ["R", "R2"]
    counting = store.searcher()
    cv = counting.variable(Result)
    counting.add(cv.links.projects.name == "P1")
    assert counting.count() == 2

    # only_latest roots: just the latest source revision, still link-filtered.
    latest = store.searcher(only_latest=True)
    lv = latest.variable(Result)
    latest.add(lv.links.projects.name == "P1")
    assert _query_labels(latest, lv) == ["R2"]
    latest_count = store.searcher(only_latest=True)
    lcv = latest_count.variable(Result)
    latest_count.add(lcv.links.projects.name == "P1")
    assert latest_count.count() == 1


def test_deep_chaining_into_target_non_scalar_is_rejected(store_factory):
    from httk.store.query import UnsupportedQueryError

    store = _sql_store(store_factory)
    store.save(Owned("O"))  # creates Owned's parent + link tables
    searcher = store.searcher()
    v = searcher.variable(Owned)
    with pytest.raises(UnsupportedQueryError, match="scalar and encoded"):
        _ = v.links.teams.members  # a child field of the target
    with pytest.raises(UnsupportedQueryError, match="not supported"):
        _ = v.links.teams.links  # nested weak links of the target


def test_link_path_projection_is_rejected(store_factory):
    from httk.store.query import UnsupportedQueryError

    store = _sql_store(store_factory)
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


def test_links_accessor_on_a_freshly_fetched_row(store_factory):
    store = _sql_store(store_factory)
    p1, p2 = Project("P1"), Project("P2")
    store.save(p1)
    store.save(p2)
    r = Result("R")
    sid = store.save(r)
    store.link(r, "projects", p1)
    store.link(r, "projects", p2)

    reopened = store_factory.reopen(store)  # fresh handle: dodge the identity cache
    fetched = reopened.fetch(Result, sid)
    assert type(fetched) is not Result  # a lazy, store-bound row

    assert _names(fetched.links.projects) == ["P1", "P2"]
    # Memoized like reference fields: the same tuple object, no re-query.
    assert fetched.links.projects is fetched.links.projects
    assert "projects" in dir(fetched.links)
    with pytest.raises(AttributeError, match="no weak link named 'ghost'"):
        _ = fetched.links.ghost
    # A plain, unbound instance has no accessor at all.
    with pytest.raises(AttributeError):
        _ = Result("plain").links
