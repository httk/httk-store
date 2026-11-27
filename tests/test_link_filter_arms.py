"""Searcher weak-link and reverse strong-link filters give the same answers on every SQL arm.

ClickHouse rejects a correlated subquery in a JOIN condition (code 48), so its
latest-of-lineage restriction (on weak-link, weak-link-target and reverse
strong-link owner joins) is rendered uncorrelated. Its stores are bulk-fenced (no
``link``/``unlink``/``replace``), so the ClickHouse arm copies the rows of a SQLite
store built through the public API verbatim: every arm searches the same
retracted, revived and superseded revisions.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
import sqlalchemy
from clickhouse_read_support import CLICKHOUSE_PARAM, clickhouse_database
from httk.core import Run, RunEdge, RunEntry
from httk.core.storage import StorageInfo, WeakLink
from postgres_support import POSTGRES_PARAM, postgres_database
from test_strong_link_search import StructureFamily, StructureRow

from httk.store import EntryIdScheme
from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql import Backend, SqlStore


@dataclass(frozen=True)
class Project:
    name: str
    note: str = ""


@dataclass(frozen=True)
class Result:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(WeakLink("projects", Project),))

    label: str


_CUTOFF = 2_000_000  # between the 1M build and the 3M supersessions (store-clock ns)
_TABLES = (
    resolve_schema(Project).table_name,
    resolve_schema(Result).table_name,
    *(link.table_name for link in resolve_schema(Result).links),
)


def _build(store: SqlStore) -> None:
    """Live, revived, retracted, unlinked and two-target sources; P1 then superseded by a v2 revision.

    Everything is stored at clock 1M except the retraction and the v2 revision, stored at 3M (after ``_CUTOFF``).
    """
    store._clock = lambda: 1_000_000
    p1, p2 = Project("P1", "v1"), Project("P2", "x")
    store.save(p1)
    store.save(p2)
    results = {label: Result(label) for label in ("live", "revived", "retracted", "none", "both")}
    for result in results.values():
        store.save(result)
    store.link(results["live"], "projects", p1)
    store.link(results["revived"], "projects", p2)
    store.unlink(results["revived"], "projects", p2)
    store.link(results["revived"], "projects", p2)
    store.link(results["retracted"], "projects", p1)
    store.link(results["both"], "projects", p1)
    store.link(results["both"], "projects", p2)
    store._clock = lambda: 3_000_000
    store.unlink(results["retracted"], "projects", p1)
    store.replace(p1, Project("P1", "v2"))


def _copy_into_clickhouse(
    database: Any,
    build: Callable[[SqlStore], None],
    scaffold: Callable[[Any], None],
    tables: tuple[str, ...],
    **options: Any,
) -> SqlStore:
    """Create the tables by a scaffold bulk ingest, then replace their rows by a SQLite reference's."""
    store = SqlStore(database, **options)
    with store.bulk_ingest(finalize="deferred", track_sids=True) as bulk:
        scaffold(bulk)
    with Backend.sqlite() as reference_database:
        reference = SqlStore(reference_database, **options)
        build(reference)
        with reference._read_connection() as source, database.engine.begin() as target:
            for name in tables:
                rows = [dict(row) for row in source.execute(sqlalchemy.select(reference._table(name))).mappings()]
                target.execute(sqlalchemy.text(f'TRUNCATE TABLE "{name}"'))
                if rows:
                    target.execute(sqlalchemy.insert(store._table(name)), rows)
    return SqlStore(database, **options)  # fresh: no identity cache of the scaffold rows


@contextmanager
def _arm(
    param: str,
    build: Callable[[SqlStore], None],
    scaffold: Callable[[Any], None],
    tables: tuple[str, ...],
    **options: Any,
) -> Iterator[SqlStore]:
    if param == "clickhousedb":
        with clickhouse_database() as database:
            yield _copy_into_clickhouse(database, build, scaffold, tables, **options)
        return
    if param == "postgresql":
        context: Any = postgres_database()
    elif param == "duckdb":
        pytest.importorskip("duckdb_engine")
        context = Backend.duckdb()
    else:
        context = Backend.sqlite()
    with context as database:
        store = SqlStore(database, **options)
        build(store)
        yield store


def _weak_scaffold(bulk: Any) -> None:
    bulk.save(Result("scaffold"), links={"projects": Project("scaffold")})


def _p2(store: SqlStore) -> Project:
    searcher = store.searcher()
    v = searcher.variable(Project)
    searcher.add(v.name == "P2")
    (row,) = searcher.results(p=v)
    return row.p


def _labels(store: SqlStore, condition: Callable[[Any, Project], Any], as_of: int | None = None) -> list[str]:
    target = _p2(store)
    searcher = store.searcher(as_of=as_of)
    v = searcher.variable(Result)
    searcher.add(condition(v, target))
    return sorted(row.r.label for row in searcher.results(r=v))


_EXPECTED: dict[str, tuple[Callable[[Any, Project], Any], list[str]]] = {
    "latest target field": (lambda v, p2: v.links.projects.note == "v2", ["both", "live"]),
    "superseded target field": (lambda v, p2: v.links.projects.note == "v1", []),
    "revived link": (lambda v, p2: v.links.projects.name == "P2", ["both", "revived"]),
    "retracted link": (lambda v, p2: v.links.projects.name == "P1", ["both", "live"]),
    "has_any": (lambda v, p2: v.links.projects.has_any(p2), ["both", "revived"]),
    "negated has_any": (lambda v, p2: ~v.links.projects.has_any(p2), ["live", "none", "retracted"]),
    "negated field": (lambda v, p2: ~(v.links.projects.name == "P1"), ["none", "retracted", "revived"]),
    "has_only": (lambda v, p2: v.links.projects.has_only(p2), ["none", "retracted", "revived"]),
    "with scalar": (lambda v, p2: (v.label != "both") & (v.links.projects.name == "P2"), ["revived"]),
}


@pytest.mark.parametrize("param", ("sqlite", "duckdb", POSTGRES_PARAM, CLICKHOUSE_PARAM))
def test_weak_link_filters_agree_across_sql_arms(param: str) -> None:
    with _arm(param, _build, _weak_scaffold, _TABLES, entry_records={}) as store:
        actual = {name: _labels(store, condition) for name, (condition, _expected) in _EXPECTED.items()}
        # As of the cutoff the retraction and P1's v2 revision lie in the future: both cutoff branches matter.
        past = [
            _labels(store, lambda v, p2, note=note: v.links.projects.note == note, as_of=_CUTOFF)
            for note in ("v1", "v2")
        ]
    assert actual == {name: expected for name, (_condition, expected) in _EXPECTED.items()}
    assert past == [["both", "live", "retracted"], []]


_STRONG_OPTIONS: dict[str, Any] = {
    "entry_records": {RunEntry: Run, StructureFamily: StructureRow},
    "entry_ids": EntryIdScheme("httk.test", "1"),
}
_STRONG_TABLES = (
    resolve_schema(StructureRow).table_name,
    resolve_schema(Run).table_name,
    resolve_schema(RunEdge).table_name,
    *(f"{resolve_schema(Run).table_name}_{child}" for child in ("inputs", "outputs", "artifacts")),
)


def _strong_build(store: SqlStore) -> None:
    """Runs whose input edges name Ca and Ti (clock 1M); at 3M the Ti run gets a revision whose input is O."""
    store._clock = lambda: 1_000_000
    ca, ti, o = (store.fetch(StructureRow, store.save(StructureRow(f)), eager=True) for f in ("Ca", "Ti", "O"))
    store.save(Run(inputs=(RunEdge("initial", "structures", ca.id),), source_id="ws:ca"))
    run_ti = store.fetch(Run, store.save(Run(inputs=(RunEdge("initial", "structures", ti.id),), source_id="ws:ti")))
    store._clock = lambda: 3_000_000
    store.replace(run_ti, Run(inputs=(RunEdge("initial", "structures", o.id),), source_id="ws:ti"))


def _strong_scaffold(bulk: Any) -> None:
    edge = RunEdge("initial", "structures", "scaffold")
    bulk.save(StructureRow("scaffold"))
    bulk.save(Run(inputs=(edge,), outputs=(edge,), artifacts=(edge,), source_id="scaffold"))


def _formulas(store: SqlStore, condition: Callable[[Any, list[Run]], Any], as_of: int | None = None) -> list[str]:
    search = store.searcher()
    run = search.variable(Run)
    runs = [row.run for row in search.results(run=run)]
    search = store.searcher(as_of=as_of)
    structure = search.variable(StructureRow)
    search.add(condition(structure, runs))
    return sorted(search.results(formula=structure.formula).scalars())


@pytest.mark.parametrize("param", ("sqlite", "duckdb", POSTGRES_PARAM, CLICKHOUSE_PARAM))
def test_reverse_strong_link_filters_agree_across_sql_arms(param: str) -> None:
    """Reverse traversal joins only the latest main owner revision: the superseded Ti input no longer counts."""
    with _arm(param, _strong_build, _strong_scaffold, _STRONG_TABLES, **_STRONG_OPTIONS) as store:
        positive = _formulas(store, lambda structure, runs: structure.links.is_input.has_any(*runs))
        negated = _formulas(store, lambda structure, runs: ~structure.links.is_input.has_any(*runs))
        past_positive = _formulas(store, lambda structure, runs: structure.links.is_input.has_any(*runs), _CUTOFF)
        past_negated = _formulas(store, lambda structure, runs: ~structure.links.is_input.has_any(*runs), _CUTOFF)
    assert (positive, negated) == (["Ca", "O"], ["Ti"])
    assert (past_positive, past_negated) == (["Ca", "Ti"], ["O"])  # before the replace, Ti's run is latest
