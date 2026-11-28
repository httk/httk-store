"""Chaining from a child field of storable records into the element fields (``s.sites.symbol``).

One spec for every backend: the SQL dialects and, live, MongoDB (gated on ``HTTK_TEST_MONGODB_URI``), and
for both layouts of the child field: a link table to an element table, and ``Inline()`` rows.
"""

import re
import uuid
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Any, ClassVar

import pytest
from clickhouse_read_support import CLICKHOUSE_PARAM, bulk_store
from httk.core.storage import Inline, Shape, StorageInfo, WeakLink
from postgres_support import POSTGRES_PARAM, postgres_database

from httk.store import FederatedStore
from httk.store.backend.sql import Backend, SqlStore
from httk.store.query import UnsupportedQueryError


@dataclass(frozen=True)
class SiteTag:
    label: str


@dataclass(frozen=True)
class SiteRow:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

    x: float
    symbol: str
    occupancy: Fraction = Fraction(1)
    tag: SiteTag | None = None


@dataclass(frozen=True)
class Parent:
    name: str
    sites: tuple[SiteRow, ...]
    n: int | None = None
    labels: tuple[str, ...] = ()


PARENTS = [
    Parent("A", (SiteRow(0.1, "Cl"), SiteRow(0.5, "Na")), 1, ("x", "y")),
    Parent("B", (SiteRow(0.5, "Cl", Fraction(1, 2)),), labels=("x",)),
    Parent("C", (), 2),
    Parent("D", (SiteRow(0.2, "Na", tag=SiteTag("t")),), labels=("y",)),
]


@dataclass(frozen=True)
class InlineSiteRow:
    x: float
    symbol: str
    occupancy: Fraction = Fraction(1)
    pos: Annotated[tuple[tuple[float, ...], ...], Shape(1, 1)] | None = None


@dataclass(frozen=True)
class InlineParent:
    name: str
    sites: Annotated[tuple[InlineSiteRow, ...], Inline()]
    n: int | None = None
    labels: tuple[str, ...] = ()


INLINE_PARENTS = [
    InlineParent(
        parent.name,
        tuple(InlineSiteRow(site.x, site.symbol, site.occupancy) for site in parent.sites),
        parent.n,
        parent.labels,
    )
    for parent in PARENTS
]
LAYOUTS = {"link": PARENTS, "inline": INLINE_PARENTS}
# The parent class each fixture store (or federation) holds, keyed by id(); unlisted stores hold Parent.
PARENT_OF: dict[int, type] = {}


def _parent_class(store: Any) -> type:
    return PARENT_OF.get(id(store), Parent)


def _mongo_store(client: Any) -> Any:
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    return MongoStore(MongoDatabase(client, f"httk_test_{uuid.uuid4().hex}"), entry_records={})


@pytest.fixture(scope="module", params=["link", "inline"])
def layout(request):
    """The child-field layout: a link table to element rows, or Inline() rows."""
    return request.param


@pytest.fixture(scope="module", params=["sqlite", "duckdb", CLICKHOUSE_PARAM, POSTGRES_PARAM, "mongo"])
def store(request, layout):
    """A populated store per supported backend and layout."""
    records = LAYOUTS[layout]
    for sql_store in _populated(request, records):
        PARENT_OF[id(sql_store)] = type(records[0])
        try:
            yield sql_store
        finally:
            PARENT_OF.pop(id(sql_store), None)


def _populated(request: Any, records: list[Any]) -> Any:
    if request.param == "mongo":
        mongo_store = _mongo_store(request.getfixturevalue("mongo_test_client"))
        try:
            for parent in records:
                mongo_store.save(parent)
            yield mongo_store
        finally:
            mongo_store._database.client.drop_database(mongo_store._database.database.name)
        return
    if request.param == "clickhousedb":
        with bulk_store(records) as sql_store:
            yield sql_store
        return
    if request.param == "postgresql":
        with postgres_database() as database:
            sql_store = SqlStore(database, entry_records={})
            for parent in records:
                sql_store.save(parent)
            yield sql_store
        return
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
    with Backend.duckdb() if request.param == "duckdb" else Backend.sqlite() as database:
        sql_store = SqlStore(database, entry_records={})
        for parent in records:
            sql_store.save(parent)
        yield sql_store


def names(store: Any, build: Any) -> list[str]:
    search = store.searcher()
    s = search.variable(_parent_class(store))
    search.add(build(s))
    return sorted(row.parent.name for row in search.results(parent=s))


def test_chained_element_field(store):
    assert names(store, lambda s: s.sites.symbol == "Cl") == ["A", "B"]
    assert names(store, lambda s: s.sites.occupancy < Fraction(1)) == ["B"]  # encoded element field


def correlated(s: Any) -> Any:
    site = s.sites
    return (site.symbol == "Cl") & (site.x > 0.4)


def not_correlated(s: Any) -> Any:
    site = s.sites
    return ~((site.symbol == "Cl") & (site.x > 0.4))


def test_one_traversal_correlates_and_separate_accesses_do_not(store):
    assert names(store, correlated) == ["B"]
    assert names(store, lambda s: (s.sites.symbol == "Cl") & (s.sites.x > 0.4)) == ["A", "B"]

    search = store.searcher()
    s = search.variable(_parent_class(store))
    site = s.sites
    _ = (site.symbol, site.x)
    if isinstance(store, SqlStore) and _parent_class(store) is InlineParent:
        assert len(s._joins) == 1  # the child alias holds the element columns
    elif isinstance(store, SqlStore):
        assert len(s._joins) == 2  # one link alias, one element alias
    elif _parent_class(store) is InlineParent:
        assert search._child_lookups == []  # the elements are embedded: no $lookup
    else:
        assert len(search._child_lookups) == 1  # one element $lookup per traversal object


def test_negation_is_set_wise(store):
    assert names(store, lambda s: ~(s.sites.symbol == "Cl")) == ["C", "D"]
    # Only B has one site that is Cl with x > 0.4; A has each property on a different site.
    assert names(store, not_correlated) == ["A", "C", "D"]


def test_negation_keeps_root_conditions_three_valued(store):
    # n: A 1, B unknown, C 2, D unknown. An unknown root comparison keeps the negated conjunction unknown.
    assert names(store, lambda s: ~((s.sites.symbol == "Cl") & (s.n == 1))) == ["C", "D"]
    assert names(store, lambda s: ~((s.sites.symbol == "Cl") | (s.n == 1))) == ["C"]

    def correlated_beside_root(s):
        site = s.sites
        return ~((site.symbol == "Cl") & (s.n == 1) & (site.x > 0.4))

    # A has no single Cl site with x > 0.4; for B the correlated pair holds and n is unknown.
    assert names(store, correlated_beside_root) == ["A", "C", "D"]


def test_set_operations_on_element_field(store):
    assert names(store, lambda s: s.sites.symbol.has_any("Na") & s.sites.symbol.has_any("Cl")) == ["A"]
    assert names(store, lambda s: s.sites.symbol.has_any("Na", "Cl")) == ["A", "B", "D"]
    assert names(store, lambda s: s.sites.symbol.has_only("Cl")) == ["B", "C"]
    assert names(store, lambda s: s.sites.symbol.has_only("Na", "Cl")) == ["A", "B", "C", "D"]
    assert names(store, lambda s: ~s.sites.symbol.has_any("Na")) == ["B", "C"]


def test_element_sid_comparison_still_works(store):
    if _parent_class(store) is InlineParent:
        # Inline rows have no element identity: the field itself refuses comparisons and set operations.
        for build in (lambda s: s.sites.has_any(1), lambda s: s.sites == 1, lambda s: s.sites.has_only()):
            with pytest.raises(UnsupportedQueryError, match="InlineParent.sites is an Inline field"):
                names(store, build)
        return
    search = store.searcher()
    site = search.variable(SiteRow)
    search.add((site.symbol == "Cl") & (site.x > 0.4))
    (sid,) = search.results(sid=site.sid).scalars("sid")
    assert names(store, lambda s: s.sites.has_any(sid)) == ["B"]
    assert names(store, lambda s: s.sites == sid) == ["B"]

    def sid_and_element(s: Any) -> Any:
        site = s.sites
        return site.has_any(sid) & (site.symbol == "Cl")

    # A sid condition and a chained one on one traversal object do not form one element condition.
    with pytest.raises(UnsupportedQueryError, match="cannot evaluate on one element"):
        names(store, sid_and_element)


def test_no_duplicate_parents(store):
    search = store.searcher()
    s = search.variable(_parent_class(store))
    search.add(s.sites.symbol.has_any("Na", "Cl"))
    rows = [row.parent.name for row in search.results(parent=s)]
    assert sorted(rows) == ["A", "B", "D"]
    assert search.count() == 3


def test_refusals(store):
    search = store.searcher()
    s = search.variable(_parent_class(store))
    if _parent_class(store) is InlineParent:
        with pytest.raises(UnsupportedQueryError, match=r"InlineSiteRow\.pos is a fixed_array field"):
            s.sites.pos  # noqa: B018
    else:
        with pytest.raises(UnsupportedQueryError, match=r"SiteRow\.tag is a reference field"):
            s.sites.tag  # noqa: B018
    with pytest.raises(AttributeError, match=r"SiteRow has no stored field 'nope'"):
        s.sites.nope  # noqa: B018
    with pytest.raises(AttributeError, match="'sid'"):
        s.sites.sid  # noqa: B018
    with pytest.raises(UnsupportedQueryError, match="field of child records"):
        search.results(symbol=s.sites.symbol)
    with pytest.raises(UnsupportedQueryError, match="field of child records"):
        search.add_sort(s.sites.x)


@pytest.mark.parametrize("chain_layout", ["link", "inline"])
@pytest.mark.parametrize("first_backend", ["sqlite", "mongo"])
def test_federated_chain(request, first_backend, chain_layout):
    records = LAYOUTS[chain_layout]
    with Backend.sqlite() as first_database, Backend.sqlite() as second_database:
        if first_backend == "mongo":
            first = _mongo_store(request.getfixturevalue("mongo_test_client"))
            request.addfinalizer(lambda: first._database.client.drop_database(first._database.database.name))
        else:
            first = SqlStore(first_database, entry_records={})
        second = SqlStore(second_database, entry_records={})
        for index, parent in enumerate(records):
            (first if index % 2 else second).save(parent)
        federation = FederatedStore({"first": first, "second": second})
        PARENT_OF[id(federation)] = type(records[0])
        request.addfinalizer(lambda: PARENT_OF.pop(id(federation), None))
        assert names(federation, lambda s: s.sites.symbol == "Cl") == ["A", "B"]
        assert names(federation, correlated) == ["B"]
        assert names(federation, lambda s: (s.sites.symbol == "Cl") & (s.sites.x > 0.4)) == ["A", "B"]
        assert names(federation, not_correlated) == ["A", "C", "D"]

        search = federation.searcher()  # correlation also spans separate add() calls
        s = search.variable(type(records[0]))
        site = s.sites
        search.add(site.symbol == "Cl")
        search.add(site.x > 0.4)
        assert [row.parent.name for row in search.results(parent=s)] == ["B"]

        # Fresh list-field accesses replay as fresh backend objects; a reused one is refused.
        assert names(federation, lambda s: s.labels.has_any("x") & ~s.labels.has_any("y")) == ["B"]
        with pytest.raises(UnsupportedQueryError) as refused:  # the source's refusal, with source context
            names(federation, list_reuse_mixed)
        assert re.search(REUSE, str(refused.value.__cause__))


def mixed_negation(s: Any) -> Any:
    site = s.sites
    return (site.symbol == "Na") & ~(site.x > 0.4)


def mixed_disjunction(s: Any) -> Any:
    site = s.sites
    return ((site.symbol == "Cl") | (s.name == "C")) & (site.x > 0.4)


def element_disjunction(s: Any) -> Any:
    site = s.sites
    return ((site.symbol == "Na") | (site.x > 0.4)) & (site.x < 0.3)


def test_uncorrelatable_combinations_are_refused(store):
    for build in (mixed_negation, mixed_disjunction):
        with pytest.raises(UnsupportedQueryError, match="cannot evaluate on one element"):
            names(store, build)
    search = store.searcher()
    s = search.variable(_parent_class(store))
    with pytest.raises(UnsupportedQueryError, match="compares only with literal values"):
        _ = s.sites.symbol == None
    with pytest.raises(UnsupportedQueryError, match="compares only with literal values"):
        _ = s.sites.symbol == s.name
    with pytest.raises(ValueError, match="None is not a valid member"):
        s.sites.symbol.has_any(None)
    # Supported mixes: | within one traversal, a fresh traversal per negation, a root condition alongside.
    assert names(store, element_disjunction) == ["D"]
    assert names(store, lambda s: (s.sites.symbol == "Na") & ~(s.sites.x > 0.4)) == ["D"]
    assert names(store, lambda s: (s.sites.symbol == "Cl") & (s.name == "A")) == ["A"]
    assert names(store, lambda s: ~s.sites.symbol.has_only("Cl")) == ["A", "D"]


@dataclass(frozen=True)
class Project:
    name: str


@dataclass(frozen=True)
class Linked:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(WeakLink("projects", Project),))

    name: str


def test_link_field_comparisons_are_not_child_record_fields():
    # Weak-link target columns are also child-style joins; only chained element fields refuse None/field operands.
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        for name, project in (("P1", "P1"), ("x", "P2"), ("y", None)):
            linked = store.fetch(Linked, store.save(Linked(name)))
            if project is not None:
                store.link(linked, "projects", store.fetch(Project, store.save(Project(project))))

        def linked_names(build: Any) -> list[str]:
            search = store.searcher()
            v = search.variable(Linked)
            search.add(build(v))
            return sorted(row.v.name for row in search.results(v=v))

        assert linked_names(lambda v: v.links.projects.name != None) == ["P1", "x"]
        assert linked_names(lambda v: v.links.projects.name == v.name) == ["P1"]


REUSE = r"Parent\.labels is used through one search object in several conditions"


def list_reuse_mixed(s: Any) -> Any:
    labels = s.labels
    return labels.has_any("x") & ~labels.has_any("y")


def list_reuse_positive(s: Any) -> Any:
    labels = s.labels
    return (labels == "x") & (labels == "y")


def test_list_field_object_reuse_constrains_one_entry(store):
    # One list-field object: its conditions hold on one entry (see test_collection_reuse); a mixed use is refused.
    with pytest.raises(UnsupportedQueryError, match=REUSE):
        names(store, list_reuse_mixed)
    assert names(store, list_reuse_positive) == []  # no single label is both "x" and "y"
    search = store.searcher()
    s = search.variable(_parent_class(store))
    labels = s.labels
    search.add(labels.has_any("x"))
    search.add(labels.contains("y"))
    assert search.count() == 0


def test_list_field_single_condition_and_fresh_accesses(store):
    def single(s):
        labels = s.labels
        return labels.has_any("x", "y")

    assert names(store, single) == ["A", "B", "D"]
    assert names(store, lambda s: s.labels.has_any("x") & ~s.labels.has_any("y")) == ["B"]
    assert names(store, lambda s: (s.labels == "x") & (s.labels == "y")) == ["A"]
