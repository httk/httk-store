"""One reuse rule for every collection-valued search object, on every backend.

A plain list field (``v.counts``), a child-record traversal (``v.sites``) and a
weak-link set (``v.links.projects``) follow the same rule: conditions taken from
ONE search object constrain the same element, ``~`` of them means "no element
satisfies them", a root condition beside them stays three-valued, and mixing a
selecting use with a negated or for-all use of the same object is refused. A
fresh access per condition stays an independent set.

Every kind reads one integer per element: A [1, 9], B [3], C [], D [7], so
"some element in (2, 5)" holds for B only, while each bound alone holds for
A, B and D.
"""

import re
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
from httk.core.storage import StorageInfo, WeakLink

from httk.store import FederatedStore
from httk.store.backend.sql import Backend, SqlStore
from httk.store.query import PageOrder, UnsupportedQueryError


@dataclass(frozen=True)
class Site:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

    value: int


@dataclass(frozen=True)
class Project:
    name: str
    value: int


@dataclass(frozen=True)
class Holder:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(WeakLink("projects", Project),))

    name: str
    n: int | None
    counts: tuple[int, ...]
    sites: tuple[Site, ...]


VALUES = {"A": (1, 9), "B": (3,), "C": (), "D": (7,)}
ROOT_N = {"A": 1, "B": None, "C": 2, "D": 1}
KINDS = ["counts", "sites", "projects"]
LABELS = {"counts": "Holder.counts", "sites": "Holder.sites", "projects": "Holder.links.projects"}


def populate(store: Any) -> dict[str, Any]:
    projects: dict[str, Any] = {}
    for name, values in VALUES.items():
        holder = store.fetch(Holder, store.save(Holder(name, ROOT_N[name], values, tuple(Site(v) for v in values))))
        for value in values:
            project = store.fetch(Project, store.save(Project(f"{name}{value}", value)))
            projects[project.name] = project
            store.link(holder, "projects", project)
    return projects


def collection(v: Any, kind: str) -> Any:
    return v.links.projects if kind == "projects" else getattr(v, kind)


def value(obj: Any, kind: str) -> Any:
    return obj if kind == "counts" else obj.value


@pytest.fixture(params=KINDS)
def kind(request):
    return request.param


@pytest.fixture
def store(store_factory):
    store = store_factory(transactions="require")  # MongoDB links need transactions
    store.projects = populate(store)
    return store


def names(store: Any, *builds: Any) -> list[str]:
    search = store.searcher()
    v = search.variable(Holder)
    for build in builds:
        search.add(build(v))
    return sorted(row.v.name for row in search.results(v=v))


def in_band(kind: str):
    def build(v):
        obj = collection(v, kind)
        return (value(obj, kind) > 2) & (value(obj, kind) < 5)

    return build


def test_one_object_constrains_one_element(store, kind):
    assert names(store, in_band(kind)) == ["B"]
    fresh = lambda v: (value(collection(v, kind), kind) > 2) & (value(collection(v, kind), kind) < 5)
    assert names(store, fresh) == ["A", "B"]


def test_negated_merged_group_means_no_element(store, kind):
    assert names(store, lambda v: ~in_band(kind)(v)) == ["A", "C", "D"]


def test_or_within_one_object(store, kind):
    def outer(v):
        obj = collection(v, kind)
        return (value(obj, kind) > 8) | (value(obj, kind) < 2)

    assert names(store, outer) == ["A"]
    assert names(store, lambda v: ~outer(v)) == ["B", "C", "D"]


def test_root_conjunct_stays_three_valued(store, kind):
    # B has an element in (2, 5) but an unknown n: the negated conjunction is unknown, so B does not match.
    assert names(store, lambda v: ~(in_band(kind)(v) & (v.n == 1))) == ["A", "C", "D"]
    assert names(store, lambda v: in_band(kind)(v) & (v.n == None)) == ["B"]


def test_mixed_use_is_refused(store, kind):
    def mixed(v):
        obj = collection(v, kind)
        return (value(obj, kind) > 2) & ~(value(obj, kind) < 5)

    with pytest.raises(UnsupportedQueryError, match=re.escape(f"{LABELS[kind]} is used through one search object")):
        names(store, mixed)
    access = "v." + LABELS[kind].split(".", 1)[1]
    with pytest.raises(UnsupportedQueryError, match=re.escape(f"{access}.has_any(a) & ~{access}.has_any(b)")):
        names(store, mixed)
    # The fresh-access rewrite the message suggests is supported.
    fresh = lambda v: (value(collection(v, kind), kind) > 2) & ~(value(collection(v, kind), kind) < 5)
    assert names(store, fresh) == ["D"]


def test_same_expression_twice_and_separate_adds(store, kind):
    search = store.searcher()
    v = search.variable(Holder)
    expression = value(collection(v, kind), kind) > 2
    search.add(expression)
    search.add(expression & expression)
    assert sorted(row.v.name for row in search.results(v=v)) == ["A", "B", "D"]

    search = store.searcher()
    v = search.variable(Holder)
    obj = collection(v, kind)
    search.add(value(obj, kind) > 2)
    search.add(value(obj, kind) < 5)
    assert sorted(row.v.name for row in search.results(v=v)) == ["B"]
    assert search.count() == 1
    page = search.results(name=v.name).page(size=1, order_by=(PageOrder("name"),), include_total=True)
    assert [row.name for row in page.rows] == ["B"]
    assert page.total == 1


def test_link_identity_and_field_share_one_element(store):
    a1, a9 = store.projects["A1"], store.projects["A9"]

    def identity_and_field(project):
        def build(v):
            link = v.links.projects
            return (link == project) & (link.value > 2)

        return build

    assert names(store, identity_and_field(a1)) == []
    assert names(store, identity_and_field(a9)) == ["A"]
    assert names(store, lambda v: (v.links.projects == a1) & (v.links.projects.value > 2)) == ["A"]
    assert names(store, lambda v: ~identity_and_field(a9)(v)) == ["B", "C", "D"]


@pytest.mark.parametrize("first_backend", ["sqlite", "mongo"])
def test_federated_list_reuse(request, first_backend):
    with Backend.sqlite() as first_database, Backend.sqlite() as second_database:
        if first_backend == "mongo":
            import uuid

            from httk.store.backend.mongo import MongoDatabase, MongoStore

            client = request.getfixturevalue("mongo_test_client")
            name = f"httk_test_{uuid.uuid4().hex}"
            first = MongoStore(MongoDatabase(client, name), entry_records={})
            request.addfinalizer(lambda: client.drop_database(name))
        else:
            first = SqlStore(first_database, entry_records={})
        second = SqlStore(second_database, entry_records={})
        for index, (name, values) in enumerate(VALUES.items()):
            (first if index % 2 else second).save(Holder(name, ROOT_N[name], values, ()))
        federation = FederatedStore({"first": first, "second": second})
        assert names(federation, in_band("counts")) == ["B"]
        assert names(federation, lambda v: ~in_band("counts")(v)) == ["A", "C", "D"]
        assert names(federation, lambda v: (v.counts > 2) & (v.counts < 5)) == ["A", "B"]

        search = federation.searcher()
        v = search.variable(Holder)
        counts = v.counts
        search.add(counts > 2)
        search.add(counts < 5)
        assert sorted(row.v.name for row in search.results(v=v)) == ["B"]
        assert search.count() == 1
