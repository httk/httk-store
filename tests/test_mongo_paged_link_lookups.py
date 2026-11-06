"""Paged MongoDB results honour weak-link and child-record predicates as unpaged results do (live, ``HTTK_TEST_MONGODB_URI``)."""

from dataclasses import dataclass
from typing import ClassVar

from httk.core.storage import StorageInfo, WeakLink

from httk.store.query import PageOrder


@dataclass(frozen=True)
class PagedProject:
    name: str


@dataclass(frozen=True)
class PagedSite:
    symbol: str


@dataclass(frozen=True)
class PagedParent:
    name: str
    sites: tuple[PagedSite, ...]


@dataclass(frozen=True)
class PagedResult:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(WeakLink("projects", PagedProject),))

    label: str


def test_paged_weak_link_predicate_matches_unpaged(mongo_test_database):
    from httk.store.backend.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_records={})
    p1, p2 = PagedProject("P1"), PagedProject("P2")
    store.save(p1)
    store.save(p2)
    for label, projects in (("R1", (p1,)), ("R2", (p2,)), ("Rboth", (p1, p2)), ("Rnone", ())):
        result = PagedResult(label)
        store.save(result)
        for project in projects:
            store.link(result, "projects", project)

    searcher = store.searcher()
    variable = searcher.variable(PagedResult)
    searcher.add(variable.links.projects.name == "P1")
    results = searcher.results(label=variable.label)
    unpaged = sorted(row.label for row in results)
    assert unpaged == ["R1", "Rboth"]

    page = results.page(size=1, order_by=(PageOrder("label"),), include_total=True)
    labels = [row.label for row in page.rows]
    while page.next is not None:
        page = results.page(size=1, order_by=(PageOrder("label"),), cursor=page.next, include_total=True)
        labels.extend(row.label for row in page.rows)
    assert labels == unpaged
    assert page.total == 2


def test_paged_child_record_predicate_drops_element_arrays(mongo_test_database):
    from httk.store.backend.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_records={})
    for name, symbols in (("A", ("Cl", "Na")), ("B", ("Cl",)), ("C", ()), ("D", ("Na",))):
        store.save(PagedParent(name, tuple(PagedSite(symbol) for symbol in symbols)))

    searcher = store.searcher()
    variable = searcher.variable(PagedParent)
    searcher.add(variable.sites.symbol == "Cl")
    seen: list[dict] = []
    candidates = searcher._candidate_documents

    def spy(pipeline):
        for document, verified in candidates(pipeline):
            seen.append(document)
            yield document, verified

    searcher._candidate_documents = spy
    results = searcher.results(name=variable.name)
    page = results.page(size=1, order_by=(PageOrder("name"),), include_total=True)
    names = [row.name for row in page.rows]
    while page.next is not None:
        page = results.page(size=1, order_by=(PageOrder("name"),), cursor=page.next, include_total=True)
        names.extend(row.name for row in page.rows)
    assert names == ["A", "B"]
    assert page.total == 2
    assert seen
    assert not any(key.startswith("_httk_child_") for document in seen for key in document)
