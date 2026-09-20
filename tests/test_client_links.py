"""No-network tests for the ``links`` relationship namespace: depth-1 dotted
predicates, set-valued outputs with automatic ``include=``, and the
``.links.<name>`` accessor on bound records returned from a query."""

import copy
import dataclasses
import json
import logging
import pickle
from urllib.parse import parse_qs, urlsplit

import pytest
from httk.atomistic import OptimadeStructure
from httk.core.optimade import OptimadeResource, parse_optimade_filter
from test_client_query import (
    STRUCTURES,
    FakeResponse,
    QueryClient,
    make_files,
    page,
    resource,
    response,
    schema_properties,
)

from httk.store import UnsupportedQueryError
from httk.store.optimade import OptimadeClientError, OptimadeResponseError, OptimadeStore


def query_params(url: str) -> dict[str, list[str]]:
    return parse_qs(urlsplit(url).query)


def page_with_included(resources: list[object], *, included: list[object], **kwargs: object) -> FakeResponse:
    """``page()`` extended with an ``included`` envelope member, for these tests only."""

    body = json.loads(page(resources, **kwargs).text)  # type: ignore[arg-type]
    body["included"] = included
    return response(body)


def make_multi(query_responses: list[FakeResponse]) -> tuple[OptimadeStore, QueryClient]:
    """Multi-endpoint discovery: a generic root ('materials'), a bound related
    'structures' endpoint, and a generic '_httk_records' related endpoint --
    for link predicates, link outputs, and the ``.links`` accessor.

    The discovery documents are mutated on the fake client before
    ``OptimadeStore`` is constructed, since discovery is eager at construction.
    """

    base = "https://example.test/v1"
    client = QueryClient(
        "materials",
        {"_anyterial_formula": {"type": "string"}},
        query_responses,
        describedby=None,
        base_url=base,
    )
    client.discovery[base + "/info"] = response(
        {
            "data": {
                "type": "info",
                "attributes": {"available_endpoints": ["materials", "structures", "_httk_records"]},
            }
        }
    )
    client.discovery[base + "/info/structures"] = response(
        {
            "data": {
                "type": "info",
                "properties": schema_properties(
                    STRUCTURES,
                    ("id", "type", "nelements", "elements", "chemical_formula_reduced"),
                    # chemical_formula_reduced's remote (wire) name differs from
                    # its local name; nelements is left unrenamed so filters
                    # against it match the plain "structures.nelements" spelling.
                    renames={"chemical_formula_reduced": "formula_reduced"},
                ),
            },
            "links": {"describedby": STRUCTURES},
        }
    )
    client.discovery[base + "/info/_httk_records"] = response(
        {"data": {"type": "info", "properties": {"_anyterial_kind": {"type": "string"}}}}
    )
    store = OptimadeStore(base, client=client)
    return store, client


def make_shadowed(query_responses: list[FakeResponse]) -> tuple[OptimadeStore, QueryClient]:
    """A root type whose own advertised property shares a name with a served type."""

    base = "https://example.test/v1"
    client = QueryClient(
        "widgets",
        {"references": {"type": "string"}},
        query_responses,
        describedby=None,
        base_url=base,
    )
    client.discovery[base + "/info"] = response(
        {"data": {"type": "info", "attributes": {"available_endpoints": ["widgets", "references"]}}}
    )
    client.discovery[base + "/info/references"] = response(
        {"data": {"type": "info", "properties": {"id": {"type": "string"}, "type": {"type": "string"}}}}
    )
    store = OptimadeStore(base, client=client)
    return store, client


def make_linked_relationships() -> dict[str, object]:
    return {
        # The block key differs from the identifier's own "type" -- a
        # StrongLink-shaped edge -- so resolution must key off the
        # identifier, never this block key.
        "_httk_has_input": {"data": [{"type": "structures", "id": "s1"}]},
        "structures": {"data": [{"type": "structures", "id": "s1"}]},
        "_httk_records": {"data": [{"type": "_httk_records", "id": "r1"}, {"type": "_httk_records", "id": "r2"}]},
        "empty_rel": {"data": []},
        "weird_rel": {"data": [{"type": "unknown-type", "id": "x"}]},
    }


# --------------------------------------------------------------- predicates


def test_dotted_relationship_filter_uses_related_types_remote_names() -> None:
    store, client = make_multi([page([])])
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    searcher.add(
        (material.links.structures.nelements > 2) & (material.links.structures.chemical_formula_reduced == "Fe2O3")
    )

    [row for row in searcher.results(item=material)]

    rendered = query_params(client.requests[-1])["filter"][0]
    assert "structures.nelements > 2" in rendered
    # chemical_formula_reduced is discovered under the wire name
    # "formula_reduced"; the rendered filter must use that remote name.
    assert 'structures.formula_reduced = "Fe2O3"' in rendered
    parse_optimade_filter(rendered)


def test_underscore_prefixed_served_type_traverses_via_relationship() -> None:
    store, client = make_multi([page([])])
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    searcher.add(material.links._httk_records._anyterial_kind == "note")

    [row for row in searcher.results(item=material)]

    rendered = query_params(client.requests[-1])["filter"][0]
    assert '_httk_records._anyterial_kind = "note"' in rendered


def test_related_field_errors_and_depth_limit_happen_before_http() -> None:
    store, client = make_multi([])
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    discovery_requests = len(client.requests)

    with pytest.raises(UnsupportedQueryError, match=r"field 'bogus' for related type 'structures'"):
        _value = material.links.structures.bogus
    with pytest.raises(UnsupportedQueryError, match="traversal"):
        _value = material.links.structures.nelements.deeper
    with pytest.raises(UnsupportedQueryError, match="related fields as outputs"):
        searcher._output(material.links.structures.nelements, "x")
    with pytest.raises(UnsupportedQueryError, match="related fields as outputs"):
        searcher.add_sort(material.links.structures.nelements)
    with pytest.raises(UnsupportedQueryError, match="related fields as outputs"):
        searcher.results(x=material.links.structures.nelements)
    with pytest.raises(UnsupportedQueryError, match="not a served entry type"):
        _value = material.links._httk_has_input.nelements
    with pytest.raises(AttributeError):
        _value = material.links.structures._ipython_canary_method_should_not_exist_
    with pytest.raises(AttributeError):
        _value = material._ipython_canary_method_should_not_exist_

    assert len(client.requests) == discovery_requests


def test_old_shortcut_of_naming_a_served_type_directly_is_unsupported() -> None:
    store, _client = make_multi([])
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))

    with pytest.raises(UnsupportedQueryError, match="field 'structures'"):
        _value = material.structures


def test_root_field_shadows_a_same_named_served_type() -> None:
    store, client = make_shadowed([page([resource("w1", "widgets", {"references": "abc"})])])
    searcher = store.searcher()
    variable = searcher.variable(store.entry_type("widgets"))
    # If "references" had resolved as a relationship namespace instead of the
    # root's own field, this comparison would not yield a _RemoteExpression
    # and add() would reject it before any request was made.
    searcher.add(variable.references == "abc")

    [row for row in searcher.results(item=variable)]

    assert 'references = "abc"' in query_params(client.requests[-1])["filter"][0]


def test_root_property_named_links_is_shadowed_by_the_namespace() -> None:
    base = "https://example.test/v1"
    client = QueryClient("widgets", {"links": {"type": "string"}}, [], describedby=None, base_url=base)
    store = OptimadeStore(base, client=client)
    searcher = store.searcher()
    variable = searcher.variable(store.entry_type("widgets"))

    from httk.store.optimade.remote_query import _RemoteLinks

    assert isinstance(variable.links, _RemoteLinks)


# ------------------------------------------------------------------ outputs


def test_link_output_adds_ordered_deduplicated_include_and_resolves_from_included() -> None:
    included = [
        resource("s1", "structures", {"nelements": 3, "elements": ["Fe", "O"], "formula_reduced": "Fe2O3"}),
        resource("r1", "_httk_records", {"_anyterial_kind": "note"}),
    ]
    relationships = make_linked_relationships()
    store, client = make_multi(
        [
            page_with_included(
                [resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)],
                included=included,
            ),
            page([resource("r2", "_httk_records", {"_anyterial_kind": "info2"})]),
            page([], returned=1),
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(
        item=material,
        records=material.links._httk_records,
        structures=material.links.structures,
    ).one()

    query_url = None
    for requested in client.requests:
        if urlsplit(requested).path.endswith("/materials"):
            query_url = requested
    assert query_url is not None
    assert query_params(query_url)["include"] == ["_httk_records,structures"]

    assert isinstance(row.structures, tuple)
    (structure,) = row.structures
    assert isinstance(structure, OptimadeStructure)
    assert structure.id == "s1"
    assert structure.unwrap().member == "included"

    r1, r2 = row.records
    assert isinstance(r1, OptimadeResource)
    assert r1.id == "r1"
    assert r1.member == "included"
    assert isinstance(r2, OptimadeResource)
    assert r2.id == "r2"
    assert r2.member == "data"

    assert row.item.links.structures == row.structures
    assert row.item.links._httk_records == row.records

    # count() must never carry include=.
    assert searcher.count() == 1
    assert "include" not in query_params(client.requests[-1])

    assert client.query_responses == []


def test_wire_key_output_adds_no_include_and_resolves_by_identifier_type() -> None:
    included = [resource("s1", "structures", {"nelements": 3})]
    relationships = {"_httk_has_input": {"data": [{"type": "structures", "id": "s1"}]}}
    store, client = make_multi(
        [
            page_with_included(
                [resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)],
                included=included,
            )
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(item=material, records=material.links._httk_has_input).one()

    query_url = next(r for r in client.requests if urlsplit(r).path.endswith("/materials"))
    assert "include" not in query_params(query_url)
    (target,) = row.records
    assert isinstance(target, OptimadeStructure)
    assert target.id == "s1"


def test_link_output_missing_from_included_performs_exactly_one_lazy_fetch_per_identifier() -> None:
    relationships = {"_httk_records": {"data": [{"type": "_httk_records", "id": "r1"}]}}
    store, client = make_multi(
        [
            page([resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)]),
            page([resource("r1", "_httk_records", {"_anyterial_kind": "note"})]),
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    requests_before = len(client.requests)
    row = searcher.results(item=material, records=material.links._httk_records).one()

    assert len(client.requests) == requests_before + 2  # one query page + one lazy fetch
    fetch_url = client.requests[-1]
    assert urlsplit(fetch_url).path.endswith("/_httk_records")
    assert query_params(fetch_url)["filter"] == ['(id = "r1")']
    (target,) = row.records
    assert target.id == "r1"
    assert client.query_responses == []


def test_link_output_missing_target_raises_client_error() -> None:
    relationships = {"_httk_records": {"data": [{"type": "_httk_records", "id": "gone"}]}}
    store, _client = make_multi(
        [
            page([resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)]),
            page([]),
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))

    with pytest.raises(OptimadeClientError, match="did not resolve to exactly one resource"):
        searcher.results(item=material, records=material.links._httk_records).one()


def test_link_output_resolves_per_row_with_distinct_order() -> None:
    """Two rows with distinct, non-overlapping-order target sets: each row's
    tuple is resolved and ordered independently -- proving resolution (and its
    memoization) is per record instance, not shared or order-confused across
    rows of the same page."""

    included = [
        resource("s1", "structures", {"nelements": 3}),
        resource("s2", "structures", {"nelements": 5}),
    ]
    store, _client = make_multi(
        [
            page_with_included(
                [
                    resource(
                        "m1",
                        "materials",
                        {"_anyterial_formula": "Fe2O3"},
                        relationships={"structures": {"data": [{"type": "structures", "id": "s1"}]}},
                    ),
                    resource(
                        "m2",
                        "materials",
                        {"_anyterial_formula": "Fe3O4"},
                        relationships={
                            "structures": {
                                "data": [{"type": "structures", "id": "s2"}, {"type": "structures", "id": "s1"}]
                            }
                        },
                    ),
                ],
                included=included,
                returned=2,
            )
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    # A list comprehension, not list(): RemoteResultSet also defines __len__,
    # and list() opportunistically calls it as a size hint, which would spend
    # this test's single queued response on a spurious count() request.
    row1, row2 = [row for row in searcher.results(item=material, structures=material.links.structures)]

    assert [structure.id for structure in row1.structures] == ["s1"]
    assert [structure.id for structure in row2.structures] == ["s2", "s1"]


# -------------------------------------------------------------- bound records


def test_bound_equals_plain_hashes_equal_and_isinstance_holds() -> None:
    included = [resource("s1", "structures", {"nelements": 3})]
    relationships = {"structures": {"data": [{"type": "structures", "id": "s1"}]}}
    store, _client = make_multi(
        [
            page_with_included(
                [resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)],
                included=included,
            )
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(structures=material.links.structures).one()
    (bound,) = row.structures

    assert isinstance(bound, OptimadeStructure)
    plain = OptimadeStructure(bound.resource)
    assert bound == plain
    assert plain == bound
    assert hash(bound) == hash(plain)
    assert plain in {bound}

    # OptimadeStructure is @dataclass(frozen=True, init=False) with a custom
    # __init__(obj, **hints); the bound subclass's __reduce_ex__ must still
    # round-trip it correctly through that constructor.
    restored = pickle.loads(pickle.dumps(bound))
    assert type(restored) is OptimadeStructure
    assert restored == plain


def test_bound_pickles_and_deep_copies_as_the_plain_class() -> None:
    store, _client = make_files([page([resource("f-1", "renamed-files")])])
    searcher = store.searcher()
    variable = searcher.variable(store.entry_types[0])
    row = searcher.results(record=variable).one()
    bound = row.record
    plain_class = type(bound).__mro__[1]  # the plain backend class bound subclasses
    assert type(bound) is not plain_class

    restored = pickle.loads(pickle.dumps(bound))
    assert type(restored) is plain_class
    assert restored == bound

    deep = copy.deepcopy(bound)
    assert type(deep) is plain_class
    assert deep == bound

    replaced = dataclasses.replace(bound)
    with pytest.raises(AttributeError, match="links"):
        _value = replaced.links


def test_generic_bound_resource_pickles_as_plain_optimade_resource() -> None:
    included = [resource("s1", "structures", {"nelements": 3})]
    relationships = {"structures": {"data": [{"type": "structures", "id": "s1"}]}}
    store, _client = make_multi(
        [
            page_with_included(
                [resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)],
                included=included,
            )
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(item=material).one()
    bound = row.item

    assert isinstance(bound, OptimadeResource)
    plain = OptimadeResource(bound.document, bound.data_index, bound.schema, bound.member)
    assert bound == plain

    restored = pickle.loads(pickle.dumps(bound))
    assert type(restored) is OptimadeResource
    assert restored == plain


def test_hop_chaining_works_on_both_included_and_lazily_fetched_records() -> None:
    """A further ``.links`` hop must work on both an included-wrapped record
    (resolved with no extra fetch) and a lazily fetched one (resolved by its
    own request): both are bound the same way, via _wrap."""

    included = [
        resource(
            "s1",
            "structures",
            {"nelements": 3},
            relationships={"empty_rel": {"data": []}},
        )
    ]
    store, client = make_multi(
        [
            page_with_included(
                [
                    resource(
                        "m1",
                        "materials",
                        {"_anyterial_formula": "Fe2O3"},
                        relationships={
                            "structures": {"data": [{"type": "structures", "id": "s1"}]},
                            "_httk_records": {"data": [{"type": "_httk_records", "id": "r1"}]},
                        },
                    )
                ],
                included=included,
            ),
            # r1 is not in included, so it resolves via one lazy fetch, and its
            # own page carries a relationship for the second hop to chase.
            page(
                [
                    resource(
                        "r1",
                        "_httk_records",
                        {"_anyterial_kind": "note"},
                        relationships={"empty_rel": {"data": []}},
                    )
                ]
            ),
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(item=material).one()

    (structure,) = row.item.links.structures
    assert structure.links.empty_rel == ()

    (record,) = row.item.links._httk_records
    assert record.links.empty_rel == ()

    assert client.query_responses == []


def test_save_of_a_bound_resource_round_trips_through_an_in_memory_sql_store() -> None:
    from httk.store.backend.sql import Backend, SqlStore

    included = [resource("s1", "structures", {"nelements": 3})]
    relationships = {"structures": {"data": [{"type": "structures", "id": "s1"}]}}
    store, _client = make_multi(
        [
            page_with_included(
                [resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)],
                included=included,
            )
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(item=material).one()
    bound = row.item  # a generic (unbound-backend) OptimadeResource
    plain = OptimadeResource(bound.document, bound.data_index, bound.schema, bound.member)

    with Backend.sqlite() as database:
        sql_store = SqlStore(database, entry_records={})
        sid = sql_store.save(bound)
        fetched = sql_store.fetch(OptimadeResource, sid)
        assert fetched == plain


def test_closed_store_rejects_links_access() -> None:
    included = [resource("s1", "structures", {"nelements": 3})]
    relationships = {"structures": {"data": [{"type": "structures", "id": "s1"}]}}
    store, _client = make_multi(
        [
            page_with_included(
                [resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)],
                included=included,
            )
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(item=material).one()

    store.close()
    with pytest.raises(OptimadeClientError, match="closed"):
        _value = row.item.links.structures


def test_unknown_relationship_key_lists_available_relationships() -> None:
    relationships = {"structures": {"data": [{"type": "structures", "id": "s1"}]}}
    store, _client = make_multi(
        [
            page_with_included(
                [resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)],
                included=[resource("s1", "structures", {"nelements": 3})],
            )
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(item=material).one()

    with pytest.raises(OptimadeClientError, match=r"no relationship 'no_such_rel'.*'structures'"):
        _value = row.item.links.no_such_rel


def test_empty_relationship_and_unknown_identifier_type_and_malformed_included() -> None:
    relationships = make_linked_relationships()
    store, _client = make_multi(
        [
            page_with_included(
                [resource("m1", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)],
                included=[resource("s1", "structures", {"nelements": 3})],
            )
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(item=material).one()

    assert row.item.links.empty_rel == ()
    with pytest.raises(OptimadeClientError, match="unknown entry type 'unknown-type'"):
        _value = row.item.links.weird_rel


def test_related_rejects_malformed_included_member() -> None:
    relationships = {"rel": {"data": [{"type": "structures", "id": "sX"}]}}
    store, _client = make_multi(
        [
            page_with_included(
                [resource("m2", "materials", {"_anyterial_formula": "Fe2O3"}, relationships=relationships)],
                included=[{"type": "structures"}],  # missing required 'id'
            )
        ]
    )
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(item=material).one()

    with pytest.raises(OptimadeResponseError, match="included"):
        _value = row.item.links.rel


def test_related_rejects_non_array_included() -> None:
    relationships = {"rel": {"data": [{"type": "structures", "id": "sX"}]}}
    body = json.loads(page([resource("m3", "materials", {}, relationships=relationships)]).text)
    body["included"] = {"not": "an array"}
    store, _client = make_multi([response(body)])
    searcher = store.searcher()
    material = searcher.variable(store.entry_type("materials"))
    row = searcher.results(item=material).one()

    with pytest.raises(OptimadeResponseError, match="'included' must be an array"):
        _value = row.item.links.rel


# ------------------------------------------------------------- meta.warnings


def test_meta_warnings_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    store, _client = make_files(
        [
            response(
                {
                    "data": [resource("f-1", "renamed-files")],
                    "meta": {
                        "data_returned": 1,
                        "warnings": [{"detail": "filter references unknown property 'nope'"}],
                    },
                }
            )
        ]
    )
    searcher = store.searcher()
    variable = searcher.variable(store.entry_types[0])

    with caplog.at_level(logging.WARNING, logger="httk.store.optimade.remote_query"):
        [row for row in searcher.results(record=variable)]

    assert any("filter references unknown property" in message for message in caplog.messages)
    assert any(getattr(rec, "context", None) == "optimade" for rec in caplog.records)
    assert any(rec.name == "httk.store.optimade.remote_query" for rec in caplog.records)
