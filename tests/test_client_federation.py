"""Real-ASGI federation coverage for the remote OPTIMADE client.

Each ``OptimadeStore`` talks to an independently constructed in-process ASGI
application.  ``AsgiSyncClient`` rejects any other host, keeping this an
integration test of the protocol client without permitting network access.
"""

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from httk.atomistic import OptimadeStructure
from httk.core import EntryProvider, EntryTypeDefinition, load_entry_type_definition
from httk.serve.optimade import adapter_from_providers, create_asgi_app

from httk.store import Backend, FederatedSourceError, FederatedStore, MultipleResultsError, SqlStore
from httk.store.optimade import OptimadeStore, OptimadeTransportError


class AsgiSyncClient:
    """A minimal blocking bridge over ASGI that cannot escape ``base_url``."""

    def __init__(self, app: Any, *, base_url: str, fail_queries: bool = False) -> None:
        self.app = app
        self.base_url = base_url
        self.fail_queries = fail_queries
        self.requests: list[str] = []
        self.closed = False

    def get(self, url: str) -> httpx.Response:
        assert urlsplit(url).netloc == urlsplit(self.base_url).netloc
        self.requests.append(url)
        if self.fail_queries and "?" in url:
            raise RuntimeError("deliberate source-specific transport failure")

        async def request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport) as client:
                return await client.get(url)

        return asyncio.run(request())

    def close(self) -> None:
        self.closed = True


class _StructuresProvider(EntryProvider):
    """Serve one or more structure endpoint spellings from fixed rows."""

    _definition = load_entry_type_definition("https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures")
    _keys: ClassVar[dict[str, str]] = {
        "id": "id",
        "type": "type",
        "elements": "elements",
        "nelements": "nelements",
        "chemical_formula_reduced": "chemical_formula_reduced",
    }

    def __init__(self, rows_by_endpoint: Mapping[str, Iterable[Mapping[str, object]]]) -> None:
        self._rows = {name: tuple(dict(row) for row in rows) for name, rows in rows_by_endpoint.items()}

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {name: self._definition for name in self._rows}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        assert entry_type in self._rows
        return self._keys

    def records(self, entry_type: str) -> Iterable[Mapping[str, object]]:
        return self._rows[entry_type]


def _row(identifier: str, endpoint: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": endpoint,
        "elements": ["Cl", "Na"],
        "nelements": 2,
        "chemical_formula_reduced": "ClNa",
    }


def _remote(
    base_url: str,
    rows_by_endpoint: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    fail_queries: bool = False,
    page_limit: int = 1,
) -> tuple[OptimadeStore, AsgiSyncClient]:
    provider = _StructuresProvider(rows_by_endpoint)
    app = create_asgi_app(adapter_from_providers([provider]), baseurl=base_url)
    client = AsgiSyncClient(app, base_url=base_url, fail_queries=fail_queries)
    return OptimadeStore(base_url, client=client, page_limit=page_limit), client


def _federated_structures(
    federation: FederatedStore,
    alpha: OptimadeStore,
    beta: OptimadeStore,
) -> tuple[object, object]:
    """Bind exact descriptors because beta's typed backend is ambiguous."""

    target = federation.target(
        "structures",
        {
            "alpha": alpha.entry_type("structures"),
            "beta": beta.entry_type("vendor-structures"),
        },
    )
    searcher = federation.searcher()
    return searcher, searcher.variable(target)


def _portable_filter(structure: Any) -> object:
    return (structure.nelements == 2) & structure.elements.has("Na")


def _query_urls(client: AsgiSyncClient) -> list[str]:
    return [url for url in client.requests if "?" in url]


def test_federated_real_asgi_sources_paginate_source_major_and_keep_remote_resources_exact() -> None:
    """Two independent endpoints remain a union, including duplicate IDs."""

    alpha, alpha_client = _remote(
        "http://alpha.test",
        {"structures": (_row("shared", "structures"), _row("alpha", "structures"))},
    )
    # Two endpoints have the same typed backend, so a direct OptimadeStructure
    # binding is ambiguous for this source.  The explicit descriptor target is
    # therefore required and retains the exact discovery-generation objects.
    beta, beta_client = _remote(
        "http://beta.test",
        {
            "other-structures": (),
            "vendor-structures": (_row("shared", "vendor-structures"), _row("beta", "vendor-structures")),
        },
        page_limit=2,
    )
    federation = FederatedStore({"alpha": alpha, "beta": beta})
    searcher, structure = _federated_structures(federation, alpha, beta)
    searcher.add(_portable_filter(structure))
    results = searcher.results(
        record=structure, identifier=structure.id, formula=structure.chemical_formula_reduced, origin=searcher.origin
    )

    rows = list(results)
    assert [(row.origin, row.identifier, row.formula) for row in rows] == [
        ("alpha", "shared", "ClNa"),
        ("alpha", "alpha", "ClNa"),
        ("beta", "shared", "ClNa"),
        ("beta", "beta", "ClNa"),
    ]
    assert rows[0].record.id == rows[2].record.id == "shared"
    assert rows[0].record is not rows[2].record
    assert all(isinstance(row.record, OptimadeStructure) for row in rows)
    assert rows[0].record.resource.document.source_url.startswith("http://alpha.test")
    assert rows[2].record.resource.document.source_url.startswith("http://beta.test")
    assert rows[0].record.resource.unwrap()["id"] == "shared"

    assert searcher.count() == 4
    assert len(results) == 4
    assert results.first() is not None
    assert results.first().identifier == "shared"  # type: ignore[union-attr]
    with pytest.raises(MultipleResultsError, match="more than one"):
        results.one()

    limited_searcher, limited_structure = _federated_structures(federation, alpha, beta)
    limited_searcher.add(_portable_filter(limited_structure))
    limited_searcher.add_offset(1)
    limited_searcher.set_limit(2)
    limited = limited_searcher.results(identifier=limited_structure.id, origin=limited_searcher.origin)
    assert [(row.origin, row.identifier) for row in limited] == [("alpha", "alpha"), ("beta", "shared")]
    assert len(limited) == 2

    one_searcher, one_structure = _federated_structures(federation, alpha, beta)
    one_searcher.add(one_structure.id == "beta")
    assert one_searcher.results(record=one_structure, origin=one_searcher.origin).one().origin == "beta"

    for client, configured_page_limit in ((alpha_client, 1), (beta_client, 2)):
        query_urls = _query_urls(client)
        assert query_urls
        requested_page_limits = [int(parse_qs(urlsplit(url).query)["page_limit"][0]) for url in query_urls]
        assert configured_page_limit in requested_page_limits
        assert all(1 <= requested <= configured_page_limit for requested in requested_page_limits)
        assert all(urlsplit(url).netloc == urlsplit(client.base_url).netloc for url in client.requests)
        assert not client.closed


def test_federated_real_asgi_source_can_be_empty_without_changing_the_other_source() -> None:
    """An empty child participates independently and contributes no phantom rows."""

    alpha, alpha_client = _remote("http://alpha.test", {"structures": (_row("alpha", "structures"),)})
    beta, beta_client = _remote("http://beta.test", {"vendor-structures": ()})
    federation = FederatedStore({"alpha": alpha, "beta": beta})
    target = federation.target(
        "structures",
        {"alpha": alpha.entry_type("structures"), "beta": beta.entry_type("vendor-structures")},
    )
    searcher = federation.searcher()
    structure = searcher.variable(target)
    searcher.add(_portable_filter(structure))
    results = searcher.results(identifier=structure.id, origin=searcher.origin)

    assert [(row.origin, row.identifier) for row in results] == [("alpha", "alpha")]
    assert searcher.count() == len(results) == 1
    assert _query_urls(alpha_client)
    assert _query_urls(beta_client)


def test_federated_remote_and_sqlite_share_a_typed_backend_with_filter_projection_and_counts() -> None:
    """A local cache and its live source preserve typed OPTIMADE record values."""

    remote, client = _remote(
        "http://remote.test",
        {"structures": (_row("shared", "structures"), _row("remote-only", "structures"))},
    )
    direct_searcher = remote.searcher()
    direct_structure = direct_searcher.variable(OptimadeStructure)
    direct_searcher.add(_portable_filter(direct_structure))
    remote_backends = [row.record for row in direct_searcher.results(record=direct_structure)]
    saved = remote_backends[0]

    with Backend.sqlite() as database:
        local = SqlStore(database, entry_records={})
        local.save(saved)
        federation = FederatedStore({"remote": remote, "local": local})
        searcher = federation.searcher()
        structure = searcher.variable(OptimadeStructure)
        searcher.add(_portable_filter(structure))
        results = searcher.results(
            record=structure,
            identifier=structure.id,
            formula=structure.chemical_formula_reduced,
            origin=searcher.origin,
        )

        rows = list(results)
        assert [(row.origin, row.identifier, row.formula) for row in rows] == [
            ("remote", "shared", "ClNa"),
            ("remote", "remote-only", "ClNa"),
            ("local", "shared", "ClNa"),
        ]
        assert all(isinstance(row.record, OptimadeStructure) for row in rows)
        assert isinstance(rows[-1].record, OptimadeStructure)
        assert rows[-1].record.resource.document == saved.resource.document
        assert rows[-1].record.resource.schema == saved.resource.schema
        assert rows[-1].record.resource.unwrap() == saved.resource.unwrap()
        assert searcher.count() == len(results) == 3

    assert _query_urls(client)


def test_federated_source_transport_failure_after_discovery_is_never_partial_success() -> None:
    """A later source failure is attributed and chains its client failure."""

    alpha, _alpha_client = _remote("http://alpha.test", {"structures": (_row("alpha", "structures"),)})
    beta, beta_client = _remote(
        "http://beta.test",
        {"vendor-structures": (_row("beta", "vendor-structures"),)},
        fail_queries=True,
    )
    federation = FederatedStore({"alpha": alpha, "beta": beta})
    target = federation.target(
        "structures",
        {"alpha": alpha.entry_type("structures"), "beta": beta.entry_type("vendor-structures")},
    )
    searcher = federation.searcher()
    structure = searcher.variable(target)
    results = iter(searcher.results(identifier=structure.id, origin=searcher.origin))

    assert next(results).identifier == "alpha"
    with pytest.raises(FederatedSourceError, match="beta") as excinfo:
        next(results)

    assert excinfo.value.source == "beta"
    assert isinstance(excinfo.value.__cause__, OptimadeTransportError)
    assert _query_urls(beta_client)
