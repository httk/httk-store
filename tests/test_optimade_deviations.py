"""No-network tests for the OPTIMADE client's specification-deviation tolerances.

The client tolerates three deviations that Materials Project (the largest public
OPTIMADE provider) exhibits -- a missing ``/versions`` endpoint at the
unversioned base, an ``/info/<entry>`` document lacking the 1.2 resource
``type`` member, and a continuation link that downgraded its scheme to ``http``
on an ``https`` host -- each through a specification-anchored fallback that is
recorded on the store and warned through the report channel, and each switched
off by ``tolerate_deviations=False``. These tests drive that behaviour entirely
through the synchronous client against fake transports, touching no network.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pytest
from httk.core.register import optimade_entry_binding

from httk.store.optimade import (
    OptimadeDiscoveryError,
    OptimadeHTTPError,
    OptimadePaginationError,
    OptimadeStore,
)

_FIXTURES = Path(__file__).parent / "data" / "optimade_info"
_FIXTURE_NAME = "materials_project_info_structures.json"

_HOST = "optimade.example.test"
_BASE = f"https://{_HOST}"
_V1 = _BASE + "/v1"
_STRUCTURES_PATH = _V1 + "/structures"
_HTTP_NEXT = f"http://{_HOST}/v1/structures?page_offset=1"
_HTTPS_NEXT = f"https://{_HOST}/v1/structures?page_offset=1"


def _require_structures_table() -> None:
    """Skip unless the installed httk-atomistic declares the structures version table.

    Standard-name completion of ``structures`` is gated by the
    ``standard_property_versions`` table that *httk-atomistic* registers on its
    binding (httk-atomistic >= 2.1.1); an older release leaves the endpoint
    generic, so these tests cannot run against it.
    """
    pytest.importorskip("httk.atomistic")
    binding = optimade_entry_binding("https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures")
    if binding is None or not binding.standard_property_versions:
        pytest.skip("installed httk-atomistic declares no standard_property_versions for structures")


@dataclass
class FakeResponse:
    status_code: int
    text: str


class FakeClient:
    """A borrowed HTTP client answering from a fixed URL map and page queues.

    Discovery responses (``fixed``) are matched by exact URL and then by path
    with the query string dropped, and answer any number of times, so the
    versions fallback's repeated ``/v1/info`` probe resolves. Query pages
    (``pages``, keyed by path) are matched by path and consumed in order.
    """

    def __init__(
        self,
        fixed: dict[str, FakeResponse],
        pages: dict[str, list[FakeResponse]] | None = None,
    ) -> None:
        self.fixed = dict(fixed)
        self.pages = {path: list(queue) for path, queue in (pages or {}).items()}
        self.requests: list[str] = []
        self.closed = False

    def get(self, url: str) -> FakeResponse:
        self.requests.append(url)
        if url in self.fixed:
            return self.fixed[url]
        path = url.split("?", 1)[0]
        if path in self.fixed:
            return self.fixed[path]
        queue = self.pages.get(path)
        if queue:
            return queue.pop(0)
        raise AssertionError(f"unexpected external request: {url}")

    def close(self) -> None:
        self.closed = True


def _json_response(value: object, status_code: int = 200) -> FakeResponse:
    return FakeResponse(status_code, json.dumps(value))


def _top_info(endpoints: list[str], api_version: str | None) -> FakeResponse:
    attributes: dict[str, object] = {"available_endpoints": endpoints}
    if api_version is not None:
        attributes["api_version"] = api_version
    return _json_response({"data": {"type": "info", "attributes": attributes}})


def _fixture_text() -> str:
    """Return the real Materials-Project ``/info/structures`` body, verbatim.

    The captured document declares ``data.id`` ``"structures"`` and no
    ``data.type`` -- exactly the deviation under test.
    """

    return (_FIXTURES / _FIXTURE_NAME).read_text()


def _fixture_mutated(**data_overrides: object) -> FakeResponse:
    """Return the fixture with its ``data`` object updated for a negative case."""

    document = json.loads(_fixture_text())
    document["data"] = {**document["data"], **data_overrides}
    return _json_response(document)


def _structure(identifier: str) -> dict[str, object]:
    """One valid ``structures`` resource with a naive (offset-free) timestamp.

    The naive ``last_modified`` mirrors the provider; its decoding is owned by
    the entry backends in *httk-core*/*httk-atomistic* and is not asserted here.
    """

    return {
        "id": identifier,
        "type": "structures",
        "attributes": {
            "last_modified": "2023-02-11T01:06:23.403000",
            "elements": ["Cl", "Na"],
            "nelements": 2,
            "elements_ratios": [0.5, 0.5],
            "chemical_formula_reduced": "ClNa",
            "chemical_formula_descriptive": "ClNa",
            "dimension_types": [1, 1, 1],
            "nperiodic_dimensions": 3,
            "lattice_vectors": [[0.0, 2.8, 2.8], [2.8, 0.0, 2.8], [2.8, 2.8, 0.0]],
            "cartesian_site_positions": [[0.0, 0.0, 0.0], [2.8, 2.8, 2.8]],
            "nsites": 2,
            "species": [
                {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
                {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
            ],
            "species_at_sites": ["Na", "Cl"],
            "structure_features": [],
        },
    }


def _page(resources: list[object], *, next_link: object) -> FakeResponse:
    value: dict[str, object] = {
        "data": resources,
        "meta": {"more_data_available": next_link is not None, "data_returned": len(resources)},
        "links": {"next": next_link},
    }
    return _json_response(value)


def _materials_project_client() -> FakeClient:
    """A fake exhibiting all three tolerated deviations of Materials Project."""

    fixed = {
        _BASE + "/versions": FakeResponse(404, "not found"),
        _V1 + "/info": _top_info(["structures"], "1.2.0"),
        _V1 + "/info/structures": FakeResponse(200, _fixture_text()),
    }
    pages = {
        _STRUCTURES_PATH: [
            _page([_structure("mp-1")], next_link=_HTTP_NEXT),
            _page([_structure("mp-2")], next_link=None),
        ]
    }
    return FakeClient(fixed, pages)


# --- The three tolerances applied together, end to end ---------------------


def test_materials_project_like_service_is_tolerated_and_recorded(caplog: pytest.LogCaptureFixture) -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    client = _materials_project_client()
    with caplog.at_level(logging.WARNING):
        store = OptimadeStore(_BASE, client=client)

        assert store.base_url.endswith("/v1")
        assert store.api_version == "1.2.0"

        descriptor = store.entry_type("structures")
        assert descriptor.backend is OptimadeStructure
        assert descriptor.binding_evidence == "standard-name"

        searcher = store.searcher()
        variable = searcher.variable(OptimadeStructure)
        ids = [row.structure.id for row in searcher.results(structure=variable)]

    # Both pages were yielded, and the continuation was followed over https.
    assert ids == ["mp-1", "mp-2"]
    assert _HTTPS_NEXT in client.requests
    assert _HTTP_NEXT not in client.requests

    by_kind = {deviation.kind: deviation for deviation in store.deviations}
    assert set(by_kind) == {"versions-endpoint", "entry-info-identity", "continuation-scheme"}
    assert len(store.deviations) == 3
    assert by_kind["versions-endpoint"].url == _BASE + "/versions"
    assert by_kind["versions-endpoint"].detail == (
        "the versions endpoint is missing at the unversioned base; /v1/info declares a major-1 service"
    )
    assert by_kind["entry-info-identity"].url == _V1 + "/info/structures"
    assert by_kind["entry-info-identity"].detail == (
        "/info/<name> data lacks the 1.2 resource 'type' member; identity established from data.id"
    )
    # The continuation-scheme deviation is recorded once per service, keyed on
    # the stable transport base URL rather than any one continuation link.
    assert by_kind["continuation-scheme"].url == _V1
    assert by_kind["continuation-scheme"].detail == (
        "continuation link downgraded scheme to http; upgraded to https for the same host"
    )

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and getattr(record, "context", None) == "optimade"
    ]
    assert len(warnings) == 3


# --- tolerate_deviations=False restores today's strict errors --------------


def test_strict_mode_fails_on_missing_versions_endpoint() -> None:
    client = _materials_project_client()
    with pytest.raises(OptimadeHTTPError) as excinfo:
        OptimadeStore(_BASE, client=client, tolerate_deviations=False)
    assert excinfo.value.status_code == 404


def test_strict_mode_fails_on_missing_entry_info_type() -> None:
    fixed = {
        _V1 + "/info": _top_info(["structures"], "1.2.0"),
        _V1 + "/info/structures": FakeResponse(200, _fixture_text()),
    }
    client = FakeClient(fixed)
    with pytest.raises(OptimadeDiscoveryError, match="data.type must be 'info'"):
        OptimadeStore(_V1, client=client, tolerate_deviations=False)


def test_strict_mode_fails_on_http_continuation() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    fixed = {
        _V1 + "/info": _top_info(["structures"], "1.2.0"),
        _V1 + "/info/structures": _fixture_mutated(type="info"),
    }
    pages = {
        _STRUCTURES_PATH: [
            _page([_structure("mp-1")], next_link=_HTTP_NEXT),
            _page([_structure("mp-2")], next_link=None),
        ]
    }
    store = OptimadeStore(_V1, client=FakeClient(fixed, pages), tolerate_deviations=False)
    searcher = store.searcher()
    variable = searcher.variable(OptimadeStructure)
    with pytest.raises(OptimadePaginationError, match="cross-origin"):
        [row.structure.id for row in searcher.results(structure=variable)]


# --- Negative cases: the fallbacks stay narrow -----------------------------


def test_non_404_versions_error_is_never_tolerated() -> None:
    client = FakeClient({_BASE + "/versions": FakeResponse(500, "boom")})
    with pytest.raises(OptimadeHTTPError) as excinfo:
        OptimadeStore(_BASE, client=client)
    assert excinfo.value.status_code == 500


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(_top_info(["structures"], "2.0.0"), id="unsupported-major"),
        pytest.param(_top_info(["structures"], None), id="no-declared-version"),
        pytest.param(
            _json_response({"data": {"type": "notinfo", "attributes": {"api_version": "1.2.0"}}}),
            id="wrong-data-type",
        ),
    ],
)
def test_versions_fallback_reraises_original_404_when_probe_is_not_major_one(probe: FakeResponse) -> None:
    client = FakeClient({_BASE + "/versions": FakeResponse(404, "nf"), _V1 + "/info": probe})
    with pytest.raises(OptimadeHTTPError) as excinfo:
        OptimadeStore(_BASE, client=client)
    assert excinfo.value.status_code == 404
    # The informative probe failure is chained onto the re-raised 404.
    assert isinstance(excinfo.value.__cause__, OptimadeDiscoveryError)


def test_entry_info_identity_requires_matching_id() -> None:
    fixed = {
        _V1 + "/info": _top_info(["structures"], "1.2.0"),
        _V1 + "/info/structures": _fixture_mutated(id="not-structures"),
    }
    with pytest.raises(OptimadeDiscoveryError, match="data.type must be 'info'"):
        OptimadeStore(_V1, client=FakeClient(fixed))


def test_entry_info_identity_rejects_a_present_but_wrong_type() -> None:
    fixed = {
        _V1 + "/info": _top_info(["structures"], "1.2.0"),
        _V1 + "/info/structures": _fixture_mutated(type="structures"),
    }
    with pytest.raises(OptimadeDiscoveryError, match="data.type must be 'info'"):
        OptimadeStore(_V1, client=FakeClient(fixed))


def _query_over(
    next_link: str,
    *,
    base: str,
    allow_cross_origin: bool = False,
    extra_pages: dict[str, list[FakeResponse]] | None = None,
    second_next_link: object = None,
) -> tuple[OptimadeStore, FakeClient, list[object]]:
    """Discover a valid ``structures`` endpoint at *base* and page once past *next_link*."""

    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    fixed = {
        base + "/info": _top_info(["structures"], "1.2.0"),
        base + "/info/structures": _fixture_mutated(type="info"),
    }
    pages: dict[str, list[FakeResponse]] = {
        base + "/structures": [
            _page([_structure("a")], next_link=next_link),
            _page([_structure("b")], next_link=second_next_link),
        ]
    }
    if extra_pages is not None:
        pages.update(extra_pages)
    client = FakeClient(fixed, pages)
    store = OptimadeStore(
        base,
        client=client,
        allow_cross_origin_pagination=allow_cross_origin,
    )
    searcher = store.searcher()
    variable = searcher.variable(OptimadeStructure)
    return store, client, [row.structure.id for row in searcher.results(structure=variable)]


def test_continuation_to_a_different_host_still_needs_cross_origin_consent() -> None:
    with pytest.raises(OptimadePaginationError, match="cross-origin"):
        _query_over("https://other.test/v1/structures?page_offset=1", base=_V1)


def test_continuation_downgrade_to_http_on_an_http_base_is_never_upgraded() -> None:
    # An http service continuing to https differs by scheme in the other
    # direction; it is never rewritten and still needs cross-origin consent.
    http_base = "http://plain.example.test/v1"
    with pytest.raises(OptimadePaginationError, match="cross-origin"):
        _query_over("https://plain.example.test/v1/structures?page_offset=1", base=http_base)


def test_continuation_to_a_different_explicit_port_still_needs_cross_origin_consent() -> None:
    with pytest.raises(OptimadePaginationError, match="cross-origin"):
        _query_over(f"http://{_HOST}:8080/v1/structures?page_offset=1", base=_V1)


def test_cross_origin_pagination_still_proceeds_when_explicitly_allowed() -> None:
    other_page = {"https://other.test/v1/structures": [_page([_structure("b")], next_link=None)]}
    store, _client, ids = _query_over(
        "https://other.test/v1/structures?page_offset=1",
        base=_V1,
        allow_cross_origin=True,
        extra_pages=other_page,
    )
    assert ids == ["a", "b"]
    # A genuine cross-origin continuation is not a recorded scheme deviation.
    assert store.deviations == ()


def test_repeated_http_continuation_is_detected_as_a_cycle() -> None:
    # Two pages whose http continuation repeats: the scheme upgrade runs before
    # the cycle check, so the second visit to the same (upgraded) URL is caught
    # as a cycle rather than looping to max_pages.
    with pytest.raises(OptimadePaginationError, match="cycle"):
        _query_over(_HTTP_NEXT, base=_V1, second_next_link=_HTTP_NEXT)


def test_scheme_upgrade_uses_the_base_authority_dropping_an_explicit_port() -> None:
    # An explicit ``:80`` on the http continuation must not survive as a wrong
    # port under https; the base's https authority replaces it.
    _store, client, ids = _query_over(
        f"http://{_HOST}:80/v1/structures?page_offset=1",
        base=_V1,
    )
    assert ids == ["a", "b"]
    assert _HTTPS_NEXT in client.requests
    assert f"http://{_HOST}:80/v1/structures?page_offset=1" not in client.requests
    assert f"{_HOST}:80" not in "".join(client.requests)
