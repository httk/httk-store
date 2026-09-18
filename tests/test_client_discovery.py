"""No-network tests for synchronous OPTIMADE client discovery."""

import json
from dataclasses import dataclass

import httpx2
import pytest
from httk.core.optimade import OptimadeResource

from httk.store.optimade import (
    ALL_ADVERTISED,
    OptimadeDiscoveryError,
    OptimadeErrorDocumentError,
    OptimadeStore,
    OptimadeTransportError,
    OptimadeVersionNegotiationError,
)

FILES = "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/files"
REFERENCES = "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/references"
STRUCTURES = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
FILE_URL = "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/url"
REFERENCE_ADDRESS = "https://schemas.optimade.org/defs/v1.2/properties/optimade/references/address"
STRUCTURE_ELEMENTS = "https://schemas.optimade.org/defs/v1.2/properties/optimade/structures/elements"
CORE_ID = "https://schemas.optimade.org/defs/v1.2/properties/core/id"
VENDOR_EXTENSION = "https://example.test/defs/properties/_vendor_extension"


@dataclass
class FakeResponse:
    status_code: int
    text: str


class FakeClient:
    def __init__(self, responses: dict[str, list[FakeResponse]]) -> None:
        self.responses = {url: list(items) for url, items in responses.items()}
        self.requests: list[str] = []
        self.closed = False

    def get(self, url: str) -> FakeResponse:
        self.requests.append(url)
        try:
            return self.responses[url].pop(0)
        except (KeyError, IndexError) as exc:
            raise AssertionError(f"unexpected external request: {url}") from exc

    def close(self) -> None:
        self.closed = True


def response(value: object, status_code: int = 200) -> FakeResponse:
    return FakeResponse(status_code, json.dumps(value))


def info(endpoints: list[object], *, api_version: object | None = None) -> dict[str, object]:
    attributes: dict[str, object] = {"available_endpoints": endpoints}
    if api_version is not None:
        attributes["api_version"] = api_version
    return {"data": {"type": "info", "attributes": attributes}}


def entry(properties: dict[str, object], describedby: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {"data": {"type": "info", "properties": properties}}
    if describedby is not None:
        value["links"] = {"describedby": describedby}
    return value


def property_definition(definition_id: object | None, *, sortable: bool = False) -> dict[str, object]:
    value: dict[str, object] = {}
    if definition_id is not None:
        value["$id"] = definition_id
    if sortable:
        value["x-optimade-implementation"] = {"sortable": True}
    return value


def make_client(entries: dict[str, dict[str, object]], *, base_url: str = "https://example.test/v1") -> FakeClient:
    responses = {base_url + "/info": [response(info(["info", "links", *entries]))]}
    responses.update({base_url + "/info/" + name: [response(value)] for name, value in entries.items()})
    return FakeClient(responses)


def test_unversioned_base_negotiates_first_supported_major_in_server_order() -> None:
    requested = "https://example.test/db"
    effective = requested + "/v1"
    client = FakeClient(
        {
            requested + "/versions": [FakeResponse(200, "version,comment\r\n2,preferred\r\n1,supported\r\n")],
            effective + "/info": [response(info(["files"]))],
            effective + "/info/files": [response(entry({"url": property_definition(FILE_URL)}))],
        }
    )

    store = OptimadeStore(requested + "/", client=client)

    assert store.requested_base_url == requested
    assert store.base_url == effective
    assert client.requests == [
        requested + "/versions",
        effective + "/info",
        effective + "/info/files",
    ]


def test_single_column_crlf_versions_csv_is_accepted() -> None:
    requested = "https://example.test/db"
    effective = requested + "/v1"
    client = FakeClient(
        {
            requested + "/versions": [FakeResponse(200, "version\r\n1\r\n")],
            effective + "/info": [response(info([]))],
        }
    )

    store = OptimadeStore(requested, client=client)

    assert store.base_url == effective
    assert client.requests == [requested + "/versions", effective + "/info"]


@pytest.mark.parametrize("suffix", ["v1", "v1.2", "v1.2.3"])
def test_explicit_supported_versions_skip_negotiation(suffix: str) -> None:
    base_url = "https://example.test/" + suffix
    client = make_client({}, base_url=base_url)

    store = OptimadeStore(base_url + "/", client=client)

    assert store.requested_base_url == base_url
    assert store.base_url == base_url
    assert client.requests == [base_url + "/info"]


@pytest.mark.parametrize("suffix", ["v0", "v2", "v2.1", "v3.0.0"])
def test_explicit_unsupported_versions_fail_before_discovery(suffix: str) -> None:
    client = FakeClient({})

    with pytest.raises(OptimadeVersionNegotiationError, match="unsupported"):
        OptimadeStore("https://example.test/" + suffix, client=client)

    assert client.requests == []
    assert not client.closed


@pytest.mark.parametrize(
    "suffix",
    ["v2beta", "v2.0.0.0", "v01", "v01.2", "v1.02", "v1.2.03", "v1-rc1"],
)
def test_malformed_version_like_paths_fail_before_discovery(suffix: str) -> None:
    client = FakeClient({})

    with pytest.raises(OptimadeVersionNegotiationError, match="malformed"):
        OptimadeStore("https://example.test/" + suffix, client=client)

    assert client.requests == []
    assert not client.closed


@pytest.mark.parametrize(
    "versions",
    [
        "",
        "versions\n1",
        "version",
        "version\n1\n\n2",
        "version\none",
        "version\n1\n1",
        'version\n"1"',
    ],
)
def test_malformed_versions_csv_fails_without_discovery(versions: str) -> None:
    requested = "https://example.test/db"
    client = FakeClient({requested + "/versions": [FakeResponse(200, versions)]})

    with pytest.raises(OptimadeVersionNegotiationError):
        OptimadeStore(requested, client=client)

    assert client.requests == [requested + "/versions"]
    assert not client.closed


def test_versions_rejects_a_standalone_leading_zero_major() -> None:
    requested = "https://example.test/db"
    client = FakeClient({requested + "/versions": [FakeResponse(200, "version\n01\n")]})

    with pytest.raises(OptimadeVersionNegotiationError, match="invalid major"):
        OptimadeStore(requested, client=client)

    assert client.requests == [requested + "/versions"]


def test_no_supported_advertised_major_fails_after_versions() -> None:
    requested = "https://example.test/db"
    client = FakeClient({requested + "/versions": [FakeResponse(200, "version\n2\n3\n")]})

    with pytest.raises(OptimadeVersionNegotiationError, match="does not advertise"):
        OptimadeStore(requested, client=client)

    assert client.requests == [requested + "/versions"]


def test_owned_client_is_closed_when_versions_negotiation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = "https://example.test/db"
    fake = FakeClient({requested + "/versions": [FakeResponse(200, "version\n2\n")]})
    monkeypatch.setattr(httpx2, "Client", lambda **_kwargs: fake)

    with pytest.raises(OptimadeVersionNegotiationError):
        OptimadeStore(requested)

    assert fake.closed


def test_credentials_stay_private_while_negotiated_transport_keeps_them() -> None:
    requested_transport = "https://user:password@example.test/db"
    effective_transport = requested_transport + "/v1"
    client = FakeClient(
        {
            requested_transport + "/versions": [FakeResponse(200, "version\n1\n")],
            effective_transport + "/info": [response(info([]))],
        }
    )

    store = OptimadeStore(requested_transport, client=client)

    assert client.requests == [requested_transport + "/versions", effective_transport + "/info"]
    assert "password" not in store.requested_base_url
    assert "password" not in store.base_url
    with pytest.raises(OptimadeTransportError) as excinfo:
        store._get(effective_transport + "/private")
    assert "password" not in str(excinfo.value)


def test_refresh_reuses_negotiated_base_without_requesting_versions_again() -> None:
    requested = "https://example.test/db"
    effective = requested + "/v1"
    client = FakeClient(
        {
            requested + "/versions": [FakeResponse(200, "version\n1\n")],
            effective + "/info": [response(info(["files"])), response(info(["files"]))],
            effective + "/info/files": [
                response(entry({"url": property_definition(FILE_URL)})),
                response(entry({"url": property_definition(FILE_URL)})),
            ],
        }
    )
    store = OptimadeStore(requested, client=client)

    store.refresh()

    assert client.requests == [
        requested + "/versions",
        effective + "/info",
        effective + "/info/files",
        effective + "/info",
        effective + "/info/files",
    ]


def test_v1_1_entry_info_uses_the_legacy_data_object_grammar() -> None:
    base_url = "https://example.test/v1"
    client = FakeClient(
        {
            base_url + "/info": [response(info(["structures"], api_version="1.1.0"))],
            base_url + "/info/structures": [
                response(
                    {
                        "data": {
                            "description": "legacy OPTIMADE 1.1 entry info",
                            "properties": {
                                "elements": {"description": "elements", "type": "list"},
                                "nelements": {"description": "number of elements", "type": "integer"},
                            },
                            "formats": ["json"],
                            "output_fields_by_format": {"json": ["elements", "nelements"]},
                        }
                    }
                )
            ],
        }
    )

    store = OptimadeStore(base_url, client=client)

    assert store.api_version == "1.1.0"
    assert store.entry_type("structures").advertised_properties == ("elements", "nelements")
    assert store.entry_type("structures").property_types == {
        "elements": ("list", None),
        "nelements": ("integer", None),
    }


def test_v1_2_entry_info_still_requires_the_info_resource_type() -> None:
    base_url = "https://example.test/v1"
    client = FakeClient(
        {
            base_url + "/info": [response(info(["structures"], api_version="1.2.0"))],
            base_url + "/info/structures": [
                response(
                    {
                        "data": {
                            "description": "malformed OPTIMADE 1.2 entry info",
                            "properties": {},
                            "formats": ["json"],
                            "output_fields_by_format": {"json": []},
                        }
                    }
                )
            ],
        }
    )

    with pytest.raises(OptimadeDiscoveryError, match=r"data\.type must be 'info'"):
        OptimadeStore(base_url, client=client)


@pytest.mark.parametrize("api_version", [1, "1.1", "v1.1.0", "01.1.0"])
def test_invalid_declared_api_version_is_rejected(api_version: object) -> None:
    base_url = "https://example.test/v1"
    client = FakeClient({base_url + "/info": [response(info([], api_version=api_version))]})

    with pytest.raises(OptimadeDiscoveryError, match="semantic-version string"):
        OptimadeStore(base_url, client=client)


def test_eager_discovery_preserves_order_and_direct_exact_binding() -> None:
    client = make_client(
        {
            "renamed-files": entry({"different_url": property_definition(FILE_URL, sortable=True)}, describedby=FILES),
            "references": entry({"id": property_definition(CORE_ID)}, describedby=REFERENCES),
        }
    )

    store = OptimadeStore("https://example.test/v1/", client=client)

    assert client.requests == [
        "https://example.test/v1/info",
        "https://example.test/v1/info/renamed-files",
        "https://example.test/v1/info/references",
    ]
    assert [item.name for item in store.entry_types] == ["renamed-files", "references"]
    remote = store.entry_type("renamed-files")
    assert remote.definition_id == FILES
    assert remote.binding is not None and remote.binding.definition_id == FILES
    assert remote.property_iris == {"different_url": FILE_URL}
    assert remote.property_names == {FILE_URL: "different_url"}
    assert remote.sortable_properties == ("different_url",)
    with pytest.raises(TypeError):
        remote.property_iris["nope"] = "nope"  # type: ignore[index]


def test_property_iri_inference_ignores_transport_and_property_names() -> None:
    client = make_client({"wholly-renamed": entry({"not-url": property_definition(FILE_URL)})})

    remote = OptimadeStore("https://example.test/v1", client=client).entry_type("wholly-renamed")

    assert remote.binding is not None and remote.binding.definition_id == FILES


def test_unknown_extensions_do_not_veto_unique_known_property_evidence() -> None:
    client = make_client(
        {
            "extended-files": entry(
                {
                    "fileish": property_definition(FILE_URL),
                    "_vendor_extension": property_definition(VENDOR_EXTENSION),
                }
            ),
            "extended-structures": entry(
                {
                    "structureish": property_definition(STRUCTURE_ELEMENTS),
                    "_vendor_extension": property_definition(VENDOR_EXTENSION),
                }
            ),
            "unknown-only": entry({"_vendor_extension": property_definition(VENDOR_EXTENSION)}),
        }
    )

    store = OptimadeStore("https://example.test/v1", client=client)

    assert store.entry_type("extended-files").binding is not None
    assert store.entry_type("extended-files").binding.definition_id == FILES
    assert store.entry_type("extended-structures").binding is not None
    assert store.entry_type("extended-structures").binding.definition_id == STRUCTURES
    assert store.entry_type("unknown-only").binding is None
    assert store.entry_type("unknown-only").backend is OptimadeResource


def test_name_only_invalid_ids_ambiguous_contradictory_and_unknown_describedby_stay_generic() -> None:
    client = make_client(
        {
            # These documents declare no api_version anywhere, so even the
            # standard endpoint name ``files`` stays generic: no declared
            # version means no standard-name binding (the namespace rule is
            # covered in test_optimade_standard_binding).
            "files": entry({"url": property_definition(None)}),
            "empty-id": entry({"url": property_definition("")}),
            "non-string-id": entry({"url": property_definition(42)}),
            "relative-id": entry({"url": property_definition("properties/url")}),
            "spaced-id": entry({"url": property_definition(" " + FILE_URL + " ")}),
            "old": entry({"id": property_definition(CORE_ID)}),
            "declared-unknown": entry({"url": property_definition(FILE_URL)}, describedby="https://unknown.test/files"),
            "contradictory": entry(
                {
                    "fileish": property_definition(FILE_URL),
                    "referenceish": property_definition(REFERENCE_ADDRESS),
                }
            ),
            "unknown-property": entry({"what": property_definition("https://unknown.test/property")}),
        }
    )
    store = OptimadeStore("https://example.test/v1", client=client)

    assert all(item.binding is None and item.backend is OptimadeResource for item in store.entry_types)
    assert store.entry_type("empty-id").property_iris == {}
    assert store.entry_type("non-string-id").property_iris == {}
    assert store.entry_type("relative-id").property_iris == {}
    assert store.entry_type("spaced-id").property_names == {}
    # ``id`` is universal across the three known built-in definitions, so it
    # establishes an explicitly ambiguous candidate set rather than a binding.
    assert store.entry_type("old").property_iris == {"id": CORE_ID}
    assert store.entry_type("old").binding is None
    # These two definition IRIs have mutually exclusive known owners.
    assert store.entry_type("contradictory").property_iris == {
        "fileish": FILE_URL,
        "referenceish": REFERENCE_ADDRESS,
    }
    assert store.entry_type("contradictory").backend is OptimadeResource
    # An unknown direct entry definition IRI must not fall back to property inference.
    assert store.entry_type("declared-unknown").property_iris == {"url": FILE_URL}
    assert store.entry_type("declared-unknown").binding is None


def test_duplicate_property_iri_is_malformed_and_owned_client_is_closed() -> None:
    base_url = "https://example.test/v1"
    fake = FakeClient(
        {
            base_url + "/info": [response(info(["things"]))],
            base_url + "/info/things": [
                response(entry({"first": property_definition(FILE_URL), "second": property_definition(FILE_URL)}))
            ],
        }
    )
    original = httpx2.Client
    httpx2.Client = lambda **_kwargs: fake  # type: ignore[assignment]
    try:
        with pytest.raises(OptimadeDiscoveryError, match="same definition IRI"):
            OptimadeStore(base_url)
    finally:
        httpx2.Client = original  # type: ignore[assignment]
    assert fake.closed


def test_raw_snapshot_redacts_credentials_without_reserializing_decimal_spelling() -> None:
    base_url = "https://user:password@example.test/v1?token=nope"
    # A query is deliberately rejected for a base URL, so use credential-only
    # transport URL and put a decimal in otherwise irrelevant metadata.
    base_url = "https://user:password@example.test/v1"
    text = '{"data":{"type":"info","properties":{"url":{"$id":"' + FILE_URL + '"}}},"meta":{"x":1.2300}}'
    client = FakeClient(
        {
            base_url + "/info": [response(info(["things"]))],
            base_url + "/info/things": [FakeResponse(200, text)],
        }
    )

    snapshot = OptimadeStore(base_url, client=client).entry_type("things").schema.info_document

    assert "password" not in snapshot.source_url
    assert "1.2300" in snapshot.text


def test_refresh_is_atomic_and_borrowed_client_is_never_closed() -> None:
    base_url = "https://example.test/v1"
    client = FakeClient(
        {
            base_url + "/info": [response(info(["files"])), response(info(["files"]))],
            base_url + "/info/files": [
                response(entry({"url": property_definition(FILE_URL)})),
                FakeResponse(500, '{"errors":[{"detail":"temporary failure at /next?token=secret"}]}'),
            ],
        }
    )
    store = OptimadeStore(base_url, client=client)
    old = store.entry_type("files")

    with pytest.raises(OptimadeErrorDocumentError, match="temporary failure") as excinfo:
        store.refresh()

    assert "secret" not in str(excinfo.value)
    assert store.entry_type("files") is old
    store.close()
    store.close()
    assert not client.closed
    with pytest.raises(Exception, match="closed"):
        store.refresh()


def test_transport_diagnostics_redact_source_and_sentinel_is_identity_safe() -> None:
    class BrokenClient:
        def get(self, url: str) -> FakeResponse:
            raise RuntimeError("offline at /next?token=secret and https://user:secret@other.test/path?token=nope")

    with pytest.raises(OptimadeTransportError) as excinfo:
        OptimadeStore("https://user:secret@example.test/v1", client=BrokenClient())

    assert "secret" not in str(excinfo.value)
    assert repr(ALL_ADVERTISED) == "ALL_ADVERTISED"
    imported_again = __import__("httk.store.optimade", fromlist=["ALL_ADVERTISED"]).ALL_ADVERTISED
    assert ALL_ADVERTISED is imported_again


def test_owned_client_uses_the_store_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The store's own HTTP client gets a 120 s default, or the configured timeout."""
    base = "https://example.test/v1"
    entries = {"files": entry({"url": property_definition(FILE_URL)})}
    created: list[object] = []

    def fake_client_factory(*, timeout: object = "unset") -> FakeClient:
        created.append(timeout)
        return make_client(entries)

    monkeypatch.setattr(httpx2, "Client", fake_client_factory)
    OptimadeStore(base).close()
    OptimadeStore(base, timeout=300).close()
    OptimadeStore(base, timeout=None).close()
    assert created == [120.0, 300, None]

    borrowed = make_client(entries)
    store = OptimadeStore(base, client=borrowed, timeout=7)
    assert store._client is borrowed and store.timeout == 7
    store.close()
    assert created == [120.0, 300, None]

    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="timeout"):
            OptimadeStore(base, timeout=bad)
