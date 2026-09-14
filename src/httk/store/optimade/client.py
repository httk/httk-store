"""Synchronous, read-only discovery for remote OPTIMADE services.

This module establishes lossless schema snapshots and strict definition-IRI
recognition. Query construction and paginated execution live in
``remote_query`` and are imported lazily by :meth:`OptimadeStore.searcher`.
"""

import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeGuard, cast
from urllib.parse import parse_qsl, quote, urlsplit

from httk.core import load_entry_type_definition
from httk.core.optimade import (
    OptimadeDocument,
    OptimadeResource,
    OptimadeSchemaSnapshot,
    redact_optimade_url,
)
from httk.core.register import (
    OptimadeEntryBinding,
    known_optimade_entry_bindings,
    optimade_entry_binding,
)

if TYPE_CHECKING:
    from httk.store.query.slicer import Slicer

    from .remote_query import RemoteSearcher


class _HTTPClient(Protocol):
    def get(self, url: str) -> object: ...


@dataclass(frozen=True, slots=True)
class _AllAdvertised:
    """Identity sentinel selecting every field advertised by a remote schema."""

    def __repr__(self) -> str:
        return "ALL_ADVERTISED"


ALL_ADVERTISED = _AllAdvertised()

_URL_TOKEN = re.compile(r"(?:https?://|/|\?)[^\s'\"<>]+")
_DIGIT_RUN = re.compile(r"\d+")
_SENSITIVE_QUERY_KEYS = frozenset({"access_token", "api_key", "apikey", "token", "key"})
_VERSION_COMPONENT = r"(?:0|[1-9][0-9]*)"
_EXPLICIT_VERSION = re.compile(rf"^v({_VERSION_COMPONENT})(?:\.({_VERSION_COMPONENT})(?:\.({_VERSION_COMPONENT}))?)?$")
_VERSION_LIKE = re.compile(r"^v[0-9]")
_VERSION_MAJOR = re.compile(r"(?:0|[1-9][0-9]*)$")
_API_VERSION = re.compile(
    rf"^({_VERSION_COMPONENT})\.({_VERSION_COMPONENT})\.({_VERSION_COMPONENT})"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_SUPPORTED_API_MAJORS = frozenset({1})


def _safe_detail(value: str) -> str:
    """Redact complete absolute or slash-relative URL tokens in diagnostics."""

    def redact(match: re.Match[str]) -> str:
        token = match.group()
        if token.startswith("?"):
            keys = {name.casefold() for name, _value in parse_qsl(token[1:], keep_blank_values=True)}
            if keys.isdisjoint(_SENSITIVE_QUERY_KEYS):
                return token
        return redact_optimade_url(token)

    return _URL_TOKEN.sub(redact, value)


class OptimadeClientError(RuntimeError):
    """Base class for safe, client-side OPTIMADE failures."""


class OptimadeTransportError(OptimadeClientError):
    """Report that the HTTP client could not complete a request.

    :param source_url: Redacted URL of the failed request.
    :param detail: Safe transport detail.
    """

    def __init__(self, source_url: str, detail: str) -> None:
        self.source_url = redact_optimade_url(source_url)
        self.detail = _safe_detail(detail)
        super().__init__(f"OPTIMADE transport request failed for {self.source_url}: {self.detail}")


class OptimadeHTTPError(OptimadeClientError):
    """Report a non-success HTTP status from a remote endpoint.

    :param source_url: Redacted URL of the response.
    :param status_code: HTTP status code returned by the service.
    :param detail: Optional safe error detail.
    """

    def __init__(self, source_url: str, status_code: int, detail: str | None = None) -> None:
        self.source_url = redact_optimade_url(source_url)
        self.status_code = status_code
        self.detail = None if detail is None else _safe_detail(detail)
        message = f"OPTIMADE request to {self.source_url} returned HTTP {status_code}"
        if self.detail:
            message += f": {self.detail}"
        super().__init__(message)


class OptimadeErrorDocumentError(OptimadeHTTPError):
    """Report a non-success response with a parseable OPTIMADE error document."""


class OptimadeDiscoveryError(OptimadeClientError):
    """Report a malformed or inconsistent ``/info`` discovery document.

    :param source_url: Redacted URL of the malformed document.
    :param detail: Safe discovery detail.
    """

    def __init__(self, source_url: str, detail: str) -> None:
        self.source_url = redact_optimade_url(source_url)
        self.detail = _safe_detail(detail)
        super().__init__(f"Malformed OPTIMADE discovery response from {self.source_url}: {self.detail}")


class OptimadeVersionNegotiationError(OptimadeClientError):
    """Report failure to negotiate a supported OPTIMADE API version.

    :param source_url: Redacted URL used for negotiation.
    :param detail: Safe negotiation detail.
    """

    def __init__(self, source_url: str, detail: str) -> None:
        self.source_url = redact_optimade_url(source_url)
        self.detail = _safe_detail(detail)
        super().__init__(f"OPTIMADE API version negotiation failed for {self.source_url}: {self.detail}")


def _frozen_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class RemoteEntryType:
    """Describe one immutable remote entry endpoint discovered from ``/info``.

    ``name`` is solely the service's transport endpoint name.  Semantic
    recognition is intentionally represented by ``binding`` and is derived
    exclusively from definition IRIs.

    :param name: Transport endpoint name.
    :param definition_id: Entry-definition IRI, when advertised.
    :param schema: Lossless schema snapshot from discovery.
    :param property_iris: Transport property names keyed by definition IRI.
    :param property_names: Local property names keyed by definition IRI.
    :param property_types: Property kinds keyed by transport name.
    :param advertised_properties: Properties advertised by the service.
    :param default_response_properties: Properties returned by default.
    :param sortable_properties: Properties accepted by remote sorting.
    :param binding: Recognized semantic binding, when available.
    :param backend: Backend class associated with the binding.
    """

    name: str
    definition_id: str | None
    schema: OptimadeSchemaSnapshot
    property_iris: Mapping[str, str]
    property_names: Mapping[str, str]
    property_types: Mapping[str, tuple[str, str | None]]
    advertised_properties: tuple[str, ...]
    default_response_properties: tuple[str, ...]
    sortable_properties: tuple[str, ...]
    binding: OptimadeEntryBinding | None
    backend: type

    def __post_init__(self) -> None:
        object.__setattr__(self, "property_iris", _frozen_mapping(self.property_iris))
        object.__setattr__(self, "property_names", _frozen_mapping(self.property_names))
        object.__setattr__(self, "property_types", MappingProxyType(dict(self.property_types)))
        object.__setattr__(self, "advertised_properties", tuple(self.advertised_properties))
        object.__setattr__(self, "default_response_properties", tuple(self.default_response_properties))
        object.__setattr__(self, "sortable_properties", tuple(self.sortable_properties))


def _parse_json(document: OptimadeDocument, *, label: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(document.text, parse_float=Decimal, parse_int=int)
    except json.JSONDecodeError as exc:
        raise OptimadeDiscoveryError(document.source_url, f"{label} is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise OptimadeDiscoveryError(document.source_url, f"{label} root must be a JSON object")
    return decoded


def _mapping(value: object, *, source_url: str, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise OptimadeDiscoveryError(source_url, f"{label} must be a JSON object")
    return value


def _nonempty_string(value: object, *, source_url: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OptimadeDiscoveryError(source_url, f"{label} must be a nonempty string")
    return value


def _is_definition_iri(value: object) -> TypeGuard[str]:
    """Return whether *value* is a minimally well-formed absolute IRI."""

    if not isinstance(value, str) or not value or value != value.strip():
        return False
    try:
        return bool(urlsplit(value).scheme)
    except ValueError:
        return False


def _error_detail(text: str) -> str | None:
    """Extract one useful but non-secret detail from an OPTIMADE error body."""

    try:
        root = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(root, dict):
        return None
    errors = root.get("errors")
    if not isinstance(errors, list) or not errors:
        return None
    first = errors[0]
    if not isinstance(first, dict):
        return None
    for key in ("detail", "title"):
        value = first.get(key)
        if isinstance(value, str) and value:
            return _safe_detail(value)
    return None


def _reduced_page_limit(detail: str | None, current: int) -> int:
    """Return the next page size to try after a service rejected ``current``.

    OPTIMADE offers no way to discover a service's maximum page size up front,
    so it is inferred from the 403 detail. Implementations state the accepted
    maximum in that message, but the wording is unspecified, so the only
    reliable signal is an integer smaller than what was asked for; the largest
    such integer is chosen, falling back to halving when the message names none.

    :param detail: Safe 403 error detail, when the service supplied one.
    :param current: Page size the service just rejected.
    :return: The next, strictly smaller, page size to attempt.
    """

    candidates = [n for n in (int(m) for m in _DIGIT_RUN.findall(detail or "")) if 1 <= n < current]
    return max(candidates) if candidates else max(1, current // 2)


class OptimadeStore:
    """Connect synchronously to a read-only OPTIMADE service and discover it eagerly.

    Unversioned bases negotiate strictly through the preference-ordered
    ``/versions`` CSV. Query pagination validates complete pages before yielding,
    uses lazy one-root exact-literal requests, and bounds continuation links by
    page count and origin.

    :param base_url: Absolute HTTP(S) service base URL.
    :param client: Optional borrowed synchronous HTTP client.
    :param page_limit: Requested default remote page size; lowered automatically when a service rejects it with HTTP 403.
    :param max_pages: Maximum continuation pages followed by one query.
    :param allow_cross_origin_pagination: Permit continuation links on another origin.
    :param response_fields: Default response-field selection for new searchers.
    :raises OptimadeVersionNegotiationError: If the service cannot select a supported version.
    :raises OptimadeDiscoveryError: If discovery documents are malformed.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client: object | None = None,
        page_limit: int = 50,
        max_pages: int = 10_000,
        allow_cross_origin_pagination: bool = False,
        response_fields: object | None = None,
    ) -> None:
        self._requested_transport_base_url = self._normalise_base_url(base_url)
        self.requested_base_url = redact_optimade_url(self._requested_transport_base_url)
        explicit_major = self._explicit_version_major(self._requested_transport_base_url)
        if explicit_major is not None and explicit_major not in _SUPPORTED_API_MAJORS:
            raise OptimadeVersionNegotiationError(
                self._requested_transport_base_url,
                f"explicit API major version {explicit_major} is unsupported; supported major is 1",
            )
        self._transport_base_url = self._requested_transport_base_url
        self.base_url = redact_optimade_url(self._transport_base_url)
        self.page_limit = self._positive_int(page_limit, "page_limit")
        self.max_pages = self._positive_int(max_pages, "max_pages")
        if not isinstance(allow_cross_origin_pagination, bool):
            raise TypeError("allow_cross_origin_pagination must be a bool")
        self.allow_cross_origin_pagination = allow_cross_origin_pagination
        self.response_fields = response_fields
        self._lock = RLock()
        self._closed = False
        self._owned_client = client is None
        if client is None:
            import httpx

            client = httpx.Client()
        self._client = client
        self._entry_types: tuple[RemoteEntryType, ...] = ()
        self._entry_types_by_name: Mapping[str, RemoteEntryType] = MappingProxyType({})
        self.api_version: str | None = None
        try:
            if explicit_major is None:
                self._transport_base_url = self._negotiate_base_url()
                self.base_url = redact_optimade_url(self._transport_base_url)
            api_version, entry_types, by_name = self._discover()
        except Exception:
            if self._owned_client:
                self._close_owned_client_after_failed_construction()
            raise
        self.api_version = api_version
        self._entry_types = entry_types
        self._entry_types_by_name = by_name

    def __repr__(self) -> str:
        return f"OptimadeStore(base_url={self.base_url!r}, api_version={self.api_version!r})"

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _normalise_base_url(base_url: str) -> str:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a nonempty absolute HTTP(S) URL")
        candidate = base_url.strip()
        try:
            split = urlsplit(candidate)
        except ValueError as exc:
            raise ValueError("base_url must be a valid absolute HTTP(S) URL") from exc
        if split.scheme not in ("http", "https") or not split.netloc or split.query or split.fragment:
            raise ValueError("base_url must be an absolute HTTP(S) URL without query or fragment")
        return candidate.rstrip("/")

    @staticmethod
    def _explicit_version_major(base_url: str) -> int | None:
        """Return the explicitly requested API major, if the final path segment is versioned."""

        final_segment = urlsplit(base_url).path.rsplit("/", 1)[-1]
        match = _EXPLICIT_VERSION.fullmatch(final_segment)
        if match is not None:
            return int(match.group(1))
        if _VERSION_LIKE.match(final_segment):
            raise OptimadeVersionNegotiationError(
                base_url,
                "final path segment is a malformed explicit API version; version suffixes are not supported",
            )
        return None

    def _negotiate_base_url(self) -> str:
        """Select httk's first supported major from an unversioned ``/versions`` response."""

        versions_url = self._requested_transport_base_url + "/versions"
        advertised_majors = self._parse_versions(self._get(versions_url), versions_url)
        for major in advertised_majors:
            if major in _SUPPORTED_API_MAJORS:
                return self._requested_transport_base_url + f"/v{major}"
        raise OptimadeVersionNegotiationError(
            versions_url,
            "server does not advertise a supported API major version (supported major is 1)",
        )

    @staticmethod
    def _parse_versions(text: str, source_url: str) -> tuple[int, ...]:
        """Parse the restricted, preference-ordered CSV specified for ``/versions``."""

        if '"' in text:
            raise OptimadeVersionNegotiationError(source_url, "/versions restricted CSV must not contain quotes")
        if "\r" in text.replace("\r\n", ""):
            raise OptimadeVersionNegotiationError(source_url, "/versions restricted CSV has an invalid line ending")
        normalized = text.replace("\r\n", "\n")
        if normalized.endswith("\n"):
            body = normalized[:-1]
        else:
            body = normalized
        rows = body.split("\n")
        if not rows or rows[0].split(",", 1)[0] != "version":
            raise OptimadeVersionNegotiationError(
                source_url, "/versions restricted CSV header must begin with 'version'"
            )
        if len(rows) == 1:
            raise OptimadeVersionNegotiationError(source_url, "/versions restricted CSV has no advertised versions")

        majors: list[int] = []
        seen: set[int] = set()
        for index, row in enumerate(rows[1:], start=2):
            if not row:
                raise OptimadeVersionNegotiationError(
                    source_url, f"/versions restricted CSV has a blank or malformed row at line {index}"
                )
            version = row.split(",", 1)[0]
            if _VERSION_MAJOR.fullmatch(version) is None:
                raise OptimadeVersionNegotiationError(
                    source_url, f"/versions restricted CSV has an invalid major version at line {index}"
                )
            major = int(version)
            if major in seen:
                raise OptimadeVersionNegotiationError(
                    source_url, f"/versions restricted CSV advertises duplicate major version {major}"
                )
            seen.add(major)
            majors.append(major)
        return tuple(majors)

    def _close_owned_client_after_failed_construction(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Preserve the original discovery exception. A best-effort
                # cleanup failure must not replace the useful schema error.
                self._closed = True
                return
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise OptimadeClientError("OPTIMADE store is closed")

    def _get(self, url: str) -> str:
        self._require_open()
        try:
            response = cast(_HTTPClient, self._client).get(url)
        except Exception as exc:
            raise OptimadeTransportError(url, str(exc)) from exc
        status_code = getattr(response, "status_code", None)
        text = getattr(response, "text", None)
        if not isinstance(status_code, int) or not isinstance(text, str):
            raise OptimadeTransportError(url, "HTTP client returned an invalid response object")
        if not 200 <= status_code < 300:
            detail = _error_detail(text)
            if detail is not None:
                raise OptimadeErrorDocumentError(url, status_code, detail)
            raise OptimadeHTTPError(url, status_code)
        return text

    def _get_page(self, build_url: Callable[[int], str], page_limit: int) -> tuple[str, str]:
        """Fetch a self-constructed first page, learning any page-size cap from a 403.

        OPTIMADE provides no way to discover a service's maximum page size up
        front, so a service that caps below the requested size is discovered
        only by its HTTP 403 rejection. On such a rejection this retries at a
        smaller size (see :func:`_reduced_page_limit`) and, once a reduction has
        succeeded, lowers :attr:`page_limit` so later queries skip the failed
        roundtrip. Any non-403 failure, or a 403 already at page size 1,
        propagates and leaves :attr:`page_limit` untouched.

        :param build_url: Build the first-page request URL for a given page size.
        :param page_limit: Page size to request first.
        :return: The successful response text and the URL that produced it.
        :raises OptimadeHTTPError: If every attempt down to page size 1 fails.
        """

        attempt = page_limit
        reduced = False
        while True:
            url = build_url(attempt)
            try:
                text = self._get(url)
            except OptimadeHTTPError as exc:
                if exc.status_code != 403 or attempt <= 1:
                    raise
                previous = attempt
                attempt = _reduced_page_limit(exc.detail, attempt)
                reduced = True
                logging.getLogger(__name__).warning(
                    "OPTIMADE service rejected page_limit=%d; retrying at page_limit=%d",
                    previous,
                    attempt,
                    extra={"context": "optimade"},
                )
                continue
            if reduced:
                self.page_limit = min(self.page_limit, attempt)
            return text, url

    @property
    def entry_types(self) -> tuple[RemoteEntryType, ...]:
        """Discovered entry endpoints in the service-advertised order."""

        return self._entry_types

    @property
    def entry_types_by_name(self) -> Mapping[str, RemoteEntryType]:
        """An immutable transport-name lookup for :attr:`entry_types`."""

        return self._entry_types_by_name

    def entry_type(self, name: str) -> RemoteEntryType:
        """Return one discovered endpoint by transport name.

        :param name: Service-advertised endpoint name.
        :return: Discovered endpoint descriptor.
        :raises KeyError: If no endpoint has that name.
        """

        try:
            return self._entry_types_by_name[name]
        except KeyError as exc:
            raise KeyError(f"No discovered OPTIMADE entry endpoint named {name!r}") from exc

    def _discover(self) -> tuple[str | None, tuple[RemoteEntryType, ...], Mapping[str, RemoteEntryType]]:
        info_url = self._transport_base_url + "/info"
        info_document = OptimadeDocument.from_response(self._get(info_url), info_url)
        info_root = _parse_json(info_document, label="/info response")
        data = _mapping(info_root.get("data"), source_url=info_document.source_url, label="/info data")
        if data.get("type") != "info":
            raise OptimadeDiscoveryError(info_document.source_url, "/info data.type must be 'info'")
        attributes = _mapping(
            data.get("attributes"), source_url=info_document.source_url, label="/info data.attributes"
        )
        api_version, entry_info_resource_identity = self._entry_info_format(
            attributes.get("api_version"), info_document.source_url
        )
        advertised = attributes.get("available_endpoints")
        if not isinstance(advertised, list):
            raise OptimadeDiscoveryError(info_document.source_url, "/info available_endpoints must be a JSON array")
        endpoint_names: list[str] = []
        seen_names: set[str] = set()
        for endpoint in advertised:
            if not isinstance(endpoint, str):
                raise OptimadeDiscoveryError(info_document.source_url, "/info available_endpoints must contain strings")
            if endpoint in ("", "/", "info", "links") or endpoint.startswith("info/"):
                continue
            if endpoint not in seen_names:
                seen_names.add(endpoint)
                endpoint_names.append(endpoint)

        descriptors: list[RemoteEntryType] = []
        by_name: dict[str, RemoteEntryType] = {}
        for name in endpoint_names:
            endpoint_url = self._transport_base_url + "/info/" + quote(name, safe="")
            document = OptimadeDocument.from_response(self._get(endpoint_url), endpoint_url)
            descriptor = self._entry_descriptor(
                name,
                document,
                require_resource_identity=entry_info_resource_identity,
            )
            descriptors.append(descriptor)
            by_name[name] = descriptor
        return api_version, tuple(descriptors), MappingProxyType(by_name)

    @staticmethod
    def _entry_info_format(api_version: object, source_url: str) -> tuple[str | None, bool]:
        """Select the entry-info grammar from the version declared by ``/info``.

        OPTIMADE 1.0 and 1.1 describe ``data`` as the entry-info object itself.
        Version 1.2 changed it into a resource object requiring ``type`` and
        ``id``. Missing version metadata retains the pre-existing strict
        resource-identity check rather than guessing an older grammar.
        """

        if api_version is None:
            return None, True
        if not isinstance(api_version, str):
            raise OptimadeDiscoveryError(source_url, "/info api_version must be a semantic-version string")
        match = _API_VERSION.fullmatch(api_version)
        if match is None:
            raise OptimadeDiscoveryError(source_url, "/info api_version must be a semantic-version string")
        major = int(match.group(1))
        minor = int(match.group(2))
        if major not in _SUPPORTED_API_MAJORS:
            raise OptimadeDiscoveryError(
                source_url,
                f"/info declares unsupported API major version {major}; supported major is 1",
            )
        return api_version, minor >= 2

    def _entry_descriptor(
        self,
        name: str,
        document: OptimadeDocument,
        *,
        require_resource_identity: bool,
    ) -> RemoteEntryType:
        root = _parse_json(document, label=f"/info/{name} response")
        data = _mapping(root.get("data"), source_url=document.source_url, label=f"/info/{name} data")
        if require_resource_identity and data.get("type") != "info":
            raise OptimadeDiscoveryError(document.source_url, f"/info/{name} data.type must be 'info'")
        properties = _mapping(
            data.get("properties"), source_url=document.source_url, label=f"/info/{name} data.properties"
        )
        property_iris: dict[str, str] = {}
        property_names: dict[str, str] = {}
        property_types: dict[str, tuple[str, str | None]] = {}
        default_response: list[str] = []
        sortable: list[str] = []
        for property_name, definition in properties.items():
            if not isinstance(property_name, str) or not property_name:
                raise OptimadeDiscoveryError(document.source_url, "property names must be nonempty strings")
            definition_mapping = _mapping(
                definition, source_url=document.source_url, label=f"property definition {property_name!r}"
            )
            definition_id = definition_mapping.get("$id")
            if _is_definition_iri(definition_id):
                previous = property_names.get(definition_id)
                if previous is not None:
                    raise OptimadeDiscoveryError(
                        document.source_url,
                        f"properties {previous!r} and {property_name!r} advertise the same definition IRI {definition_id!r}",
                    )
                property_iris[property_name] = definition_id
                property_names[definition_id] = property_name
            optimade_type = definition_mapping.get("x-optimade-type")
            if not isinstance(optimade_type, str):
                legacy_type = definition_mapping.get("type")
                optimade_type = legacy_type if isinstance(legacy_type, str) else "unknown"
            items = definition_mapping.get("items")
            item_type = (
                cast(str, items.get("x-optimade-type"))
                if isinstance(items, Mapping) and isinstance(items.get("x-optimade-type"), str)
                else None
            )
            property_types[property_name] = (optimade_type, item_type)
            implementation = definition_mapping.get("x-optimade-implementation")
            if implementation is not None:
                implementation_mapping = _mapping(
                    implementation,
                    source_url=document.source_url,
                    label=f"property {property_name!r} x-optimade-implementation",
                )
                if "sortable" in implementation_mapping:
                    sortable_value = implementation_mapping["sortable"]
                    if not isinstance(sortable_value, bool):
                        raise OptimadeDiscoveryError(
                            document.source_url, f"property {property_name!r} sortable metadata must be a bool"
                        )
                    if sortable_value:
                        sortable.append(property_name)
                if "response-default" in implementation_mapping:
                    default_value = implementation_mapping["response-default"]
                    if not isinstance(default_value, bool):
                        raise OptimadeDiscoveryError(
                            document.source_url,
                            f"property {property_name!r} response-default metadata must be a bool",
                        )
                    if default_value:
                        default_response.append(property_name)

        describedby: str | None = None
        links = root.get("links")
        if links is not None:
            links_mapping = _mapping(links, source_url=document.source_url, label=f"/info/{name} links")
            if "describedby" in links_mapping:
                describedby = _nonempty_string(
                    links_mapping["describedby"],
                    source_url=document.source_url,
                    label=f"/info/{name} links.describedby",
                )
        binding = self._recognise_binding(describedby, frozenset(property_names))
        try:
            backend = OptimadeResource if binding is None else binding.resolve_backend()
        except Exception as exc:
            binding_id = describedby if describedby is not None else binding.definition_id if binding else None
            raise OptimadeDiscoveryError(
                document.source_url, f"could not resolve OPTIMADE binding backend for {binding_id!r}: {exc}"
            ) from exc
        return RemoteEntryType(
            name=name,
            definition_id=describedby,
            schema=OptimadeSchemaSnapshot(name, document),
            property_iris=property_iris,
            property_names=property_names,
            property_types=property_types,
            advertised_properties=tuple(properties),
            default_response_properties=tuple(default_response),
            sortable_properties=tuple(sortable),
            binding=binding,
            backend=backend,
        )

    @staticmethod
    def _recognise_binding(
        describedby: str | None, remote_property_iris: frozenset[str]
    ) -> OptimadeEntryBinding | None:
        if describedby is not None:
            return optimade_entry_binding(describedby)

        binding_ids = known_optimade_entry_bindings()
        if not binding_ids:
            return None
        bindings: dict[str, OptimadeEntryBinding] = {}
        property_owners: dict[str, set[str]] = {}
        for definition_id in binding_ids:
            binding = optimade_entry_binding(definition_id)
            if binding is None:
                raise OptimadeDiscoveryError(
                    "(local registry)", f"binding {definition_id!r} disappeared during discovery"
                )
            try:
                definition = load_entry_type_definition(definition_id)
            except Exception as exc:
                raise OptimadeDiscoveryError(
                    "(local registry)", f"could not load entry-type definition for binding {definition_id!r}: {exc}"
                ) from exc
            bindings[definition_id] = binding
            for property_definition in definition.properties.values():
                property_owners.setdefault(property_definition.definition_id, set()).add(definition_id)

        candidates = set(binding_ids)
        universal = set(binding_ids)
        for property_iri in remote_property_iris:
            owners = property_owners.get(property_iri, set())
            # A locally unknown extension IRI carries no evidence about the
            # entry type. Known non-universal IRIs do: mutually exclusive
            # evidence still empties the candidate set and stays generic.
            if owners and owners != universal:
                candidates.intersection_update(owners)
        if len(candidates) == 1:
            return bindings[candidates.pop()]
        return None

    def refresh(self) -> None:
        """Refresh discovery state after a fully successful rediscovery.

        :raises OptimadeClientError: If the store is closed or discovery fails.
        """

        with self._lock:
            self._require_open()
            api_version, entry_types, by_name = self._discover()
            self.api_version = api_version
            self._entry_types = entry_types
            self._entry_types_by_name = by_name

    def close(self) -> None:
        """Close an internally owned HTTP client; borrowed clients stay open."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owned_client:
                close = getattr(self._client, "close", None)
                if callable(close):
                    close()

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def searcher(self, *, response_fields: object = ..., as_of: object = None) -> "RemoteSearcher":
        """Create one synchronous, read-only remote search plan.

        Passing ``response_fields`` overrides the store-level selection. An
        omitted value inherits it, while explicit ``None`` requests the
        service default.

        :param response_fields: Per-search field selection override.
        :param as_of: Historic cutoff; unsupported because remote snapshot negotiation is unavailable.
        :return: New remote search plan.
        :raises OptimadeClientError: If the store is closed.
        :raises ValueError: If a historic cutoff is requested.
        """

        if as_of is not None:
            raise ValueError("OptimadeStore cannot honor as_of; remote historic snapshot negotiation is unsupported")

        from .remote_query import RemoteSearcher

        selected = self.response_fields if response_fields is ... else response_fields
        return RemoteSearcher(self, response_fields=selected)

    def slicer(self, target: "RemoteEntryType | str") -> "Slicer":
        """A pandas-style ``[]`` indexing view over one discovered entry endpoint.

        ``target`` is a discovered :class:`RemoteEntryType`, or its transport
        endpoint name resolved the same way as :meth:`entry_type`. Each
        terminal indexing operation runs its own fresh search against a
        searcher created with this store's default ``response_fields``
        policy. No sorting is offered here -- use :meth:`searcher` directly
        for a sorted or relationship query.

        :param target: A discovered endpoint descriptor, or its endpoint name.
        :return: A slicer over the endpoint's records.
        :raises KeyError: If ``target`` is a name with no discovered endpoint.
        """
        descriptor = self.entry_type(target) if isinstance(target, str) else target
        return self.searcher().slicer(descriptor)
