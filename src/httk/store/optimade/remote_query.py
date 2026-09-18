"""Neutral synchronous query/result protocols over a remote OPTIMADE service.

Attribute access on a bound query variable (``variable.some_field``) resolves
against the endpoint's declared field map, including OPTIMADE provider-prefixed
properties such as ``variable._anyterial_max_spin_splitting`` -- on a generic,
unregistered entry type the local field names are the wire names verbatim, and
provider prefixes are the norm there.  A double-leading-underscore name (a
dunder, e.g. ``__deepcopy__``) is always rejected with a bare
``AttributeError`` before the field map is even consulted, so interpreter and
library introspection stay cheap and can never collide with a field name.  A
single-leading-underscore name that is *not* a declared field also raises a
bare ``AttributeError`` rather than the descriptive
:class:`~httk.store.UnsupportedQueryError` used for other unknown names: this
keeps probes shaped like a provider field but not one, such as IPython's
``_ipython_canary_method_should_not_exist_``, indistinguishable from "no such
attribute" instead of surfacing as a backend failure.
"""

import dataclasses
import datetime
import functools
import json
import logging
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote, urlencode, urljoin, urlsplit, urlunsplit

from httk.core import load_entry_type_definition
from httk.core.optimade import (
    OptimadeDocument,
    OptimadeResource,
    optimade_document_root,
    redact_optimade_url,
)

from httk.store.query.portable import (
    PortableQueryCapabilities,
    portable_query_capabilities,
    portable_query_fields,
)
from httk.store.query.protocols import CountUnavailableError as NeutralCountUnavailableError
from httk.store.query.protocols import (
    MultipleResultsError,
    NoResultError,
    ResultRow,
    SearchResult,
    UnsupportedQueryError,
)

from .client import ALL_ADVERTISED, OptimadeClientError, OptimadeStore, RemoteEntryType

if TYPE_CHECKING:
    from httk.store.query.slicer import Slicer

__all__ = [
    "CountUnavailableError",
    "OptimadePaginationError",
    "OptimadeResponseError",
    "RemoteResultColumn",
    "RemoteResultSet",
    "RemoteSearcher",
]

_CORE_ID = "https://schemas.optimade.org/defs/v1.2/properties/core/id"
_CORE_TYPE = "https://schemas.optimade.org/defs/v1.2/properties/core/type"


class OptimadeResponseError(OptimadeClientError):
    """A successful HTTP response was not a usable OPTIMADE entry document."""


class OptimadePaginationError(OptimadeResponseError):
    """A remote continuation was unsafe, malformed, or non-terminating."""


class CountUnavailableError(OptimadeResponseError, NeutralCountUnavailableError):
    """The client could not obtain an exact filtered result count."""


@dataclass(slots=True)
class _CountCache:
    value: int | None = None


def _unsupported(detail: str) -> UnsupportedQueryError:
    return UnsupportedQueryError(f"remote OPTIMADE query does not support {detail}")


def _scalar_value(value: object, generic: bool, field: "_RemoteField") -> object:
    """Extract one scalar output value from one materialized query result.

    A bound (non-generic) backend materializes a typed object with real
    attributes matching *field*'s local name, so a plain ``getattr`` is
    exact and unchanged here. A schema-unknown advertised name (a provider
    extension) has no decoded typed attribute, but its bound row exposes it raw
    by exact wire name (see :func:`_bound_class`), so the same ``getattr`` still
    resolves it.

    A generic (unregistered) backend has no typed object at all: *value* is
    the raw :class:`~httk.core.optimade.OptimadeResource` itself, a
    source-exact lazy ``Mapping`` whose only real attributes are ``id``,
    ``type``, ``document``, ``data_index``, ``schema``, and ``unwrap()``.
    ``id``/``type`` are protocol envelope members and are read straight off
    the resource; every other field is read from ``unwrap()["attributes"]``
    by the field's *wire* name (``field._remote_name``) rather than its
    local name -- for a generic descriptor the two happen to be identical
    today, but the wire name is what the response envelope actually keys on,
    so a future bound-name divergence cannot silently misroute this lookup.

    An attribute absent from ``attributes`` yields ``None``, never a
    ``KeyError``: an OPTIMADE server may legitimately omit a property whose
    value is unknown (httk-serve itself does this), so "absent" and
    "present with an explicit null" are deliberately not distinguished here.

    :param value: Materialized query result object.
    :param generic: Whether the owning descriptor's backend is ``OptimadeResource``.
    :param field: Field being projected.
    :return: Extracted scalar value, or ``None`` for an absent generic attribute.
    """

    if not generic:
        return getattr(value, field._local_name)
    resource = cast(OptimadeResource, value)
    if field._local_name in ("id", "type"):
        return getattr(resource, field._local_name)
    attributes = resource.unwrap().get("attributes")
    if not isinstance(attributes, Mapping):
        return None
    return attributes.get(field._remote_name)


def _origin(url: str) -> tuple[str, str, int | None]:
    split = urlsplit(url)
    scheme = split.scheme.casefold()
    port = split.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (split.hostname or "").casefold(), port


def _is_scheme_only_http_upgrade(next_url: str, base_url: str) -> bool:
    """Whether *next_url* differs from *base_url*'s origin only by ``http`` vs ``https``.

    The continuation must use ``http`` where the base uses ``https``, on the
    same casefolded host, with equal effective ports once scheme defaults are
    applied -- so a plain ``http`` continuation (default port 80) counts as the
    same origin as an ``https`` base (default port 443). Any other origin
    difference, including an explicit differing port, returns ``False``.

    :param next_url: The continuation link the service supplied.
    :param base_url: The service's own transport base URL.
    :return: Whether the sole difference is a downgraded ``http`` scheme.
    """

    next_split = urlsplit(next_url)
    base_split = urlsplit(base_url)
    if next_split.scheme.casefold() != "http" or base_split.scheme.casefold() != "https":
        return False
    if (next_split.hostname or "").casefold() != (base_split.hostname or "").casefold():
        return False
    next_port = next_split.port if next_split.port is not None else 80
    base_port = base_split.port if base_split.port is not None else 443
    return next_port == base_port or (next_port == 80 and base_port == 443)


def _upgraded_to_https(url: str, base_url: str) -> str:
    """Return *url* over the base's ``https`` authority, keeping path/query/fragment.

    The continuation's own authority is discarded in favour of the base URL's
    (the host matches by construction), so an explicit ``:80`` on the ``http``
    link cannot survive as a wrong port under ``https``.

    :param url: The continuation link to rewrite.
    :param base_url: The service's transport base URL, supplying the authority.
    :return: The continuation over the base's ``https`` authority.
    """

    split = urlsplit(url)
    base_split = urlsplit(base_url)
    return urlunsplit(("https", base_split.netloc, split.path, split.query, split.fragment))


def _safe_source(url: str) -> str:
    return redact_optimade_url(url)


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise _unsupported("non-finite numeric literals")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in ("", "-0"):
        return "0"
    return rendered


def _fraction_text(value: Fraction) -> str:
    denominator = value.denominator
    twos = 0
    fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise _unsupported(
            f"the non-terminating Fraction literal {value!s}; OPTIMADE numbers cannot represent it exactly"
        )
    scale = max(twos, fives)
    scaled = value.numerator * (2 ** (scale - twos)) * (5 ** (scale - fives))
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(scale + 1, "0")
    if scale == 0:
        return sign + digits
    return sign + digits[:-scale] + "." + digits[-scale:]


def _literal(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, Fraction):
        return _fraction_text(value)
    if isinstance(value, datetime.datetime):
        if value.utcoffset() is None:
            raise _unsupported("naive timestamp literals")
        return json.dumps(value.isoformat(), ensure_ascii=False)
    if value is None:
        raise _unsupported("NULL outside equality or inequality comparisons")
    if isinstance(value, float):
        # A float literal renders as repr()'s shortest round-tripping decimal --
        # the text the caller typed -- so nothing binary leaks onto the wire.
        # Exactness beyond float precision is stated with Decimal instead.
        if not math.isfinite(value):
            raise _unsupported("non-finite numeric literals")
        return _decimal_text(Decimal(repr(value)))
    raise _unsupported(f"literal values of type {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class _RemoteExpression:
    searcher: "RemoteSearcher"
    text: str

    def _other(self, other: object) -> "_RemoteExpression":
        if not isinstance(other, _RemoteExpression) or other.searcher is not self.searcher:
            raise _unsupported("combining expressions from different searchers")
        return other

    def __and__(self, other: object) -> "_RemoteExpression":
        right = self._other(other)
        return _RemoteExpression(self.searcher, f"({self.text}) AND ({right.text})")

    def __or__(self, other: object) -> "_RemoteExpression":
        right = self._other(other)
        return _RemoteExpression(self.searcher, f"({self.text}) OR ({right.text})")

    def __invert__(self) -> "_RemoteExpression":
        return _RemoteExpression(self.searcher, f"NOT ({self.text})")


class _RemoteField:
    __slots__ = (
        "_capabilities",
        "_definition_id",
        "_item_kind",
        "_kind",
        "_local_name",
        "_remote_name",
        "_searcher",
    )

    def __init__(
        self,
        searcher: "RemoteSearcher",
        local_name: str,
        definition_id: str,
        remote_name: str,
        kind: str,
        item_kind: str | None,
        capabilities: PortableQueryCapabilities | None,
    ) -> None:
        self._searcher = searcher
        self._local_name = local_name
        self._definition_id = definition_id
        self._remote_name = remote_name
        self._kind = kind.casefold()
        self._item_kind = None if item_kind is None else item_kind.casefold()
        self._capabilities = capabilities

    def __getattr__(self, name: str) -> object:
        # Unlike _RemoteVariable, a bound field has no map of its own fields
        # to resolve against: nested/relationship traversal is wholly
        # unimplemented, so every name here is a miss regardless of an
        # underscore prefix. Single-leading-underscore names (provider-prefix
        # shaped or not, e.g. IPython's canary probe) therefore still get a
        # plain AttributeError rather than the descriptive error below -- the
        # same non-field-shaped-name safety net as _RemoteVariable, at zero
        # functional cost since there is nothing they could ever resolve to.
        if name.startswith("_"):
            raise AttributeError(name)
        raise _unsupported("field traversal or relationship queries")

    def _comparison(self, operator: str, value: object) -> _RemoteExpression:
        if isinstance(value, _RemoteField):
            raise _unsupported("field-to-field comparisons")
        if self._kind == "list":
            raise _unsupported(f"scalar comparisons on list field {self._local_name!r}")
        if self._kind == "boolean" and operator not in ("=", "!="):
            raise _unsupported(f"ordered comparisons on boolean field {self._local_name!r}")
        self._require("equality" if operator in ("=", "!=") else "ordering")
        if value is None:
            if operator == "=":
                return _RemoteExpression(self._searcher, f"{self._remote_name} IS UNKNOWN")
            if operator == "!=":
                return _RemoteExpression(self._searcher, f"{self._remote_name} IS KNOWN")
            raise _unsupported("ordered comparisons with NULL")
        self._validate(value, self._kind)
        return _RemoteExpression(self._searcher, f"{self._remote_name} {operator} {_literal(value)}")

    def _validate(self, value: object, kind: str | None) -> None:
        if kind in (None, "unknown"):
            return
        valid = (
            (kind == "string" and isinstance(value, str))
            or (kind == "timestamp" and isinstance(value, str | datetime.datetime))
            or (kind == "boolean" and isinstance(value, bool))
            or (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (kind == "float" and isinstance(value, int | float | Decimal | Fraction) and not isinstance(value, bool))
        )
        if not valid:
            raise _unsupported(f"{kind} field {self._local_name!r} with {type(value).__name__} literal")

    def __eq__(self, value: object) -> _RemoteExpression:  # type: ignore[override]
        return self._comparison("=", value)

    def __ne__(self, value: object) -> _RemoteExpression:  # type: ignore[override]
        return self._comparison("!=", value)

    def __lt__(self, value: object) -> _RemoteExpression:
        return self._comparison("<", value)

    def __le__(self, value: object) -> _RemoteExpression:
        return self._comparison("<=", value)

    def __gt__(self, value: object) -> _RemoteExpression:
        return self._comparison(">", value)

    def __ge__(self, value: object) -> _RemoteExpression:
        return self._comparison(">=", value)

    def _string_match(self, operator: str, value: object) -> _RemoteExpression:
        if self._kind != "string":
            raise _unsupported(f"string matching on {self._kind} field {self._local_name!r}")
        self._require("stringmatching")
        if not isinstance(value, str):
            raise _unsupported(f"{operator.lower()} with a non-string literal")
        return _RemoteExpression(self._searcher, f"{self._remote_name} {operator} {_literal(value)}")

    def contains(self, text: str) -> _RemoteExpression:
        return self._string_match("CONTAINS", text)

    def startswith(self, prefix: str) -> _RemoteExpression:
        return self._string_match("STARTS WITH", prefix)

    def endswith(self, suffix: str) -> _RemoteExpression:
        return self._string_match("ENDS WITH", suffix)

    def has(self, value: object) -> _RemoteExpression:
        if self._kind != "list":
            raise _unsupported(f"HAS on non-list field {self._local_name!r}")
        self._require("set")
        self._validate(value, self._item_kind)
        return _RemoteExpression(self._searcher, f"{self._remote_name} HAS {_literal(value)}")

    def has_any(self, *values: object) -> _RemoteExpression:
        if self._kind != "list":
            raise _unsupported(f"HAS ANY on non-list field {self._local_name!r}")
        self._require("set")
        if not values:
            return self._searcher._constant(False)
        for value in values:
            self._validate(value, self._item_kind)
        rendered = ", ".join(_literal(value) for value in values)
        return _RemoteExpression(self._searcher, f"{self._remote_name} HAS ANY {rendered}")

    def has_only(self, *values: object) -> _RemoteExpression:
        if self._kind != "list":
            raise _unsupported(f"HAS ONLY on non-list field {self._local_name!r}")
        self._require("set")
        if not values:
            raise _unsupported("HAS ONLY without values")
        for value in values:
            self._validate(value, self._item_kind)
        rendered = ", ".join(_literal(value) for value in values)
        return _RemoteExpression(self._searcher, f"{self._remote_name} HAS ONLY {rendered}")

    def is_in(self, *values: object) -> _RemoteExpression:
        if self._kind == "list":
            raise _unsupported(f"scalar is_in on list field {self._local_name!r}")
        if not values:
            return self._searcher._constant(False)
        comparisons = [self._comparison("=", value) for value in values]
        expression = comparisons[0]
        for comparison in comparisons[1:]:
            expression = expression | comparison
        return expression

    def _require(self, operation: str) -> None:
        if self._capabilities is None or self._capabilities.supports(operation):
            return
        support = self._capabilities.query_support or "unspecified"
        raise _unsupported(
            f"{operation} on field {self._local_name!r}; its definition declares query-support {support!r}"
        )


class _RemoteVariable:
    __slots__ = ("_fields", "_searcher")

    def __init__(self, searcher: "RemoteSearcher", fields: Mapping[str, _RemoteField]) -> None:
        self._searcher = searcher
        self._fields = fields

    def always_true(self) -> _RemoteExpression:
        return self._searcher._constant(True)

    def always_false(self) -> _RemoteExpression:
        return self._searcher._constant(False)

    def __getattr__(self, name: str) -> "_RemoteField | _RemoteLinks":
        if name.startswith("__"):
            # Dunder probes (``__deepcopy__``, ``__iter__``, ...) never name a
            # field, so reject them before the field map is even consulted.
            raise AttributeError(name)
        if name == "links":
            # Checked before the field map: ``links`` is a JSON:API-reserved
            # member, so even a generic endpoint that happens to advertise a
            # same-named property must still yield the relationship
            # namespace, never that property.
            return _RemoteLinks(self._searcher, self)
        try:
            return self._fields[name]
        except KeyError:
            if name.startswith("_"):
                # A single leading underscore is the OPTIMADE provider-prefix
                # shape (``_<prefix>_<property>``) as well as the shape used
                # by non-field introspection probes such as IPython's
                # ``_ipython_canary_method_should_not_exist_``. Neither is a
                # declared field here, so both get a plain AttributeError
                # rather than the descriptive error below.
                raise AttributeError(name) from None
            descriptor = self._searcher._descriptor
            endpoint = descriptor.name if descriptor is not None else "(unbound)"
            raise _unsupported(f"field {name!r} for endpoint {endpoint!r}") from None


class _RemoteLinks:
    """The ``links`` relationship namespace bound to a root query variable.

    Returned for ``variable.links``. Attribute access names one relationship
    of the queried resource -- a served entry type (``material.links.structures``)
    or a wire-only relationship key such as a StrongLink provenance edge
    (``run.links._httk_has_input``) -- and never rejects the name up front:
    the actual relationship set of a resource is only known from a response,
    so an unresolvable name surfaces as :class:`~httk.store.optimade.client.OptimadeClientError`
    at resolution time (a predicate, output, or ``.links.<name>`` access),
    listing the resource's real relationships.
    """

    __slots__ = ("_searcher", "_variable")

    def __init__(self, searcher: "RemoteSearcher", variable: _RemoteVariable) -> None:
        self._searcher = searcher
        self._variable = variable

    def __getattr__(self, name: str) -> "_RemoteLinkSet":
        if name.startswith("__"):
            raise AttributeError(name)
        descriptor = self._searcher._store.entry_types_by_name.get(name)
        return _RemoteLinkSet(self._searcher, self._variable, name, descriptor)


class _RemoteLinkSet:
    """A depth-1 relationship namespace bound to one named relationship.

    Usable as a predicate root (field chaining renders ``<name>.<field>`` in
    filter text, one level deep and filter-only) and as a set-valued output
    or ``results()`` projection, which requests server-side ``include=`` for
    a served relationship's *name* and yields a tuple of bound related
    records per row. Field chaining is unavailable for a relationship whose
    name is not itself a served entry type (a wire-only key): its members can
    still be projected as an output, just not traversed further in a filter.

    :param searcher: Owning search plan.
    :param variable: Root query variable this relationship namespace belongs to.
    :param name: Relationship name.
    :param descriptor: The related entry type, or ``None`` for a wire-only relationship key.
    """

    __slots__ = ("_descriptor", "_name", "_searcher", "_specs", "_variable")

    def __init__(
        self,
        searcher: "RemoteSearcher",
        variable: _RemoteVariable,
        name: str,
        descriptor: RemoteEntryType | None,
    ) -> None:
        self._searcher = searcher
        self._variable = variable
        self._name = name
        self._descriptor = descriptor
        # Computed once here rather than per attribute access: same specs
        # _bind_descriptor uses for the root variable's own field map. Empty
        # (never None) for a wire-only relationship key, so a probe below
        # needs no descriptor-is-None special case.
        self._specs = {} if descriptor is None else RemoteSearcher._field_specs(descriptor)

    def __getattr__(self, name: str) -> _RemoteField:
        if name.startswith("__"):
            raise AttributeError(name)
        if name.startswith("_") and name not in self._specs:
            # Same provider-prefix/introspection-probe safety net as
            # _RemoteVariable.__getattr__, applied to the related type's own
            # field map -- checked before the not-a-served-entry-type
            # rejection below, so an introspection probe (e.g. IPython's
            # canary) gets a plain AttributeError on a wire-only relationship
            # key too, not the descriptive error.
            raise AttributeError(name) from None
        if self._descriptor is None:
            raise _unsupported(f"field chaining on relationship {self._name!r}: it is not a served entry type")
        spec = self._specs.get(name)
        if spec is None:
            raise _unsupported(f"field {name!r} for related type {self._name!r}")
        definition_id, remote_name, kind, item_kind, capabilities = spec
        return _RemoteField(
            self._searcher,
            f"{self._name}.{name}",
            definition_id,
            f"{self._descriptor.name}.{remote_name}",
            kind,
            item_kind,
            capabilities,
        )


@dataclass(frozen=True, slots=True)
class _Output:
    name: str
    variable: _RemoteVariable | None
    field: _RemoteField | None
    link: str | None = None


_BOUND_STORE = "_httk_bound_store"
_BOUND_RESOURCE = "_httk_bound_resource"
_BOUND_LINKS = "_httk_bound_links"
_BOUND_ATTRIBUTES = "_httk_bound_attributes"
_BOUND_ATTRIBUTE_CACHE = "_httk_bound_attribute_cache"
_MISSING = object()


def _bound_raw_attribute(instance: Any, source: OptimadeResource, name: str) -> object:
    """Read one advertised provider-extension attribute raw from *source*.

    The single raw-lookup body shared by the generic ``OptimadeResource`` bound
    class and every typed bound class's chained ``__getattr__``: the value is
    taken from the source resource's ``attributes``, then this row's
    per-instance cache, then one lazy single-resource fetch. *name* is assumed
    already admitted (an advertised name the endpoint schema does not identify);
    a fetch with no retained store leaves it a plain missing attribute.

    :param instance: The bound row carrying the hidden store, resource, and cache state.
    :param source: The resource whose ``attributes`` hold the raw value.
    :param name: The advertised transport name to read.
    :return: The raw attribute value, present or fetched.
    :raises AttributeError: If the value is absent and no store is retained to fetch it.
    """

    attributes = source.unwrap().get("attributes")
    if isinstance(attributes, Mapping) and name in attributes:
        return attributes[name]
    cached = instance.__dict__.get(_BOUND_ATTRIBUTE_CACHE)
    if isinstance(cached, Mapping) and name in cached:
        return cached[name]
    try:
        store = instance.__dict__[_BOUND_STORE]
        resource = instance.__dict__[_BOUND_RESOURCE]
    except KeyError:
        raise AttributeError(name) from None
    value = RemoteSearcher._fetch_attribute(store, resource, name)
    values = dict(cached) if isinstance(cached, Mapping) else {}
    values[name] = value
    object.__setattr__(instance, _BOUND_ATTRIBUTE_CACHE, MappingProxyType(values))
    return value


@functools.cache
def _bound_class(cls: type) -> type:
    """Return the cached thin per-backend subclass carrying a ``.links`` accessor.

    Mirrors ``httk.store.backend.mongo.store._bound_record_class``: the
    subclass keeps the base dataclass's equality, hash, and repr semantics
    intact (a bound instance and a plain one compare and hash equal) and
    adds the hidden state :func:`RemoteSearcher._wrap` sets, the ``links``
    descriptor, and ``__reduce_ex__`` -- so pickling, copying, or
    :func:`dataclasses.replace` on a bound instance always produces the plain
    base class, never one still carrying a live store reference. Both the
    generic ``OptimadeResource`` class and every typed class expose advertised
    OPTIMADE attributes the endpoint schema does not identify (provider
    extensions) by their transport names -- the generic class exposes every
    advertised name, a typed class only the unidentified ones, so its decoded
    typed properties always win; omitted values are fetched once from the
    single-resource endpoint and memoized per bound record. A typed class
    chains to the backend's own ``__getattr__`` first, preserving any
    record-field delegation.

    :param cls: The plain backend dataclass to derive a bound subclass from.
    :return: The cached bound subclass.
    """

    fields = dataclasses.fields(cls)
    field_names = tuple(field.name for field in fields)
    compare_fields = tuple(field for field in fields if field.compare)
    hash_fields = tuple(field for field in fields if field.hash is True or (field.hash is None and field.compare))
    repr_fields = tuple(field for field in fields if field.repr)
    bound_type: type

    def eq(self: Any, other: Any) -> Any:
        if type(other) is not cls and type(other) is not bound_type:
            return NotImplemented
        return tuple(getattr(self, field.name) for field in compare_fields) == tuple(
            getattr(other, field.name) for field in compare_fields
        )

    def ne(self: Any, other: Any) -> Any:
        result = eq(self, other)
        return NotImplemented if result is NotImplemented else not result

    def bound_hash(self: Any) -> int:
        return hash(tuple(getattr(self, field.name) for field in hash_fields))

    def bound_repr(self: Any) -> str:
        values = ", ".join(f"{field.name}={getattr(self, field.name)!r}" for field in repr_fields)
        return f"{cls.__qualname__}({values})"

    def reduce(self: Any, _protocol: int) -> tuple[type, tuple[Any, ...]]:
        return (cls, tuple(getattr(self, name) for name in field_names))

    # A base ``__getattr__`` to preserve when chaining on a typed bound class
    # (e.g. an entry backend that delegates record fields); ``None`` when the
    # backend defines none, as the OPTIMADE structure and entry backends do.
    base_getattr = None if cls is OptimadeResource else getattr(cls, "__getattr__", None)

    def resource_getattr(self: OptimadeResource, name: str) -> object:
        # The generic OptimadeResource bound class, unchanged: every advertised
        # transport name is exposed raw, read straight off the resource itself.
        if name.startswith("__"):
            raise AttributeError(name)
        advertised = self.__dict__.get(_BOUND_ATTRIBUTES, ())
        if name not in advertised:
            raise AttributeError(name)
        return _bound_raw_attribute(self, self, name)

    def typed_getattr(self: Any, name: str) -> object:
        # A typed bound class: decoded typed properties win via normal lookup
        # (this runs only when that fails). Chain to the backend's own
        # ``__getattr__`` first, when it has one, so record-field delegation is
        # preserved; then expose advertised names the schema does not identify
        # (provider extensions) raw, read from the retained source resource, so
        # standard-name completion never withdraws a provider field.
        if base_getattr is not None:
            try:
                return base_getattr(self, name)
            except AttributeError:
                pass
        if name.startswith("__"):
            raise AttributeError(name)
        advertised = self.__dict__.get(_BOUND_ATTRIBUTES, ())
        if name not in advertised:
            raise AttributeError(name)
        source = self.__dict__.get(_BOUND_RESOURCE)
        if source is None:
            raise AttributeError(name)
        return _bound_raw_attribute(self, cast(OptimadeResource, source), name)

    attrs: dict[str, Any] = {
        "__module__": cls.__module__,
        "__httk_storage_record__": cls,
        "__httk_row_base__": cls,
        "__eq__": eq,
        "__ne__": ne,
        "__hash__": bound_hash,
        "__repr__": bound_repr,
        "__reduce_ex__": reduce,
        "links": _RemoteLinksDescriptor(),
    }
    attrs["__getattr__"] = resource_getattr if cls is OptimadeResource else typed_getattr
    bound_type = type(f"{cls.__name__}Remote", (cls,), attrs)
    return bound_type


class _RemoteLinksDescriptor:
    """The ``links`` non-data descriptor of one bound remote record instance.

    Mirrors ``httk.store.store_common._LinksDescriptor``: it reads the
    hidden store/resource reference :func:`RemoteSearcher._wrap` sets, then
    returns a memoized :class:`_RemoteLinksAccessor`. A plain instance, or a
    bound instance produced by :func:`dataclasses.replace` (which always
    constructs a fresh instance through the ordinary ``__init__``), carries
    neither and so simply has no ``links`` attribute.
    """

    __slots__ = ()

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        try:
            store = instance.__dict__[_BOUND_STORE]
            resource = instance.__dict__[_BOUND_RESOURCE]
        except KeyError:
            raise AttributeError("links") from None
        cached = instance.__dict__.get(_BOUND_LINKS, _MISSING)
        if cached is not _MISSING:
            return cached
        accessor = _RemoteLinksAccessor(store, resource)
        object.__setattr__(instance, _BOUND_LINKS, accessor)
        return accessor


class _RemoteLinksAccessor:
    """Lazy, per-instance ``.links.<name>`` resolution for one bound record.

    Identifiers are matched against the response's ``included`` array by
    their own ``(type, id)``, never by the relationship block key *name*
    itself: some relationships (StrongLink provenance edges) use wire keys
    that differ from the identifier's own ``type``. A related resource
    already present in ``included`` -- via an output's automatic ``include=``
    or a service's own default includes -- is wrapped in place at no extra
    cost; one absent from ``included`` costs one HTTP request per missing
    identifier. Each named relationship resolves once and is memoized.

    :param store: Owning store, used for discovery and any lazy fetch.
    :param resource: The bound record's own source resource.
    :raises OptimadeClientError: If the store is closed, the relationship or an identifier's entry type is unresolvable, or a lazy fetch does not resolve to exactly one resource.
    :raises OptimadeResponseError: If ``included`` or a matched member is malformed.
    """

    __slots__ = ("_cache", "_resource", "_store")

    def __init__(self, store: OptimadeStore, resource: OptimadeResource) -> None:
        self._store = store
        self._resource = resource
        self._cache: dict[str, tuple[object, ...]] = {}

    def __getattr__(self, name: str) -> tuple[object, ...]:
        if name.startswith("__"):
            # Unlike a store-declared link name, an OPTIMADE relationship key
            # routinely carries a single leading underscore (a provider
            # prefix, e.g. ``_httk_records``) or names a StrongLink wire key
            # (``_httk_has_input``), so only actual dunder introspection
            # probes are rejected here.
            raise AttributeError(name)
        if name not in self._cache:
            self._cache[name] = self._resolve(name)
        return self._cache[name]

    def _resolve(self, name: str) -> tuple[object, ...]:
        self._store._require_open()
        resource = self._resource
        relationships = resource.get("relationships")
        if not isinstance(relationships, Mapping) or name not in relationships:
            available = sorted(relationships) if isinstance(relationships, Mapping) else []
            raise OptimadeClientError(
                f"resource {resource.id!r} of type {resource.type!r} has no relationship {name!r}; "
                f"available relationships: {available}"
            )
        block = relationships[name]
        data_raw = block.get("data") if isinstance(block, Mapping) else None
        if not data_raw:
            return ()
        if not isinstance(data_raw, tuple):
            raise OptimadeResponseError(f"relationship {name!r} data must be an array")
        data = data_raw

        included_raw = optimade_document_root(resource.document).get("included")
        if included_raw is None:
            included: tuple[object, ...] = ()
        elif isinstance(included_raw, tuple):
            included = included_raw
        else:
            raise OptimadeResponseError("OPTIMADE document 'included' must be an array when present")
        validated_included = [
            RemoteSearcher._validate_resource_item(item, index, member="included")
            for index, item in enumerate(included)
        ]

        results: list[object] = []
        for identifier in data:
            if not isinstance(identifier, Mapping):
                raise OptimadeResponseError(f"relationship {name!r} identifier must be an object")
            id_type = identifier.get("type")
            id_value = identifier.get("id")
            if not isinstance(id_type, str) or not id_type or not isinstance(id_value, str) or not id_value:
                raise OptimadeResponseError(
                    f"relationship {name!r} identifier must have nonempty string 'type' and 'id'"
                )
            descriptor = self._store.entry_types_by_name.get(id_type)
            if descriptor is None:
                raise OptimadeClientError(f"relationship {name!r} identifier names unknown entry type {id_type!r}")
            match_index = next(
                (
                    index
                    for index, item in enumerate(validated_included)
                    if item["type"] == id_type and item["id"] == id_value
                ),
                None,
            )
            if match_index is not None:
                included_resource = OptimadeResource(
                    resource.document, match_index, descriptor.schema, member="included"
                )
                results.append(RemoteSearcher._wrap(self._store, descriptor, included_resource))
                continue
            # Not included: one lazy fetch per missing identifier.
            search = self._store.searcher()
            variable = search.variable(descriptor)
            search.add(variable.id == id_value)
            try:
                row = search.results(item=variable).one()
            except (NoResultError, MultipleResultsError) as exc:
                raise OptimadeClientError(
                    f"related {id_type!r} identifier {id_value!r} did not resolve to exactly one resource"
                ) from exc
            results.append(row.item)
        return tuple(results)


class RemoteSearcher:
    """Build one portable single-root OPTIMADE query.

    :param store: Remote OPTIMADE store used for discovery and requests.
    :param response_fields: Optional field-selection policy for this search.
    """

    def __init__(self, store: OptimadeStore, *, response_fields: object = None) -> None:
        self._store = store
        self._response_fields_setting = response_fields
        self._response_transport_fields: tuple[str, ...] | None = None
        self._descriptor: RemoteEntryType | None = None
        self._variable: _RemoteVariable | None = None
        self._fields: Mapping[str, _RemoteField] = MappingProxyType({})
        self._expressions: list[_RemoteExpression] = []
        self._sorts: list[tuple[_RemoteField, bool]] = []
        self._outputs: list[_Output] = []
        self._limit: int | None = None
        self.offset = 0
        self._count_cache = _CountCache()

    def _clone(self) -> "RemoteSearcher":
        clone = type(self)(self._store, response_fields=self._response_fields_setting)
        if self._descriptor is not None:
            clone._bind_descriptor(self._descriptor)
            clone._response_transport_fields = self._response_transport_fields
            clone._expressions = [_RemoteExpression(clone, expression.text) for expression in self._expressions]
            clone._sorts = [(clone._fields[field._local_name], descending) for field, descending in self._sorts]
            clone._outputs = [
                _Output(
                    output.name,
                    clone._variable if output.variable is not None else None,
                    clone._fields[cast(_RemoteField, output.field)._local_name] if output.field is not None else None,
                    output.link,
                )
                for output in self._outputs
            ]
        clone._limit = self._limit
        clone.offset = self.offset
        clone._count_cache = self._count_cache
        return clone

    def _invalidate_count(self) -> None:
        self._count_cache = _CountCache()

    def _resolve_descriptor(self, target: object) -> RemoteEntryType:
        if isinstance(target, RemoteEntryType):
            if not any(target is item for item in self._store.entry_types):
                raise _unsupported("a RemoteEntryType from another store or stale discovery snapshot")
            return target
        if isinstance(target, type):
            # A previously returned bound record's own class (see _bound_class)
            # names its plain base here, so variable(type(some_result)) resolves
            # exactly like variable(<the plain backend class>).
            base = cast(type, getattr(target, "__httk_row_base__", target))
            matches = tuple(item for item in self._store.entry_types if item.backend is base)
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise _unsupported(f"the unrecognized backend class {target.__name__}")
            raise _unsupported(
                f"backend class {target.__name__} because it matches multiple endpoints; pass a RemoteEntryType"
            )
        raise _unsupported("query targets other than a RemoteEntryType or registered backend class")

    @staticmethod
    def _typed_maps(
        descriptor: RemoteEntryType,
    ) -> tuple[
        dict[str, str],
        dict[str, str],
        dict[str, tuple[str, str | None]],
        dict[str, PortableQueryCapabilities],
    ]:
        binding = descriptor.binding
        if binding is None:
            generic_fields = {name: name for name in descriptor.advertised_properties}
            generic_all_fields = dict(generic_fields)
            generic_kinds: dict[str, tuple[str, str | None]] = {
                **descriptor.property_types,
                "id": ("string", None),
                "type": ("string", None),
            }
            for generic_local_name, generic_definition_id in (("id", _CORE_ID), ("type", _CORE_TYPE)):
                # id/type are protocol envelope members even for old/generic
                # schemas that predate property definition IRIs.
                generic_remote_name = descriptor.property_names.get(generic_definition_id, generic_local_name)
                generic_fields[generic_local_name] = generic_remote_name
                generic_all_fields[generic_local_name] = generic_remote_name
            return generic_fields, generic_all_fields, generic_kinds, {}

        schema = load_entry_type_definition(binding.definition_id)
        definitions_by_iri = {definition.definition_id: name for name, definition in schema.properties.items()}
        typed_all_fields = {
            name: descriptor.property_names[definition.definition_id]
            for name, definition in schema.properties.items()
            if definition.definition_id in descriptor.property_names
        }
        if binding.query_fields is None:
            local_names = portable_query_fields(schema)
            query_iris = tuple(schema.properties[name].definition_id for name in local_names)
        else:
            query_iris = binding.query_fields
        typed_fields: dict[str, str] = {}
        typed_kinds: dict[str, tuple[str, str | None]] = {}
        typed_capabilities: dict[str, PortableQueryCapabilities] = {}
        for definition_id in query_iris:
            typed_local_name = definitions_by_iri.get(definition_id)
            typed_remote_name = descriptor.property_names.get(definition_id)
            if typed_local_name is not None and typed_remote_name is not None:
                typed_fields[typed_local_name] = typed_remote_name
                definition = schema.properties[typed_local_name]
                payload = definition.as_optimade()
                items = payload.get("items")
                item_kind = (
                    cast(str, items.get("x-optimade-type"))
                    if isinstance(items, Mapping) and isinstance(items.get("x-optimade-type"), str)
                    else None
                )
                typed_kinds[typed_local_name] = (definition.optimade_type, item_kind)
                typed_capabilities[typed_local_name] = portable_query_capabilities(definition)
        # Advertised names the schema does not identify (no declared or inferred
        # definition IRI) -- provider-prefixed extensions above all -- stay
        # queryable and projectable under their exact wire names, exactly as a
        # generic endpoint exposes them. Standard-name completion only adds
        # semantic identities; it never withdraws a provider field. A name that
        # is a standard local name of this entry type is skipped even without an
        # identity here: it belongs to the semantic vocabulary (its typed
        # mapping, when present, may carry a different wire name) and must never
        # be shadowed by a same-spelled unidentified advertised field.
        for advertised_name in descriptor.advertised_properties:
            if (
                advertised_name in descriptor.property_iris
                or advertised_name in schema.properties
                or advertised_name in typed_all_fields
            ):
                continue
            typed_fields[advertised_name] = advertised_name
            typed_all_fields[advertised_name] = advertised_name
            typed_kinds[advertised_name] = descriptor.property_types.get(advertised_name, ("unknown", None))
        return typed_fields, typed_all_fields, typed_kinds, typed_capabilities

    def _select_response_fields(
        self,
        descriptor: RemoteEntryType,
        all_fields: Mapping[str, str],
    ) -> tuple[str, ...] | None:
        setting = self._response_fields_setting
        if setting is None:
            return None
        if setting is ALL_ADVERTISED:
            return descriptor.advertised_properties
        if isinstance(setting, str):
            raise TypeError("response_fields must be an iterable of field names, ALL_ADVERTISED, or None")
        if not isinstance(setting, Iterable):
            raise TypeError("response_fields must be an iterable of field names, ALL_ADVERTISED, or None")
        selected: list[str] = []
        seen: set[str] = set()
        for name in setting:
            if not isinstance(name, str):
                raise TypeError("response_fields names must be strings")
            remote_name = all_fields.get(name)
            if remote_name is None and descriptor.binding is None and name in descriptor.advertised_properties:
                # Generic descriptors have no local semantic vocabulary beyond
                # id/type; explicit response selection therefore uses their
                # exact advertised transport names.
                remote_name = name
            if remote_name is None:
                raise _unsupported(f"response field {name!r} for endpoint {descriptor.name!r}")
            if remote_name not in seen:
                seen.add(remote_name)
                selected.append(remote_name)
        return tuple(selected)

    @staticmethod
    def _field_specs(
        descriptor: RemoteEntryType,
    ) -> dict[str, tuple[str, str, str, str | None, PortableQueryCapabilities | None]]:
        """Return one ``(definition_id, remote_name, kind, item_kind, capabilities)``
        spec per query-supported field, keyed by local name -- the exact
        ingredients :class:`_RemoteField` needs.

        Shared by :meth:`_bind_descriptor` (building a root query variable)
        and :class:`_RemoteLinkSet` (building depth-1 relationship fields),
        so definition-IRI derivation for a generic vs. a bound descriptor is
        written, and computed, exactly once for both call sites.

        :param descriptor: Entry type to derive field specs for.
        :return: Field specs keyed by local name.
        """

        query_fields, _all_fields, kinds, capabilities = RemoteSearcher._typed_maps(descriptor)
        if descriptor.binding is None:
            definition_ids = {
                name: descriptor.property_iris.get(name, name)
                for name in ("id", "type", *descriptor.advertised_properties)
            }
            definition_ids["id"] = _CORE_ID
            definition_ids["type"] = _CORE_TYPE
        else:
            schema = load_entry_type_definition(descriptor.binding.definition_id)
            definition_ids = {name: definition.definition_id for name, definition in schema.properties.items()}
        return {
            # A schema-unknown advertised name (a provider extension exposed
            # generically) has no schema IRI; its own transport name stands in,
            # exactly as for a generic descriptor above.
            local_name: (
                definition_ids.get(local_name, descriptor.property_iris.get(local_name, local_name)),
                remote_name,
                *kinds[local_name],
                capabilities.get(local_name),
            )
            for local_name, remote_name in query_fields.items()
        }

    def _bind_descriptor(self, descriptor: RemoteEntryType) -> _RemoteVariable:
        _query_fields, all_fields, _kinds, _capabilities = self._typed_maps(descriptor)
        fields = {
            local_name: _RemoteField(self, local_name, *spec)
            for local_name, spec in self._field_specs(descriptor).items()
        }
        self._descriptor = descriptor
        self._fields = MappingProxyType(fields)
        self._response_transport_fields = self._select_response_fields(descriptor, all_fields)
        self._variable = _RemoteVariable(self, self._fields)
        self._invalidate_count()
        return self._variable

    def variable(self, target: object) -> _RemoteVariable:
        """Bind the query to one discovered remote entry type.

        :param target: Discovered entry descriptor or registered backend class.
        :return: Query variable exposing portable fields.
        :raises httk.store.query.protocols.UnsupportedQueryError: If the target is not recognized or a root variable is already bound.
        """
        if self._variable is not None:
            raise _unsupported("a second root variable")
        return self._bind_descriptor(self._resolve_descriptor(target))

    def _require_variable(self) -> tuple[RemoteEntryType, _RemoteVariable]:
        if self._descriptor is None or self._variable is None:
            raise ValueError("this searcher has no query variable; call variable() first")
        return self._descriptor, self._variable

    def _constant(self, value: bool) -> _RemoteExpression:
        _descriptor, variable = self._require_variable()
        try:
            id_field = variable.id
        except UnsupportedQueryError as exc:
            raise _unsupported("constant expressions when semantic id is not advertised") from exc
        state = "KNOWN" if value else "UNKNOWN"
        return _RemoteExpression(self, f"{id_field._remote_name} IS {state}")

    def add(self, expression: object) -> None:
        """Add a filter expression to the query.

        :param expression: Expression created by this searcher.
        :raises ValueError: If no query variable is bound.
        :raises httk.store.UnsupportedQueryError: If the expression belongs elsewhere.
        """
        self._require_variable()
        if not isinstance(expression, _RemoteExpression) or expression.searcher is not self:
            raise _unsupported("expressions from another backend or searcher")
        self._expressions.append(expression)
        self._invalidate_count()

    def output(self, variable: object, name: str) -> None:
        """Declare a whole-record, scalar, or set-valued relationship output.

        A relationship namespace (``variable.links.<name>``) is a set-valued
        output: it yields a tuple of bound related records per row, resolved
        from the response's ``included`` array or one lazy fetch per missing
        identifier -- the same resolution a returned record's own
        ``.links.<name>`` performs. Served-entry-type relationship names are
        also added to the request's ``include=`` parameter automatically.

        :param variable: Root variable, portable scalar field, or relationship namespace to project.
        :param name: Output name.
        :raises ValueError: If the name is empty or duplicated.
        :raises httk.store.UnsupportedQueryError: If the output belongs elsewhere.
        """
        _descriptor, root = self._require_variable()
        if not isinstance(name, str) or not name:
            raise ValueError("output name must be a nonempty string")
        if any(output.name == name for output in self._outputs):
            raise ValueError(f"duplicate output name: {name!r}")
        if variable is root:
            output = _Output(name, root, None)
        elif isinstance(variable, _RemoteField) and variable._searcher is self:
            if self._fields.get(variable._local_name) is not variable:
                # A chained relationship field (_RemoteLinkSet.__getattr__) is
                # never a value of this searcher's own field map -- it is
                # filter-only.
                raise _unsupported("related fields as outputs or sort keys")
            output = _Output(name, None, variable)
        elif isinstance(variable, _RemoteLinkSet) and variable._searcher is self:
            if variable._variable is not root:
                raise _unsupported("link sets from another backend or searcher")
            output = _Output(name, None, None, variable._name)
        else:
            raise _unsupported("outputs other than the root record or a portable scalar field")
        self._outputs.append(output)

    def _effective_response_fields(self) -> tuple[str, ...] | None:
        """Return the request field selection implied by outputs and user policy.

        A scalar projection has to be requested when the service's default is
        not known to include it.  For a scalar-only result that is harmless.
        For a whole-resource result, retain every field the endpoint declares
        default-response and add the scalar.  If the endpoint omits that
        implementation metadata, the conservative fallback is every
        advertised field rather than a thin scalar-only resource.
        """

        descriptor, _variable = self._require_variable()
        scalar_fields = tuple(
            dict.fromkeys(
                cast(_RemoteField, output.field)._remote_name for output in self._outputs if output.field is not None
            )
        )
        if not scalar_fields:
            return self._response_transport_fields
        if self._response_transport_fields is None:
            has_record = any(output.variable is not None for output in self._outputs)
            if has_record:
                # id and type are JSON:API resource members, independent of
                # response field policy.  The remaining default fields must be
                # declared by the endpoint itself.
                all_fields = self._typed_maps(descriptor)[1]
                protocol_fields = tuple(
                    remote_name
                    for local_name in ("id", "type")
                    if (remote_name := all_fields.get(local_name)) is not None
                )
                default_fields = set(descriptor.default_response_properties)
                default_fields.update(protocol_fields)
                if all(field in default_fields for field in scalar_fields):
                    # No explicit parameter is needed when every projection
                    # is already guaranteed by the selected default policy.
                    return None
                if descriptor.default_response_properties:
                    selected = list(dict.fromkeys((*descriptor.default_response_properties, *protocol_fields)))
                else:
                    selected = list(descriptor.advertised_properties)
                for field in scalar_fields:
                    if field not in selected:
                        selected.append(field)
                return tuple(selected)
            # A scalar-only result has no raw-resource completeness promise,
            # so ask for exactly the fields it projects.
            return scalar_fields
        selected = list(self._response_transport_fields)
        for field in scalar_fields:
            if field not in selected:
                selected.append(field)
        return tuple(selected)

    def add_sort(self, field: object, descending: bool = False) -> None:
        """Append a sortable field to the remote query.

        :param field: Field exposed by this searcher's variable.
        :param descending: Sort in descending order when true.
        :raises httk.store.UnsupportedQueryError: If the field is not portable or sortable.
        """
        descriptor, _variable = self._require_variable()
        if not isinstance(field, _RemoteField) or field._searcher is not self:
            raise _unsupported("sort keys from another backend or searcher")
        if self._fields.get(field._local_name) is not field:
            raise _unsupported("related fields as outputs or sort keys")
        if not isinstance(descending, bool):
            raise TypeError("descending must be a bool")
        if field._remote_name not in descriptor.sortable_properties:
            raise _unsupported(f"sorting by non-sortable field {field._local_name!r}")
        self._sorts.append((field, descending))

    def set_limit(self, limit: int) -> None:
        """Set the query result limit.

        :param limit: Nonnegative limit, or a negative value for no bound.
        """
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer")
        self._limit = None if limit < 0 else limit

    def add_offset(self, offset: int) -> None:
        """Advance the query offset.

        :param offset: Nonnegative number of matching rows to skip.
        """
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise TypeError("offset must be an integer")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        self.offset += offset

    def _include_names(self) -> tuple[str, ...]:
        """Return the ordered, de-duplicated ``include=`` targets implied by outputs.

        Only relationship outputs naming a served entry type are included:
        a wire-only relationship key (a StrongLink edge such as
        ``_httk_has_input``) is never a served entry type and always
        resolves by lazy per-identifier fetch instead.
        """

        return tuple(
            dict.fromkeys(
                output.link
                for output in self._outputs
                if output.link is not None and output.link in self._store.entry_types_by_name
            )
        )

    def _filter_text(self) -> str | None:
        if not self._expressions:
            return None
        return " AND ".join(f"({expression.text})" for expression in self._expressions)

    def _request_url(
        self,
        *,
        page_limit: int,
        include_sort: bool = True,
        include_related: bool = True,
        response_fields: tuple[str, ...] | None | object = ...,
    ) -> str:
        descriptor, _variable = self._require_variable()
        parameters: list[tuple[str, str]] = []
        filter_text = self._filter_text()
        if filter_text is not None:
            parameters.append(("filter", filter_text))
        fields = self._effective_response_fields() if response_fields is ... else response_fields
        if fields:
            parameters.append(("response_fields", ",".join(cast(tuple[str, ...], fields))))
        if include_sort and self._sorts:
            parameters.append(
                (
                    "sort",
                    ",".join(("-" if descending else "") + field._remote_name for field, descending in self._sorts),
                )
            )
        if include_related:
            include_names = self._include_names()
            if include_names:
                parameters.append(("include", ",".join(include_names)))
        parameters.append(("page_limit", str(page_limit)))
        return self._store._transport_base_url + "/" + quote(descriptor.name, safe="") + "?" + urlencode(parameters)

    @staticmethod
    def _raw_root(text: str, source_url: str) -> Mapping[str, object]:
        try:
            root = json.loads(text, parse_float=Decimal, parse_int=int)
        except json.JSONDecodeError as exc:
            raise OptimadeResponseError(
                f"OPTIMADE response from {_safe_source(source_url)!r} is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(root, dict):
            raise OptimadeResponseError(f"OPTIMADE response from {_safe_source(source_url)!r} root must be an object")
        errors = root.get("errors")
        if isinstance(errors, list) and errors:
            raise OptimadeResponseError(
                f"OPTIMADE response from {_safe_source(source_url)!r} contains an errors document"
            )
        return root

    @staticmethod
    def _next_link(root: Mapping[str, object], source_url: str) -> str | None:
        links = root.get("links")
        if links is None:
            return None
        if not isinstance(links, dict):
            raise OptimadePaginationError(f"OPTIMADE response from {_safe_source(source_url)!r} has non-object links")
        next_link = links.get("next")
        if next_link is None:
            return None
        if isinstance(next_link, str) and next_link and next_link == next_link.strip():
            return next_link
        if isinstance(next_link, dict):
            href = next_link.get("href")
            if isinstance(href, str) and href and href == href.strip():
                return href
        raise OptimadePaginationError(f"OPTIMADE response from {_safe_source(source_url)!r} has malformed links.next")

    @staticmethod
    def _more_data(root: Mapping[str, object], source_url: str) -> bool:
        meta = root.get("meta")
        if meta is None:
            return False
        if not isinstance(meta, dict):
            raise OptimadeResponseError(f"OPTIMADE response from {_safe_source(source_url)!r} has non-object meta")
        value = meta.get("more_data_available", False)
        if not isinstance(value, bool):
            raise OptimadeResponseError(
                f"OPTIMADE response from {_safe_source(source_url)!r} has non-boolean meta.more_data_available"
            )
        return value

    @staticmethod
    def _validate_entry_page(
        root: Mapping[str, object],
        source_url: str,
        endpoint: str,
    ) -> list[Mapping[str, object]]:
        """Validate a complete successful entry envelope before yielding it.

        OPTIMADE list responses use an array ``data`` member and an object
        ``meta`` member.  When present, ``data_returned`` is the total number of
        data objects for the current filter query independent of pagination, so
        it is a nonnegative integer bearing no fixed relation to this page's
        length.  Resource-level checks
        intentionally stop at JSON:API envelope shapes: extension members and
        arbitrary link objects remain lossless raw data rather than being
        interpreted or rejected here.
        """

        raw_data = root.get("data")
        if not isinstance(raw_data, list):
            raise OptimadeResponseError(f"OPTIMADE response from {_safe_source(source_url)!r} data must be an array")
        meta = root.get("meta")
        if not isinstance(meta, dict):
            raise OptimadeResponseError(f"OPTIMADE response from {_safe_source(source_url)!r} has no object meta")
        data_returned = meta.get("data_returned")
        if data_returned is not None and (
            not isinstance(data_returned, int) or isinstance(data_returned, bool) or data_returned < 0
        ):
            raise OptimadeResponseError(
                f"OPTIMADE response from {_safe_source(source_url)!r} has invalid meta.data_returned"
            )
        more_data = meta.get("more_data_available", False)
        if not isinstance(more_data, bool):
            raise OptimadeResponseError(
                f"OPTIMADE response from {_safe_source(source_url)!r} has non-boolean meta.more_data_available"
            )

        return [
            RemoteSearcher._validate_resource_item(item, data_index, member="data", endpoint=endpoint)
            for data_index, item in enumerate(raw_data)
        ]

    @staticmethod
    def _validate_resource_item(
        item: object,
        index: int | None,
        *,
        member: str,
        endpoint: str | None = None,
    ) -> Mapping[str, object]:
        """Validate one JSON:API resource object from a response envelope.

        Shared by primary ``data`` page validation (``_validate_entry_page``)
        and included-member validation (``_RemoteLinksAccessor``): both apply
        the identical envelope-shape checks -- an object with nonempty string
        ``id``/``type`` and object-valued ``attributes``/``relationships``
        when present -- differing only in whether ``type`` must match one known
        endpoint.

        :param item: Candidate resource object.
        :param index: Index within the ``member`` array, or ``None`` for a single-resource member.
        :param member: Envelope member name, for diagnostics (``"data"`` or ``"included"``).
        :param endpoint: Required resource ``type``, or ``None`` to accept any type.
        :return: The validated resource object.
        :raises OptimadeResponseError: If the resource object is malformed.
        """

        # Mapping, not dict: primary-page items come from plain json.loads()
        # dicts, but included-member items (_RemoteLinksAccessor) come from
        # the frozen, already-redacted document root, whose objects are
        # immutable MappingProxyType instances rather than dicts.
        location = member if index is None else f"{member}[{index}]"
        if not isinstance(item, Mapping):
            raise OptimadeResponseError(f"OPTIMADE response {location} must be an object")
        for envelope_member in ("id", "type"):
            envelope_value = item.get(envelope_member)
            if not isinstance(envelope_value, str) or not envelope_value:
                raise OptimadeResponseError(f"OPTIMADE response {location}.{envelope_member} must be a nonempty string")
        if endpoint is not None and item["type"] != endpoint:
            raise OptimadeResponseError(
                f"OPTIMADE response {location}.type does not match queried endpoint {endpoint!r}"
            )
        for sub_member in ("attributes", "relationships"):
            if sub_member in item and not isinstance(item[sub_member], Mapping):
                raise OptimadeResponseError(f"OPTIMADE response {location}.{sub_member} must be an object when present")
        return item

    @staticmethod
    def _fetch_attribute(store: OptimadeStore, resource: OptimadeResource, name: str) -> object:
        """Fetch one omitted advertised attribute from its single-resource endpoint."""

        url = (
            store._transport_base_url
            + "/"
            + quote(resource.type, safe="")
            + "/"
            + quote(resource.id, safe="")
            + "?"
            + urlencode({"response_fields": name})
        )
        text = store._get(url)
        root = RemoteSearcher._raw_root(text, url)
        RemoteSearcher._log_page_warnings(root)
        item = RemoteSearcher._validate_resource_item(root.get("data"), None, member="data", endpoint=resource.type)
        if item["id"] != resource.id:
            raise OptimadeResponseError(
                f"OPTIMADE single-resource response id {item['id']!r} does not match requested id {resource.id!r}"
            )
        document = OptimadeDocument.from_response(text, url)
        fetched = OptimadeResource(document, 0, resource.schema)
        attributes = fetched.unwrap().get("attributes")
        return attributes.get(name) if isinstance(attributes, Mapping) else None

    @staticmethod
    def _wrap(store: OptimadeStore, descriptor: RemoteEntryType, resource: OptimadeResource) -> object:
        """Wrap one resource with its entry type's backend, bound for ``.links`` access.

        The single construction point for every whole-record result: the
        returned instance is always of a thin per-backend bound subclass
        (see :func:`_bound_class`) carrying a hidden store/resource
        reference, so its ``links`` accessor and any relationship output
        resolved from it share exactly one resolution path
        (:class:`_RemoteLinksAccessor`).

        :param store: Owning store, retained for lazy relationship resolution.
        :param descriptor: Entry type descriptor owning the wrapping backend.
        :param resource: Resource to wrap.
        :return: The bound backend instance, or a bound generic resource for an unbound entry type.
        """

        if descriptor.backend is OptimadeResource:
            bound_cls = _bound_class(OptimadeResource)
            instance = bound_cls(resource.document, resource.data_index, resource.schema, resource.member)
            object.__setattr__(instance, _BOUND_ATTRIBUTES, descriptor.advertised_properties)
        else:
            bound_cls = _bound_class(descriptor.backend)
            instance = cast(Callable[[OptimadeResource], object], bound_cls)(resource)
            # Advertised names the schema does not identify (provider extensions)
            # stay reachable as raw attributes on the typed row, exactly as on a
            # generic row. A name is excluded both when it carries a definition
            # IRI (declared or inferred) and when it is a standard local name of
            # this entry type -- a version-gated standard name (e.g. a files
            # ``url`` on a 1.1.0 service) belongs to the semantic vocabulary and
            # must never leak a raw value under a standard spelling. This mirrors
            # the exclusion in :meth:`_typed_maps`. ``binding`` is never ``None``
            # here, since a non-generic backend is always derived from one.
            binding = descriptor.binding
            schema_names = () if binding is None else load_entry_type_definition(binding.definition_id).properties
            object.__setattr__(
                instance,
                _BOUND_ATTRIBUTES,
                tuple(
                    name
                    for name in descriptor.advertised_properties
                    if name not in descriptor.property_iris and name not in schema_names
                ),
            )
        object.__setattr__(instance, _BOUND_STORE, store)
        object.__setattr__(instance, _BOUND_RESOURCE, resource)
        return instance

    @staticmethod
    def _log_page_warnings(root: Mapping[str, object]) -> None:
        """Log every ``meta.warnings`` entry of one validated response page.

        A dotted filter whose first segment is a served type but not a
        relationship of the queried endpoint returns HTTP 200 with zero rows
        and only a ``meta.warnings`` entry to explain why -- this is the
        client's only surface for that condition, so it is logged rather than
        silently dropped.
        """

        meta = root.get("meta")
        if not isinstance(meta, dict):
            return
        warnings = meta.get("warnings")
        if not isinstance(warnings, list) or not warnings:
            return
        logger = logging.getLogger(__name__)
        for warning in warnings:
            detail = warning.get("detail") if isinstance(warning, Mapping) else None
            logger.warning(detail if isinstance(detail, str) else json.dumps(warning), extra={"context": "optimade"})

    def _objects(
        self,
        *,
        maximum: int | None = None,
        page_limit_override: int | None = None,
    ) -> Iterator[tuple[object, OptimadeResource]]:
        descriptor, _variable = self._require_variable()
        effective_limit = self._limit
        if maximum is not None:
            effective_limit = maximum if effective_limit is None else min(effective_limit, maximum)
        if effective_limit == 0:
            return iter(())
        request_page_limit = self._store.page_limit
        if page_limit_override is not None:
            request_page_limit = min(request_page_limit, page_limit_override)
        if effective_limit is not None:
            request_page_limit = max(1, min(request_page_limit, effective_limit + self.offset))
        base_origin = _origin(self._store._transport_base_url)

        def resources() -> Iterator[tuple[object, OptimadeResource]]:
            raw_text: str | None
            raw_text, first_url = self._store._get_page(
                lambda limit: self._request_url(page_limit=limit), request_page_limit
            )
            next_url: str | None = first_url
            seen: set[str] = set()
            pages = 0
            skipped = 0
            emitted = 0
            while next_url is not None:
                split = urlsplit(next_url)
                if split.scheme not in ("http", "https") or not split.netloc:
                    raise OptimadePaginationError("OPTIMADE pagination continuation must be an absolute HTTP(S) URL")
                if _origin(next_url) != base_origin:
                    # A continuation that only downgraded https to http on the
                    # same host is upgraded back to https (the http URL is a
                    # spec deviation and typically answers a redirect); every
                    # other origin difference still needs cross-origin consent.
                    # The upgrade happens before the cycle check so a service
                    # repeating the same http link is still caught as a cycle.
                    if self._store.tolerate_deviations and _is_scheme_only_http_upgrade(
                        next_url, self._store._transport_base_url
                    ):
                        self._store._record_deviation(
                            "continuation-scheme",
                            self._store._transport_base_url,
                            "continuation link downgraded scheme to http; upgraded to https for the same host",
                        )
                        next_url = _upgraded_to_https(next_url, self._store._transport_base_url)
                    elif not self._store.allow_cross_origin_pagination:
                        raise OptimadePaginationError("OPTIMADE pagination attempted a cross-origin request")
                if next_url in seen:
                    raise OptimadePaginationError("OPTIMADE pagination cycle detected")
                if pages >= self._store.max_pages:
                    raise OptimadePaginationError(f"OPTIMADE pagination exceeded max_pages={self._store.max_pages}")
                seen.add(next_url)
                pages += 1
                if raw_text is None:
                    raw_text = self._store._get(next_url)
                # Continuations are extracted from the ephemeral raw response
                # so a credential-bearing cursor remains usable. Only the
                # redacted document below is retained by yielded resources.
                raw_root = self._raw_root(raw_text, next_url)
                raw_data = self._validate_entry_page(raw_root, next_url, descriptor.name)
                self._log_page_warnings(raw_root)
                document = OptimadeDocument.from_response(raw_text, next_url)
                safe_root = optimade_document_root(document)
                safe_data = safe_root.get("data")
                if not isinstance(safe_data, tuple) or len(safe_data) != len(raw_data):
                    raise OptimadeResponseError("redacted OPTIMADE document changed the data envelope")
                for data_index, item in enumerate(raw_data):
                    if skipped < self.offset:
                        skipped += 1
                        continue
                    resource = OptimadeResource(document, data_index, descriptor.schema)
                    yield self._wrap(self._store, descriptor, resource), resource
                    emitted += 1
                    if effective_limit is not None and emitted >= effective_limit:
                        return
                continuation = self._next_link(raw_root, next_url)
                if continuation is None:
                    if self._more_data(raw_root, next_url):
                        raise OptimadePaginationError(
                            "OPTIMADE response reports more_data_available without a usable links.next"
                        )
                    return
                raw_text = None
                next_url = urljoin(next_url, continuation)

        return resources()

    def _search_results(self, *, maximum: int | None = None) -> Iterator[SearchResult]:
        if not self._outputs:
            raise ValueError("this search has no outputs; call output() before iterating")
        descriptor, _variable = self._require_variable()
        generic = descriptor.backend is OptimadeResource
        names = tuple(output.name for output in self._outputs)
        for value, _resource in self._objects(maximum=maximum, page_limit_override=maximum):
            # value is always the bound whole-record instance (see _wrap), so
            # a relationship output resolves through its own .links accessor
            # even when the record itself is not separately declared as an
            # output -- exactly one resolution path either way.
            values = tuple(
                value
                if output.variable is not None
                else getattr(cast(Any, value).links, output.link)
                if output.link is not None
                else _scalar_value(value, generic, cast(_RemoteField, output.field))
                for output in self._outputs
            )
            yield SearchResult(values, names)

    def __iter__(self) -> Iterator[SearchResult]:
        """Yield remote search results in output order."""

        return self._search_results()

    def count(self) -> int:
        """Return the filtered remote count.

        A valid optional ``meta.data_returned`` is the fast path. If it is
        absent or ``null``, counting raises unless the store enables ID
        pagination explicitly.

        :return: Exact number of filtered remote results.
        :raises CountUnavailableError: If no valid count is reported and ID pagination is disabled.
        """
        descriptor, _variable = self._require_variable()
        if self._count_cache.value is not None:
            return self._count_cache.value
        url = self._request_url(
            page_limit=1,
            include_sort=False,
            include_related=False,
            response_fields=(self._fields["id"]._remote_name,) if "id" in self._fields else None,
        )
        raw_text = self._store._get(url)
        root = self._raw_root(raw_text, url)
        meta = root.get("meta")
        if not isinstance(meta, dict):
            raise CountUnavailableError("OPTIMADE count response has no object meta")
        value = meta.get("data_returned")
        if value is None:
            if not self._store.count_by_pagination:
                raise CountUnavailableError(
                    "OPTIMADE count response omitted meta.data_returned; pass "
                    "count_by_pagination=True to OptimadeStore to count by fetching ID pages"
                )
            data = root.get("data")
            if (
                isinstance(data, list)
                and meta.get("more_data_available") is False
                and self._next_link(root, url) is None
            ):
                self._validate_entry_page(root, url, descriptor.name)
                value = len(data)
            else:
                counter = self._clone()
                counter._sorts = []
                counter._outputs = []
                counter._limit = None
                counter.offset = 0
                id_field = counter._fields.get("id")
                if id_field is None:
                    raise CountUnavailableError("OPTIMADE count fallback requires the protocol id field")
                counter._response_transport_fields = (id_field._remote_name,)
                value = sum(1 for _value, _resource in counter._objects())
        elif not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CountUnavailableError("OPTIMADE count response has invalid nonnegative meta.data_returned")
        self._count_cache.value = value
        return value

    def results(self, **outputs: object) -> "RemoteResultSet":
        """Freeze the query as a lazy, re-iterable result set.

        :param \\*\\*outputs: Optional output names mapped to this searcher's projections.
        :return: Frozen remote result plan.
        :raises ValueError: If no outputs are declared.
        """
        return RemoteResultSet(self, outputs or None)

    def slicer(self, target: RemoteEntryType) -> "Slicer":
        """A pandas-style ``[]`` indexing view over one discovered entry endpoint.

        Each terminal indexing operation runs against a fresh searcher minted
        with this searcher's ``response_fields`` policy, so slicer operations
        never share filter state. No sorting is offered here -- use
        :meth:`~httk.store.optimade.client.OptimadeStore.searcher` and :meth:`add_sort` directly for a
        sorted or relationship query.

        :param target: The discovered remote entry endpoint to index.
        :return: A slicer over ``target``.
        """
        from httk.store.query.slicer import Slicer

        def _make() -> "RemoteSearcher":
            return self._store.searcher(response_fields=self._response_fields_setting)

        return Slicer(_make, target)


class RemoteResultColumn:
    """Expose one named scalar projection from a lazy result set.

    :param result: Result set owning the projection.
    :param index: Zero-based projection index.
    """

    def __init__(self, result: "RemoteResultSet", index: int) -> None:
        self._result = result
        self._index = index
        self.name = result.names[index]

    def __len__(self) -> int:
        """Return the number of projected results.

        :return: Result count after query paging.
        """

        return len(self._result)

    def __iter__(self) -> Iterator[object]:
        """Yield projected scalar values."""

        return (row[self._index] for row in self._result)


class RemoteResultSet:
    """Represent a frozen, lazy, and re-iterable remote result plan.

    :param searcher: Search plan to clone.
    :param outputs: Optional output names mapped to the searcher's projections.
    """

    def __init__(self, searcher: RemoteSearcher, outputs: Mapping[str, object] | None = None) -> None:
        self._plan = searcher._clone()
        if outputs:
            self._plan._outputs = []
            for name, value in outputs.items():
                if value is searcher._variable:
                    self._plan.output(self._plan._variable, name)
                elif isinstance(value, _RemoteField) and value._searcher is searcher:
                    if searcher._fields.get(value._local_name) is not value:
                        raise _unsupported("related fields as outputs or sort keys")
                    self._plan.output(self._plan._fields[value._local_name], name)
                elif isinstance(value, _RemoteLinkSet) and value._searcher is searcher:
                    if value._variable is not searcher._variable:
                        raise _unsupported("link sets from another backend or searcher")
                    self._plan.output(
                        _RemoteLinkSet(
                            self._plan, cast(_RemoteVariable, self._plan._variable), value._name, value._descriptor
                        ),
                        name,
                    )
                else:
                    raise _unsupported("result projections from another backend or searcher")
        if not self._plan._outputs:
            raise ValueError("this search has no outputs; declare outputs or pass them to results()")
        self.names = tuple(output.name for output in self._plan._outputs)

    def __getitem__(self, item: slice) -> "RemoteResultSet":
        """Return a derived result plan for a unit-step slice.

        :param item: Nonnegative slice with no step or a unit step.
        :return: Derived lazy result plan.
        :raises TypeError: If integer indexing is requested.
        :raises ValueError: If slice bounds or step are unsupported.
        """
        if not isinstance(item, slice):
            raise TypeError("remote result sets support slicing, not integer indexing")
        if item.step not in (None, 1):
            raise ValueError("remote result slices require a unit step")
        start = 0 if item.start is None else item.start
        stop = item.stop
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or start < 0
            or (stop is not None and (not isinstance(stop, int) or isinstance(stop, bool) or stop < 0))
        ):
            raise ValueError("remote result slice bounds must be nonnegative integers")
        derived = object.__new__(type(self))
        derived._plan = self._plan._clone()
        derived.names = self.names
        available_limit = derived._plan._limit
        derived._plan.offset += start
        if available_limit is not None:
            available_limit = max(0, available_limit - start)
        if stop is not None:
            span = max(0, stop - start)
            available_limit = span if available_limit is None else min(available_limit, span)
        derived._plan._limit = available_limit
        return derived

    def __iter__(self) -> Iterator[ResultRow]:
        """Yield projected rows as result rows."""

        return (ResultRow(result.values, result.names) for result in self._plan._search_results())

    def __len__(self) -> int:
        """Return the bounded result count.

        :return: Number of rows available under this result plan.
        """

        available = max(0, self._plan.count() - self._plan.offset)
        if self._plan._limit is not None:
            available = min(available, self._plan._limit)
        return available

    def first(self) -> ResultRow | None:
        """Return the first result, if present.

        :return: First row or ``None``.
        """

        iterator = self._plan._search_results(maximum=1)
        try:
            result = next(iterator)
        except StopIteration:
            return None
        return ResultRow(result.values, result.names)

    def one(self) -> ResultRow:
        """Return the only result.

        :return: Sole result row.
        :raises httk.store.NoResultError: If no result exists.
        :raises httk.store.MultipleResultsError: If more than one result exists.
        """

        iterator = self._plan._search_results(maximum=2)
        try:
            first = next(iterator)
        except StopIteration:
            raise NoResultError("expected exactly one result, found none") from None
        try:
            next(iterator)
        except StopIteration:
            return ResultRow(first.values, first.names)
        raise MultipleResultsError("expected exactly one result, found more than one")

    def scalars(self, name: str | None = None) -> Iterator[object]:
        """Iterate one named scalar output from each result.

        :param name: Output name, or ``None`` when exactly one exists.
        :return: Iterator over scalar values.
        :raises KeyError: If the named output is unknown.
        :raises ValueError: If no name is given and multiple outputs exist.
        """
        if name is None:
            if len(self.names) != 1:
                raise ValueError(f"scalars() without a name requires exactly one output; declared: {self.names}")
            name = self.names[0]
        try:
            index = self.names.index(name)
        except ValueError:
            raise KeyError(f"unknown output {name!r}; declared: {self.names}") from None
        return (row[index] for row in self)

    def column(self, name: str) -> RemoteResultColumn:
        """Return a lazy column for a scalar output.

        :param name: Scalar output name.
        :return: Lazy result column.
        :raises KeyError: If the output is unknown.
        :raises TypeError: If the output is a whole-record projection.
        """
        try:
            index = self.names.index(name)
        except ValueError:
            raise KeyError(f"unknown output {name!r}; declared: {self.names}") from None
        if self._plan._outputs[index].field is None:
            scalar_names = tuple(output.name for output in self._plan._outputs if output.field is not None)
            raise TypeError(f"column {name!r} is an object output; scalar projections: {scalar_names}")
        return RemoteResultColumn(self, index)

    def cursor(self) -> Iterator[ResultRow]:
        """Reject unsupported cursor access.

        :return: Never; remote OPTIMADE cursors are unsupported.
        :raises NotImplementedError: Remote OPTIMADE cursors are unavailable.
        """

        raise NotImplementedError("remote OPTIMADE cursor rows are not supported")
