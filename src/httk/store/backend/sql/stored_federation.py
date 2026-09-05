"""Bounded, durable federation of stored entry-family property plans.

Unlike :mod:`httk.store.federated_store`, which is the general portable query
protocol, this module joins only configured durable entry families.
It can therefore retain a stable backing inventory, push candidate filtering
and bounds into SQL, and delay record hydration until a global page is known.
"""

import heapq
import json
import warnings
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import batched
from types import MappingProxyType
from typing import Any, Final, cast

import sqlalchemy
from httk.core import ALTERNATIVE_KIND_PATTERN, RelatedEntry
from httk.core.optimade import FilterAst

import httk.store.store_common
from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql.entry_provider import _live_targets_by_source
from httk.store.backend.sql.mapping import (
    LOGICAL_ID_COLUMN,
    RETRACTED_COLUMN,
    SID_COLUMN,
    SOURCE_LID_COLUMN,
    TARGET_LID_COLUMN,
)
from httk.store.backend.sql.provenance_edges import (
    StrongLinkFamily,
    forward_run_edges,
    reverse_run_edges,
    strong_link_families,
    wire_type_for_internal,
)
from httk.store.backend.sql.store import SqlStore, _served_definition
from httk.store.backend.sql.stored_properties import (
    RelationshipSourceMap,
    StoredPropertySqlCandidateStream,
    StoredPropertySqlPlan,
    _served_type_for_target,
)
from httk.store.entry_providers import wire_relationship_key
from httk.store.query.optimade_filters import RelatedPropertyResolver
from httk.store.store_common import EntryStore

# The relationship channel keyed by (wire-translated) related entry type.
_RelatedMap = Mapping[str, tuple[RelatedEntry, ...]]
_EMPTY_RELATIONSHIPS: Final[_RelatedMap] = MappingProxyType({})

# A per-store depth-1 related-property resolver source: called with the store
# whose row is being filtered and its relationship mount context, it returns a
# resolver restricted to that store's sibling plans, or None when unavailable.
type RelatedResolverFactory = Callable[
    [httk.store.store_common.EntryStore, RelationshipSourceMap | None], RelatedPropertyResolver | None
]

__all__ = [
    "DuplicateEntryIdError",
    "RelatedResolverFactory",
    "StoredEntryFederation",
    "StoredEntryOrigin",
    "StoredEntryPage",
    "StoredEntrySource",
    "related_property_resolver_factory",
]


def related_property_resolver_factory(
    plans: Sequence[StoredPropertySqlPlan],
) -> RelatedResolverFactory:
    """Build a same-store depth-1 related-property resolver factory over sibling plans.

    ``plans`` are the family plans available to a serving edge (one per source).
    The returned factory accepts the store whose row is being filtered and an
    optional :class:`~httk.store.backend.sql.stored_properties.RelationshipSourceMap`, yielding a
    :data:`~httk.store.query.optimade_filters.RelatedPropertyResolver` that resolves a
    dotted ``<related_type>.<prop>`` filter to the matching related-entry ids by
    running the stripped sub-filter through the sibling plan for ``related_type``
    **in that same store** (same-store scope, mirroring reverse serving). Only
    the sibling's own properties are consulted; the sub-search runs
    ``only_latest=True`` over mains, so stale revisions and named alternatives can
    never satisfy the filter. With a relationship map, each matching candidate
    receives its concrete backing's target prefix before ids are combined;
    that same prefix applies to ``id`` predicates in the sibling search.
    Unrelated backings are excluded, and custom wire types map back to the
    original plan type. Calling the factory with only the store returns raw
    stored ids instead. These ids feed the ``<related_type>.id HAS ...`` handler
    used by the semi-join rewrite.

    :param plans: The family plans (one per source) available to the serving edge.
    :return: A per-store resolver factory (a miss returns an empty tuple, i.e. matches nothing).
    """
    plans_by_type: dict[str, list[StoredPropertySqlPlan]] = {}
    for plan in plans:
        plans_by_type.setdefault(plan.entry_type, []).append(plan)

    def factory(store: EntryStore, source_map: RelationshipSourceMap | None = None) -> RelatedPropertyResolver | None:
        original_types = (
            {
                wire: wire_type_for_internal(cast(SqlStore, store), internal)
                for internal, wire in source_map.wire_types.items()
            }
            if source_map is not None
            else {}
        )

        def resolve(related_type: str, sub_ast: FilterAst) -> tuple[str, ...]:
            matched: dict[str, None] = {}
            for plan in plans_by_type.get(original_types.get(related_type, related_type), ()):
                if plan.store is not store:
                    continue
                prefix = (
                    next(
                        (
                            source_map.backing_prefixes[backing]
                            for backing in plan.backings
                            if backing in source_map.backing_prefixes
                        ),
                        None,
                    )
                    if source_map is not None
                    else ""
                )
                if prefix is None:
                    continue
                for stream in plan.candidate_searchers(sub_ast, only_latest=True, public_id_prefix=prefix):
                    if source_map is not None and stream.backing not in source_map.backing_prefixes:
                        continue
                    for values, _names in stream.searcher:
                        matched.setdefault(prefix + str(values[1]))
            return tuple(matched)

        return resolve

    return factory


_AUDIT_BATCH_SIZE: Final = 1_000
# Match the hydrator's conservative bound for SQL IN parameters.
_RELATIONSHIP_BATCH_SIZE: Final = 500


@dataclass(frozen=True)
class StoredEntrySource:
    """One named configured family in one durable entry store.

    ``public_id_prefix`` is concatenated with every backing's store-minted
    lineage id.  It is intentionally not required to be unique: callers may
    retain a legacy unprefixed source, in which case collisions are detected
    when their visible ids are fetched or explicitly audited.

    :param store: The durable entry store containing the entry family.
    :param entry_family: The logical entry-family class to serve.
    :param name: The unique name used to identify this source.
    :param public_id_prefix: The prefix prepended to store-minted lineage ids.
    :param relationship_sources: Explicit target-family to same-store source-name selections for ambiguous mounts.
    """

    store: EntryStore
    entry_family: type
    name: str
    public_id_prefix: str = ""
    relationship_sources: Mapping[type, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.store, EntryStore):
            raise TypeError("StoredEntrySource.store must be an EntryStore")
        if not isinstance(self.entry_family, type):
            raise TypeError("StoredEntrySource.entry_family must be an entry-family class")
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("StoredEntrySource.name must be a non-empty stripped string")
        if not isinstance(self.public_id_prefix, str):
            raise TypeError("StoredEntrySource.public_id_prefix must be a string")
        if not isinstance(self.relationship_sources, Mapping) or not all(
            isinstance(family, type) and isinstance(name, str) and name
            for family, name in self.relationship_sources.items()
        ):
            raise TypeError("StoredEntrySource.relationship_sources must map family classes to source names")
        object.__setattr__(
            self,
            "relationship_sources",
            MappingProxyType(dict(self.relationship_sources)),
        )


@dataclass(frozen=True)
class StoredEntryOrigin:
    """The durable source of one public entry id.

    :param source: The configured source name.
    :param source_index: The source's position in the federation.
    :param backing: The concrete backing name.
    :param entry_id: The store-minted lineage id claimed by the backing.
    """

    source: str
    source_index: int
    backing: str
    entry_id: str


class DuplicateEntryIdError(RuntimeError):
    """Several durable origins claim the same public entry id.

    Call :meth:`StoredEntryFederation.audit_duplicate_ids` to perform the
    intentionally explicit complete audit; ordinary pages inspect only the
    candidates they would otherwise return.

    :param public_id: The public id claimed by multiple origins.
    :param origins: The durable origins claiming the id.
    """

    def __init__(self, public_id: str, origins: Sequence[StoredEntryOrigin]) -> None:
        self.public_id = public_id
        self.origins = tuple(origins)
        rendered = ", ".join(f"{item.source}/{item.backing}" for item in self.origins)
        super().__init__(
            f"duplicate public entry id {public_id!r} from {rendered}; "
            "call audit_duplicate_ids() to audit the complete federation"
        )


@dataclass(frozen=True)
class StoredEntryPage:
    """One immutable globally paginated response.

    ``total_count`` is the exact filtered count before global offset/limit.
    The sentinel establishing :attr:`more_data_available` is ID-only and is
    never present in :attr:`rows`.

    :param rows: The rows visible in this page.
    :param total_count: The exact filtered count before paging bounds.
    :param more_data_available: Whether another row exists after this page.
    :param relationships: The per-row exposed weak-link relationships, aligned
        with :attr:`rows` (an empty mapping for a row carrying none).
    """

    rows: tuple[Mapping[str, Any], ...]
    total_count: int
    more_data_available: bool
    relationships: tuple[Mapping[str, tuple[RelatedEntry, ...]], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            object.__setattr__(self, "rows", tuple(self.rows))
        if isinstance(self.total_count, bool) or not isinstance(self.total_count, int) or self.total_count < 0:
            raise ValueError("StoredEntryPage.total_count must be a non-negative integer")
        if not isinstance(self.more_data_available, bool):
            raise TypeError("StoredEntryPage.more_data_available must be bool")
        if not isinstance(self.relationships, tuple):
            object.__setattr__(self, "relationships", tuple(self.relationships))
        if len(self.relationships) != len(self.rows):
            raise ValueError("StoredEntryPage.relationships must be row-aligned with rows")


@dataclass(frozen=True)
class _ResolvedSource:
    source: StoredEntrySource
    source_index: int
    plan: StoredPropertySqlPlan


@dataclass(frozen=True)
class _Stream:
    source: _ResolvedSource
    backing: type
    backing_name: str
    backing_index: int
    candidate_stream: StoredPropertySqlCandidateStream


@dataclass(frozen=True)
class _Candidate:
    stream: _Stream
    sid: int
    entry_id: str
    immutable_id: str
    alt_kind: str | None
    sort_values: tuple[Any, ...]
    store_timestamp: int | None = None

    @property
    def public_id(self) -> str:
        return self.stream.source.source.public_id_prefix + self.entry_id

    @property
    def revision_public_id(self) -> str:
        """Return this immutable revision's public id."""
        return self.stream.source.source.public_id_prefix + self.immutable_id

    @property
    def alternative_public_id(self) -> str:
        """Return this alternative's composite ``<prefix><id>~<kind>`` public id."""
        return f"{self.stream.source.source.public_id_prefix}{self.entry_id}~{self.alt_kind}"

    @property
    def origin(self) -> StoredEntryOrigin:
        return StoredEntryOrigin(
            self.stream.source.source.name,
            self.stream.source.source_index,
            self.stream.backing_name,
            self.entry_id,
        )


class _Descending:
    """Heap key wrapper reversing only one comparable non-null value."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __lt__(self, other: "_Descending") -> bool:
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Descending) and self.value == other.value


class StoredEntryFederation:
    """Merge one or more configured durable entry-family sources.

    Sources preserve caller order.  Without a sort, rows remain in source,
    persisted-backing, and native database order and candidate SQL contains no
    ``ORDER BY``.  With a sort, each backing stream orders in SQL and this
    object performs a bounded heap merge with a deterministic public-id/source
    /backing tie-breaker.

    Pages probe all sibling backings in prefixes shared by multiple sources.
    Within-source corruption is otherwise audit-only; use
    :meth:`audit_duplicate_ids` to detect it.

    :param sources: The configured sources to merge in caller order.
    :param served_type_names: An optional internal-to-wire map applied to the
        entry-type names emitted on served relationships; unmapped names pass
        through unchanged.
    :param related_resolver_factory: An optional per-store factory (see
        :func:`related_property_resolver_factory`) enabling depth-1
        related-property filtering (``references.doi CONTAINS ...``); without it
        such dotted filters match nothing, while ``<type>.id HAS ...`` still works.
        Called with the store and its relationship source map (or ``None``).
    :param source_inventory: All mounted families, used to resolve relationship target prefixes; defaults to sources.
    """

    def __init__(
        self,
        sources: Sequence[StoredEntrySource],
        *,
        served_type_names: Mapping[str, str] | None = None,
        related_resolver_factory: RelatedResolverFactory | None = None,
        source_inventory: Sequence[StoredEntrySource] | None = None,
    ) -> None:
        if served_type_names is not None and not isinstance(served_type_names, Mapping):
            raise TypeError("StoredEntryFederation.served_type_names must be a mapping or None")
        if related_resolver_factory is not None and not callable(related_resolver_factory):
            raise TypeError("StoredEntryFederation.related_resolver_factory must be callable or None")
        self._served_type_names: Mapping[str, str] = dict(served_type_names or {})
        self._related_resolver_factory = related_resolver_factory
        if isinstance(sources, (str, bytes)):
            raise TypeError("StoredEntryFederation.sources must be a sequence of StoredEntrySource values")
        values = tuple(sources)
        if not values:
            raise ValueError("StoredEntryFederation requires at least one source")
        if not all(isinstance(item, StoredEntrySource) for item in values):
            raise TypeError("StoredEntryFederation.sources must contain StoredEntrySource values")
        names = tuple(item.name for item in values)
        if len(set(names)) != len(names):
            raise ValueError("StoredEntryFederation source names must be unique")
        entry_family = values[0].entry_family
        if any(item.entry_family is not entry_family for item in values[1:]):
            raise ValueError("StoredEntryFederation sources must use one exact entry_family")
        self._source_inventory = tuple(source_inventory) if source_inventory is not None else values
        if not all(isinstance(source, StoredEntrySource) for source in self._source_inventory):
            raise TypeError("StoredEntryFederation.source_inventory must contain StoredEntrySource values")
        inventory_by_name = {source.name: source for source in self._source_inventory}
        if len(inventory_by_name) != len(self._source_inventory):
            raise ValueError("StoredEntryFederation source inventory names must be unique")
        if any(inventory_by_name.get(source.name) is not source for source in values):
            raise ValueError("StoredEntryFederation source inventory must contain its sources")
        for source in self._source_inventory:
            for family, name in source.relationship_sources.items():
                target = inventory_by_name.get(name)
                if target is None or target.store is not source.store or target.entry_family is not family:
                    raise ValueError(f"Invalid relationship source {name!r} for {source.name!r}/{family.__name__}")
        if sum(item.public_id_prefix == "" for item in values) > 1:
            warnings.warn(
                "multiple StoredEntrySource values use an empty public_id_prefix; "
                "duplicate ids remain lazy until fetch(), a visible page, or audit_duplicate_ids()",
                RuntimeWarning,
                stacklevel=2,
            )
        resolved_sources = tuple(
            _ResolvedSource(
                source,
                index,
                source.store.stored_property_plan(source.entry_family),
            )
            for index, source in enumerate(values)
        )
        entry_type = resolved_sources[0].plan.entry_type
        definition_id = _definition_id(resolved_sources[0].plan)
        if any(
            item.plan.entry_type != entry_type or _definition_id(item.plan) != definition_id
            for item in resolved_sources[1:]
        ):
            raise ValueError("StoredEntryFederation sources must use equal entry type and definition")
        self._sources = resolved_sources
        self._relationship_maps = {
            source.name: self._relationship_map(source) for source in values if isinstance(source.store, SqlStore)
        }
        stream_groups: dict[str, list[tuple[int, int]]] = {}
        for resolved in resolved_sources:
            for backing_index in range(len(resolved.plan.backings)):
                stream_groups.setdefault(resolved.source.public_id_prefix, []).append(
                    (resolved.source_index, backing_index)
                )
        self._page_colliding_streams = {
            prefix: frozenset(streams)
            for prefix, streams in stream_groups.items()
            if len({source_index for source_index, _backing_index in streams}) > 1
        }
        self._audit_streams = {prefix: frozenset(streams) for prefix, streams in stream_groups.items()}

    @property
    def sources(self) -> tuple[StoredEntrySource, ...]:
        """Return the immutable declared source order.

        :return: The declared sources in caller order.
        """
        return tuple(item.source for item in self._sources)

    def snapshot_cutoff_ns(self, now_ns: int) -> int | None:
        """Return the latest instant strictly before every capable source's current bucket.

        Per-source floor conversion then selects each store's last completed
        timestamp unit, so future monotonic writes cannot enter the snapshot,
        even when sources use different timestamp resolutions. Timestamp-
        disabled sources are ignored; ``None`` means no source is capable.

        :param now_ns: Current time in nanoseconds.
        :return: A nanosecond cutoff, or ``None`` when no source stores timestamps.
        """
        completed_buckets = []
        for source in self._sources:
            store = cast(Any, source.source.store)
            if store.store_timestamps:
                resolution = store.store_timestamp_resolution
                completed_buckets.append((now_ns // resolution) * resolution)
        return None if not completed_buckets else min(completed_buckets) - 1

    @property
    def _colliding_streams(self) -> Mapping[str, frozenset[tuple[int, int]]]:
        """Cross-source stream groups requiring page-time duplicate probes.

        Page serving intentionally does not detect duplicates between record
        classes in one source: the store write path maintains dispatch
        consistency, and such duplicates require out-of-band modification.
        :meth:`audit_duplicate_ids` is the designed detector for that
        corruption class.
        """
        return self._page_colliding_streams

    def query(
        self,
        filter_string: str | FilterAst | None = None,
        *,
        sort: Sequence[tuple[str, bool]] = (),
        offset: int = 0,
        limit: int | None = None,
        as_of: object = None,
        fields: Collection[str] | None = None,
        revisions: bool = False,
        alternatives: bool = False,
    ) -> StoredEntryPage:
        """Return one globally merged page with an exact filtered total.

        ``limit=0`` intentionally runs only count plus an ID-only sentinel:
        it is suitable for metadata initialization and never duplicate-probes
        or hydrates a candidate.

        :param filter_string: The OPTIMADE filter, parsed filter tree, or no filter.
        :param sort: The property sort keys and directions.
        :param offset: The number of matching rows to skip globally.
        :param limit: The maximum number of rows to return, or no maximum.
        :param as_of: Optional historic cutoff. Sources without store timestamps
            deliberately omit this cutoff and serve their current state.
            Dependencies of a visible row remain visible because references
            only point at earlier-or-equal rows from the same transaction.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :param revisions: Whether to stream immutable revisions of mains instead of latest mains.
        :param alternatives: Whether to stream latest named alternatives with composite ``<id>~<kind>`` ids.
        :return: The globally merged page.
        :raises DuplicateEntryIdError: If a visible id has multiple cross-source origins.
        """
        _validate_page_bounds(offset, limit)
        ordered = _normalized_sort(sort)
        stream_sort = _stream_sort(ordered) if ordered else ()
        streams = self._streams(filter_string, stream_sort, as_of=as_of, revisions=revisions, alternatives=alternatives)
        counts = tuple(stream.candidate_stream.searcher.count() for stream in streams)
        total_count = sum(counts)
        if ordered:
            candidates = self._sorted_candidates(streams, ordered, offset, limit)
        else:
            candidates = self._unsorted_candidates(streams, counts, offset, limit)
        visible_count = len(candidates) if limit is None else min(limit, len(candidates))
        visible = candidates[:visible_count]
        more = False if limit is None else len(candidates) > visible_count
        # A public page may detect collisions only for ids it would expose.
        # The ID-only sentinel is deliberately excluded, especially for
        # limit=0 metadata calls.
        for candidate in visible:
            self._probe_candidate(candidate, as_of=as_of, revisions=revisions, alternatives=alternatives)
        rows = self._render_page(visible, fields, revisions=revisions, alternatives=alternatives)
        collected = self._collect_relationships(visible, alternatives=alternatives)
        relationships = tuple(collected.get(id(candidate), _EMPTY_RELATIONSHIPS) for candidate in visible)
        return StoredEntryPage(rows, total_count, more, relationships)

    def fetch(
        self,
        public_id: str,
        *,
        as_of: object = None,
        fields: Collection[str] | None = None,
        revisions: bool = False,
        alternatives: bool = False,
    ) -> tuple[Mapping[str, Any], Mapping[str, tuple[RelatedEntry, ...]]] | None:
        """Fetch one public id and detect a collision among its possible origins.

        :param public_id: The public id to fetch.
        :param as_of: Optional historic cutoff. Sources without store timestamps
            deliberately omit this cutoff and serve their current state.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :param revisions: Whether ``public_id`` addresses an immutable revision instead of a main.
        :param alternatives: Whether ``public_id`` is a composite ``<id>~<kind>`` alternative id.
        :return: The fetched ``(row, relationships)`` pair, or ``None`` when absent.
        :raises DuplicateEntryIdError: If the id has multiple origins.
        """
        if not isinstance(public_id, str):
            raise TypeError("StoredEntryFederation.fetch public_id must be a string")
        matches = self._probe_public_id(public_id, as_of=as_of, revisions=revisions, alternatives=alternatives)
        if not matches:
            return None
        return self._row_with_relationships(matches[0], fields, revisions=revisions, alternatives=alternatives)

    def fetch_revision(
        self,
        entry_id: str,
        immutable_id: str,
        *,
        as_of: object = None,
        fields: Collection[str] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, tuple[RelatedEntry, ...]]] | None:
        """Fetch one immutable revision addressed by its lineage and revision ids.

        :param entry_id: The public lineage id of the revision.
        :param immutable_id: The public immutable revision id.
        :param as_of: Optional historic cutoff. Sources without store timestamps
            deliberately omit this cutoff and serve their current state.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :return: The fetched ``(row, relationships)`` pair, or ``None`` when absent.
        :raises DuplicateEntryIdError: If the id has multiple origins.
        """
        if not isinstance(entry_id, str) or not isinstance(immutable_id, str):
            raise TypeError("StoredEntryFederation.fetch_revision ids must be strings")
        matches: list[_Candidate] = []
        for source in self._sources:
            raw_entry = _entry_id_for_public_id(entry_id, source.source.public_id_prefix)
            raw_immutable = _entry_id_for_public_id(immutable_id, source.source.public_id_prefix)
            if raw_entry is None or raw_immutable is None:
                continue
            source_as_of = as_of if getattr(source.source.store, "store_timestamps", False) else None
            for backing_index, candidate_stream in enumerate(
                source.plan.candidate_searchers(
                    "immutable_id = " + json.dumps(raw_immutable),
                    public_id_prefix=source.source.public_id_prefix,
                    as_of=source_as_of,
                    only_latest=False,
                    revisions=True,
                )
            ):
                candidate_stream.searcher.set_limit(1)
                stream = _Stream(
                    source, candidate_stream.backing, candidate_stream.backing_name, backing_index, candidate_stream
                )
                matches.extend(candidate for candidate in _candidates(stream) if candidate.entry_id == raw_entry)
        if not matches:
            return None
        if len(matches) > 1:
            raise DuplicateEntryIdError(entry_id, tuple(item.origin for item in matches))
        return self._row_with_relationships(matches[0], fields, revisions=True)

    def fetch_alternative(
        self,
        entry_id: str,
        kind: str,
        *,
        as_of: object = None,
        fields: Collection[str] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, tuple[RelatedEntry, ...]]] | None:
        """Fetch one named alternative addressed by its group entry id and kind.

        The returned alternative is its own latest revision.  A malformed kind
        misses like an absent one, matching :meth:`fetch_revision`.

        :param entry_id: The public group lineage id of the alternative.
        :param kind: The alternative kind selecting the named alternative.
        :param as_of: Optional historic cutoff. Sources without store timestamps
            deliberately omit this cutoff and serve their current state.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :return: The fetched ``(row, relationships)`` pair, or ``None`` when absent.
        :raises DuplicateEntryIdError: If the id has multiple origins.
        """
        if not isinstance(entry_id, str) or not isinstance(kind, str):
            raise TypeError("StoredEntryFederation.fetch_alternative arguments must be strings")
        if ALTERNATIVE_KIND_PATTERN.fullmatch(kind) is None:
            return None
        matches: list[_Candidate] = []
        for source in self._sources:
            raw_entry = _entry_id_for_public_id(entry_id, source.source.public_id_prefix)
            if raw_entry is None:
                continue
            source_as_of = as_of if getattr(source.source.store, "store_timestamps", False) else None
            for backing_index, candidate_stream in enumerate(
                source.plan.candidate_searchers(
                    "_httk_id = " + json.dumps(entry_id),
                    public_id_prefix=source.source.public_id_prefix,
                    as_of=source_as_of,
                    alternatives=True,
                )
            ):
                stream = _Stream(
                    source, candidate_stream.backing, candidate_stream.backing_name, backing_index, candidate_stream
                )
                matches.extend(candidate for candidate in _candidates(stream) if candidate.alt_kind == kind)
        if not matches:
            return None
        if len(matches) > 1:
            raise DuplicateEntryIdError(f"{entry_id}~{kind}", tuple(item.origin for item in matches))
        return self._row_with_relationships(matches[0], fields, alternatives=True)

    def audit_duplicate_ids(self, *, batch_size: int = _AUDIT_BATCH_SIZE) -> None:
        """Lazily scan sorted ID-only batches and raise on the first collision.

        The audit includes duplicate ids across backings within one source as
        well as duplicates across sources.

        :param batch_size: The maximum number of candidate ids read per batch.
        :return: None.
        :raises DuplicateEntryIdError: If any public id has multiple origins.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("audit_duplicate_ids batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("audit_duplicate_ids batch_size must be positive")
        stream_keys = frozenset(stream for group in self._audit_streams.values() for stream in group)
        if not stream_keys:
            return
        streams = self._streams(None, (("id", False),), stream_keys)
        iterators = [_BatchedCandidateIterator(stream, batch_size) for stream in streams]
        heap: list[tuple[tuple[Any, ...], int, _Candidate]] = []
        for index, iterator in enumerate(iterators):
            candidate = _next_or_none(iterator)
            if candidate is not None:
                heapq.heappush(heap, ((candidate.public_id,), index, candidate))
        while heap:
            _key, index, candidate = heapq.heappop(heap)
            group = [candidate]
            following = _next_or_none(iterators[index])
            if following is not None:
                heapq.heappush(heap, ((following.public_id,), index, following))
            while heap and heap[0][2].public_id == candidate.public_id:
                _same_key, same_index, same = heapq.heappop(heap)
                group.append(same)
                following = _next_or_none(iterators[same_index])
                if following is not None:
                    heapq.heappush(heap, ((following.public_id,), same_index, following))
            if len(group) > 1:
                raise DuplicateEntryIdError(candidate.public_id, tuple(item.origin for item in group))

    def _streams(
        self,
        filter_string: str | FilterAst | None,
        sort: Sequence[tuple[str, bool]],
        stream_keys: frozenset[tuple[int, int]] | None = None,
        *,
        as_of: object = None,
        revisions: bool = False,
        alternatives: bool = False,
    ) -> tuple[_Stream, ...]:
        streams: list[_Stream] = []
        for source in self._sources:
            source_as_of = as_of if getattr(source.source.store, "store_timestamps", False) else None
            # A depth-1 related-property resolver is bound to THIS source's store
            # (same-store scope). It is only passed when a factory is configured,
            # so a directly constructed (or non-SQL) federation keeps the plain
            # matches-nothing behavior and its plan signature untouched.
            resolver_kwargs: dict[str, Any] = {}
            source_map = self._relationship_maps.get(source.source.name)
            if source_map is not None:
                resolver_kwargs["relationship_source_map"] = source_map
            if self._related_resolver_factory is not None:
                resolver_kwargs["related_property_resolver"] = self._related_resolver_factory(
                    source.source.store, source_map
                )
            candidates = source.plan.candidate_searchers(
                filter_string,
                sort=sort,
                public_id_prefix=source.source.public_id_prefix,
                as_of=source_as_of,
                only_latest=not revisions,
                revisions=revisions,
                alternatives=alternatives,
                **resolver_kwargs,
            )
            for backing_index, candidate in enumerate(candidates):
                if stream_keys is not None and (source.source_index, backing_index) not in stream_keys:
                    continue
                streams.append(_Stream(source, candidate.backing, candidate.backing_name, backing_index, candidate))
        return tuple(streams)

    def _unsorted_candidates(
        self,
        streams: Sequence[_Stream],
        counts: Sequence[int],
        offset: int,
        limit: int | None,
    ) -> list[_Candidate]:
        remaining = None if limit is None else limit + 1
        skipped = offset
        result: list[_Candidate] = []
        for stream, count in zip(streams, counts, strict=True):
            if skipped >= count:
                skipped -= count
                continue
            take = count - skipped if remaining is None else min(count - skipped, remaining)
            if take <= 0:
                break
            searcher = stream.candidate_stream.searcher
            if skipped:
                searcher.add_offset(skipped)
            searcher.set_limit(take)
            fetched = tuple(_candidates(stream))
            result.extend(fetched)
            skipped = 0
            if remaining is not None:
                remaining -= len(fetched)
                if remaining == 0:
                    break
        return result

    def _sorted_candidates(
        self,
        streams: Sequence[_Stream],
        sort: Sequence[tuple[str, bool]],
        offset: int,
        limit: int | None,
    ) -> list[_Candidate]:
        needed = None if limit is None else offset + limit + 1
        iterators: list[Iterator[_Candidate]] = []
        heap: list[tuple[tuple[Any, ...], int, _Candidate]] = []
        for index, stream in enumerate(streams):
            if needed is not None:
                stream.candidate_stream.searcher.set_limit(needed)
            iterator = iter(_candidates(stream))
            iterators.append(iterator)
            candidate = _next_or_none(iterator)
            if candidate is not None:
                heapq.heappush(heap, (_sort_key(candidate, sort), index, candidate))
        result: list[_Candidate] = []
        target = None if limit is None else offset + limit + 1
        while heap and (target is None or len(result) < target):
            _key, index, candidate = heapq.heappop(heap)
            result.append(candidate)
            following = _next_or_none(iterators[index])
            if following is not None:
                heapq.heappush(heap, (_sort_key(following, sort), index, following))
        return result[offset:]

    def _probe_candidate(
        self, candidate: _Candidate, *, as_of: object = None, revisions: bool = False, alternatives: bool = False
    ) -> None:
        colliding = self._colliding_streams.get(candidate.stream.source.source.public_id_prefix)
        if colliding is None:
            return
        matches = [candidate]
        public_id = (
            candidate.alternative_public_id
            if alternatives
            else (candidate.revision_public_id if revisions else candidate.public_id)
        )
        filter_string = "id = " + json.dumps(public_id)
        candidate_key = (candidate.stream.source.source_index, candidate.stream.backing_index)
        for stream in self._streams(
            filter_string, (), colliding - {candidate_key}, as_of=as_of, revisions=revisions, alternatives=alternatives
        ):
            stream.candidate_stream.searcher.set_limit(1)
            matches.extend(_candidates(stream))
        if len(matches) > 1:
            raise DuplicateEntryIdError(public_id, tuple(item.origin for item in matches))

    def _probe_public_id(
        self, public_id: str, *, as_of: object = None, revisions: bool = False, alternatives: bool = False
    ) -> tuple[_Candidate, ...]:
        stream_keys = frozenset(
            (source.source_index, backing_index)
            for source in self._sources
            if _entry_id_for_public_id(public_id, source.source.public_id_prefix) is not None
            for backing_index in range(len(source.plan.backings))
        )
        if not stream_keys:
            return ()
        filter_string = "id = " + json.dumps(public_id)
        matches: list[_Candidate] = []
        for stream in self._streams(
            filter_string, (), stream_keys, as_of=as_of, revisions=revisions, alternatives=alternatives
        ):
            stream.candidate_stream.searcher.set_limit(1)
            matches.extend(_candidates(stream))
        if len(matches) > 1:
            raise DuplicateEntryIdError(public_id, tuple(item.origin for item in matches))
        return tuple(matches)

    @staticmethod
    def _render_page(
        visible: Sequence[_Candidate], fields: Collection[str] | None, *, revisions: bool, alternatives: bool
    ) -> tuple[Mapping[str, Any], ...]:
        """Render visible candidates, batching the record fetch per source backing.

        Candidates are grouped by their originating ``_Stream`` (one per
        ``(source, backing)`` per ``_streams()`` call, and every visible
        candidate comes from one such call) so each distinct backing table is
        hydrated in one batched ``fetch_many`` call instead of one ``fetch`` per
        row.  Grouping is by object identity; streams need not be hashable by
        value.  Rows are then rendered in the original ``visible`` order.

        :param visible: The page candidates in final output order.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :param revisions: Whether ids render immutable revisions instead of mains.
        :param alternatives: Whether ids render composite ``<id>~<kind>`` alternatives.
        :return: The rendered response rows in ``visible`` order.
        """
        groups: dict[int, list[_Candidate]] = {}
        for candidate in visible:
            groups.setdefault(id(candidate.stream), []).append(candidate)
        record_by_candidate: dict[int, object] = {}
        # A full render (fields is None) touches every configured property, so
        # eager hydration is kept to preserve the shipped batched profile. A
        # field subset renders lazily: rows.py chunk-batches child tables, so
        # untouched child tables are simply never SELECTed.
        eager = fields is None
        for group in groups.values():
            store = group[0].stream.source.source.store
            backing = group[0].stream.backing
            records: list[object] = store.fetch_many(backing, [candidate.sid for candidate in group], eager=eager)
            for candidate, record in zip(group, records, strict=True):
                record_by_candidate[id(candidate)] = record
        return tuple(
            StoredEntryFederation._render(
                candidate, record_by_candidate[id(candidate)], fields, revisions=revisions, alternatives=alternatives
            )
            for candidate in visible
        )

    @staticmethod
    def _render(
        candidate: _Candidate,
        record: object,
        fields: Collection[str] | None,
        *,
        revisions: bool,
        alternatives: bool,
    ) -> Mapping[str, Any]:
        """Render one already-fetched record into its public response row.

        :param candidate: The visible candidate to render.
        :param record: The hydrated backing record for ``candidate``.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :param revisions: Whether ids render immutable revisions instead of mains.
        :param alternatives: Whether ids render composite ``<id>~<kind>`` alternatives.
        :return: The immutable response row.
        """
        source = candidate.stream.source
        if alternatives:
            public_id = candidate.alternative_public_id
        elif revisions:
            public_id = candidate.revision_public_id
        else:
            public_id = candidate.public_id
        row = source.plan.response_row(
            candidate.stream.backing,
            record,
            public_id=public_id,
            httk_id=candidate.public_id,
            kind=candidate.alt_kind if alternatives else None,
            store_timestamp=candidate.store_timestamp,
            fields=fields,
            revisions=revisions,
        )
        return MappingProxyType(dict(row))

    @staticmethod
    def _row(
        candidate: _Candidate, fields: Collection[str] | None, *, revisions: bool = False, alternatives: bool = False
    ) -> Mapping[str, Any]:
        source = candidate.stream.source
        eager = fields is None
        record: object = source.source.store.fetch(candidate.stream.backing, candidate.sid, eager=eager)
        return StoredEntryFederation._render(candidate, record, fields, revisions=revisions, alternatives=alternatives)

    def _row_with_relationships(
        self,
        candidate: _Candidate,
        fields: Collection[str] | None,
        *,
        revisions: bool = False,
        alternatives: bool = False,
    ) -> tuple[Mapping[str, Any], _RelatedMap]:
        row = self._row(candidate, fields, revisions=revisions, alternatives=alternatives)
        collected = self._collect_relationships((candidate,), alternatives=alternatives)
        return row, collected.get(id(candidate), _EMPTY_RELATIONSHIPS)

    def _relationship_target(self, source: StoredEntrySource, family: type) -> StoredEntrySource | None:
        """Select the actual target mount using explicit, self, unique, then same-prefix resolution."""
        candidates = [
            item for item in self._source_inventory if item.store is source.store and item.entry_family is family
        ]
        override = source.relationship_sources.get(family)
        if override is not None:
            return next(item for item in candidates if item.name == override)
        if family is source.entry_family:
            return source
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None
        same_prefix = [item for item in candidates if item.public_id_prefix == source.public_id_prefix]
        if len(same_prefix) == 1:
            return same_prefix[0]
        raise ValueError(
            f"Ambiguous relationship target {family.__name__} from source {source.name!r}; "
            "set relationship_sources for that family"
        )

    def _relationship_map(self, source: StoredEntrySource) -> RelationshipSourceMap:
        """Resolve one source's forward and reverse prefixes from the same mount selections."""
        assert isinstance(source.store, SqlStore)
        prefixes: dict[str, str] = {}
        target_backings: dict[str, tuple[type, ...]] = {}
        reverse: dict[type, tuple[str, ...]] = {}
        backing_prefixes: dict[type, str] = {}
        wire_types: dict[str, str] = {}
        strong = strong_link_families(source.store)
        strong_backings = {family.backing for family in strong}
        own_layout = next(layout for layout in source.store.layout.families if layout.family is source.entry_family)
        targets = {
            spec.target
            for backing in own_layout.records
            for spec in resolve_schema(backing).fields
            if spec.role in ("reference", "child") and (spec.related is None or spec.related.serve)
        } | {
            link.target
            for backing in own_layout.records
            for link in resolve_schema(backing).links
            if link.exposed_relationship
        }
        loose_edges = any(family.backing in own_layout.records for family in strong)
        for layout in source.store.layout.families:
            internal = getattr(layout.family, "type", None)
            if not isinstance(internal, str):
                continue
            if any(backing in targets for backing in layout.records):
                target = self._relationship_target(source, layout.family)
                prefix = target.public_id_prefix if target is not None else ""
                for backing in layout.records:
                    backing_prefixes[backing] = prefix
            if loose_edges:
                loose_targets = self._loose_relationship_targets(source, internal)
                prefixes[internal] = loose_targets[0].public_id_prefix if loose_targets else ""
                target_backings[internal] = tuple(
                    backing
                    for target in loose_targets
                    for target_layout in source.store.layout.families
                    if target_layout.family is target.entry_family
                    for backing in target_layout.records
                )
            wire_types[internal] = self._served_type_names.get(internal, wire_type_for_internal(source.store, internal))
            if not any(backing in strong_backings for backing in layout.records):
                continue
            mounts = [
                item
                for item in self._source_inventory
                if item.store is source.store and item.entry_family is layout.family
            ]
            selected_prefixes = (
                tuple(
                    item.public_id_prefix
                    for item in mounts
                    if any(
                        target is source
                        for target in self._loose_relationship_targets(item, getattr(source.entry_family, "type", ""))
                    )
                )
                if mounts
                else ("",)
            )
            for backing in layout.records:
                if backing in strong_backings:
                    reverse[backing] = selected_prefixes
        return RelationshipSourceMap(prefixes, reverse, wire_types, backing_prefixes, target_backings)

    def _loose_relationship_targets(self, source: StoredEntrySource, internal: str) -> tuple[StoredEntrySource, ...]:
        """Resolve a loose edge's type only when mounted families agree on its prefix."""
        families = dict.fromkeys(
            item.entry_family
            for item in self._source_inventory
            if item.store is source.store and getattr(item.entry_family, "type", None) == internal
        )
        targets = tuple(self._relationship_target(source, family) for family in families)
        selected = tuple(target for target in targets if target is not None)
        if len({target.public_id_prefix for target in selected}) > 1:
            raise ValueError(
                f"Ambiguous relationship target type {internal!r} from source {source.name!r}: "
                "loose edges cannot distinguish entry families with different public id prefixes"
            )
        return selected

    def _collect_relationships(
        self, candidates: Sequence[_Candidate], *, alternatives: bool = False
    ) -> dict[int, _RelatedMap]:
        """Collect exposed weak-link and StrongLink relationships for a page's candidates.

        Candidates are grouped per ``(source, backing)`` by ``_Stream`` identity
        exactly like :meth:`_render_page`, so each backing's link and edge tables
        are scanned once per group.  Four independent collections run per group,
        each issuing no query when it does not apply:

        - **Weak links**: exposed ``WeakLink`` specs on the backing, mirroring the
          in-memory provider path (links bind lineages, retracted links excluded,
          each target at its lineage's latest revision).
        - **Forward StrongLink edges**: a run backing's own provenance edges under
          their forward wire key (e.g. ``_httk_has_input``).
        - **Reverse StrongLink edges**: the runs that point at each candidate,
          derived by scanning THIS candidate's own source store's StrongLink
          families (store-scoped, never global), lineage-level (only a run
          lineage's latest main revision contributes). Reverse edges are
          suppressed on an alternatives page: an alternative cell must not claim a
          reverse relationship.
        - **Related reference/child fields**: the backing's own reference and
          child-of-storable fields whose target is a served family (mirroring
          the in-memory provider's ``_relationship_specs``, honoring
          ``Related(serve=False)``). These are record content: the row's own FK
          values, so they are revision-pinned (each ``~revs`` row carries its own)
          and — like forward StrongLink edges — appear on ``~alts`` rows too. Each
          entry carries ``relationship=None`` so it groups under the target's
          served wire type.

        Non-SQL sources are skipped (their relationship serving is a separate
        backend concern).  Relationships reflect the LIVE link/edge state
        regardless of a page's ``as_of``, so a historic page pairs its rows with
        the current state.

        :param candidates: The candidates whose relationships are collected.
        :param alternatives: Whether this is an alternatives page (reverse edges suppressed).
        :return: Row relationships keyed by ``id(candidate)``; absent for a candidate carrying none.
        """
        groups: dict[int, list[_Candidate]] = {}
        for candidate in candidates:
            groups.setdefault(id(candidate.stream), []).append(candidate)
        collected: dict[int, dict[str, list[RelatedEntry]]] = {}
        for group in groups.values():
            store = group[0].stream.source.source.store
            backing = group[0].stream.backing
            if not isinstance(store, SqlStore):
                continue
            link_specs = [spec for spec in resolve_schema(backing).links if spec.exposed_relationship]
            if link_specs:
                self._collect_group_relationships(store, backing, link_specs, group, collected)
            relation_specs = [
                (spec, related_type)
                for spec in resolve_schema(backing).fields
                if spec.role in ("reference", "child")
                and spec.target is not None
                and (spec.related is None or spec.related.serve)
                and (related_type := _served_type_for_target(store, spec.target)) is not None
            ]
            if relation_specs and not store._missing_tables_for_read((backing,)):
                with store._read_connection() as connection:
                    for spec, related_type in relation_specs:
                        self._collect_related_field(connection, store, backing, spec, related_type, group, collected)
            strong = strong_link_families(store)
            forward_family = next((family for family in strong if family.backing is backing), None)
            if forward_family is not None:
                self._collect_forward_edges(store, forward_family, group, collected)
            if not alternatives and strong:
                internal_target = getattr(group[0].stream.source.source.entry_family, "type", None)
                if isinstance(internal_target, str):
                    self._collect_reverse_edges(store, internal_target, strong, group, collected)
        return {
            candidate_id: MappingProxyType(
                {related: tuple(dict.fromkeys(entries)) for related, entries in mapping.items()}
            )
            for candidate_id, mapping in collected.items()
        }

    def _collect_group_relationships(
        self,
        store: SqlStore,
        backing: type,
        link_specs: Sequence[Any],
        group: Sequence[_Candidate],
        collected: dict[int, dict[str, list[RelatedEntry]]],
    ) -> None:
        """Scan one backing's exposed link tables for a group of same-backing candidates.

        :param store: The SQL store backing this group.
        :param backing: The concrete backing class of the group.
        :param link_specs: The backing's exposed weak-link specs.
        :param group: The candidates sharing this ``(source, backing)`` stream.
        :param collected: The mutable relationship accumulator keyed by candidate id.
        :return: None.
        """
        if store._missing_tables_for_read((backing,)):
            return
        table = store._table(resolve_schema(backing).table_name)
        with store._read_connection() as connection:
            # Each candidate's source-lineage logical_id keys the live link scan.
            lid_by_sid: dict[int, int] = {
                int(sid): int(lid)
                for sid, lid in connection.execute(
                    sqlalchemy.select(table.c[SID_COLUMN], table.c[LOGICAL_ID_COLUMN]).where(
                        table.c[SID_COLUMN].in_([candidate.sid for candidate in group])
                    )
                )
            }
            for link_spec in link_specs:
                self._collect_link(connection, store, link_spec, group, lid_by_sid, collected)

    def _collect_forward_edges(
        self,
        store: SqlStore,
        family: StrongLinkFamily,
        group: Sequence[_Candidate],
        collected: dict[int, dict[str, list[RelatedEntry]]],
    ) -> None:
        """Attach a run backing's own forward provenance edges to its candidates.

        :param store: The SQL store backing this group.
        :param family: The group backing's StrongLink family.
        :param group: The candidates sharing this ``(source, backing)`` stream.
        :param collected: The mutable relationship accumulator keyed by candidate id.
        :return: None.
        """
        if store._missing_tables_for_read((family.backing,)):
            return
        source_map = self._relationship_maps[group[0].stream.source.source.name]
        with store._read_connection() as connection:
            edges_by_sid = forward_run_edges(
                connection,
                store,
                family,
                [candidate.sid for candidate in group],
                target_prefixes=source_map.prefixes,
                target_backings=source_map.target_backings,
            )
        for candidate in group:
            source_map = self._relationship_maps[candidate.stream.source.source.name]
            for (edge_type, edge_id, label), marker in edges_by_sid.get(int(candidate.sid), []):
                key = wire_relationship_key(marker.relationship, family.definition_id)
                collected.setdefault(id(candidate), {}).setdefault(key, []).append(
                    RelatedEntry(
                        source_map.wire_types.get(edge_type, edge_type),
                        edge_id,
                        role=marker.role,
                        label=label,
                        relationship=key,
                    )
                )

    def _collect_reverse_edges(
        self,
        store: SqlStore,
        internal_target: str,
        families: Sequence[StrongLinkFamily],
        group: Sequence[_Candidate],
        collected: dict[int, dict[str, list[RelatedEntry]]],
    ) -> None:
        """Attach the reverse provenance edges naming runs that point at each candidate.

        Reverse matching is against the candidate's raw stored id (F: custom
        ``id_of`` remapping is not attempted).  Only this store's StrongLink
        families are scanned, so a target served from a store without the run
        family gets no reverse edges.

        :param store: The candidate's own source store.
        :param internal_target: The candidate family's internal (unprefixed) type name.
        :param families: The store's StrongLink families whose reverse edges are derived.
        :param group: The candidates sharing this ``(source, backing)`` stream.
        :param collected: The mutable relationship accumulator keyed by candidate id.
        :return: None.
        """
        # A revisions page shares one raw entry_id across every revision of a
        # lineage, so a raw id maps to a LIST of candidates; each reverse hit
        # must attach to all of them (not just the last in group order).
        candidates_by_raw: dict[str, list[_Candidate]] = {}
        for candidate in group:
            candidates_by_raw.setdefault(candidate.entry_id, []).append(candidate)
        target_ids = list(candidates_by_raw)
        with store._read_connection() as connection:
            for family in families:
                if store._missing_tables_for_read((family.backing,)):
                    continue
                reverse = reverse_run_edges(connection, store, family, internal_target, target_ids)
                for raw_id, hits in reverse.items():
                    for run_id, label, marker in hits:
                        if marker.reverse is None:
                            continue
                        key = wire_relationship_key(marker.reverse, family.definition_id)
                        for candidate in candidates_by_raw.get(raw_id, ()):
                            source_map = self._relationship_maps[candidate.stream.source.source.name]
                            for prefix in source_map.reverse_prefixes.get(family.backing, ()):
                                collected.setdefault(id(candidate), {}).setdefault(key, []).append(
                                    RelatedEntry(
                                        source_map.wire_types.get(family.internal_type, family.wire_type),
                                        prefix + run_id,
                                        role=marker.role,
                                        label=label,
                                        relationship=key,
                                    )
                                )

    def _collect_related_field(
        self,
        connection: Any,
        store: SqlStore,
        backing: type,
        spec: Any,
        related_type: str,
        group: Sequence[_Candidate],
        collected: dict[int, dict[str, list[RelatedEntry]]],
    ) -> None:
        """Attach one reference or child field's targets to the group's candidates.

        A reference field contributes at most one target per row (its FK); a
        child-of-storable field contributes an ordered list (its element sids).
        Both are the candidate row's own content — the FK value of the exact
        revision/alternative row — so no lineage/latest resolution is applied to
        the OWNING row; only the TARGET sid is resolved to its raw stored ``id``.
        ``role``/``description`` are carried from the ``Related`` marker (absent
        when the field is unmarked); ``relationship`` is left ``None`` so the
        entry groups under ``related_type``.

        :param connection: The open read connection to ``store``.
        :param store: The SQL store backing this group.
        :param backing: The concrete backing class of the group.
        :param spec: The reference or child :class:`~httk.store.backend.schema.FieldSpec`.
        :param related_type: The target's served (wire) relationship type.
        :param group: The candidates sharing this ``(source, backing)`` stream.
        :param collected: The mutable relationship accumulator keyed by candidate id.
        :return: None.
        """
        assert spec.target is not None
        schema = resolve_schema(backing)
        table = store._table(schema.table_name)
        sids = [candidate.sid for candidate in group]
        marker = spec.related
        description = marker.description if marker is not None else None
        role = marker.role if marker is not None else None
        # ponytail: one query per (backing, field) per group, plus one id lookup;
        # batch across fields sharing a target table if these show up in profiles.
        target_sids_by_sid: dict[int, list[int]] = {}
        if spec.role == "reference":
            fk_column = table.c[spec.columns[0].name]
            for row_sid, value in connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN], fk_column).where(table.c[SID_COLUMN].in_(sids))
            ):
                if value is not None:
                    target_sids_by_sid.setdefault(int(row_sid), []).append(int(value))
        else:
            assert spec.child is not None
            child_table = store._table(spec.child.table_name)
            parent_column = child_table.c[f"{schema.table_name}_sid"]
            element_column = child_table.c[spec.child.element_columns[0].name]
            statement = (
                sqlalchemy.select(parent_column, element_column)
                .where(parent_column.in_(sids))
                .order_by(parent_column, child_table.c[f"{spec.field}_index"])
            )
            for parent_sid, element_sid in connection.execute(statement):
                target_sids_by_sid.setdefault(int(parent_sid), []).append(int(element_sid))
        if not target_sids_by_sid:
            return
        id_by_sid = self._raw_ids_for_sids(
            connection,
            store,
            spec.target,
            {sid for sids_ in target_sids_by_sid.values() for sid in sids_},
        )
        for candidate in group:
            source_map = self._relationship_maps[candidate.stream.source.source.name]
            internal = store._entry_record_types[spec.target][0]
            wire_type = source_map.wire_types.get(internal, related_type)
            for target_sid in target_sids_by_sid.get(int(candidate.sid), ()):
                target_id = id_by_sid.get(target_sid)
                if target_id is None:
                    continue
                collected.setdefault(id(candidate), {}).setdefault(wire_type, []).append(
                    RelatedEntry(
                        wire_type,
                        source_map.backing_prefixes.get(spec.target, "") + target_id,
                        description=description,
                        role=role,
                    )
                )

    @staticmethod
    def _raw_ids_for_sids(connection: Any, store: SqlStore, target: type, sids: Collection[int]) -> dict[int, str]:
        """Resolve target row sids to their raw stored ``id`` column (revision-pinned)."""
        if not sids:
            return {}
        target_table = store._table(resolve_schema(target).table_name)
        return {
            int(sid): str(value)
            for sid, value in connection.execute(
                sqlalchemy.select(target_table.c[SID_COLUMN], target_table.c["id"]).where(
                    target_table.c[SID_COLUMN].in_(sorted(sids))
                )
            )
        }

    def _collect_link(
        self,
        connection: Any,
        store: SqlStore,
        link_spec: Any,
        group: Sequence[_Candidate],
        lid_by_sid: Mapping[int, int],
        collected: dict[int, dict[str, list[RelatedEntry]]],
    ) -> None:
        """Attach one exposed weak link's live relationships to the group's candidates.

        The related entry type is ``served_type_names[internal type]`` when
        mapped, else the target family's served (wire) name, else the internal
        type as a last resort. The target mount's prefix is applied to its latest
        lineage id; unmounted targets retain raw ids and dangling targets are skipped.

        :param connection: The open read connection to ``store``.
        :param store: The SQL store backing this group.
        :param link_spec: The exposed weak-link spec being scanned.
        :param group: The candidates sharing this ``(source, backing)`` stream.
        :param lid_by_sid: The candidate sid to source-lineage logical_id map.
        :param collected: The mutable relationship accumulator keyed by candidate id.
        :return: None.
        """
        internal = store._entry_record_types.get(link_spec.target)
        if internal is None:
            return
        related_type = self._served_type_names.get(internal[0])
        if related_type is None:
            # No explicit mapping: fall back to the target family's served (wire)
            # name so an unmapped target still keys the block in the one wire
            # vocabulary, never the internal one (e.g. "_httk_runs", not "runs").
            family_layout = store._family_for_backing(link_spec.target)
            served = _served_definition(family_layout.family) if family_layout is not None else None
            related_type = served.name if served is not None else internal[0]
        link_table = store._table(link_spec.table_name)
        source_lids = sorted(set(lid_by_sid.values()))
        targets_by_source = _live_targets_by_source(
            row
            for batch in batched(source_lids, _RELATIONSHIP_BATCH_SIZE)
            for row in connection.execute(
                sqlalchemy.select(
                    link_table.c[SOURCE_LID_COLUMN],
                    link_table.c[TARGET_LID_COLUMN],
                    link_table.c[LOGICAL_ID_COLUMN],
                    link_table.c[SID_COLUMN],
                    link_table.c[RETRACTED_COLUMN],
                ).where(link_table.c[SOURCE_LID_COLUMN].in_(batch))
            )
        )
        if not targets_by_source:
            return
        id_by_lid = self._target_ids(connection, store, link_spec.target, targets_by_source)
        for candidate in group:
            source_map = self._relationship_maps[candidate.stream.source.source.name]
            source_lid = lid_by_sid.get(int(candidate.sid))
            if source_lid is None:
                continue
            for target_lid in targets_by_source.get(source_lid, ()):
                target_id = id_by_lid.get(target_lid)
                if target_id is None:
                    continue
                collected.setdefault(id(candidate), {}).setdefault(related_type, []).append(
                    RelatedEntry(
                        related_type,
                        source_map.backing_prefixes.get(link_spec.target, "") + target_id,
                        description=link_spec.description,
                        role=link_spec.role,
                        label=link_spec.name,
                    )
                )

    @staticmethod
    def _target_ids(
        connection: Any,
        store: SqlStore,
        target: type,
        targets_by_source: Mapping[int, Sequence[int]],
    ) -> dict[int, str]:
        """Resolve each linked target lineage to its latest revision's stored id.

        :param connection: The open read connection to ``store``.
        :param store: The SQL store backing the targets.
        :param target: The target storable class.
        :param targets_by_source: The live target lineage ids keyed by source lineage.
        :return: The raw stored ``id`` column keyed by target lineage id (dangling lineages omitted).
        """
        target_lids = sorted({lid for lids in targets_by_source.values() for lid in lids})
        target_table = store._table(resolve_schema(target).table_name)
        max_sid_by_lid: dict[int, int] = {
            int(lid): int(max_sid)
            for batch in batched(target_lids, _RELATIONSHIP_BATCH_SIZE)
            for lid, max_sid in connection.execute(
                sqlalchemy.select(
                    target_table.c[LOGICAL_ID_COLUMN],
                    sqlalchemy.func.max(target_table.c[SID_COLUMN]),
                )
                .where(target_table.c[LOGICAL_ID_COLUMN].in_(batch))
                .group_by(target_table.c[LOGICAL_ID_COLUMN])
            )
            if max_sid is not None  # a None max is a dangling link; fsck reports it
        }
        id_by_sid: dict[int, str] = {
            int(sid): str(value)
            for batch in batched(max_sid_by_lid.values(), _RELATIONSHIP_BATCH_SIZE)
            for sid, value in connection.execute(
                sqlalchemy.select(target_table.c[SID_COLUMN], target_table.c["id"]).where(
                    target_table.c[SID_COLUMN].in_(batch)
                )
            )
        }
        return {lid: id_by_sid[sid] for lid, sid in max_sid_by_lid.items() if sid in id_by_sid}


class _BatchedCandidateIterator:
    """One sorted ID-only stream fetched in bounded SQL batches."""

    def __init__(self, stream: _Stream, batch_size: int) -> None:
        self._stream = stream
        self._batch_size = batch_size
        self._last_public_id: str | None = None
        self._rows: Iterator[_Candidate] = iter(())
        self._done = False

    def __iter__(self) -> "_BatchedCandidateIterator":
        return self

    def __next__(self) -> _Candidate:
        while True:
            candidate = _next_or_none(self._rows)
            if candidate is not None:
                return candidate
            if self._done:
                raise StopIteration
            filter_string = None if self._last_public_id is None else "id > " + json.dumps(self._last_public_id)
            fresh = self._stream.source.plan.candidate_searchers(
                filter_string,
                sort=(("id", False),),
                public_id_prefix=self._stream.source.source.public_id_prefix,
                only_latest=True,
            )[self._stream.backing_index]
            fresh.searcher.set_limit(self._batch_size)
            batch_stream = _Stream(
                self._stream.source,
                fresh.backing,
                fresh.backing_name,
                self._stream.backing_index,
                fresh,
            )
            values = tuple(_candidates(batch_stream))
            if values:
                self._last_public_id = values[-1].public_id
            self._done = len(values) < self._batch_size
            self._rows = iter(values)


def _candidates(stream: _Stream) -> Iterator[_Candidate]:
    for values, _names in stream.candidate_stream.searcher:
        expected_width = 4 + stream.candidate_stream.sort_count + int(stream.candidate_stream.timestamp_output)
        if len(values) != expected_width:
            raise RuntimeError(
                f"candidate stream {stream.source.source.name}/{stream.backing_name} returned "
                f"{len(values)} values; expected {expected_width}"
            )
        sort_end = 4 + stream.candidate_stream.sort_count
        timestamp = values[sort_end] if stream.candidate_stream.timestamp_output else None
        alt_kind = None if values[3] is None else str(values[3])
        yield _Candidate(
            stream, int(values[0]), str(values[1]), str(values[2]), alt_kind, tuple(values[4:sort_end]), timestamp
        )


def _next_or_none(iterator: Iterator[_Candidate]) -> _Candidate | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _entry_id_for_public_id(public_id: str, prefix: str) -> str | None:
    """Return an entry-id suffix when ``public_id`` has a non-empty suffix after ``prefix``."""
    if not public_id.startswith(prefix):
        return None
    value = public_id[len(prefix) :]
    return value or None


def _definition_id(plan: StoredPropertySqlPlan) -> str | None:
    """The concrete definition id, including inherited standard definitions."""
    return plan.definition.definition_id or plan.definition.extends_id


def _validate_page_bounds(offset: int, limit: int | None) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("StoredEntryFederation.query offset must be an integer")
    if offset < 0:
        raise ValueError("StoredEntryFederation.query offset must be non-negative")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
        raise TypeError("StoredEntryFederation.query limit must be an integer or None")
    if limit is not None and limit < 0:
        raise ValueError("StoredEntryFederation.query limit must be non-negative or None")


def _normalized_sort(sort: Sequence[tuple[str, bool]]) -> tuple[tuple[str, bool], ...]:
    if isinstance(sort, (str, bytes)):
        raise TypeError("StoredEntryFederation.query sort must be a sequence of (property, descending) pairs")
    result: list[tuple[str, bool]] = []
    for item in sort:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("StoredEntryFederation.query sort entries must be (property, descending) pairs")
        name, descending = item
        if not isinstance(name, str) or not name:
            raise ValueError("StoredEntryFederation.query sort property names must be non-empty strings")
        if not isinstance(descending, bool):
            raise TypeError("StoredEntryFederation.query sort descending flags must be bool")
        if name in {existing for existing, _direction in result}:
            raise ValueError(f"StoredEntryFederation.query sort repeats {name!r}")
        result.append((name, descending))
    return tuple(result)


def _stream_sort(sort: Sequence[tuple[str, bool]]) -> tuple[tuple[str, bool], ...]:
    """Append the public-id stream tie-break only when a user key lacks it."""
    return tuple(sort) if any(name == "id" for name, _descending in sort) else (*sort, ("id", False))


def _sort_key(candidate: _Candidate, sort: Sequence[tuple[str, bool]]) -> tuple[Any, ...]:
    values: list[Any] = []
    for value, (_name, descending) in zip(candidate.sort_values[: len(sort)], sort, strict=True):
        values.append((1, None) if value is None else (0, _Descending(value) if descending else value))
    return (*values, candidate.public_id, candidate.stream.source.source_index, candidate.stream.backing_name)
