"""Bounded, durable federation of stored entry-family property plans.

Unlike :mod:`httk.store.federated_store`, which is the general portable query
protocol, this module joins only configured durable entry families.
It can therefore retain a stable backing inventory, push candidate filtering
and bounds into SQL, and delay record hydration until a global page is known.
"""

import heapq
import json
import warnings
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

import sqlalchemy
from httk.core import ALTERNATIVE_KIND_PATTERN, RelatedEntry
from httk.core.optimade import FilterAst

from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql.entry_provider import _live_targets_by_source
from httk.store.backend.sql.mapping import (
    LOGICAL_ID_COLUMN,
    RETRACTED_COLUMN,
    SID_COLUMN,
    SOURCE_LID_COLUMN,
    TARGET_LID_COLUMN,
)
from httk.store.backend.sql.store import SqlStore, _served_definition
from httk.store.backend.sql.stored_properties import (
    StoredPropertySqlCandidateStream,
    StoredPropertySqlPlan,
)
from httk.store.store_common import EntryStore

# The relationship channel keyed by (wire-translated) related entry type.
_RelatedMap = Mapping[str, tuple[RelatedEntry, ...]]
_EMPTY_RELATIONSHIPS: Final[_RelatedMap] = MappingProxyType({})

__all__ = [
    "DuplicateEntryIdError",
    "StoredEntryFederation",
    "StoredEntryOrigin",
    "StoredEntryPage",
    "StoredEntrySource",
]


_AUDIT_BATCH_SIZE: Final = 1_000


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
    """

    store: EntryStore
    entry_family: type
    name: str
    public_id_prefix: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.store, EntryStore):
            raise TypeError("StoredEntrySource.store must be an EntryStore")
        if not isinstance(self.entry_family, type):
            raise TypeError("StoredEntrySource.entry_family must be an entry-family class")
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("StoredEntrySource.name must be a non-empty stripped string")
        if not isinstance(self.public_id_prefix, str):
            raise TypeError("StoredEntrySource.public_id_prefix must be a string")


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
    """

    def __init__(
        self,
        sources: Sequence[StoredEntrySource],
        *,
        served_type_names: Mapping[str, str] | None = None,
    ) -> None:
        if served_type_names is not None and not isinstance(served_type_names, Mapping):
            raise TypeError("StoredEntryFederation.served_type_names must be a mapping or None")
        self._served_type_names: Mapping[str, str] = dict(served_type_names or {})
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
        stream_groups: dict[str, list[tuple[int, int]]] = {}
        for source in resolved_sources:
            for backing_index in range(len(source.plan.backings)):
                stream_groups.setdefault(source.source.public_id_prefix, []).append(
                    (source.source_index, backing_index)
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
        collected = self._collect_relationships(visible)
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
            candidates = source.plan.candidate_searchers(
                filter_string,
                sort=sort,
                public_id_prefix=source.source.public_id_prefix,
                as_of=source_as_of,
                only_latest=not revisions,
                revisions=revisions,
                alternatives=alternatives,
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
        collected = self._collect_relationships((candidate,))
        return row, collected.get(id(candidate), _EMPTY_RELATIONSHIPS)

    def _collect_relationships(self, candidates: Sequence[_Candidate]) -> dict[int, _RelatedMap]:
        """Collect exposed weak-link relationships for a page's candidates.

        Candidates are grouped per ``(source, backing)`` by ``_Stream`` identity
        exactly like :meth:`_render_page`, so each backing's link tables are
        scanned once per group.  Only backings that declare exposed
        ``WeakLink`` specs are scanned; every other group issues no query.  The
        SQL scan mirrors the in-memory provider path
        (``StoreEntryProvider._collect_weak_relationships``): links bind
        lineages, retracted links are excluded, and each target resolves at its
        lineage's latest revision.  Non-SQL sources are skipped (their
        relationship serving is a separate backend concern).  Relationships
        reflect the LIVE link state regardless of a page's ``as_of``: like the
        lineage-level provider path, a retraction applies retroactively, so a
        historic page pairs its rows with the current link state.

        :param candidates: The candidates whose relationships are collected.
        :return: Row relationships keyed by ``id(candidate)``; absent for a candidate carrying none.
        """
        groups: dict[int, list[_Candidate]] = {}
        for candidate in candidates:
            groups.setdefault(id(candidate.stream), []).append(candidate)
        collected: dict[int, dict[str, list[RelatedEntry]]] = {}
        for group in groups.values():
            store = group[0].stream.source.source.store
            backing = group[0].stream.backing
            link_specs = [spec for spec in resolve_schema(backing).links if spec.exposed_relationship]
            if not link_specs or not isinstance(store, SqlStore):
                continue
            self._collect_group_relationships(store, backing, link_specs, group, collected)
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
        type as a last resort.  The related id is the target row's raw
        stored ``id`` column at its lineage's latest revision (F9: it matches a
        mounted target endpoint only when that target source's
        ``public_id_prefix`` is empty; dangling targets are skipped).

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
        # ponytail: full link-table scan per page; add WHERE source_lid IN (group lids) if link tables grow large.
        targets_by_source = _live_targets_by_source(
            connection.execute(
                sqlalchemy.select(
                    link_table.c[SOURCE_LID_COLUMN],
                    link_table.c[TARGET_LID_COLUMN],
                    link_table.c[LOGICAL_ID_COLUMN],
                    link_table.c[SID_COLUMN],
                    link_table.c[RETRACTED_COLUMN],
                )
            )
        )
        if not targets_by_source:
            return
        id_by_lid = self._target_ids(connection, store, link_spec.target, targets_by_source)
        for candidate in group:
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
                        target_id,
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
            for lid, max_sid in connection.execute(
                sqlalchemy.select(target_table.c[LOGICAL_ID_COLUMN], sqlalchemy.func.max(target_table.c[SID_COLUMN]))
                .where(target_table.c[LOGICAL_ID_COLUMN].in_(target_lids))
                .group_by(target_table.c[LOGICAL_ID_COLUMN])
            )
            if max_sid is not None  # a None max is a dangling link; fsck reports it
        }
        id_by_sid: dict[int, str] = {
            int(sid): str(value)
            for sid, value in connection.execute(
                sqlalchemy.select(target_table.c[SID_COLUMN], target_table.c["id"]).where(
                    target_table.c[SID_COLUMN].in_(list(max_sid_by_lid.values()))
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
