"""MongoDB store layout initialization and collection preparation."""

import contextlib
import dataclasses
import datetime
import functools
import json
import logging
import threading
import time
import typing
from collections.abc import Mapping, Sequence
from typing import Any, cast

from httk.core import EntryTypeDefinition, FracVector
from httk.core.entry_ids import (
    ALTERNATIVE_KIND_PATTERN,
    check_entry_id,
    check_immutable_id,
    format_alternative_id,
    format_entry_id,
    format_immutable_id,
)
from httk.core.storage import StorageProjectionCycleError, resolve_storage_record
from pymongo import IndexModel, ReturnDocument
from pymongo.errors import BulkWriteError, CollectionInvalid, DuplicateKeyError, PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern

from httk.store.backend.codecs import codec_named, decode_fracvector_exact
from httk.store.backend.schema import LinkSpec, SchemaError, TableSchema, resolve_schema
from httk.store.storage_layout import (
    ADDITIVE_DECLARATION_UPGRADE_HINT,
    ADDITIVE_UPGRADE_HINT,
    DECLARATION_PROTOCOL_VERSION,
    ENTRY_ID_OFFSETS_KEY,
    AdditiveUpgradePlan,
    DeclarationUpgradePlan,
    EntryFamilyDeclaration,
    EntryFamilyLayout,
    EntryLayoutBindingError,
    StorageLayout,
    StorageLayoutUpgradeRequiredError,
    _layout_from_declaration,
    _merge_storage_layouts,
    classify_declaration_upgrade,
    classify_schema_upgrade,
    declaration_is_superseded,
    declaration_json,
    entry_id_number,
    entry_id_offsets_json,
    family_entry_type_definition,
    next_entry_id_offset,
    normalize_entry_families,
    normalize_entry_records,
    parse_entry_id_offsets,
    schema_fingerprint_diff,
    schema_fingerprint_json,
    validate_entry_id_fields,
)
from httk.store.store_common import (
    EntryDispatchIntegrityError,
    EntryIdConflictError,
    EntryIdScheme,
    EntryMetadataConflictError,
    EntryReplacementError,
    IdentityCaches,
    SaveProjection,
    _LinksAccessor,
    _metadata_plan,
    reject_cursor_proxy,
)
from httk.store.store_timestamp import (
    StoreClockRegressionError,
    advance_store_timestamp_mark,
    capture_store_timestamp,
    encode_store_timestamp_state,
    ns_operand_to_store_units,
    parse_store_timestamp_state,
)

from .database import MongoDatabase, TransactionConflictError, TransactionsUnavailableError
from .documents import decode_record, encode_record, preflight_document
from .fsck import FsckSummary, main_backing_documents
from .leases import (
    DEFAULT_TIMING,
    FsckLeasePresentError,
    StoreLockedError,
    WriterDrainTimeoutError,
    WriterLease,
    acquire_fsck,
    acquire_writer,
    clear_stale_lock,
)
from .mapping import (
    COUNTERS_COLLECTION,
    METADATA_COLLECTION,
    collection_name_for,
    counter_next,
    dispatch_index_specs,
    dispatch_validator_for,
    document_fields_for,
    entry_dispatch_table_name,
    index_specs_for,
    link_index_specs_for,
    link_validator_for,
    validator_for,
)

__all__ = ["MongoStore", "StoreClockRegressionError"]

_DOCUMENT_LAYOUT = "mongo-v2"
_IDENTITY_OWNERS = "_httk_identity_owners"
_IDENTITY_VERSION = "1"
_RESERVED_PREFIX = "_httk_"
_LINK_COLLECTION_PREFIX = "_httk_link_"
_METADATA_KEYS = frozenset(
    {
        "_id",
        "protocol",
        "entry_declaration",
        "entry_schemas",
        "document_layout",
        "generation",
        "store_timestamps",
        "identity_ownership",
    }
)
_INTERRUPTED_UPGRADE_HINT = (
    "a declaration upgrade appears to have been interrupted; reopen with the upgraded declaration and upgrade=True"
)
_NEWER_STORE_HINT = (
    "the store was upgraded to a newer declaration; open it with that declaration (a newer httk), not this older one"
)
_STALE_LAYOUT_HINT = "the store layout changed since this store was opened; reopen it"
_DISPATCH_BATCH = 500
# How long a declaration upgrade waits for live writers to drain (SQL parity
# with its _UPGRADE_LOCK_TIMEOUT); fsck() itself keeps waiting indefinitely.
_UPGRADE_DRAIN_BUDGET = 30.0
type _Remedy = typing.Literal["upgrade", "rebuild", "reopen", "retry"]
_LOGGER = logging.getLogger("httk.store.backend.mongo")
_TRANSACTION_ATTEMPTS = 5
_MISSING = object()
_BOUND_STORE = "_httk_bound_store"
_BOUND_LID = "_httk_bound_lid"
_BOUND_LINKS = "_httk_bound_links"


def _schema_is_ahead(stored: str, current: str) -> bool:
    """Whether the stored fingerprint additively extends ``current`` with further entry backing tables.

    That is what a declaration upgrade's partial restamp leaves behind when the
    store is reopened with the pre-upgrade declaration.
    """
    if not isinstance(classify_schema_upgrade(current, stored), AdditiveUpgradePlan):
        return False
    try:
        stored_tables = set(json.loads(stored)["entry_id_tables"])
        current_tables = set(json.loads(current)["entry_id_tables"])
    except (TypeError, KeyError, ValueError):
        return False
    return stored_tables > current_tables


def _iter_link_targets(targets: object) -> typing.Iterator[Any]:
    """Yield one target, or each target of an iterable-of-targets (list/tuple/set)."""
    if isinstance(targets, (list, tuple, set, frozenset)):
        yield from targets
    else:
        yield targets


class _LinksDescriptor:
    """The ``links`` non-data descriptor of a fetched, store-bound record.

    Mirrors the SQL ``row_class`` accessor: it reads the hidden store reference
    and lineage id set on the instance by :meth:`MongoStore._fetch`, then returns
    a memoized shared :class:`~httk.store.store_common._LinksAccessor`. A plain,
    unbound instance carries neither and so simply has no ``links`` attribute.
    """

    __slots__ = ("_specs",)

    def __init__(self, specs: tuple[LinkSpec, ...]) -> None:
        self._specs = specs

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        try:
            store = instance.__dict__[_BOUND_STORE]
            lid = instance.__dict__[_BOUND_LID]
        except KeyError:
            raise AttributeError("links") from None
        cached = instance.__dict__.get(_BOUND_LINKS, _MISSING)
        if cached is not _MISSING:
            return cached
        accessor = _LinksAccessor(self._specs, lambda spec: store._linked_by_lid(spec, lid))
        object.__setattr__(instance, _BOUND_LINKS, accessor)
        return accessor


@functools.cache
def _bound_record_class(cls: type) -> type:
    """Return the cached thin subclass of a link-declaring record with a ``.links`` accessor.

    The subclass keeps the base dataclass behaviour intact — equality, hash and
    repr compare the same fields and treat a bound instance and a plain one as
    equal — and adds only the ``links`` descriptor plus the hidden store/lid slots
    set by :meth:`MongoStore._fetch`.
    """
    specs = resolve_schema(cls).links
    fields = dataclasses.fields(cls)
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

    attrs: dict[str, Any] = {
        "__module__": cls.__module__,
        "__httk_storage_record__": cls,
        "__httk_row_base__": cls,
        "__eq__": eq,
        "__ne__": ne,
        "__hash__": bound_hash,
        "__repr__": bound_repr,
        "links": _LinksDescriptor(specs),
    }
    bound_type = type(f"{cls.__name__}Linked", (cls,), attrs)
    return bound_type


def _served_definition(family: EntryFamilyLayout) -> EntryTypeDefinition | None:
    """Return the wire (served) form of a family's internal definition, if resolvable.

    Resolved exactly as :func:`stored_property_mongo_plan` resolves its internal
    definition (:func:`~httk.store.storage_layout.family_entry_type_definition`).
    Returns ``None`` when the family class declares neither
    ``entry_type_definition()`` nor a ``definition_id``, or the definition is
    malformed, leaving the plan builder to raise its own error.

    :param family: The configured entry family to resolve.
    :return: The served definition, or ``None`` when it cannot be resolved.
    """
    definition_id = getattr(family.family, "definition_id", None)
    if not callable(getattr(family.family, "entry_type_definition", None)) and (
        not isinstance(definition_id, str) or not definition_id
    ):
        return None
    try:
        return family_entry_type_definition(family).served_form()
    except TypeError:
        return None


class _AlternativeRequest(typing.NamedTuple):
    """One top-level save's unresolved alternative-group inputs.

    A fresh :meth:`MongoStore.save` carries ``alternative_of``/
    ``alternative_kind`` (resolved against the record's backing collection); a
    :meth:`MongoStore.replace` of an alternative carries the predecessor's
    already-resolved ``replace_alt_*`` values instead.
    """

    alternative_of: str | None = None
    alternative_kind: str | None = None
    replace_alt_group: int | None = None
    replace_alt_kind: str | None = None
    replace_alt_main_id: str | None = None


class _HydrationContext:
    """Per-fetch document cache shared by one recursive hydration."""

    def __init__(self) -> None:
        self.documents: dict[tuple[type, int], Mapping[str, Any]] = {}


class _TransactionState:
    """Thread-local transaction session and deferred identity-cache entries."""

    def __init__(self, session: Any, lease: WriterLease | None) -> None:
        self.session = session
        self.lease = lease
        self.pending: dict[tuple[type, int], tuple[Any, bool]] = {}
        self.pending_sids: dict[tuple[type, int], int] = {}
        self.timestamp_initialized = False
        self.store_timestamp: int | None = None


class MongoStore:
    """Object store foundation for MongoDB-backed storable records.

    Construction stamps a new empty database or validates the existing layout
    declaration.  Record collections are deliberately created only by the
    explicit :meth:`ensure_collections` operation; save and fetch belong to a
    later phase.

    :param database: The MongoDB database wrapper.
    :param entry_records: The required entry-family declaration on first open.
    :param entry_families: Application-owned declarations which bypass global registration.
    :param entry_ids: Optional scheme used to mint ids for defined entry families.
    :param store_timestamps: Whether saved parent documents receive timestamps.
    :param store_timestamp_resolution: Nanoseconds represented by one stored unit.
    :param allow_clock_regression: Whether to disable the process-local clock guard.
    :param clock_regression_grace: Whether to wait briefly for sub-millisecond regressions.
    :param upgrade: Whether to apply a purely additive change on reopen instead
        of raising: a schema-fingerprint change (documents are schemaless, so
        only the stored fingerprint is re-stamped) and/or a declaration change
        that only appends record kinds to existing families or adds families
        (dispatch validators widened and back-filled, entry-id offsets
        recorded, all under the exclusive fsck lease).  Non-additive or other
        differences still raise; every refusal carries a typed ``remedy``.
    :raises TypeError: If the first open omits both declaration forms.
    :raises ~httk.store.storage_layout.StorageLayoutUpgradeRequiredError: If the
        persisted layout is not trusted by this implementation.
    """

    supports_page = True
    """Whether this backend implements keyset result paging."""

    def __init__(
        self,
        database: MongoDatabase,
        *,
        entry_records: Mapping[type, type | tuple[type, ...]] | None = None,
        entry_families: Sequence[EntryFamilyDeclaration] | None = None,
        entry_ids: EntryIdScheme | None = None,
        store_timestamps: bool = True,
        store_timestamp_resolution: int = 1000,
        allow_clock_regression: bool = False,
        clock_regression_grace: bool = True,
        upgrade: bool = False,
    ) -> None:
        if (
            not isinstance(store_timestamp_resolution, int)
            or isinstance(store_timestamp_resolution, bool)
            or store_timestamp_resolution <= 0
        ):
            raise ValueError("store_timestamp_resolution must be a positive integer")
        self._database = database
        self._entry_ids = entry_ids
        self._upgrade = upgrade
        self._identity_ownership_ready = True
        self._store_timestamps = store_timestamps
        self._store_timestamp_resolution = store_timestamp_resolution
        self._allow_clock_regression = allow_clock_regression
        self._clock_regression_grace = clock_regression_grace
        self._clock = time.time_ns
        self._store_timestamp_mark: int | None = None
        self._layout: StorageLayout | None = None
        # The persisted declaration this instance opened (or upgraded to); every
        # writer-lease acquisition compares it to refuse a stale writer.
        self._opened_declaration: str | None = None
        # Per-family entry-id numbering offsets (absent family = 0).
        self._entry_id_offsets: dict[str, int] = {}
        self._collections_ready: set[str] = set()
        # Layout declarations describe the persistent roots; this additional
        # set lets fsck also attribute arbitrary record classes saved through
        # this live store instance.
        self._known_record_types: set[type] = set()
        self._identity = IdentityCaches()
        self._write_lock = threading.RLock()
        self._local = threading.local()
        self._failed_identities: set[tuple[type, int]] = set()
        hello = database.client.admin.command("hello")
        self._max_bson_size = int(hello.get("maxBsonObjectSize", 16 * 1024 * 1024))
        if not database.supports_transactions:
            _LOGGER.warning(
                "MongoStore is running in degraded mode without multi-document transactions",
                extra={"context": "storage"},
            )
        layouts = []
        if entry_records is not None:
            layouts.append(normalize_entry_records(entry_records))
        if entry_families is not None:
            layouts.append(normalize_entry_families(entry_families))
        supplied = _merge_storage_layouts(*layouts) if layouts else None
        if supplied is not None:
            validate_entry_id_fields(supplied)
        self._initialize_layout(supplied)
        for family in self.layout.families:
            self._known_record_types.update(family.records)
        self._entry_record_types: dict[type, tuple[str, int, int]] = {
            record: (
                self._family_entry_type(family.family),
                len(family.records),
                backing_index,
            )
            for family in self.layout.families
            if family.definition_id is not None
            for backing_index, record in enumerate(family.records)
        }
        self._entry_id_offset_by_record: dict[type, int] = {
            record: self._entry_id_offsets.get(family.name, 0)
            for family in self.layout.families
            if family.definition_id is not None
            for record in family.records
        }
        self._last_generation = self._layout_generation()
        self._initialize_store_timestamp_mark()

    def __repr__(self) -> str:
        return f"MongoStore(database={self._database!r})"

    @staticmethod
    def _family_entry_type(family: type) -> str:
        """Return the validated served entry type declared by ``family``."""
        entry_type = getattr(family, "type", None)
        if not isinstance(entry_type, str) or not entry_type or entry_type != entry_type.strip():
            raise ValueError(f"{family.__name__}.type must be a non-empty stripped entry type")
        return entry_type

    def _entry_id_number(self, record_type: type, logical_id: int) -> int:
        """Return the family-unique numeric component for a record lineage.

        The number is ``offset + logical_id * B + backing_index`` exactly as in
        :class:`~httk.store.backend.sql.store.SqlStore`: ``offset`` is ``0``
        unless an additive upgrade appended record kinds to the family, so a
        family whose record list never changed mints exactly as before.
        """
        # Uniqueness: within one (offset, B) epoch distinct (logical_id, index)
        # pairs give distinct numbers because 0 <= index < B.  An upgrade that
        # changes B sets offset = max(every existing id number and ownership
        # claim, previous offset) + 1, and every later number is >= offset + B
        # (logical_id >= 1), so it exceeds every number minted before; the
        # ownership collection still refuses any collision loudly.
        _entry_type, backing_count, backing_index = self._entry_record_types[record_type]
        return self._entry_id_offset_by_record.get(record_type, 0) + logical_id * backing_count + backing_index

    @property
    def layout(self) -> StorageLayout:
        """Return the immutable persisted entry declaration.

        :return: The normalized storage layout.
        """
        assert self._layout is not None
        return self._layout

    @property
    def entry_layout(self) -> tuple[EntryFamilyLayout, ...]:
        """Return configured entry-family layouts in stable order.

        :return: The configured entry-family layouts.
        """
        return self.layout.families

    @property
    def entry_records(self) -> Mapping[type, tuple[type, ...]]:
        """Return configured family classes mapped to backing classes.

        :return: The normalized entry declaration keyed by family class.
        """
        return self.layout.entry_records

    @property
    def store_timestamps(self) -> bool:
        """Whether parent documents carry store-managed timestamps."""
        return self._store_timestamps

    @property
    def store_timestamp_resolution(self) -> int | None:
        """Return nanoseconds per stored timestamp unit, or ``None`` when disabled."""
        return self._store_timestamp_resolution if self._store_timestamps else None

    @property
    def _store_timestamp_state(self) -> str:
        return encode_store_timestamp_state(self._store_timestamps, self._store_timestamp_resolution)

    def _initialize_layout(self, supplied: StorageLayout | None) -> None:
        database = self._database.database
        names = {name for name in database.list_collection_names() if not name.startswith("system.")}
        metadata_exists = METADATA_COLLECTION in names
        stored = database[METADATA_COLLECTION].find_one({"_id": "layout"}) if metadata_exists else None

        if not metadata_exists and not names:
            if supplied is None:
                raise TypeError("entry_records or entry_families is required when opening an uninitialized database")
            self._validate_layout_names(supplied)
            document = {
                "_id": "layout",
                "protocol": DECLARATION_PROTOCOL_VERSION,
                "entry_declaration": declaration_json(supplied),
                "entry_schemas": schema_fingerprint_json(supplied),
                "document_layout": _DOCUMENT_LAYOUT,
                "generation": 0,
                "store_timestamps": self._store_timestamp_state,
            }
            try:
                database[METADATA_COLLECTION].insert_one(document)
            except DuplicateKeyError:
                # Another opener won the single-document first-open race.
                stored = database[METADATA_COLLECTION].find_one({"_id": "layout"})
            else:
                self._install_layout(supplied)
                self._ensure_identity_owners()
                database[METADATA_COLLECTION].update_one(
                    {"_id": "layout"},
                    {"$set": {"identity_ownership": _IDENTITY_VERSION}},
                )
                return

        if stored is None:
            self._raise_unversioned(names)
        assert stored is not None
        self._open_marked_layout(stored, supplied, names)

    def _open_marked_layout(
        self,
        stored: Mapping[str, Any],
        supplied: StorageLayout | None,
        collection_names: set[str],
    ) -> None:
        diff: dict[str, object] = {}

        def declaration_diff() -> dict[str, object]:
            # Each check contributes one named aspect; independent mismatches
            # accumulate instead of overwriting a single "declaration" payload.
            return typing.cast("dict[str, object]", diff.setdefault("declaration", {}))

        ownership_pending = "identity_ownership" not in stored
        # ``entry_id_offsets`` is the one optional key; only an additive
        # declaration upgrade writes it (older httk-store refuses it).
        present_keys = set(stored) - {ENTRY_ID_OFFSETS_KEY}
        if present_keys != (_METADATA_KEYS - {"identity_ownership"} if ownership_pending else _METADATA_KEYS):
            declaration_diff()["metadata_keys"] = {
                "expected": tuple(sorted(_METADATA_KEYS)),
                "optional": (ENTRY_ID_OFFSETS_KEY,),
                "actual": tuple(sorted(stored)),
            }
        if not ownership_pending and stored.get("identity_ownership") != _IDENTITY_VERSION:
            declaration_diff()["identity_ownership"] = {
                "expected": _IDENTITY_VERSION,
                "actual": stored.get("identity_ownership"),
            }
        protocol_actual = stored.get("protocol")
        document_layout_actual = stored.get("document_layout")
        if protocol_actual != DECLARATION_PROTOCOL_VERSION or document_layout_actual != _DOCUMENT_LAYOUT:
            diff["protocol"] = {
                "expected": {
                    "protocol": DECLARATION_PROTOCOL_VERSION,
                    "document_layout": _DOCUMENT_LAYOUT,
                },
                "actual": {
                    "protocol": protocol_actual,
                    "document_layout": document_layout_actual,
                },
            }

        persisted_timestamps = stored.get("store_timestamps")
        parsed_timestamps = parse_store_timestamp_state(persisted_timestamps)
        if (
            parsed_timestamps is None
            or parsed_timestamps[0] != self._store_timestamps
            or (parsed_timestamps[0] and parsed_timestamps[1] != self._store_timestamp_resolution)
        ):
            declaration_diff()["store_timestamps"] = {
                "expected": self._store_timestamp_state,
                "actual": persisted_timestamps,
            }

        persisted: StorageLayout | None = None
        # An additive declaration upgrade makes ``persisted`` the *target*
        # layout from here on: the schema diff, the reserved-collection
        # expectations and the install all use it, so a re-run after a crash
        # that left target collections behind is not refused.
        declaration_upgrade: DeclarationUpgradePlan | None = None
        declaration_rejection: str | None = None
        declaration = stored.get("entry_declaration")
        if supplied is not None and isinstance(declaration, str):
            if declaration == declaration_json(supplied):
                persisted = supplied
            else:
                classified = classify_declaration_upgrade(declaration, supplied)
                if isinstance(classified, DeclarationUpgradePlan):
                    # Read-only: a collection the upgrade would newly attach must
                    # be empty, so upgrade=False never advertises an additive
                    # change that upgrade=True would then refuse.
                    foreign = self._attached_collections_with_documents(declaration, supplied, classified)
                    if foreign is not None:
                        classified = foreign
                if isinstance(classified, DeclarationUpgradePlan):
                    persisted = supplied
                    declaration_upgrade = classified
                elif declaration_is_superseded(declaration, supplied):
                    # A client one declaration behind: the store is healthy and newer.
                    declaration_rejection = _NEWER_STORE_HINT
                    declaration_diff()["entry_declaration"] = {
                        "expected": declaration,
                        "actual": declaration_json(supplied),
                        "additive": False,
                        "stored_is_newer": True,
                        "reason": classified,
                    }
                else:
                    declaration_rejection = f"{classified}; the declaration change is not additive, rebuild the store"
                    declaration_diff()["entry_declaration"] = {
                        "expected": declaration,
                        "actual": declaration_json(supplied),
                        "additive": False,
                        "reason": classified,
                    }
        else:
            try:
                if not isinstance(declaration, str):
                    raise ValueError("metadata is missing entry_declaration")
                persisted = _layout_from_declaration(declaration)
                if declaration_json(persisted) != declaration:
                    raise ValueError("stored entry declaration is not in its canonical deterministic encoding")
            except EntryLayoutBindingError:
                raise
            except (TypeError, ValueError) as error:
                declaration_diff()["entry_declaration"] = {
                    "expected": "canonical registered declaration or explicit entry_families binding",
                    "actual": declaration,
                    "error": str(error),
                }
        entry_id_offsets: dict[str, int] = {}
        if persisted is not None and ENTRY_ID_OFFSETS_KEY in stored:
            raw_offsets = stored[ENTRY_ID_OFFSETS_KEY]
            try:
                if not isinstance(raw_offsets, str):
                    raise ValueError("entry_id_offsets must be a JSON string")
                entry_id_offsets = parse_entry_id_offsets(raw_offsets, persisted)
            except ValueError as error:
                declaration_diff()[ENTRY_ID_OFFSETS_KEY] = {"actual": raw_offsets, "error": str(error)}
        schema_plan: AdditiveUpgradePlan | str | None = None
        if persisted is not None and "entry_schemas" in stored:
            # Absence of the key is already reported by the exact key-set check.
            current_schemas = schema_fingerprint_json(persisted)
            schema_diff = schema_fingerprint_diff(stored["entry_schemas"], current_schemas)
            if schema_diff:
                diff["schema"] = schema_diff
                schema_plan = classify_schema_upgrade(stored["entry_schemas"], current_schemas)

        def pending_declaration_diff(reason: str | None = None) -> None:
            # Report a pending declaration change beside whatever else keeps
            # the store from opening; ``reason`` names a non-additive schema part.
            if declaration_upgrade is not None:
                assert persisted is not None
                entry = {"expected": declaration, "actual": declaration_json(persisted), "additive": reason is None}
                if reason is not None:
                    entry["reason"] = reason
                declaration_diff()["entry_declaration"] = entry

        def remedy() -> _Remedy:
            # rebuild dominates (nothing else can fix it), then reopen (fix the
            # declaration or options first), then upgrade (everything left is additive).
            aspects = typing.cast("dict[str, object]", diff.get("declaration", {}))
            declaration_entry = aspects.get("entry_declaration")
            newer_store = isinstance(declaration_entry, Mapping) and declaration_entry.get("stored_is_newer") is True
            rebuild = (
                "protocol" in diff
                or isinstance(schema_plan, str)
                or any(key in aspects for key in ("metadata_keys", "identity_ownership", ENTRY_ID_OFFSETS_KEY))
                or (
                    isinstance(declaration_entry, Mapping)
                    and declaration_entry.get("additive") is not True
                    and not newer_store
                )
            )
            if rebuild:
                return "rebuild"
            if newer_store or "store_timestamps" in aspects:
                return "reopen"
            return "upgrade"

        upgrade_pending = False
        additive_diff: dict[str, object] = {}
        additive_hint = ADDITIVE_UPGRADE_HINT if declaration_upgrade is None else ADDITIVE_DECLARATION_UPGRADE_HINT
        if set(diff) == {"schema"} or (declaration_upgrade is not None and set(diff) <= {"schema"}):
            assert persisted is not None
            plan = schema_plan if schema_plan is not None else AdditiveUpgradePlan({})
            if isinstance(plan, str) and declaration_upgrade is not None:
                pending_declaration_diff(plan)
                raise StorageLayoutUpgradeRequiredError(
                    diff, hint=f"{plan}; the layout change is not additive, rebuild the store", remedy="rebuild"
                )
            if isinstance(plan, AdditiveUpgradePlan):
                # Neither advertised (upgrade=False) nor applied (upgrade=True)
                # until the read-only physical checks below have passed, so
                # remedy "upgrade" is never emitted for a store upgrade=True
                # would refuse; Mongo has no transaction to roll a bad open back.
                upgrade_pending = True
                additive_diff = diff
                diff = {}
        if diff:
            pending_declaration_diff()
            rejection_hint = declaration_rejection
            chosen = remedy()
            if (
                rejection_hint is None
                and set(diff) == {"schema"}
                and declaration_upgrade is None
                and persisted is not None
                and _schema_is_ahead(stored["entry_schemas"], schema_fingerprint_json(persisted))
            ):
                # The stored fingerprint already carries backing collections this
                # (older) declaration lacks: a declaration upgrade's partial
                # restamp.  Only the upgraded declaration can finish it.
                rejection_hint, chosen = _INTERRUPTED_UPGRADE_HINT, "reopen"
            raise StorageLayoutUpgradeRequiredError(diff, hint=rejection_hint, remedy=chosen)
        assert persisted is not None
        self._validate_layout_names(persisted)
        self._check_physical_layout(persisted, collection_names, ownership_pending)
        if upgrade_pending and not self._upgrade:
            diff = additive_diff
            pending_declaration_diff()
            raise StorageLayoutUpgradeRequiredError(diff, hint=additive_hint, remedy="upgrade")
        self._install_layout(persisted)
        self._identity_ownership_ready = not ownership_pending
        if ownership_pending and self._upgrade:
            lease = acquire_fsck(self._database.database)
            try:
                self._ensure_identity_owners()
                self._sync_identity_ownership(lease)
                self._database.database[METADATA_COLLECTION].update_one(
                    {"_id": "layout"},
                    {"$set": {"identity_ownership": _IDENTITY_VERSION}},
                )
                self._identity_ownership_ready = True
            finally:
                lease.release()
        if upgrade_pending and declaration_upgrade is not None:
            assert isinstance(declaration, str)
            entry_id_offsets = self._apply_declaration_upgrade(
                declaration, persisted, declaration_upgrade, entry_id_offsets
            )
        elif upgrade_pending:
            self._restamp_entry_schemas(persisted)
        self._entry_id_offsets = entry_id_offsets

    def _check_physical_layout(
        self, layout: StorageLayout, collection_names: set[str], ownership_pending: bool
    ) -> None:
        """Run the read-only physical checks of a reopen against ``layout`` (the target while upgrading).

        Reserved ``_httk_`` collections and the identity-ownership indexes.
        They run before an additive change is advertised or applied.

        :param layout: The layout the store will be opened with.
        :param collection_names: The database's collection names.
        :param ownership_pending: Whether the ownership capability is still to be installed.
        :return: None.
        :raises StorageLayoutUpgradeRequiredError: On the first problem found.
        """
        expected_reserved = {
            METADATA_COLLECTION,
            COUNTERS_COLLECTION,
            _IDENTITY_OWNERS,
            *(entry_dispatch_table_name(family.name) for family in layout.families if len(family.records) > 1),
        }
        # Dispatch collections of families this layout declares with one
        # backing: the residue of an interrupted 1→N declaration upgrade.
        interrupted_dispatch = {
            entry_dispatch_table_name(family.name) for family in layout.families if len(family.records) == 1
        }
        problems: dict[str, object] = {}
        for name in collection_names:
            # Weak-link collections (``_httk_link_*``) belong to storable source
            # classes that need not be declared to reopen a store (e.g. non-entry
            # records), so they cannot be enumerated here; fsck attributes and
            # structurally validates them instead.
            if name.startswith(_LINK_COLLECTION_PREFIX):
                continue
            if name.startswith(_RESERVED_PREFIX) and name not in expected_reserved:
                problems[name] = {
                    "reserved": True,
                    "message": "unexpected collection uses the MongoStore-reserved _httk_ prefix",
                }
        if problems:
            if set(problems) <= interrupted_dispatch:
                # Only the upgraded declaration can finish (or own) this residue.
                raise StorageLayoutUpgradeRequiredError(
                    {"schema": problems}, hint=_INTERRUPTED_UPGRADE_HINT, remedy="reopen"
                )
            raise StorageLayoutUpgradeRequiredError({"schema": problems}, remedy="rebuild")
        if ownership_pending:
            return
        indexes = self._database.database[_IDENTITY_OWNERS].index_information()
        expected_indexes = {
            "identity_value": [("family", 1), ("kind", 1), ("value", 1)],
            "identity_owner": [
                ("family", 1),
                ("kind", 1),
                ("backing", 1),
                ("owner", 1),
            ],
        }
        if any(
            indexes.get(name, {}).get("key") != keys
            or not indexes.get(name, {}).get("unique")
            or "partialFilterExpression" in indexes.get(name, {})
            for name, keys in expected_indexes.items()
        ):
            raise StorageLayoutUpgradeRequiredError(
                {"schema": {_IDENTITY_OWNERS: "missing identity uniqueness indexes"}}, remedy="rebuild"
            )

    def _attached_collections_with_documents(
        self, stored_declaration: str, target: StorageLayout, plan: DeclarationUpgradePlan
    ) -> str | None:
        """Name a collection the upgrade would newly attach to a family that already holds documents, if any.

        Such documents were stored outside the family (an ad-hoc save of the
        class) and would enter it without family identity, so the change is not
        additive.  Read-only (``find_one`` per attached collection).
        """
        families = {family.name: family for family in target.families}
        # Already validated by classify_declaration_upgrade.
        stored_counts = {item["family"]: len(item["records"]) for item in json.loads(stored_declaration)["families"]}
        attached: list[tuple[str, type]] = []
        for name in plan.changed_families:
            attached.extend((name, record) for record in families[name].records[stored_counts[name] :])
        for name in plan.new_families:
            attached.extend((name, record) for record in families[name].records)
        database = self._database.database
        for family_name, record in attached:
            collection_name = collection_name_for(resolve_schema(record))
            if database[collection_name].find_one({}, {"_id": 1}) is not None:
                return (
                    f"collection {collection_name!r} of record {record.__name__} newly declared in family "
                    f"{family_name!r} already holds documents stored outside the family"
                )
        return None

    def _after_upgrade_step(self, step: str) -> None:
        """Test seam after each durable step of an additive declaration upgrade.

        Production leaves it a no-op.  The crash-convergence tests raise from it
        to simulate a process death after the steps named ``prepared``,
        ``claimed``, ``dispatch``, ``offsets``, ``restamp-schemas``,
        ``restamp-offsets`` and ``restamped``.  Every Mongo step is durable as it
        happens (there is no DDL transaction), so each seam is a real crash point.
        """

    def _apply_declaration_upgrade(
        self,
        stored_declaration: str,
        target: StorageLayout,
        plan: DeclarationUpgradePlan,
        stored_offsets: Mapping[str, int],
    ) -> dict[str, int]:
        """Apply an additive declaration upgrade (appended record kinds, new families) crash-convergently.

        The steps mirror :class:`~httk.store.backend.sql.store.SqlStore`'s:

        0. *exclude writers and claim* — take the singleton fsck lease, which
           drains every fresh writer lease and refuses new ones for the whole
           upgrade (a busy store raises remedy ``"retry"``), then re-read the
           layout document: if its declaration is no longer the one this open
           read, the store already holds the identical target (adopt it) or a
           different one (remedy ``"reopen"``);
        1. re-check that no collection being newly attached to a family holds
           documents (the open already refused that read-only);
        2. for every family whose record list grew, ``collMod`` (or create) its
           dispatch collection with the full target validator and index, then
           back-fill one dispatch document per main backing document
           (:func:`~httk.store.backend.mongo.fsck.main_backing_documents`, the
           fsck repair derivation) with unordered inserts that tolerate
           already-present identical documents;
        3. compute each grown family's entry-id numbering offset over every
           stored ``f.id``/``f.immutable_id`` and every ownership claim;
        4. restamp ``entry_schemas``, then ``entry_id_offsets``, then — last —
           the declaration with ``find_one_and_update`` conditioned on its old
           value (also advancing ``generation``, so live handles drop caches).

        Every step is idempotent and planned against the target layout, and
        the declaration is written last, so a crash at any point leaves the old
        declaration and a re-run with ``upgrade=True`` converges; a re-run after
        the offsets were written only raises the offset further.

        :param stored_declaration: The persisted declaration this open read.
        :param target: The upgraded (target) layout.
        :param plan: The additive declaration classification.
        :param stored_offsets: The persisted entry-id offsets.
        :return: The entry-id offsets now in force.
        :raises StorageLayoutUpgradeRequiredError: If the store is busy, a
            concurrent upgrade restamped a different layout, or an attached
            collection already holds foreign documents.
        """
        database = self._database.database
        metadata = database[METADATA_COLLECTION]
        target_declaration = declaration_json(target)
        try:
            lease = acquire_fsck(
                database, timing=dataclasses.replace(DEFAULT_TIMING, drain_budget=_UPGRADE_DRAIN_BUDGET)
            )
        except StoreLockedError as error:
            if isinstance(error, FsckLeasePresentError):
                hint = (
                    "an fsck or upgrade lease is present; if its owner is dead, clear it once it is stale "
                    f"(default {DEFAULT_TIMING.stale_after:g} s) with "
                    "httk.store.backend.mongo.clear_stale_lock(database), then retry"
                )
            elif isinstance(error, WriterDrainTimeoutError):
                hint = (
                    f"live writers did not finish within {_UPGRADE_DRAIN_BUDGET:g} s; retry the upgrade when they have"
                )
            else:
                hint = f"{error}; retry the upgrade once those writers have finished or been verified dead"
            raise StorageLayoutUpgradeRequiredError(
                {"declaration": {"entry_declaration": {"expected": stored_declaration, "error": str(error)}}},
                hint=hint,
                remedy="retry",
            ) from error
        try:
            self._after_upgrade_step("prepared")
            current = metadata.find_one({"_id": "layout"}, {"entry_declaration": 1})
            if current is None or current.get("entry_declaration") != stored_declaration:
                return self._resolve_lost_declaration_upgrade(stored_declaration, target)
            self._after_upgrade_step("claimed")

            foreign = self._attached_collections_with_documents(stored_declaration, target, plan)
            if foreign is not None:
                raise StorageLayoutUpgradeRequiredError(
                    {
                        "declaration": {
                            "entry_declaration": {
                                "expected": stored_declaration,
                                "actual": target_declaration,
                                "additive": False,
                                "reason": foreign,
                            }
                        }
                    },
                    hint=f"{foreign}; the declaration change is not additive, rebuild the store",
                    remedy="rebuild",
                )

            families = {family.name: family for family in target.families}
            for name in plan.changed_families:
                self._rebuild_entry_dispatch(families[name], lease)
            self._after_upgrade_step("dispatch")

            offsets = dict(stored_offsets)
            for name in plan.changed_families:
                family = families[name]
                if family.definition_id is None:
                    continue  # no store-minted ids in a family without a definition id
                offset = next_entry_id_offset(self._existing_entry_id_numbers(family, lease), offsets.get(name, 0))
                if offset:
                    offsets[name] = offset
            self._after_upgrade_step("offsets")

            # Declaration last: any partial restamp leaves the old declaration,
            # so a retry re-enters this routine (never the schema-only path).
            if metadata.find_one({"_id": "layout", "entry_declaration": stored_declaration}, {"_id": 1}) is None:
                return self._resolve_lost_declaration_upgrade(stored_declaration, target)
            metadata.update_one({"_id": "layout"}, {"$set": {"entry_schemas": schema_fingerprint_json(target)}})
            self._after_upgrade_step("restamp-schemas")
            if offsets:
                metadata.update_one({"_id": "layout"}, {"$set": {ENTRY_ID_OFFSETS_KEY: entry_id_offsets_json(offsets)}})
            self._after_upgrade_step("restamp-offsets")
            won = metadata.find_one_and_update(
                {"_id": "layout", "entry_declaration": stored_declaration},
                {"$set": {"entry_declaration": target_declaration}, "$inc": {"generation": 1}},
                return_document=ReturnDocument.AFTER,
            )
            if won is None:
                # Unreachable while the fsck lease is held.
                now = metadata.find_one({"_id": "layout"}, {"entry_declaration": 1})
                actual = None if now is None else now.get("entry_declaration")
                raise StorageLayoutUpgradeRequiredError(
                    {"declaration": {"entry_declaration": {"expected": stored_declaration, "actual": actual}}},
                    hint="the store layout changed concurrently during the upgrade's restamp; retry the upgrade",
                    remedy="retry" if actual == target_declaration else "reopen",
                )
            self._after_upgrade_step("restamped")
            return offsets
        finally:
            lease.release()

    def _resolve_lost_declaration_upgrade(self, stored_declaration: str, target: StorageLayout) -> dict[str, int]:
        """Settle a declaration that moved under the upgrade: a no-op for the identical target, else reopen."""
        current = self._database.database[METADATA_COLLECTION].find_one({"_id": "layout"}) or {}
        if current.get("entry_declaration") == declaration_json(target) and current.get(
            "entry_schemas"
        ) == schema_fingerprint_json(target):
            offsets = current.get(ENTRY_ID_OFFSETS_KEY)
            try:
                if offsets is not None and not isinstance(offsets, str):
                    raise ValueError("entry_id_offsets must be a JSON string")
                return {} if offsets is None else parse_entry_id_offsets(offsets, target)
            except ValueError as error:
                raise StorageLayoutUpgradeRequiredError(
                    {"declaration": {ENTRY_ID_OFFSETS_KEY: {"actual": offsets, "error": str(error)}}},
                    remedy="rebuild",
                ) from error
        raise StorageLayoutUpgradeRequiredError(
            {
                "declaration": {
                    "entry_declaration": {
                        "expected": stored_declaration,
                        "actual": current.get("entry_declaration"),
                    }
                }
            },
            hint="a concurrent upgrade changed the store layout to a different declaration; reopen it",
            remedy="reopen",
        )

    def _rebuild_entry_dispatch(self, family: EntryFamilyLayout, lease: Any) -> None:
        """Bring one grown family's dispatch collection to the target shape and back-fill it idempotently.

        Dispatch documents (``{_id: content_id, record, sid}``) do not depend
        on the number of backings, so nothing is dropped: the validator's
        ``record`` enum is widened by ``collMod`` (or the collection created for
        a 1→N growth), its unique ``(record, sid)`` index ensured, and one
        document inserted per main backing document that lacks one.
        """
        name = entry_dispatch_table_name(family.name)
        database = self._database.database
        self._ensure_collection(name, dispatch_validator_for(family))
        models = []
        for spec in dispatch_index_specs(family):
            options: dict[str, Any] = {"name": spec.name, "unique": spec.unique}
            if spec.partial_filter_expression is not None:
                options["partialFilterExpression"] = spec.partial_filter_expression
            models.append(IndexModel(list(spec.keys), **options))
        if models:
            database[name].create_indexes(models)
        batch: list[dict[str, Any]] = []
        for _collection_name, record_name, document in main_backing_documents(database, family):
            content_id = document.get("content_id")
            if not isinstance(content_id, str):
                continue  # reported by fsck; it can have no dispatch document
            batch.append({"_id": content_id, "record": record_name, "sid": document["_id"]})
            if len(batch) >= _DISPATCH_BATCH:
                self._insert_dispatch_batch(family, database[name], batch)
                batch = []
                lease.refresh_heartbeat()
        if batch:
            self._insert_dispatch_batch(family, database[name], batch)

    @staticmethod
    def _insert_dispatch_batch(family: EntryFamilyLayout, dispatch: Any, batch: list[dict[str, Any]]) -> None:
        """Insert dispatch documents unordered, accepting only already-present *identical* documents."""
        try:
            dispatch.insert_many(batch, ordered=False)
        except BulkWriteError as error:
            if error.details.get("writeConcernErrors"):
                # pymongo raises for write-concern errors even with no write
                # error: the back-fill may be rolled back by a failover, so the
                # upgrade must not restamp on top of it.
                raise
            for write_error in error.details.get("writeErrors", ()):
                if write_error.get("code") != 11000:
                    raise
                wanted = batch[int(write_error["index"])]
                existing = dispatch.find_one({"_id": wanted["_id"]})
                if (
                    existing is None
                    or existing.get("record") != wanted["record"]
                    or existing.get("sid") != wanted["sid"]
                ):
                    raise EntryDispatchIntegrityError(
                        f"entry dispatch {family.name!r} already maps content_id {wanted['_id']!r} or backing sid "
                        f"{wanted['sid']} to a different association; run fsck(repair_conflicts=True) or rebuild"
                    ) from error

    def _existing_entry_id_numbers(self, family: EntryFamilyLayout, lease: Any) -> typing.Iterator[int]:
        """Yield the numeric suffix of every id ``family`` has stored or claimed.

        Covers every ``f.id``/``f.immutable_id`` of every backing plus every
        ownership claim of the family, so an orphan claim left by a failed
        non-transactional write can never be re-minted either.
        """
        database = self._database.database
        for record in family.records:
            collection = database[collection_name_for(resolve_schema(record))]
            for document in collection.find({}, {"f.id": 1, "f.immutable_id": 1}):
                lease.refresh_heartbeat()
                fields = document.get("f") or {}
                for value in (fields.get("id"), fields.get("immutable_id")):
                    number = entry_id_number(value)
                    if number is not None:
                        yield number
        for claim in database[_IDENTITY_OWNERS].find({"family": family.name}, {"value": 1}):
            number = entry_id_number(claim.get("value"))
            if number is not None:
                yield number

    def _stored_declaration(self) -> str | None:
        """Read the persisted ``entry_declaration`` value."""
        document = self._database.database[METADATA_COLLECTION].find_one({"_id": "layout"}, {"entry_declaration": 1})
        value = None if document is None else document.get("entry_declaration")
        return value if isinstance(value, str) else None

    def _stale_layout_error(self, current: str | None) -> StorageLayoutUpgradeRequiredError:
        """Build the remedy-``reopen`` error for a handle whose layout another instance changed."""
        return StorageLayoutUpgradeRequiredError(
            {
                "declaration": {
                    "entry_declaration": {"expected": self._opened_declaration, "actual": current, "stale": True}
                }
            },
            hint=_STALE_LAYOUT_HINT,
            remedy="reopen",
        )

    def _raise_if_layout_moved(self) -> None:
        """On a read miss or integrity failure, report a stale handle (remedy ``"reopen"``) instead."""
        current = self._stored_declaration()
        if current != self._opened_declaration:
            raise self._stale_layout_error(current)

    def _acquire_writer_lease(self) -> WriterLease:
        """Acquire a writer lease and refuse it when the layout moved since this store was opened.

        The lease protocol orders this check against declaration upgrades:
        the writer's registration is durable before it reads the layout, and an
        upgrade drains every fresh registration behind the fsck singleton before
        it touches anything.  So either the upgrade finished before this read
        (the new declaration is seen and the write refused) or it waits for this
        writer to release.

        :return: The live writer lease.
        :raises StorageLayoutUpgradeRequiredError: If the declaration changed
            since this store was opened (remedy ``"reopen"``).
        """
        lease = acquire_writer(self._database.database)
        if lease.declaration != self._opened_declaration:
            lease.release()
            raise self._stale_layout_error(lease.declaration)
        return lease

    def _require_identity_ownership(self) -> None:
        if not self._identity_ownership_ready:
            raise StorageLayoutUpgradeRequiredError(
                {
                    "declaration": {
                        "identity_ownership": {
                            "expected": _IDENTITY_VERSION,
                            "actual": None,
                        }
                    }
                },
                hint="reopen with upgrade=True to validate durable entry-id ownership before writing",
                remedy="upgrade",
            )

    def _ensure_identity_owners(self) -> None:
        collection = self._database.database[_IDENTITY_OWNERS]
        collection.create_index(
            [("family", 1), ("kind", 1), ("value", 1)],
            unique=True,
            name="identity_value",
        )
        collection.create_index(
            [("family", 1), ("kind", 1), ("backing", 1), ("owner", 1)],
            unique=True,
            name="identity_owner",
        )

    def _claim_entry_identity(self, record_type: type, document: Mapping[str, Any]) -> None:
        """Reserve identities before publishing the parent document.

        Non-transactional failures deliberately retain reservations. Only fsck,
        after excluding writers, may reclaim claims whose owners do not exist.
        """
        family = self._family_for_backing(record_type)
        if family is None or family.definition_id is None:
            return
        backing = family.record_names[family.records.index(record_type)]
        for kind, owner in (
            ("id", document["alt_id"]),
            ("immutable_id", document["_id"]),
        ):
            value = document["f"].get(kind)
            if value is None:
                raise RuntimeError(
                    f"entry family {family.name!r} backing {backing!r} owner {owner} is missing {kind!r}"
                )
            claim = {
                "family": family.name,
                "kind": kind,
                "value": value,
                "backing": backing,
                "owner": owner,
            }
            try:
                self._database.database[_IDENTITY_OWNERS].update_one(
                    claim,
                    {"$setOnInsert": claim},
                    upsert=True,
                    **self._session_kwargs(),
                )
            except DuplicateKeyError as error:
                raise EntryIdConflictError(family.name, str(value), None, int(owner)) from error

    def _sync_identity_ownership(self, lease: Any, *, reconcile: bool = False) -> None:
        """Validate/backfill claims under the existing exclusive fsck fence."""
        owners = self._database.database[_IDENTITY_OWNERS]
        for family in self.layout.families:
            if family.definition_id is None:
                continue
            for backing, record in zip(family.record_names, family.records, strict=True):
                collection = self._database.database[collection_name_for(resolve_schema(record))]
                if reconcile:
                    for claim in owners.find({"family": family.name, "backing": backing}):
                        lease.refresh_heartbeat()
                        key = "alt_id" if claim["kind"] == "id" else "_id"
                        if collection.find_one({key: claim["owner"]}, {"_id": 1}) is None:
                            owners.delete_one({"_id": claim["_id"]})
                for document in collection.find({}, {"f.id": 1, "f.immutable_id": 1, "alt_id": 1}):
                    lease.refresh_heartbeat()
                    self._claim_entry_identity(record, document)

    def _raise_unversioned(self, collection_names: set[str]) -> None:
        schema: dict[str, object] = {METADATA_COLLECTION: {"missing": True}}
        for name in sorted(collection_names):
            schema[name] = (
                {
                    "reserved": True,
                    "message": "unexpected collection uses the MongoStore-reserved _httk_ prefix",
                }
                if name.startswith(_RESERVED_PREFIX)
                else {
                    "unversioned": True,
                    "message": "a nonempty database without MongoStore metadata cannot be adopted",
                }
            )
        raise StorageLayoutUpgradeRequiredError(
            {
                "protocol": {"expected": DECLARATION_PROTOCOL_VERSION, "actual": None},
                "declaration": {
                    "expected": "canonical registered declaration",
                    "actual": None,
                },
                "schema": schema,
            },
            remedy="rebuild",
        )

    @staticmethod
    def _validate_layout_names(layout: StorageLayout) -> None:
        owners: dict[str, type] = {}
        visited: set[type] = set()

        def visit(record: type) -> None:
            if record in visited:
                return
            visited.add(record)
            schema = resolve_schema(record)
            names = [collection_name_for(schema)]
            names.extend(spec.child.table_name for spec in schema.fields if spec.child is not None)
            for name in names:
                if name.startswith(_RESERVED_PREFIX):
                    raise ValueError(f"record {record.__name__} claims reserved MongoStore collection name {name!r}")
                previous = owners.get(name)
                if previous is not None and previous is not record:
                    raise ValueError(
                        f"records {previous.__name__} and {record.__name__} collide on physical collection name {name!r}"
                    )
                owners[name] = record
            for target in schema.referenced_classes():
                visit(target)

        for family in layout.families:
            for record in family.records:
                visit(record)
            dispatch_name = entry_dispatch_table_name(family.name) if len(family.records) > 1 else None
            if dispatch_name is not None:
                if dispatch_name in owners:
                    raise ValueError(
                        f"entry family {family.name!r} dispatch collection collides with a record collection"
                    )
                owners[dispatch_name] = family.family

    def _install_layout(self, layout: StorageLayout) -> None:
        self._layout = layout
        self._opened_declaration = declaration_json(layout)

    def _restamp_entry_schemas(self, layout: StorageLayout) -> None:
        """Re-stamp the metadata layout document's fingerprint after an additive upgrade."""
        self._database.database[METADATA_COLLECTION].update_one(
            {"_id": "layout"},
            {"$set": {"entry_schemas": schema_fingerprint_json(layout)}},
        )

    def ensure_collections(self, *classes: type) -> None:
        r"""Synchronously create or update record collections and their indexes.

        Collection DDL is lease-ordered: when this thread holds no writer lease
        (outside :meth:`transaction` and :meth:`save`) and there is DDL to do, a
        writer lease is taken for its duration, exactly as for a write.

        :param \*classes: Storable record classes whose collections should be
            prepared.  A configured multi-record family also prepares its
            dispatch collection.
        :return: None.
        :raises ValueError: If a requested physical name is reserved.
        :raises ~httk.store.storage_layout.StorageLayoutUpgradeRequiredError: If another
            instance has since upgraded the declaration (remedy ``"reopen"``).
        :raises ~httk.store.backend.mongo.leases.StoreLockedError: If an fsck or a declaration
            upgrade holds the store.
        :raises RuntimeError: If the metadata layout document lacks its generation counter.
        """
        self._require_identity_ownership()
        requested: list[tuple[str, dict[str, Any], list[Any]]] = []
        seen: set[str] = set()
        for cls in classes:
            schema = resolve_schema(cls)
            name = collection_name_for(schema)
            if name not in seen:
                requested.append(
                    (
                        name,
                        validator_for(schema, store_timestamps=self._store_timestamps),
                        index_specs_for(schema, store_timestamps=self._store_timestamps),
                    )
                )
                seen.add(name)
            # A source class's weak-link collections are created together with
            # its own collection, so the searcher and linked() may assume they
            # exist whenever the source collection does (SQL parity).
            for link in schema.links:
                if link.table_name in seen:
                    continue
                requested.append(
                    (
                        link.table_name,
                        link_validator_for(link, store_timestamps=self._store_timestamps),
                        link_index_specs_for(link),
                    )
                )
                seen.add(link.table_name)
            for family in self.layout.families:
                if cls not in family.records or len(family.records) < 2:
                    continue
                dispatch_name = entry_dispatch_table_name(family.name)
                if dispatch_name in seen:
                    continue
                requested.append(
                    (
                        dispatch_name,
                        dispatch_validator_for(family),
                        dispatch_index_specs(family),
                    )
                )
                seen.add(dispatch_name)

        pending = [item for item in requested if item[0] not in self._collections_ready]
        if not pending:
            return
        holds_lease = self._current_transaction() is not None or getattr(self._local, "writer_lease", None) is not None
        # Collection DDL rewrites validators (a dispatch validator's ``record``
        # enum in particular), so it is ordered against declaration upgrades
        # like any write: under a writer lease whose declaration must still be
        # the one this store opened.  A stale handle must never collMod a grown
        # family's dispatch validator back to its old, narrower enum.
        lease = None if holds_lease else self._acquire_writer_lease()
        try:
            self._prepare_collections(pending)
        finally:
            if lease is not None:
                lease.release()

    def _prepare_collections(self, requested: list[tuple[str, dict[str, Any], list[Any]]]) -> None:
        """Create or update the requested collections and indexes (caller holds a writer lease)."""
        for name, validator, specs in requested:
            if name in self._collections_ready:
                continue
            self._ensure_collection(name, validator)
            collection = self._database.database[name]
            models = []
            for spec in specs:
                options: dict[str, Any] = {"name": spec.name, "unique": spec.unique}
                if spec.partial_filter_expression is not None:
                    options["partialFilterExpression"] = spec.partial_filter_expression
                models.append(IndexModel(list(spec.keys), **options))
            if models:
                collection.create_indexes(models)
            self._collections_ready.add(name)

    def _ensure_collection(self, name: str, validator: dict[str, Any]) -> None:
        database = self._database.database
        try:
            database.create_collection(
                name,
                validator=validator,
                validationLevel="strict",
                validationAction="error",
            )
        except CollectionInvalid:
            database.command(
                "collMod",
                name,
                validator=validator,
                validationLevel="strict",
                validationAction="error",
            )

    def _initialize_store_timestamp_mark(self) -> None:
        """Derive the writable process-local timestamp mark from present collections."""
        if not self._store_timestamps or self._allow_clock_regression:
            self._store_timestamp_mark = None
            return
        maximum: int | None = None
        all_names = self._database.database.list_collection_names()
        record_names = {
            name for name in all_names if not name.startswith("system.") and not name.startswith(_RESERVED_PREFIX)
        }
        # Weak-link revisions carry the same clock as record saves, so a link
        # holding the newest timestamp must advance the reopen mark too (SQL
        # parity). Link docs have no ``_httk_role``, so scan them unfiltered.
        link_names = {name for name in all_names if name.startswith(_LINK_COLLECTION_PREFIX)}
        for name in record_names:
            document = self._database.database[name].find_one(
                {"_httk_role": {"$in": ["main", "dep"]}},
                {"store_timestamp": 1},
                sort=[("store_timestamp", -1)],
            )
            if document is not None and document.get("store_timestamp") is not None:
                value = int(document["store_timestamp"])
                maximum = value if maximum is None else max(maximum, value)
        for name in link_names:
            document = self._database.database[name].find_one(
                {}, {"store_timestamp": 1}, sort=[("store_timestamp", -1)]
            )
            if document is not None and document.get("store_timestamp") is not None:
                value = int(document["store_timestamp"])
                maximum = value if maximum is None else max(maximum, value)
        self._store_timestamp_mark = maximum

    def _capture_store_timestamp(self) -> int | None:
        """Capture one guarded store-unit timestamp for a save."""
        if not self._store_timestamps:
            return None
        return capture_store_timestamp(
            self._clock,
            self._store_timestamp_resolution,
            self._store_timestamp_mark,
            allow_clock_regression=self._allow_clock_regression,
            clock_regression_grace=self._clock_regression_grace,
        )

    def _advance_store_timestamp_mark(self, captured: int | None) -> None:
        self._store_timestamp_mark = advance_store_timestamp_mark(
            self._store_timestamp_mark,
            captured,
            allow_clock_regression=self._allow_clock_regression,
        )

    # ------------------------------------------------------------------ leases and transactions

    def _layout_generation(self) -> int:
        document = self._database.database[METADATA_COLLECTION].find_one({"_id": "layout"}, {"generation": 1})
        if document is None or not isinstance(document.get("generation"), int):
            raise RuntimeError("MongoStore metadata layout document is missing its generation counter")
        return int(document["generation"])

    def _observe_generation(self, generation: int) -> None:
        if generation != self._last_generation:
            self._identity._clear_identity_caches()
            self._last_generation = generation

    def _clear_identity_caches(self) -> None:
        """Clear cached hydrated records (a test and maintenance seam)."""
        self._identity._clear_identity_caches()

    def _transaction_stack(self) -> list[_TransactionState]:
        stack = getattr(self._local, "transactions", None)
        if stack is None:
            stack = []
            self._local.transactions = stack
        return typing.cast(list[_TransactionState], stack)

    def _current_transaction(self) -> _TransactionState | None:
        stack = self._transaction_stack()
        return stack[-1] if stack else None

    def _session_kwargs(self) -> dict[str, Any]:
        transaction = self._current_transaction()
        if transaction is not None:
            return {"session": transaction.session}
        session = getattr(self._local, "write_session", None)
        return {} if session is None else {"session": session}

    def _write_session(self) -> Any:
        transaction = self._current_transaction()
        return getattr(self._local, "write_session", None) if transaction is None else transaction.session

    @staticmethod
    def _has_label(error: BaseException, label: str) -> bool:
        return isinstance(error, PyMongoError) and error.has_error_label(label)

    def _is_protocol_duplicate(self, error: BaseException) -> bool:
        """Return whether a duplicate belongs to a MongoStore protocol index."""
        if not isinstance(error, DuplicateKeyError):
            return False
        details = error.details or {}
        key_pattern = details.get("keyPattern")
        if key_pattern == {"content_id": 1}:
            return True
        message = str(details.get("errmsg", error))
        return any(entry_dispatch_table_name(family.name) in message for family in self.layout.families)

    def _start_transaction(self, session: Any) -> None:
        session.start_transaction(
            read_concern=ReadConcern("majority"),
            write_concern=WriteConcern("majority", j=True),
        )

    def _commit(self, session: Any) -> None:
        while True:
            try:
                session.commit_transaction()
                return
            except BaseException as error:
                if self._has_label(error, "UnknownTransactionCommitResult"):
                    continue
                raise

    @staticmethod
    def _abort(session: Any) -> None:
        try:
            session.abort_transaction()
        except PyMongoError:
            pass

    def _publish_transaction_cache(self, transaction: _TransactionState) -> None:
        for (cls, sid), (obj, cache_instance) in transaction.pending.items():
            self._identity._remember(cls, sid, obj, cache_instance=cache_instance)
        self._failed_identities.difference_update(transaction.pending_sids)

    @contextlib.contextmanager
    def _transaction_scope(self) -> typing.Iterator[None]:
        self._require_identity_ownership()
        current = self._current_transaction()
        if current is not None:
            yield
            return
        if not self._database.supports_transactions:
            raise TransactionsUnavailableError("MongoDB transactions require a replica-set deployment")
        with self._write_lock:
            lease = self._acquire_writer_lease()
            try:
                self._observe_generation(lease.generation)
                with self._database.client.start_session(causal_consistency=True) as session:
                    transaction = _TransactionState(session, lease)
                    stack = self._transaction_stack()
                    stack.append(transaction)
                    try:
                        self._start_transaction(session)
                        yield
                        self._commit(session)
                    except BaseException:
                        self._abort(session)
                        self._identity._clear_identity_caches()
                        self._failed_identities.update(transaction.pending_sids)
                        raise
                    else:
                        self._advance_store_timestamp_mark(transaction.store_timestamp)
                        self._publish_transaction_cache(transaction)
                    finally:
                        stack.pop()
            finally:
                lease.release()

    def transaction(self) -> contextlib.AbstractContextManager[None]:
        """Return a flat explicit MongoDB transaction context manager.

        :return: A context that commits on normal exit and aborts on exception.
        :raises TransactionsUnavailableError: If this store is in degraded mode.
        """
        return self._transaction_scope()

    def clear_stale_lock(self) -> None:
        """Clear a stale fsck lease after verifying its owner is dead.

        This is an administrative operation. Clearing a merely slow fsck can
        corrupt the store because the lease protocol intentionally has no
        fencing token.

        :return: None.
        :raises StoreLockedError: If the fsck lease is still fresh.
        """
        clear_stale_lock(self._database.database)

    def fsck(
        self,
        *,
        repair: bool = True,
        collect_garbage: bool = True,
        repair_conflicts: bool = False,
        force: bool = False,
        clamp_future_timestamps: bool = False,
        known_types: tuple[type, ...] = (),
    ) -> FsckSummary:
        """Exclusively repair dispatch integrity and collect orphan dependencies.

        Main-role records and dispatch-addressed records are roots. Only
        dependency-role documents are eligible for collection; fsck never
        creates a dispatch for a dependency-role backing.

        :param repair: Insert missing dispatches for main multi-family backings.
        :param collect_garbage: Delete unmarked dependency documents.
        :param repair_conflicts: Delete invalid dispatch documents after reporting them.
        :param force: Administrative stale-lease override for the fsck handshake.
        :param clamp_future_timestamps: Clamp timestamps beyond the allowed future slack when repairing.
        :param known_types: Record classes that attribute ordinary collections
            from earlier store sessions, allowing a safe sweep after reopen.
        :return: An immutable :class:`~httk.store.backend.mongo.fsck.FsckSummary`.
        """
        from .fsck import run_fsck

        self._require_identity_ownership()

        return run_fsck(
            self,
            repair=repair,
            collect_garbage=collect_garbage,
            repair_conflicts=repair_conflicts,
            force=force,
            clamp_future_timestamps=clamp_future_timestamps,
            known_types=known_types,
        )

    def _refresh_writer_lease(self) -> None:
        transaction = self._current_transaction()
        if transaction is not None and transaction.lease is not None:
            transaction.lease.refresh_heartbeat()
            return
        lease = getattr(self._local, "writer_lease", None)
        if lease is not None:
            lease.refresh_heartbeat()

    # ------------------------------------------------------------------ object storage

    def save(
        self,
        obj: Any,
        *,
        as_record: type | None = None,
        id_series: str | None = None,
        alternative_of: str | None = None,
        alternative_kind: str | None = None,
        links: Mapping[str, object] | None = None,
    ) -> int:
        """Store an object graph and return its integer sid.

        Passing ``alternative_of`` (a stored main entry's id) with
        ``alternative_kind`` saves ``obj`` as a named ALTERNATIVE representation
        of that main: it copies the main's public ``id``, joins the main's
        alternative group, and hashes with the group identity folded in so its
        content never dedups onto the main. The main must live in ``obj``'s own
        backing collection and must itself be a main (not another alternative).

        :param obj: The object or projected domain object to store.
        :param as_record: An explicit alternate storage-record class.
        :param id_series: Override the configured entry-id series when an id must be minted.
        :param alternative_of: The stored main entry's id this record is an alternative of, if any.
        :param alternative_kind: The alternative kind name (grammar ``[a-z][a-z0-9_]*``); required with ``alternative_of``.
        :param links: Weak links to add after saving, mapping each declared link name to a target or iterable of targets; the save and every link are applied in one atomic transaction.
        :return: The stored sid.
        :raises TypeError: If ``obj`` is a cursor proxy.
        :raises ValueError: If exactly one of ``alternative_of``/``alternative_kind`` is given, the kind is malformed, or the named main is missing, in another backing collection, or itself an alternative.
        :raises ~httk.core.storage.StorageProjectionCycleError: If the projected graph cycles.
        :raises ~httk.store.store_common.EntryMetadataConflictError: If identity-excluded metadata conflicts.
        :raises ~httk.store.backend.mongo.database.TransactionConflictError: If the save keeps losing write conflicts against a concurrent transaction.
        """
        if (alternative_of is None) != (alternative_kind is None):
            raise ValueError("alternative_of and alternative_kind must be given together, or neither")
        if alternative_kind is not None and ALTERNATIVE_KIND_PATTERN.fullmatch(alternative_kind) is None:
            raise ValueError(
                f"invalid alternative_kind {alternative_kind!r}; expected {ALTERNATIVE_KIND_PATTERN.pattern}"
            )
        if not links:
            return self._save_graph(
                obj,
                as_record=as_record,
                replace_logical_id=None,
                id_series=id_series,
                alternative_of=alternative_of,
                alternative_kind=alternative_kind,
            )
        with self.transaction():
            sid = self._save_graph(
                obj,
                as_record=as_record,
                replace_logical_id=None,
                id_series=id_series,
                alternative_of=alternative_of,
                alternative_kind=alternative_kind,
            )
            self._apply_save_links(resolve_storage_record(obj, as_record=as_record), sid, links)
        return sid

    def _save_graph(
        self,
        obj: Any,
        *,
        as_record: type | None,
        replace_logical_id: int | None,
        id_series: str | None,
        replacement_entry_id: str | None = None,
        alternative_of: str | None = None,
        alternative_kind: str | None = None,
        replace_alt_group: int | None = None,
        replace_alt_kind: str | None = None,
        replace_alt_main_id: str | None = None,
    ) -> int:
        """Run one top-level save, optionally as a replacement carrying ``replace_logical_id``.

        Alternatives enter one of two ways: a fresh :meth:`save` resolves
        ``alternative_of``/``alternative_kind`` against the record's own backing
        collection; a :meth:`replace` of an alternative carries the predecessor's
        already-resolved ``replace_alt_*`` values so the replacement hashes with
        the same group extras as revision 1.
        """
        reject_cursor_proxy(obj)
        self._require_identity_ownership()
        record_type = resolve_storage_record(obj, as_record=as_record)
        alt_request = (
            _AlternativeRequest(
                alternative_of=alternative_of,
                alternative_kind=alternative_kind,
                replace_alt_group=replace_alt_group,
                replace_alt_kind=replace_alt_kind,
                replace_alt_main_id=replace_alt_main_id,
            )
            if alternative_of is not None or replace_alt_group is not None
            else None
        )
        if self._current_transaction() is not None:
            self._ensure_graph_collections(record_type)
            self._ensure_counter_collection()
            transaction = self._current_transaction()
            assert transaction is not None
            if not transaction.timestamp_initialized:
                transaction.store_timestamp = self._capture_store_timestamp()
                transaction.timestamp_initialized = True
            sid = self._save_once(
                record_type,
                obj,
                transaction.store_timestamp,
                replace_logical_id,
                id_series,
                replacement_entry_id,
                alt_request,
            )
            return sid
        with self._write_lock:
            lease = self._acquire_writer_lease()
            previous_lease = getattr(self._local, "writer_lease", None)
            self._local.writer_lease = lease
            try:
                self._observe_generation(lease.generation)
                self._ensure_graph_collections(record_type)
                self._ensure_counter_collection()
                captured = self._capture_store_timestamp()
                if not self._database.supports_transactions:
                    with self._database.client.start_session(causal_consistency=True) as session:
                        previous_session = getattr(self._local, "write_session", None)
                        self._local.write_session = session
                        try:
                            sid = self._save_once(
                                record_type,
                                obj,
                                captured,
                                replace_logical_id,
                                id_series,
                                replacement_entry_id,
                                alt_request,
                            )
                        finally:
                            self._local.write_session = previous_session
                else:
                    sid = self._save_implicit_transaction(
                        record_type,
                        obj,
                        typing.cast(WriterLease, lease),
                        captured,
                        replace_logical_id,
                        id_series,
                        replacement_entry_id,
                        alt_request,
                    )
                self._advance_store_timestamp_mark(captured)
                return sid
            finally:
                self._local.writer_lease = previous_lease
                lease.release()

    def _save_once(
        self,
        record_type: type,
        obj: Any,
        store_timestamp: int | None,
        replace_logical_id: int | None = None,
        id_series: str | None = None,
        replacement_entry_id: str | None = None,
        alt_request: "_AlternativeRequest | None" = None,
    ) -> int:
        projection = SaveProjection(store_timestamp=store_timestamp)
        self._projection_state(projection)
        alt_group: int | None
        alt_kind: str | None
        alt_main_id: str | None
        if alt_request is not None:
            if alt_request.alternative_of is not None:
                alt_group, alt_main_id = self._resolve_alternative_main(record_type, alt_request.alternative_of)
                alt_kind = alt_request.alternative_kind
            else:
                alt_group = alt_request.replace_alt_group
                alt_kind = alt_request.replace_alt_kind
                alt_main_id = alt_request.replace_alt_main_id
        else:
            alt_group = alt_kind = alt_main_id = None
        alt_extras: Mapping[str, object] | None = (
            {"alternative_of": alt_main_id, "alternative_kind": alt_kind} if alt_kind is not None else None
        )
        try:
            sid = self._save(
                record_type,
                obj,
                projection,
                "",
                top_level=True,
                replace_logical_id=replace_logical_id,
                id_series=id_series,
                replacement_entry_id=replacement_entry_id,
                alt_group=alt_group,
                alt_kind=alt_kind,
                alt_main_id=alt_main_id,
                alt_extras=alt_extras,
            )
            family = self._family_for_backing(record_type)
            if family is not None:
                # The dispatch hash is a top-level content id, so it carries the
                # same alternative-group extras as the dedup key below.
                self._save_entry_dispatch(
                    family,
                    record_type,
                    sid,
                    projection.content_id(record_type, obj, extras=alt_extras),
                )
            for (saved_type, identity), saved_sid in self._projection_sids(projection).items():
                source = self._projection_sources(projection)[(saved_type, identity)]
                # Entry-id minting augments the persisted ``f`` document; the
                # caller's frozen source still carries ``id=None``.  Do not
                # cache that pre-mint instance, so serving/fetching hydrates
                # the physical stored ids instead.
                self._remember(
                    saved_type,
                    saved_sid,
                    source,
                    cache_instance=type(source) is saved_type and saved_type not in self._entry_record_types,
                )
                self._failed_identities.discard((saved_type, identity))
            return sid
        except BaseException:
            self._failed_identities.update(self._projection_sources(projection))
            raise

    def _save_implicit_transaction(
        self,
        record_type: type,
        obj: Any,
        lease: WriterLease,
        store_timestamp: int | None,
        replace_logical_id: int | None = None,
        id_series: str | None = None,
        replacement_entry_id: str | None = None,
        alt_request: "_AlternativeRequest | None" = None,
    ) -> int:
        last_error: BaseException | None = None
        for attempt in range(_TRANSACTION_ATTEMPTS):
            with self._database.client.start_session(causal_consistency=True) as session:
                transaction = _TransactionState(session, lease)
                stack = self._transaction_stack()
                stack.append(transaction)
                try:
                    self._start_transaction(session)
                    sid = self._save_once(
                        record_type,
                        obj,
                        store_timestamp,
                        replace_logical_id,
                        id_series,
                        replacement_entry_id,
                        alt_request,
                    )
                    self._commit(session)
                except BaseException as error:
                    self._abort(session)
                    last_error = error
                    # A duplicate content-id within a transaction is retried as
                    # a fresh callback so its first lookup can observe the winner.
                    if self._is_protocol_duplicate(error) or self._has_label(error, "TransientTransactionError"):
                        time.sleep(min(0.01 * (2**attempt), 0.1))
                        continue
                    raise
                else:
                    self._publish_transaction_cache(transaction)
                    return sid
                finally:
                    stack.pop()
        assert last_error is not None
        self._identity._clear_identity_caches()
        raise TransactionConflictError(
            f"save gave up after {_TRANSACTION_ATTEMPTS} write conflicts with a concurrent transaction; "
            "retry once the competing transaction has ended"
        ) from last_error

    @staticmethod
    def _projection_state(projection: SaveProjection) -> None:
        projection.__dict__["mongo_sids"] = {}
        projection.__dict__["mongo_sources"] = {}

    @staticmethod
    def _projection_sids(projection: SaveProjection) -> dict[tuple[type, int], int]:
        return typing.cast(dict[tuple[type, int], int], projection.__dict__["mongo_sids"])

    @staticmethod
    def _projection_sources(projection: SaveProjection) -> dict[tuple[type, int], Any]:
        return typing.cast(dict[tuple[type, int], Any], projection.__dict__["mongo_sources"])

    def _ensure_counter_collection(self) -> None:
        try:
            self._database.database.create_collection(COUNTERS_COLLECTION)
        except CollectionInvalid:
            pass

    def _ensure_graph_collections(self, record_type: type) -> None:
        pending = [record_type]
        seen: set[type] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            self._known_record_types.add(current)
            schema = resolve_schema(current)
            self.ensure_collections(current)
            pending.extend(spec.target for spec in schema.fields if spec.target is not None)

    def _save(
        self,
        record_type: type,
        source: Any,
        projection: SaveProjection,
        path: str,
        *,
        top_level: bool,
        replace_logical_id: int | None = None,
        id_series: str | None = None,
        replacement_entry_id: str | None = None,
        alt_group: int | None = None,
        alt_kind: str | None = None,
        alt_main_id: str | None = None,
        alt_extras: Mapping[str, object] | None = None,
    ) -> int:
        self._refresh_writer_lease()
        key = (record_type, id(source))
        if key in projection.active:
            raise StorageProjectionCycleError(path, record_type)
        existing = self._projection_sids(projection).get(key)
        if existing is not None:
            return existing
        projection.active.add(key)
        self._projection_sources(projection)[key] = source
        try:
            return self._save_active(
                record_type,
                source,
                projection,
                path,
                top_level=top_level,
                replace_logical_id=replace_logical_id,
                id_series=id_series,
                replacement_entry_id=replacement_entry_id,
                alt_group=alt_group,
                alt_kind=alt_kind,
                alt_main_id=alt_main_id,
                alt_extras=alt_extras,
            )
        finally:
            projection.active.remove(key)

    def _save_active(
        self,
        record_type: type,
        source: Any,
        projection: SaveProjection,
        path: str,
        *,
        top_level: bool,
        replace_logical_id: int | None = None,
        id_series: str | None = None,
        replacement_entry_id: str | None = None,
        alt_group: int | None = None,
        alt_kind: str | None = None,
        alt_main_id: str | None = None,
        alt_extras: Mapping[str, object] | None = None,
    ) -> int:
        # The replacement lineage applies ONLY to the top-level parent document;
        # a nested record reached through references/ownership keeps its own-sid
        # lineage. A fresh document's logical_id is its own sid. Alternative-group
        # identity is likewise a top-level concept only: nested records never
        # carry a group id, kind, or extras.
        replacement = replace_logical_id if top_level else None
        alt_group = alt_group if top_level else None
        alt_kind = alt_kind if top_level else None
        alt_main_id = alt_main_id if top_level else None
        alt_extras = alt_extras if top_level else None
        schema = resolve_schema(record_type)
        projected = projection.projector(record_type, source)
        if record_type in self._entry_record_types:
            self._validate_projected_entry_ids(record_type, projected)
            if top_level and replacement_entry_id is not None:
                entry_id = projected.get("id")
                if entry_id is not None and entry_id != replacement_entry_id:
                    raise EntryIdConflictError(
                        collection_name_for(schema),
                        str(entry_id),
                        replacement,
                        replacement,
                    )
        validation_key = (record_type, id(source))
        if type(source) is record_type and validation_key not in projection.validated:
            validator = vars(record_type).get("__httk_validate__")
            if validator is not None:
                validator.__get__(None, record_type)(source)
            projection.validated.add(validation_key)

        content_key: str | None = None
        collection = self._database.database[collection_name_for(schema)]
        if schema.dedup == "content_id":
            content_key = projection.content_id(record_type, source, extras=alt_extras)
            found = collection.find_one({"content_id": content_key}, **self._session_kwargs())
            if found is not None:
                sid = int(found["_id"])
                if replacement is not None:
                    self._apply_replacement_collision(collection, sid, replacement)
                self._check_metadata(record_type, sid, source, projection)
                self._projection_sids(projection)[validation_key] = sid
                if top_level and found.get("_httk_role") == "dep":
                    collection.update_one(
                        {"_id": sid},
                        {"$set": {"_httk_role": "main"}},
                        **self._session_kwargs(),
                    )
                return sid

        checkpoint = len(projection.inserted)
        f_document = encode_record(
            schema,
            projected,
            source,
            record_type,
            lambda target, value, field: self._save(
                target,
                value,
                projection,
                self._field_path(path, field),
                top_level=False,
                id_series=id_series,
            ),
        )
        if schema.dedup == "by_value":
            query = self._by_value_query(schema, f_document)
            found = collection.find_one(query, {"_id": 1, "_httk_role": 1}, **self._session_kwargs())
            if found is not None:
                sid = int(found["_id"])
                self._discard_inserts(projection, checkpoint)
                if replacement is not None:
                    self._apply_replacement_collision(collection, sid, replacement)
                self._check_metadata(record_type, sid, source, projection)
                self._projection_sids(projection)[validation_key] = sid
                if top_level and found.get("_httk_role") == "dep":
                    collection.update_one(
                        {"_id": sid},
                        {"$set": {"_httk_role": "main"}},
                        **self._session_kwargs(),
                    )
                return sid

        sid = counter_next(self._database.database, schema.table_name, session=self._write_session())
        document: dict[str, Any] = {
            "_id": sid,
            "_httk_role": "main" if top_level else "dep",
            "f": f_document,
        }
        # The sid is allocated pre-insert, so the lineage is known directly: a
        # replacement copies its predecessor's, a fresh document uses its own sid.
        document["logical_id"] = replacement if replacement is not None else sid
        # alt_id is a replacement's/alternative's group (both carried in
        # alt_group), else the fresh main's own sid; alt_kind is absent for a
        # main.  Both are final pre-insert here (no write-after-insert).
        document["alt_id"] = alt_group if alt_group is not None else sid
        if alt_kind is not None:
            document["alt_kind"] = alt_kind
        if record_type in self._entry_record_types:
            self._prepare_entry_ids(
                record_type,
                f_document,
                lineage=int(document["logical_id"]),
                sid=sid,
                id_series=id_series,
                replacement_entry_id=replacement_entry_id if top_level else None,
                alt_group=alt_group,
                alt_kind=alt_kind,
                alt_main_id=alt_main_id,
            )
        if projection.store_timestamp is not None:
            document["store_timestamp"] = projection.store_timestamp
        if content_key is not None:
            document["content_id"] = content_key
        preflight_document(document, self._max_bson_size, record_type)
        if not self._database.supports_transactions:
            self._claim_entry_identity(record_type, document)
        try:
            collection.insert_one(document, **self._session_kwargs())
        except DuplicateKeyError as error:
            if "immutable_id" in str(error):
                raise EntryIdConflictError(collection.name, str(f_document.get("immutable_id")), None, sid) from error
            if not self._is_protocol_duplicate(error) or content_key is None:
                raise
            winner = collection.find_one({"content_id": content_key}, **self._session_kwargs())
            if winner is None:
                raise
            sid = int(winner["_id"])
            self._discard_inserts(projection, checkpoint)
            if replacement is not None:
                self._apply_replacement_collision(collection, sid, replacement)
            self._check_metadata(record_type, sid, source, projection)
            self._projection_sids(projection)[validation_key] = sid
            if top_level and winner.get("_httk_role") == "dep":
                collection.update_one(
                    {"_id": sid},
                    {"$set": {"_httk_role": "main"}},
                    **self._session_kwargs(),
                )
            return sid
        projection.inserted.append((record_type, sid))
        if self._database.supports_transactions:
            self._claim_entry_identity(record_type, document)
        self._projection_sids(projection)[validation_key] = sid
        return sid

    def _resolve_alternative_main(self, record_type: type, alternative_of: str) -> tuple[int, str]:
        """Resolve ``alternative_of`` to its ``(group id, main entry id)``.

        The main must live in ``record_type``'s own backing collection
        (alternatives share their main's backing) and must itself be a main, not
        another alternative.

        :param record_type: The alternative's record class, fixing the backing collection to search.
        :param alternative_of: The main entry's id this record is an alternative of.
        :return: The main's ``logical_id`` (the alternative group id) and its entry id.
        :raises ValueError: If the main is missing, lives in another family backing collection, or is itself an alternative.
        """
        collection = self._database.database[collection_name_for(resolve_schema(record_type))]
        # Alternatives copy their main's id, so an id names a whole group; the
        # group's MAIN is the one document with no alt_kind.
        main = collection.find_one(
            {"f.id": alternative_of, "alt_kind": {"$exists": False}},
            {"logical_id": 1},
            **self._session_kwargs(),
        )
        if main is not None:
            return int(main["logical_id"]), alternative_of
        # No main here: the id may exist only as an alternative (defensive; an
        # orphan without its main cannot arise through the public API), in a
        # sibling backing collection, or nowhere.
        if collection.find_one({"f.id": alternative_of}, {"_id": 1}, **self._session_kwargs()) is not None:
            raise ValueError(
                f"alternative_of {alternative_of!r} names an alternative, not a main; "
                "alternatives of alternatives are not allowed"
            )
        for sibling in self._entry_family_collections(record_type):
            if sibling.name == collection.name:
                continue
            if sibling.find_one({"f.id": alternative_of}, {"_id": 1}, **self._session_kwargs()) is not None:
                raise ValueError(
                    f"alternative_of {alternative_of!r} is stored in collection {sibling.name!r}, but an "
                    f"alternative must share its main's backing collection {collection.name!r}"
                )
        raise ValueError(f"alternative_of {alternative_of!r} names no entry in collection {collection.name!r}")

    def _prepare_entry_ids(
        self,
        record_type: type,
        values: dict[str, Any],
        *,
        lineage: int,
        sid: int,
        id_series: str | None,
        replacement_entry_id: str | None,
        alt_group: int | None = None,
        alt_kind: str | None = None,
        alt_main_id: str | None = None,
    ) -> None:
        """Validate ownership and mint entry ids for one parent document.

        An alternative copies its group main's id (``alt_main_id``) rather than
        minting, owns id-space per its alternative group (``alt_group``) instead
        of its own lineage, and stamps an alternative immutable id under
        ``alt_kind``. Mains keep the original per-lineage behaviour.
        """
        collection = self._database.database[collection_name_for(resolve_schema(record_type))]
        entry_id = values.get("id")
        immutable_id = values.get("immutable_id")
        # An alternative copies its main's id; a replacement copies its
        # predecessor's. Both forbid an explicit-but-different id on the record.
        forced_entry_id = replacement_entry_id if replacement_entry_id is not None else alt_main_id
        # Ownership is per alternative group, not per lineage: a fresh main's
        # group is its own sid, so it falls back to the lineage exactly as
        # before; an alternative/replacement carries its group id.
        group_id = alt_group if alt_group is not None else lineage
        if forced_entry_id is not None:
            if entry_id is None:
                entry_id = forced_entry_id
                values["id"] = entry_id
            elif entry_id != forced_entry_id:
                raise EntryIdConflictError(collection.name, str(entry_id), group_id, group_id)
        if entry_id is not None:
            for sibling in self._entry_family_collections(record_type):
                existing = sibling.find_one({"f.id": entry_id}, {"alt_id": 1}, **self._session_kwargs())
                if existing is None:
                    continue
                existing_alt_id = self._require_alt_id(existing, sibling.name)
                if sibling.name != collection.name or (int(existing["_id"]) != sid and existing_alt_id != group_id):
                    raise EntryIdConflictError(sibling.name, str(entry_id), existing_alt_id, group_id)
        else:
            if self._entry_ids is None:
                raise ValueError(
                    f"{record_type.__name__} has no id and MongoStore(entry_ids=EntryIdScheme(...)) was not declared; "
                    "pass an explicit id or declare a scheme"
                )
            base = self._entry_ids.base
            if self._entry_ids.type_in_base:
                base = f"{base}.{self._entry_record_types[record_type][0]}"
            entry_id = format_entry_id(
                base,
                id_series or self._entry_ids.series,
                self._entry_id_number(record_type, lineage),
            )
            values["id"] = entry_id
        # One alternative lineage per (alt_id, alt_kind): a fresh alternative
        # (its own new lineage) conflicts with ANY existing row of that (group,
        # kind); a replacement excludes its own predecessor lineage. This runs
        # before this document is inserted, so it never matches itself, and it
        # closes the hole an explicit ``<id>~<kind>~N`` immutable id would
        # otherwise punch through the incidental revision-1 collision.
        if alt_kind is not None:
            existing_lineage = collection.find_one(
                {
                    "alt_id": alt_group,
                    "alt_kind": alt_kind,
                    "logical_id": {"$ne": lineage},
                },
                {"logical_id": 1},
                **self._session_kwargs(),
            )
            if existing_lineage is not None:
                raise EntryIdConflictError(
                    collection.name,
                    str(entry_id),
                    int(existing_lineage["logical_id"]),
                    lineage,
                )
        if immutable_id is not None:
            for sibling in self._entry_family_collections(record_type):
                existing = sibling.find_one(
                    {"f.immutable_id": immutable_id},
                    {"_id": 1},
                    **self._session_kwargs(),
                )
                if existing is not None:
                    raise EntryIdConflictError(sibling.name, str(immutable_id), int(existing["_id"]), sid)
            return
        revision = 1 + collection.count_documents(
            {"logical_id": lineage, "_id": {"$ne": sid}}, **self._session_kwargs()
        )
        if alt_kind is not None:
            # Each alternative is its own lineage, so the per-logical_id revision
            # counter is correct here; the id namespace is the kind-qualified one.
            values["immutable_id"] = format_alternative_id(str(entry_id), alt_kind, revision)
        else:
            values["immutable_id"] = format_immutable_id(str(entry_id), revision)

    @staticmethod
    def _require_alt_id(document: Mapping[str, Any], collection_name: str) -> int:
        """Return a parent document's ``alt_id``, refusing pre-alternatives documents.

        :param document: A parent document that must carry the alternatives axis.
        :param collection_name: The reading collection, for the diagnostic.
        :return: The document's alternative-group identity.
        :raises RuntimeError: If ``alt_id`` is absent (the store predates the axis).
        """
        value = document.get("alt_id")
        if value is None:
            raise RuntimeError(
                f"record collection {collection_name!r} has a parent document without an alt_id; "
                "this store predates the alternatives axis; rebuild"
            )
        return int(value)

    @staticmethod
    def _validate_projected_entry_ids(record_type: type, projected: Mapping[str, object]) -> None:
        """Validate supplied entry ids before deduplication can write dependencies."""
        del record_type
        entry_id = projected.get("id")
        immutable_id = projected.get("immutable_id")
        if entry_id is not None:
            check_entry_id(cast(str, entry_id))
        if immutable_id is not None:
            check_immutable_id(cast(str, immutable_id))

    def _entry_family_collections(self, record_type: type) -> tuple[Any, ...]:
        """Return every backing collection sharing an enforced entry-id namespace."""
        family = self._family_for_backing(record_type)
        if family is None:
            return (self._database.database[collection_name_for(resolve_schema(record_type))],)
        return tuple(
            self._database.database[collection_name_for(resolve_schema(backing))] for backing in family.records
        )

    @staticmethod
    def _field_path(path: str, field: str) -> str:
        return f"{path}.{field}" if path else field

    @staticmethod
    def _by_value_query(schema: Any, f_document: Mapping[str, Any]) -> dict[str, Any]:
        query: dict[str, Any] = {}
        child_fields = {spec.field: spec for spec in schema.fields if spec.role == "child"}
        field_plans = {plan.field: plan for plan in document_fields_for(schema)}
        for key, value in f_document.items():
            if key not in child_fields:
                query[f"f.{key}"] = value
        for spec in schema.fields:
            if spec.role == "child" or not spec.optional:
                continue
            plan = field_plans[spec.field]
            for key in plan.keys:
                if key not in f_document:
                    query[f"f.{key}"] = None
        for spec in child_fields.values():
            if spec.optional and spec.field not in f_document:
                query[f"f.{spec.field}"] = {"$exists": False}
            elif spec.field in f_document:
                query[f"f.{spec.field}"] = {"$type": "array"}
        return query

    def _discard_inserts(self, projection: SaveProjection, checkpoint: int) -> None:
        if checkpoint == len(projection.inserted):
            return
        if self._current_transaction() is None:
            return
        sids = self._projection_sids(projection)
        for record_type, sid in reversed(projection.inserted[checkpoint:]):
            self._database.database[collection_name_for(resolve_schema(record_type))].delete_one(
                {"_id": sid}, **self._session_kwargs()
            )
            family = self._family_for_backing(record_type)
            if family is not None and family.definition_id is not None:
                backing = family.record_names[family.records.index(record_type)]
                owners = self._database.database[_IDENTITY_OWNERS]
                owners.delete_one(
                    {
                        "family": family.name,
                        "kind": "immutable_id",
                        "backing": backing,
                        "owner": sid,
                    },
                    **self._session_kwargs(),
                )
                collection = self._database.database[collection_name_for(resolve_schema(record_type))]
                if collection.find_one({"alt_id": sid}, {"_id": 1}, **self._session_kwargs()) is None:
                    owners.delete_one(
                        {
                            "family": family.name,
                            "kind": "id",
                            "backing": backing,
                            "owner": sid,
                        },
                        **self._session_kwargs(),
                    )
            for key, value in tuple(sids.items()):
                if value == sid and key[0] is record_type:
                    del sids[key]
        del projection.inserted[checkpoint:]

    def _apply_replacement_collision(self, collection: Any, hit_sid: int, predecessor_logical_id: int) -> None:
        """Enforce :meth:`replace`'s dedup collision policy against a deduplicated hit document.

        A replacement whose content deduplicates onto an existing document is an
        idempotent no-op when that document shares the predecessor's lineage, and
        an :class:`~httk.store.store_common.EntryReplacementError` when it belongs to a
        different one.

        :param collection: The record collection holding the deduplicated document.
        :param hit_sid: The sid of the deduplicated document.
        :param predecessor_logical_id: The predecessor's lineage identity.
        :raises ~httk.store.store_common.EntryReplacementError: If the hit belongs to a different lineage.
        """
        document = collection.find_one({"_id": hit_sid}, {"logical_id": 1}, **self._session_kwargs())
        if document is None:
            raise RuntimeError(
                f"record collection {collection.name!r} is missing the deduplicated document for sid {hit_sid}"
            )
        existing = int(document["logical_id"])
        if existing != predecessor_logical_id:
            raise EntryReplacementError(collection.name, predecessor_logical_id, existing)

    def _family_for_backing(self, record_type: type) -> Any:
        return next(
            (family for family in self.layout.families if record_type in family.records),
            None,
        )

    def _save_entry_dispatch(self, family: Any, backing: type, sid: int, key: str) -> None:
        if len(family.records) == 1:
            return
        collection = self._database.database[entry_dispatch_table_name(family.name)]
        record_name = family.record_names[family.records.index(backing)]
        existing = collection.find_one({"_id": key}, **self._session_kwargs())
        if existing is not None:
            if existing.get("record") == record_name and int(existing.get("sid", -1)) == sid:
                return
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} maps content_id {key!r} to a conflicting backing row"
            )
        try:
            collection.insert_one(
                {"_id": key, "record": record_name, "sid": sid},
                **self._session_kwargs(),
            )
            return
        except DuplicateKeyError:
            existing = collection.find_one({"_id": key}, **self._session_kwargs())
            if existing is not None:
                if existing.get("record") == record_name and int(existing.get("sid", -1)) == sid:
                    return
                raise EntryDispatchIntegrityError(
                    f"entry dispatch {family.name!r} maps content_id {key!r} to a conflicting backing row"
                ) from None
            owner = collection.find_one({"record": record_name, "sid": sid}, **self._session_kwargs())
            if owner is not None:
                raise EntryDispatchIntegrityError(
                    f"entry dispatch {family.name!r} already maps backing sid {sid} to content_id {owner['_id']!r}, "
                    f"not {key!r}"
                ) from None
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} declined content_id {key!r} without a discoverable conflicting row"
            ) from None

    # ------------------------------------------------------------------ reads

    def fetch[T](self, cls: type[T], sid: int, *, eager: bool = False) -> T:
        """Fetch and hydrate ``cls`` at ``sid``.

        The ``eager`` flag is accepted for backend transparency with
        :class:`~httk.store.backend.sql.store.SqlStore`; the Mongo document is fully in
        memory at read, so the returned record is always materialized and its
        values and semantics are identical either way.

        :param cls: The storable record class.
        :param sid: The integer sid.
        :param eager: Accepted for interface parity; a materialized record is always returned.
        :return: The hydrated record.
        :raises KeyError: If the record does not exist.
        """
        return typing.cast(T, self._fetch(cls, int(sid), _HydrationContext()))

    def fetch_many[T](self, cls: type[T], sids: Sequence[int], *, eager: bool = False) -> list[T]:
        """Fetch and hydrate ``cls`` at each sid in ``sids``.

        The ``eager`` flag is accepted for backend transparency; Mongo always
        returns materialized records (see :meth:`fetch`).

        :param cls: The storable record class.
        :param sids: The integer sids to fetch.
        :param eager: Accepted for interface parity; materialized records are always returned.
        :return: The hydrated records in ``sids`` order.
        :raises KeyError: If any record does not exist.
        """
        return [self.fetch(cls, sid) for sid in sids]

    def _fetch(self, cls: type, sid: int, context: _HydrationContext | None = None) -> Any:
        if context is None:
            context = _HydrationContext()
        key = (cls, int(sid))
        transaction = self._current_transaction()
        pending = None if transaction is None else transaction.pending.get(key)
        cached = None if pending is None or not pending[1] else pending[0]
        if cached is None:
            cached = self._identity._instances.get(key)
        if cached is not None:
            return cached
        schema = resolve_schema(cls)
        document = context.documents.get(key)
        if document is None:
            document = self._database.database[collection_name_for(schema)].find_one(
                {"_id": int(sid)}, **self._session_kwargs()
            )
        if document is None:
            raise KeyError(cls, int(sid))
        self._prefetch_references(schema, document, context)
        # A class declaring weak links is fetched as a cached thin subclass
        # carrying the store reference and its lineage id, so ``.links`` resolves
        # on the fetched record. Identity-cache/pending hits above return the
        # user's plain saved instance without ``.links`` (documented limitation).
        record_class = _bound_record_class(cls) if schema.links else None
        instance = decode_record(
            schema,
            document,
            lambda target, target_sid: self._fetch(target, target_sid, context),
            record_class=record_class,
        )
        if record_class is not None:
            object.__setattr__(instance, _BOUND_STORE, self)
            object.__setattr__(instance, _BOUND_LID, int(document["logical_id"]))
        self._remember(cls, int(sid), instance)
        return instance

    def _prefetch_references(self, schema: Any, document: Mapping[str, Any], context: _HydrationContext) -> None:
        targets: dict[type, set[int]] = {}
        embedded = document.get("f", {})
        for spec in schema.fields:
            if spec.target is None:
                continue
            if spec.role == "reference":
                sid = embedded.get(spec.columns[0].name)
                if sid is not None:
                    targets.setdefault(spec.target, set()).add(int(sid))
            elif spec.role == "child" and spec.child is not None:
                for element in embedded.get(spec.field, ()):
                    sid = element.get(spec.child.element_columns[0].name)
                    if sid is not None:
                        targets.setdefault(spec.target, set()).add(int(sid))
        cache = context.documents
        transaction = self._current_transaction()
        for target, sids in targets.items():
            missing = [
                sid
                for sid in sids
                if (target, sid) not in cache
                and (transaction is None or (target, sid) not in transaction.pending)
                and self._identity._instances.get((target, sid)) is None
            ]
            if not missing:
                continue
            target_schema = resolve_schema(target)
            for item in self._database.database[collection_name_for(target_schema)].find(
                {"_id": {"$in": missing}}, **self._session_kwargs()
            ):
                cache[(target, int(item["_id"]))] = item

    def fetch_by_content_id[T](self, cls: type[T], key: str, *, eager: bool = False) -> T | None:
        """Fetch a content-addressed record, or return ``None``.

        The ``eager`` flag is accepted for backend transparency; Mongo always
        returns a materialized record (see :meth:`fetch`).

        :param cls: The storable record class.
        :param key: The content identity.
        :param eager: Accepted for interface parity; a materialized record is always returned.
        :return: The hydrated record or ``None``.
        :raises ~httk.store.backend.schema.SchemaError: If ``cls`` is not content-id deduplicated.
        """
        schema = resolve_schema(cls)
        if schema.dedup != "content_id":
            raise SchemaError(
                f"{cls.__name__} has dedup policy {schema.dedup!r}; only classes with the "
                f"'content_id' policy have a content identity column"
            )
        document = self._database.database[collection_name_for(schema)].find_one(
            {"content_id": key}, {"_id": 1}, **self._session_kwargs()
        )
        return None if document is None else self.fetch(cls, int(document["_id"]))

    def fetch_entry(self, family_cls: type, content_id: str, *, eager: bool = False) -> object | None:
        """Fetch the concrete backing record for an entry-family identity.

        The ``eager`` flag is accepted for backend transparency; Mongo always
        returns a materialized record (see :meth:`fetch`).

        Only records stored top-level carry a dispatch document; a record stored
        only as a dependency of another record is resolved directly from its
        backing collection, exactly as in a single-backing family.

        :param family_cls: The configured entry-family class.
        :param content_id: The entry content identity.
        :param eager: Accepted for interface parity; a materialized record is always returned.
        :return: The backing record or ``None``.
        :raises ValueError: If the family is not configured.
        :raises ~httk.store.store_common.EntryDispatchIntegrityError: If dispatch and backing disagree.
        :raises ~httk.store.storage_layout.StorageLayoutUpgradeRequiredError: If the lookup misses or
            meets a dispatch document this handle cannot interpret because another instance has since
            upgraded the declaration (remedy ``"reopen"``).
        """
        family = next((item for item in self.layout.families if item.family is family_cls), None)
        if family is None:
            raise ValueError(f"{family_cls.__name__} is not a configured entry family in this MongoStore")
        if len(family.records) == 1:
            found_single = self.fetch_by_content_id(family.records[0], content_id)
            if found_single is None:
                # A stale single-backing handle cannot see an entry stored in a
                # backing appended since it was opened: say so, not None.
                self._raise_if_layout_moved()
            return found_single
        dispatch = self._database.database[entry_dispatch_table_name(family.name)]
        row = dispatch.find_one({"_id": content_id}, **self._session_kwargs())
        if row is None:
            for backing in family.records:
                found = self._database.database[collection_name_for(resolve_schema(backing))].find_one(
                    {"content_id": content_id}, {"_id": 1, "_httk_role": 1}, **self._session_kwargs()
                )
                if found is None:
                    continue
                if found.get("_httk_role") == "dep":
                    # Dependency-only documents carry no dispatch (fsck's rule):
                    # resolve directly, as a single-backing family does.
                    return self.fetch(backing, int(found["_id"]))
                self._raise_if_layout_moved()
                raise EntryDispatchIntegrityError(
                    f"entry dispatch {family.name!r} is missing for stored content_id {content_id!r}"
                )
            self._raise_if_layout_moved()
            return None
        record_name = row.get("record")
        try:
            index = family.record_names.index(record_name)
        except ValueError:
            # A handle opened before another instance appended record kinds
            # meets a backing name it does not know: that is staleness.
            self._raise_if_layout_moved()
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} names an unknown backing {record_name!r}"
            ) from None
        backing = family.records[index]
        sid = row.get("sid")
        if not isinstance(sid, int):
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} has an invalid sid for content_id {content_id!r}"
            )
        backing_document = self._database.database[collection_name_for(resolve_schema(backing))].find_one(
            {"_id": sid}, {"content_id": 1}, **self._session_kwargs()
        )
        backing_key = None if backing_document is None else backing_document.get("content_id")
        if backing_key != content_id:
            self._raise_if_layout_moved()
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} maps content_id {content_id!r} to backing sid {sid} "
                f"whose content_id is {backing_key!r}"
            )
        return self.fetch(backing, sid)

    def sid_of(self, obj: Any, *, as_record: type | None = None) -> int | None:
        """Return the sid known for ``obj``, using content lookup when allowed.

        :param obj: The object whose sid is requested.
        :param as_record: An explicit alternate record class.
        :return: The sid, or ``None``.
        """
        record_type = resolve_storage_record(obj, as_record=as_record)
        if (record_type, id(obj)) in self._failed_identities:
            return None
        transaction = self._current_transaction()
        if transaction is not None:
            pending = transaction.pending_sids.get((record_type, id(obj)))
            if pending is not None:
                return pending
        try:
            cached = self._identity._sids.get(obj, {}).get(record_type)
        except TypeError:
            cached = None
        if cached is None:
            cached = self._identity._sids_by_identity.get((record_type, id(obj)))
        if cached is not None:
            return cached
        schema = resolve_schema(record_type)
        if schema.dedup != "content_id":
            return None
        projection = SaveProjection()
        key = projection.content_id(record_type, obj)
        document = self._database.database[collection_name_for(schema)].find_one(
            {"content_id": key}, {"_id": 1}, **self._session_kwargs()
        )
        if document is None:
            return None
        sid = int(document["_id"])
        self._remember(record_type, sid, obj, cache_instance=type(obj) is record_type)
        return sid

    def replace(
        self,
        predecessor: Any,
        obj: Any,
        *,
        id_series: str | None = None,
        links: Mapping[str, object] | None = None,
    ) -> int:
        """Store ``obj`` as a logical replacement of ``predecessor`` and return its sid.

        The saved document copies ``predecessor``'s ``logical_id`` (its lineage
        identity) instead of starting a fresh one, so both documents share the
        lineage :meth:`history` walks. Nothing is updated or deleted: plain
        :meth:`fetch` and :meth:`searcher` queries keep returning both documents,
        and the lineage's latest document is simply the one with the highest sid.
        ``predecessor`` need not itself be the latest document of its lineage —
        replacing an already-replaced document is allowed and extends the same
        lineage. ``obj`` is saved through the ordinary :meth:`save` path, so its
        dedup policy, timestamp capture, identity caching and entry-family
        dispatch all behave exactly as they do there.

        If ``obj``'s content deduplicates onto an existing document (under the
        ``"content_id"`` or ``"by_value"`` policies), that document's lineage is
        compared with ``predecessor``'s: an equal lineage (including ``obj``
        equalling ``predecessor`` itself) is an idempotent no-op returning the
        existing sid, while a different lineage raises
        :class:`~httk.store.store_common.EntryReplacementError`.

        ``links`` optionally adds weak links to the replacement's lineage, as in
        :meth:`save`; the replace and every link are applied in one atomic
        transaction.

        :param predecessor: The stored instance being replaced; it must have been stored or fetched through this store.
        :param obj: The replacement object to store.
        :param id_series: Override the configured entry-id series when an id must be minted.
        :param links: Weak links to add, mapping each declared link name to a target or iterable of targets.
        :return: The stored replacement document's sid.
        :raises ValueError: If ``predecessor`` is not known to this store, or ``obj``'s record collection differs from ``predecessor``'s.
        :raises ~httk.store.store_common.EntryReplacementError: If ``obj`` deduplicates onto a document from a different lineage.
        """
        if links:
            with self.transaction():
                sid = self._replace_core(predecessor, obj, id_series)
                self._apply_save_links(resolve_storage_record(obj), sid, links)
            return sid
        return self._replace_core(predecessor, obj, id_series)

    def _replace_core(self, predecessor: Any, obj: Any, id_series: str | None) -> int:
        predecessor_sid = self.sid_of(predecessor)
        if predecessor_sid is None:
            raise ValueError(
                f"the {type(predecessor).__name__} instance has not been stored or fetched through this store"
            )
        predecessor_type = resolve_storage_record(predecessor)
        obj_type = resolve_storage_record(obj)
        predecessor_collection = collection_name_for(resolve_schema(predecessor_type))
        obj_collection = collection_name_for(resolve_schema(obj_type))
        if obj_collection != predecessor_collection:
            raise ValueError(
                f"cannot replace a record stored in collection {predecessor_collection!r} with a "
                f"{obj_type.__name__} record stored in collection {obj_collection!r}"
            )
        document = self._database.database[predecessor_collection].find_one(
            {"_id": predecessor_sid},
            {"logical_id": 1, "f.id": 1, "alt_id": 1, "alt_kind": 1},
            **self._session_kwargs(),
        )
        if document is None:
            raise RuntimeError(
                f"record collection {predecessor_collection!r} is missing the predecessor document for sid "
                f"{predecessor_sid}"
            )
        predecessor_logical_id = int(document["logical_id"])
        predecessor_alt_id = self._require_alt_id(document, predecessor_collection)
        predecessor_alt_kind = document.get("alt_kind")
        predecessor_entry_id = None
        alt_main_id = None
        if obj_type in self._entry_record_types:
            predecessor_entry_id = document.get("f", {}).get("id")
            if not isinstance(predecessor_entry_id, str):
                raise RuntimeError(
                    f"record collection {predecessor_collection!r} has no entry id on predecessor sid {predecessor_sid}"
                )
            if predecessor_alt_kind is not None:
                # An alternative copied its group main's id, so the predecessor's
                # own id is the group main's id: the replacement hashes with the
                # same group extras as revision 1.
                alt_main_id = predecessor_entry_id
        return self._save_graph(
            obj,
            as_record=None,
            replace_logical_id=predecessor_logical_id,
            id_series=id_series,
            replacement_entry_id=predecessor_entry_id,
            replace_alt_group=predecessor_alt_id,
            replace_alt_kind=(predecessor_alt_kind if isinstance(predecessor_alt_kind, str) else None),
            replace_alt_main_id=alt_main_id,
        )

    # ------------------------------------------------------------------ weak links

    def _link_spec(self, source_cls: type, name: str) -> LinkSpec:
        """Resolve the source class's declared weak link named ``name``."""
        schema = resolve_schema(source_cls)
        for spec in schema.links:
            if spec.name == name:
                return spec
        declared = ", ".join(link.name for link in schema.links) or "none"
        raise SchemaError(f"{source_cls.__name__} declares no weak link named {name!r} (declared links: {declared})")

    def _lid_from_sid(self, schema: TableSchema, sid: int) -> int:
        """The lineage id (``logical_id``) of the document ``sid`` in ``schema``'s collection."""
        document = self._database.database[collection_name_for(schema)].find_one(
            {"_id": int(sid)}, {"logical_id": 1}, **self._session_kwargs()
        )
        if document is None:
            raise ValueError(f"no {schema.cls.__name__} document exists for sid {sid}")
        return int(document["logical_id"])

    def _lid_of(self, cls: type, obj: Any) -> int:
        """The lineage id of ``obj`` in ``cls``'s collection; raises if ``obj`` is not stored."""
        sid = self.sid_of(obj, as_record=cls)
        if sid is None:
            raise ValueError(f"the {type(obj).__name__} instance has not been stored or fetched through this store")
        return self._lid_from_sid(resolve_schema(cls), sid)

    def _link_target_lid(self, spec: LinkSpec, target: Any, source_cls: type, name: str) -> int:
        """Validate ``target``'s type against ``spec`` and resolve its lineage id."""
        target_cls = resolve_storage_record(target)
        if collection_name_for(resolve_schema(target_cls)) != collection_name_for(resolve_schema(spec.target)):
            raise TypeError(
                f"weak link {name!r} on {source_cls.__name__} expects a {spec.target.__name__} target, "
                f"got {target_cls.__name__}"
            )
        return self._lid_of(target_cls, target)

    def _latest_link_by_lineage(self, spec: LinkSpec, source_lid: int, target_lid: int) -> dict[int, tuple[int, int]]:
        """Map each pair lineage (``logical_id``) to its latest ``(sid, retracted)``."""
        collection = self._database.database[spec.table_name]
        latest: dict[int, tuple[int, int]] = {}
        for row in collection.find(
            {"source_lid": source_lid, "target_lid": target_lid},
            {"logical_id": 1, "_id": 1, "retracted": 1},
            **self._session_kwargs(),
        ):
            lineage = int(row["logical_id"])
            sid, retracted = int(row["_id"]), int(row["retracted"])
            previous = latest.get(lineage)
            if previous is None or sid > previous[0]:
                latest[lineage] = (sid, retracted)
        return latest

    def _insert_link_row(
        self,
        spec: LinkSpec,
        *,
        logical_id: int | None,
        source_lid: int,
        target_lid: int,
        retracted: int,
        store_timestamp: int | None,
    ) -> None:
        """Append one link revision. ``logical_id=None`` founds a fresh own-sid lineage.

        Mongo mints the sid up front (unlike SQL's write-after-insert), so a
        fresh lineage simply sets ``logical_id`` to that sid before the insert.
        """
        sid = counter_next(self._database.database, spec.table_name, session=self._write_session())
        document: dict[str, Any] = {
            "_id": sid,
            "logical_id": sid if logical_id is None else logical_id,
            "source_lid": source_lid,
            "target_lid": target_lid,
            "retracted": retracted,
        }
        if store_timestamp is not None:
            document["store_timestamp"] = store_timestamp
        self._database.database[spec.table_name].insert_one(document, **self._session_kwargs())

    def _do_link(
        self,
        spec: LinkSpec,
        source_lid: int,
        target_lid: int,
        store_timestamp: int | None,
    ) -> None:
        """Idempotently assert the pair ``(source_lid, target_lid)`` is linked."""
        latest = self._latest_link_by_lineage(spec, source_lid, target_lid)
        if any(retracted == 0 for _sid, retracted in latest.values()):
            return  # some lineage is already live: no-op
        self._insert_link_row(
            spec,
            # All lineages retracted: revive the max-logical_id one. None founds a fresh lineage.
            logical_id=max(latest) if latest else None,
            source_lid=source_lid,
            target_lid=target_lid,
            retracted=0,
            store_timestamp=store_timestamp,
        )

    def _do_unlink(
        self,
        spec: LinkSpec,
        source_lid: int,
        target_lid: int,
        store_timestamp: int | None,
    ) -> None:
        """Retract every live lineage of the pair ``(source_lid, target_lid)``."""
        latest = self._latest_link_by_lineage(spec, source_lid, target_lid)
        for lineage, (_sid, retracted) in latest.items():
            if retracted == 0:
                self._insert_link_row(
                    spec,
                    logical_id=lineage,
                    source_lid=source_lid,
                    target_lid=target_lid,
                    retracted=1,
                    store_timestamp=store_timestamp,
                )

    def _apply_save_links(self, source_cls: type, source_sid: int, links: Mapping[str, object]) -> None:
        """Add each declared link for the just-saved document, inside the open save transaction."""
        transaction = self._current_transaction()
        assert transaction is not None, "_apply_save_links runs inside an open transaction"
        source_lid = self._lid_from_sid(resolve_schema(source_cls), source_sid)
        for name, targets in links.items():
            spec = self._link_spec(source_cls, name)
            for target in _iter_link_targets(targets):
                target_lid = self._link_target_lid(spec, target, source_cls, name)
                self._do_link(spec, source_lid, target_lid, transaction.store_timestamp)

    def _link_timestamp(self) -> int | None:
        """Capture (once per transaction) the store timestamp for a link revision."""
        transaction = self._current_transaction()
        assert transaction is not None, "_link_timestamp runs inside an open transaction"
        if not transaction.timestamp_initialized:
            transaction.store_timestamp = self._capture_store_timestamp()
            transaction.timestamp_initialized = True
        return transaction.store_timestamp

    def link(self, source: Any, name: str, target: Any) -> None:
        """Assert a weak link named ``name`` from ``source``'s lineage to ``target``'s lineage.

        Weak links live in a dedicated append-only link collection and bind
        *lineages*: both endpoints always resolve to their latest revision. The
        operation is idempotent per the pair ``(source lineage, target
        lineage)`` — a live pair is a no-op, a retracted pair is revived, an
        absent pair founds a fresh link lineage — and duplicate pair lineages
        (from concurrent writers, which Mongo transactions do not exclude) are
        tolerated: the pair is live if any lineage is live.

        :param source: The stored source instance declaring the link.
        :param name: The declared weak-link name.
        :param target: The stored target instance whose type matches the link declaration.
        :return: None.
        :raises httk.store.backend.schema.SchemaError: If ``source``'s class declares no link named ``name``.
        :raises TypeError: If ``target``'s type does not match the link's declared target.
        :raises ValueError: If ``source`` or ``target`` is not stored in this store.
        :raises TransactionsUnavailableError: If this store is in degraded mode.
        """
        source_cls = resolve_storage_record(source)
        spec = self._link_spec(source_cls, name)
        if self._current_transaction() is None:
            self._ensure_graph_collections(source_cls)
            self._ensure_counter_collection()
        with self.transaction():
            store_timestamp = self._link_timestamp()
            source_lid = self._lid_of(source_cls, source)
            target_lid = self._link_target_lid(spec, target, source_cls, name)
            self._do_link(spec, source_lid, target_lid, store_timestamp)

    def unlink(self, source: Any, name: str, target: Any) -> None:
        """Retract the weak link named ``name`` from ``source``'s lineage to ``target``'s lineage.

        Retracts every live lineage of the pair (duplicate-tolerant); an absent
        or already-retracted pair is a no-op. Retraction appends a revision — no
        document is deleted, and history/``as_of`` still see the earlier live rows.

        :param source: The stored source instance declaring the link.
        :param name: The declared weak-link name.
        :param target: The stored target instance whose type matches the link declaration.
        :return: None.
        :raises httk.store.backend.schema.SchemaError: If ``source``'s class declares no link named ``name``.
        :raises TypeError: If ``target``'s type does not match the link's declared target.
        :raises ValueError: If ``source`` or ``target`` is not stored in this store.
        :raises TransactionsUnavailableError: If this store is in degraded mode.
        """
        source_cls = resolve_storage_record(source)
        spec = self._link_spec(source_cls, name)
        if self._current_transaction() is None:
            self._ensure_graph_collections(source_cls)
            self._ensure_counter_collection()
        with self.transaction():
            store_timestamp = self._link_timestamp()
            source_lid = self._lid_of(source_cls, source)
            target_lid = self._link_target_lid(spec, target, source_cls, name)
            self._do_unlink(spec, source_lid, target_lid, store_timestamp)

    def linked(self, source: Any, name: str, *, eager: bool = False) -> tuple[Any, ...]:
        """Return the latest revisions of the targets currently linked from ``source`` under ``name``.

        Targets are deduplicated by lineage and ordered by first-link order (the
        link lineage's root ``logical_id``, ascending; stable across
        retract+relink). Each returned object is the latest revision of its
        target lineage. ``eager`` is accepted for interface parity with the SQL
        store; a Mongo fetch is always fully materialized.

        :param source: The stored source instance declaring the link.
        :param name: The declared weak-link name.
        :param eager: Accepted for interface parity; a materialized record is always returned.
        :return: The linked targets' latest revisions, deduplicated and ordered by first-link order.
        :raises httk.store.backend.schema.SchemaError: If ``source``'s class declares no link named ``name``.
        :raises ValueError: If ``source`` is not stored in this store.
        """
        source_cls = resolve_storage_record(source)
        spec = self._link_spec(source_cls, name)
        source_lid = self._lid_of(source_cls, source)
        return self._linked_by_lid(spec, source_lid)

    def _linked_by_lid(self, spec: LinkSpec, source_lid: int, *, as_of: object = None) -> tuple[Any, ...]:
        """Latest live-linked targets of ``source_lid`` under ``spec`` (the query core of :meth:`linked`).

        Shared by :meth:`linked` (which resolves the source lineage id first),
        the fetched record's ``.links`` accessor (which already holds the
        lineage id), and a link-set search output (which passes its searcher's
        ``as_of``). Targets are deduplicated by lineage and ordered by
        first-link order; each is the latest revision of its target lineage.
        When ``as_of`` is given, both the link documents and each target's
        latest revision are bounded by that cutoff — the same semantics the
        predicate path applies to link rows and targets.
        """
        as_of_units = (
            None if as_of is None else ns_operand_to_store_units(as_of, cast(int, self._store_timestamp_resolution))
        )
        collection = self._database.database[spec.table_name]
        link_filter: dict[str, Any] = {"source_lid": source_lid}
        if as_of_units is not None:
            link_filter["store_timestamp"] = {"$lte": as_of_units}
        latest: dict[int, tuple[int, int, int]] = {}  # logical_id -> (sid, retracted, target_lid)
        for row in collection.find(
            link_filter,
            {"target_lid": 1, "logical_id": 1, "_id": 1, "retracted": 1},
            **self._session_kwargs(),
        ):
            lineage = int(row["logical_id"])
            sid, retracted, target_lid = (
                int(row["_id"]),
                int(row["retracted"]),
                int(row["target_lid"]),
            )
            previous = latest.get(lineage)
            if previous is None or sid > previous[0]:
                latest[lineage] = (sid, retracted, target_lid)
        live_root: dict[int, int] = {}  # target_lid -> smallest live lineage root
        for lineage, (_sid, retracted, target_lid) in latest.items():
            if retracted != 0:
                continue
            existing = live_root.get(target_lid)
            if existing is None or lineage < existing:
                live_root[target_lid] = lineage
        ordered_target_lids = [
            target_lid for target_lid, _root in sorted(live_root.items(), key=lambda item: (item[1], item[0]))
        ]
        if not ordered_target_lids:
            return ()
        target_schema = resolve_schema(spec.target)
        target_collection = self._database.database[collection_name_for(target_schema)]
        target_sids: list[int] = []
        for target_lid in ordered_target_lids:
            target_filter: dict[str, Any] = {"logical_id": target_lid}
            if as_of_units is not None:
                target_filter["store_timestamp"] = {"$lte": as_of_units}
            document = target_collection.find_one(
                target_filter,
                {"_id": 1},
                sort=[("_id", -1)],
                **self._session_kwargs(),
            )
            if document is not None:  # a missing target is a dangling link; fsck reports it
                target_sids.append(int(document["_id"]))
        return tuple(self.fetch_many(spec.target, target_sids))

    def history(self, obj: Any) -> tuple[Any, ...]:
        """Return every record in ``obj``'s replacement lineage, oldest first.

        The lineage is the set of documents sharing ``obj``'s ``logical_id`` — the
        fresh record that started it and every :meth:`replace` of it — ordered by
        sid ascending (the fresh record first, the latest replacement last).
        Records are reconstructed through the same machinery as :meth:`fetch`.

        :param obj: A stored instance whose lineage to walk; it must have been stored or fetched through this store.
        :return: The lineage's records ordered by ascending sid.
        :raises ValueError: If ``obj`` is not known to this store.
        """
        obj_sid = self.sid_of(obj)
        if obj_sid is None:
            raise ValueError(f"the {type(obj).__name__} instance has not been stored or fetched through this store")
        record_type = resolve_storage_record(obj)
        collection = self._database.database[collection_name_for(resolve_schema(record_type))]
        document = collection.find_one({"_id": obj_sid}, {"logical_id": 1}, **self._session_kwargs())
        if document is None:
            raise RuntimeError(f"record collection {collection.name!r} is missing the document for sid {obj_sid}")
        # A lineage is one logical_id; alternatives own distinct logical_ids,
        # so a main and its alternatives never share a history() walk.
        logical_id = int(document["logical_id"])
        sids = [
            int(item["_id"])
            for item in collection.find({"logical_id": logical_id}, {"_id": 1}, **self._session_kwargs()).sort("_id", 1)
        ]
        return tuple(self.fetch(record_type, sid) for sid in sids)

    def searcher(
        self,
        *,
        as_of: object = None,
        only_latest: bool = False,
        only_main_alt: bool = True,
    ) -> Any:
        """Return a Mongo searcher bound to this store's read path.

        Queries use the active transaction session when one is open, so they
        see that transaction's uncommitted writes, and object outputs hydrate
        through :meth:`fetch`, preserving the identity-cache contract.

        :param as_of: Optional historic cutoff in canonical timestamp form.
        :param only_latest: Whether root variables are restricted to the latest document of each
            ``logical_id`` lineage by sid (bounded by ``as_of`` when given). Reference/child
            scopes stay unfiltered so pinned references may still resolve replaced documents.
        :param only_main_alt: Whether root variables are restricted to mains (``alt_kind`` absent),
            hiding named alternatives. Defaults to ``True``; pass ``False`` to reveal alternatives.
        :return: A new MongoDB searcher bound to this store.
        """
        if as_of is not None:
            if not self._store_timestamps:
                raise ValueError("as_of queries require MongoStore(store_timestamps=True)")
            ns_operand_to_store_units(as_of, self._store_timestamp_resolution)
        from .searcher import MongoSearcher

        return MongoSearcher(self, as_of=as_of, only_latest=only_latest, only_main_alt=only_main_alt)

    def stored_property_plan(self, family: type) -> Any:
        """Return the wire-form Mongo stored-property plan for one configured entry family.

        This method is a serving edge: like the entry providers, it serves the
        family's definition in OPTIMADE wire form (``served_form()``), so a
        prefixed family whose ``__httk_stored_properties__`` keys are served
        names plans correctly. ``served_form()`` is idempotent for standard
        (unprefixed) families, so their plans are byte-identical.

        :param family: The logical entry-family class to plan.
        :return: The validated Mongo stored-property plan.
        """
        from .stored_properties import stored_property_mongo_plan

        layout = next((item for item in self.entry_layout if item.family is family), None)
        return stored_property_mongo_plan(self, family, served=None if layout is None else _served_definition(layout))

    def referring(self, cls: type, *, field: str, to: Any, eager: bool = False) -> list[Any]:
        """Return records whose reference field points at ``to``, ordered by sid.

        The ``eager`` flag is accepted for backend transparency; Mongo always
        returns materialized records (see :meth:`fetch`).

        :param cls: The referring record class.
        :param field: The reference field.
        :param to: The stored target instance.
        :param eager: Accepted for interface parity; materialized records are always returned.
        :return: Matching records ordered by sid.
        :raises ~httk.store.backend.schema.SchemaError: If the field or target class is incompatible.
        :raises ValueError: If ``to`` is not stored or fetched here.
        """
        schema = resolve_schema(cls)
        spec = schema.field(field)
        if spec.role != "reference":
            raise SchemaError(f"{cls.__name__}.{field} is not a reference field (its role is {spec.role!r})")
        assert spec.target is not None
        if not isinstance(to, spec.target):
            raise SchemaError(f"{cls.__name__}.{field} references {spec.target.__name__}, not {type(to).__name__}")
        sid = self.sid_of(to)
        if sid is None:
            raise ValueError(f"the {type(to).__name__} instance has not been stored or fetched through this store")
        path = f"f.{spec.columns[0].name}"
        collection = self._database.database[collection_name_for(schema)]
        return [
            self.fetch(cls, int(document["_id"]))
            for document in collection.find({path: sid}, {"_id": 1}, **self._session_kwargs()).sort("_id", 1)
        ]

    def _remember(self, cls: type, sid: int, obj: Any, *, cache_instance: bool = True) -> None:
        transaction = self._current_transaction()
        if transaction is None:
            self._identity._remember(cls, sid, obj, cache_instance=cache_instance)
            return
        transaction.pending[(cls, sid)] = (obj, cache_instance)
        transaction.pending_sids[(cls, id(obj))] = sid

    # ------------------------------------------------------------------ metadata comparison

    def _check_metadata(self, record_type: type, sid: int, source: Any, projection: SaveProjection) -> None:
        plan = _metadata_plan(record_type)
        if plan is not None:
            self._check_metadata_at(record_type, sid, source, projection, record_type.__name__, plan)
        if record_type in self._entry_record_types:
            document = self._database.database[collection_name_for(resolve_schema(record_type))].find_one(
                {"_id": sid}, **self._session_kwargs()
            )
            assert document is not None
            self._claim_entry_identity(record_type, document)

    def _metadata_parent_document(
        self, record_type: type, sid: int, plan: Any, projection: SaveProjection
    ) -> Mapping[str, Any]:
        key = (record_type, int(sid))
        cached = projection.metadata_rows.get(key)
        if cached is not None:
            return cached
        paths: dict[str, int] = {}
        for spec in (*plan.skipped_specs, *plan.skipped_nested, *plan.descend_specs):
            if spec.role == "child":
                paths[f"f.{spec.field}"] = 1
            else:
                for column in spec.columns:
                    paths[f"f.{column.name}"] = 1
        collection = self._database.database[collection_name_for(resolve_schema(record_type))]
        document = collection.find_one({"_id": int(sid)}, paths, **self._session_kwargs())
        if document is None:
            raise KeyError(record_type, int(sid))
        projection.metadata_rows[key] = document
        return document

    @staticmethod
    def _metadata_scalar_equal(left: Any, right: Any) -> bool:
        if isinstance(left, list | tuple) or isinstance(right, list | tuple):
            return (
                type(left) is type(right)
                and len(left) == len(right)
                and all(MongoStore._metadata_scalar_equal(a, b) for a, b in zip(left, right, strict=True))
            )
        if isinstance(left, datetime.datetime) and isinstance(right, datetime.datetime):
            if (left.utcoffset() is None) != (right.utcoffset() is None):
                return False
            if left.utcoffset() is not None:
                return left.astimezone(datetime.UTC) == right.astimezone(datetime.UTC)
        return bool(left == right)

    def _metadata_value(self, spec: Any, document: Mapping[str, Any]) -> Any:
        embedded = document.get("f", {})
        if spec.role == "child":
            if spec.optional and spec.field not in embedded:
                return None
            entries = embedded.get(spec.field, [])
            assert spec.child is not None
            if spec.shape is not None:
                return FracVector(
                    [
                        decode_fracvector_exact(item[f"{spec.field}_exact"], 1, spec.shape.cols).to_fractions()[0]
                        for item in entries
                    ]
                )
            if spec.target is not None:
                return [int(item[spec.child.element_columns[0].name]) for item in entries]
            if spec.codec_name is not None:
                codec = codec_named(spec.codec_name)
                values = [
                    codec.decode(tuple(item[column.name] for column in spec.child.element_columns)) for item in entries
                ]
            else:
                values = [item[spec.child.element_columns[0].name] for item in entries]
            return tuple(values) if typing.get_origin(spec.python_type) is tuple else values
        embedded = document.get("f", {})
        if spec.role == "scalar":
            return embedded.get(spec.columns[0].name)
        if spec.role == "encoded":
            parts = tuple(embedded.get(column.name) for column in spec.columns)
            return None if all(part is None for part in parts) else codec_named(spec.codec_name).decode(parts)
        if spec.role == "fixed_array":
            exact = embedded.get(f"{spec.field}_exact")
            return None if exact is None else decode_fracvector_exact(exact, spec.shape.rows, spec.shape.cols)
        return embedded.get(spec.columns[0].name)

    def _check_metadata_at(
        self,
        record_type: type,
        sid: int,
        source: Any,
        projection: SaveProjection,
        path: str,
        plan: Any,
    ) -> None:
        schema = resolve_schema(record_type)
        document = self._metadata_parent_document(record_type, sid, plan, projection)
        values = projection.projector(record_type, source)
        skipped = {spec.field for spec in plan.skipped_specs}
        nested = {spec.field for spec in plan.skipped_nested}
        descend = {spec.field for spec in plan.descend_specs}
        for spec in schema.fields:
            if spec.derived:
                continue
            field_path = self._field_path(path, spec.field)
            if spec.field in skipped:
                incoming = values[spec.field]
                if (
                    record_type in self._entry_record_types
                    and spec.field in {"id", "immutable_id"}
                    and incoming is None
                ):
                    continue
                stored = self._metadata_value(spec, document)
                if not self._metadata_scalar_equal(incoming, stored):
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {field_path}: stored {stored!r}, received {incoming!r}"
                    )
            elif spec.field in nested:
                self._check_metadata_nested(
                    schema,
                    document,
                    sid,
                    spec,
                    values[spec.field],
                    projection,
                    field_path,
                    True,
                )
            elif spec.field in descend:
                self._check_metadata_nested(
                    schema,
                    document,
                    sid,
                    spec,
                    values[spec.field],
                    projection,
                    field_path,
                    False,
                )

    def _check_metadata_nested(
        self,
        schema: Any,
        document: Mapping[str, Any],
        sid: int,
        spec: Any,
        incoming: Any,
        projection: SaveProjection,
        path: str,
        compare_content: bool,
    ) -> None:
        stored = self._metadata_value(spec, document)
        if spec.role == "reference":
            if incoming is None or stored is None:
                if incoming is not None or stored is not None:
                    if compare_content:
                        existing = None if stored is None else self._fetch(spec.target, int(stored))
                        raise EntryMetadataConflictError(
                            f"metadata conflict for {path}: stored {existing!r}, received {incoming!r}"
                        )
                    raise EntryMetadataConflictError(f"metadata conflict for {path}")
                return
            self._check_metadata_target(spec.target, int(stored), incoming, projection, path, compare_content)
            return
        if spec.target is None:
            if not self._metadata_scalar_equal(incoming, stored):
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {stored!r}, received {incoming!r}"
                )
            return
        if incoming is None or stored is None:
            if incoming is not stored:
                if compare_content:
                    existing = None if stored is None else [self._fetch(spec.target, int(item)) for item in stored]
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {path}: stored {existing!r}, received {incoming!r}"
                    )
                raise EntryMetadataConflictError(f"metadata conflict for {path}")
            return
        if len(incoming) != len(stored):
            if compare_content:
                existing = [self._fetch(spec.target, int(item)) for item in stored]
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {existing!r}, received {incoming!r}"
                )
            raise EntryMetadataConflictError(f"metadata conflict for {path}")
        for index, (incoming_item, stored_sid) in enumerate(zip(incoming, stored, strict=True)):
            self._check_metadata_target(
                spec.target,
                int(stored_sid),
                incoming_item,
                projection,
                f"{path}[{index}]",
                compare_content,
            )

    def _check_metadata_target(
        self,
        record_type: type,
        sid: int,
        source: Any,
        projection: SaveProjection,
        path: str,
        compare_content: bool,
    ) -> None:
        if compare_content:
            collection = self._database.database[collection_name_for(resolve_schema(record_type))]
            stored = collection.find_one({"_id": sid}, {"content_id": 1}, **self._session_kwargs())
            if resolve_schema(record_type).dedup == "content_id":
                stored_key = None if stored is None else stored.get("content_id")
            else:
                stored_key = projection.content_id(record_type, self._fetch(record_type, sid))
            incoming_key = projection.content_id(record_type, source)
            if incoming_key != stored_key:
                existing = self._fetch(record_type, sid)
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {existing!r}, received {source!r}"
                )
        plan = _metadata_plan(record_type)
        if plan is not None:
            self._check_metadata_at(record_type, sid, source, projection, path, plan)
