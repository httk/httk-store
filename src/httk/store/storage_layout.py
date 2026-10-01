"""Backend-neutral store declaration machinery shared by storage backends.

This module owns the logical entry-family declaration, its canonical JSON
encoding, and trust-on-reopen validation.  Physical names and backend-specific
layout validation belong to each storage backend.
"""

import dataclasses
import json
import sys
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, Literal, cast

from httk.core import EntryTypeDefinition, PropertyDefinition
from httk.core.entry_ids import parse_alternative_id, parse_entry_id, parse_immutable_id
from httk.core.register import (
    entry_family_info,
    entry_record_info,
    known_entry_families,
    known_entry_records,
    resolve_entry_family,
    resolve_entry_record,
)
from httk.core.storage import IdentitySkip, Indexed, Unique, storage_identity_name

from httk.store.backend.schema import ChildTableSpec, ColumnSpec, FieldSpec, SchemaError, TableSchema, resolve_schema

__all__ = [
    "ADDITIVE_DECLARATION_UPGRADE_HINT",
    "ADDITIVE_UPGRADE_HINT",
    "DECLARATION_PROTOCOL_VERSION",
    "ENTRY_ID_OFFSETS_KEY",
    "AdditiveUpgradePlan",
    "DeclarationUpgradePlan",
    "EntryFamilyDeclaration",
    "EntryFamilyLayout",
    "EntryLayoutBindingError",
    "EntryRecordDeclaration",
    "StorageLayout",
    "StorageLayoutUpgradeRequiredError",
    "classify_declaration_upgrade",
    "classify_schema_upgrade",
    "declaration_is_superseded",
    "declaration_json",
    "entry_id_number",
    "entry_id_offsets_json",
    "family_entry_type_definition",
    "next_entry_id_offset",
    "normalize_entry_families",
    "normalize_entry_records",
    "normalize_entry_types",
    "parse_entry_id_offsets",
    "schema_fingerprint_diff",
    "schema_fingerprint_json",
    "validate_entry_id_fields",
]

DECLARATION_PROTOCOL_VERSION: Final = "2"
"""The current backend-neutral declaration protocol.

The value is the major generation only, compared for strict equality on reopen
and never parsed. Bump it to "3" solely on a breaking change to the declaration
protocol that an existing store could not be reopened against.
"""

ADDITIVE_UPGRADE_HINT: Final = (
    "the schema difference is purely additive (new nullable columns / lazily created tables); "
    "reopen with upgrade=True to apply it"
)
"""The reopen hint appended when an additive-only schema mismatch is not applied."""

ADDITIVE_DECLARATION_UPGRADE_HINT: Final = (
    "the declaration change is purely additive (record kinds appended to existing families and/or new "
    "families); reopen with upgrade=True to apply it"
)
"""The reopen hint appended when an additive declaration change is not applied."""

ENTRY_ID_OFFSETS_KEY: Final = "entry_id_offsets"
"""The optional store-metadata key holding the per-family entry-id numbering offsets.

Its value is the canonical JSON object ``{family name: offset}`` (sorted keys,
compact separators) with positive integer offsets.  Only an additive declaration
upgrade writes it; an absent key means every offset is zero.
"""


class StorageLayoutUpgradeRequiredError(RuntimeError):
    """A database does not exactly implement the current persisted store layout.

    ``diff`` is immutable and JSON-shaped.  Its top-level keys are stable
    categories (currently ``protocol``, ``declaration`` and ``schema``), so a
    caller can present a precise upgrade diagnostic without parsing
    the human-readable exception message.  The ``declaration`` category maps
    named aspect keys (``metadata_keys``, ``store_timestamps``,
    ``write_profile``, ``entry_declaration``) to their own diagnostics, so
    several independent declaration mismatches are reported together; the
    exception message names the mismatched aspects.

    ``remedy`` is the machine-readable action a caller should take, so a tool
    (``httk collect --upgrade``, say) never has to parse ``hint``:

    - ``"upgrade"`` — the difference is additive (or an additive upgrade was
      interrupted); reopen with the target declaration and ``upgrade=True``;
    - ``"rebuild"`` — the difference cannot be applied in place; rebuild the store;
    - ``"reopen"`` — this handle is out of date or was opened with mismatching
      options (another instance upgraded the layout, a concurrent upgrade won
      with a different layout, or ``store_timestamps``/write profile differ);
      reopen it with the right declaration and options;
    - ``"retry"`` — a transient conflict (the store was locked by writers, or a
      concurrent write conflicted with the upgrade's claim); retry the same call.

    SQL stores set it on every raise; ``None`` means the raising backend does
    not classify the difference (currently the Mongo store).

    :param diff: The immutable JSON-shaped category-keyed difference.
    :param hint: An optional remediation appended to the message and exposed as
        :attr:`hint` (e.g. that a purely additive schema change can be applied
        with ``upgrade=True``).
    :param remedy: The machine-readable remedy, exposed as :attr:`remedy`.
    """

    def __init__(
        self,
        diff: Mapping[str, object],
        *,
        hint: str | None = None,
        remedy: Literal["upgrade", "rebuild", "reopen", "retry"] | None = None,
    ) -> None:
        frozen = _freeze_mapping(diff)
        self.diff: Mapping[str, object] = frozen
        self.hint: str | None = hint
        self.remedy: Literal["upgrade", "rebuild", "reopen", "retry"] | None = remedy
        categories = ", ".join(frozen) or "unknown layout difference"
        details = frozen.get("declaration")
        detail_names = f": {', '.join(details)}" if isinstance(details, Mapping) else ""
        hint_text = f"; {hint}" if hint else ""
        super().__init__(f"Store layout upgrade is required ({categories}{detail_names}){hint_text}")


class EntryLayoutBindingError(ValueError):
    """A persisted application-owned layout needs explicit Python class bindings."""


@dataclasses.dataclass(frozen=True)
class EntryRecordDeclaration:
    """Bind one stable store-local name to a concrete record class.

    :param name: Stable record identity persisted in the store declaration.
    :param record: Concrete frozen dataclass used for storage and hydration.
    :param definition_id: Optional entry-type definition IRI described by the record.
    """

    name: str
    record: type
    definition_id: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, label="entry record name")
        if not isinstance(self.record, type):
            raise TypeError("entry record must be a class")
        params = getattr(self.record, "__dataclass_params__", None)
        if not dataclasses.is_dataclass(self.record) or params is None or not params.frozen:
            raise TypeError("entry record must be a frozen dataclass class")
        _validate_optional_definition_id(self.definition_id)


@dataclasses.dataclass(frozen=True)
class EntryFamilyDeclaration:
    """Declare one application-owned entry family without global registration.

    Explicit declarations provide stable persistence identities directly to a
    store.  They are intended for application-private families which should
    not participate in plugin discovery.  The same declaration must be
    supplied whenever such a store is reopened.

    :param name: Stable family identity persisted in the store declaration.
    :param family: Logical entry-family class exposed through ``entry_layout``.
    :param records: Ordered concrete record declarations belonging to the family.
    :param definition_id: Optional entry-type definition IRI for the family.
    """

    name: str
    family: type
    records: tuple[EntryRecordDeclaration, ...]
    definition_id: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, label="entry family name")
        if not isinstance(self.family, type):
            raise TypeError("entry family must be a class")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("entry family records must be a nonempty tuple")
        if any(not isinstance(record, EntryRecordDeclaration) for record in self.records):
            raise TypeError("entry family records must contain EntryRecordDeclaration values")
        if len({record.name for record in self.records}) != len(self.records):
            raise ValueError("entry family repeats a record name")
        if len({record.record for record in self.records}) != len(self.records):
            raise ValueError("entry family repeats a record class")
        _validate_optional_definition_id(self.definition_id)


@dataclasses.dataclass(frozen=True)
class EntryFamilyLayout:
    """One immutable configured entry family and its concrete records."""

    name: str
    family: type
    definition_id: str | None
    record_names: tuple[str, ...]
    records: tuple[type, ...]
    record_definition_ids: tuple[str | None, ...]


@dataclasses.dataclass(frozen=True)
class StorageLayout:
    """The immutable normalized entry declaration of an initialized store."""

    protocol_version: str
    families: tuple[EntryFamilyLayout, ...]

    @property
    def entry_records(self) -> Mapping[type, tuple[type, ...]]:
        """Configured family classes mapped to their ordered concrete record classes."""
        return MappingProxyType({family.family: family.records for family in self.families})

    @property
    def declaration(self) -> Mapping[str, tuple[str, ...]]:
        """Configured stable family names mapped to their ordered stable record names."""
        return MappingProxyType({family.name: family.record_names for family in self.families})


def validate_entry_id_fields(layout: StorageLayout) -> None:
    """Require entry-id fields on every backing of a defined entry family."""
    required = (
        ("id", "id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)"),
        (
            "immutable_id",
            "immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)",
        ),
    )
    for family in layout.families:
        if family.definition_id is None:
            continue
        for record in family.records:
            schema = resolve_schema(record)
            hints = typing.get_type_hints(record, include_extras=True)
            fields = {item.name: item for item in dataclasses.fields(record)}
            for name, declaration in required:
                try:
                    spec = schema.field(name)
                except SchemaError:
                    spec = None
                valid = (
                    spec is not None
                    and spec.role == "scalar"
                    and spec.python_type is str
                    and len(spec.columns) == 1
                    and (name != "id" or (_has_indexed(hints.get(name)) and not _has_unique(hints.get(name))))
                    and (name != "immutable_id" or _has_unique(hints.get(name)))
                    and _is_optional_string(hints.get(name))
                    and _has_identity_skip(hints.get(name))
                    and fields.get(name) is not None
                    and fields[name].default is None
                    and fields[name].compare is False
                )
                if not valid:
                    raise SchemaError(
                        f"entry record {record.__name__} in family {family.name!r} must declare {declaration}"
                    )


def _is_optional_string(annotation: object) -> bool:
    """Whether an annotated field's value type is exactly ``str | None``."""
    if typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation == (str | None)


def _has_indexed(annotation: object) -> bool:
    """Whether an annotation carries the ``Indexed`` marker."""
    return any(isinstance(marker, Indexed) for marker in _annotation_metadata(annotation))


def _has_unique(annotation: object) -> bool:
    """Whether an annotation carries the ``Unique`` marker."""
    return any(isinstance(marker, Unique) for marker in _annotation_metadata(annotation))


def _has_identity_skip(annotation: object) -> bool:
    """Whether an annotation carries the ``IdentitySkip`` marker."""
    return any(isinstance(marker, IdentitySkip) for marker in _annotation_metadata(annotation))


def _annotation_metadata(annotation: object) -> tuple[object, ...]:
    """Return the metadata carried by an ``Annotated`` declaration."""
    if typing.get_origin(annotation) is not typing.Annotated:
        return ()
    return tuple(typing.get_args(annotation)[1:])


def normalize_entry_records(entry_records: Mapping[type, type | tuple[type, ...]]) -> StorageLayout:
    """Validate an explicit class declaration and replace it with stable registry names.

    Registry aliases are rejected rather than selected arbitrarily: a
    persistent declaration must have exactly one stable spelling for every
    supplied class.
    """
    if not isinstance(entry_records, Mapping):
        raise TypeError("entry_records must be a mapping from entry-family classes to record classes")
    declarations: list[EntryFamilyDeclaration] = []
    for family, supplied_records in entry_records.items():
        if not isinstance(family, type):
            raise TypeError("entry_records keys must be entry-family classes")
        family_name = _registered_family_name(family)
        records: tuple[type, ...]
        if isinstance(supplied_records, type):
            records = (supplied_records,)
        elif isinstance(supplied_records, tuple):
            records = supplied_records
        else:
            raise TypeError(f"entry_records[{family.__name__}] must be a record class or a tuple of record classes")
        if not records:
            raise ValueError(f"entry_records[{family.__name__}] cannot be an empty tuple")
        if any(not isinstance(record, type) for record in records):
            raise TypeError(f"entry_records[{family.__name__}] contains a non-class record")
        if len(set(records)) != len(records):
            raise ValueError(f"entry_records[{family.__name__}] repeats a record class")
        record_declarations: list[EntryRecordDeclaration] = []
        for record in records:
            record_name = _registered_record_name(record)
            _, registered_family_name, definition_id = entry_record_info(record_name)
            if registered_family_name is None:
                raise ValueError(
                    f"entry record {record_name!r} has no registered family and cannot be used in a family store"
                )
            if registered_family_name != family_name:
                raise ValueError(
                    f"entry record {record.__name__} belongs to registered family {registered_family_name!r}, "
                    f"not {family_name!r}"
                )
            record_declarations.append(
                EntryRecordDeclaration(name=record_name, record=record, definition_id=definition_id)
            )
        _, family_definition_id = entry_family_info(family_name)
        declarations.append(
            EntryFamilyDeclaration(
                name=family_name,
                family=family,
                records=tuple(record_declarations),
                definition_id=family_definition_id,
            )
        )
    return _normalize_entry_families(declarations, explicit=False)


def normalize_entry_families(entry_families: Sequence[EntryFamilyDeclaration]) -> StorageLayout:
    """Validate application-owned entry declarations and build a store layout.

    Unlike :func:`normalize_entry_records`, this path does not require the
    family or record classes to be globally registered.  Stable names and
    optional definition identities are supplied by the application itself.

    :param entry_families: Explicit family declarations in any order.
    :return: The immutable normalized storage layout.
    :raises TypeError: If the declaration container or its members are invalid.
    :raises ValueError: If names, classes, definitions, or storage schemas conflict.
    """
    if not isinstance(entry_families, Sequence) or isinstance(entry_families, str | bytes):
        raise TypeError("entry_families must be a sequence of EntryFamilyDeclaration values")
    if any(not isinstance(item, EntryFamilyDeclaration) for item in entry_families):
        raise TypeError("entry_families must contain EntryFamilyDeclaration values")
    return _normalize_entry_families(entry_families, explicit=True)


def normalize_entry_types(records: Sequence[type]) -> StorageLayout:
    """Build a family declaration from decorated application record classes.

    ``records`` is the short declaration used by the beginner-facing store
    API.  Each class supplies its stable ``__httk_entry_name__`` and inherits
    the family ``type`` and ``definition_id`` from a core entry-record base.
    Registered record classes referenced by those records are included
    recursively, so a record containing (for example) a structure also makes
    the structure family available to a serving provider.

    :param records: Decorated frozen entry-record classes, or registered core records such as ``Run``, to serve.
    :return: The normalized storage layout.
    :raises TypeError: If ``records`` is not a sequence of classes.
    :raises ValueError: If a class is neither decorated nor registered, has incomplete or conflicting entry
        metadata, or two families containing explicitly listed
        classes would serve the same entry type.
    """
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise TypeError("records must be a sequence of entry-record classes")
    if not records:
        raise ValueError("records cannot be empty")
    if any(not isinstance(record, type) for record in records):
        raise TypeError("records must contain entry-record classes")
    if len(set(records)) != len(records):
        raise ValueError("records repeats an entry-record class")

    declarations: dict[str, EntryFamilyDeclaration] = {}
    decorated_groups: dict[tuple[str, str], list[tuple[str, type]]] = {}
    pending = [(record, "registered") for record in records]
    listed = set(records)
    visited: set[type] = set()
    while pending:
        record, mode = pending.pop(0)
        if record in visited:
            continue
        visited.add(record)
        decorated_name = vars(record).get("__httk_entry_name__")
        if mode == "decorated" or decorated_name is not None:
            name = decorated_name
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError(f"{record.__name__} must declare a nonempty __httk_entry_name__")
            registered_name = None
        else:
            if mode == "private":
                _queue_referenced_records(record, pending)
                continue
            try:
                registered_name = _registered_record_name(record)
            except ValueError:
                if record not in listed:
                    raise
                raise ValueError(
                    f"{record.__name__} is neither decorated nor registered: decorate it with "
                    "httk.core.entry_record or register it with httk.core.register_entry_record"
                ) from None
            name = registered_name
        family_name = getattr(record, "type", None)
        definition_id = getattr(record, "definition_id", None)
        if registered_name is not None:
            _, family_name, record_definition_id = entry_record_info(registered_name)
            if family_name is None:
                raise ValueError(f"registered entry record {registered_name!r} has no entry family")
            family_definition_id = entry_family_info(family_name)[1]
            definition_id = family_definition_id
        else:
            record_definition_id = definition_id
        if not isinstance(family_name, str) or not family_name.strip() or family_name != family_name.strip():
            raise ValueError(f"{record.__name__}.type must be a nonempty entry-family name")
        if definition_id is not None and (not isinstance(definition_id, str) or not definition_id.strip()):
            raise ValueError(f"{record.__name__}.definition_id must be a nonempty string or None")
        if registered_name is None:
            if not isinstance(definition_id, str):
                raise ValueError(f"{record.__name__}.definition_id must be a nonempty string")
            decorated_groups.setdefault((family_name, definition_id), []).append((name, record))
            _queue_referenced_records(record, pending)
            continue
        else:
            try:
                family_reference, registered_definition = entry_family_info(family_name)
                family = resolve_entry_family(family_name)
            except ValueError as error:
                raise ValueError(f"{record.__name__}.type {family_name!r} is not a registered entry family") from error
            if not isinstance(family_reference, str):  # pragma: no cover - registry validates this
                raise TypeError(f"entry family {family_name!r} has an invalid registry reference")
        if definition_id != registered_definition:
            raise ValueError(
                f"{record.__name__}.definition_id {definition_id!r} does not match family "
                f"{family_name!r} definition {registered_definition!r}"
            )
        declaration = declarations.get(family_name)
        record_declaration = EntryRecordDeclaration(name=name, record=record, definition_id=record_definition_id)
        if declaration is None:
            declarations[family_name] = EntryFamilyDeclaration(
                name=family_name,
                family=family,
                records=(record_declaration,),
                definition_id=definition_id,
            )
        else:
            if declaration.family is not family or declaration.definition_id != definition_id:
                raise ValueError(f"entry record {name!r} conflicts with family {family_name!r}")
            declarations[family_name] = dataclasses.replace(
                declaration, records=declaration.records + (record_declaration,)
            )

        _queue_referenced_records(record, pending)

    entry_type_definitions: dict[str, str] = {}
    for (entry_type, definition_id), group in decorated_groups.items():
        previous_definition = entry_type_definitions.get(entry_type)
        if previous_definition is not None and previous_definition != definition_id:
            raise ValueError(f"decorated records use conflicting definitions for entry type {entry_type!r}")
        entry_type_definitions[entry_type] = definition_id
        family_name = f"__httk_{entry_type}"
        if family_name in declarations or family_name in known_entry_families():
            raise ValueError(f"application entry family name {family_name!r} conflicts with an existing family")
        family = _application_entry_family(family_name, entry_type, definition_id, tuple(record for _, record in group))
        declarations[family_name] = EntryFamilyDeclaration(
            name=family_name,
            family=family,
            records=tuple(
                EntryRecordDeclaration(name=name, record=record, definition_id=definition_id) for name, record in group
            ),
            definition_id=definition_id,
        )

    served: dict[str, str] = {}
    for declaration in declarations.values():
        if not any(item.record in listed for item in declaration.records):
            continue
        entry_type = getattr(declaration.family, "type", declaration.name)
        if entry_type in served:
            raise ValueError(
                f"entry families {served[entry_type]!r} and {declaration.name!r} both serve entry type "
                f"{entry_type!r}; serve only one of them (for example a DataEntryRecord subclass instead of DataRecord)"
            )
        served[entry_type] = declaration.name
    return _normalize_entry_families(tuple(declarations.values()), explicit=True)


def _queue_referenced_records(record: type, pending: list[tuple[type, str]]) -> None:
    """Queue decorated and registered records reachable from ``record``."""
    for target in resolve_schema(record).referenced_classes():
        if vars(target).get("__httk_entry_name__") is not None:
            pending.append((target, "decorated"))
            continue
        try:
            target_name = _registered_record_name(target)
        except ValueError:
            pending.append((target, "private"))
        else:
            _, target_family, _ = entry_record_info(target_name)
            pending.append(
                (resolve_entry_record(target_name), "registered") if target_family is not None else (target, "private")
            )


def family_entry_type_definition(family: EntryFamilyLayout) -> EntryTypeDefinition:
    """Return the internal entry-type definition a configured family serves.

    A family class that defines ``entry_type_definition()`` is authoritative and
    its result is returned unchanged. Otherwise the family's registered
    definition is extended with the union of the property definitions its
    record backings declare in ``__httk_property_definitions__``, so a typed
    backing (for example a record with a native total-energy column) serves its
    property next to generic backings of the same family. Backings without
    ``__httk_property_definitions__`` contribute nothing, names already in the
    registered definition are left to it, and a family whose backings add
    nothing gets the registered definition itself. The result is the internal
    form; serve it through ``served_form()``.

    :param family: The configured entry family whose definition is derived.
    :return: The family's internal entry-type definition.
    :raises TypeError: If the family's ``entry_type_definition()`` does not
        return an :class:`~httk.core.EntryTypeDefinition`, or a backing declares
        malformed property metadata.
    :raises ValueError: If the family has neither a definition id nor
        ``entry_type_definition()``, or two backings declare different
        definitions under one property name.
    """
    factory = getattr(family.family, "entry_type_definition", None)
    if callable(factory):
        definition = factory()
        if not isinstance(definition, EntryTypeDefinition):
            raise TypeError(f"{family.family.__name__}.entry_type_definition() must return EntryTypeDefinition")
        return definition
    if family.definition_id is None:
        raise ValueError(f"entry family {family.name!r} has no entry-type definition")
    from httk.core import load_entry_type_definition

    base = load_entry_type_definition(family.definition_id)
    contributions = [
        (record, f"{record.__name__}.__httk_property_definitions__", declared)
        for record in family.records
        if (declared := getattr(record, "__httk_property_definitions__", None)) is not None
    ]
    properties = _record_property_union(base, contributions)
    return base.extended(properties) if properties else base


def _record_property_union(
    base: EntryTypeDefinition, contributions: Sequence[tuple[type, str, object]]
) -> dict[str, PropertyDefinition]:
    """Return the property definitions records add to ``base``, rejecting disagreements.

    Each contribution is ``(record, source, properties)``: the declaring record,
    a diagnostic label, and its name-to-definition mapping (read only through
    the ``Mapping`` interface).  Names already in ``base`` are skipped; one name
    declared with two different definitions raises :class:`ValueError` naming
    both records.
    """
    properties: dict[str, PropertyDefinition] = {}
    owners: dict[str, type] = {}
    for record, source, declared in contributions:
        if not isinstance(declared, Mapping):
            raise TypeError(f"{source} is not a properties mapping")
        for property_name, property_definition in cast(Mapping[object, object], declared).items():
            if not isinstance(property_name, str) or not isinstance(property_definition, PropertyDefinition):
                raise TypeError(f"{source} has invalid property metadata")
            if property_name in base.properties:
                continue
            previous = properties.get(property_name)
            if previous is not None and previous != property_definition:
                raise ValueError(
                    f"records {owners[property_name].__name__} and {record.__name__} disagree about "
                    f"property {property_name!r}"
                )
            properties[property_name] = property_definition
            owners.setdefault(property_name, record)
    return properties


def _application_entry_family(name: str, entry_type: str, definition_id: str, records: tuple[type, ...]) -> type:
    """Create the local logical family shared by decorated records of one type."""
    from httk.core import load_entry_type_definition

    base = load_entry_type_definition(definition_id)
    contributions: list[tuple[type, str, object]] = []
    for record in records:
        factory = getattr(record, "entry_type_definition", None)
        if not callable(factory):
            raise TypeError(f"{record.__name__} must provide entry_type_definition()")
        definition_properties = getattr(factory(), "properties", None)
        if not isinstance(definition_properties, Mapping):
            raise TypeError(f"{record.__name__}.entry_type_definition() has no properties mapping")
        contributions.append((record, f"{record.__name__}.entry_type_definition()", definition_properties))
    extended = base.extended(_record_property_union(base, contributions))

    def entry_type_definition(cls: type) -> object:
        return extended

    family = type(
        "_ApplicationEntryFamily",
        (),
        {
            "__module__": __name__,
            "__httk_entry_name__": name,
            "type": entry_type,
            "definition_id": definition_id,
            "entry_type_definition": classmethod(entry_type_definition),
        },
    )
    return family


def _normalize_entry_families(declarations: Sequence[EntryFamilyDeclaration], *, explicit: bool) -> StorageLayout:
    entries: list[EntryFamilyLayout] = []
    for declaration in declarations:
        if explicit:
            _reject_registry_conflicts(declaration)
        for record in declaration.records:
            schema = resolve_schema(record.record)
            if schema.dedup != "content_id":
                raise ValueError(
                    f"configured entry record {record.record.__name__} must use "
                    f"dedup='content_id', got {schema.dedup!r}"
                )
        entries.append(
            EntryFamilyLayout(
                name=declaration.name,
                family=declaration.family,
                definition_id=declaration.definition_id,
                record_names=tuple(record.name for record in declaration.records),
                records=tuple(record.record for record in declaration.records),
                record_definition_ids=tuple(record.definition_id for record in declaration.records),
            )
        )
    return _storage_layout(entries)


def _merge_storage_layouts(*layouts: StorageLayout) -> StorageLayout:
    return _storage_layout([family for layout in layouts for family in layout.families])


def _storage_layout(entries: list[EntryFamilyLayout]) -> StorageLayout:
    entries.sort(key=lambda entry: entry.name)
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("entry declaration repeats a family name")
    if len({entry.family for entry in entries}) != len(entries):
        raise ValueError("entry declaration repeats a family class")
    record_names = [name for entry in entries for name in entry.record_names]
    if len(set(record_names)) != len(record_names):
        raise ValueError("entry declaration repeats a record name")
    records = [record for entry in entries for record in entry.records]
    if len(set(records)) != len(records):
        raise ValueError("entry declaration repeats a record class")
    return StorageLayout(DECLARATION_PROTOCOL_VERSION, tuple(entries))


def declaration_json(layout: StorageLayout) -> str:
    """Serialize a normalized declaration in its exact deterministic persisted form."""
    document = {
        "families": [
            {
                "definition_id": family.definition_id,
                "family": family.name,
                "records": [
                    {"definition_id": definition_id, "record": name}
                    for name, definition_id in zip(family.record_names, family.record_definition_ids, strict=True)
                ],
            }
            for family in layout.families
        ],
        "format": 2,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _walk_closure(layout: StorageLayout, visit: Callable[[TableSchema], None]) -> None:
    """Invoke ``visit`` once per resolved schema across the declared closure.

    Each declared record class and the transitive closure of its referenced
    storable classes is resolved and passed to ``visit`` exactly once.

    :param layout: The normalized storage layout to walk.
    :param visit: The callback invoked with each distinct resolved schema.
    :return: None.
    """
    seen: set[type] = set()

    def descend(record: type) -> None:
        if record in seen:
            return
        seen.add(record)
        schema = resolve_schema(record)
        visit(schema)
        for target in schema.referenced_classes():
            descend(target)

    for family in layout.families:
        for record in family.records:
            descend(record)


def schema_fingerprint_json(layout: StorageLayout) -> str:
    """Serialize the resolved per-table schema of ``layout`` in deterministic form.

    The fingerprint covers every declared record class plus the transitive
    closure of referenced storable classes, resolved through
    :func:`~httk.store.backend.schema.resolve_schema`.  It captures what determines
    the on-disk layout, the stored value encoding, and the *content identity* of
    each table — the logical identity name, dedup policy, composite indexes,
    relationship links, and per-field roles, codecs, shapes, columns, child
    tables, identity participation, and list-vs-tuple container — so that
    reopening a store whose record classes changed is rejected up front.

    A code move or rename is safe only when the record pins an explicit
    :attr:`~httk.core.storage.StorageInfo.identity_name` (every shipped httk
    record does); without a pin the qualified class name *is* the content
    identity, so the move changes ``content_id`` and the store correctly
    refuses to open.  ``cls`` and ``python_type`` themselves are excluded — the
    identity name and resolved columns capture everything the store depends on.

    :param layout: The normalized storage layout to fingerprint.
    :return: A deterministic ``sort_keys`` JSON document describing tables and
        definition-backed entry-id tables.
    """
    # Duplicate physical table names across the closure are already rejected by
    # each backend's physical-name validation, which walks the identical
    # closure; keying by table_name here needs no second collision guard.
    schemas: dict[str, TableSchema] = {}

    def collect(schema: TableSchema) -> None:
        schemas.setdefault(schema.table_name, schema)

    _walk_closure(layout, collect)
    entry_id_tables = sorted(
        resolve_schema(record).table_name
        for family in layout.families
        if family.definition_id is not None
        for record in family.records
    )
    document = {
        "entry_id_tables": entry_id_tables,
        "tables": {name: _table_fingerprint(schema) for name, schema in schemas.items()},
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def schema_fingerprint_diff(stored: str | None, current: str) -> dict[str, object]:
    """Diff a stored fingerprint against the current one, per differing table.

    The persisted fingerprint shape is versioned by the store protocol, so this
    only needs to survive a corrupt stored value gracefully.

    :param stored: The persisted fingerprint JSON, or ``None`` when absent.
    :param current: The fingerprint recomputed from the persisted layout.
    :return: A mapping of table name to ``{"expected", "actual"}`` for each table
        that differs; ``{}`` when the two fingerprints are byte-equal.  A stored
        value that is not a parseable fingerprint yields a single
        ``"<fingerprint>"`` entry, and differing ``entry_id_tables`` (the backing
        tables of defined families) an ``"<entry_id_tables>"`` entry — they
        differ even when every table is unchanged, e.g. when a record kind whose
        table was already reachable by reference is appended to a family.
    """
    current_document = json.loads(current)
    current_tables = current_document["tables"]
    try:
        stored_document = json.loads(stored) if stored is not None else None
        stored_tables = stored_document["tables"] if isinstance(stored_document, dict) else None
    except (TypeError, KeyError, json.JSONDecodeError):
        stored_document, stored_tables = None, None
    if not isinstance(stored_tables, dict) or not isinstance(stored_document, dict):
        return {"<fingerprint>": {"expected": "schema fingerprint", "actual": stored}}
    diff: dict[str, object] = {}
    for name in sorted(set(stored_tables) | set(current_tables)):
        stored_table = stored_tables.get(name)
        current_table = current_tables.get(name)
        if stored_table != current_table:
            diff[name] = {"expected": stored_table, "actual": current_table}
    stored_entry_tables = stored_document.get("entry_id_tables")
    current_entry_tables = current_document.get("entry_id_tables")
    if stored_entry_tables is not None and stored_entry_tables != current_entry_tables:
        diff["<entry_id_tables>"] = {"expected": stored_entry_tables, "actual": current_entry_tables}
    return diff


def _table_fingerprint(schema: TableSchema) -> dict[str, Any]:
    """Render the identity- and layout-determining attributes of one resolved table schema."""
    hints = typing.get_type_hints(schema.cls, include_extras=True)
    return {
        "identity_name": storage_identity_name(schema.cls),
        "dedup": schema.dedup,
        "composite_indexes": [list(index) for index in schema.composite_indexes],
        # Name-keyed (reorder-immune) and json-safe: the class-valued target is
        # rendered as its resolved table name, never the class object itself.
        "links": {
            link.name: {
                "target": resolve_schema(link.target).table_name,
                "exposed_relationship": link.exposed_relationship,
                "role": link.role,
                "description": link.description,
            }
            for link in schema.links
        },
        # Keyed by field name so a pure dataclass field reorder (no column,
        # value, or identity change) does not force a store rebuild.
        "fields": {spec.field: _field_fingerprint(spec, hints.get(spec.field)) for spec in schema.fields},
    }


def _field_fingerprint(spec: FieldSpec, annotation: Any) -> dict[str, Any]:
    # Local import: store_common imports EntryFamilyLayout from this module, so a
    # module-level import would form a cycle. The helper is a pure annotation
    # inspector; importing it lazily here is order-independent.
    from httk.store.store_common import _has_identity_skip

    origin = typing.get_origin(spec.python_type)
    return {
        "role": spec.role,
        "codec_name": spec.codec_name,
        "shape": None if spec.shape is None else [spec.shape.rows, spec.shape.cols],
        "optional": spec.optional,
        "derived": spec.derived,
        # list-vs-tuple and identity participation both change content_id.
        "container": origin.__name__ if origin in (list, tuple) else None,
        "identity_skipped": _has_identity_skip(annotation),
        "columns": [_column_fingerprint(column) for column in spec.columns],
        "child": None if spec.child is None else _child_fingerprint(spec.child),
        "target": None if spec.target is None else resolve_schema(spec.target).table_name,
        "related": None if spec.related is None else dataclasses.asdict(spec.related),
    }


def _child_fingerprint(child: ChildTableSpec) -> dict[str, Any]:
    return {
        "table_name": child.table_name,
        "element_columns": [_column_fingerprint(column) for column in child.element_columns],
        "target": None if child.target is None else resolve_schema(child.target).table_name,
    }


def _column_fingerprint(column: ColumnSpec) -> dict[str, Any]:
    return {
        "name": column.name,
        "kind": column.kind,
        "nullable": column.nullable,
        "indexed": column.indexed,
        "unique": column.unique,
    }


# The immutable value keys of one fingerprinted table besides its fields; an
# additive upgrade requires every one of these to stay byte-equal.
_TABLE_INVARIANT_KEYS: Final = ("identity_name", "dedup", "composite_indexes", "links")
# Every top-level key a fingerprinted table doc is allowed to carry; a table
# growing an unrecognized key can no longer be trusted as additive.
_KNOWN_TABLE_KEYS: Final = frozenset({*_TABLE_INVARIANT_KEYS, "fields"})


@dataclasses.dataclass(frozen=True)
class AdditiveUpgradePlan:
    """The nullable parent columns an additive fingerprint upgrade must add per table.

    :param added_columns: Physical table name mapped to the ordered
        :class:`~httk.store.backend.schema.ColumnSpec` values newly present in the
        current fingerprint.  New tables carry no entry (they are created whole);
        a table appears only when it already exists in the stored fingerprint
        and gained one or more nullable parent columns.
    """

    added_columns: Mapping[str, tuple[ColumnSpec, ...]]


def classify_schema_upgrade(stored: str | None, current: str) -> AdditiveUpgradePlan | str:
    """Classify a fingerprint mismatch as an additive upgrade plan or a rejection.

    The whole diff must be additive.  A table present only in the current
    fingerprint is a new table (additive; it is created whole).  A table present
    in both is additive only when its ``identity_name``, ``dedup``,
    ``composite_indexes`` and ``links`` are byte-equal, it carries no
    unrecognized top-level key, every stored field is present and byte-equal in
    the current fingerprint, and every added field is a non-child, non-derived,
    content-identity-excluded (``IdentitySkip``) field whose parent columns are
    all nullable.  Identity participation is required so a pre-existing row's
    ``content_id`` (and therefore dedup, dispatch, and federation identity) is
    unchanged by the upgrade.  ``entry_id_tables`` may grow (a declaration
    upgrade attaches backings) but never lose a table.

    :param stored: The persisted fingerprint JSON, or ``None`` when absent.
    :param current: The fingerprint recomputed from the persisted layout.
    :return: An :class:`AdditiveUpgradePlan` when fully additive, otherwise a
        human-readable rejection reason naming the offending table/field/column.
    """
    current_document = json.loads(current)
    current_tables = current_document["tables"]
    entry_id_tables = frozenset(current_document.get("entry_id_tables", ()))
    try:
        stored_document = json.loads(stored) if stored is not None else None
        stored_tables = stored_document["tables"] if isinstance(stored_document, dict) else None
    except (TypeError, KeyError, json.JSONDecodeError):
        stored_document, stored_tables = None, None
    if not isinstance(stored_tables, dict) or not isinstance(stored_document, dict):
        return "stored schema fingerprint is not parseable"
    # Entry-id tables may only be gained (record kinds appended, families
    # added), never lost: losing one is a declaration shrink, not additive.
    lost = sorted(set(stored_document.get("entry_id_tables", ())) - entry_id_tables)
    if lost:
        return f"entry-id tables {lost!r} are no longer declared"
    added: dict[str, tuple[ColumnSpec, ...]] = {}
    for name in sorted(set(stored_tables) | set(current_tables)):
        stored_table = stored_tables.get(name)
        current_table = current_tables.get(name)
        if stored_table == current_table:
            continue
        if stored_table is None:
            continue  # new table: additive, created whole by the upgrade
        if current_table is None:
            return f"table {name!r} was removed"
        if not _KNOWN_TABLE_KEYS >= set(stored_table) or not _KNOWN_TABLE_KEYS >= set(current_table):
            return f"table {name!r} has an unrecognized fingerprint key"
        for key in _TABLE_INVARIANT_KEYS:
            if stored_table.get(key) != current_table.get(key):
                return f"table {name!r} changed {key}"
        stored_fields = stored_table["fields"]
        current_fields = current_table["fields"]
        columns: list[ColumnSpec] = []
        for field, field_doc in stored_fields.items():
            if field not in current_fields:
                return f"table {name!r} dropped field {field!r}"
            if current_fields[field] != field_doc:
                return f"table {name!r} changed field {field!r}"
        for field, field_doc in current_fields.items():
            if field in stored_fields:
                continue
            if name in entry_id_tables and field in {"id", "immutable_id"}:
                return f"table {name!r} adds enforced entry-id field {field!r}; rebuild the store"
            reason, field_columns = _added_parent_columns(name, field, field_doc)
            if reason is not None:
                return reason
            columns.extend(field_columns)
        if columns:
            added[name] = tuple(columns)
    return AdditiveUpgradePlan(added)


def _added_parent_columns(table: str, field: str, field_doc: Mapping[str, Any]) -> tuple[str | None, list[ColumnSpec]]:
    """Return the nullable parent columns an added field contributes, or a rejection reason."""
    if field_doc["role"] == "child":
        # A child field reconstructs old rows to an empty/absent collection that
        # ignores the declared type and defaults; there is no safe backfill.
        return (
            f"table {table!r} adds child field {field!r}; child-field backfill is unsupported, rebuild the store",
            [],
        )
    if field_doc["derived"]:
        # Old rows hold NULL where the true computed value differs, so queries
        # would silently under-report until every row is rewritten.
        return (
            f"table {table!r} adds derived field {field!r}; stored-property backfill is unimplemented, rebuild the store",
            [],
        )
    if not field_doc["identity_skipped"]:
        reason = (
            f"table {table!r} adds field {field!r} that participates in content identity; "
            f"mark it IdentitySkip or rebuild the store"
        )
        return reason, []
    for column in field_doc["columns"]:
        if not column["nullable"]:
            return f"table {table!r} adds field {field!r} with non-nullable column {column['name']!r}", []
    return None, [
        ColumnSpec(
            name=column["name"],
            kind=column["kind"],
            nullable=column["nullable"],
            indexed=column["indexed"],
            unique=column["unique"],
        )
        for column in field_doc["columns"]
    ]


@dataclasses.dataclass(frozen=True)
class DeclarationUpgradePlan:
    """The families an additive declaration upgrade touches.

    :param changed_families: Names of stored families whose record list gained
        appended record kinds; their dispatch tables are rebuilt and their
        entry-id numbering offsets advanced.
    :param new_families: Names of families present only in the target declaration.
    """

    changed_families: tuple[str, ...]
    new_families: tuple[str, ...]


def _stored_declaration_families(value: str) -> dict[str, tuple[str | None, tuple[tuple[str, str | None], ...]]] | None:
    """Parse the names-only format-2 declaration, or return ``None`` when it is malformed."""
    try:
        document = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict) or set(document) != {"families", "format"} or document["format"] != 2:
        return None
    families = document["families"]
    if not isinstance(families, list):
        return None
    parsed: dict[str, tuple[str | None, tuple[tuple[str, str | None], ...]]] = {}
    for item in families:
        if not isinstance(item, dict) or set(item) != {"definition_id", "records", "family"}:
            return None
        name, definition_id, records = item["family"], item["definition_id"], item["records"]
        if not isinstance(name, str) or name in parsed or not isinstance(records, list) or not records:
            return None
        if definition_id is not None and not isinstance(definition_id, str):
            return None
        entries: list[tuple[str, str | None]] = []
        for record in records:
            if not isinstance(record, dict) or set(record) != {"definition_id", "record"}:
                return None
            record_name, record_definition_id = record["record"], record["definition_id"]
            if not isinstance(record_name, str):
                return None
            if record_definition_id is not None and not isinstance(record_definition_id, str):
                return None
            entries.append((record_name, record_definition_id))
        parsed[name] = (definition_id, tuple(entries))
    return parsed


def classify_declaration_upgrade(stored: str | None, target: StorageLayout) -> DeclarationUpgradePlan | str:
    """Classify a persisted-declaration mismatch as an additive upgrade plan or a rejection.

    The change is additive iff every stored family is still declared under the
    same name with the same family ``definition_id``, and the stored record list
    of every such family (registry names *and* record ``definition_id`` values)
    is an order-preserving prefix of the target family's record list, so record
    kinds are only ever appended.  New families are allowed.  Record order is
    part of the entry-id numbering scheme (a record's position is its backing
    index), which is why a reorder is never additive.  The comparison works on
    the names-only persisted declaration, so it is shared by every backend.

    :param stored: The persisted ``entry_declaration`` JSON, or ``None`` when absent.
    :param target: The declaration the store is being reopened with.
    :return: A :class:`DeclarationUpgradePlan` when the change is additive,
        otherwise a human-readable rejection naming the offending family and record.
    """
    parsed = None if stored is None else _stored_declaration_families(stored)
    if parsed is None:
        return "stored entry declaration is not parseable"
    targets = {
        family.name: (family.definition_id, tuple(zip(family.record_names, family.record_definition_ids, strict=True)))
        for family in target.families
    }
    return _classify_declaration_families(parsed, targets)


def declaration_is_superseded(stored: str | None, supplied: StorageLayout) -> bool:
    """Whether the persisted declaration is an additive extension of ``supplied``.

    That is the situation of a client one declaration behind: the store was
    already upgraded (record kinds appended or families added) by a newer
    declaration, and ``supplied`` is the older one.  The store is healthy; the
    remedy is to open it with the newer declaration, never to rebuild it.

    :param stored: The persisted ``entry_declaration`` JSON, or ``None`` when absent.
    :param supplied: The declaration the store is being opened with.
    :return: ``True`` when ``supplied`` → stored is an additive change.
    """
    parsed = None if stored is None else _stored_declaration_families(stored)
    if parsed is None:
        return False
    older = {
        family.name: (family.definition_id, tuple(zip(family.record_names, family.record_definition_ids, strict=True)))
        for family in supplied.families
    }
    return isinstance(_classify_declaration_families(older, parsed), DeclarationUpgradePlan)


def _classify_declaration_families(
    stored: Mapping[str, tuple[str | None, tuple[tuple[str, str | None], ...]]],
    target: Mapping[str, tuple[str | None, tuple[tuple[str, str | None], ...]]],
) -> DeclarationUpgradePlan | str:
    """Apply the append-only rule to two names-only declarations (see :func:`classify_declaration_upgrade`)."""
    changed: list[str] = []
    for name, (definition_id, stored_records) in sorted(stored.items()):
        found = target.get(name)
        if found is None:
            return f"family {name!r} was removed from the declaration"
        target_definition, target_records = found
        if target_definition != definition_id:
            return f"family {name!r} changed its definition_id from {definition_id!r} to {target_definition!r}"
        target_names = [record_name for record_name, _definition in target_records]
        for position, (record_name, record_definition_id) in enumerate(stored_records):
            if position >= len(target_records):
                return f"family {name!r} no longer declares record {record_name!r}"
            target_name, target_definition_id = target_records[position]
            if target_name == record_name:
                if target_definition_id != record_definition_id:
                    return (
                        f"family {name!r} record {record_name!r} changed its definition_id from "
                        f"{record_definition_id!r} to {target_definition_id!r}"
                    )
                continue
            if record_name in target_names:
                return (
                    f"family {name!r} moved record {record_name!r} from position {position}; "
                    "record kinds may only be appended"
                )
            return (
                f"family {name!r} no longer declares record {record_name!r} at position {position} "
                f"(found {target_name!r}); record kinds may only be appended"
            )
        if len(target_records) > len(stored_records):
            changed.append(name)
    new = sorted(name for name in target if name not in stored)
    return DeclarationUpgradePlan(tuple(changed), tuple(new))


def entry_id_number(value: object) -> int | None:
    """Return the numeric suffix of an entry, immutable, or alternative identifier.

    Identifiers are parsed exactly as the store mints and validates them
    (:func:`httk.core.entry_ids.parse_entry_id` and its immutable/alternative
    siblings).  An identifier outside the recommended ``<base>-<series>-<number>``
    form carries no store-mintable number and yields ``None``: it can never
    collide with a minted identifier.

    :param value: A stored ``id`` or ``immutable_id`` value (``None`` allowed).
    :return: The embedded entry number, or ``None``.
    """
    if not isinstance(value, str):
        return None
    parsed = parse_entry_id(value)
    if parsed is not None:
        return parsed[2]
    embedded: str | None = None
    immutable = parse_immutable_id(value)
    if immutable is not None:
        embedded = immutable[0]
    else:
        alternative = parse_alternative_id(value)
        if alternative is not None:
            embedded = alternative[0]
    if embedded is None:
        return None
    parsed = parse_entry_id(embedded)
    return None if parsed is None else parsed[2]


def next_entry_id_offset(numbers: Iterable[int], previous: int) -> int:
    """Return a family's entry-id numbering offset after its record list changes.

    Store-minted numbers are ``offset + logical_id * B + backing_index`` with
    ``logical_id >= 1`` and ``0 <= backing_index < B``, so every number minted
    under the returned offset is at least ``offset + B``.  The offset is one
    more than the largest number any existing identifier of the family carries
    (over every backing and every id series) and than the previous offset, so
    every later number exceeds every earlier one: identifiers minted before the
    change can never be re-minted after it, whatever ``B`` becomes, and repeated
    upgrades keep the property.  A family with neither existing numbers nor a
    previous offset keeps offset ``0`` and so mints exactly like a fresh store.

    :param numbers: The numeric suffixes of the family's existing identifiers.
    :param previous: The family's current offset (``0`` when none is recorded).
    :return: The new offset.
    """
    highest = max(numbers, default=0)
    if highest == 0 and previous == 0:
        return 0
    return max(highest, previous) + 1


def entry_id_offsets_json(offsets: Mapping[str, int]) -> str:
    """Serialize per-family entry-id offsets in their canonical persisted form.

    :param offsets: Family name mapped to its positive offset (zero offsets are omitted).
    :return: The canonical JSON object persisted under :data:`ENTRY_ID_OFFSETS_KEY`.
    """
    return json.dumps(
        {name: offset for name, offset in offsets.items() if offset}, sort_keys=True, separators=(",", ":")
    )


def parse_entry_id_offsets(value: str, layout: StorageLayout) -> dict[str, int]:
    """Parse and validate the persisted per-family entry-id offsets.

    :param value: The persisted :data:`ENTRY_ID_OFFSETS_KEY` value.
    :param layout: The layout whose families the offsets must name.
    :return: Family name mapped to its positive offset.
    :raises ValueError: If the value is not the canonical encoding of a mapping
        from declared family names to positive integers.
    """
    try:
        document = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("entry_id_offsets is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("entry_id_offsets must be a JSON object")
    names = {family.name for family in layout.families}
    offsets: dict[str, int] = {}
    for name, offset in document.items():
        if name not in names:
            raise ValueError(f"entry_id_offsets names undeclared family {name!r}")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset <= 0:
            raise ValueError(f"entry_id_offsets[{name!r}] must be a positive integer")
        offsets[name] = offset
    if entry_id_offsets_json(offsets) != value:
        raise ValueError("entry_id_offsets is not in its canonical encoding")
    return offsets


def _layout_from_declaration(value: str) -> StorageLayout:
    try:
        document = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("stored entry declaration is not valid JSON") from error
    if not isinstance(document, dict) or set(document) != {"families", "format"} or document["format"] != 2:
        raise ValueError("stored entry declaration does not use format 2")
    families = document["families"]
    if not isinstance(families, list):
        raise ValueError("stored entry declaration families must be a list")
    supplied: dict[type, tuple[type, ...]] = {}
    previous = ""
    for item in families:
        if not isinstance(item, dict) or set(item) != {"definition_id", "records", "family"}:
            raise ValueError("stored entry declaration family entry is malformed")
        family_name = item["family"]
        family_definition_id = item["definition_id"]
        record_names = item["records"]
        if not isinstance(family_name, str) or not isinstance(record_names, list) or not record_names:
            raise ValueError("stored entry declaration has an invalid family or record list")
        _validate_name(family_name, label="stored entry family name")
        _validate_optional_definition_id(family_definition_id)
        if family_name <= previous:
            raise ValueError("stored entry declaration families are not deterministically ordered")
        previous = family_name
        if family_name not in known_entry_families():
            raise EntryLayoutBindingError(
                f"stored entry family {family_name!r} is not registered; reopen the store with entry_families"
            )
        _, registered_family_definition_id = entry_family_info(family_name)
        if family_definition_id != registered_family_definition_id:
            raise ValueError(f"stored entry family {family_name!r} definition does not match its registration")
        family = resolve_entry_family(family_name)
        resolved_records: list[type] = []
        for record_item in record_names:
            if not isinstance(record_item, dict) or set(record_item) != {"definition_id", "record"}:
                raise ValueError("stored entry declaration record entry is malformed")
            record_name = record_item["record"]
            record_definition_id = record_item["definition_id"]
            if not isinstance(record_name, str):
                raise ValueError("stored entry declaration record names must be strings")
            _validate_name(record_name, label="stored entry record name")
            _validate_optional_definition_id(record_definition_id)
            if record_name not in known_entry_records():
                raise EntryLayoutBindingError(
                    f"stored entry record {record_name!r} is not registered; reopen the store with entry_families"
                )
            _, declared_family, registered_definition_id = entry_record_info(record_name)
            if declared_family is None:
                raise ValueError(
                    f"entry record {record_name!r} has no registered family and cannot be used in a family store"
                )
            if declared_family != family_name:
                raise ValueError(
                    f"stored entry record {record_name!r} is registered for {declared_family!r}, not {family_name!r}"
                )
            if record_definition_id != registered_definition_id:
                raise ValueError(f"stored entry record {record_name!r} definition does not match its registration")
            resolved_records.append(resolve_entry_record(record_name))
        supplied[family] = tuple(resolved_records)
    layout = normalize_entry_records(supplied)
    if declaration_json(layout) != value:
        raise ValueError("stored entry declaration is not in its canonical deterministic encoding")
    return layout


def _reject_registry_conflicts(declaration: EntryFamilyDeclaration) -> None:
    if declaration.name in known_entry_families():
        reference, definition_id = entry_family_info(declaration.name)
        matches = _reference_matches_class(reference, declaration.family)
        if not matches or definition_id != declaration.definition_id:
            raise ValueError(f"explicit entry family {declaration.name!r} conflicts with a global registration")
    for record in declaration.records:
        if record.name not in known_entry_records():
            continue
        reference, family_name, definition_id = entry_record_info(record.name)
        if (
            not _reference_matches_class(reference, record.record)
            or family_name != declaration.name
            or definition_id != record.definition_id
        ):
            raise ValueError(f"explicit entry record {record.name!r} conflicts with a global registration")


def _reference_matches_class(reference: str, cls: type) -> bool:
    canonical = f"{cls.__module__}:{cls.__name__}"
    if reference == canonical:
        return True
    module_name, separator, attribute = reference.partition(":")
    module = sys.modules.get(module_name) if separator else None
    return module is not None and getattr(module, attribute, None) is cls


def _validate_name(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")


def _validate_optional_definition_id(value: object) -> None:
    if value is not None:
        _validate_name(value, label="definition_id")


def _registered_family_name(family: type) -> str:
    matches = _registered_names_for(family, known_entry_families(), entry_family_info)
    if len(matches) != 1:
        found = ", ".join(matches) or "none"
        raise ValueError(f"entry family {family.__name__} must resolve to exactly one registered name (found {found})")
    return matches[0]


def _registered_record_name(record: type) -> str:
    matches = _registered_names_for(record, known_entry_records(), entry_record_info)
    if len(matches) != 1:
        found = ", ".join(matches) or "none"
        raise ValueError(f"entry record {record.__name__} must resolve to exactly one registered name (found {found})")
    return matches[0]


def _registered_names_for(record: type, names: list[str], info: object) -> list[str]:
    """Return registry names for ``record`` without importing unrelated lazy entries.

    A store declaration already has the concrete class in hand.  Resolving
    every registry reference just to find its stable name turns that harmless
    validation into a transitive import of every optional entry package.  Some
    such packages are deliberately heavyweight; more importantly, repeated
    store construction must not retain their import-time state.

    Registry references conventionally name the class's defining module.  A
    loaded alias remains supported by identity, while an unloaded unrelated
    entry is never imported merely for declaration validation.
    """
    get_info = info
    if not callable(get_info):  # pragma: no cover - defensive narrowing for the registry seam
        raise TypeError("registry info lookup must be callable")
    canonical = f"{record.__module__}:{record.__name__}"
    matches: list[str] = []
    for name in names:
        reference = cast('tuple[str, ...]', get_info(name))[0]
        if reference == canonical:
            matches.append(name)
            continue
        module_name, separator, attribute = reference.partition(":")
        module = sys.modules.get(module_name) if separator else None
        if module is not None and getattr(module, attribute, None) is record:
            matches.append(name)
    return matches


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(member) for key, member in item.items()})
        if isinstance(item, list):
            return tuple(freeze(member) for member in item)
        if isinstance(item, tuple):
            return tuple(freeze(member) for member in item)
        if isinstance(item, set | frozenset):
            return tuple(sorted((freeze(member) for member in item), key=repr))
        return item

    return MappingProxyType({str(key): freeze(member) for key, member in value.items()})
