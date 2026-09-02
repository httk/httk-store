"""Serve MongoDB-backed records through the neutral entry-provider contract.

Mongo entry identities are the store-managed physical ``id`` values carried
by the hydrated records.
Configured entry families are rendered through their Mongo stored-property plan;
configured backing records are also accepted for the schema-derived provider
surface used by the SQL provider's parity tests.
"""

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from httk.core import (
    EntryProvider,
    EntryTypeDefinition,
    FracVector,
    PropertyDefinition,
    RelatedEntry,
    known_definition_prefixes,
)

from httk.store.backend.codecs import codec_named
from httk.store.backend.schema import (
    FieldSpec,
    SchemaError,
    TableSchema,
    resolve_schema,
)
from httk.store.query import ID_FIELD
from httk.store.served_specs import served_specs

from .documents import _as_fixed_tensor
from .stored_properties import MongoStoredPropertyPlan

__all__ = ["StoreEntryProvider", "auto_definition", "served_specs"]


def _default_id(_entry_type: str, _sid: int, obj: Any) -> str:
    """Return the store-minted identifier carried by ``obj``."""
    value = getattr(obj, "id", None)
    if value is None:
        raise ValueError(f"{_entry_type} record sid {_sid} has no stored id")
    if not isinstance(value, str):
        raise ValueError(f"{_entry_type} record sid {_sid} has a non-string stored id")
    return value


def _query_index(codec: Any) -> int:
    """Return the stored codec component used for query/response values."""
    return next(
        (index for index, (suffix, _kind) in enumerate(codec.columns) if suffix == codec.query_suffix),
        0,
    )


def auto_definition(entry_type: str, schema: TableSchema, prefix: str) -> EntryTypeDefinition:
    """Build a definition for the JSON-able fields of one backing class."""
    base = EntryTypeDefinition(
        entry_type,
        f"The '{entry_type}' entry type, generated from the stored class {schema.cls.__name__}.",
        {
            "id": PropertyDefinition.from_simple("id", description="The unique entry id.", required_response=True),
            "type": PropertyDefinition.from_simple(
                "type",
                description="The name of the entry type.",
                required_response=True,
            ),
        },
    )
    extra: dict[str, PropertyDefinition] = {}
    for name, spec, fulltype in served_specs(schema, prefix):
        dimensions: dict[str, Any] | None = None
        if spec.role == "fixed_array":
            assert spec.shape is not None
            dimensions = {
                "names": ["rows", "cols"],
                "sizes": [spec.shape.rows, spec.shape.cols],
            }
        kind = "stored property" if spec.derived else "stored field"
        extra[name] = PropertyDefinition.from_simple(
            name,
            description=f"The {kind} '{spec.field}' of {schema.cls.__name__}.",
            fulltype=fulltype,
            dimensions=dimensions,
        )
    return base.extended(extra)


class StoreEntryProvider(EntryProvider):
    """Serve configured Mongo entry families or their concrete backings.

    ``classes`` maps public entry-type names to either configured entry-family
    classes or configured concrete backing classes. Family classes use the
    family's :class:`~httk.store.backend.mongo.stored_properties.MongoStoredPropertyPlan`; backing classes use the same
    schema-derived property contract as the SQL provider. ``id_of`` receives
    ``(entry_type, sid, hydrated_record)`` and defaults to the record's
    stored ``id`` field. ``only_latest`` restricts served searchers' root
    variables to the latest document of each lineage.
    """

    def __init__(
        self,
        store: Any,
        classes: Mapping[str, type],
        *,
        definitions: Mapping[str, EntryTypeDefinition] | None = None,
        prefix: str = "_httk_",
        id_of: Callable[[str, int, Any], str] | None = None,
        only_latest: bool = True,
    ) -> None:
        if prefix not in known_definition_prefixes():
            raise ValueError(
                f"the property-name prefix {prefix!r} is not registered; register it with "
                f"httk.core.register_definition_prefix() (registered prefixes: "
                f"{', '.join(known_definition_prefixes())})"
            )
        if id_of is None and not only_latest:
            raise ValueError(
                "StoreEntryProvider(only_latest=False) requires an id_of override; "
                "all-revision serving must use immutable ids"
            )
        # This provider serves mains only: the store searcher defaults to
        # only_main_alt=True, so named alternatives never appear here and their
        # revisions never enter the revision stream. Alternative serving is
        # available through StoredEntryFederation, not this provider.
        self._store = store
        self._classes = dict(classes)
        self._only_latest = only_latest
        self._prefix = prefix
        self._id_of = id_of if id_of is not None else _default_id
        self._definitions = dict(definitions or {})
        unknown = sorted(name for name in self._definitions if name not in self._classes)
        if unknown:
            raise ValueError(
                f"definitions were supplied for entry types this provider does not serve: {', '.join(unknown)}"
            )

        self._families: dict[str, type] = {}
        self._plans: dict[str, MongoStoredPropertyPlan] = {}
        self._record_classes: dict[str, tuple[type, ...]] = {}
        self._schemas: dict[type, TableSchema] = {}
        for entry_type, selected in self._classes.items():
            family = self._family_for(selected, entry_type)
            if family is not None:
                plan = store.stored_property_plan(family)
                self._families[entry_type] = family
                self._plans[entry_type] = plan
                self._record_classes[entry_type] = plan.backings
            else:
                self._record_classes[entry_type] = (selected,)
            for record in self._record_classes[entry_type]:
                self._schemas[record] = resolve_schema(record)

        if id_of is None:
            required = "id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)"
            for record, schema in self._schemas.items():
                try:
                    spec = schema.field("id")
                except SchemaError:
                    spec = None
                if spec is None or spec.role != "scalar" or spec.python_type is not str:
                    raise TypeError(f"{record.__name__} must declare {required} when served without id_of")

        # Keyed by served (wire) entry-type names: relationship targets are
        # served directly, so this map resolves a target class to the name it is
        # served under.
        self._type_for_class: dict[type, str] = {}
        for entry_type, records in self._record_classes.items():
            for record in records:
                self._type_for_class[record] = entry_type

        for entry_type in self._classes:
            definition = self._definitions.get(entry_type)
            if definition is not None:
                actual_keys = self._actual_property_keys(entry_type)
                missing = sorted(set(actual_keys) - set(definition.properties))
                if missing:
                    raise ValueError(
                        f"the supplied definition for entry type {entry_type!r} does not describe the served "
                        f"propert{'y' if len(missing) == 1 else 'ies'}: {', '.join(missing)}"
                    )

    def _family_for(self, selected: type, entry_type: str) -> type | None:
        for layout in self._store.entry_layout:
            if selected is layout.family:
                family_type = getattr(selected, "type", entry_type)
                if entry_type != family_type:
                    raise ValueError(f"entry family {selected.__name__} has type {family_type!r}, not {entry_type!r}")
                return selected
            if selected in layout.records:
                return None
        raise ValueError(
            f"record class {selected.__name__} is not configured in this MongoStore; "
            "serve a configured entry family or one of its configured backing records"
        )

    def _require_entry_type(self, entry_type: str) -> None:
        if entry_type not in self._classes:
            raise KeyError(f"StoreEntryProvider serves only the entry type(s): {', '.join(sorted(self._classes))}")

    def _definition(self, entry_type: str) -> EntryTypeDefinition:
        supplied = self._definitions.get(entry_type)
        if supplied is not None:
            return supplied
        plan = self._plans.get(entry_type)
        if plan is not None:
            return plan.definition
        return auto_definition(entry_type, self._schemas[self._record_classes[entry_type][0]], self._prefix)

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        """Return definitions for all served entry types.

        This provider is an OPTIMADE serving edge, so each definition is returned
        in its wire form via ``EntryTypeDefinition.served_form()``
        (idempotent for the already-prefixed supplied and auto-generated
        definitions).

        :return: The served entry-type definitions keyed by entry type.
        """
        return {entry_type: self._definition(entry_type).served_form() for entry_type in self._classes}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        """Return public property names mapped to Mongo response keys."""
        self._require_entry_type(entry_type)
        return self._actual_property_keys(entry_type)

    def _actual_property_keys(self, entry_type: str) -> Mapping[str, str]:
        """Return keys actually emitted by :meth:`records`, independent of overrides."""
        if entry_type in self._plans:
            return {name: name for name in self._plans[entry_type].definition.properties}
        return {
            "id": ID_FIELD,
            "type": "type",
            **{name: name for name, _spec, _ in self._served_specs(entry_type)},
        }

    def _served_specs(self, entry_type: str) -> list[tuple[str, FieldSpec, str]]:
        record = self._record_classes[entry_type][0]
        return served_specs(self._schemas[record], self._prefix)

    def _iter_records(self, record: type) -> Iterator[tuple[Any, int]]:
        searcher = self._store.searcher(only_latest=self._only_latest)
        variable = searcher.variable(record)
        searcher.add_sort(variable.sid)
        searcher.output(variable, "record")
        searcher.output(variable.sid, "sid")
        for result in searcher:
            yield result[0][0], int(result[0][1])

    def records(self, entry_type: str) -> Iterator[Mapping[str, Any]]:
        """Yield JSON-able records for one served entry type."""
        self._require_entry_type(entry_type)
        plan = self._plans.get(entry_type)
        if plan is not None:
            for backing in plan.backings:
                for record, sid in self._iter_records(backing):
                    yield plan.response_row(backing, record, public_id=self._id_of(entry_type, sid, record))
            return
        specs = self._served_specs(entry_type)
        schema = self._schemas[self._record_classes[entry_type][0]]
        for record, sid in self._iter_records(self._record_classes[entry_type][0]):
            row: dict[str, Any] = {
                ID_FIELD: self._id_of(entry_type, sid, record),
                "type": entry_type,
            }
            for name, spec, _fulltype in specs:
                row[name] = _json_value(schema, spec, getattr(record, spec.field))
            yield row

    def _relationship_specs(self, record: type) -> list[tuple[FieldSpec, str]]:
        result: list[tuple[FieldSpec, str]] = []
        for spec in self._schemas[record].fields:
            if spec.role not in {"reference", "child"} or spec.target is None:
                continue
            if spec.related is not None and not spec.related.serve:
                continue
            related = self._type_for_class.get(spec.target)
            if related is not None:
                result.append((spec, related))
        return result

    def _exposed_link_specs(self, record: type) -> list[tuple[Any, str]]:
        """The ``(LinkSpec, related entry type)`` exposed weak links whose target is served.

        Only ``exposed_relationship=True`` links contribute; ``False`` links are
        served nowhere.
        """
        result: list[tuple[Any, str]] = []
        for spec in self._schemas[record].links:
            if not spec.exposed_relationship:
                continue
            related = self._type_for_class.get(spec.target)
            if related is not None:
                result.append((spec, related))
        return result

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        """Return direct relationships grouped by source id."""
        self._require_entry_type(entry_type)
        result: dict[str, list[RelatedEntry]] = {}
        for record_type in self._record_classes[entry_type]:
            for source, sid in self._iter_records(record_type):
                entries: list[RelatedEntry] = []
                relation_specs = self._relationship_specs(record_type)
                relation_specs.sort(key=lambda item: item[0].role != "reference")
                for spec, related_type in relation_specs:
                    values = (
                        (getattr(source, spec.field),)
                        if spec.role == "reference"
                        else (getattr(source, spec.field) or ())
                    )
                    marker = spec.related
                    for target in values:
                        if target is None:
                            continue
                        target_type = self._type_for_class.get(type(target))
                        if target_type != related_type:
                            continue
                        target_sid = self._store.sid_of(target, as_record=type(target))
                        if target_sid is None:
                            continue
                        entries.append(
                            RelatedEntry(
                                related_type,
                                self._id_of(related_type, target_sid, target),
                                description=(marker.description if marker is not None else None),
                                role=marker.role if marker is not None else None,
                            )
                        )
                for link_spec, related_type in self._exposed_link_specs(record_type):
                    # Weak links bind lineages: linked() returns the live latest
                    # target revisions (deduped, retracted dropped), so id
                    # resolution is lineage-level on either side.
                    for target in self._store.linked(source, link_spec.name):
                        target_sid = self._store.sid_of(target, as_record=type(target))
                        if target_sid is None:
                            continue
                        entries.append(
                            RelatedEntry(
                                related_type,
                                self._id_of(related_type, target_sid, target),
                                description=link_spec.description,
                                role=link_spec.role,
                                label=link_spec.name,
                            )
                        )
                if entries:
                    result[self._id_of(entry_type, sid, source)] = entries

        return {key: tuple(dict.fromkeys(values)) for key, values in result.items()}


def _json_value(schema: TableSchema, spec: FieldSpec, value: Any) -> Any:
    if value is None:
        return None
    if spec.role == "scalar":
        return value
    if spec.role == "encoded":
        assert spec.codec_name is not None
        codec = codec_named(spec.codec_name)
        return codec.encode(value)[_query_index(codec)]
    if spec.role == "fixed_array":
        assert spec.shape is not None
        return _as_fixed_tensor(schema, spec, spec.shape, value).to_floats()
    assert spec.role == "child"
    if spec.shape is not None:
        tensor = FracVector(value)
        return [] if tensor.dim in {(), (0,)} else tensor.to_floats()
    if spec.codec_name is not None:
        codec = codec_named(spec.codec_name)
        index = _query_index(codec)
        return [codec.encode(element)[index] for element in value]
    return list(value)
