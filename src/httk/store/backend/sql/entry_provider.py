"""Serve stored dataclasses through the httk-core entry-provider contract.

:class:`StoreEntryProvider` bridges the SQL storage layer to the neutral
:class:`~httk.core.EntryProvider` contract: it serves the rows of one or more
storable classes in a :class:`~httk.store.backend.sql.store.SqlStore` as described,
JSON-able entry-type records, so a serving module (such as *httk-serve*)
can expose a database as an OPTIMADE API without either side depending on the
other.

For each served class the provider either passes through a supplied
:class:`~httk.core.EntryTypeDefinition` (validated to describe every served
property) or auto-generates one from the class's resolved
:class:`~httk.store.backend.schema.TableSchema`: the OPTIMADE core ``id``/``type``
properties plus one :meth:`~httk.core.PropertyDefinition.from_simple`
definition per servable stored field, each named with a registered
database-specific ``prefix`` (default ``"_httk_"``) and merged in via
:meth:`~httk.core.EntryTypeDefinition.extended` — the same construction route
the other httk entry providers use.

The schema-to-OPTIMADE type mapping is:

- ``str``/``int``/``bool``/``float`` fields — ``string``/``integer``/
  ``boolean``/``float``;
- rational fields (:class:`fractions.Fraction`, :class:`~httk.core.FracScalar`,
  :class:`~httk.core.SurdScalar`) — ``float``, served as the nearest float
  (stored values themselves remain exact; only the *served* value is
  approximate);
- :class:`datetime.datetime` fields — ``timestamp``, served as ISO-8601 text;
- fixed-shape (``Shape(r, c)``) and variable-rows (``Shape(0, c)``)
  :class:`~httk.core.FracVector` fields — ``list of list of float`` (the fixed
  shape also declares its dimension sizes);
- ``list``/``tuple`` fields of scalars or of the codec types above — ``list
  of`` the mapped element type.

Not every stored field can be served as a property: ``bytes`` fields (and
fields encoded by a custom, non-built-in value codec) have no OPTIMADE value
representation and are skipped, while reference fields and child fields of
storable elements surface through :meth:`StoreEntryProvider.relationships`
instead — when their target class is itself served, each record declares its
related entries as a flat tuple of :class:`~httk.core.RelatedEntry` values,
carrying the ``role``/``description`` metadata of an optional
:class:`~httk.core.storage.Related` field marker (``Related(serve=False)`` suppresses
the field as a relationship). Exposed weak links
(``WeakLink`` declared ``exposed_relationship=True``)
whose target class is served also surface through
:meth:`StoreEntryProvider.relationships`: each source lineage's live latest
link rows become related entries carrying the link's ``role``/``description``,
resolved to the target lineage's latest revision id (links declared
``exposed_relationship=False`` are served nowhere).
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import sqlalchemy
from httk.core import (
    EntryProvider,
    EntryTypeDefinition,
    FracVector,
    PropertyDefinition,
    RelatedEntry,
    known_definition_prefixes,
)

from httk.store.backend.codecs import codec_named
from httk.store.backend.schema import FieldSpec, SchemaError, TableSchema, resolve_schema
from httk.store.backend.sql.mapping import (
    LOGICAL_ID_COLUMN,
    RETRACTED_COLUMN,
    SID_COLUMN,
    SOURCE_LID_COLUMN,
    TARGET_LID_COLUMN,
)
from httk.store.backend.sql.provenance_edges import (
    forward_run_edges,
    latest_main_run_sids,
    reverse_run_edges,
    strong_link_families,
    wire_type_for_internal,
)
from httk.store.backend.sql.searcher import SqlColumn, _query_index
from httk.store.backend.sql.store import SqlStore, _as_fixed_tensor
from httk.store.entry_providers import wire_relationship_key
from httk.store.query import ID_FIELD
from httk.store.served_specs import _fulltype_of as _served_fulltype_of
from httk.store.served_specs import served_specs

_fulltype_of = _served_fulltype_of

__all__ = [
    "StoreEntryProvider",
    "auto_definition",
    "served_specs",
]


def _live_targets_by_source(rows: Any) -> dict[int, list[int]]:
    """The live latest-of-lineage linked target lids per source lid, from raw link rows.

    ``rows`` yields ``(source_lid, target_lid, logical_id, sid, retracted)``.
    Each pair-lineage (``logical_id``) is reduced to its latest revision; a
    target is live under a source if any of its pair-lineages is live. The
    returned target lids are deduplicated and ordered by first-link order (the
    smallest live pair-lineage root), matching ``linked()``.
    """
    latest: dict[int, tuple[int, int, int, int]] = {}  # logical_id -> (sid, retracted, target_lid, source_lid)
    for source_lid, target_lid, logical_id, sid, retracted in rows:
        lineage = int(logical_id)
        previous = latest.get(lineage)
        if previous is None or int(sid) > previous[0]:
            latest[lineage] = (int(sid), int(retracted), int(target_lid), int(source_lid))
    live_root: dict[int, dict[int, int]] = {}  # source_lid -> {target_lid -> smallest live root}
    for lineage, (_sid, retracted, target_lid, source_lid) in latest.items():
        if retracted != 0:
            continue
        roots = live_root.setdefault(source_lid, {})
        existing = roots.get(target_lid)
        if existing is None or lineage < existing:
            roots[target_lid] = lineage
    return {
        source_lid: [tl for tl, _root in sorted(roots.items(), key=lambda item: (item[1], item[0]))]
        for source_lid, roots in live_root.items()
    }


def _default_id(entry_type: str, sid: int, obj: Any) -> str:
    """Return the store-minted identifier carried by ``obj``."""
    value = getattr(obj, "id", None)
    if value is None:
        raise ValueError(f"{entry_type} record sid {sid} has no stored id")
    if not isinstance(value, str):
        raise ValueError(f"{entry_type} record sid {sid} has a non-string stored id")
    return value


def auto_definition(entry_type: str, schema: TableSchema, prefix: str) -> EntryTypeDefinition:
    """Auto-generate the :class:`~httk.core.EntryTypeDefinition` of a storable class.

    The definition carries the OPTIMADE core ``id``/``type`` properties plus
    one :meth:`~httk.core.PropertyDefinition.from_simple` definition per triple
    of :func:`served_specs`, named in the ``custom_`` sub-namespace of
    ``prefix`` so generated names cannot collide with curated prefixed
    definitions, merged in via
    :meth:`~httk.core.EntryTypeDefinition.extended` (so ``prefix`` must be a
    registered definition prefix).

    :param entry_type: The entry type name to define.
    :param schema: The resolved schema of the stored class.
    :param prefix: The registered prefix used for generated property names.
    :return: The generated entry-type definition.
    """
    cls = schema.cls
    base = EntryTypeDefinition(
        entry_type,
        f"The '{entry_type}' entry type, generated from the stored class {cls.__name__}.",
        {
            "id": PropertyDefinition.from_simple("id", description="The unique entry id.", required_response=True),
            "type": PropertyDefinition.from_simple(
                "type", description="The name of the entry type.", required_response=True
            ),
        },
    )
    extra: dict[str, PropertyDefinition] = {}
    for name, spec, fulltype in served_specs(schema, prefix):
        kind = "stored property" if spec.derived else "stored field"
        dimensions: dict[str, Any] | None = None
        if spec.role == "fixed_array":
            assert spec.shape is not None
            dimensions = {"names": ["rows", "cols"], "sizes": [spec.shape.rows, spec.shape.cols]}
        extra[name] = PropertyDefinition.from_simple(
            name,
            description=f"The {kind} '{spec.field}' of {cls.__name__}.",
            fulltype=fulltype,
            dimensions=dimensions,
        )
    return base.extended(extra)


class StoreEntryProvider(EntryProvider):
    """Serves the stored rows of storable classes as httk-core entry types.

    ``classes`` maps each served entry-type name to its storable dataclass;
    the classes' tables are read through ``store``. ``definitions`` optionally
    supplies the :class:`~httk.core.EntryTypeDefinition` of an entry type
    (validated: it must describe every property the provider serves for it);
    entry types without a supplied definition get one auto-generated from the
    class's schema, with every schema-derived property name carrying
    ``prefix`` (which must be registered, see
    :func:`~httk.core.register_definition_prefix`). ``id_of`` maps
    ``(entry_type, sid, instance)`` to the served entry id; the default reads
    the record's store-minted ``id`` field.

    See the module docstring for which stored fields are served as properties
    (and how their types map), which are skipped, and which surface through
    :meth:`relationships` instead.

    :param store: The SQL store containing the served records.
    :param classes: The served entry-type names and their storable classes.
    :param definitions: Optional definitions to use instead of auto-generation.
    :param prefix: The registered prefix for generated property names.
    :param id_of: The function that maps a served record to its public id.
    :param only_latest: Whether served searchers restrict root variables to the latest row of each lineage.
    """

    def __init__(
        self,
        store: SqlStore,
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
        self._classes: dict[str, type] = dict(classes)
        self._only_latest = only_latest
        self._prefix = prefix
        self._id_of: Callable[[str, int, Any], str] = id_of if id_of is not None else _default_id
        # Keyed by served (wire) entry-type names: relationship targets are
        # served directly, so this map must resolve a target class to the name
        # it is served under.
        self._entry_type_of: dict[type, str] = {cls: name for name, cls in self._classes.items()}
        self._schemas: dict[str, TableSchema] = {name: resolve_schema(cls) for name, cls in self._classes.items()}
        if id_of is None:
            required = "id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)"
            for cls, schema in zip(self._classes.values(), self._schemas.values(), strict=True):
                try:
                    spec = schema.field("id")
                except SchemaError:
                    spec = None
                if spec is None or spec.role != "scalar" or spec.python_type is not str:
                    raise TypeError(f"{cls.__name__} must declare {required} when served without id_of")
        self._definitions: dict[str, EntryTypeDefinition] = dict(definitions or {})
        unknown = sorted(name for name in self._definitions if name not in self._classes)
        if unknown:
            raise ValueError(
                f"definitions were supplied for entry types this provider does not serve: {', '.join(unknown)}"
            )
        for entry_type in self._classes:
            definition = self._definitions.get(entry_type)
            if definition is not None:
                described = definition.properties
                missing = sorted(name for name in self.property_keys(entry_type) if name not in described)
                if missing:
                    raise ValueError(
                        f"the supplied definition for entry type {entry_type!r} does not describe the "
                        f"served propert{'y' if len(missing) == 1 else 'ies'}: {', '.join(missing)}"
                    )

    # ------------------------------------------------------------------ the served subset

    def _require_entry_type(self, entry_type: str) -> type:
        try:
            return self._classes[entry_type]
        except KeyError:
            raise KeyError(
                f"StoreEntryProvider serves only the entry type(s): {', '.join(sorted(self._classes))}"
            ) from None

    def _served_specs(self, entry_type: str) -> list[tuple[str, FieldSpec, str]]:
        """The served ``(property name, field spec, fulltype)`` triples of ``entry_type``."""
        return served_specs(self._schemas[entry_type], self._prefix)

    def _relationship_specs(self, entry_type: str) -> list[tuple[FieldSpec, str]]:
        """The ``(field spec, related entry type)`` pairs whose target class is also served.

        Fields suppressed with ``Related(serve=False)`` are excluded.
        """
        pairs: list[tuple[FieldSpec, str]] = []
        for spec in self._schemas[entry_type].fields:
            if spec.role not in ("reference", "child") or spec.target is None:
                continue
            if spec.related is not None and not spec.related.serve:
                continue
            related = self._entry_type_of.get(spec.target)
            if related is not None:
                pairs.append((spec, related))
        return pairs

    def _exposed_link_specs(self, entry_type: str) -> list[tuple[Any, str]]:
        """The ``(LinkSpec, related entry type)`` weak links served as relationships.

        Only weak links declared ``exposed_relationship=True`` whose target
        class is itself served contribute; ``exposed_relationship=False`` links
        contribute nothing anywhere in serving.
        """
        pairs: list[tuple[Any, str]] = []
        for spec in self._schemas[entry_type].links:
            if not spec.exposed_relationship:
                continue
            related = self._entry_type_of.get(spec.target)
            if related is not None:
                pairs.append((spec, related))
        return pairs

    # ------------------------------------------------------------------ the provider contract

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        """Return the definitions of all served entry types.

        This provider is an OPTIMADE serving edge, so each definition is
        returned in its wire form via
        ``EntryTypeDefinition.served_form()`` (idempotent for the
        already-prefixed supplied and auto-generated definitions).

        :return: The served entry-type definitions keyed by entry type.
        """
        return {
            entry_type: (
                self._definitions[entry_type] if entry_type in self._definitions else self._auto_definition(entry_type)
            ).served_form()
            for entry_type in self._classes
        }

    def _auto_definition(self, entry_type: str) -> EntryTypeDefinition:
        return auto_definition(entry_type, self._schemas[entry_type], self._prefix)

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        """Return the served property-to-storage-key mapping for an entry type.

        :param entry_type: The served entry type to inspect.
        :return: The public property names and their storage keys.
        :raises KeyError: If ``entry_type`` is not served.
        """
        self._require_entry_type(entry_type)
        property_keys = {"id": ID_FIELD, "type": "type"}
        property_keys.update({name: name for name, _spec, _fulltype in self._served_specs(entry_type)})
        return property_keys

    def records(self, entry_type: str) -> Iterator[Mapping[str, Any]]:
        """Yield JSON-able records for a served entry type.

        :param entry_type: The served entry type whose records are read.
        :yield: A served record.
        :raises KeyError: If ``entry_type`` is not served.
        """
        cls = self._require_entry_type(entry_type)
        served = self._served_specs(entry_type)
        schema = self._schemas[entry_type]
        searcher = self._store.searcher(only_latest=self._only_latest)
        variable = searcher.variable(cls)
        sid_column = SqlColumn(searcher, variable._alias.c[SID_COLUMN])
        searcher.output(variable, "obj")
        searcher.output(sid_column, "sid")
        matches: Iterator[tuple[Any, Any]] = ((obj, sid) for (obj, sid), _names in searcher)
        for obj, sid in matches:
            row: dict[str, Any] = {
                ID_FIELD: self._id_of(entry_type, int(sid), obj),
                "type": entry_type,
            }
            for name, spec, _fulltype in served:
                row[name] = _json_value(schema, spec, getattr(obj, spec.field))
            yield row

    def _internal_type(self, cls: type) -> str | None:
        """Return the internal (unprefixed) family type name backing ``cls``, if configured."""
        family = self._store._family_for_backing(cls)
        if family is None:
            return None
        internal = getattr(family.family, "type", None)
        return internal if isinstance(internal, str) else None

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        """Return relationships grouped by source entry id.

        Related entries come from stored reference fields, child fields, exposed
        weak links targeting served storable classes, and StrongLink provenance
        edges in both directions: a run's own edges under their forward wire key
        (e.g. ``_httk_has_input``), and the derived reverse edges naming the runs
        that point at each served target under their reverse wire key (e.g.
        ``_httk_is_input``). The reverse view is store-scoped (only this store's
        StrongLink families are scanned) and lineage-level (only a run lineage's
        latest main revision contributes).

        :param entry_type: The served entry type whose relationships are read.
        :return: Related entries keyed by source entry id.
        :raises KeyError: If ``entry_type`` is not served.
        """
        cls = self._require_entry_type(entry_type)
        relation_specs = self._relationship_specs(entry_type)
        link_specs = self._exposed_link_specs(entry_type)
        store = self._store
        strong_families = strong_link_families(store)
        forward_family = next((family for family in strong_families if family.backing is cls), None)
        internal_target = self._internal_type(cls)
        reverse_families = [family for family in strong_families if family.markers] if internal_target else []
        if not relation_specs and not link_specs and forward_family is None and not reverse_families:
            return {}
        if store._missing_tables_for_read((cls,)):
            return {}
        schema = self._schemas[entry_type]
        table = store._table(schema.table_name)
        reference_specs = [(spec, related) for spec, related in relation_specs if spec.role == "reference"]
        child_specs = [(spec, related) for spec, related in relation_specs if spec.role == "child"]
        # The default id lives on the stored object, so relationships must
        # hydrate related records just like a custom id function does.
        fast_ids = False

        def stored_id(connection: Any, related_type: str, related_cls: type, sid: int) -> str:
            return self._id_of(
                related_type,
                sid,
                None if fast_ids else store._fetch(connection, related_cls, sid),
            )

        result: dict[str, tuple[RelatedEntry, ...]] = {}
        with store._read_connection() as connection:
            # The related (target class, target sid, related entry type,
            # description, role, label) tuples per record sid: forward reference
            # fields first (in field order), then child fields (by row order).
            # ``label`` is the weak-link name on the link path, None otherwise.
            related_by_sid: dict[int, list[tuple[type, int, str, str | None, str | None, str | None]]] = {}
            if reference_specs:
                columns = [table.c[SID_COLUMN]] + [table.c[spec.columns[0].name] for spec, _related in reference_specs]
                for row in connection.execute(sqlalchemy.select(*columns).order_by(table.c[SID_COLUMN])):
                    sid = int(row[0])
                    for (spec, related), value in zip(reference_specs, row[1:], strict=True):
                        if value is not None:
                            assert spec.target is not None
                            marker = spec.related
                            related_by_sid.setdefault(sid, []).append(
                                (
                                    spec.target,
                                    int(value),
                                    related,
                                    marker.description if marker is not None else None,
                                    marker.role if marker is not None else None,
                                    None,
                                )
                            )
            for spec, related in child_specs:
                assert spec.child is not None and spec.target is not None
                child_table = store._table(spec.child.table_name)
                parent_column = child_table.c[f"{schema.table_name}_sid"]
                element_column = child_table.c[spec.child.element_columns[0].name]
                statement = sqlalchemy.select(parent_column, element_column).order_by(
                    parent_column, child_table.c[f"{spec.field}_index"]
                )
                marker = spec.related
                for parent_sid, element_sid in connection.execute(statement):
                    related_by_sid.setdefault(int(parent_sid), []).append(
                        (
                            spec.target,
                            int(element_sid),
                            related,
                            marker.description if marker is not None else None,
                            marker.role if marker is not None else None,
                            None,
                        )
                    )
            for link_spec, related in link_specs:
                self._collect_weak_relationships(connection, schema, table, link_spec, related, related_by_sid)
            accumulated: dict[str, list[RelatedEntry]] = {}
            for sid in sorted(related_by_sid):
                record_id = stored_id(connection, entry_type, cls, sid)
                accumulated.setdefault(record_id, []).extend(
                    RelatedEntry(
                        related,
                        stored_id(connection, related, target_cls, target_sid),
                        description=description,
                        role=role,
                        label=label,
                    )
                    for target_cls, target_sid, related, description, role, label in related_by_sid[sid]
                )
            if forward_family is not None:
                self._collect_forward_edges(connection, cls, entry_type, forward_family, stored_id, accumulated)
            if reverse_families and internal_target is not None:
                self._collect_reverse_edges(
                    connection, cls, entry_type, table, internal_target, reverse_families, stored_id, accumulated
                )
            # Dedup by exact RelatedEntry equality, preserving first occurrence;
            # entries differing only in metadata are both kept.
            for record_id, entries in accumulated.items():
                result[record_id] = tuple(dict.fromkeys(entries))
        return result

    def _collect_forward_edges(
        self,
        connection: Any,
        cls: type,
        entry_type: str,
        family: Any,
        stored_id: Callable[[Any, str, type, int], str],
        accumulated: dict[str, list[RelatedEntry]],
    ) -> None:
        """Append a run family's own forward edges to ``accumulated`` by run id.

        :param connection: The open read connection.
        :param cls: The served run backing class.
        :param entry_type: The served run entry type.
        :param family: The run's :class:`~httk.store.backend.sql.provenance_edges.StrongLinkFamily`.
        :param stored_id: The hydrating id resolver ``(connection, entry_type, cls, sid) -> id``.
        :param accumulated: The mutable per-record-id related-entry accumulator.
        :return: None.
        """
        store = self._store
        run_table = store._table(family.schema.table_name)
        run_sids = set(latest_main_run_sids(connection, run_table).values())
        edges_by_sid = forward_run_edges(connection, store, family, run_sids)
        for sid, edges in edges_by_sid.items():
            record_id = stored_id(connection, entry_type, cls, sid)
            for (edge_type, edge_id, label), marker in edges:
                accumulated.setdefault(record_id, []).append(
                    RelatedEntry(
                        wire_type_for_internal(store, edge_type),
                        edge_id,
                        role=marker.role,
                        label=label,
                        relationship=wire_relationship_key(marker.relationship, family.definition_id),
                    )
                )

    def _collect_reverse_edges(
        self,
        connection: Any,
        cls: type,
        entry_type: str,
        table: Any,
        internal_target: str,
        families: Sequence[Any],
        stored_id: Callable[[Any, str, type, int], str],
        accumulated: dict[str, list[RelatedEntry]],
    ) -> None:
        """Append the reverse edges naming runs that point at each served target.

        :param connection: The open read connection.
        :param cls: The served target backing class.
        :param entry_type: The served target entry type.
        :param table: The target family's parent table.
        :param internal_target: The target family's internal (unprefixed) type name.
        :param families: The store's StrongLink families whose reverse edges are derived.
        :param stored_id: The hydrating id resolver ``(connection, entry_type, cls, sid) -> id``.
        :param accumulated: The mutable per-record-id related-entry accumulator.
        :return: None.
        """
        store = self._store
        sid_by_raw: dict[str, int] = {
            str(raw_id): int(sid)
            for sid, raw_id in connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN], table.c["id"]).where(
                    table.c[SID_COLUMN].in_(sorted(set(latest_main_run_sids(connection, table).values())))
                )
            )
        }
        if not sid_by_raw:
            return
        target_ids = sorted(sid_by_raw)
        for family in families:
            reverse = reverse_run_edges(connection, store, family, internal_target, target_ids)
            for raw_id, hits in reverse.items():
                record_id = stored_id(connection, entry_type, cls, sid_by_raw[raw_id])
                for run_id, label, marker in hits:
                    if marker.reverse is None:
                        continue
                    accumulated.setdefault(record_id, []).append(
                        RelatedEntry(
                            family.wire_type,
                            run_id,
                            role=marker.role,
                            label=label,
                            relationship=wire_relationship_key(marker.reverse, family.definition_id),
                        )
                    )

    def _collect_weak_relationships(
        self,
        connection: Any,
        schema: TableSchema,
        table: Any,
        link_spec: Any,
        related: str,
        related_by_sid: dict[int, list[tuple[type, int, str, str | None, str | None, str | None]]],
    ) -> None:
        """Append one exposed weak link's live relationships to ``related_by_sid``.

        Weak links bind lineages: the live latest link rows are scanned in bulk,
        keyed by source ``logical_id``, and each is attached to *every* revision
        row of the source lineage (so the served record carries it whether or
        not ``only_latest`` restricts the revision stream), exactly as reference
        and child relationships attach to their rows. Each target lid resolves
        through the target lineage's latest row — the id is lineage-level.
        """
        store = self._store
        link_table = store._table(link_spec.table_name)
        rows = connection.execute(
            sqlalchemy.select(
                link_table.c[SOURCE_LID_COLUMN],
                link_table.c[TARGET_LID_COLUMN],
                link_table.c[LOGICAL_ID_COLUMN],
                link_table.c[SID_COLUMN],
                link_table.c[RETRACTED_COLUMN],
            )
        )
        targets_by_source = _live_targets_by_source(rows)
        if not targets_by_source:
            return
        # Every revision sid of each live source lineage carries the link.
        source_sids_by_lid: dict[int, list[int]] = {}
        for sid_value, lid_value in connection.execute(
            sqlalchemy.select(table.c[SID_COLUMN], table.c[LOGICAL_ID_COLUMN]).where(
                table.c[LOGICAL_ID_COLUMN].in_(list(targets_by_source))
            )
        ):
            source_sids_by_lid.setdefault(int(lid_value), []).append(int(sid_value))
        # Each live target lineage resolves to its latest revision row.
        target_lids = sorted({tl for tls in targets_by_source.values() for tl in tls})
        target_table = store._table(resolve_schema(link_spec.target).table_name)
        target_sid_by_lid: dict[int, int] = {}
        for lid_value, max_sid in connection.execute(
            sqlalchemy.select(target_table.c[LOGICAL_ID_COLUMN], sqlalchemy.func.max(target_table.c[SID_COLUMN]))
            .where(target_table.c[LOGICAL_ID_COLUMN].in_(target_lids))
            .group_by(target_table.c[LOGICAL_ID_COLUMN])
        ):
            if max_sid is not None:  # a None max is a dangling link; fsck reports it
                target_sid_by_lid[int(lid_value)] = int(max_sid)
        for source_lid, target_list in targets_by_source.items():
            for source_sid in source_sids_by_lid.get(source_lid, ()):
                for target_lid in target_list:
                    target_sid = target_sid_by_lid.get(target_lid)
                    if target_sid is None:
                        continue
                    related_by_sid.setdefault(source_sid, []).append(
                        (link_spec.target, target_sid, related, link_spec.description, link_spec.role, link_spec.name)
                    )


def _json_value(schema: TableSchema, spec: FieldSpec, value: Any) -> Any:
    """The JSON-able served form of one stored field value (see the module docstring)."""
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
        if tensor.dim in ((), (0,)):
            return []
        return tensor.to_floats()
    if spec.codec_name is not None:
        codec = codec_named(spec.codec_name)
        index = _query_index(codec)
        return [codec.encode(element)[index] for element in value]
    return list(value)
