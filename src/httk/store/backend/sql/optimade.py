"""Serve OPTIMADE filters over stored dataclasses through the SQL query layer.

:func:`optimade_filter_searcher` builds a :class:`~httk.store.backend.sql.searcher.SqlSearcher`
over a storable class directly from an OPTIMADE filter string, deriving the
recognized property names and their types from the class's resolved
:class:`~httk.store.backend.schema.TableSchema` — the same derivation
:class:`~httk.store.backend.sql.entry_provider.StoreEntryProvider` uses to serve the
class (via the shared :func:`~httk.store.backend.sql.entry_provider.served_specs` /
:func:`~httk.store.backend.sql.entry_provider.auto_definition` helpers), so a filter
that works against the served API works against the store.

Filter property names are the served names: ``{prefix}{field}`` (default
``_httk_<field>``) for every servable stored field. Related storable classes
declared via ``related_classes`` can be filtered relationally —
``references.id HAS "references-3"`` and depth-1 related-property filters like
``references._httk_doi CONTAINS "10.1"`` — over the class's reference or
child-of-storable fields.

"""

from collections.abc import Callable, Mapping
from typing import Any

import sqlalchemy
from httk.core import EntryTypeDefinition
from httk.core.optimade import FilterAst

from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql.mapping import LOGICAL_ID_COLUMN, SID_COLUMN
from httk.store.backend.sql.searcher import SqlColumn, SqlExpression, SqlSearcher, SqlVariable
from httk.store.backend.sql.store import SqlStore
from httk.store.query import Searcher, SearchExpression, SearchVariable
from httk.store.query.optimade_filters import (
    FilterTranslationError,
    filter_searcher,
    known_unknown_handler,
    simple_property_handlers,
)
from httk.store.served_specs import definition_fulltype, served_specs

__all__ = [
    "field_id_has_handlers",
    "optimade_filter_searcher",
]


def field_id_has_handlers(field: str, sids_for: Callable[[Any], Any]) -> Mapping[str, Callable[..., Any]]:
    """A ``'<related_type>.id'`` HAS handler matching a reference/child field against related sids.

    ``sids_for(values)`` returns the sub-select of related-row sids the field's
    foreign key must match for those values; the returned handler composes the
    HAS family over ``field`` with the correct set semantics. ``HAS ALL`` ANDs
    one fresh ``has_any`` per value (each over an independently joined child
    row, so a row must carry a child matching *every* value), ``HAS ANY`` is a
    single ``has_any`` over all values, and ``HAS ONLY`` uses ``has_only`` (a
    row with no such children matches vacuously).

    :param field: The backend reference or child-of-storable field name.
    :param sids_for: A callable mapping filter values to a related-sid sub-select.
    :return: A handler mapping containing the ``HAS`` operation.
    :raises ~httk.store.query.optimade_filters.FilterTranslationError: If an unexpected set operator is supplied.
    """

    def has_handler(
        entry: str, ops: Any, values: Any, search_variable: SearchVariable, has_type: str
    ) -> SearchExpression:
        if has_type == 'HAS_ALL':
            # A fresh child alias per conjunct (each ``getattr`` access joins a
            # fresh alias) so the ANDed predicates constrain independent rows.
            search = getattr(search_variable, field).has_any(sids_for(values[:1]))
            for value in values[1:]:
                search = search & getattr(search_variable, field).has_any(sids_for([value]))
            return search
        if has_type == 'HAS_ANY':
            return getattr(search_variable, field).has_any(sids_for(values))
        if has_type == 'HAS_ONLY':
            return getattr(search_variable, field).has_only(_never_empty(sids_for(values)))
        raise FilterTranslationError("Unexpected set operator type: " + str(has_type), "internal")

    return {'HAS': has_handler}


def _never_empty(selected: Any) -> Any:
    """Pad a HAS ONLY membership sub-select so it is never the empty set.

    ``has_only`` renders as "no value is outside the allowed set", i.e.
    ``value NOT IN (<sub-select>)``, and relies on NULL propagation to keep a
    referent-less row (or a no-child LEFT JOIN row) out of the outsider count.
    SQL breaks that propagation for an empty set: ``NULL NOT IN ()`` is TRUE, so
    ids matching no row of the target table would wrongly drop exactly the rows
    that match vacuously. Padding with an impossible sid/lineage (``-1`` is never
    minted) keeps the set non-empty without matching anything.

    :param selected: The sub-select of allowed sids or target lineages.
    :return: The same sub-select padded with one impossible member.
    """
    return sqlalchemy.union_all(selected, sqlalchemy.select(sqlalchemy.literal(-1)))


def _related_id_has_handlers(store: SqlStore, related_cls: type, field: str) -> Mapping[str, Callable[..., Any]]:
    """The ``'<related_type>.id'`` HAS handler over a reference or child-of-storable field."""
    table = store._table(resolve_schema(related_cls).table_name)
    return field_id_has_handlers(
        field, lambda values: sqlalchemy.select(table.c[SID_COLUMN]).where(table.c["id"].in_(values))
    )


def _weak_link_id_has_handlers(store: SqlStore, related_cls: type, name: str) -> Mapping[str, Callable[..., Any]]:
    """The ``'<related_type>.id'`` HAS handler over an exposed weak link.

    Each served id resolves to the target lineage id through a subquery over the
    target table's physical ``id`` column, matched against the link's live
    latest target lineages. ``HAS ALL`` composes as ANDed per-value ``has_any``
    over fresh link aliases (mirroring :func:`field_id_has_handlers`); ``HAS ONLY`` uses
    the set machinery's ``has_only`` (a no-links source matches vacuously), which
    expresses the multi-valued semantics faithfully via the subquery passthrough.
    """

    def has_handler(
        entry: str, ops: Any, values: Any, search_variable: SearchVariable, has_type: str
    ) -> SearchExpression:
        table = store._table(resolve_schema(related_cls).table_name)

        def lineages(selected: Any) -> Any:
            return sqlalchemy.select(table.c[LOGICAL_ID_COLUMN]).where(table.c["id"].in_(selected))

        if has_type == 'HAS_ALL':
            # Fresh link alias per conjunct so the ANDed predicates constrain
            # independent linked rows (a source linked to every id).
            search = getattr(search_variable.links, name).has_any(lineages(values[:1]))
            for value in values[1:]:
                search = search & getattr(search_variable.links, name).has_any(lineages([value]))
            return search
        if has_type == 'HAS_ANY':
            return getattr(search_variable.links, name).has_any(lineages(values))
        if has_type == 'HAS_ONLY':
            return getattr(search_variable.links, name).has_only(_never_empty(lineages(values)))
        raise FilterTranslationError("Unexpected set operator type: " + str(has_type), "internal")

    return {'HAS': has_handler}


def _own_id_handlers() -> Mapping[str, Callable[..., Any]]:
    """Handlers serving the related class's own ``id`` property in a nested sub-search."""

    def comparison(entry: str, op: str, value: Any, search_variable: SqlVariable) -> SqlExpression:
        if op not in ('=', '!='):
            raise FilterTranslationError("Ordering comparisons on relationship ids not implemented.", "not-implemented")
        column = SqlColumn(search_variable._searcher, search_variable._alias.c["id"])
        return column == value if op == "=" else column != value

    return {'comparison': comparison, 'unknown': known_unknown_handler}


def optimade_filter_searcher(
    store: SqlStore,
    cls: type,
    filter_string: str | FilterAst,
    *,
    prefix: str = "_httk_",
    definition: EntryTypeDefinition | None = None,
    extra_handlers: Mapping[str, Mapping[str, Callable[..., Any]]] | None = None,
    related_classes: Mapping[str, type] | None = None,
) -> Searcher:
    """Build a searcher over the stored rows of ``cls`` from an OPTIMADE filter.

    ``filter_string`` is an OPTIMADE filter string or an already-parsed
    :py:type:`~httk.core.optimade.FilterAst`. The returned searcher outputs the matching
    stored instances (``item[0][0]`` per match).

    Property names: every servable stored field of ``cls`` (per
    :func:`~httk.store.backend.sql.entry_provider.served_specs`) is filterable as
    ``{prefix}{field}``, with its schema-derived fulltype driving constant
    conversion and handler dispatch (rational fields compare on their
    documented-approximate float query column). Unknown names carrying
    ``prefix`` raise :class:`~httk.store.query.optimade_filters.FilterTranslationError`
    (``"unrecognized-property"``); other unknown names match nothing, per the
    OPTIMADE specification. Unprefixed property names (beyond ``id``/``type``)
    are recognized only when a ``definition`` describes them — and even then
    they translate only if ``extra_handlers`` supplies their handlers, since
    the store knows no column for them. The OPTIMADE core ``id`` and ``type``
    properties are recognized but **not supported** without ``extra_handlers``
    entries (a store row has no served id: ids are minted at serving time), so
    filtering on them raises ``"not-implemented"``.

    Relationships: ``related_classes`` maps relationship-type names to
    storable classes; each must be the target of exactly one reference or
    child-of-storable (``list[Target]``) field of ``cls`` (anything else
    raises :class:`ValueError`). For each such ``(rtype, rcls)``:

    - ``<rtype>.id HAS "<stored id>"`` resolves the supplied stored public id
      through a correlated subquery over the related table's physical ``id``
      column, then compares its sid to the reference (or child-element)
      column.  The same stored-id rule is used by
      :class:`~httk.store.backend.sql.entry_provider.StoreEntryProvider`.
    - Depth-1 related-property filters (``<rtype>._httk_doi CONTAINS "10.1"``,
      ``<rtype>.id != "..."``, ``<rtype>._httk_year IS KNOWN``, ...) resolve by
      a two-phase semi-join: a nested ``optimade_filter_searcher`` over
      ``rcls`` (with the same ``prefix``; no further relationship nesting)
      collects the matching related sids, which are then matched as
      ``<rtype>.id HAS ANY ...``. Each dotted filter node resolves
      independently — see :func:`~httk.store.query.optimade_filters.translate_filter_ast`.

    ``extra_handlers`` entries are merged over the derived handler table last
    (so they can also override derived handlers); extra property names that are
    otherwise unknown are recognized with fulltype ``"unknown"`` (filter
    constants pass through unconverted).

    :param store: The SQL store containing the rows to query.
    :param cls: The storable class whose rows are searched.
    :param filter_string: The OPTIMADE filter string or parsed filter tree.
    :param prefix: The registered prefix used for served property names.
    :param definition: An optional entry definition supplying additional property types.
    :param extra_handlers: Optional handlers for ids, types, or additional properties.
    :param related_classes: Related entry types and their storable classes.
    :return: A searcher yielding the matching stored instances.
    :raises ~httk.store.query.optimade_filters.FilterTranslationError: When the filter cannot be translated.
    :raises ~httk.core.optimade.ParserSyntaxError: When a filter string does not parse.
    :raises ValueError: When a ``related_classes`` entry does not match exactly one
        reference or child-of-storable field of ``cls``.
    """
    schema = resolve_schema(cls)
    served = served_specs(schema, prefix)
    property_fulltypes = {"id": "string", "type": "string"}
    property_fulltypes.update({name: fulltype for name, _spec, fulltype in served})
    property_keys = {"id": "id"}
    property_keys.update({name: spec.field for name, spec, _fulltype in served})
    if store.store_timestamps:
        property_fulltypes[f"{prefix}store_timestamp"] = "integer"
        property_keys[f"{prefix}store_timestamp"] = "store_timestamp"
    # The store-managed lineage id is unconditional: every parent row carries a
    # ``logical_id`` regardless of the timestamp option, and it needs no unit
    # scaling (a plain integer resolved through ``SqlVariable.logical_id``).
    property_fulltypes[f"{prefix}logical_id"] = "integer"
    property_keys[f"{prefix}logical_id"] = "logical_id"
    if definition is not None:
        for name, prop in definition.properties.items():
            if name in ("id", "type"):
                continue
            property_fulltypes[name] = definition_fulltype(prop)

    handlers = simple_property_handlers(cls.__name__, property_keys, property_fulltypes)
    # The default type handler is a serving-layer constant, but the stored id
    # is a real physical column and remains queryable.
    del handlers["type"]
    entry_type = cls.__name__

    relationship_targets: tuple[str, ...] = ()
    resolver = None
    if related_classes:
        related = dict(related_classes)
        for related_type, related_cls in related.items():
            fields = [
                spec for spec in schema.fields if spec.target is related_cls and spec.role in ("reference", "child")
            ]
            links = [spec for spec in schema.links if spec.exposed_relationship and spec.target is related_cls]
            total = len(fields) + len(links)
            if total != 1:
                raise ValueError(
                    f"related_classes entry {related_type!r} ({related_cls.__name__}) matches "
                    f"{'no' if not total else str(total)} reference or child-of-storable field"
                    f"{'' if total == 1 else 's'} or exposed weak link{'' if total == 1 else 's'} "
                    f"of {cls.__name__}; exactly one is required"
                )
            handlers[f"{related_type}.id"] = (
                _related_id_has_handlers(store, related_cls, fields[0].field)
                if fields
                else _weak_link_id_has_handlers(store, related_cls, links[0].name)
            )
        relationship_targets = tuple(related)

        def resolve_related(related_type: str, sub_ast: FilterAst) -> tuple[str, ...]:
            nested = optimade_filter_searcher(
                store,
                related[related_type],
                sub_ast,
                prefix=prefix,
                extra_handlers={"id": _own_id_handlers()},
            )
            assert isinstance(nested, SqlSearcher)
            sid_column = nested._variables[0].sid
            assert isinstance(sid_column, SqlColumn)
            nested._outputs.clear()
            nested.output(sid_column, "sid")
            related_table = store._table(resolve_schema(related[related_type]).table_name)
            values = tuple(int(item[0]) for item, _names in nested)
            with store._read_connection() as connection:
                return tuple(
                    str(value[0])
                    for value in connection.execute(
                        sqlalchemy.select(related_table.c["id"]).where(related_table.c[SID_COLUMN].in_(values))
                    )
                )

        resolver = resolve_related

    if extra_handlers:
        for name in extra_handlers:
            if "." not in name:
                property_fulltypes.setdefault(name, "unknown")
        handlers.update(extra_handlers)

    return filter_searcher(
        store,
        cls,
        filter_string,
        entry_type=entry_type,
        property_fulltypes=property_fulltypes,
        handlers=handlers,
        recognized_prefixes=(prefix,),
        relationship_targets=relationship_targets,
        related_property_resolver=resolver,
    )
