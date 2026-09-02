"""Serve OPTIMADE filters over stored dataclasses through MongoDB.

The property and filter translation layers are backend-neutral. This module
only supplies Mongo-specific field handlers for stored properties and for
relationship ids, plus the semi-join that resolves dotted relationship
filters through a nested Mongo searcher.
"""

from collections.abc import Callable, Mapping
from typing import Any

from httk.core import EntryTypeDefinition
from httk.core.optimade import FilterAst

from httk.store.backend.mongo.mapping import collection_name_for
from httk.store.backend.mongo.searcher import (
    LinkPredicateNode,
    MongoExpression,
    MongoField,
    MongoReference,
    MongoSearcher,
    MongoVariable,
)
from httk.store.backend.mongo.store import MongoStore
from httk.store.backend.schema import resolve_schema
from httk.store.query import Searcher
from httk.store.query.optimade_filters import (
    FilterTranslationError,
    filter_searcher,
    known_unknown_handler,
    simple_property_handlers,
)
from httk.store.served_specs import definition_fulltype, served_specs

__all__ = ["optimade_filter_searcher"]


def _related_sids(store: MongoStore, related_cls: type, values: Any) -> tuple[int, ...]:
    """Resolve served ids to the related collection's physical SIDs."""
    requested = tuple(values)
    if not requested:
        return ()
    strings = tuple(value for value in requested if isinstance(value, str))
    if not strings:
        return tuple(-1 for _value in requested)
    collection = store._database.database[collection_name_for(resolve_schema(related_cls))]
    resolved = {
        str(document["f"]["id"]): int(document["_id"])
        for document in collection.find({"f.id": {"$in": strings}}, {"_id": 1, "f.id": 1}, **store._session_kwargs())
    }
    return tuple(resolved.get(value, -1) if isinstance(value, str) else -1 for value in requested)


def _related_id_has_handlers(
    store: MongoStore, related_cls: type, field: str, role: str
) -> Mapping[str, Callable[..., Any]]:
    """Build the ``'<related_type>.id'`` handler over a reference or child SID field."""

    def has_handler(
        entry: str, ops: Any, values: Any, search_variable: MongoVariable, has_type: str
    ) -> MongoExpression:
        sids = _related_sids(store, related_cls, values)
        relation = getattr(search_variable, field)
        query_field = relation._field if isinstance(relation, MongoReference) else relation
        if not isinstance(query_field, MongoField):
            raise FilterTranslationError("Relationship id field is not queryable.", "internal")
        if role == "reference":

            def member(sid: int) -> MongoExpression:
                # Preserve SQL's outer-join complement: a negated relationship
                # predicate includes rows whose nullable reference is absent.
                return query_field.is_in(sid) & ~(query_field == None)

            if has_type == "HAS_ALL":
                expression = member(sids[0])
                for sid in sids[1:]:
                    expression = expression & member(sid)
                return expression
            if has_type == "HAS_ANY":
                expression = member(sids[0])
                for sid in sids[1:]:
                    expression = expression | member(sid)
                return expression
            if has_type == "HAS_ONLY":
                # A reference is a set of zero or one elements: the empty
                # set satisfies HAS ONLY vacuously, as SqlReference.has_only
                # does. Mongo's nullable membership expression includes the
                # absent/null reference alongside the supplied SIDs.
                return query_field.is_in(None, *sids)
        elif role == "child":
            if has_type == "HAS_ALL":
                expression = query_field.has_any(sids[0])
                for sid in sids[1:]:
                    expression = expression & query_field.has_any(sid)
                return expression
            if has_type == "HAS_ANY":
                return query_field.has_any(*sids)
            if has_type == "HAS_ONLY":
                return query_field.has_only(*sids)
        raise FilterTranslationError("Unexpected set operator type: " + str(has_type), "internal")

    return {"HAS": has_handler}


def _weak_target_lids(store: MongoStore, related_cls: type, values: Any) -> list[int]:
    """Resolve served ids to deduplicated target lineage ids (unknown ids -> sentinel -1).

    All revisions of a target lineage share both id and ``logical_id``, so the
    last-revision-wins concern of the reference resolver does not arise; lids are
    deduplicated to keep the membership predicate minimal.
    """
    requested = tuple(values)
    strings = [value for value in requested if isinstance(value, str)]
    resolved: dict[str, int] = {}
    if strings:
        collection = store._database.database[collection_name_for(resolve_schema(related_cls))]
        for document in collection.find(
            {"f.id": {"$in": strings}}, {"logical_id": 1, "f.id": 1}, **store._session_kwargs()
        ):
            resolved[str(document["f"]["id"])] = int(document["logical_id"])
    lids = [resolved.get(value, -1) if isinstance(value, str) else -1 for value in requested]
    return list(dict.fromkeys(lids))


def _weak_link_id_has_handlers(store: MongoStore, related_cls: type, name: str) -> Mapping[str, Callable[..., Any]]:
    """Build the ``'<related_type>.id'`` handler over an exposed weak link.

    Served ids resolve to target lineage ids, then drive the same no-``$unwind``
    link membership predicate as :class:`~httk.store.backend.mongo.searcher.MongoLinkSet`
    (``has_any`` / ``has_only``), with the SQL branch's op semantics: ``HAS ALL``
    is ANDed per-lid existence, ``HAS ANY`` is one ``$in`` membership, ``HAS
    ONLY`` is the universal ``$nin`` outsider test (a no-links source matches).
    An unknown id resolves to a sentinel that no lineage carries, so it fails
    ``HAS ALL`` / ``HAS ANY`` and is inert in ``HAS ONLY``.
    """

    def has_handler(
        entry: str, ops: Any, values: Any, search_variable: MongoVariable, has_type: str
    ) -> MongoExpression:
        lids = _weak_target_lids(store, related_cls, values)
        # The link set registers the lookup; MongoLinkSet.has_any/has_only accept
        # only stored records, so the lid membership node is built directly (the
        # same LinkPredicateNode those methods emit).
        path = getattr(search_variable.links, name)._path
        if has_type == "HAS_ALL":
            expression = MongoExpression(LinkPredicateNode(path, {"target_lid": {"$in": [lids[0]]}}, False))
            for lid in lids[1:]:
                expression = expression & MongoExpression(
                    LinkPredicateNode(path, {"target_lid": {"$in": [lid]}}, False)
                )
            return expression
        if has_type == "HAS_ANY":
            return MongoExpression(LinkPredicateNode(path, {"target_lid": {"$in": lids}}, False))
        if has_type == "HAS_ONLY":
            return MongoExpression(LinkPredicateNode(path, {"target_lid": {"$nin": lids}}, True))
        raise FilterTranslationError("Unexpected set operator type: " + str(has_type), "internal")

    return {"HAS": has_handler}


def _own_id_handlers() -> Mapping[str, Callable[..., Any]]:
    """Build handlers for a related class's own ``id`` in a nested search."""

    def comparison(entry: str, op: str, value: Any, search_variable: MongoVariable) -> MongoExpression:
        if op not in ("=", "!="):
            raise FilterTranslationError("Ordering comparisons on relationship ids not implemented.", "not-implemented")
        field = search_variable.id
        return field == value if op == "=" else field != value

    return {"comparison": comparison, "unknown": known_unknown_handler}


def optimade_filter_searcher(
    store: MongoStore,
    cls: type,
    filter_string: str | FilterAst,
    *,
    prefix: str = "_httk_",
    definition: EntryTypeDefinition | None = None,
    extra_handlers: Mapping[str, Mapping[str, Callable[..., Any]]] | None = None,
    related_classes: Mapping[str, type] | None = None,
) -> Searcher:
    """Build a Mongo searcher over ``cls`` from an OPTIMADE filter.

    :param store: The Mongo store containing the rows to query.
    :param cls: The storable class whose rows are searched.
    :param filter_string: An OPTIMADE filter string or parsed filter tree.
    :param prefix: The registered prefix used for served property names.
    :param definition: An optional definition supplying additional property types.
    :param extra_handlers: Optional handlers for ids, types, or extra properties.
    :param related_classes: Related entry types and their storable classes.
    :return: A Mongo searcher yielding matching stored instances.
    :raises ~httk.store.query.optimade_filters.FilterTranslationError: If the filter cannot be translated.
    :raises ValueError: If a related class does not match exactly one relationship field.
    """
    schema = resolve_schema(cls)
    served = served_specs(schema, prefix)
    property_fulltypes = {"id": "string", "type": "string"}
    property_fulltypes.update({name: fulltype for name, _spec, fulltype in served})
    property_keys = {"id": "id"}
    property_keys.update({name: spec.field for name, spec, _fulltype in served})
    if definition is not None:
        for name, prop in definition.properties.items():
            if name not in ("id", "type"):
                property_fulltypes[name] = definition_fulltype(prop)

    handlers = simple_property_handlers(cls.__name__, property_keys, property_fulltypes)
    del handlers["type"]
    entry_type = cls.__name__

    relationship_targets: tuple[str, ...] = ()
    resolver: Callable[[str, FilterAst], tuple[str, ...]] | None = None
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
                _related_id_has_handlers(store, related_cls, fields[0].field, fields[0].role)
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
            assert isinstance(nested, MongoSearcher)
            result = nested.results(sid=nested._root_sid_field())
            sids = tuple(int(sid) for sid in result.scalars("sid"))
            if not sids:
                return ()
            collection = store._database.database[collection_name_for(resolve_schema(related[related_type]))]
            return tuple(
                str(document["f"]["id"])
                for document in collection.find({"_id": {"$in": sids}}, {"f.id": 1}, **store._session_kwargs())
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
