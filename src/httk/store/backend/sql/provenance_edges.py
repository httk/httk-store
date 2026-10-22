"""Store-scoped SQL scanning of :class:`~httk.core.storage.StrongLink` edges.

Run provenance edges (``inputs``/``artifacts``/``outputs`` on
:class:`~httk.core.Run`) are child fields whose element class is an edge record
carrying the string fields ``label``, ``entry_type``, and ``entry_id``. Both the
database-backed :class:`~httk.store.backend.sql.entry_provider.StoreEntryProvider`
and the durable :class:`~httk.store.backend.sql.stored_federation.StoredEntryFederation`
serve these edges as OPTIMADE relationships in both directions. This module holds
the shared discovery and SQL scanning so both serve them identically, and so the
declaring family is always discovered from the store's registered families (never
a hardcoded record class).

The forward view projects one run's own edges. The reverse view is derived,
never stored: it filters ``alt_kind IS NULL`` (run alternatives are separate
lineages that must not emit reverse identifiers) and reduces each run lineage to
its latest main row, so a target entry names only the runs whose latest main
revision still references it.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import sqlalchemy
from httk.core import ALTERNATIVE_KIND_PATTERN
from httk.core.storage import StrongLink

from httk.store.backend.schema import (
    FieldSpec,
    TableSchema,
    _index_column,
    _parent_sid_column,
    resolve_schema,
)
from httk.store.backend.sql.mapping import ALT_KIND_COLUMN, LOGICAL_ID_COLUMN, SID_COLUMN
from httk.store.backend.sql.store import SqlStore, _served_definition
from httk.store.entry_providers import strong_link_markers
from httk.store.storage_layout import EntryFamilyLayout

__all__ = [
    "StrongLinkFamily",
    "forward_run_edges",
    "latest_main_condition",
    "latest_main_run_sids",
    "reverse_run_edges",
    "strong_link_families",
    "wire_type_for_internal",
]


@dataclass(frozen=True)
class StrongLinkFamily:
    """One store-registered family whose backing declares StrongLink edge fields.

    :param internal_type: The family's internal (unprefixed) entry-type name.
    :param wire_type: The family's served (wire) entry-type name.
    :param definition_id: The family's entry-type definition IRI, whose prefix names the wire relationship keys.
    :param backing: The concrete backing record class carrying the edge fields.
    :param schema: The backing's resolved storage schema.
    :param markers: The backing's StrongLink markers keyed by field name.
    """

    internal_type: str
    wire_type: str
    definition_id: str | None
    backing: type
    schema: TableSchema
    markers: Mapping[str, StrongLink]


def _served_name(family: EntryFamilyLayout, internal: str) -> str:
    """Return a family's served (wire) name, falling back to its internal name."""
    served = _served_definition(family)
    return served.name if served is not None else internal


def strong_link_families(store: SqlStore) -> list[StrongLinkFamily]:
    """Return the store's registered families whose backings declare StrongLink fields.

    :param store: The SQL store whose configured families are inspected.
    :return: One :class:`StrongLinkFamily` per StrongLink-declaring backing.
    """
    families: list[StrongLinkFamily] = []
    for family in store.layout.families:
        internal = getattr(family.family, "type", None)
        if not isinstance(internal, str):
            continue
        wire = _served_name(family, internal)
        for backing in family.records:
            markers = strong_link_markers(backing)
            if markers:
                families.append(
                    StrongLinkFamily(internal, wire, family.definition_id, backing, resolve_schema(backing), markers)
                )
    return families


def wire_type_for_internal(store: SqlStore, internal_type: str) -> str:
    """Return the served (wire) entry-type name for an edge's internal target type.

    Resolves via the target family's served definition — the same
    ``EntryTypeDefinition.served_form()`` source used elsewhere — so a target such
    as ``records`` serves as ``_httk_records`` while a standard name (or any
    unregistered target) passes through unchanged.

    :param store: The SQL store whose configured families supply the mapping.
    :param internal_type: The edge's internal (unprefixed) target entry-type name.
    :return: The served (wire) entry-type name, or ``internal_type`` unchanged.
    """
    for family in store.layout.families:
        if family.definition_id is not None and getattr(family.family, "type", None) == internal_type:
            return _served_name(family, internal_type)
    return internal_type


def _edge_columns(store: SqlStore, parent_table: str, spec: FieldSpec) -> tuple[Any, sqlalchemy.Table, Any]:
    """Return ``(child parent-sid column, edge table, child edge-fk column)`` for a strong field."""
    assert spec.child is not None and spec.target is not None
    child_table = store._table(spec.child.table_name)
    parent_column = child_table.c[_parent_sid_column(parent_table)]
    fk_column = child_table.c[spec.child.element_columns[0].name]
    edge_table = store._table(resolve_schema(spec.target).table_name)
    return parent_column, edge_table, fk_column


def forward_run_edges(
    connection: Any,
    store: SqlStore,
    family: StrongLinkFamily,
    run_sids: Collection[int],
    *,
    target_prefixes: Mapping[str, str] | None = None,
    target_backings: Mapping[str, tuple[type, ...]] | None = None,
) -> dict[int, list[tuple[tuple[str, str, str], StrongLink]]]:
    """Return each run's own edges keyed by run sid, in field then row order.

    :param connection: An open read connection to ``store``.
    :param store: The SQL store backing the run family.
    :param family: The StrongLink family whose edges are read.
    :param run_sids: The run row sids to project edges for.
    :param target_prefixes: Mounted prefixes to apply only to locally resolved targets.
    :param target_backings: Mounted target record classes keyed by internal entry type.
    :return: ``run sid -> [((internal_target_type, target_id, label), marker)]``.
    """
    if not run_sids:
        return {}
    prefixes = target_prefixes or {}
    backings = target_backings or {}
    rows: list[tuple[int, str, str, str, str, StrongLink]] = []
    schema = family.schema
    for field_name, marker in family.markers.items():
        spec = schema.field(field_name)
        parent_column, edge_table, fk_column = _edge_columns(store, schema.table_name, spec)
        child_table = store._table(spec.child.table_name)  # type: ignore[union-attr]
        statement = (
            sqlalchemy.select(
                parent_column,
                edge_table.c["entry_type"],
                edge_table.c["entry_id"],
                local_edge_public_id(store, edge_table, prefixes, backings),
                edge_table.c["label"],
            )
            .select_from(child_table.join(edge_table, fk_column == edge_table.c[SID_COLUMN]))
            .where(parent_column.in_(list(run_sids)))
            .order_by(parent_column, child_table.c[_index_column(field_name)])
        )
        for parent_sid, entry_type, raw, public, label in connection.execute(statement):
            rows.append((int(parent_sid), str(entry_type), str(raw), str(public), str(label), marker))
    # Edges still unprefixed may name a mounted alternative: resolve the page's
    # ``<id>~<kind>`` ids with one batched id-indexed lookup per mounted backing.
    pending: dict[str, set[str]] = {}
    for _sid, entry_type, raw, public, _label, _marker in rows:
        if public == raw and prefixes.get(entry_type) and _split_alternative(raw) is not None:
            pending.setdefault(entry_type, set()).add(raw)
    resolved: set[tuple[str, str]] = set()
    for internal, raws in pending.items():
        wanted = {split for raw in raws if (split := _split_alternative(raw)) is not None}
        for table in _mounted_tables(store, backings, internal):
            statement = sqlalchemy.select(table.c["id"], table.c[ALT_KIND_COLUMN]).where(
                table.c["id"].in_(sorted({entry_id for entry_id, _kind in wanted})),
                table.c[ALT_KIND_COLUMN].is_not(None),
            )
            resolved.update(
                (internal, f"{entry_id}~{kind}")
                for entry_id, kind in connection.execute(statement)
                if (str(entry_id), str(kind)) in wanted
            )
    result: dict[int, list[tuple[tuple[str, str, str], StrongLink]]] = {}
    for sid, entry_type, raw, public, label, marker in rows:
        if (entry_type, raw) in resolved:
            public = prefixes[entry_type] + raw
        result.setdefault(sid, []).append(((entry_type, public, label), marker))
    return result


def _split_alternative(raw: str) -> tuple[str, str] | None:
    """Split an ``<id>~<kind>`` edge id at its last ``~`` (kinds never contain one)."""
    entry_id, separator, kind = raw.rpartition("~")
    if separator and entry_id and ALTERNATIVE_KIND_PATTERN.fullmatch(kind) is not None:
        return entry_id, kind
    return None


def _mounted_tables(store: SqlStore, backings: Mapping[str, tuple[type, ...]], internal: str) -> list[Any]:
    """Return the existing tables of the mounted backings for one internal entry type."""
    return [
        store._table(resolve_schema(backing).table_name)
        for backing in backings.get(internal, ())
        if not store._missing_tables_for_read((backing,))
    ]


def local_edge_public_id(
    store: SqlStore,
    edge: Any,
    prefixes: Mapping[str, str],
    backings: Mapping[str, tuple[type, ...]],
) -> Any:
    """Prefix a loose id only when a mounted target table contains its main lineage.

    Indexed, correlated probes avoid either loading every target id or issuing a
    query for every edge. Edges naming an alternative (``<id>~<kind>``) are
    resolved afterwards in :func:`forward_run_edges`, batched per page.

    :param store: The store containing the source and mounted target tables.
    :param edge: The edge table or alias carrying ``entry_type`` and ``entry_id``.
    :param prefixes: Selected public prefixes keyed by internal entry type.
    :param backings: Mounted target record classes keyed by internal entry type.
    :return: The SQL expression for the resolved public id, or the unchanged raw id.
    """
    cases = []
    for internal, prefix in prefixes.items():
        if not prefix:
            continue
        matches = []
        for backing in backings.get(internal, ()):
            if store._missing_tables_for_read((backing,)):
                continue
            target = store._table(resolve_schema(backing).table_name).alias()
            matches.append(
                sqlalchemy.exists(
                    sqlalchemy.select(sqlalchemy.literal(1)).where(
                        target.c["id"] == edge.c["entry_id"],
                        target.c[ALT_KIND_COLUMN].is_(None),
                    )
                )
            )
        if matches:
            cases.append(
                (
                    sqlalchemy.and_(edge.c["entry_type"] == internal, sqlalchemy.or_(*matches)),
                    sqlalchemy.literal(prefix) + edge.c["entry_id"],
                )
            )
    return sqlalchemy.case(*cases, else_=edge.c["entry_id"]) if cases else edge.c["entry_id"]


def edge_public_id_in(
    store: SqlStore,
    edge: Any,
    values: Sequence[Any],
    prefixes: Mapping[str, str],
    backings: Mapping[str, tuple[type, ...]],
) -> Any:
    """Return whether an edge's served public id is one of ``values``.

    The filtering counterpart of the rendering in :func:`forward_run_edges`. Each
    value is resolved in Python: a mount prefix is stripped from the value (never
    prepended to the column), an ``<id>~<kind>`` remainder is split at its last
    ``~``, and the edge matches on ``entry_id = <raw>`` while uncorrelated,
    literal-keyed probes on the target ``id`` index confirm the target exists. A
    value that is no resolvable target matches only an unprefixed loose edge.

    :param store: The store containing the source and mounted target tables.
    :param edge: The edge table or alias carrying ``entry_type`` and ``entry_id``.
    :param values: The filter's public id values.
    :param prefixes: Selected public prefixes keyed by internal entry type.
    :param backings: Mounted target record classes keyed by internal entry type.
    :return: The SQL boolean clause.
    """

    def resolves(internal: str, raw: str) -> Any:
        split = _split_alternative(raw)
        probes = []
        for table in _mounted_tables(store, backings, internal):
            main = table.alias()
            probes.append(
                sqlalchemy.exists(
                    sqlalchemy.select(sqlalchemy.literal(1)).where(
                        main.c["id"] == raw, main.c[ALT_KIND_COLUMN].is_(None)
                    )
                )
            )
            if split is not None:
                alternative = table.alias()
                probes.append(
                    sqlalchemy.exists(
                        sqlalchemy.select(sqlalchemy.literal(1)).where(
                            alternative.c["id"] == split[0], alternative.c[ALT_KIND_COLUMN] == split[1]
                        )
                    )
                )
        return sqlalchemy.or_(*probes) if probes else None

    clauses = []
    for value in values:
        if not isinstance(value, str):
            clauses.append(edge.c["entry_id"] == value)
            continue
        loose_blockers = []
        for internal, prefix in prefixes.items():
            if not prefix:
                continue
            if value.startswith(prefix) and (found := resolves(internal, value[len(prefix) :])) is not None:
                clauses.append(
                    sqlalchemy.and_(edge.c["entry_type"] == internal, edge.c["entry_id"] == value[len(prefix) :], found)
                )
            if (blocked := resolves(internal, value)) is not None:
                loose_blockers.append(sqlalchemy.and_(edge.c["entry_type"] == internal, blocked))
        loose = edge.c["entry_id"] == value
        clauses.append(
            sqlalchemy.and_(loose, sqlalchemy.not_(sqlalchemy.or_(*loose_blockers))) if loose_blockers else loose
        )
    return sqlalchemy.or_(sqlalchemy.false(), *clauses)


def latest_main_run_sids(connection: Any, run_table: sqlalchemy.Table) -> dict[int, int]:
    """Return each run lineage's latest main row sid (``alt_kind IS NULL``).

    :param connection: An open read connection.
    :param run_table: The run family's parent table.
    :return: ``logical_id -> latest main sid``.
    """
    return {
        int(logical_id): int(max_sid)
        for logical_id, max_sid in connection.execute(
            sqlalchemy.select(run_table.c[LOGICAL_ID_COLUMN], sqlalchemy.func.max(run_table.c[SID_COLUMN]))
            .where(run_table.c[ALT_KIND_COLUMN].is_(None))
            .group_by(run_table.c[LOGICAL_ID_COLUMN])
        )
        if max_sid is not None
    }


def latest_main_condition(
    run_table: sqlalchemy.Table, run_alias: sqlalchemy.FromClause
) -> sqlalchemy.ColumnElement[bool]:
    """Return the correlated predicate selecting a run lineage's latest main row.

    The SQL-predicate counterpart of :func:`latest_main_run_sids` for use inside
    a correlated ``EXISTS`` (e.g. reverse-relationship filtering): ``run_alias``
    is a main row (``alt_kind IS NULL``) and no newer main row of the same
    lineage exists. The same constraint the reverse serving path applies through
    a Python-side max-sid-per-lineage scan.

    :param run_table: The run family's parent table (used to alias the newer-row probe).
    :param run_alias: The run parent alias being constrained inside the outer query.
    :return: The ``alt_kind IS NULL`` and latest-of-lineage boolean predicate.
    """
    newer = run_table.alias()
    return sqlalchemy.and_(
        run_alias.c[ALT_KIND_COLUMN].is_(None),
        sqlalchemy.not_(
            sqlalchemy.exists(
                sqlalchemy.select(sqlalchemy.literal(1)).where(
                    newer.c[LOGICAL_ID_COLUMN] == run_alias.c[LOGICAL_ID_COLUMN],
                    newer.c[ALT_KIND_COLUMN].is_(None),
                    newer.c[SID_COLUMN] > run_alias.c[SID_COLUMN],
                )
            )
        ),
    )


def reverse_run_edges(
    connection: Any,
    store: SqlStore,
    family: StrongLinkFamily,
    target_type: str,
    target_ids: Sequence[str],
) -> dict[str, list[tuple[str, str, StrongLink]]]:
    """Return the reverse edges pointing at each target id, keyed by target id.

    Only a run lineage's latest main row contributes (``alt_kind IS NULL``,
    latest sid per lineage); an edge deduped across lineages still yields one
    reverse identifier per referencing lineage. Because the join back through the
    child tables has no index, this is a per-page scan of the run edges.

    :param connection: An open read connection to ``store``.
    :param store: The SQL store backing the run family.
    :param family: The StrongLink family whose reverse edges are derived.
    :param target_type: The edge's internal (unprefixed) target entry-type name to match.
    :param target_ids: The edge-named target ids to match exactly: a main's raw
        stored id, or an alternative's composite ``<id>~<kind>``.
    :return: ``target id -> [(run raw id, edge label, marker)]``, ordered per
        target by ``(run raw id, field/marker order, edge row order)``.
    """
    if not target_ids:
        return {}
    run_table = store._table(family.schema.table_name)
    latest_main = latest_main_run_sids(connection, run_table)
    if not latest_main:
        return {}
    latest_sids = set(latest_main.values())
    run_id_by_sid: dict[int, str] = {
        int(sid): str(value)
        for sid, value in connection.execute(
            sqlalchemy.select(run_table.c[SID_COLUMN], run_table.c["id"]).where(
                run_table.c[SID_COLUMN].in_(sorted(latest_sids))
            )
        )
    }
    # Deterministic reverse order: collect each hit with its sort key
    # (run raw id, marker/field index, edge row index) then sort per target.
    keyed: dict[str, list[tuple[tuple[str, int, int], tuple[str, str, StrongLink]]]] = {}
    # ponytail: per-page reverse scan joins edge -> child -> run with no index on
    # the join-back; mirrors the full link-table scan precedent. Ceiling: run
    # edge/child tables scanned per page. Upgrade: add a (target_type, target_id)
    # covering index feeding a source-lineage filter if run edges grow large.
    for marker_index, (field_name, marker) in enumerate(family.markers.items()):
        spec = family.schema.field(field_name)
        parent_column, edge_table, fk_column = _edge_columns(store, family.schema.table_name, spec)
        child_table = store._table(spec.child.table_name)  # type: ignore[union-attr]
        index_column = child_table.c[_index_column(field_name)]
        statement = (
            sqlalchemy.select(edge_table.c["entry_id"], edge_table.c["label"], parent_column, index_column)
            .select_from(edge_table.join(child_table, fk_column == edge_table.c[SID_COLUMN]))
            .where(
                edge_table.c["entry_type"] == target_type,
                edge_table.c["entry_id"].in_(list(target_ids)),
                parent_column.in_(sorted(latest_sids)),
            )
        )
        for entry_id, label, parent_sid, edge_index in connection.execute(statement):
            run_id = run_id_by_sid.get(int(parent_sid))
            if run_id is None:
                continue
            keyed.setdefault(str(entry_id), []).append(
                ((run_id, marker_index, int(edge_index)), (run_id, str(label), marker))
            )
    return {
        target_id: [hit for _key, hit in sorted(hits, key=lambda item: item[0])] for target_id, hits in keyed.items()
    }
