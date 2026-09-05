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
from httk.core.storage import StrongLink

from httk.store.backend.schema import FieldSpec, TableSchema, resolve_schema
from httk.store.backend.sql.mapping import ALT_KIND_COLUMN, LOGICAL_ID_COLUMN, SID_COLUMN
from httk.store.backend.sql.store import SqlStore, _served_definition
from httk.store.entry_providers import strong_link_markers

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


def _served_name(family: type, internal: str) -> str:
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
        wire = _served_name(family.family, internal)
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
            return _served_name(family.family, internal_type)
    return internal_type


def _edge_columns(store: SqlStore, parent_table: str, spec: FieldSpec) -> tuple[Any, sqlalchemy.Table, Any]:
    """Return ``(child parent-sid column, edge table, child edge-fk column)`` for a strong field."""
    assert spec.child is not None and spec.target is not None
    child_table = store._table(spec.child.table_name)
    parent_column = child_table.c[f"{parent_table}_sid"]
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
    result: dict[int, list[tuple[tuple[str, str, str], StrongLink]]] = {}
    schema = family.schema
    for field_name, marker in family.markers.items():
        spec = schema.field(field_name)
        parent_column, edge_table, fk_column = _edge_columns(store, schema.table_name, spec)
        child_table = store._table(spec.child.table_name)  # type: ignore[union-attr]
        statement = (
            sqlalchemy.select(
                parent_column,
                edge_table.c["entry_type"],
                local_edge_public_id(store, edge_table, target_prefixes or {}, target_backings or {}),
                edge_table.c["label"],
            )
            .select_from(child_table.join(edge_table, fk_column == edge_table.c[SID_COLUMN]))
            .where(parent_column.in_(list(run_sids)))
            .order_by(parent_column, child_table.c[f"{field_name}_index"])
        )
        for parent_sid, entry_type, entry_id, label in connection.execute(statement):
            result.setdefault(int(parent_sid), []).append(((str(entry_type), str(entry_id), str(label)), marker))
    return result


def local_edge_public_id(
    store: SqlStore,
    edge: Any,
    prefixes: Mapping[str, str],
    backings: Mapping[str, tuple[type, ...]],
) -> Any:
    """Prefix a loose id only when a mounted target table contains its main lineage.

    Rendering and filtering share this expression. Indexed, correlated probes
    avoid either loading every target id or issuing a query for every edge.

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
    :param target_ids: The raw stored target ids to match.
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
        index_column = child_table.c[f"{field_name}_index"]
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
