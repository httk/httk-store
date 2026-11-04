"""Versioned physical layout for :class:`httk.store.backend.sql.store.SqlStore`."""

import dataclasses
import difflib
import json
from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal

import sqlalchemy

from httk.store.backend.schema import (
    TableSchema,
    _index_column,
    _parent_sid_column,
    _present_column,
    _sid_sequence_name,
    resolve_schema,
)
from httk.store.backend.sql.mapping import (
    IndexPolicy,
    backing_dispatch_column_name,
    dispatch_table_for,
    entry_dispatch_table_name,
    identity_owner_tables,
    table_for,
)
from httk.store.storage_layout import (
    EntryFamilyDeclaration,
    EntryFamilyLayout,
    StorageLayout,
    StorageLayoutUpgradeRequiredError,
    _merge_storage_layouts,
    _walk_closure,
    declaration_json,
    validate_entry_id_fields,
)
from httk.store.storage_layout import (
    normalize_entry_families as _normalize_entry_families,
)
from httk.store.storage_layout import (
    normalize_entry_records as _normalize_entry_records,
)
from httk.store.storage_layout import (
    normalize_entry_types as _normalize_entry_types,
)

__all__ = [
    "METADATA_TABLE_NAME",
    "STORAGE_PROTOCOL_VERSION",
    "WRITE_PROFILE_VOCABULARY",
    "BackendFacts",
    "EntryFamilyLayout",
    "StorageLayout",
    "StorageLayoutUpgradeRequiredError",
    "StoreUnderConstructionError",
    "actual_columns",
    "actual_index_names",
    "actual_schema_objects",
    "actual_table_names",
    "backend_facts_for_dialect",
    "declaration_json",
    "expected_metadata",
    "metadata_table_for",
    "normalize_entry_declaration",
    "normalize_entry_families",
    "normalize_entry_records",
    "normalize_entry_types",
    "read_store_metadata",
]

STORAGE_PROTOCOL_VERSION: Final = "2"
# The value is the major store generation only, compared for strict equality on
# reopen; it is never parsed. Bump it to "3" solely on a breaking change to the
# physical SqlStore layout that an existing store could not be reopened against.
"""The persisted SqlStore layout protocol implemented by this package."""

METADATA_TABLE_NAME: Final = "_httk_store_metadata"
"""Reserved key/value table holding the store protocol and entry declaration."""

_METADATA_PROTOCOL_KEY: Final = "protocol"
_METADATA_DECLARATION_KEY: Final = "entry_declaration"
_RESERVED_PREFIX: Final = "_httk_"
WRITE_PROFILE_VOCABULARY: Final = frozenset({"transactional", "degraded", "bulk-fenced"})

_INDEX_POLICY_KEY: Final = "index_policy"
"""Metadata key of a store's index policy, stamped only when it differs from all indexes."""


class StoreUnderConstructionError(RuntimeError):
    """A new open found an interrupted empty-store bulk ingest.

    Crash window for new SQLite/DuckDB opens: before the marker commits the
    old clean state remains accepted; after the marker and through ingest,
    finalize, or before marker clear the store is rejected; after clear it is
    accepted again.  The marker is intentionally not a resume protocol.

    ClickHouse marker residue is fail-closed: the default recovery is to drop
    the database and re-ingest. Clearing the marker is valid only after an
    operator has restored and verified the declared empty-store invariant.
    """


@dataclasses.dataclass(frozen=True)
class BackendFacts:
    """Dialect capabilities used by the SQL storage protocol."""

    transactional_ddl: bool
    transactional_dml: bool
    supports_sequences: bool
    atomic_upsert: bool
    serial_stage_format: Literal["sqlite", "duckdb-attach", "parquet"]
    parallel_shard_format: Literal["sqlite", "parquet"]
    supports_deferred_finalize: bool
    supports_degraded: bool
    write_profiles: tuple[str, ...]
    metadata_backend: Literal["table", "keepermap"]
    supports_incremental_save: bool
    system_catalog: Literal["sqlite", "duckdb", "clickhouse", "postgresql"]
    stage_load: Literal["attach", "duckdb-views", "client-stream"]
    finalize_map_maintenance: Literal["update", "swap"]
    supports_adhoc_indexes: bool
    default_indexes: Literal["all", "minimal"]


_BACKEND_FACTS: Final[dict[str, BackendFacts]] = {
    "sqlite": BackendFacts(
        transactional_ddl=False,
        transactional_dml=True,
        supports_sequences=False,
        atomic_upsert=True,
        serial_stage_format="sqlite",
        parallel_shard_format="sqlite",
        supports_deferred_finalize=True,
        supports_degraded=True,
        write_profiles=("transactional", "degraded"),
        metadata_backend="table",
        supports_incremental_save=True,
        system_catalog="sqlite",
        stage_load="attach",
        finalize_map_maintenance="update",
        supports_adhoc_indexes=True,
        default_indexes="all",
    ),
    "duckdb": BackendFacts(
        transactional_ddl=True,
        transactional_dml=True,
        supports_sequences=True,
        atomic_upsert=True,
        serial_stage_format="duckdb-attach",
        parallel_shard_format="parquet",
        supports_deferred_finalize=True,
        supports_degraded=False,
        write_profiles=("transactional",),
        metadata_backend="table",
        supports_incremental_save=True,
        system_catalog="duckdb",
        stage_load="duckdb-views",
        finalize_map_maintenance="update",
        supports_adhoc_indexes=True,
        default_indexes="minimal",
    ),
    "postgresql": BackendFacts(
        transactional_ddl=True,
        transactional_dml=True,
        supports_sequences=True,
        atomic_upsert=True,
        serial_stage_format="sqlite",
        parallel_shard_format="sqlite",
        supports_deferred_finalize=True,
        supports_degraded=False,
        write_profiles=("transactional",),
        metadata_backend="table",
        supports_incremental_save=True,
        system_catalog="postgresql",
        stage_load="attach",
        finalize_map_maintenance="update",
        supports_adhoc_indexes=True,
        default_indexes="all",
    ),
    "clickhousedb": BackendFacts(
        transactional_ddl=False,
        transactional_dml=False,
        supports_sequences=False,
        atomic_upsert=False,
        serial_stage_format="parquet",
        parallel_shard_format="parquet",
        supports_deferred_finalize=True,
        supports_degraded=False,
        write_profiles=("bulk-fenced",),
        metadata_backend="keepermap",
        supports_incremental_save=False,
        system_catalog="clickhouse",
        stage_load="client-stream",
        finalize_map_maintenance="swap",
        supports_adhoc_indexes=False,
        default_indexes="all",
    ),
}


def backend_facts_for_dialect(dialect_name: str) -> BackendFacts:
    """Resolve the hardcoded protocol facts for one supported dialect."""
    try:
        return _BACKEND_FACTS[dialect_name]
    except KeyError as error:
        raise ValueError(f"SqlStore layout validation does not support dialect {dialect_name!r}") from error


def normalize_entry_records(entry_records: Mapping[type, type | tuple[type, ...]]) -> StorageLayout:
    """Normalize a declaration and apply SQL physical-name validation."""
    layout = _normalize_entry_records(entry_records)
    _validate_physical_names(layout)
    return layout


def normalize_entry_families(entry_families: Sequence[EntryFamilyDeclaration]) -> StorageLayout:
    """Normalize application-owned declarations and apply SQL physical-name validation."""
    layout = _normalize_entry_families(entry_families)
    _validate_physical_names(layout)
    return layout


def normalize_entry_types(records: Sequence[type]) -> StorageLayout:
    """Normalize decorated entry-record classes and validate SQL names.

    :param records: Explicit decorated application record classes.
    :return: The complete layout, including referenced entry families.
    :raises TypeError: If the declaration has unsupported types.
    :raises ValueError: If entry declarations or physical names conflict.
    """
    layout = _normalize_entry_types(records)
    _validate_physical_names(layout)
    return layout


def normalize_entry_declaration(
    entry_records: Mapping[type, type | tuple[type, ...]] | None,
    entry_families: Sequence[EntryFamilyDeclaration] | None,
    records: Sequence[type] | None = None,
) -> StorageLayout | None:
    """Merge store declarations and validate SQL names."""
    layouts = []
    if entry_records is not None:
        layouts.append(_normalize_entry_records(entry_records))
    if entry_families is not None:
        layouts.append(_normalize_entry_families(entry_families))
    if records is not None:
        layouts.append(_normalize_entry_types(records))
    if not layouts:
        return None
    layout = _merge_storage_layouts(*layouts)
    validate_entry_id_fields(layout)
    _validate_physical_names(layout)
    return layout


def _layout_from_declaration(value: str) -> StorageLayout:
    """Parse a declaration and apply SQL physical-name validation."""
    from httk.store.storage_layout import _layout_from_declaration as parse_declaration

    layout = parse_declaration(value)
    _validate_physical_names(layout)
    return layout


def expected_metadata(
    layout: StorageLayout, *, store_timestamps: bool = True, indexes: IndexPolicy | None = None
) -> sqlalchemy.MetaData:
    """Return SQLAlchemy metadata for all protocol-owned tables of ``layout``."""
    metadata = sqlalchemy.MetaData()
    metadata_table_for(metadata)
    identity_owner_tables(metadata)
    for family in layout.families:
        schemas = tuple(resolve_schema(record) for record in family.records)
        for schema in schemas:
            table_for(schema, metadata, store_timestamps=store_timestamps, indexes=indexes)
        if len(schemas) > 1:
            dispatch_table_for(family.name, tuple(zip(family.record_names, schemas, strict=True)), metadata)
    return metadata


def _index_policy_for(
    layout: StorageLayout, indexes: Literal["all", "minimal"], skip_indexes: Collection[str]
) -> IndexPolicy:
    """Return the :class:`~httk.store.backend.sql.mapping.IndexPolicy` of a store's index settings.

    :param layout: The store layout; its entry-family backing tables keep their bookkeeping indexes.
    :param indexes: ``"all"`` or ``"minimal"``.
    :param skip_indexes: Further non-unique index names to leave out.
    :return: The policy.
    """
    entry_tables = frozenset(
        resolve_schema(record).table_name for family in layout.families for record in family.records
    )
    return IndexPolicy(minimal=indexes == "minimal", entry_tables=entry_tables, skip=frozenset(skip_indexes))


def _parse_index_policy(value: str | None) -> tuple[Literal["all", "minimal"], frozenset[str]] | None:
    """Parse a stamped index policy (absent means all indexes), returning ``None`` when malformed.

    :param value: The stamped ``index_policy`` metadata value, or ``None`` when absent.
    :return: The ``indexes`` setting and the skipped index names.
    """
    if value is None:
        return "all", frozenset()
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"indexes", "skip_indexes"}:
        return None
    indexes, skip = parsed["indexes"], parsed["skip_indexes"]
    if indexes not in ("all", "minimal") or not isinstance(skip, list) or not all(isinstance(n, str) for n in skip):
        return None
    return indexes, frozenset(skip)


def _validate_skip_indexes(layout: StorageLayout, names: Collection[str], *, store_timestamps: bool) -> None:
    """Refuse ``skip_indexes`` names that are integrity constraints or that ``layout`` never creates.

    :param layout: The store layout.
    :param names: The requested index names.
    :param store_timestamps: Whether the store keeps timestamps (and so their indexes).
    :raises ValueError: If a name is a unique index or primary key, or is not an index of ``layout``.
    """
    metadata = expected_metadata(layout, store_timestamps=store_timestamps)
    declared = {str(index.name): bool(index.unique) for table in metadata.tables.values() for index in table.indexes}
    for name in sorted(names):
        primary = name.startswith("sqlite_autoindex_") or (name.endswith("_pkey") and name[:-5] in metadata.tables)
        if declared.get(name) or primary:
            raise ValueError(
                f"cannot skip index {name!r}: unique indexes and primary keys are integrity constraints "
                "and cannot be skipped"
            )
        if name not in declared:
            close = difflib.get_close_matches(name, [other for other, unique in declared.items() if not unique])
            hint = f"; close matches: {', '.join(close)}" if close else ""
            raise ValueError(f"cannot skip index {name!r}: the declared records never create it{hint}")


def metadata_table_for(metadata: sqlalchemy.MetaData) -> sqlalchemy.Table:
    """Return the reserved protocol key/value table in ``metadata``."""
    existing = metadata.tables.get(METADATA_TABLE_NAME)
    if existing is not None:
        return existing
    return sqlalchemy.Table(
        METADATA_TABLE_NAME,
        metadata,
        sqlalchemy.Column("key", sqlalchemy.Text, primary_key=True, nullable=False),
        sqlalchemy.Column("value", sqlalchemy.Text, nullable=False),
        info={"httk_metadata": True},
    )


def actual_schema_objects(connection: sqlalchemy.Connection) -> Mapping[str, frozenset[str]]:
    """Return application schema-object names mapped to their stable object kinds.

    The DuckDB SQLAlchemy inspector presently routes column inspection through
    a PostgreSQL catalogue relation DuckDB does not expose, so the whole
    layout path intentionally uses the dialect catalogues directly.
    """
    facts = backend_facts_for_dialect(connection.dialect.name)
    if facts.system_catalog == "sqlite":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
            )
        )
    elif facts.system_catalog == "duckdb":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT table_name, lower(table_type) FROM information_schema.tables "
                "WHERE table_catalog = current_database() AND table_schema = current_schema() "
                "UNION ALL "
                "SELECT sequence_name, 'sequence' FROM duckdb_sequences() "
                "WHERE database_name = current_database() AND schema_name = current_schema()"
            )
        )
    elif facts.system_catalog == "postgresql":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT table_name, lower(table_type) FROM information_schema.tables "
                "WHERE table_catalog = current_database() AND table_schema = current_schema() "
                "UNION ALL "
                "SELECT sequence_name, 'sequence' FROM information_schema.sequences "
                "WHERE sequence_catalog = current_database() AND sequence_schema = current_schema() "
                "UNION ALL "
                # Materialized views are absent from information_schema.tables; a
                # stray one must still count as a non-empty schema object.
                "SELECT matviewname, 'view' FROM pg_matviews WHERE schemaname = current_schema()"
            )
        )
    else:
        from httk.store.backend.clickhouse.support import actual_schema_objects as clickhouse_schema_objects

        return clickhouse_schema_objects(connection)
    result: dict[str, set[str]] = {}
    for name, kind in rows:
        result.setdefault(str(name), set()).add(str(kind).lower().replace("base ", ""))
    return MappingProxyType({name: frozenset(kinds) for name, kinds in result.items()})


def actual_table_names(connection: sqlalchemy.Connection) -> frozenset[str]:
    """Return application base-table names without SQLAlchemy reflection."""
    return frozenset(name for name, kinds in actual_schema_objects(connection).items() if "table" in kinds)


def actual_columns(connection: sqlalchemy.Connection, table_name: str) -> frozenset[str]:
    """Return the physical column names of ``table_name`` without SQLAlchemy reflection.

    The DuckDB SQLAlchemy inspector routes column inspection through a
    PostgreSQL catalogue relation DuckDB does not expose (see
    :func:`actual_schema_objects`), so this reads the dialect catalogues directly.

    :param connection: The open connection whose dialect selects the catalogue.
    :param table_name: The table whose columns are read.
    :return: The set of physical column names, empty when the table is absent.
    """
    facts = backend_facts_for_dialect(connection.dialect.name)
    if facts.system_catalog == "sqlite":
        # PRAGMA cannot bind parameters, so the name is quoted, not bound.
        quoted = connection.dialect.identifier_preparer.quote(table_name)
        pragma_rows = connection.execute(sqlalchemy.text(f"PRAGMA table_info({quoted})")).mappings().all()
        return frozenset(str(row["name"]) for row in pragma_rows)
    if facts.system_catalog in ("duckdb", "postgresql"):
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_catalog = current_database() AND table_schema = current_schema() "
                "AND table_name = :name"
            ),
            {"name": table_name},
        ).all()
        return frozenset(str(row[0]) for row in rows)
    from httk.store.backend.clickhouse.support import actual_columns as clickhouse_actual_columns

    return frozenset(clickhouse_actual_columns(connection, table_name))


def actual_index_names(connection: sqlalchemy.Connection) -> frozenset[str]:
    """Return the index names in the current schema without SQLAlchemy reflection.

    The DuckDB SQLAlchemy inspector does not reflect indexes, so this reads the
    dialect catalogues directly, like :func:`actual_schema_objects`.

    :param connection: The open SQLite, DuckDB, or PostgreSQL connection.
    :return: The set of index names.
    :raises ValueError: If the dialect has no index catalogue here (ClickHouse).
    """
    catalog = backend_facts_for_dialect(connection.dialect.name).system_catalog
    queries = {
        "sqlite": "SELECT name FROM sqlite_master WHERE type = 'index'",
        "duckdb": "SELECT index_name FROM duckdb_indexes() "
        "WHERE database_name = current_database() AND schema_name = current_schema()",
        "postgresql": "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()",
    }
    if catalog not in queries:
        raise ValueError(f"no index catalogue for {connection.dialect.name!r}")
    return frozenset(str(row[0]) for row in connection.execute(sqlalchemy.text(queries[catalog])))


def read_store_metadata(connection: sqlalchemy.Connection) -> Mapping[str, str] | None:
    """Read the marker values, or return ``None`` when no metadata table exists."""
    if METADATA_TABLE_NAME not in actual_table_names(connection):
        return None
    table = metadata_table_for(sqlalchemy.MetaData())
    rows = connection.execute(sqlalchemy.select(table.c.key, table.c.value)).all()
    result: dict[str, str] = {}
    for key, value in rows:
        if not isinstance(key, str) or not isinstance(value, str) or key in result:
            raise ValueError("stored metadata rows are malformed")
        result[key] = value
    return MappingProxyType(result)


class _UniqueNames:
    """Reject two distinct sources whose physical names are the same identifier.

    Every derived name is hash-shortened to 63 characters at resolution, so no
    dialect rejects or truncates it; what remains possible is an exact
    collision: a hash collision, or a declared name equal to another's
    shortened form.
    """

    def __init__(self, what: str) -> None:
        self._what = what
        self._seen: dict[str, str] = {}

    def add(self, name: str, source: str) -> None:
        other = self._seen.setdefault(name, source)
        if other != source:
            raise ValueError(f"{self._what}: {other} and {source} both map to the identifier {name!r}")


def _validate_physical_names(layout: StorageLayout) -> None:
    owners: dict[str, type] = {}
    tables = _UniqueNames("table names")

    def check(schema: TableSchema) -> None:
        record = schema.cls
        label = f"{record.__module__}.{record.__qualname__}"
        names = [schema.table_name]
        names.extend(spec.child.table_name for spec in schema.fields if spec.child is not None)
        for name in names:
            if name.startswith(_RESERVED_PREFIX):
                raise ValueError(f"record {record.__name__} claims reserved SqlStore table name {name!r}")
            previous = owners.get(name)
            if previous is not None and previous is not record:
                raise ValueError(
                    f"records {previous.__name__} and {record.__name__} collide on physical table name {name!r}"
                )
            owners[name] = record
        tables.add(schema.table_name, f"record {label}")
        # PostgreSQL tables, sequences and indexes share one namespace.
        tables.add(_sid_sequence_name(schema.table_name), f"the sid sequence of record {label}")
        parent_columns = _UniqueNames(f"columns of table {schema.table_name!r}")
        for spec in schema.fields:
            if spec.child is not None:
                tables.add(spec.child.table_name, f"child table of {label}.{spec.field}")
                child_columns = _UniqueNames(f"columns of table {spec.child.table_name!r}")
                child_columns.add(_parent_sid_column(schema.table_name), "the parent sid column")
                child_columns.add(_index_column(spec.field), "the index column")
                for position, column in enumerate(spec.child.element_columns):
                    child_columns.add(column.name, f"{label}.{spec.field} element column {position}")
                if spec.optional:
                    parent_columns.add(_present_column(spec.field), f"the presence column of {label}.{spec.field}")
            else:
                for position, column_spec in enumerate(spec.columns):
                    parent_columns.add(column_spec.name, f"{label}.{spec.field} column {position}")
        for link in schema.links:
            tables.add(link.table_name, f"weak link {label}.{link.name}")
            tables.add(_sid_sequence_name(link.table_name), f"the sid sequence of weak link {label}.{link.name}")

    _walk_closure(layout, check)
    for family in layout.families:
        dispatch_name = entry_dispatch_table_name(family.name) if len(family.records) > 1 else None
        if dispatch_name is not None:
            if dispatch_name in owners:
                raise ValueError(
                    f"entry family {family.name!r} dispatch table collides with record table {dispatch_name!r}"
                )
            owners[dispatch_name] = family.family
            tables.add(dispatch_name, f"the dispatch table of entry family {family.name!r}")
            dispatch_columns = _UniqueNames(f"columns of table {dispatch_name!r}")
            for record_name in family.record_names:
                dispatch_columns.add(
                    backing_dispatch_column_name(record_name), f"the dispatch column of {record_name!r}"
                )
