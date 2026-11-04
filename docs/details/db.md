# Backend storage in detail

`httk.store.backend.sql` is the database storage layer of *httk₂*: it stores **plain
frozen dataclasses** in a relational database, makes them queryable through a
backend-agnostic search DSL, and serves them through the neutral
`httk.core.EntryProvider` contract (e.g. as an OPTIMADE API via
*httk-serve*). SQL generation and dialect handling run on SQLAlchemy Core
internally; the public API exposes no SQLAlchemy types.

## Installing

The SQL layer is an optional extra (plain `import httk.store` works without it):

```bash
python -m pip install "httk-store[db]"      # SQLite (built into Python) via sqlalchemy
python -m pip install "httk-store[duckdb]"  # additionally the DuckDB backend
python -m pip install "httk-store[postgresql]"  # PostgreSQL backend (psycopg 3)
python -m pip install "httk-store[clickhouse]"  # ClickHouse backend
```

## Opening a store

The store class names the engine; its first argument is only the location:

```python
from httk.store import SqliteStore

store = SqliteStore("results.sqlite", entry_records={})   # first open declares the store
# reopen later with just: SqliteStore("results.sqlite"); in memory: SqliteStore(entry_records={})
```

`DuckdbStore("results.duckdb")`, `PostgresqlStore(url)`, and
`ClickhouseStore(url)` are the sibling classes; every keyword of `SqlStore`
(`entry_records`, `entry_ids`, `upgrade`, ...) is accepted as a keyword here. A
store built this way owns its database connection and disposes it on
`store.close()` or when leaving a `with SqliteStore(...) as store:` block.

`PostgresqlStore(url)` opens a PostgreSQL store from a `postgresql://` URL. It
is fully transactional and rides the ordinary `transactional` write profile
with no special-casing, and it supports bulk ingestion (`store.bulk_ingest()`)
with the same parity/deferred/parallel behavior as SQLite and DuckDB. Only the
psycopg 3 driver is supported: a bare `postgresql://` URL is normalized to
`postgresql+psycopg://` and any other explicit driver is rejected. See the
[PostgreSQL testing guide](../postgres-testing.md) for local setup.

### Index policy

`indexes=` chooses which non-unique indexes a new SQL store creates. A new
`DuckdbStore` defaults to `indexes="minimal"`: it leaves out every child
table's parent-sid index and the `store_timestamp`, `logical_id`, `alt_id` and
`(alt_id, alt_kind)` indexes of record tables outside the entry families.
DuckDB keeps automatic min/max zonemaps and stores child rows in parent order,
so those indexes only cost space (over 40% of a structure store) and slow
per-record fetches. SQLite and PostgreSQL stores default to `indexes="all"`:
their fetches need the child-table indexes. Primary keys and unique indexes
(`uq_<table>_content_id`, `immutable_id`) are integrity constraints and are
always created.

`skip_indexes=` names further non-unique indexes to leave out, exactly as they
are named in the database (`ix_<table>_<column>`), on any SQL store:

```python
from httk.atomistic import StructureEntry, UnitcellStructureRecord

store = SqliteStore(
    "structures.sqlite",
    entry_records={StructureEntry: UnitcellStructureRecord},
    skip_indexes=["ix_atomistic_unitcell_structure_optimization_type"],
)
```

Naming a unique index, a primary key, or an index the declared records never
create raises `ValueError`. The policy is stamped in the store on creation and
a reopen adopts it; reopening with a different explicit policy raises
`ValueError`, so changing it means rebuilding the store. Stores created before
index policies existed open as `indexes="all"`.

### Advanced: `Backend` and `SqlStore`

The per-dialect classes are thin subclasses of `SqlStore`. Use the two-object
form `SqlStore(Backend.sqlite("results.sqlite"))` directly when you need a
custom SQLAlchemy engine, one `Backend` shared across several stores or a
`with Backend.sqlite(...) as db:` block, or `degraded=True` recovery. `Backend`
names *where* data lives (`Backend.sqlite`, `Backend.duckdb`,
`Backend.postgresql`, `Backend.clickhouse`) and owns the connection pool; the
caller then owns its lifecycle, so `SqlStore.close()` leaves it open. Both names
remain importable from `httk.store` and `httk.store.backend.sql`.

Touching a SQL-backed name (such as `httk.store.backend.sql.Backend`) without the extra
installed raises an `ImportError` naming it.

## Topics

```{toctree}
:maxdepth: 1

db-records
db-schema
db-revisions
db-relationships
db-timestamps
db-recovery
db-bulk-ingestion
db-querying
db-serving
db-tables
db-optimade-client
```

## Previous section links

The sections of this guide now have their own pages. Existing section links
land here; follow the matching link to the full discussion.

(declaring-a-storable-class)=
- [Declaring a storable class](db-records.md#declaring-a-storable-class)

(storing-and-fetching)=
- [Storing and fetching](db-records.md#storing-and-fetching)

(lazy-records)=
- [Lazy records](db-records.md#lazy-records)

(vocabulary)=
- [Vocabulary](db-schema.md#vocabulary)

(applying-a-purely-additive-change-with-upgrade-true)=
(applying-a-purely-additive-change-with-upgradetrue)=
- [Applying a purely additive change with `upgrade=True`](db-schema.md#applying-a-purely-additive-change-with-upgradetrue)

(record-replacement-and-lineages)=
- [Record replacement and lineages](db-revisions.md#record-replacement-and-lineages)

(entry-ids)=
- [Entry ids](db-revisions.md#entry-ids)

(alternatives)=
- [Alternatives](db-revisions.md#alternatives)

(weak-links)=
- [Weak links](db-relationships.md#weak-links)

(strong-links-provenance-edges)=
- [Strong links (provenance edges)](db-relationships.md#strong-links-provenance-edges)

(store-timestamps)=
- [Store timestamps](db-timestamps.md#store-timestamps)

(permanentization-degraded-writes-and-fsck)=
- [Permanentization, degraded writes, and fsck](db-recovery.md#permanentization-degraded-writes-and-fsck)

(clickhouse-bulk-fenced-writes)=
- [ClickHouse bulk-fenced writes](db-recovery.md#clickhouse-bulk-fenced-writes)

(bulk-ingestion)=
- [Bulk ingestion](db-bulk-ingestion.md#bulk-ingestion)

(contract)=
- [Contract](db-bulk-ingestion.md#contract)

(performance)=
- [Performance](db-bulk-ingestion.md#performance)

(parallel-ingestion)=
- [Parallel ingestion](db-bulk-ingestion.md#parallel-ingestion)

(searching)=
- [Searching](db-querying.md#searching)

(pandas-style-slicer-indexing)=
- [Pandas-style slicer indexing](db-querying.md#pandas-style-slicer-indexing)

(continuation-pages)=
- [Continuation pages](db-querying.md#continuation-pages)

(portable-protocol)=
- [Portable protocol](db-querying.md#portable-protocol)

(neutral-portable-store-profile)=
- [Neutral portable Store profile](db-querying.md#neutral-portable-store-profile)

(result-and-identity-semantics)=
- [Result and identity semantics](db-querying.md#result-and-identity-semantics)

(memory-and-statement-cost)=
- [Memory and statement cost](db-querying.md#memory-and-statement-cost)

(exact-rationals-approximate-comparisons)=
- [Exact rationals, approximate comparisons](db-querying.md#exact-rationals-approximate-comparisons)

(serving-through-optimade)=
- [Serving through OPTIMADE](db-serving.md#serving-through-optimade)

(reading-a-remote-optimade-service)=
- [Reading a remote OPTIMADE service](db-optimade-client.md#reading-a-remote-optimade-service)

(binding-entry-types)=
- [Binding entry types](db-optimade-client.md#binding-entry-types)

(the-standard-namespace-rule)=
- [The standard-namespace rule](db-optimade-client.md#the-standard-namespace-rule)

(strict-definition-only-discovery)=
- [Strict definition-only discovery](db-optimade-client.md#strict-definition-only-discovery)

(querying-a-remote-service)=
- [Querying a remote service](db-optimade-client.md#querying-a-remote-service)

(non-conforming-services)=
- [Non-conforming services](db-optimade-client.md#non-conforming-services)
