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

`Backend.postgresql(url)` opens a PostgreSQL store from a `postgresql://` URL.
It is fully transactional and rides the ordinary `transactional` write profile
with no special-casing, and it supports bulk ingestion (`store.bulk_ingest()`)
with the same parity/deferred/parallel behavior as SQLite and DuckDB. Only the
psycopg 3 driver is supported: a bare `postgresql://` URL is normalized to
`postgresql+psycopg://` and any other explicit driver is rejected. See the
[PostgreSQL testing guide](../postgres-testing.md) for local setup.

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

(low-level-portable-protocol)=
- [Low-level portable protocol](db-querying.md#low-level-portable-protocol)

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
