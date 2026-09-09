# Bulk ingestion

For SQLite, DuckDB, and PostgreSQL, `store.bulk_ingest()` is a faster path than
a `save()` loop for **building a store from scratch or appending a large
increment** to one. It returns a
`httk.store.backend.sql.bulk.BulkIngest` context manager that mirrors `save()` but buffers
encoded rows with pre-assigned sids and appends them in `executemany` batches
inside one transaction, instead of one statement round-trip and an in-database
deduplication protocol per record. It is a near drop-in for the save loop:

ClickHouse bulk ingestion is currently fresh-store-only and stops at the P2
lease-plus-marker boundary until P3 supplies its nontransactional loader and
finalizer. It does not provide rollback or exact restoration; marker residue
fails closed and the default recovery is drop-and-reingest.

**Known limitation — PostgreSQL bulk `NaN` in a list-of-floats field.** Under
PostgreSQL bulk ingest, a `NaN` value inside a stored **list-of-floats (child)
field** is not preserved: it reads back as `NULL`. Bulk ingest stages rows
through SQLite shards, which cannot represent `NaN`, so the value is lost in the
list column. A **scalar** float `NaN` IS preserved under bulk ingest, and the
serial `save()` path preserves `NaN` in both scalar and list-of-floats fields on
every backend.

```python
# Per-record save loop
with store.transaction():
    for structure in structures:
        store.save(structure)
```

```python
# Bulk-ingest drop-in
with store.bulk_ingest() as bulk:
    for structure in structures:
        bulk.save(structure)
```

Reach for it when the increment is large; for a handful of records the ordinary
`save()` path is simpler and the round-trips it saves are negligible.

## Contract

**Exclusive write ownership.** While a `bulk_ingest()` context is open the
store's ordinary write path belongs to it: `save()`, `ensure_tables()`, and
`transaction()` on the same `httk.store.backend.sql.SqlStore` raise `RuntimeError`, and a
second `bulk_ingest()` context on the same store is refused. Reads from an
already-open store remain available; a new open is rejected while an
empty-store ingest marker is present.

**SQLite/DuckDB transaction and restoration.** On SQLite and DuckDB, the whole
ingest runs in a single transaction that commits only on clean exit. Any exception — a metadata
conflict, a uniqueness violation, or one you raise inside the block — rolls the
transaction back, drops every table the context created, restores any index it
dropped, removes its staging tables, and clears the store's identity caches,
leaving the store exactly as it was before the context opened. For an
empty-store ingest, cleanup verifies that only the metadata table remains and
then clears its marker, so retrying is safe. A hard crash can leave the marker
behind; subsequent opens reject that store and require dropping and re-ingesting
it.

These transaction and restoration guarantees do not apply to ClickHouse. Its
nontransactional P3 ingest will use the marker as a fail-closed recovery gate;
an interrupted marker defaults to drop-and-re-ingest.

**Deduplication and uniqueness are post-conditions, not per-row checks.** Within
the stream, records deduplicate set-wise in memory by the class's
`StorageInfo.dedup` policy (content identity by default, `by_value`, or `none`),
exactly as `save()` would. Global uniqueness against what is already stored is
enforced at the boundaries rather than per row: on a physically empty store the
record tables are created index-less and their separable indexes (content-id
uniqueness, `Indexed`/`Unique`, composite, and child parent-sid) are built once
the stream completes — building the unique index *is* the verification, and a
duplicate aborts the ingest. On a populated store each flushed chunk is staged
into an ordinary `bulkstage_<table>` table and resolved set-wise against the
target: a content-id anti-join, a `by_value` whole-parent-column anti-join with
null-safe equality, and a sid remap that rewrites every still-buffered reference
to the deduplicated existing sid.

**Returned sids are provisional.** `bulk.save()` returns an integer sid like
`save()`, but it is provisional while the context is open: a record that
deduplicates against a row the store already held is remapped to that existing
sid at flush. After the context exits cleanly,
`httk.store.backend.sql.bulk.BulkIngest.resolved_sid` maps any returned sid — provisional
or final — to its durable stored sid. It keys on the bare sid value, so resolve
a returned sid against the type it was saved as (sids are allocated per table,
and one value can recur across tables).

**Nested entry promotion.** `bulk.save(envelope, promote=StructureRecord)`
makes every nested `StructureRecord` occurrence a top-level entry while keeping
the envelope as the returned root. Pass an iterable of classes to promote more
than one record type. Each class must be reachable from the envelope's stored
schema; projection, role marking, and entry dispatch all remain inside the same
worker task.

**`verify_metadata`** (default `True`, a plain `bool`) controls whether a
content-id hit compares its identity-excluded metadata against the first
in-memory occurrence — or against the stored row for a hit against existing
data — reproducing `save()` and raising
`httk.store.store_common.EntryMetadataConflictError` on a conflict. Pass
`verify_metadata=False` to skip the comparison when the stream is
known-consistent.

**`index_strategy`** (`"auto"`, `"keep"`, or `"rebuild"`, default `"auto"`)
governs only how an *existing* table's separable indexes are handled during an
append: `"keep"` appends through them, `"rebuild"` drops and recreates them at
the end (where the unique-index creation re-verifies global uniqueness), and
`"auto"` chooses per table by the staged-to-existing row ratio. On DuckDB, which
reserves a dropped index's name until commit, a `"rebuild"` decision instead
keeps the indexes in place — relying on their incremental maintenance — and
verifies content-id uniqueness with a duplicate scan at finalize; the final
indexes are identical either way.

**`finalize`** (`"auto"`, `"parity"`, or `"deferred"`, default `"auto"`)
chooses the finalization profile. `"deferred"` is an explicit fresh-store
profile at any worker count; `"parity"` is the historical in-database path.
`"auto"` selects deferred only for a physically empty, supported serial ingest;
it selects parity for every other case, including `workers>1`. At current batch
scales the parallel in-database merge is faster, while serial deferred gains
about 36%.

**Nested conflict paths differ by prefix.** Because the bulk encoder resolves
referenced and child records eagerly and only discovers their existing-row hits
at flush, an `httk.store.store_common.EntryMetadataConflictError` reached through
a `descend` field (a non-skipped reference whose target itself carries skipped
metadata) is reported at the descendant record's own path (`"Leaf.note"`) rather
than the ancestor field path `save()` would use (`"Root.primary.note"`). The
exception type, message template, and roll-back are identical; only the path
prefix differs.

**`chunk_size`** (default `100_000`) is the number of top-level `save()` calls
buffered before a flush. Buffered rows and the in-memory dedup indexes are held
until the next flush, so peak memory scales with the chunk size and each
record's fan-out into child and reference rows: lower it for very wide records
or a tight memory budget, raise it to amortize the staging round-trips over more
rows. Identity caches are deliberately not populated by bulk ingestion.

**`on_progress`** is an optional `(records_buffered_total, rows_flushed_total)`
callback invoked after each flush, for progress reporting over a long build.

## Performance

Bulk ingestion gains most on flat records with little fan-out: measured against
the per-record `save()` loop it is roughly **30x** faster on DuckDB and **13x**
on SQLite for flat rows, easing to about **5x** (DuckDB) and **4x** (SQLite) for
structure-shaped records whose child and reference tables dominate the row
count. These figures come from single-threaded runs against a tmpfs database, so
the per-record baseline they improve on is already I/O-favorable; both the
speed-up and the absolute throughput will differ on slower storage.

## Parallel ingestion

For the *offline build* of a store from a large stream, `bulk_ingest(workers=N)`
with `N > 1` encodes the stream in a pool of forked worker processes and merges
their per-table shards set-wise. Encoding — the bottleneck for structure-shaped
records — runs across cores; the merge (loading shards, collapsing cross-worker
duplicates, renumbering to compact sids, and building the indexes) runs once in
the main process inside the ingest's single transaction.

```python
with store.bulk_ingest(workers=12) as bulk:
    bulk.save(layout_record)
    for material in materials:
        bulk.save(material)
```

On DuckDB workers hand rows off as Parquet shards, so parallel mode there needs
`pyarrow`; install it with the combined extra:

```console
$ pip install "httk-store[duckdb,parallel]"
```

SQLite workers write one native shard database each and need no extra dependency.

**Empty target only.** Parallel mode is for building a fresh store, not for
appending: opening `workers>1` on a store that already holds application rows is
refused (use `workers=1` for incremental appends). On DuckDB the restriction is
stronger — *any* pre-existing application table is refused, because the merge
renumbers and deletes rows in place and DuckDB will not do that through a live
foreign-key constraint.

**Physical schema is foreign-key free.** SQLite and DuckDB use the same FK-free
physical DDL for serial and parallel builds. Logical reference, ownership,
child-element, and dispatch edges remain available to the storage algorithms,
while column types, keys, checks, and indexes are unchanged.

**Provisional tokens.** Because a worker encodes each object asynchronously, the
sid is not known when `save` returns; in parallel mode `save` returns an opaque
token instead. After the context exits cleanly,
`httk.store.backend.sql.bulk.BulkIngest.resolved_sid` maps each returned token to its
durable stored sid, exactly as it maps a provisional sid on the serial path. A
lost task (an unpicklable object, or a worker that crashed or was killed) aborts
the ingest rather than committing a partial store, and `on_progress` is rejected
up front because per-flush counts are not observable across processes.

**Identity-excluded metadata restriction.** The merge verifies identity-excluded
(`IdentitySkip`) metadata with a grouped column scan rather than by reconstructing
every duplicate record. That covers scalar skip columns and skipped references to
content-addressed or by_value records, and it reports a conflict against the
schema field. A few shapes fall outside it and are rejected up front (naming
`workers=1`): an identity-excluded child *sequence*, an identity-excluded
reference to — or `descend` into — a non-deduplicated (`dedup="none"`) record,
and a self-referential identity-excluded reference. Opening with
`verify_metadata=False` lifts the restriction.

**Measured speed-up.** Building the ~9,000-material altermagnets store into a
file-backed DuckDB database, parallel mode reaches about **6.6x** at 24 workers
when replicas share substructure (the realistic case, where the merge collapses
many cross-worker duplicates) and about **11x** at 24 workers with distinct roots
and shared atomic descendants (each material and its structure distinct, their
cells/sites/species still shared, so the merge collapses much less). The encode
phase scales with the worker count; the merge is a small fixed fraction of the total.
The benefit is real only for large builds — the pool fork, the shard round-trip,
and the merge are pure overhead on a small stream — so `workers` defaults to `1`.
Reproduce with `benchmarks/bench50_parallel.py`.
