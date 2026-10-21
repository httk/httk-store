# Permanentization, degraded writes, and fsck

SQL stores use a storage-only `_httk_role` parent column: `1` marks a record
saved at the public top level and `0` marks a recursively saved dependency.
The column is not part of content identity, by-value matching, canonical
encoding, hydrated records, or query results. Saving a dependency again at the
top level promotes its existing row to main; bulk canonicalization likewise
keeps the maximum role of all collapsed occurrences.

The usual `Backend.sqlite(...)` and `Backend.duckdb(...)` stores have the
persisted `transactional` write profile (the absent metadata value means the
same thing). SQLite additionally exposes an explicitly opt-in, artificial
transactionless conformance vehicle:

```python
db = Backend.sqlite("recovery-test.sqlite", degraded=True)
store = SqlStore(db, entry_records={})
```

That construction stamps the `degraded` profile and can only reopen through a
similarly configured database. Opening validates the live SQLite DB-API
autocommit state, not just the construction flag; a transactional profile also
rejects an autocommit engine. It is SQLite-only in this release: it uses DB-API
autocommit to model SQL-like backends that cannot provide transaction rollback.
The profile is deliberately single-writer. A database-visible writer lease is
acquired on mutation and held until `Backend.dispose()`; another instance can
inspect the holder/age and explicitly call `store.steal_lease()` when recovery
authority is clear.

Degraded saves permanently write dependencies first, then child-element rows
under a preallocated monotonic sid, then the parent sid row last. Thus a visible
parent means its subtree is complete; a failed write may leave only dependency
or child residue. No compensation deletion is attempted. Per-operation dirty
markers cost one lookup, one upsert, and one conditional delete per touched
table; a leftover marker arranges a targeted ownerless-child sweep before the
next write to that table. Sid counters are created and initialized lazily at
the first allocation for each parent table. `bulk_ingest()` is intentionally unavailable for degraded stores in
v2.3.0; use ordered `save()` calls.

Run `store.fsck(known_types=(...))` after a failed degraded writer (or for an
integrity audit). It repairs missing dispatch rows for main entries, sweeps
ownerless child rows, marks from main and dispatch roots, removes unreachable
dependency rows, and reports dangling logical references. It refuses garbage
collection if it finds an ordinary application table it cannot attribute to
the declared layout or `known_types`; no unrelated table is guessed or swept.
SQLite transactional fsck uses `BEGIN IMMEDIATE`. DuckDB callers must pass
`exclusive=True`, which is an explicit acknowledgement that the database is
offline from all writers for the entire fsck; DuckDB cannot otherwise enforce
the necessary read/delete exclusion. Invalid role values are violations; with
`repair=True` fsck normalizes them to dependency role `0` rather than inventing
a new root.

Weak-link tables (see [Weak links](db-relationships.md#weak-links)) are checked separately and are
never ownership or reachability edges — a weak link neither retains nor
garbage-protects rows. fsck reports a link row whose `source_lid` or
`target_lid` does not resolve to an existing lineage in its parent table
(dangling), a link lineage whose `logical_id` is not its first row's sid (broken
lineage integrity), and a `retracted` value outside `{0, 1}`. More than one live
lineage for one `(source, target)` pair is reported as a *repairable note*, not
corruption (concurrent writers may mint duplicate pairs).

Pass `data_dir` to also verify the files that stored `FileRecord` rows point
at. It is the local directory that stored `data/...` URLs are relative to (the
directory published under serve's `--data-path`):

```python
summary = store.fsck(data_dir="/srv/project/data")
summary.files_missing     # ((sid, url), ...) of files that are absent, symlinks, or unreadable
summary.files_mismatched  # ((sid, url), ...) of files whose SHA-256 differs from sha256
```

Every `FileRecord` row is read, whether it was saved on its own or as a field of
another record. A relative `data/<rest>` URL is checked as `data_dir/<rest>`:
the file must exist and its streamed SHA-256 must equal the row's `sha256`.
Symlinks are not followed; a path with a symlink in any component below
`data_dir` is reported as missing. `files_checked` counts these rows.
`files_unchecked` counts the others: absolute URLs (`https://…`, `file://…`,
`/abs/...`) and paths that contain `..`. A row with a NULL `url` is a lost file
(see below): it is not checked and is counted in `files_lost`, not reported as
missing. The rows are read under the fsck lock
and the files are hashed after it is released, so hashing does not block
writers. The check reads files only. It never changes a row, and a problem it
finds is reported in these fields rather than raised. Rows removed by the same
run's garbage collection are not checked. `MongoStore.fsck` accepts the same
keyword and reports the same fields.

## Lost files and `relocate_file`

A `FileRecord` whose `url` is `None` records that the bytes are lost: the record
still names the content by its `sha256`, but there is no location to read it
from. A workflow that mutates or deletes a large input during its run records
the input this way. Served `files` entries render such a `url` as the RFC 6920
content URI `ni:///sha-256;<base64url digest>`, so the entry never claims a
location it does not have.

A lost file does not stay lost by accident. `FileRecord` deduplicates by
content, and its `url` is `KeepStored`, which means the first non-null value
wins: a later save of the same bytes from a real location fills the stored
NULL in place, while a stored non-null `url` is never overwritten by a later
save, whether that save's `url` is `None` or different. Filling is an in-place
metadata update, so it creates no new revision and the content id is unchanged.

To set a location explicitly, for example after a workspace is deleted or the
bytes are moved, use `relocate_file`:

```python
store.relocate_file(sid, "data/archive/OUTCAR")  # another copy of the same bytes
store.relocate_file(sid, None)                   # mark the bytes lost
```

It updates the `url` of the `FileRecord` row with that sid in place, in one
transaction, and raises `KeyError` when no such row exists. The caller is
responsible for the new location holding the same bytes; `fsck(data_dir=...)`
verifies them. fsck itself stays report-only and never relocates a file.
`MongoStore.relocate_file` behaves the same.

## ClickHouse bulk-fenced writes

For local/CI server setup and the required `_httk_bootstrap` KeeperMap DDL,
see the [ClickHouse testing guide](../clickhouse-testing.md).

ClickHouse uses KeeperMap metadata and the persisted `bulk-fenced` profile.
Reads do not acquire a lease. A bulk writer acquires a fresh, never-reused
token with a strict insert, verifies that exact value during the P2 bulk-entry
and marker operations, and releases it with an exact-value delete when
`Backend.dispose()` runs. P3 adds verification around its durable phases. The
`ingest_state` marker is also a strict insert and carries the lease token plus
a fresh per-ingest nonce; it is cleared only by an exact-value delete after a
successful ingest.
`steal_lease()` is intentionally unavailable.

If a writer dies with only a lease residue, inspect `_httk_store_metadata`,
verify that the writer is no longer alive, and delete only the observed lease
value with a ClickHouse client:

```sql
SELECT key, value FROM _httk_store_metadata WHERE key = 'lease';
SET keeper_map_strict_mode = 1;
DELETE FROM _httk_store_metadata
WHERE key = 'lease' AND value = '<observed lease JSON>';
```

Never clear `ingest_state` merely because its lease was removed. Its presence
means the store may contain partial or inconsistent physical state, so the
default remedy is `DROP DATABASE`, recreate the bootstrap table, and re-ingest.
Only after a verified cleanup/rebuild has restored the declared empty-store
invariant may an operator clear the exact observed marker value. Use the same
strict setting:

```sql
SET keeper_map_strict_mode = 1;
DELETE FROM _httk_store_metadata
WHERE key = 'ingest_state' AND value = '<observed marker JSON>';
```

Do not delete values belonging to a live writer or use broad key-only deletes.
