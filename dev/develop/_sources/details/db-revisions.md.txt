# Record replacement and lineages

The store is append-only, but a record can be marked as the logical successor
of an earlier one. Every stored row carries a `logical_id` lineage identity:
a freshly saved record starts a lineage whose id is its own sid, while
`store.replace(predecessor, obj)` saves `obj` copying the predecessor's
`logical_id` instead of starting a new one. Nothing is updated or deleted —
both rows remain fetchable — and the lineage's *latest* row is simply the one
with the highest sid.

```python
from dataclasses import dataclass
from typing import Annotated

from httk.core.storage import Indexed
from httk.store import SqliteStore


@dataclass(frozen=True)
class Note:
    key: Annotated[str, Indexed()]
    text: str


store = SqliteStore(entry_records={})
with store.transaction():
    first = Note("n", "first")
    store.save(first)
    second = store.replace(first, Note("n", "second"))  # replace the stored instance
    latest = store.replace(store.fetch(Note, second), Note("n", "third"))  # a lazy proxy works too
```

`replace()` goes through the ordinary `save()` path, so its dedup policy,
timestamp capture, identity caching, and entry dispatch behave exactly as they
do there; it returns the new row's sid. The `predecessor` (a stored instance or
lazy proxy) need not itself be the latest row of its lineage — replacing an
already-replaced row extends the same lineage. If `obj` deduplicates onto an
existing row, an equal lineage (including replacing a record with itself) is an
idempotent no-op returning that sid, while a different lineage raises
`EntryReplacementError`. Replacing across record tables raises `ValueError`.

`store.history(obj)` returns every record sharing that lineage, oldest first
(the fresh record, then each replacement), reconstructed lazily like `fetch()`:

```python
[record.text for record in store.history(store.fetch(Note, latest))]
# ['first', 'second', 'third']
```

## Entry ids

Defined entry families also carry two human-readable identifiers. `id` is
one-to-one with a `logical_id` lineage and is consequently shared by every
replacement; `immutable_id` is one-to-one with a stored row. Configure minting
with `SqlStore(..., entry_ids=EntryIdScheme("httk.mydb", "1"))`, or pass
`id_series=` to `save`, `replace`, or `bulk_ingest` to select a campaign
series for that call. `MongoStore` has the same `EntryIdScheme` and per-call
kwargs (it has no bulk ingest); it indexes `f.id` and creates a unique partial
index for nullable `f.immutable_id`. The recommended form is
`httk.mydb-1-42` for an entry id and `httk.mydb-1-42~3` for its third revision.
Mongo stored-property serving uses these same human ids as SQL: ordinary pages
serve the latest row of each lineage, while revisions pages serve immutable ids
and expose the lineage id as `_httk_id`.
`EntryIdScheme(type_in_base=True)` appends the served entry type to the base.
For multi-backing families, the number is `logical_id * backing_count + backing_index`, which keeps ids unique across
the family's backing tables. After an additive upgrade has appended record kinds
to a family (see [below](#adding-record-kinds-or-families-to-an-existing-store)),
the number is `offset + logical_id * backing_count + backing_index`; a family
whose record list never changed has offset 0 and mints exactly as before.
Entry-ID ownership is enforced across every backing in a defined family.
SQL ownership claims share the record transaction. An identity conflict aborts
that transaction, even if the exception is caught inside its context; earlier
writes roll back and lazy records from it expire. Degraded and bulk-fenced
profiles retain their exclusive lease and recovery guarantees.

Existing stores without the ownership capability remain readable, but writes
require reopening with `upgrade=True`. This validates and backfills ownership
without changing IDs, revision numbers, or content IDs; conflicts abort without
rewriting records. Keep a backup before any upgrade. Mongo uses unique ownership
indexes and its existing transaction sessions. Standalone writes reserve IDs
before publishing records, using ordinary writer leases rather than an exclusive
maintenance lease or a full-store rescan. A failed write can leave an orphan
reservation that blocks reuse of its ID; explicit `store.fsck()` reclaims it
under the maintenance fence. After a killed process, use `force=True` only after
verifying that the stale lease's writer is no longer running.

An explicit URL-safe id which does not match that recommended form is accepted
with a warning; unsafe ids are rejected. Backing records of every family with
a definition id must declare nullable, identity-skipped `id` (indexed) and
`immutable_id` (unique) fields. Because this adds physical columns and a
unique index, stores created before this change must be rebuilt rather than
reopened with the old schema. This older column-layout requirement is distinct
from the additive ownership-capability upgrade above; immutable IDs are now
unique across every backing of their defined family.

Plain `fetch()` and `searcher()` queries keep returning **all** rows of a
lineage. Pass `only_latest=True` to `store.searcher()` to restrict *root*
variables to the highest-sid row of each `logical_id` (bounded by `as_of` when
given); reference and child variables stay unfiltered, and it does not require
`store_timestamps=True`. Weak links are the lineage-following complement to
these sid-pinned reference and child fields — they always resolve to the latest
revision regardless of `only_latest` (see [Weak links](db-relationships.md#weak-links)):

```python
search = store.searcher(only_latest=True)
note = search.variable(Note)
current = [row.note for row in search.results(note=note)]  # one row per lineage
```

### Adding record kinds or families to an existing store

Adding a new typed record kind to an existing family (for example a further
record class in the `records` family) or adding a whole new family is an
**additive** change: new tables and dispatch columns, never a rebuild. Reopening
with the extended declaration raises `StorageLayoutUpgradeRequiredError` whose
`hint` says the change is additive; reopening with `upgrade=True` applies it:

```python
store = SqlStore(database, entry_families=(extended_family,), entry_ids=scheme, upgrade=True)
```

The rules:

- **Append only.** Every stored family must still be declared with the same
  name and family `definition_id`, and its stored record list (record names and
  record `definition_id`s) must be an order-preserving prefix of the new one.
  A record's position in its family is its `backing_index`, part of the id
  numbering, so a reordered, removed, renamed, or re-defined record, a removed
  family, or a changed family `definition_id` is not additive; the error then
  names the offending family and record and the store must be rebuilt. A table
  newly attached to a family must be empty (a class previously saved ad hoc
  into its own table would bring rows without family identity; rebuild instead).
- **Numbering offset.** When a family's record list grows, its backing count `B`
  changes, and `logical_id * B + backing_index` could re-mint an old number. The
  upgrade therefore records, per grown family, `offset = max(n, previous offset) + 1`
  where `n` is the largest number carried by any existing `id` or
  `immutable_id` of any backing and any id series, and later numbers are
  `offset + logical_id * B + backing_index`. Every such number is at least
  `offset + B`, larger than every number minted before the upgrade, so no id
  is ever minted twice, however often record kinds are appended. Existing ids
  are never rewritten, and the family-wide ownership tables still refuse any
  collision with `EntryIdConflictError`. Offsets are stored in one optional
  metadata key, `entry_id_offsets` (canonical JSON `{"<family>": <offset>}`),
  written only by such an upgrade.
- **Dispatch.** The family's `_httk_entry_dispatch_*` table is dropped and
  recreated in its full new shape (one nullable unique sid column per backing
  and the exactly-one CHECK) and refilled with one row per main row of every
  backing, the same derivation `store.fsck()` uses to repair it. Old entries
  keep resolving by content id through `fetch_entry`, and their revision
  history is untouched.
- **Crash safety.** The upgrade restamps `entry_schemas`, then the offsets, and
  the declaration **last**, as a compare-and-set on the old declaration.
  SQLite (in the transactional profile), DuckDB and PostgreSQL run the whole
  upgrade in one transaction; independently of that, every step is idempotent
  and planned against the new layout, so an upgrade interrupted at any point —
  the degraded, autocommit SQLite profile can stop between any two statements —
  leaves the old declaration in place and converges when the `upgrade=True` open
  is retried. A retry after a partial restamp may choose an offset one higher
  than an uninterrupted run would: offsets only ever move up, which keeps every
  id unique. Reopening such a store with the *old* declaration reports "a
  declaration upgrade appears to have been interrupted" with remedy `"reopen"`:
  only the upgraded declaration (with `upgrade=True`) can finish it.
  Two concurrent upgraders are serialized: the second waits for the first, then
  either finds the identical layout already applied and opens it, or raises.
- **Stale writers and readers.** Every write takes the backend's
  write-ordering lock and then re-reads the stored declaration, immediately
  before its first DML, so an upgrade can never commit between that check and
  the writes: SQLite takes its database write lock (`BEGIN IMMEDIATE`) before
  the check; DuckDB, which admits one read-write process per file, uses an
  in-process lock that write scopes hold shared and an upgrading open holds
  exclusively from before its snapshot until it commits (a waiting upgrade has
  preference over new write scopes); PostgreSQL uses transaction-scoped advisory
  locks keyed by the store's metadata table (shared for writers, exclusive for
  the upgrader); the degraded SQLite profile is serialized by its exclusive
  writer lease, which an upgrade must also take. Inside `store.transaction()`
  the lock and the check are taken at the scope's **first write** and held to
  commit, so a read-only `transaction()` never blocks other writers (on DuckDB
  the in-process lock is held for the whole scope). A write already in flight
  finishes first and the upgrade covers its rows; a store instance opened before
  another instance upgraded the layout raises `StorageLayoutUpgradeRequiredError`
  ("the store layout changed since this store was opened; reopen it", remedy
  `"reopen"`) on its next write and changes nothing. A read that misses, or meets
  a dispatch row it cannot interpret, checks the declaration too and reports the
  same remedy instead of `None` or a corruption error. An upgrade that cannot get
  the lock in time fails with remedy `"retry"`; from inside the same thread's own
  open write scope it can never succeed, and the message says to close the scope
  first. SQLite waits for a held write lock up to the connection's busy timeout:
  pysqlite's default is 5 seconds, and `Backend(sqlalchemy.create_engine(url,
  connect_args={"timeout": seconds}))` sets another. One in-memory
  `Backend.sqlite()` shares a single connection between every store opened on
  it, so it must not be used by concurrently writing stores (their transactions
  would interleave on that connection); use a file database for concurrent
  writers.
- **Compatibility.** A store upgraded this way carries the `entry_id_offsets`
  key, which older httk-store versions do not recognize: they refuse to open it.
  Keep a backup before any upgrade.

`StorageLayoutUpgradeRequiredError.remedy` tells a tool what to do without
parsing the message:

- `"upgrade"` — the supplied declaration is an additive extension of the stored
  one (including finishing an interrupted upgrade): reopen with it and
  `upgrade=True`. It is advertised only after every read-only check that
  `upgrade=True` would also run has passed, so if `upgrade=False` reports
  `"upgrade"`, `upgrade=True` with the same declaration succeeds.
- `"rebuild"` — the difference cannot be applied in place (including a table
  newly attached to a family that already holds rows, or an alien reserved
  `_httk_` object).
- `"reopen"` — open the store differently: with the newer declaration it was
  upgraded to (an older client one declaration behind; the hint says "the store
  was upgraded to a newer declaration"), with the upgraded declaration after an
  interrupted upgrade (reopening with the old one can never finish it), after
  another instance upgraded it (a stale handle), after a concurrent upgrade to a
  different declaration, or with matching `store_timestamps`/write-profile
  options.
- `"retry"` — a transient conflict: the store was locked by writers, or a
  concurrent write conflicted with the upgrade's claim.

The declaration diff's `entry_declaration` entry carries a boolean `additive`
whenever the declaration differs, and `stored_is_newer: true` for the older
client case.

In a multi-backing family, only entries stored top-level carry a dispatch row;
`fetch_entry` resolves a record stored only as a dependency of another record
directly from its backing table, exactly as a single-backing family does, so
fresh and upgraded stores agree. A family record first stored by reference also
creates its family's sibling tables, as a top-level save does. Stores written
before that, in which such referencing records read as absent, are healed by the
first write that reaches the family.

Adding a family whose record list is new, without growing an existing family,
writes no offset: the new family mints exactly like a fresh store. The ClickHouse
bulk-fenced backend refuses these upgrades.

`MongoStore` offers the same upgrade with the same rules, offsets, remedies and
error diffs; only the mechanics differ. The upgrade runs under the singleton
**fsck lease**, which drains every live writer lease and refuses new writers
until it is released. Writers that do not finish within 30 seconds make the
upgrade give up with remedy `"retry"` (the lease is released again, nothing is
changed); `fsck()` itself keeps waiting for writers indefinitely. A grown
family's dispatch collection keeps its documents (`{_id: content_id, record,
sid}` does not depend on the number of backings): its validator's `record` enum is
widened with `collMod` (or the collection is created for a one-to-many growth) and
the missing dispatch documents are back-filled from the main documents of every
backing with unordered inserts that accept already-present identical documents.
The offsets live in the `entry_id_offsets` field of the layout document, and the
restamp writes `entry_schemas`, then `entry_id_offsets`, then the declaration last
(a `find_one_and_update` conditioned on the old declaration, which also advances
`generation`). MongoDB has no transactional DDL, so every step is durable as it
happens; each is idempotent and planned against the new layout, so an
interrupted upgrade converges on a retried `upgrade=True` open. Unlike on SQL, a
process that *dies* mid-upgrade leaves its fsck lease (`lease/fsck`) behind, and
the retry then fails with remedy `"retry"` and a hint naming the cure: once that
lease is stale (8 seconds after its last heartbeat by default) and its owner is
verified dead, clear it with the module-level
`httk.store.backend.mongo.clear_stale_lock(database)` (no `MongoStore` needs to
be constructible for that), then retry. After an interrupted N→N+1 upgrade, a
handle with the old declaration can still open (its dispatch collection is
expected) and its first collection preparation narrows the dispatch validator
again; that is harmless, because the newly attached collection is guaranteed
empty, and the retried upgrade widens it once more. Stale writers are excluded by the writer-lease protocol rather than by a
per-write lock: a writer registers its lease durably *before* reading the layout
document, and the upgrade drains every fresh registration before touching
anything, so a writer either saw the new declaration (and is refused with remedy
`"reopen"`) or finished before the upgrade began. That covers `save`, `replace`,
`transaction()` (whose lease is taken at entry, so a stale handle is refused
there), `link`/`unlink`, and collection preparation (`ensure_collections`), so a
stale handle can never `collMod` a grown dispatch validator back to its old enum.
The protocol's one residual assumption is the lease freshness interval: a writer
whose lease went stale without heartbeats (8 seconds by default) makes the
upgrade refuse with remedy `"retry"`, and only an administrator's forced
override could let an upgrade run past a writer that is in fact still alive.

### Alternatives

A stored entry may carry named **alternative representations** — a conventional
cell beside a primitive one, say — that share the main entry's public `id` but
are addressed by a composite identifier. Two store-managed columns back this:
`alt_id` (`BigInteger`, not null; the group id — the main's `logical_id`, and a
main's own `alt_id` is itself) and `alt_kind` (`Text`, null; `NULL` on mains, a
kind token matching `[a-z][a-z0-9_]*` on alternatives).

Save an alternative by naming its main and kind:

```python
main = store.fetch(Structure, store.save(Structure(...)))
store.save(conventional_cell, alternative_of=main.id, alternative_kind="conventional")
```

The alternative copies the main's `id`, gets its own lineage and revision
history, and mints immutable ids of the form `<id>~<kind>~<n>`
(`httk.mydb-1-42~conventional~3`); a listed alternative without a revision
suffix is written `<id>~<kind>`. Both `alternative_of` and `alternative_kind`
must be given together, one kind per group, and bulk ingest saves mains only.

`store.searcher()` defaults to `only_main_alt=True`, so **mains are the default
everywhere**: ordinary and revisions queries never surface alternatives, and an
alternative's revisions never enter a revision stream. Pass
`only_main_alt=False` to include them. Stored-property serving exposes
alternatives through `StoredEntryFederation` (see [Federated stores](../federation.md)), not through
`StoreEntryProvider`, which stays main-only.

Because `alt_id`/`alt_kind` are new physical columns, a store created before
this change is not version-bumped automatically. An old SQL store still reads
its mains correctly, but the first query or write that touches the alternative
columns fails loudly with a missing-column error; the remedy is to rebuild the
store rather than reopen the old schema.
