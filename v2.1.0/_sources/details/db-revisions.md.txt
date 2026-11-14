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
from httk.store.backend.sql import Backend, SqlStore


@dataclass(frozen=True)
class Note:
    key: Annotated[str, Indexed()]
    text: str


store = SqlStore(Backend.sqlite(), entry_records={})
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
the family's backing tables.
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
search.output(note, "note")
current = [row.values[0] for row in search]  # one row per lineage
```

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
