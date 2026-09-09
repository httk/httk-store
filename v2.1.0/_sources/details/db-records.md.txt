# Declaring, storing, and fetching records

## Declaring a storable class

Storability is non-intrusive: any frozen dataclass whose fields resolve is
storable — there is no base class. The stdlib-only marker vocabulary lives in
*httk-core* (`Indexed`, `Unique`, `Skip`, `Shape`, `StorageInfo`, `WeakLink`,
`StrongLink`, `stored_property`), so domain modules can declare storable classes without
depending on httk-store:

```python
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

from httk.core import FracVector
from httk.core.storage import Indexed, Shape, StorageInfo, stored_property


@dataclass(frozen=True)
class Author:
    name: str
    year: int


@dataclass(frozen=True)
class StructureRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(indexes=(("spacegroup", "formula"),))

    formula: Annotated[str, Indexed()]
    spacegroup: int
    energy: Fraction  # stored exactly
    cell_basis: Annotated[FracVector, Shape(3, 3)]  # fixed-shape tensor, stored inline
    reduced_coords: Annotated[FracVector, Shape(0, 3)]  # variable rows, child table
    symbols: list[str]  # child table
    reference: Author | None = None  # foreign key, saved recursively

    @stored_property
    def natoms(self) -> int:  # stored & queryable; recomputed on load
        return len(self.symbols)
```

Scalars (`int`/`str`/`bool`/`bytes`) become columns, while `float` gets a query
`DOUBLE` plus an exact hexadecimal text companion so signed zero and every
other finite binary64 value round-trip unchanged. `X | None` makes fields
nullable, rationals and datetimes are encoded by value codecs, lists and tuples
become child tables, and nested storable dataclasses become foreign keys (saved
recursively first). Classes you cannot modify can be described externally with
`register_schema_override`. A class-level `StorageInfo(links=...)` declaration
adds store-managed, lineage-following associations that live outside record
fields (see [Weak links](db-relationships.md#weak-links)).

For the distinction between exact storage and approximate SQL comparisons, see
[Exact rationals, approximate comparisons](db-querying.md#exact-rationals-approximate-comparisons).

## Storing and fetching

`Backend` names where data lives; `SqlStore` saves and reconstructs
instances. Saving deduplicates per the class's `StorageInfo.dedup` policy
(by content identity by default), and `transaction()` scopes several
operations into one database transaction:

```python
from httk.store.backend.sql import Backend, SqlStore

db = Backend.sqlite("example.sqlite")  # or Backend.sqlite() in memory,
store = SqlStore(db, entry_records={})  # first-time custom-record store
# Reopen an initialized database with: SqlStore(db)

with store.transaction():
    sid = store.save(record)  # returns the integer sid; dedups; recurses

same_record = store.fetch(StructureRecord, sid)  # a lazy row, decoded on access
eager = store.fetch(StructureRecord, sid, eager=True)  # fully materialized now
```

A source object with an exact `__httk_storage_record__` can be saved directly;
`save(source, as_record=OtherRecord)` selects another declared projection.
Nested record fields are projected recursively. A projected source must expose
any derived `stored_property` declared by its target record because storage
does not construct an intermediate record merely to evaluate that property.
Record validation runs at this storage boundary through `__httk_validate__`.
Optional child fields use presence columns, so `None` remains distinct from an
empty child value.

Repeated default fetches of one live `(class, sid)` return the same object (see
[Lazy records](#lazy-records) for the full identity contract). Join-objects
pointing at a stored instance are found with
`store.referring(TagClass, field="structure", to=record)`, which returns lazy
rows by default and accepts `eager=True`.

### Lazy records

`fetch()`, `fetch_many()`, `fetch_by_content_id()`, `fetch_entry()`, and
`referring()` return **lazy rows** by default: the parent row is read now, but
every child, reference, and derived field decodes only when first accessed, and
recursively — a lazy record's children are lazy rows too. Pass `eager=True` to
any of them to fully materialize the base dataclass up front. A lazy row is a
subclass of the storable class (`isinstance(row, StructureRecord)` holds, and
`type(row)` is the row subclass), exposes its database `sid`, and reuses the
memoized value on repeated access to one field.

**Identity.** Repeated default fetches of one live `(class, sid)` return the
same object; a live materialized instance takes precedence over creating a
proxy, so an `eager=True` fetch after a lazy one hands back a fresh instance and
replaces the cached slot (materialized-wins). Mixing eager and lazy access can
therefore yield two distinct but equal objects for one sid while a caller still
holds the older one. Eager materialization reuses live *materialized* nested
objects but not lazily fetched ones (the exact-type guard skips a proxy).
Internal cache maintenance — a failed write, dedup compensation after a save —
may re-materialize a later fetch. Strict `is` identity across arbitrary call
sequences is **not** promised, and never truly was: these cache clears already
qualified the older "very same object" wording.

**Lifetime.** A lazy row's deferred reads open a *fresh* connection outside the
originating transaction, so a record that must outlive its transaction,
connection, or engine should be fetched with `eager=True`. Accessing a lazy row
whose originating transaction rolled back raises `ExpiredLazyRecordError`
(naming the class and sid) — including on a field that was read before the
rollback, and including a child sequence whose deferred read executed inside the
rolled-back transaction. Committed-transaction rows and rows fetched outside any
transaction are unaffected. Engine disposal does **not** invalidate proxies: the
pool silently reconnects for file-backed databases, so the lifetime rule is
correctness discipline, not an enforced guard.

**Memory and cost.** A single live lazy row pins not just its own chunk (up to
500 parent rows plus any child blocks already read) but the transitive closure
of chunks and hydrators reached through its shared hydration context —
reference targets are pinned wholesale when first read, so one held record can
retain a broad slice of the graph until the cyclic garbage collector reclaims
it (proxies sit in reference cycles with their pinned chunk, so `gc.collect()`
is what frees them). Reading a field is a descriptor call plus the liveness
guard, so a repeated read of an already-decoded field costs a few hundred
nanoseconds rather than a plain-attribute access — bind a field to a local in
hot loops.

**Append-only framing.** The public store API is append-only, so rows a caller
fetched are not deleted or mutated in normal use. Under the lazy default, two
abnormal-deletion cases are honest caveats rather than guarded errors: abnormal
external deletion of a *referenced* row surfaces at attribute access as
`StaleResultError` naming the referenced class and sid, while abnormally deleted
*child* rows are indistinguishable from a legitimately empty sequence (child
rows carry no tombstones) and cannot raise. `KeyError` remains the at-call-time
contract for an absent parent sid, in both modes. A reference *field* memoizes
its target proxy without recording a per-field rollback token (unlike a child
sequence, which does): correctness still holds because the target's own chunk
carries the token and raises `ExpiredLazyRecordError` when the target's fields
are touched.

**Other behaviors.** A lazy row exposes stored/codec values without re-running
`__post_init__` (record authors keep `__post_init__` idempotent with respect to
stored representations); `eager=True` runs it. `dataclasses.replace(row, ...)`
returns a plain instance of the row class and does run `__post_init__`. Lazy
rows reject `copy.copy`, `copy.deepcopy`, and pickling (materialize with
`eager=True` first). Reference cycles are tolerated lazily — each hop is a fresh
proxy — whereas `eager=True` on a cyclic graph raises `SchemaError`. Storable
records must not declare `@dataclass(order=True)` — ordering comparisons
between a lazy row and a materialized instance raise `TypeError` (the generated
`__lt__` requires the exact class) — and none do in the workspace today.
