# Backend storage

`httk.store.backend.sql` stores **plain frozen dataclasses** in a relational database
(SQLite, DuckDB, or PostgreSQL), makes them queryable through a
backend-agnostic search DSL, and serves them through the neutral `httk.core.EntryProvider` contract —
no SQLAlchemy types in the public API, no base class to inherit:

```python
from httk.store import SqliteStore

store = SqliteStore("results.sqlite", entry_records={})   # first open declares the store
# reopen later with just: SqliteStore("results.sqlite")
# in memory instead: SqliteStore(entry_records={})

with store.transaction():
    sid = store.save(record)             # dedups and recurses automatically

same_record = store.fetch(type(record), sid)   # a lazy row; add eager=True to materialize
```

The class names the engine and the argument is only the location:
`DuckdbStore("results.duckdb")`, `PostgresqlStore(url)`, `ClickhouseStore(url)`.
A store built this way owns its database connection and disposes it on
`store.close()` or when leaving a `with SqliteStore(...) as store:` block.

**Advanced: `Backend` and `SqlStore`.** Use the two-object form
`SqlStore(Backend.sqlite("results.sqlite"))` when you need a custom SQLAlchemy
engine, one `Backend` shared across several stores or a
`with Backend.sqlite(...) as db:` block, or `degraded=True` recovery. There the
caller owns the `Backend`, so `SqlStore.close()` leaves it open. Both names stay
importable from `httk.store`.

Records are content-addressed (`content_id`) as well as locally numbered
(`sid`), and identical content saves to one row however many times it arrives.

Store timestamps are enabled by default. They support historic predicates such
as `store_timestamp <= T`; configure their unit size with
`store_timestamp_resolution` (default: microseconds, `time_ns() // 1000`).
The [detailed guide](details/db-timestamps.md#store-timestamps) covers the query syntax,
deduplication semantics, clock guard, and fsck repair behavior.

Append-only record replacement is available too: `store.replace(predecessor,
obj)` saves a logical successor sharing the predecessor's lineage, `store.history()`
walks a lineage, and `store.searcher(only_latest=True)` restricts root variables
to each lineage's latest row.

Entries can be connected by exact record-valued fields, immutable provenance
edges (`StrongLink`), or editable lineage-level curation (`WeakLink`). See
{doc}`connecting_entries` for the differences and complete declaration
examples.

An entry may also carry named **alternative representations** — a conventional
cell beside a primitive one, say: `store.save(obj, alternative_of=<main entry id>,
alternative_kind="conventional")` stores a sibling that shares the main's public
`id` (addressed as `<id>~<kind>`, with its own revision lineage), while ordinary
queries stay mains-only by default (`only_main_alt=True`). See
[the detailed guide](details/db-revisions.md#alternatives).

The full guide, {doc}`details/db`, covers declaring storable classes with the
httk-core marker vocabulary, entry families and multi-record dispatch, the
search DSL and stored properties, record replacement lineages, bulk ingestion
(including `bulk_ingest(workers=N)` and the crash-safe `finalize="deferred"`
fresh-store profile), the permanentization role model with `store.fsck()`
recovery, OPTIMADE serving, and store-layout versioning.

(serving-application-records)=
## Serving application records

For a small application-defined OPTIMADE dataset, *httk-core* provides
`EntryRecord`, `DataEntryRecord`, and `entry_record`. The decorator supplies
the durable identifiers and direct property mappings; pass the decorated
classes to the SQL store with `records=`:

```python
from typing import Annotated

from httk.core import DataEntryRecord, Property, entry_record
from httk.store import EntryIdScheme, SqliteStore


@entry_record("example.result")
class Result(DataEntryRecord):
    formation_energy: Annotated[
        float,
        Property(description="Formation energy per atom.", unit="eV"),
    ]


store = SqliteStore(
    "results.sqlite",
    records=[Result],
    entry_ids=EntryIdScheme("example", "1"),
)
store.save(Result(formation_energy=-1.25))
store.close()
```

The same `records=[Result]` declaration is supplied when reopening the store.
Records referenced by fields are discovered recursively when their classes are
registered or decorated, so a `structure: UnitcellStructureRecord` field also
declares the registered structures family. Use `entry_records=` or `entry_families=` for
the existing explicit declaration APIs; `records=` is mutually exclusive with
those forms. Multiple decorated records with the same family type are grouped
into one local family and their property definitions must agree where names
overlap. Existing plain frozen dataclasses should continue to use the explicit
declaration APIs.
