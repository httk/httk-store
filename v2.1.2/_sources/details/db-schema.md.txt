# Entry families and schema layout

## Vocabulary

An entry family is a logical key such as `StructureEntry`.
A record is a durable frozen-dataclass representation; a family may have several.
Backend/View is the representation pattern: a backend owns data, and a view presents it.
A content id identifies the record's content across stores; a SID is only a local row id.

Every database starts with a persisted, versioned layout declaration. Passing
`entry_records={}` says that this is a private/custom-record store with no
queryable entry families. An entry store instead maps each registered logical
family to the exact durable Record representation or representations it may
contain:

```python
store = SqlStore(
    db,
    entry_records={StructureEntry: UnitcellStructureRecord},
)
```

Applications may keep a private entry family out of global plugin discovery.
Supply its stable persistence names and classes directly with
`EntryFamilyDeclaration` and `EntryRecordDeclaration`:

```python
from httk.store import EntryFamilyDeclaration, EntryRecordDeclaration

private_entries = EntryFamilyDeclaration(
    name="my-application-publications",
    family=PublicationEntry,
    records=(
        EntryRecordDeclaration(
            name="my-application-publication",
            record=PublicationRecord,
        ),
    ),
)
store = SqlStore(db, entry_families=(private_entries,))
```

This is a store-local binding, not a registry operation. The store persists
the stable names and optional entry-definition IRIs but never persists or
imports arbitrary Python paths. Consequently, every reopen of a store with
application-owned declarations must supply the same `entry_families` value.
Omitting it raises `EntryLayoutBindingError`. Installed reusable modules should
continue to use registry-backed `entry_records`, which permits automatic
resolution on `SqlStore(db)`.
Both arguments may be supplied together when one store combines reusable
module families with application-private families; name or class collisions
are rejected while constructing the combined layout.

A single record is queried directly. A tuple of two or more records creates
a small family dispatch table, while the representation-specific data remains
in its normalized Record tables. Saving an exact configured record (including
saving a naturally bound domain object) makes it discoverable through
`fetch_entry(StructureEntry, content_id)`; that method returns the actual
concrete Record.

Later registry-backed `SqlStore(db)` calls trust the persisted declaration for
which classes and families are stored. Beyond the declaration, reopen also
verifies a per-table *schema fingerprint*: the resolved on-disk layout and
content identity of every declared class and its referenced classes — the
logical `identity_name`, dedup, indexes, links, and each field's role, codec,
columns, child tables, identity participation, and list-vs-tuple container. A
fingerprint JSON document has a `tables` mapping plus `entry_id_tables`, the
physical backing-table names of families with an entry definition id. A
record class whose stored shape or identity changed since creation — a gained or
retyped field, a new codec, a changed index, a `list`↔`tuple` swap, an added
`IdentitySkip`, a changed `identity_name` — is rejected up front with
`StorageLayoutUpgradeRequiredError`, whose diff names the offending tables
(`{"schema": {table: {"expected", "actual"}}}`) rather than failing later at use
(or, worse, silently breaking `content_id` deduplication). A code move or rename
is safe only when the record pins an explicit `identity_name` (every shipped
httk record does); without a pin the qualified class name *is* the content
identity, so the move changes `content_id` and the store correctly refuses to
open. Tables are still created lazily on the first write; reads never issue DDL.
Old, unversioned, or incompatible layouts raise
`StorageLayoutUpgradeRequiredError`; this redesign does not migrate old stores,
so rebuild them explicitly — with one exception below.

The internal `Run.source_id` field is served under its wire name `_httk_source_id`
on the `_httk_runs` entry type. That prefixing is not hand-written at the serving
edge: it is produced by `EntryTypeDefinition.served_form()`, the single wire-naming
authority, which prefixes the internal `runs`/`source_id` names when the definition
is served. `_httk_source_id` is a nullable, queryable, sortable string containing
the identifier assigned by the system that executed the run (for example, an
httk-workflow `<workspace_id>:<job_id>`), and it participates in the run's content
identity. Adding it therefore changes the
`core_run` schema fingerprint and requires rebuilding existing stores; it is not
an additive `upgrade=True` change.

### Applying a purely additive change with `upgrade=True`

When the *only* difference is additive, the reopen is applied instead of
rejected by passing `SqlStore(db, ..., upgrade=True)`. Additive means: new
tables, plus new fields that are each **non-child, non-derived, marked
`IdentitySkip`, and whose columns are all nullable** — with every pre-existing
table attribute and field byte-identical. The `IdentitySkip` requirement is the
key one: a field that participates in content identity would change the
`content_id` of byte-identical pre-existing rows, silently diverging dedup,
dispatch, and federation identity, so such an added field is rejected (the error
names the field and tells you to mark it `IdentitySkip` or rebuild). Added child,
derived (`stored_property`), non-nullable, removed, or retyped fields, changed
table attributes, and any protocol or declaration difference all still raise;
`upgrade=True` never widens or drops.

The apply creates every not-yet-created declared table whole (so a pre-existing
row that references a *new* table no longer reads as absent), adds each new
nullable column to the tables that already exist via `ALTER TABLE ... ADD
COLUMN` (plus any declared single-column index), then re-stamps the stored
fingerprint last. Old rows read back with the new fields as `None`, and their
`content_id` is unchanged. Every step is idempotent — already-present columns
are skipped and the index create is `IF NOT EXISTS` — and the re-stamp runs only
after all other verification passes, so a store interrupted mid-upgrade (SQLite
DDL escapes the open transaction) heals cleanly when you retry the same
`upgrade=True` open. When `upgrade` is left `False` and the difference is exactly
additive, the raised error carries a `hint` pointing at `upgrade=True`. Additive
upgrade is not offered on the ClickHouse bulk-fenced backend.

Because an added field must be `IdentitySkip`, it is identity-excluded metadata:
the store's metadata-agreement check means the new field can only carry a
non-`None` value on content first saved *after* the upgrade. Re-saving content
that already exists in order to populate the new field on it raises
`EntryMetadataConflictError` (the existing, correct guard against silently
mutating stored metadata), so plan to backfill by rebuilding rather than by
re-saving old content.
