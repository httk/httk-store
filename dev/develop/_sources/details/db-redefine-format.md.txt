# Redefine the underlying database format

This guide shows how to store structures in a physical table layout of your
own. The worked example keeps the sites of each structure in a table with one
row per site and the columns `x`, `y`, `z` and `symbol`. Each step includes the
code from two runnable example scripts:
{doc}`../examples/custom_layout/sites_layout` declares the records, and
{doc}`../examples/custom_layout/build_sites_store` builds, queries and reopens
a store of them.

The approach needs no change to *httk-store*, *httk-atomistic* or
*httk-serve*. You declare your own record class, register it as a `structures`
entry record, and declare how it serves the OPTIMADE properties. It is
possible rather than easy: you take over work that the built-in records do for
you.

## 1. Decide whether you need this

The built-in structure records of *httk-atomistic* are the supported default.
Before you write your own, check whether an existing option covers the need:

- `NumericUnitcellStructureRecord` stores structures as plain floats instead
  of exact values.
- `--records structures=NAME` on `httk collect --into` selects which
  registered record a collected store uses.
- `indexes=` and `skip_indexes=` control which indexes a store creates (see
  [Index policy](db.md#index-policy)).
- {doc}`db-tables` serves an existing SQL database in place, in whatever layout
  it already has.

Write your own record when you need a specific physical layout that none of
these give you.

## 2. Plan the tables

The store derives the tables from the fields of a frozen dataclass. The rules
that matter here are short:

- A scalar field (`float`, `int`, `str`) becomes one column.
- A fixed-shape float tensor, `Annotated[tuple[tuple[float, ...], ...], Shape(3, 3)]`,
  becomes nine inline `DOUBLE` columns `basis_0` to `basis_8`.
- A tuple of a storable dataclass, `tuple[SiteRecord, ...]`, becomes an element
  table for `SiteRecord` plus a link table that keeps the order.

[Declaring, storing, and fetching records](db-records.md) has the full set of
rules.

The store adds its own columns to every record table. `sid` is the local row
id. `_httk_role` marks main rows and dependency rows (see
[Permanentization](db-recovery.md#permanentization-degraded-writes-and-fsck)).
`store_timestamp` records when the row was stored (see
[Store timestamps](db-timestamps.md#store-timestamps)). `logical_id` is the
lineage of the row (see [Record replacement](db-revisions.md#record-replacement-and-lineages)).
`alt_id` and `alt_kind` link alternative representations (see
[Alternatives](db-revisions.md#alternatives)). A content-deduplicated table
also has a unique `content_id` column.

Deduplication decides whether rows are shared. The structure table keeps the
default `dedup="content_id"`, so saving an identical structure twice returns
the existing row. The site table uses `dedup="none"`, so every structure gets
its own site rows. With the default `"content_id"` on the site class,
identical site rows would be shared between structures instead.

On SQLite the example produces these tables (indexes and constraints left out):

```text
CREATE TABLE example_sites_structure (
    sid INTEGER NOT NULL,
    _httk_role SMALLINT NOT NULL,
    store_timestamp BIGINT NOT NULL,
    logical_id BIGINT NOT NULL,
    alt_id BIGINT NOT NULL,
    alt_kind TEXT,
    content_id TEXT NOT NULL,
    basis_0 DOUBLE NOT NULL,
    ...
    basis_8 DOUBLE NOT NULL,
    id TEXT,
    immutable_id TEXT,
    last_modified TEXT,
    PRIMARY KEY (sid)
)
CREATE TABLE example_sites_structure_sites (
    example_sites_structure_sid INTEGER NOT NULL,
    sites_index INTEGER NOT NULL,
    sites_sid INTEGER NOT NULL
)
CREATE TABLE example_site (
    sid INTEGER NOT NULL,
    _httk_role SMALLINT NOT NULL,
    store_timestamp BIGINT NOT NULL,
    logical_id BIGINT NOT NULL,
    alt_id BIGINT NOT NULL,
    alt_kind TEXT,
    x DOUBLE NOT NULL,
    y DOUBLE NOT NULL,
    z DOUBLE NOT NULL,
    symbol TEXT NOT NULL,
    PRIMARY KEY (sid)
)
```

DuckDB creates the same tables with `VARCHAR` in place of `TEXT`. The site
table has no `content_id` column because it is not deduplicated. The link
table row `(structure sid, sites_index, site sid)` places one site at one
position of one structure.

## 3. Declare the site record

The record module starts with these imports, which the snippets below rely on:

```{literalinclude} ../../examples/custom_layout/sites_layout.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

The site record is an ordinary frozen dataclass. `storage_name` sets the table
name. `identity_name` pins its content identity, so moving the module later
keeps stored content ids valid.

```{literalinclude} ../../examples/custom_layout/sites_layout.py
:language: python
:start-after: "# [site-record]"
:end-before: "# [/site-record]"
```

## 4. Declare the structure record and its projection

The structure record holds the lattice basis and the ordered sites. It also
carries the two store-minted ids, `id` and `immutable_id`, and an optional
`last_modified`. These three fields are marked `IdentitySkip()`, so they do
not take part in content identity.

```{literalinclude} ../../examples/custom_layout/sites_layout.py
:language: python
:start-after: "# [structure-record]"
:end-before: "# [/structure-record]"
```

Three class-level declarations connect the record to *httk₂* structures:

- `__httk_canonical_source__` names the structure class the record is built
  from. `UnitcellStructure` is the one that works for structures read from
  files, because `UnitcellStructureView` subclasses it. A
  `NumericUnitcellStructure` is refused with a `TypeError`.
- `__httk_project__` builds the record fields from such a structure. It must
  name every field not marked `Skip`, including `id` and `immutable_id`, which it sets to
  `None` so the store mints them.
- The `type` property returns `"structures"`, the OPTIMADE entry type.

The store validates projected values against the field types. Saving with
`as_record=` does not run `__post_init__`, so the projection must return
canonical values itself. That is why it calls `float_matrix` on the lattice
vectors.

## 5. Declare the served properties

`__httk_stored_properties__` maps OPTIMADE property names to a
{py:class}`~httk.core.storage.StoredPropertyProjection`. A projection has a
`response`, which computes the served value from a fetched record. It may also
have a `query`, which translates a filter into a database predicate, and a
`sort`.

```{literalinclude} ../../examples/custom_layout/sites_layout.py
:language: python
:start-after: "# [stored-properties]"
:end-before: "# [/stored-properties]"
:dedent: 4
```

The rules for this mapping are:

- The only property you must project is `structure_features`. It is the one
  non-nullable `structures` property that the store does not supply itself.
- Unprojected nullable properties are served as `null`.
- `store.stored_property_plan(StructureEntry)` validates the mapping. It names
  any missing mandatory property and rejects names that `structures` does not
  define.
- A property projected without `query=` is response-only. A filter on it
  answers 501 "not implemented". `/info/structures` does not tell
  response-only and filterable properties apart.

The responses are plain functions of the fetched record. The declared
callable type takes `object`, so the example types the argument as
`record: Any`:

```{literalinclude} ../../examples/custom_layout/sites_layout.py
:language: python
:start-after: "# [responses]"
:end-before: "# [/responses]"
```

The queries receive a {py:class}`~httk.core.storage.QueryContext`, the filter
operator and its literal. `context.scope("sites")` reaches the site rows
through the link table. On that scope, `context.count` counts the sites,
`context.distinct_count` counts distinct symbols, and `context.exists` and
`context.filtered` express the `HAS`, `HAS ALL`, `HAS ANY` and `HAS ONLY`
operators on `elements`. A query must handle or reject every operator it can
receive, including `IS KNOWN` and `IS UNKNOWN`. It raises `QueryLiteralError`
for a literal it cannot use.

```{literalinclude} ../../examples/custom_layout/sites_layout.py
:language: python
:start-after: "# [queries]"
:end-before: "# [/queries]"
```

[Serving through OPTIMADE](db-serving.md) describes how stores are served.
[Searching and exact values](db-querying.md) describes the query protocol the
predicates run on.

## 6. Register it as a `structures` record

The store persists the record's registry name in its layout. Every process
that opens the store must therefore know that name. For a script, a
module-level call at the end of the record module is enough:

```{literalinclude} ../../examples/custom_layout/sites_layout.py
:language: python
:start-after: "# [registration]"
:end-before: "# [/registration]"
```

The `record=` reference is a lazy `"module:Class"` string. It must name the
module as other processes import it, here `sites_layout`.

For permanent use, ship a registration package instead. It is the PEP 420
package `httk/registry/entries/<name>/__init__.py`, provided by an installed
distribution or by a directory on `PYTHONPATH`. The directories `httk/`,
`httk/registry/` and `httk/registry/entries/` must not contain an
`__init__.py`:

```text
my_records/
├── httk/
│   └── registry/
│       └── entries/
│           └── sites_layout/
│               └── __init__.py
└── sites_layout.py
```

```python
"""Register the sites-table structures record."""

from httk.core.register import register_entry_record

register_entry_record(name="example-sites-structure", family="structures", record="sites_layout:SitesStructureRecord")
```

The package must not import the record module itself. Remove the module-level
call from `sites_layout.py` once the package exists, since a second
registration of the same name raises `ValueError`. Check the result with:

```console
$ PYTHONPATH=my_records python -c "from httk.core.register import known_entry_records; print(known_entry_records('structures'))"
```

The [module registry guide](https://docs.httk.org/httk-core/dev/main/details/registry.html)
in *httk-core* describes discovery in full.

A store can also bind a record without the registry, through
`entry_families=` (see [Entry families and schema layout](db-schema.md)).
Every reopen then has to pass the same declaration, and the command-line tools
below cannot use it.

## 7. Create the store and save

Declare the record for the `structures` family when you create the store.
Then save each structure with `as_record=SitesStructureRecord`:

```{literalinclude} ../../examples/custom_layout/build_sites_store.py
:language: python
:start-after: "# [create-and-save]"
:end-before: "# [/create-and-save]"
:dedent: 8
```

Pass `as_record=`. A store that declares entry families refuses to save an
entry as a record its family does not declare: `store.save(structure)` without
it raises `ValueError` naming the declared record, instead of writing the
default *httk-atomistic* tables.

The example builds its two structures exactly in code:

```{literalinclude} ../../examples/custom_layout/build_sites_store.py
:language: python
:start-after: "# [structures]"
:end-before: "# [/structures]"
```

## 8. Query it in Python

The search DSL chains from the `sites` field into the site table.
`s.sites.symbol == "Cl"` finds the structures that have a chlorine site.
Conditions on one `s.sites` object apply to the same site, so the second search
finds only the structures with a chlorine site near the origin. A fresh
`s.sites` access matches its own site instead:

```{literalinclude} ../../examples/custom_layout/build_sites_store.py
:language: python
:start-after: "# [search]"
:end-before: "# [/search]"
:dedent: 12
```

A reopened store finds the record class through its registration:

```{literalinclude} ../../examples/custom_layout/build_sites_store.py
:language: python
:start-after: "# [reopen]"
:end-before: "# [/reopen]"
:dedent: 8
```

## 9. Serve it

*httk-serve* serves the store as an OPTIMADE API. With a registration package
on the import path, no extra option is needed. For a one-off, `--records`
imports the record module, and with it the module-level registration, from the
working directory:

```console
$ httk serve optimade check sites.duckdb --records sites_layout:SitesStructureRecord
$ httk serve optimade serve sites.duckdb --records sites_layout:SitesStructureRecord
```

`check` opens the store and tests every endpoint without serving. Filters
then work on the queried properties:

```text
/structures?filter=nsites>2
/structures?filter=elements HAS ALL "Na","Cl"
/structures?sort=nelements
```

The *httk-serve* page
[Deploying a store as an OPTIMADE service](https://docs.httk.org/httk-serve/dev/main/optimade/deployment.html#application-record-classes)
describes `--records` and the other options.

## 10. Collect into it from workflows

`httk collect --into` stores collected structures as your record when you
select it by name:

```console
$ httk collect --into results.sqlite --id-base mydb --records structures=example-sites-structure
```

The collector converts each collected `UnitcellStructure` through the
record's `__httk_canonical_source__` and `__httk_project__`. Records are
selected by registered name only, so the record module must be importable
through a registration package. A collector may also return the record
directly. The *httk-workflow* guide
[Choosing record kinds](https://docs.httk.org/httk-workflow/dev/main/details/collecting.html#choosing-record-kinds)
covers the selection rules.

A job whose structure came from a staged input file, such as a POSCAR staged
by `httk job new`, is collected as two entries. The second one is a
file-recording alternative that carries the staged file as `source_file`. To
collect such jobs, add this field to the structure record and set it to
`None` in `__httk_project__`:

```python
source_file: Annotated[FileRecord | None, IdentitySkipWhenNone()] = None
```

`FileRecord` comes from `httk.core.files` and `IdentitySkipWhenNone` from
`httk.core.storage`. The field adds a reference column to the store's files
table. Without it, such a job reports a `storage_error` that names the missing
`source_file` field.

## 11. Limitations

- A tuple of a storable dataclass always goes through a link table. Inline
  multi-column child rows without a link table are not supported.
- A filter on a response-only property answers 501 "not implemented".
- Every reopen needs the registration, whether from the module-level call,
  `--records` or a registration package.
- Numeric structures are refused unless your projection accepts them through
  its canonical source.
- The structure row's content identity is that of your record. The same
  structure stored as your record and as an *httk-atomistic* record gives two
  different entries.
