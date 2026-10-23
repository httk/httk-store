# Serving existing tables

`TableStore` queries an existing SQL database in place, read-only, through the
same neutral `Store`/`Searcher` protocol as the httk-store backends. Use it
when a database already exists and should be searched or served without
ingesting it into httk-store's own layout. Every table is reflected with
SQLAlchemy when the store is constructed, so a missing table or column fails
immediately with a `ValueError` naming it.

```python
from httk.store import DelimitedColumn, ListTable, TableSource, TableStore

store = TableStore(
    "sqlite:///materials.sqlite",  # a URL: the store owns the engine; or pass an Engine you own
    {
        "materials": TableSource(
            "materials",  # key defaults to the single-column primary key
            lists={
                "species": ListTable("material_species", parent_column="mat_id", value_column="symbol"),
                "elements": DelimitedColumn("elements_csv", separator=","),
            },
        )
    },
)
searcher = store.searcher()
v = searcher.variable("materials")
searcher.add((v.nsites > 2) & v.elements.has_only("O", "Si"))
searcher.add_sort(v.nsites, True)
for row in searcher.results(material=v):
    print(row.material["mat_id"], row.material["elements"])
store.close()  # disposes only an engine the store created from a URL
```

A list field is stored in one of two ways:

- `ListTable`: one child row per value, joined on the parent's key column.
  Set operations are uncorrelated `IN` subqueries on the child table and
  assume a non-NULL key such as a primary key: `has`/`has_any` and their
  negations never match a NULL-key row. NULL child values are ignored.
- `DelimitedColumn`: delimited text in a parent column. Empty items are
  ignored, and members must be non-empty strings without the separator.
  Items are taken literally, without whitespace stripping: `"p, q"` yields
  `"p"` and `" q"`.

A NULL or absent list is the empty set. Queries follow the neutral truth table
described in
[Implementing the protocol for another backend](db-querying.md#implementing-the-protocol-for-another-backend).

A result row for the search variable is a read-only mapping of the parent
row's columns, with the values the driver returns through SQLAlchemy's column
types (for example `datetime` and `Decimal`) and no presentation coercion,
plus each declared list as a tuple ordered by value (a NULL delimited column
is `None`). Lists are loaded with one query per `ListTable` per batch of up to
500 page keys.

String literals are coerced to the compared column: ISO-8601 text against a
date/time column is parsed as a timestamp. Decimal integer text such as `"3"`
against an integer column compares as the integer 3, and other text compares
against the column's text form, so string ids from a request work against an
integer key.

Limitations:

- Read-only. There are no `as_of` historic queries, revisions, continuation
  pages, JSON-array columns, reference chaining, or multi-table joins.
- ClickHouse is untested. DuckDB primary keys are not reflected by
  `duckdb_engine`, so pass `key=` explicitly there.
- SQLite `LIKE` is ASCII case-insensitive. SQLite datetimes compare as
  timestamps only when they are stored in SQLAlchemy's `DATETIME` text format
  (`YYYY-MM-DD HH:MM:SS.ffffff`). SQLite `NUMERIC` values are floating point.

*httk-serve* serves a `TableStore` over OPTIMADE.
