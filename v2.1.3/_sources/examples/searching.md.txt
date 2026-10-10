# Querying a store with the search DSL

Once frozen dataclasses are stored (see the *storable records* example), you
query them with `store.searcher()`. The DSL is deliberately not SQL and not an
ORM: it is a small backend-agnostic protocol (`httk.store.query`) that the SQL
layer implements and that other stores — including *httk-serve*'s in-memory
reference store — implement identically, so the same query program runs
unchanged against either.

A search is built, not written. Four calls do everything:

`variable(cls)`
: Bind a class to a query variable. Two variables of the same class self-join,
  so "another record with the same spacegroup" is expressible.

`add(expression)`
: Add a condition. Expressions come from comparing a variable's fields;
  `&`, `|` and `~` combine them. Several `add()` calls are ANDed.

`output(variable_or_field, name)`
: Declare what a match yields — the whole reconstructed object, one field, or a
  weak-link set (see below). Declaration order is result order.

iteration / `count()`
: Run it. Iterating yields one `SearchResult` per match; `count()` returns how
  many there are, ignoring `set_limit`/`add_offset`.

## What the fields give you

Comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) work on scalar fields. String
fields add `contains`, `startswith` and `endswith`, and these always match
**literal** text: `%` and `_` are ordinary characters, not wildcards. `is_in`
is membership on a root field.

A reference field *chains*: `v.ref.doi == "10.1/a"` joins the referenced table
automatically, and every condition written through the same reference field
shares one join. `v.ref == None` is the IS NULL test, and `v.ref == some_object`
compares against an already-stored instance.

## Variable-length fields are sets

A `list[...]` field lives in a child table, so a condition on it is a statement
about a *set* of rows, and the DSL says which one you mean:

- `v.symbols.has_any("O", "Ca")` — at least one element is in the set.
- `v.symbols.has_only("O", "Ca", "Ti")` — every element is in the set (a
  record with no elements at all satisfies this: the empty set is a subset of
  anything). `is_in` on a child field means exactly the same thing.
- `~` negates the *set* statement rather than the row: `~v.symbols.has_any("O")`
  is "no element is O". For the same reason `~(v.symbols == "O")` also means
  "no element is O", not "some element is not O".

`has_any(a) & has_any(b)` is the "HAS ALL" pattern, and works because each
access to a child field mints an independent join.

## Weak links

Weak links are store-managed associations between record lineages, declared in
`StorageInfo.links` on the source class and reached through a `links` namespace:

- `v.links.<name>.<field> == value` chains into the linked target as a set
  predicate, exactly like a child field; `~`, `has_any` and `has_only` behave
  set-wise, and each access mints an independent join.
- `v.links.<name>` on its own is a set-valued output:
  `results(projects=v.links.projects)` yields, per matched row, a tuple of the
  currently linked targets (latest revision, first-link order), resolved after
  the query and honouring `as_of`. Chaining a field onto a link *output* stays
  rejected — that is a set predicate, not a projectable value.

The full weak-link contract is in the versioned database guide.

## Results

The low-level protocol yields `SearchResult`, a named 2-tuple of `values` (one
per `output()`, in declaration order) and `names`. The canonical `results()`
API yields named lazy rows; use the portable protocol form
`for (structure,), _names in search:` when needed. Result rows bypass the
identity cache; `fetch()` returns lazy rows too (`eager=True` materializes),
and repeated default fetches of one live sid return the same object.

Sorting on a rational field runs on its float companion column, so it is
documented-approximate; the values themselves are still exact.

```{literalinclude} ../../examples/searching.py
:language: python
:lines: 84-
```
