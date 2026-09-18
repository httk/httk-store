# Pandas-style slicer indexing over a store

`store.searcher().slicer(cls)` wraps the search DSL (see the *searching*
example) in a `[]` indexing surface that reads like pandas. It compiles bracket
indexing into the same `variable`/`add`/`output`/`results` calls and adds no
query capability of its own — so anything the slicer expresses, the plain
searcher expresses too.

```python
note = store.searcher().slicer(Structure)
list(note["formula"])          # one field's values
list(note[note["energy"] < 0]) # the records a boolean mask selects
len(note[note["spacegroup"] == 225])
```

Two indexing keys are accepted, and nothing else:

`note["field"]`
: A field-name string gives a *column* — iterate it for that field's decoded
  values. Comparisons on a column (`==`, `!=`, `<`, `<=`, `>`, `>=`) and the
  helpers `isin`, `isna`, `notna`, `between`, and `.str.contains`/`startswith`/
  `endswith` build a boolean *mask*. String matching is literal: `%` and `_`
  match themselves, never as wildcards.

`note[mask]`
: A boolean mask gives a *selection* — iterate it for the matching
  reconstructed records, or take its `len()`. Masks combine with `&`, `|`, `^`
  and `~` (both operands must be masks of the same slicer).

Every operation runs against a *fresh* searcher, so operations never share
filter state: a filtered selection never leaks its condition into the next one.
Iterating the slicer itself yields every record; `len(note)` counts them.

The slicer never sorts (some stores reject ordering), and it does not offer
`.loc`/`.iloc`, integer or slice indexing, or multi-column selection — reach for
the plain searcher when you need those.

```{literalinclude} ../../examples/slicing.py
:language: python
:lines: 38-
```
