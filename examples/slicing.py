"""Pandas-style slicer indexing over a store

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
"""

from dataclasses import dataclass
from fractions import Fraction

from httk.store.backend.sql import Backend, SqlStore

HTTK_EXAMPLE_REQUIRES = ["sqlalchemy"]


@dataclass(frozen=True)
class Structure:
    formula: str
    spacegroup: int
    energy: Fraction
    rating: int | None = None


STRUCTURES = [
    Structure("CaTiO3", 221, Fraction(-1, 3), 4),
    Structure("NaCl", 225, Fraction(1, 2), None),
    Structure("MgO", 225, Fraction(-5, 4), 5),
    Structure("CaO", 225, Fraction(0), None),
    Structure("SrCaTiO", 62, Fraction(3, 2), 2),
    Structure("Vacuum", 1, Fraction(7, 8), 1),
]


def populate() -> SqlStore:
    """An in-memory store holding the structures."""
    store = SqlStore(Backend.sqlite(), entry_records={})
    with store.transaction():
        for structure in STRUCTURES:
            store.save(structure)
    return store


def main() -> None:
    store = populate()
    note = store.searcher().slicer(Structure)

    print("== Whole slicer ==")
    print(f"  all formulas          -> {sorted(note['formula'])}")
    print(f"  len(note)             -> {len(note)}")

    print("== A single boolean mask ==")
    print(f"  energy < 0            -> {sorted(s.formula for s in note[note['energy'] < 0])}")
    print(f"  spacegroup == 225     -> {sorted(s.formula for s in note[note['spacegroup'] == 225])}")
    print(f"  count of the above    -> {len(note[note['spacegroup'] == 225])}")

    print("== Combined masks (& | ~ ^) ==")
    compound = (note["spacegroup"] == 225) & (note["formula"] != "CaO") | (note["energy"] > Fraction(1))
    print(f"  225 & !CaO | energy>1 -> {sorted(s.formula for s in note[compound])}")
    print(f"  ~(spacegroup == 225)  -> {sorted(s.formula for s in note[~(note['spacegroup'] == 225)])}")

    print("== Helper predicates ==")
    print(f"  formula.isin([...])   -> {sorted(s.formula for s in note[note['formula'].isin(['NaCl', 'MgO'])])}")
    print(f"  rating.isna()         -> {sorted(s.formula for s in note[note['rating'].isna()])}")
    print(f"  rating.notna()        -> {sorted(s.formula for s in note[note['rating'].notna()])}")
    print(f"  spacegroup.between()  -> {sorted(s.formula for s in note[note['spacegroup'].between(62, 225)])}")
    print(f"  formula.str.contains  -> {sorted(s.formula for s in note[note['formula'].str.contains('Ca')])}")


if __name__ == "__main__":
    main()
