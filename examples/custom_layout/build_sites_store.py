"""Building and checking a store with a custom sites table

This driver uses the record module {doc}`sites_layout` to build a DuckDB store
whose structures keep their sites in a table of their own. It is the runnable
counterpart of the guide {doc}`../../details/db-redefine-format`.

In order, the script:

1. builds rock-salt NaCl and CsCl as exact `UnitcellStructure` values;
2. saves both with `as_record=SitesStructureRecord`;
3. validates the served properties with `store.stored_property_plan`;
4. searches the site table for chlorine sites;
5. prints the physical tables and their columns with plain `duckdb`;
6. reopens the store and fetches a record back.

The assertions pin the layout, so the example fails if it changes.
"""

import tempfile
from fractions import Fraction
from pathlib import Path

import duckdb
from httk.atomistic import StructureEntry, UnitcellStructure
from sites_layout import SiteRecord, SitesStructureRecord

from httk.store import DuckdbStore, EntryIdScheme

HTTK_EXAMPLE_REQUIRES = ["httk.atomistic", "duckdb"]

# [structures]
HALF = Fraction(1, 2)


def cubic(a: str) -> list[list[Fraction]]:
    """A cubic cell with the exact edge length ``a`` (ångström)."""
    edge = Fraction(a)
    return [[edge, Fraction(0), Fraction(0)], [Fraction(0), edge, Fraction(0)], [Fraction(0), Fraction(0), edge]]


NACL = UnitcellStructure(
    cell=cubic("5.64"),
    sites=[[0, 0, 0], [0, HALF, HALF], [HALF, 0, HALF], [HALF, HALF, 0]]
    + [[HALF, HALF, HALF], [HALF, 0, 0], [0, HALF, 0], [0, 0, HALF]],
    species_at_sites=["Na"] * 4 + ["Cl"] * 4,
)
CSCL = UnitcellStructure(cell=cubic("4.12"), sites=[[0, 0, 0], [HALF, HALF, HALF]], species_at_sites=["Cs", "Cl"])
# [/structures]


def show_layout(path: Path) -> dict[str, int]:
    """Print every example table with its columns, and return the row counts."""
    with duckdb.connect(str(path), read_only=True) as connection:
        tables = [
            name
            for (name,) in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'example%' ORDER BY 1"
            ).fetchall()
        ]
        counts: dict[str, int] = {}
        for table in tables:
            columns = connection.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
            row = connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()
            counts[table] = 0 if row is None else row[0]
            print(f"  {table} ({counts[table]} rows)")
            print("    " + ", ".join(f"{name} {kind}" for name, kind in columns))
    return counts


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sites.duckdb"

        # [create-and-save]
        with DuckdbStore(
            path, entry_records={StructureEntry: SitesStructureRecord}, entry_ids=EntryIdScheme("example", "1")
        ) as store:
            nacl_sid = store.save(NACL, as_record=SitesStructureRecord)
            store.save(CSCL, as_record=SitesStructureRecord)
            plan = store.stored_property_plan(StructureEntry)  # raises if a projection is wrong
            print(f"stored-property plan for {plan.entry_type!r} is valid")
            # [/create-and-save]

            # [search]
            search = store.searcher()
            site = search.variable(SiteRecord)
            search.add(site.symbol == "Cl")
            chlorine = search.results(site=site)
            print(f"{len(chlorine)} chlorine sites:", [(row.site.x, row.site.y, row.site.z) for row in chlorine])
            # [/search]
            assert len(chlorine) == 5

        print("physical tables:")
        counts = show_layout(path)
        assert counts == {"example_site": 10, "example_sites_structure": 2, "example_sites_structure_sites": 10}

        # [reopen]
        with DuckdbStore(path) as store:  # sites_layout is imported, so the registration is known
            record = store.fetch(SitesStructureRecord, nacl_sid)
            print(f"fetched {record.id}: {[site.symbol for site in record.sites]}")
            # [/reopen]
            assert [site.symbol for site in record.sites] == ["Na"] * 4 + ["Cl"] * 4
            assert record.basis == ((5.64, 0.0, 0.0), (0.0, 5.64, 0.0), (0.0, 0.0, 5.64))


if __name__ == "__main__":
    main()
