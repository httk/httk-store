"""A structures record with its own sites table

This module declares a user-defined `structures` record whose sites live in a
table of their own, one row per site with the columns `x`, `y`, `z` and
`symbol`. It is the record module of the guide
{doc}`../../details/db-redefine-format`, which explains each part.

`SiteRecord` is one row of the site table. `SitesStructureRecord` holds the
lattice basis and the ordered sites, projects itself from any httk
`UnitcellStructure`, and declares the OPTIMADE properties it serves. The call
at the bottom registers it as the `structures` record
`example-sites-structure`.

Run on its own, the script prints the registration and the served
properties. The driver {doc}`build_sites_store` builds, queries and reopens a
store of these records.
"""

# [imports]
import datetime
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

from httk.atomistic import UnitcellStructure
from httk.core.register import register_entry_record
from httk.core.storage import (
    IdentitySkip,
    Indexed,
    QueryContext,
    QueryExpression,
    QueryLiteralError,
    QueryValue,
    Shape,
    StorageInfo,
    StoredPropertyProjection,
    Unique,
    float_matrix,
)

# [/imports]

HTTK_EXAMPLE_REQUIRES = ["httk.atomistic"]


# [site-record]
@dataclass(frozen=True)
class SiteRecord:
    """One site: reduced coordinates and the element symbol occupying it.

    :param x: The first reduced coordinate.
    :param y: The second reduced coordinate.
    :param z: The third reduced coordinate.
    :param symbol: The chemical symbol of the atom at the site.
    """

    # identity_name pins content identity, so moving the module keeps stored ids valid.
    # dedup="none": every saved structure gets its own site rows.
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="example_site", identity_name="example_site", dedup="none"
    )

    x: float
    y: float
    z: float
    symbol: str


# [/site-record]


# [responses]
# The store calls each response with a fetched SitesStructureRecord. The declared
# callable type is Callable[[object], object], hence ``record: Any``.
def _symbols(record: Any) -> list[str]:
    return [site.symbol for site in record.sites]


def _elements(record: Any) -> list[str]:
    return sorted(set(_symbols(record)))


def _nelements_value(record: Any) -> int:
    return len(set(_symbols(record)))


def _nsites_value(record: Any) -> int:
    return len(record.sites)


def _formula(counts: Mapping[str, int], order: Sequence[str]) -> str:
    divisor = math.gcd(*counts.values())
    return "".join(f"{symbol}{n if (n := counts[symbol] // divisor) != 1 else ''}" for symbol in order)


def _reduced(record: Any) -> str:
    counts = Counter(_symbols(record))
    return _formula(counts, sorted(counts))


def _anonymous(record: Any) -> str:
    # ponytail: letters A..Z only, so at most 26 elements; OPTIMADE continues with Aa, Ba, ...
    counts = Counter(_symbols(record))
    divisor = math.gcd(*counts.values())
    amounts = sorted((n // divisor for n in counts.values()), reverse=True)
    return "".join(f"{chr(ord('A') + i)}{n if n != 1 else ''}" for i, n in enumerate(amounts))


def _ratios(record: Any) -> list[float]:
    counts = Counter(_symbols(record))
    return [counts[symbol] / len(record.sites) for symbol in sorted(counts)]


def _lattice_vectors(record: Any) -> list[list[float]]:
    return [list(row) for row in record.basis]


def _fractional(record: Any) -> list[list[float]]:
    return [[site.x, site.y, site.z] for site in record.sites]


def _cartesian(record: Any) -> list[list[float]]:
    return [
        [sum(c * record.basis[k][j] for k, c in enumerate((site.x, site.y, site.z))) for j in range(3)]
        for site in record.sites
    ]


def _species(record: Any) -> list[dict[str, object]]:
    return [{"name": symbol, "chemical_symbols": [symbol], "concentration": [1.0]} for symbol in _elements(record)]


def _last_modified(record: Any) -> str | None:
    return None if record.last_modified is None else record.last_modified.isoformat()


# [/responses]


# [queries]
def _nsites(context: QueryContext) -> QueryValue:
    """The number of site rows of the structure."""
    return context.count(context.scope("sites"))


def _nelements(context: QueryContext) -> QueryValue:
    """The number of distinct ``symbol`` values among the sites."""
    sites = context.scope("sites")
    return context.distinct_count(sites, sites.field("symbol"))


def _integer_filter(
    name: str, value: Callable[[QueryContext], QueryValue]
) -> Callable[[QueryContext, str, object], QueryExpression]:
    """Filter an always-known integer property computed by ``value(context)``."""

    def query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
        if operator == "IS_KNOWN":
            return context.always_true()
        if operator == "IS_UNKNOWN":
            return context.always_false()
        if operator not in {"=", "!=", "<", "<=", ">", ">="}:
            raise QueryLiteralError(f"{name} supports comparisons and IS KNOWN/IS UNKNOWN")
        if not isinstance(literal, int) or isinstance(literal, bool):
            raise QueryLiteralError(f"{name} requires an integer literal")
        left, right = value(context), context.constant(literal)
        if operator == "=":
            return context.equal(left, right)
        if operator == "!=":
            return context.not_(context.equal(left, right))
        return context.compare(left, operator, right)

    return query


def _elements_query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
    """Filter ``elements`` (the distinct site symbols) with HAS ALL/ANY/ONLY and LENGTH."""
    if operator == "IS_KNOWN":
        return context.always_true()
    if operator == "IS_UNKNOWN":
        return context.always_false()
    if operator.startswith("LENGTH "):
        return _integer_filter("elements LENGTH", _nelements)(context, operator.removeprefix("LENGTH "), literal)
    if operator not in {"HAS_ALL", "HAS_ANY", "HAS_ONLY"}:
        raise QueryLiteralError("elements supports HAS ALL/ANY/ONLY, LENGTH and IS KNOWN/IS UNKNOWN")
    values = literal if isinstance(literal, tuple) else (literal,)
    if not values or not all(isinstance(value, str) for value in values):
        raise QueryLiteralError("elements requires element-symbol string literals")
    sites = context.scope("sites")
    matches = [context.equal(sites.field("symbol"), context.constant(value)) for value in values]
    if operator == "HAS_ONLY":
        others = context.filtered(sites, context.not_(context.or_(*matches)))
        return context.equal(context.count(others), context.constant(0))
    present = [context.exists(sites, match) for match in matches]
    return context.and_(*present) if operator == "HAS_ALL" else context.or_(*present)


# [/queries]


# [structure-record]
@dataclass(frozen=True)
class SitesStructureRecord:
    """A periodic structure stored as a basis plus a table of sites.

    :param basis: The three lattice vectors, one per row, in ångström.
    :param sites: The sites, in order.
    :param id: The human-readable entry id; minted by the store when None.
    :param immutable_id: The per-revision immutable id; minted by the store when None.
    :param last_modified: The source modification timestamp, if stated.
    """

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="example_sites_structure", identity_name="example_sites_structure", indexes=(("last_modified",),)
    )
    # store.save(structure, as_record=SitesStructureRecord) accepts any UnitcellStructure,
    # including a UnitcellStructureView loaded from a file.
    __httk_canonical_source__: ClassVar[type] = UnitcellStructure

    basis: Annotated[tuple[tuple[float, ...], ...], Shape(3, 3)]
    sites: tuple[SiteRecord, ...]
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis", float_matrix(self.basis, 3, rows=3))
        object.__setattr__(self, "sites", tuple(self.sites))

    @property
    def type(self) -> str:
        """Expose the OPTIMADE entry type.

        :return: ``structures``.
        """
        return "structures"

    @classmethod
    def __httk_project__(cls, structure: UnitcellStructure) -> Mapping[str, object]:
        """Build the record fields from a unit-cell structure, as plain floats.

        :param structure: The structure to store.
        :return: The record fields.
        :raises ValueError: If a site is not occupied by exactly one element.
        """
        symbol_of: dict[str, str] = {}
        for species in structure.species:
            if len(species.chemical_symbols) != 1 or tuple(species.concentration) != (1,):
                raise ValueError(f"species {species.name!r} is not a single fully occupying element")
            symbol_of[species.name] = species.chemical_symbols[0]
        positions = zip(structure.fractional_site_positions, structure.species_at_sites, strict=True)
        return {
            # No __post_init__ runs on these values: they must already have the field types.
            "basis": float_matrix(structure.lattice_vectors, 3, rows=3),
            "sites": tuple(SiteRecord(x, y, z, symbol_of[name]) for (x, y, z), name in positions),
            # A projection names every field; the store mints the two ids.
            "id": None,
            "immutable_id": None,
            "last_modified": structure.last_modified,
        }

    # [/structure-record]

    # [stored-properties]
    # Served OPTIMADE properties. Unlisted nullable properties are served as null.
    __httk_stored_properties__: ClassVar[Mapping[str, StoredPropertyProjection]] = {
        "structure_features": StoredPropertyProjection(response=lambda record: []),  # the one mandatory property
        "last_modified": StoredPropertyProjection(response=_last_modified),
        "elements": StoredPropertyProjection(response=_elements, query=_elements_query),
        "nelements": StoredPropertyProjection(
            response=_nelements_value, query=_integer_filter("nelements", _nelements), sort=_nelements
        ),
        "elements_ratios": StoredPropertyProjection(response=_ratios),
        "chemical_formula_reduced": StoredPropertyProjection(response=_reduced),
        "chemical_formula_anonymous": StoredPropertyProjection(response=_anonymous),
        "dimension_types": StoredPropertyProjection(response=lambda record: [1, 1, 1]),
        "nperiodic_dimensions": StoredPropertyProjection(response=lambda record: 3),
        "lattice_vectors": StoredPropertyProjection(response=_lattice_vectors),
        "fractional_site_positions": StoredPropertyProjection(response=_fractional),
        "cartesian_site_positions": StoredPropertyProjection(response=_cartesian),
        "nsites": StoredPropertyProjection(
            response=_nsites_value, query=_integer_filter("nsites", _nsites), sort=_nsites
        ),
        "species": StoredPropertyProjection(response=_species),
        "species_at_sites": StoredPropertyProjection(response=_symbols),
    }
    # [/stored-properties]


# [registration]
register_entry_record(name="example-sites-structure", family="structures", record="sites_layout:SitesStructureRecord")
# [/registration]

if __name__ == "__main__":
    from httk.core.register import known_entry_records

    assert "example-sites-structure" in known_entry_records("structures")
    print("registered structures records:", ", ".join(known_entry_records("structures")))
    print("served properties:", ", ".join(SitesStructureRecord.__httk_stored_properties__))
