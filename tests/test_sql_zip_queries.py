"""Correlated (zip) list filters on the SQL stored-property plan: ``aligned`` and ``'HAS_ZIP'`` routing."""

from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

import pytest
from clickhouse_read_support import CLICKHOUSE_PARAM, bulk_store
from httk.core import PropertyDefinition, load_entry_type_definition
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, StoredPropertyProjection, Unique, ZipLiteral
from postgres_support import POSTGRES_PARAM, postgres_database

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore, StoredPropertySqlConfigurationError, stored_property_sql_plan
from httk.store.backend.sql.stored_properties import _projection_handlers, _SqlQueryContext
from httk.store.query.optimade_filters import FilterTranslationError

pytestmark = pytest.mark.xdist_group("clickhouse_read_corpus")

CALCULATIONS_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations"
LABELS = "_httk_test_zip_labels"
COUNTS = "_httk_test_zip_counts"
WEIGHTS = "_httk_test_zip_weights"


def _list(name: str, optimade_type: str, json_type: str) -> PropertyDefinition:
    return PropertyDefinition.from_optimade(
        name,
        {
            "$id": f"https://schemas.httk.org/ad-hoc/defs/properties/{name}",
            "description": "Zip test list.",
            "x-optimade-type": "list",
            "x-optimade-unit": "dimensionless",
            "type": ["array", "null"],
            "items": {"x-optimade-type": optimade_type, "x-optimade-unit": "dimensionless", "type": json_type},
        },
    )


class ZipEntry:
    type = "calculations"
    definition_id = CALCULATIONS_DEFINITION

    @staticmethod
    def entry_type_definition():
        return load_entry_type_definition(CALCULATIONS_DEFINITION).extended(
            {
                LABELS: _list(LABELS, "string", "string"),
                COUNTS: _list(COUNTS, "integer", "integer"),
                WEIGHTS: _list(WEIGHTS, "integer", "integer"),
            }
        )


# Served name -> (child scope, field read on the aligned view).
_SLOTS = {LABELS: ("labels", "value"), COUNTS: ("counts", "value"), WEIGHTS: ("sites", "weight")}


def _zip_query(ctx, operator: str, literal: ZipLiteral):
    """Hand-written zip realization over sibling child lists of the root record."""
    if any(path not in _SLOTS for path in literal.paths):
        return None
    root_scopes = [ctx.scope(_SLOTS[path][0]) for path in literal.paths]

    def matches(views, operators, values):
        return ctx.and_(
            *(
                ctx.compare(view.field(_SLOTS[path][1]), op, ctx.constant(value))
                for view, path, op, value in zip(views, literal.paths, operators, values, strict=True)
            )
        )

    def tuple_hit(operators, values):
        views = ctx.aligned(*root_scopes)
        return ctx.compare(ctx.count(ctx.filtered(views[0], matches(views, operators, values))), ">", ctx.constant(0))

    pairs = tuple(zip(literal.operators, literal.values, strict=True))
    if operator == "HAS_ZIP_ALL":
        return ctx.and_(*(tuple_hit(ops, values) for ops, values in pairs))
    if operator == "HAS_ZIP_ANY":
        return ctx.or_(*(tuple_hit(ops, values) for ops, values in pairs))
    views = ctx.aligned(*root_scopes)
    others = ctx.not_(ctx.or_(*(matches(views, ops, values) for ops, values in pairs)))
    return ctx.compare(ctx.count(ctx.filtered(views[0], others)), "=", ctx.constant(0))


@dataclass(frozen=True)
class ZipSite:
    weight: int


@dataclass(frozen=True)
class ZipRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="zip_query_record")

    name: str
    labels: tuple[str, ...]
    counts: tuple[int, ...]
    sites: tuple[ZipSite, ...] = ()
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = {
        LABELS: StoredPropertyProjection(lambda r: list(r.labels), zip_query=_zip_query),
        COUNTS: StoredPropertyProjection(lambda r: list(r.counts)),
        WEIGHTS: StoredPropertyProjection(lambda r: [site.weight for site in r.sites]),
    }


@dataclass(frozen=True)
class ZipPlain:
    """A second backing of the family that projects none of the zipped lists."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="zip_query_plain")

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


register_entry_family(name="test-zip-queries", family=f"{__name__}:ZipEntry", definition_id=CALCULATIONS_DEFINITION)
register_entry_record(name="test-zip-record", family="test-zip-queries", record=f"{__name__}:ZipRecord")
register_entry_record(name="test-zip-plain", family="test-zip-queries", record=f"{__name__}:ZipPlain")


def _sites(*weights: int) -> tuple[ZipSite, ...]:
    return tuple(ZipSite(weight) for weight in weights)


RECORDS = (
    ZipRecord("r1", ("a", "b"), (1, 2), _sites(10, 20)),
    ZipRecord("r2", ("a", "b"), (2, 1), _sites(20, 10)),
    ZipRecord("r3", ("a",), (1,), _sites(10)),
    ZipRecord("r4", ("c", "a", "a"), (5, 3, 1), _sites(1, 2, 3)),
    ZipRecord("r5", (), ()),
    ZipPlain("p1"),
)
ENTRY_RECORDS = {ZipEntry: (ZipRecord, ZipPlain)}
ALL_ZIP = {"r1", "r2", "r3", "r4", "r5"}


def _lit(paths: tuple[str, ...], *tuples: tuple[tuple[str, object], ...]) -> ZipLiteral:
    return ZipLiteral(
        paths,
        tuple(tuple(op for op, _value in slots) for slots in tuples),
        tuple(tuple(value for _op, value in slots) for slots in tuples),
    )


LC = (LABELS, COUNTS)
CASES = (
    ("HAS_ZIP_ALL", _lit(LC, (("=", "a"), ("=", 1))), {"r1", "r3", "r4"}),
    # Positions are correlated: r2 has a and 1, but never at one position.
    ("HAS_ZIP_ALL", _lit(LC, (("=", "a"), ("=", 2))), {"r2"}),
    ("HAS_ZIP_ALL", _lit(LC, (("=", "a"), ("=", 1)), (("=", "b"), ("=", 2))), {"r1"}),
    ("HAS_ZIP_ANY", _lit(LC, (("=", "a"), ("=", 2)), (("=", "c"), ("=", 5))), {"r2", "r4"}),
    ("HAS_ZIP_ONLY", _lit(LC, (("=", "a"), ("=", 1)), (("=", "b"), ("=", 2))), {"r1", "r3", "r5"}),
    ("HAS_ZIP_ALL", _lit(LC, (("=", "a"), (">", 1))), {"r2", "r4"}),
    ("HAS_ZIP_ALL", _lit(LC, (("!=", "a"), ("<=", 2))), {"r1", "r2"}),
    ("HAS_ZIP_ONLY", _lit(LC, (("=", "a"), (">=", 1)), (("=", "c"), ("=", 5))), {"r3", "r4", "r5"}),
    # A storable-element child list aligns with a scalar child list.
    ("HAS_ZIP_ALL", _lit((LABELS, WEIGHTS), (("=", "a"), ("=", 20))), {"r2"}),
    ("HAS_ZIP_ALL", _lit((LABELS, COUNTS, WEIGHTS), (("=", "a"), ("=", 1), ("=", 3))), {"r4"}),
)


def _context(plan, backing: type) -> tuple[Any, Any, Any]:
    searcher = plan.store.searcher()
    variable = searcher.variable(backing)
    return searcher, variable, _SqlQueryContext(searcher, variable)


def _zip(plan, backing: type, has_type: str, literal: ZipLiteral, *, negate: bool = False) -> set[str]:
    """Translate one zip through the plan's handler table and execute it like the plan does."""
    searcher, variable, context = _context(plan, backing)
    configured = next(item for item in plan._backings if item.backing is backing)
    handlers, _targets = plan._handlers(configured, context, "", False, False)
    predicate = handlers[LABELS]["HAS_ZIP"](LABELS, literal, None, has_type)
    if negate:
        predicate = ~predicate
    plan._validate_clickhouse_correlation(predicate)
    searcher.add(predicate)
    searcher._output(variable, "record")
    return {row[0].name for row in searcher.results()}


def _names(plan, has_type: str, literal: ZipLiteral, *, negate: bool = False) -> set[str]:
    return _zip(plan, ZipRecord, has_type, literal, negate=negate) | _zip(
        plan, ZipPlain, has_type, literal, negate=negate
    )


@pytest.fixture(scope="module", params=("sqlite", "duckdb", CLICKHOUSE_PARAM, POSTGRES_PARAM))
def sql_plan(request):
    if request.param == "clickhousedb":
        with bulk_store(RECORDS, entry_records=ENTRY_RECORDS) as store:
            yield stored_property_sql_plan(store, ZipEntry)
        return
    if request.param == "postgresql":
        context: Any = postgres_database()
    elif request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        context = Backend.duckdb()
    else:
        context = Backend.sqlite()
    with context as database:
        store = SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
        for record in RECORDS:
            store.save(record)
        store._clear_identity_caches()
        yield stored_property_sql_plan(store, ZipEntry)


@pytest.mark.parametrize(("has_type", "literal", "expected"), CASES)
def test_zip_handler_correlates_positions(sql_plan, has_type, literal, expected):
    assert _names(sql_plan, has_type, literal) == expected


@pytest.mark.parametrize(("has_type", "literal", "expected"), CASES)
def test_negated_zip_complements_within_the_projecting_backing(sql_plan, has_type, literal, expected):
    # The non-projecting backing is unknown: it matches neither the zip nor its negation.
    assert _names(sql_plan, has_type, literal, negate=True) == ALL_ZIP - expected


def test_backing_without_the_owner_is_unknown(sql_plan):
    literal = _lit(LC, (("=", "a"), ("=", 1)))
    for has_type in ("HAS_ZIP_ALL", "HAS_ZIP_ANY", "HAS_ZIP_ONLY"):
        assert _zip(sql_plan, ZipPlain, has_type, literal) == set()
        assert _zip(sql_plan, ZipPlain, has_type, literal, negate=True) == set()


def test_unsupported_zip_is_not_implemented(sql_plan):
    with pytest.raises(FilterTranslationError) as excinfo:
        _zip(sql_plan, ZipRecord, "HAS_ZIP_ALL", _lit((LABELS, "other"), (("=", "a"), ("=", 1))))
    assert excinfo.value.category == "not-implemented"
    assert LABELS in str(excinfo.value)


@pytest.mark.parametrize("has_type", ("HAS_ZIP_ALL", "HAS_ZIP_ANY", "HAS_ZIP_ONLY"))
def test_zip_count_form_stays_within_one_correlation_level(has_type):
    """The aligned count form is ClickHouse-safe: correlation depth at most one, also negated."""
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
        searcher = store.searcher()
        context = _SqlQueryContext(searcher, searcher.variable(ZipRecord))
        literal = _lit((LABELS, COUNTS, WEIGHTS), (("=", "a"), ("=", 1), ("=", 3)), (("=", "b"), (">", 1), ("<", 9)))
        predicate = _zip_query(context, has_type, literal)
        assert predicate.correlation_depth <= 1
        assert (~predicate).correlation_depth <= 1
        views = context.aligned(context.scope("labels"), context.scope("sites"))
        assert all(view.correlation_depth == 1 for view in views)
        assert context.count(views[1]).correlation_depth == 1


def test_aligned_views_are_fresh_per_call():
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
        searcher = store.searcher()
        context = _SqlQueryContext(searcher, searcher.variable(ZipRecord))
        labels, counts = context.scope("labels"), context.scope("counts")
        first = context.aligned(labels, counts)
        second = context.aligned(labels, counts)
        assert first[0].froms == first[1].froms
        assert not {id(alias) for alias in first[0].froms} & {id(alias) for alias in second[0].froms}
        assert not {id(alias) for alias in first[0].froms} & {id(alias) for alias in labels.froms}


def test_aligned_rejects_non_sibling_scopes():
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
        searcher = store.searcher()
        context = _SqlQueryContext(searcher, searcher.variable(ZipRecord))
        labels = context.scope("labels")
        sites = context.scope("sites")
        with pytest.raises(StoredPropertySqlConfigurationError, match="at least two"):
            context.aligned(labels)
        with pytest.raises(StoredPropertySqlConfigurationError, match="unfiltered child"):
            context.aligned(labels, context.filtered(sites, context.always_true()))
        with pytest.raises(StoredPropertySqlConfigurationError, match="unfiltered child"):
            context.aligned(context._root, labels)
        other = _SqlQueryContext(searcher, searcher.variable(ZipRecord))
        with pytest.raises(StoredPropertySqlConfigurationError, match="same parent"):
            context.aligned(labels, other.scope("counts"))


def test_foreign_zip_result_is_a_configuration_error(sql_plan):
    _searcher, _variable, context = _context(sql_plan, ZipRecord)
    projection = StoredPropertyProjection(lambda r: None, zip_query=lambda ctx, op, lit: object())
    table = _projection_handlers(projection, context)
    assert set(table) == {"HAS_ZIP"}
    with pytest.raises(StoredPropertySqlConfigurationError, match="foreign expression"):
        table["HAS_ZIP"](LABELS, _lit(LC, (("=", "a"), ("=", 1))), None, "HAS_ZIP_ALL")


_LZ = f"{LABELS}:{COUNTS}"
FILTER_CASES = (
    (f'{_LZ} HAS "a":1', {"r1", "r3", "r4"}),
    (f'{_LZ} HAS "a":2', {"r2"}),
    (f'{_LZ} HAS ALL "a":1, "b":2', {"r1"}),
    (f'{_LZ} HAS ANY "a":2, "c":5', {"r2", "r4"}),
    (f'{_LZ} HAS ONLY "a":1, "b":2', {"r1", "r3", "r5"}),
    (f'{_LZ} HAS "a":>1', {"r2", "r4"}),
    (f'{_LZ} HAS !="a":<=2', {"r1", "r2"}),
    # The backing without the zipped lists is unknown: it matches neither form.
    (f'NOT {_LZ} HAS "a":1', {"r2", "r5"}),
    (f'{LABELS}:{WEIGHTS} HAS "a":20', {"r2"}),
    (f'{LABELS}:{COUNTS}:{WEIGHTS} HAS "a":1:3', {"r4"}),
)


@pytest.mark.parametrize(("filter_string", "expected"), FILTER_CASES)
def test_zip_filter_strings_route_through_the_plan(sql_plan, filter_string, expected):
    names = {row[0].name for searcher in sql_plan.filter_searchers(filter_string) for row in searcher.results()}
    assert names == expected


@pytest.mark.parametrize(
    ("filter_string", "category"),
    (
        # The owner is the first named property; counts declares no zip_query.
        (f'{COUNTS}:{LABELS} HAS 1:"a"', "not-implemented"),
        (f'{_LZ} HAS "a":"x"', "type-mismatch"),
    ),
)
def test_zip_filter_string_errors(sql_plan, filter_string, category):
    with pytest.raises(FilterTranslationError) as excinfo:
        sql_plan.filter_searchers(filter_string)
    assert excinfo.value.category == category
