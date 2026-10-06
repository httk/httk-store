"""Nested (dotted) property names routed through the SQL and Mongo stored-property plans."""

from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, cast

import pytest
from clickhouse_read_support import CLICKHOUSE_PARAM, bulk_store
from httk.core import PropertyDefinition, known_definition_prefixes, load_entry_type_definition
from httk.core.optimade import parse_optimade_filter
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, QueryLiteralError, StorageInfo, StoredPropertyProjection, Unique
from postgres_support import POSTGRES_PARAM, postgres_database

from httk.store import EntryIdScheme
from httk.store.backend.mongo import MongoStore, stored_property_mongo_plan
from httk.store.backend.mongo import stored_properties as mongo_plans
from httk.store.backend.mongo.evaluator import evaluate
from httk.store.backend.sql import Backend, SqlStore, stored_property_sql_plan
from httk.store.backend.sql.stored_properties import _SqlQueryContext
from httk.store.query.optimade_filters import FilterTranslationError, translate_filter_ast

pytestmark = pytest.mark.xdist_group("clickhouse_read_corpus")

CALCULATIONS_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations"


def _member(optimade_type: str, json_type: object, **extra: Any) -> dict[str, Any]:
    return {"x-optimade-type": optimade_type, "x-optimade-unit": "dimensionless", "type": json_type, **extra}


_FLOAT_LIST = _member("list", "array", items=_member("float", "number"))
ERRORS = PropertyDefinition.from_optimade(
    "_httk_test_errors",
    {
        "$id": "https://schemas.httk.org/ad-hoc/defs/properties/_httk_test_errors",
        "description": "Test error statistics.",
        "x-optimade-type": "dictionary",
        "x-optimade-unit": "dimensionless",
        "type": ["object", "null"],
        "properties": {
            "weighting": _member("string", "string"),
            "rmse": _member("float", "number"),
            "offset": _member("float", ["number", "null"]),
            "labels": _member("list", "array", items=_member("string", "string")),
            "extra": _FLOAT_LIST,
            "grid": _member("list", "array", items=_FLOAT_LIST),
        },
        "required": ["weighting", "rmse", "offset", "labels", "grid"],
    },
)


class NestedEntry:
    type = "calculations"
    definition_id = CALCULATIONS_DEFINITION

    @staticmethod
    def entry_type_definition():
        return load_entry_type_definition(CALCULATIONS_DEFINITION).extended({"_httk_test_errors": ERRORS})


# ---------------------------------------------------------------------- hand-written member projections


def _scalar_query(name: str):
    def query(ctx, operator: str, literal: object):
        value = ctx.field(name)
        if operator == "IS_UNKNOWN":
            return ctx.is_null(value)
        if operator == "IS_KNOWN":
            return ctx.not_(ctx.is_null(value))
        if operator in {"=", "!=", "<", "<=", ">", ">=", "CONTAINS", "STARTS", "ENDS"}:
            return ctx.compare(value, operator, ctx.constant(literal))
        raise QueryLiteralError(f"{operator} is not supported on {name}")

    return query


def _list_query(name: str, inner: int = 1, optional: bool = False):
    def query(ctx, operator: str, literal: object):
        present = ctx.equal(ctx.field(f"{name}_present"), ctx.constant(True)) if optional else ctx.always_true()
        if operator == "IS_KNOWN":
            return present
        if operator == "IS_UNKNOWN":
            return ctx.not_(present)
        s = ctx.scope(name)
        value = s.field("value")
        if operator in {"HAS_ALL", "HAS_ANY"}:
            hits = [
                ctx.compare(ctx.count(ctx.filtered(s, ctx.equal(value, ctx.constant(v)))), ">", ctx.constant(0))
                for v in cast(tuple[object, ...], literal)
            ]
            predicate = ctx.and_(*hits) if operator == "HAS_ALL" else ctx.or_(*hits)
        elif operator == "HAS_ONLY":
            others = ctx.and_(*(ctx.not_(ctx.equal(value, ctx.constant(v))) for v in cast(tuple[object, ...], literal)))
            predicate = ctx.compare(ctx.count(ctx.filtered(s, others)), "=", ctx.constant(0))
        elif operator.startswith("LENGTH "):
            predicate = ctx.compare(ctx.count(s), operator[len("LENGTH ") :], ctx.constant(cast(int, literal) * inner))
        else:
            raise QueryLiteralError(f"{operator} is not supported on {name}")
        return ctx.when_known(present, predicate) if optional else predicate

    return query


def _dict_query(ctx, operator: str, literal: object):
    if operator == "IS_KNOWN":
        return ctx.always_true()
    if operator == "IS_UNKNOWN":
        return ctx.always_false()
    raise QueryLiteralError("only IS KNOWN/UNKNOWN apply to _httk_test_errors")


def _errors_response(record: "NestedErrors") -> dict[str, object]:
    out: dict[str, object] = {
        "weighting": record.weighting,
        "rmse": record.rmse,
        "offset": record.offset,
        "labels": list(record.labels),
        "grid": [list(record.grid[:2]), list(record.grid[2:])],
    }
    if record.extra is not None:
        out["extra"] = list(record.extra)
    return out


@dataclass(frozen=True)
class NestedErrors:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="nested_property_errors")

    name: str
    weighting: str
    rmse: float
    offset: float | None
    labels: tuple[str, ...]
    grid: tuple[float, ...]
    extra: tuple[float, ...] | None = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = {
        "_httk_test_errors": StoredPropertyProjection(
            response=_errors_response,
            query=_dict_query,
            members={
                "weighting": StoredPropertyProjection(lambda r: r.weighting, _scalar_query("weighting")),
                "rmse": StoredPropertyProjection(lambda r: r.rmse, _scalar_query("rmse")),
                "offset": StoredPropertyProjection(lambda r: r.offset, _scalar_query("offset")),
                "labels": StoredPropertyProjection(lambda r: list(r.labels), _list_query("labels")),
                "extra": StoredPropertyProjection(lambda r: r.extra, _list_query("extra", optional=True)),
                "grid": StoredPropertyProjection(lambda r: r.grid, _list_query("grid", inner=2)),
            },
        ),
    }


@dataclass(frozen=True)
class NestedPlain:
    """A second backing of the family that does not project ``_httk_test_errors``."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="nested_property_plain")

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


register_entry_family(
    name="test-nested-properties", family=f"{__name__}:NestedEntry", definition_id=CALCULATIONS_DEFINITION
)
register_entry_record(name="test-nested-errors", family="test-nested-properties", record=f"{__name__}:NestedErrors")
register_entry_record(name="test-nested-plain", family="test-nested-properties", record=f"{__name__}:NestedPlain")

GRID = (1.0, 0.0, 0.0, 1.0)
RECORDS = (
    NestedErrors("a1", "atom", 0.1, 0.5, ("O", "H"), GRID, (1.0, 2.0)),
    NestedErrors("a2", "structure", 0.3, None, ("O",), GRID),
    NestedErrors("a3", "atomic", 0.2, 2.0, ("Si", "O", "H"), GRID, (3.0,)),
    NestedErrors("a4", "atom", 0.4, -1.0, ("H",), GRID, ()),
    NestedPlain("b1"),
)
ENTRY_RECORDS = {NestedEntry: (NestedErrors, NestedPlain)}
ALL_ERRORS = {"a1", "a2", "a3", "a4"}

CASES = (
    ("_httk_test_errors.rmse < 0.25", {"a1", "a3"}),
    ("0.25 > _httk_test_errors.rmse", {"a1", "a3"}),
    ('_httk_test_errors.weighting = "atom"', {"a1", "a4"}),
    ('_httk_test_errors.weighting CONTAINS "atom"', {"a1", "a3", "a4"}),
    ('_httk_test_errors.weighting STARTS WITH "struct"', {"a2"}),
    ("_httk_test_errors.offset IS UNKNOWN", {"a2", "b1"}),
    ("_httk_test_errors.offset IS KNOWN", {"a1", "a3", "a4"}),
    ("NOT _httk_test_errors.offset > 1.0", {"a1", "a4"}),
    ('_httk_test_errors.labels HAS "O"', {"a1", "a2", "a3"}),
    ('_httk_test_errors.labels HAS ALL "O","H"', {"a1", "a3"}),
    ('_httk_test_errors.labels HAS ANY "Si","H"', {"a1", "a3", "a4"}),
    ('_httk_test_errors.labels HAS ONLY "O","H"', {"a1", "a2", "a4"}),
    ("_httk_test_errors.labels LENGTH 2", {"a1"}),
    ("_httk_test_errors.grid LENGTH 2", ALL_ERRORS),
    ("_httk_test_errors.grid LENGTH > 2", set()),
    ("_httk_test_errors.extra HAS 1.0", {"a1"}),
    ("NOT _httk_test_errors.extra HAS 1.0", {"a3", "a4"}),
    ("_httk_test_errors.extra LENGTH 1", {"a3"}),
    ("_httk_test_errors.extra IS UNKNOWN", {"a2", "b1"}),
    ("_httk_test_errors IS KNOWN", ALL_ERRORS),
    ("_httk_test_errors IS UNKNOWN", {"b1"}),
)
ERROR_CASES = (
    ("_httk_test_errors.nope = 1", "unrecognized-property"),
    ("_httk_test_errors.grid HAS 1.0", "type-mismatch"),
)


def _names(searchers) -> set[str]:
    return {row[0].name for searcher in searchers for row in searcher.results()}


# ---------------------------------------------------------------------- SQL


@pytest.fixture(scope="module", params=("sqlite", "duckdb", CLICKHOUSE_PARAM, POSTGRES_PARAM))
def sql_plan(request):
    if request.param == "clickhousedb":
        with bulk_store(RECORDS, entry_records=ENTRY_RECORDS) as store:
            yield stored_property_sql_plan(store, NestedEntry)
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
        yield stored_property_sql_plan(store, NestedEntry)


@pytest.mark.parametrize(("filter_string", "expected"), CASES)
def test_sql_plan_routes_nested_member_filters(sql_plan, filter_string, expected):
    assert _names(sql_plan.filter_searchers(filter_string)) == expected


@pytest.mark.parametrize(("filter_string", "category"), ERROR_CASES)
def test_sql_plan_nested_member_errors(sql_plan, filter_string, category):
    with pytest.raises(FilterTranslationError) as excinfo:
        sql_plan.filter_searchers(filter_string)
    assert excinfo.value.category == category


def test_sql_rows_serve_the_whole_dictionary(sql_plan):
    values = [row["_httk_test_errors"] for row in sql_plan.records()]
    assert values.count(None) == 1
    served = sorted((value for value in values if value is not None), key=lambda value: value["rmse"])
    assert [value["weighting"] for value in served] == ["atom", "atomic", "structure", "atom"]
    assert served[0] == {
        "weighting": "atom",
        "rmse": 0.1,
        "offset": 0.5,
        "labels": ["O", "H"],
        "grid": [[1.0, 0.0], [0.0, 1.0]],
        "extra": [1.0, 2.0],
    }


@pytest.mark.parametrize("aggregate", ("count", "distinct_count"))
def test_sql_correlated_count_is_zero_without_matching_children(sql_plan, aggregate):
    """A parent without matching child rows counts 0: ClickHouse's decorrelated subquery reads NULL there."""
    searcher = sql_plan.store.searcher()
    variable = searcher.variable(NestedErrors)
    ctx = _SqlQueryContext(searcher, variable)
    labels = ctx.scope("labels")
    silicon = ctx.filtered(labels, ctx.equal(labels.field("value"), ctx.constant("Si")))
    value = ctx.count(silicon) if aggregate == "count" else ctx.distinct_count(silicon, silicon.field("value"))
    predicate = ctx.compare(value, "=", ctx.constant(0))
    sql_plan._validate_clickhouse_correlation(predicate)
    searcher.add(predicate)
    searcher._output(variable, "record")
    assert {row[0].name for row in searcher.results()} == {"a1", "a2", "a4"}


@pytest.mark.parametrize(
    ("member", "operator", "literal"),
    (
        ("labels", "HAS_ALL", ("O", "H")),
        ("labels", "HAS_ANY", ("O", "H")),
        ("labels", "HAS_ONLY", ("O", "H")),
        ("labels", "LENGTH =", 2),
        ("grid", "LENGTH >=", 2),
        ("extra", "HAS_ALL", (1.0,)),
        ("extra", "HAS_ONLY", (1.0, 2.0)),
        ("extra", "LENGTH =", 1),
    ),
)
def test_sql_list_member_predicates_stay_within_one_correlation_level(member, operator, literal):
    """Every list predicate form is ClickHouse-safe: correlation depth at most one."""
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
        searcher = store.searcher()
        context = _SqlQueryContext(searcher, searcher.variable(NestedErrors))
        query = NestedErrors.__httk_stored_properties__["_httk_test_errors"].members[member].query
        predicate = query(context, operator, literal)
        assert predicate.correlation_depth <= 1
        assert (~predicate).correlation_depth <= 1


# ---------------------------------------------------------------------- Mongo


def _mongo_in_process_names(filter_string: str) -> set[str]:
    """Run the Mongo plan's handlers and evaluator in process (no server).

    The plan only reads the store's entry layout, so a SQLite store supplies
    it; each backing's predicate is then evaluated against the records exactly
    as the plan's row verifier does.
    """
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
        plan = stored_property_mongo_plan(store, NestedEntry)
    matched: set[str] = set()
    for backing in plan._backings:
        context = mongo_plans._MongoQueryContext(backing.backing)
        try:
            predicate = translate_filter_ast(
                parse_optimade_filter(filter_string),
                cast(Any, None),
                mongo_plans._property_fulltypes(plan.definition),
                plan._handlers(backing, context, "", False, False),
                known_definition_prefixes(),
            )
        except QueryLiteralError as error:
            raise FilterTranslationError(str(error), "type-mismatch") from error
        matched |= {
            record.name for record in RECORDS if type(record) is backing.backing and evaluate(predicate, record) is True
        }
    return matched


@pytest.mark.parametrize(("filter_string", "expected"), CASES)
def test_mongo_plan_routes_nested_member_filters_in_process(filter_string, expected):
    assert _mongo_in_process_names(filter_string) == expected


@pytest.mark.parametrize(("filter_string", "category"), ERROR_CASES)
def test_mongo_plan_nested_member_errors_in_process(filter_string, category):
    with pytest.raises(FilterTranslationError) as excinfo:
        _mongo_in_process_names(filter_string)
    assert excinfo.value.category == category


@pytest.mark.parametrize(("filter_string", "expected"), CASES)
def test_mongo_plan_routes_nested_member_filters_live(mongo_test_database, filter_string, expected):
    store = MongoStore(mongo_test_database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
    for record in RECORDS:
        store.save(record)
    assert _names(store.stored_property_plan(NestedEntry).filter_searchers(filter_string)) == expected
