"""Generated typed records from httk-core stored, served and filtered through the SQL and Mongo plans.

Covers the generated core kinds next to generic and hand-written records, a dictionary-typed kind declared in
the generator's exact form (its definition supplied in-test), and adding a kind to an existing store as an
additive declaration upgrade.
"""

import datetime
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, cast

import pytest
from clickhouse_read_support import CLICKHOUSE_PARAM, clickhouse_database
from httk.core import (
    IdentitySkip,
    Indexed,
    PropertyDefinition,
    RunEdge,
    StorageInfo,
    StrongLink,
    TypedRecord,
    TypedRecordSpec,
    Unique,
    known_definition_prefixes,
    load_property_definition,
    typed_records,
)
from httk.core.data_records import RECORDS_DEFINITION_ID, DataRecord, DataRecordEntry, TotalEnergyRecord
from httk.core.definition_ids import STRESS_TENSOR, TEMPERATURE, TOTAL_ENERGY
from httk.core.optimade import parse_optimade_filter
from httk.core.property_records import StressTensorRecord, TemperatureRecord, VolumeRecord
from httk.core.storage import QueryLiteralError, StoredPropertyProjection
from postgres_support import POSTGRES_PARAM, postgres_database

from httk.store import EntryIdScheme
from httk.store.backend.mongo import MongoStore, stored_property_mongo_plan
from httk.store.backend.mongo import stored_properties as mongo_plans
from httk.store.backend.mongo.evaluator import evaluate
from httk.store.backend.sql import Backend, SqlStore, stored_property_sql_plan
from httk.store.backend.sql.stored_properties import _SqlQueryContext
from httk.store.query.optimade_filters import FilterTranslationError, translate_filter_ast
from httk.store.storage_layout import (
    ADDITIVE_DECLARATION_UPGRADE_HINT,
    EntryFamilyDeclaration,
    EntryRecordDeclaration,
    StorageLayoutUpgradeRequiredError,
    family_entry_type_definition,
)

pytestmark = pytest.mark.xdist_group("clickhouse_read_corpus")

IDS = EntryIdScheme("httk.test", "1")
SQL_PARAMS = ("sqlite", "duckdb", CLICKHOUSE_PARAM, POSTGRES_PARAM)


@contextmanager
def _sql_database(param: str) -> Iterator[Any]:
    if param == "clickhousedb":
        context: Any = clickhouse_database()
    elif param == "postgresql":
        context = postgres_database()
    elif param == "duckdb":
        pytest.importorskip("duckdb_engine")
        context = Backend.duckdb()
    else:
        context = Backend.sqlite()
    with context as database:
        yield database


def _populate(store: Any, records: Any, param: str) -> None:
    """Save *records*: ClickHouse through a deferred bulk ingest, as the read suites do."""
    if param == "clickhousedb":
        with store.bulk_ingest(finalize="deferred") as bulk:
            for record in records:
                bulk.save(record)
        return
    for record in records:
        store.save(record)


def _plan(store: SqlStore) -> Any:
    """Plan the records family against its served definition, as the store's serving edge does."""
    layout = next(item for item in store.entry_layout if item.family is DataRecordEntry)
    served = family_entry_type_definition(layout).served_form()
    plan = stored_property_sql_plan(store, DataRecordEntry, served=served)
    assert plan.definition == served == store.stored_property_plan(DataRecordEntry).definition
    return plan


def _edge(target: str) -> tuple[RunEdge, ...]:
    return (RunEdge("structure", "structures", f"ext-{target}"),)


# ---------------------------------------------------------------------- generated core kinds

ABOVE = math.nextafter(300.5, math.inf)  # One ulp above 300.5: exact equality must tell them apart.
CORE = {
    "e": DataRecord.from_value(TOTAL_ENERGY, "e", -1.0, product_of=_edge("e")),
    "total": TotalEnergyRecord(-2.0, product_of=_edge("total")),
    "t250": TemperatureRecord(cast(Any, 250), product_of=_edge("t250")),  # An int literal coerces to float.
    "t300.5": TemperatureRecord(300.5, product_of=_edge("t300.5")),
    "t300.5+": TemperatureRecord(ABOVE, product_of=_edge("t300.5+")),
    "t310": TemperatureRecord(310.0),
    "t1000": TemperatureRecord(1000.25, product_of=(*_edge("t1000"), RunEdge("run", "runs", "ext-run"))),
    "s1": StressTensorRecord((1.5, 1.5, 1.5, 0.0, 0.0, 0.0), product_of=_edge("s1")),
    "s2": StressTensorRecord((1.5, 2.0, 3.0, 0.0, 0.0, 0.25)),
    "s3": StressTensorRecord(cast(Any, [-1, -1, -1, 0, 0, 0]), product_of=_edge("s3")),
    "v": VolumeRecord(42.0, product_of=_edge("v")),
}
CORE_LABELS = {record: label for label, record in CORE.items()}
CORE_ENTRY_RECORDS = {
    DataRecordEntry: (DataRecord, TemperatureRecord, StressTensorRecord, VolumeRecord, TotalEnergyRecord)
}
TEMPERATURES = {"t250", "t300.5", "t300.5+", "t310", "t1000"}
STRESSES = {"s1", "s2", "s3"}

CORE_CASES = (
    ("_httk_temperature > 300", {"t300.5", "t300.5+", "t310", "t1000"}),
    ("_httk_temperature > 300.5", {"t300.5+", "t310", "t1000"}),
    ("_httk_temperature = 300.5", {"t300.5"}),
    ("_httk_temperature = 250", {"t250"}),
    ("NOT _httk_temperature > 300", {"t250"}),
    ("_httk_temperature IS UNKNOWN", set(CORE) - TEMPERATURES),
    ("_httk_temperature IS KNOWN", TEMPERATURES),
    ("_httk_stress_tensor HAS 1.5", {"s1", "s2"}),
    ("_httk_stress_tensor HAS ALL 1.5, 2.0", {"s2"}),
    ("_httk_stress_tensor HAS ANY 3.0, -1.0", {"s2", "s3"}),
    ("_httk_stress_tensor HAS ONLY 1.5, 0.0", {"s1"}),
    ("NOT _httk_stress_tensor HAS 1.5", {"s3"}),
    ("_httk_stress_tensor LENGTH 6", STRESSES),
    ("_httk_stress_tensor IS UNKNOWN", set(CORE) - STRESSES),
    ("_httk_volume < 50 OR _httk_total_energy < -1.5", {"v", "total"}),
)


@pytest.fixture(scope="module", params=SQL_PARAMS)
def core_database(request: pytest.FixtureRequest) -> Iterator[Any]:
    with _sql_database(request.param) as database:
        _populate(SqlStore(database, entry_records=CORE_ENTRY_RECORDS, entry_ids=IDS), CORE.values(), request.param)
        yield database


@pytest.fixture
def core_store(core_database: Any) -> SqlStore:
    """A fresh open (reopen) of the populated store: nothing is served from the writer's caches."""
    return SqlStore(core_database, entry_records=CORE_ENTRY_RECORDS, entry_ids=IDS)


def _stored(store: Any, backing: type) -> list[Any]:
    search = store.searcher()
    variable = search.variable(backing)
    return [row.record for row in search.results(record=variable)]


def test_core_records_round_trip_exactly(core_store: SqlStore) -> None:
    for backing in CORE_ENTRY_RECORDS[DataRecordEntry]:
        stored = _stored(core_store, backing)
        expected = [record for record in CORE.values() if type(record) is backing]
        assert len(stored) == len(expected) > 0
        assert set(stored) == set(expected)  # Value fields and product_of edges, compared exactly.
        for record in stored:
            original = next(item for item in expected if item == record)
            assert record.product_of == original.product_of
            if isinstance(original, TypedRecord):
                assert record.value == original.value
            assert isinstance(record.id, str) and record.immutable_id == f"{record.id}~1"
    temperatures = {record.temperature for record in _stored(core_store, TemperatureRecord)}
    assert temperatures == {250.0, 300.5, ABOVE, 310.0, 1000.25}
    assert all(type(value) is float for value in temperatures)
    s3 = next(record for record in _stored(core_store, StressTensorRecord) if record.stress_tensor[0] < 0)
    assert s3.stress_tensor == (-1.0, -1.0, -1.0, 0.0, 0.0, 0.0) and s3.value == [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]
    assert all(type(component) is float for component in s3.stress_tensor)


def test_core_served_definition_declares_every_kind(core_store: SqlStore) -> None:
    properties = _plan(core_store).definition.properties
    assert properties["_httk_temperature"].definition_id == TEMPERATURE
    assert properties["_httk_temperature"] == load_property_definition(TEMPERATURE).served_form()
    assert properties["_httk_stress_tensor"].definition_id == STRESS_TENSOR
    assert properties["_httk_total_energy"].definition_id == TOTAL_ENERGY
    assert "_httk_volume" in properties


def _core_labels(searchers: Any) -> set[str]:
    return {CORE_LABELS[row[0]] for searcher in searchers for row in searcher.results()}


@pytest.mark.parametrize(("filter_string", "expected"), CORE_CASES)
def test_core_filters(core_store: SqlStore, filter_string: str, expected: set[str]) -> None:
    assert expected and expected < set(CORE)
    assert _core_labels(_plan(core_store).filter_searchers(filter_string)) == expected


@pytest.mark.parametrize("descending", (False, True))
def test_core_sort_by_temperature(core_store: SqlStore, descending: bool) -> None:
    searchers = _plan(core_store).filter_searchers(
        "_httk_temperature IS KNOWN", sort=(("_httk_temperature", descending),)
    )
    ordered = [row[0].temperature for searcher in searchers for row in searcher.results()]
    assert ordered == sorted([250.0, 300.5, ABOVE, 310.0, 1000.25], reverse=descending)


def test_core_response_rows(core_store: SqlStore) -> None:
    rows = {row["id"]: row for row in _plan(core_store).records()}
    assert len(rows) == len(CORE)
    for record in _stored(core_store, TemperatureRecord):
        row = rows[record.id]
        assert type(row["_httk_temperature"]) is float and row["_httk_temperature"] == record.temperature
        assert row["_httk_stress_tensor"] is None and row["_httk_total_energy"] is None
    for record in _stored(core_store, StressTensorRecord):
        row = rows[record.id]
        assert row["_httk_stress_tensor"] == list(record.stress_tensor) and len(row["_httk_stress_tensor"]) == 6
        assert all(type(component) is float for component in row["_httk_stress_tensor"])
        assert row["_httk_temperature"] is None
    (volume,) = _stored(core_store, VolumeRecord)
    assert rows[volume.id]["_httk_volume"] == 42.0 and rows[volume.id]["_httk_temperature"] is None
    (total,) = _stored(core_store, TotalEnergyRecord)
    assert rows[total.id]["_httk_total_energy"] == -2.0
    (generic,) = _stored(core_store, DataRecord)
    assert {key: value for key, value in rows[generic.id].items() if key.startswith("_httk_")} == {
        "_httk_temperature": None,
        "_httk_stress_tensor": None,
        "_httk_volume": None,
        "_httk_total_energy": None,
    }
    assert {row["type"] for row in rows.values()} == {"_httk_records"}


@pytest.mark.parametrize(
    ("operator", "literal"),
    (("HAS_ALL", (1.5, 2.0)), ("HAS_ANY", (1.5, 2.0)), ("HAS_ONLY", (1.5, 0.0)), ("LENGTH =", 6), ("LENGTH >", 5)),
)
def test_stress_tensor_predicates_stay_within_one_correlation_level(operator: str, literal: object) -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records=CORE_ENTRY_RECORDS, entry_ids=IDS)
        searcher = store.searcher()
        context: Any = _SqlQueryContext(searcher, searcher.variable(StressTensorRecord))
        query = StressTensorRecord.__httk_stored_properties__["_httk_stress_tensor"].query
        assert query is not None
        predicate: Any = query(context, operator, literal)
        assert predicate.correlation_depth <= 1
        assert (~predicate).correlation_depth <= 1


# ---------------------------------------------------------------------- a generated dictionary kind

ERRORS_ID = "https://schemas.httk.org/defs/v0.1/properties/test/typed_errors"


def _leaf(kind: str, json_type: list[str], **extra: Any) -> dict[str, Any]:
    return {"x-optimade-type": kind, "x-optimade-unit": "dimensionless", "type": json_type, **extra}


def _list(items: dict[str, Any], name: str, size: int | None) -> dict[str, Any]:
    dims = {"names": [name], "sizes": [size]}
    return {"x-optimade-type": "list", "type": ["array"], "x-optimade-dimensions": dims, "items": items}


ERRORS = PropertyDefinition.from_optimade(
    "typed_errors",
    {
        "$id": ERRORS_ID,
        "description": "Test error statistics.",
        "x-optimade-type": "dictionary",
        "x-optimade-unit": "dimensionless",
        "type": ["object", "null"],
        "properties": {
            "weighting": _leaf("string", ["string"], enum=["atom", "atomic", "structure"]),
            "rmse": _leaf("float", ["number"]),
            "offset": _leaf("float", ["number", "null"]),
            "labels": _list(_leaf("string", ["string"]), "labels", None),
            "extra": _list(_leaf("float", ["number"]), "extra", None),
            "grid": _list(_list(_leaf("float", ["number"]), "cols", 2), "rows", 2),
        },
        "required": ["weighting", "rmse", "offset", "labels", "grid"],
    },
)


@pytest.fixture(scope="module", autouse=True)
def _errors_definition() -> Iterator[None]:
    """Make the definition loader return ``ERRORS``, as the core typed-record runtime tests do."""

    def load(definition_id: str) -> PropertyDefinition:
        return {ERRORS_ID: ERRORS, ZIPS_ID: ZIPS}.get(definition_id) or load_property_definition(definition_id)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(typed_records, "load_property_definition", load)
        yield


@dataclass(frozen=True)
class TypedErrorsRecord(TypedRecord):
    """A dictionary-typed kind in the exact form ``httk registry gen property-records`` emits."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="test_typed_errors",
        identity_name="test_typed_errors",
    )
    __httk_typed_record__: ClassVar[TypedRecordSpec] = TypedRecordSpec(ERRORS_ID, "_httk_typed_errors")
    __httk_property_definitions__: ClassVar[Mapping[str, PropertyDefinition]] = __httk_typed_record__.definitions()
    __httk_stored_properties__: ClassVar[Mapping[str, StoredPropertyProjection]] = __httk_typed_record__.projections()

    weighting: str
    rmse: float
    offset: float | None
    labels: tuple[str, ...]
    grid: tuple[float, ...]
    extra: tuple[float, ...] | None = None
    product_of: Annotated[tuple[RunEdge, ...], StrongLink("product_of", reverse="has_product", role="subject")] = ()
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)


ERRORS_FAMILIES = (
    EntryFamilyDeclaration(
        name="records",
        family=DataRecordEntry,
        definition_id=RECORDS_DEFINITION_ID,
        records=(
            EntryRecordDeclaration(name="core-data-record", record=DataRecord, definition_id=RECORDS_DEFINITION_ID),
            EntryRecordDeclaration(
                name="test-typed-errors", record=TypedErrorsRecord, definition_id=RECORDS_DEFINITION_ID
            ),
        ),
    ),
)
IDENTITY = [[1.0, 0.0], [0.0, 1.0]]
ERROR_VALUES: dict[str, dict[str, Any]] = {
    "a1": {
        "weighting": "atom",
        "rmse": 0.1,
        "offset": 0.5,
        "labels": ["O", "H"],
        "grid": IDENTITY,
        "extra": [1.0, 2.0],
    },
    "a2": {"weighting": "structure", "rmse": 0.3, "offset": None, "labels": ["O"], "grid": [[0.5, 0.25], [0.125, 2.0]]},
    "a3": {
        "weighting": "atomic",
        "rmse": 0.2,
        "offset": 2.0,
        "labels": ["Si", "O", "H"],
        "grid": IDENTITY,
        "extra": [3.0],
    },
    "a4": {"weighting": "atom", "rmse": 0.4, "offset": -1.0, "labels": ["H"], "grid": IDENTITY, "extra": []},
}
ALL_ERRORS = set(ERROR_VALUES)


def _error_records() -> dict[str, Any]:
    """Build the records at use time: construction loads the patched definition."""
    records: dict[str, Any] = {
        label: TypedErrorsRecord.from_value(value, product_of=_edge(label)) for label, value in ERROR_VALUES.items()
    }
    records["b1"] = DataRecord.from_value(TOTAL_ENERGY, "e", -1.0)
    return records


ERROR_CASES = (
    ("_httk_typed_errors.rmse < 0.25", {"a1", "a3"}),
    ("0.25 > _httk_typed_errors.rmse", {"a1", "a3"}),
    ("_httk_typed_errors.rmse = 0.2", {"a3"}),
    ('_httk_typed_errors.weighting = "atom"', {"a1", "a4"}),
    ('_httk_typed_errors.weighting CONTAINS "atom"', {"a1", "a3", "a4"}),
    ('_httk_typed_errors.weighting STARTS WITH "struct"', {"a2"}),
    ('_httk_typed_errors.weighting ENDS WITH "ic"', {"a3"}),
    ("_httk_typed_errors.offset IS UNKNOWN", {"a2", "b1"}),
    ("_httk_typed_errors.offset IS KNOWN", {"a1", "a3", "a4"}),
    ("NOT _httk_typed_errors.offset > 1.0", {"a1", "a4"}),
    ('_httk_typed_errors.labels HAS "O"', {"a1", "a2", "a3"}),
    ('_httk_typed_errors.labels HAS ALL "O","H"', {"a1", "a3"}),
    ('_httk_typed_errors.labels HAS ANY "Si","H"', {"a1", "a3", "a4"}),
    ('_httk_typed_errors.labels HAS ONLY "O","H"', {"a1", "a2", "a4"}),
    ("_httk_typed_errors.labels LENGTH 2", {"a1"}),
    ("_httk_typed_errors.grid LENGTH 2", ALL_ERRORS),
    ("_httk_typed_errors.extra HAS 1.0", {"a1"}),
    ("NOT _httk_typed_errors.extra HAS 1.0", {"a3", "a4"}),
    ("_httk_typed_errors.extra LENGTH 1", {"a3"}),
    ("_httk_typed_errors.extra LENGTH 0", {"a4"}),
    ("_httk_typed_errors.extra IS UNKNOWN", {"a2", "b1"}),
    ("NOT _httk_typed_errors.extra LENGTH 1", {"a1", "a4"}),
    ("_httk_typed_errors IS KNOWN", ALL_ERRORS),
    ("_httk_typed_errors IS UNKNOWN", {"b1"}),
)
ERROR_FAILURES = (
    ("_httk_typed_errors.nope = 1", "unrecognized-property"),
    ("_httk_typed_errors.grid HAS 1.0", "type-mismatch"),
)


def _labels(searchers: Any, records: Mapping[str, Any]) -> set[str]:
    found = [row[0] for searcher in searchers for row in searcher.results()]
    return {
        label
        for label, record in records.items()
        for item in found
        if isinstance(item, type(record)) and item == record
    }


@pytest.fixture(scope="module", params=SQL_PARAMS)
def errors_plan(request: pytest.FixtureRequest) -> Iterator[Any]:
    with _sql_database(request.param) as database:
        _populate(
            SqlStore(database, entry_families=ERRORS_FAMILIES, entry_ids=IDS), _error_records().values(), request.param
        )
        yield _plan(SqlStore(database, entry_families=ERRORS_FAMILIES, entry_ids=IDS))


@pytest.mark.parametrize(("filter_string", "expected"), ERROR_CASES)
def test_sql_dictionary_member_filters(errors_plan: Any, filter_string: str, expected: set[str]) -> None:
    assert _labels(errors_plan.filter_searchers(filter_string), _error_records()) == expected


@pytest.mark.parametrize(("filter_string", "category"), ERROR_FAILURES)
def test_sql_dictionary_member_errors(errors_plan: Any, filter_string: str, category: str) -> None:
    with pytest.raises(FilterTranslationError) as excinfo:
        errors_plan.filter_searchers(filter_string)
    assert excinfo.value.category == category


def test_sql_dictionary_served_definition_and_rows(errors_plan: Any) -> None:
    served = errors_plan.definition.properties["_httk_typed_errors"]
    assert served == ERRORS.served_form() and served.definition_id == ERRORS_ID
    values = [row["_httk_typed_errors"] for row in errors_plan.records()]
    assert values.count(None) == 1
    # Required members always present (null offset stays); the optional member only when stored.
    expected = sorted(ERROR_VALUES.values(), key=lambda value: value["rmse"])
    assert sorted((value for value in values if value is not None), key=lambda value: value["rmse"]) == expected
    a2 = next(value for value in values if value is not None and value["weighting"] == "structure")
    assert "extra" not in a2 and a2["offset"] is None and a2["grid"] == [[0.5, 0.25], [0.125, 2.0]]


def test_dictionary_record_round_trip(errors_plan: Any) -> None:
    stored = _stored(errors_plan.store, TypedErrorsRecord)
    records = _error_records()
    assert set(stored) == {records[label] for label in ERROR_VALUES}
    for record in stored:
        label = next(label for label in ERROR_VALUES if records[label] == record)
        assert record.value == ERROR_VALUES[label]
        assert record.product_of == _edge(label)


# ---------------------------------------------------------------------- Mongo


def _mongo_in_process_labels(
    filter_string: str, families: Any = ERRORS_FAMILIES, records: Mapping[str, Any] | None = None
) -> set[str]:
    """Run the Mongo plan's handlers and evaluator in process (no server), as the nested-name suite does."""
    records = _error_records() if records is None else records
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_families=families, entry_ids=IDS)
        layout = next(item for item in store.entry_layout if item.family is DataRecordEntry)
        plan = stored_property_mongo_plan(
            store, DataRecordEntry, served=family_entry_type_definition(layout).served_form()
        )
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
            label
            for label, record in records.items()
            if type(record) is backing.backing and evaluate(cast(Any, predicate), record) is True
        }
    return matched


@pytest.mark.parametrize(("filter_string", "expected"), ERROR_CASES)
def test_mongo_dictionary_member_filters_in_process(filter_string: str, expected: set[str]) -> None:
    assert _mongo_in_process_labels(filter_string) == expected


@pytest.mark.parametrize(("filter_string", "category"), ERROR_FAILURES)
def test_mongo_dictionary_member_errors_in_process(filter_string: str, category: str) -> None:
    with pytest.raises(FilterTranslationError) as excinfo:
        _mongo_in_process_labels(filter_string)
    assert excinfo.value.category == category


@pytest.mark.parametrize(("filter_string", "expected"), ERROR_CASES)
def test_mongo_dictionary_member_filters_live(mongo_test_database: Any, filter_string: str, expected: set[str]) -> None:
    store = MongoStore(mongo_test_database, entry_families=ERRORS_FAMILIES, entry_ids=IDS)
    records = _error_records()
    for record in records.values():
        store.save(record)
    plan = store.stored_property_plan(DataRecordEntry)
    assert _labels(plan.filter_searchers(filter_string), records) == expected


# ---------------------------------------------------------------------- generated zip (correlated list) filters

ZIPS_ID = "https://schemas.httk.org/defs/v0.1/properties/test/typed_zips"
ZIPS = PropertyDefinition.from_optimade(
    "typed_zips",
    {
        "$id": ZIPS_ID,
        "description": "Test site lists sharing one dimension.",
        "x-optimade-type": "dictionary",
        "x-optimade-unit": "dimensionless",
        "type": ["object", "null"],
        "properties": {
            "labels": _list(_leaf("string", ["string"]), "sites", None),
            "counts": _list(_leaf("integer", ["integer"]), "sites", None),
            "weights": _list(_leaf("float", ["number"]), "sites", None),
            "masses": _list(_leaf("float", ["number", "null"]), "sites", None),
            "other": _list(_leaf("integer", ["integer"]), "other", None),
        },
        "required": ["labels", "counts", "masses", "other"],
    },
)


@dataclass(frozen=True)
class TypedZipsRecord(TypedRecord):
    """A dictionary kind whose one-dimensional members share the ``sites`` dimension (zippable)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="test_typed_zips", identity_name="test_typed_zips"
    )
    __httk_typed_record__: ClassVar[TypedRecordSpec] = TypedRecordSpec(ZIPS_ID, "_httk_typed_zips")
    __httk_property_definitions__: ClassVar[Mapping[str, PropertyDefinition]] = __httk_typed_record__.definitions()
    __httk_stored_properties__: ClassVar[Mapping[str, StoredPropertyProjection]] = __httk_typed_record__.projections()

    labels: tuple[str, ...]
    counts: tuple[int, ...]
    masses: tuple[float | None, ...]
    other: tuple[int, ...]
    weights: tuple[float, ...] | None = None
    product_of: Annotated[tuple[RunEdge, ...], StrongLink("product_of", reverse="has_product", role="subject")] = ()
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)
    last_modified: Annotated[datetime.datetime | None, IdentitySkip()] = field(default=None, compare=False)


ZIPS_FAMILIES = (
    EntryFamilyDeclaration(
        name="records",
        family=DataRecordEntry,
        definition_id=RECORDS_DEFINITION_ID,
        records=(
            EntryRecordDeclaration(name="core-data-record", record=DataRecord, definition_id=RECORDS_DEFINITION_ID),
            EntryRecordDeclaration(name="test-typed-zips", record=TypedZipsRecord, definition_id=RECORDS_DEFINITION_ID),
        ),
    ),
)
ZIP_VALUES: dict[str, dict[str, Any]] = {
    "z1": {"labels": ["a", "b"], "counts": [1, 2], "weights": [0.5, 1.5], "masses": [1.0, None], "other": [7]},
    "z2": {"labels": ["a", "b"], "counts": [2, 1], "masses": [2.0, 2.0], "other": []},
    "z3": {"labels": ["a"], "counts": [1], "weights": [2.0], "masses": [None], "other": [1, 2]},
    "z4": {
        "labels": ["c", "a", "a"],
        "counts": [5, 3, 1],
        "weights": [1.0, 1.0, 1.0],
        "masses": [1.0] * 3,
        "other": [],
    },
    "z5": {"labels": [], "counts": [], "masses": [], "other": []},  # The empty zip: HAS ONLY vacuously true.
}
ALL_ZIPS = {*ZIP_VALUES, "b1"}


def _zip_records() -> dict[str, Any]:
    records: dict[str, Any] = {
        label: TypedZipsRecord.from_value(value, product_of=_edge(label)) for label, value in ZIP_VALUES.items()
    }
    records["b1"] = DataRecord.from_value(TOTAL_ENERGY, "e", -1.0)
    return records


_LC = "_httk_typed_zips.labels:_httk_typed_zips.counts"
_LW = "_httk_typed_zips.labels:_httk_typed_zips.weights"
_LM = "_httk_typed_zips.labels:_httk_typed_zips.masses"
ZIP_CASES = (
    (f'{_LC} HAS "a":1', {"z1", "z3", "z4"}),
    (f'{_LC} HAS "a":2', {"z2"}),  # Positions correlate: z1 holds "a" and 2, never at one position.
    (f'{_LC} HAS ALL "a":1, "b":2', {"z1"}),
    (f'{_LC} HAS ANY "a":2, "c":5', {"z2", "z4"}),
    (f'{_LC} HAS ONLY "a":1, "b":2', {"z1", "z3", "z5"}),
    (f'{_LC} HAS "a":>1', {"z2", "z4"}),
    (f'NOT {_LC} HAS "a":1', {"z2", "z5"}),  # The unprojecting backing (b1) never matches.
    (f'{_LW} HAS "a":1.0', {"z4"}),
    (f'NOT {_LW} HAS "a":1.0', {"z1", "z3"}),  # z2 and z5 lack the optional weights: unknown, neither form.
    # A null slot makes its position unknown: HAS ONLY ignores it (z1's "b", z3's only position) ...
    (f'{_LM} HAS ONLY "a":1.0, "b":2.0', {"z1", "z3", "z5"}),
    ("_httk_typed_zips.masses HAS ONLY 1.0", {"z1", "z3", "z4", "z5"}),
    # ... and it never matches HAS, so z1's ("b", null) is not ("b", 2.0); the empty zip never HAS.
    (f'{_LM} HAS "b":2.0', {"z2"}),
    (f'{_LM} HAS ANY "a":1.0, "b":2.0', {"z1", "z2", "z4"}),
    (f'NOT {_LM} HAS "b":2.0', {"z1", "z3", "z4", "z5"}),
)
ZIP_FAILURES = (("_httk_typed_zips.labels:_httk_typed_zips.other HAS \"a\":1", "not-implemented"),)


@pytest.fixture(scope="module", params=SQL_PARAMS)
def zips_plan(request: pytest.FixtureRequest) -> Iterator[Any]:
    with _sql_database(request.param) as database:
        _populate(
            SqlStore(database, entry_families=ZIPS_FAMILIES, entry_ids=IDS), _zip_records().values(), request.param
        )
        yield _plan(SqlStore(database, entry_families=ZIPS_FAMILIES, entry_ids=IDS))


@pytest.mark.parametrize(("filter_string", "expected"), ZIP_CASES)
def test_sql_generated_zip_filters(zips_plan: Any, filter_string: str, expected: set[str]) -> None:
    assert expected and expected < ALL_ZIPS
    assert _labels(zips_plan.filter_searchers(filter_string), _zip_records()) == expected


@pytest.mark.parametrize(("filter_string", "category"), ZIP_FAILURES)
def test_sql_generated_zip_errors(zips_plan: Any, filter_string: str, category: str) -> None:
    with pytest.raises(FilterTranslationError) as excinfo:
        zips_plan.filter_searchers(filter_string)
    assert excinfo.value.category == category


@pytest.mark.parametrize(("filter_string", "expected"), ZIP_CASES)
def test_mongo_generated_zip_filters_in_process(filter_string: str, expected: set[str]) -> None:
    assert _mongo_in_process_labels(filter_string, ZIPS_FAMILIES, _zip_records()) == expected


@pytest.mark.parametrize(("filter_string", "category"), ZIP_FAILURES)
def test_mongo_generated_zip_errors_in_process(filter_string: str, category: str) -> None:
    with pytest.raises(FilterTranslationError) as excinfo:
        _mongo_in_process_labels(filter_string, ZIPS_FAMILIES, _zip_records())
    assert excinfo.value.category == category


# ---------------------------------------------------------------------- additive upgrade


def test_adding_a_core_kind_is_an_additive_declaration_upgrade(tmp_path: Any) -> None:
    path = tmp_path / "records.sqlite"
    before = {DataRecordEntry: (DataRecord,)}
    after = {DataRecordEntry: (DataRecord, TemperatureRecord)}
    old = [DataRecord.from_value(TOTAL_ENERGY, "e", value) for value in (-1.0, -2.0)]
    with Backend.sqlite(path) as database:
        store = SqlStore(database, entry_records=before, entry_ids=IDS)
        sids = [store.save(record) for record in old]
        ids = [store.fetch(DataRecord, sid, eager=True).id for sid in sids]

    with Backend.sqlite(path) as database, pytest.raises(StorageLayoutUpgradeRequiredError) as refused:
        SqlStore(database, entry_records=after, entry_ids=IDS)
    assert refused.value.remedy == "upgrade" and refused.value.hint == ADDITIVE_DECLARATION_UPGRADE_HINT

    warm = TemperatureRecord(305.0)
    with Backend.sqlite(path) as database:
        store = SqlStore(database, entry_records=after, entry_ids=IDS, upgrade=True)
        for sid, entry_id, record in zip(sids, ids, old, strict=True):
            fetched = store.fetch(DataRecord, sid, eager=True)
            assert fetched == record and fetched.id == entry_id and fetched.value == record.value
        warm_id = store.fetch(TemperatureRecord, store.save(warm), eager=True).id
        assert warm_id not in ids

    with Backend.sqlite(path) as database:
        store = SqlStore(database, entry_records=after, entry_ids=IDS)  # The upgraded declaration now opens plainly.
        plan = _plan(store)

        def matched(filter_string: str) -> set[str]:
            return {row[0].id for searcher in plan.filter_searchers(filter_string) for row in searcher.results()}

        assert matched("_httk_temperature > 300") == {warm_id}
        assert matched("_httk_temperature IS UNKNOWN") == set(ids)
        assert {row["id"]: row["_httk_temperature"] for row in plan.records()} == {**dict.fromkeys(ids), warm_id: 305.0}
        assert _stored(store, TemperatureRecord) == [warm]
