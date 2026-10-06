"""Correlated (zip) list filters on the Mongo stored-property context and plan."""

from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, cast

import pytest
from httk.core import PropertyDefinition, known_definition_prefixes, load_entry_type_definition
from httk.core.optimade import parse_optimade_filter
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, StoredPropertyProjection, Unique, ZipLiteral

from httk.store import EntryIdScheme
from httk.store.backend.mongo import MongoStore, MongoStoredPropertyConfigurationError, stored_property_mongo_plan
from httk.store.backend.mongo import stored_properties as mongo_plans
from httk.store.backend.mongo.evaluator import canonical_predicate, evaluate
from httk.store.backend.sql import Backend, SqlStore
from httk.store.query.optimade_filters import FilterTranslationError, translate_filter_ast

CALCULATIONS_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations"
PATHS = ("labels", "counts")


def _member(optimade_type: str, json_type: str) -> dict[str, Any]:
    return {"x-optimade-type": optimade_type, "x-optimade-unit": "dimensionless", "type": json_type}


ZIP = PropertyDefinition.from_optimade(
    "_httk_test_zip",
    {
        "$id": "https://schemas.httk.org/ad-hoc/defs/properties/_httk_test_zip",
        "description": "Test zipped lists.",
        "x-optimade-type": "dictionary",
        "x-optimade-unit": "dimensionless",
        "type": ["object", "null"],
        "properties": {
            "labels": {**_member("list", "array"), "items": _member("string", "string")},
            "counts": {**_member("list", "array"), "items": _member("integer", "integer")},
        },
        "required": ["labels", "counts"],
    },
)


class ZipEntry:
    type = "calculations"
    definition_id = CALCULATIONS_DEFINITION

    @staticmethod
    def entry_type_definition():
        return load_entry_type_definition(CALCULATIONS_DEFINITION).extended({"_httk_test_zip": ZIP})


def _zip_query(ctx, operator: str, literal: ZipLiteral):
    """Per value tuple, ``count(filtered(view0, and(slot predicates))) > 0`` over aligned views."""
    if sorted(literal.paths) != sorted(PATHS):
        return None
    views = dict(zip(PATHS, ctx.aligned(*(ctx.scope(path) for path in PATHS)), strict=True))
    ordered = [views[path] for path in literal.paths]
    tuples = [
        ctx.and_(
            *(
                ctx.compare(view.field("value"), op, ctx.constant(value))
                for view, op, value in zip(ordered, ops, values, strict=True)
            )
        )
        for ops, values in zip(literal.operators, literal.values, strict=True)
    ]
    if operator == "HAS_ZIP_ONLY":
        return ctx.compare(ctx.count(ctx.filtered(ordered[0], ctx.not_(ctx.or_(*tuples)))), "=", ctx.constant(0))
    hits = [ctx.compare(ctx.count(ctx.filtered(ordered[0], item)), ">", ctx.constant(0)) for item in tuples]
    return ctx.and_(*hits) if operator == "HAS_ZIP_ALL" else ctx.or_(*hits)


@dataclass(frozen=True)
class ZipRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="mongo_zip_records")

    name: str
    labels: tuple[str, ...]
    counts: tuple[int, ...]
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = {
        "_httk_test_zip": StoredPropertyProjection(
            response=lambda r: {"labels": list(r.labels), "counts": list(r.counts)},
            zip_query=_zip_query,
        ),
    }


@dataclass(frozen=True)
class ZipPlain:
    """A second backing of the family that does not project ``_httk_test_zip``."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="mongo_zip_plain")

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


register_entry_family(name="test-mongo-zip", family=f"{__name__}:ZipEntry", definition_id=CALCULATIONS_DEFINITION)
register_entry_record(name="test-mongo-zip-records", family="test-mongo-zip", record=f"{__name__}:ZipRecord")
register_entry_record(name="test-mongo-zip-plain", family="test-mongo-zip", record=f"{__name__}:ZipPlain")

RECORDS = (
    ZipRecord("z1", ("a", "b"), (1, 2)),
    ZipRecord("z2", ("a", "b"), (2, 1)),
    ZipRecord("z3", ("a",), (1,)),
    ZipRecord("z4", (), ()),
    ZipRecord("z5", ("a", "b"), (1,)),  # Misaligned: every zip over it is UNKNOWN.
    ZipPlain("p1"),
)
ENTRY_RECORDS = {ZipEntry: (ZipRecord, ZipPlain)}


def _literal(*items: tuple[str, str, str, int], paths: tuple[str, str] = PATHS) -> ZipLiteral:
    """Build a literal from ``(label_op, label, count_op, count)`` items (given in ``PATHS`` order)."""
    flip = paths != PATHS
    return ZipLiteral(
        paths,
        tuple((c_op, l_op) if flip else (l_op, c_op) for l_op, _l, c_op, _c in items),
        tuple((c, label) if flip else (label, c) for _lo, label, _co, c in items),
    )


def _eq(*pairs: tuple[str, int]) -> tuple[tuple[str, str, str, int], ...]:
    return tuple(("=", label, "=", count) for label, count in pairs)


CASES = (
    ("HAS_ZIP_ALL", _eq(("a", 1), ("b", 2)), {"z1"}),
    ("HAS_ZIP_ALL", _eq(("a", 1)), {"z1", "z3"}),
    ("HAS_ZIP_ALL", _eq(("a", 2)), {"z2"}),
    ("HAS_ZIP_ALL", _eq(("a", 2), ("b", 2)), set()),
    ("HAS_ZIP_ANY", _eq(("a", 2), ("b", 2)), {"z1", "z2"}),
    ("HAS_ZIP_ONLY", _eq(("a", 1), ("b", 2)), {"z1", "z3", "z4"}),
    ("HAS_ZIP_ONLY", _eq(("a", 2), ("b", 1)), {"z2", "z4"}),
    ("HAS_ZIP_ALL", (("=", "a", ">", 1),), {"z2"}),
    ("HAS_ZIP_ALL", (("!=", "a", "=", 1),), {"z2"}),
    ("HAS_ZIP_ALL", (("=", "b", "<=", 1),), {"z2"}),
    ("HAS_ZIP_ALL", (("=", "a", "<", 2), (">=", "b", "!=", 1)), {"z1"}),
)


def _names(predicate: object, backing: type) -> set[str]:
    return {r.name for r in RECORDS if type(r) is backing and evaluate(cast(Any, predicate), r) is True}


@pytest.mark.parametrize(("operator", "items", "expected"), CASES)
@pytest.mark.parametrize("paths", (PATHS, PATHS[::-1]))
def test_context_zip_semantics(operator, items, expected, paths):
    ctx = mongo_plans._MongoQueryContext(ZipRecord)
    predicate = _zip_query(ctx, operator, _literal(*items, paths=paths))
    assert _names(predicate, ZipRecord) == expected


@pytest.mark.parametrize(("operator", "items", "expected"), CASES)
def test_negation_excludes_unknown_misaligned_records(operator, items, expected):
    ctx = mongo_plans._MongoQueryContext(ZipRecord)
    predicate = _zip_query(ctx, operator, _literal(*items))
    assert evaluate(predicate, RECORDS[4]) is None
    assert evaluate(ctx.not_(predicate), RECORDS[4]) is None
    assert _names(ctx.not_(predicate), ZipRecord) == {"z1", "z2", "z3", "z4"} - expected


def test_aligned_views_read_their_own_member_at_each_position():
    ctx = mongo_plans._MongoQueryContext(ZipRecord)
    labels, counts = ctx.aligned(ctx.scope("labels"), ctx.scope("counts"))
    assert labels.identifier == counts.identifier
    assert evaluate(ctx.compare(ctx.count(counts), "=", ctx.constant(2)), RECORDS[0]) is True
    b_two = ctx.and_(
        ctx.equal(labels.field("value"), ctx.constant("b")), ctx.equal(counts.field("value"), ctx.constant(2))
    )
    assert evaluate(ctx.exists(counts, b_two), RECORDS[0]) is True
    assert evaluate(ctx.exists(counts, b_two), RECORDS[1]) is False
    assert (
        evaluate(ctx.compare(ctx.distinct_count(labels, labels.field("value")), "=", ctx.constant(2)), RECORDS[0])
        is True
    )


def test_aligned_rejects_non_sibling_or_single_scopes():
    ctx = mongo_plans._MongoQueryContext(ZipRecord)
    labels = ctx.scope("labels")
    for scopes in (
        (labels,),
        (ctx._root, labels),
        (ctx.filtered(labels, ctx.always_true()), ctx.scope("counts")),
        (ctx.aligned(labels, ctx.scope("counts"))[0], ctx.scope("counts")),
    ):
        with pytest.raises(MongoStoredPropertyConfigurationError):
            ctx.aligned(*scopes)


def test_canonical_identity_is_stable_and_distinct_per_group():
    def build():
        ctx = mongo_plans._MongoQueryContext(ZipRecord)
        return ctx, canonical_predicate(_zip_query(ctx, "HAS_ZIP_ALL", _literal(*_eq(("a", 1)))))

    ctx, first = build()
    assert first == build()[1]
    one = ctx.aligned(ctx.scope("labels"), ctx.scope("counts"))
    two = ctx.aligned(ctx.scope("labels"), ctx.scope("counts"))
    identities = {canonical_predicate(ctx.compare(ctx.count(view), ">", ctx.constant(0))) for view in (*one, *two)}
    assert len(identities) == 4


def _plan():
    with Backend.sqlite() as database:  # The plan only reads the entry layout.
        store = SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
        return stored_property_mongo_plan(store, ZipEntry)


def _plan_names(plan, operator: str, literal: ZipLiteral) -> set[str]:
    matched: set[str] = set()
    for backing in plan._backings:
        context = mongo_plans._MongoQueryContext(backing.backing)
        handler = plan._handlers(backing, context, "", False, False)["_httk_test_zip"]["HAS_ZIP"]
        matched |= _names(handler("_httk_test_zip", literal, None, operator), backing.backing)
    return matched


@pytest.mark.parametrize(("operator", "items", "expected"), CASES)
def test_plan_routes_has_zip_to_the_owning_projection(operator, items, expected):
    assert _plan_names(_plan(), operator, _literal(*items)) == expected


def test_plan_unprojecting_backing_is_unknown_and_unsupported_is_not_implemented():
    plan = _plan()
    plain = next(backing for backing in plan._backings if backing.backing is ZipPlain)
    context = mongo_plans._MongoQueryContext(ZipPlain)
    handler = plan._handlers(plain, context, "", False, False)["_httk_test_zip"]["HAS_ZIP"]
    predicate = handler("_httk_test_zip", _literal(*_eq(("a", 1))), None, "HAS_ZIP_ALL")
    assert evaluate(predicate, RECORDS[5]) is None
    assert evaluate(context.not_(predicate), RECORDS[5]) is None
    with pytest.raises(FilterTranslationError) as excinfo:
        _plan_names(plan, "HAS_ZIP_ALL", ZipLiteral(("labels",), (("=",),), (("a",),)))
    assert excinfo.value.category == "not-implemented"
    assert "_httk_test_zip with correlated (zip) values" in str(excinfo.value)


def test_dictionary_fulltype_matches_sql_spelling():
    fulltypes = mongo_plans._property_fulltypes(_plan().definition)
    assert fulltypes["_httk_test_zip"] == "dict"
    assert fulltypes["_httk_test_zip.labels"] == "list of string"


Z = "_httk_test_zip.labels:_httk_test_zip.counts"
FILTER_CASES = (
    (f'{Z} HAS "a":1', {"z1", "z3"}),
    (f'{Z} HAS ALL "a":1, "b":2', {"z1"}),
    (f'{Z} HAS ANY "a":2, "b":2', {"z1", "z2"}),
    (f'{Z} HAS ONLY "a":1, "b":2', {"z1", "z3", "z4"}),
    ('_httk_test_zip.counts:_httk_test_zip.labels HAS 2:"a"', {"z2"}),
    (f'{Z} HAS "a":>1', {"z2"}),
    (f'{Z} HAS !="a":1', {"z2"}),
    (f'NOT {Z} HAS "a":1', {"z2", "z4"}),
)


def _in_process_filter_names(filter_string: str) -> set[str]:
    plan = _plan()
    matched: set[str] = set()
    for backing in plan._backings:
        context = mongo_plans._MongoQueryContext(backing.backing)
        predicate = translate_filter_ast(
            parse_optimade_filter(filter_string),
            cast(Any, None),
            mongo_plans._property_fulltypes(plan.definition),
            plan._handlers(backing, context, "", False, False),
            known_definition_prefixes(),
        )
        matched |= _names(predicate, backing.backing)
    return matched


@pytest.mark.parametrize(("filter_string", "expected"), FILTER_CASES)
def test_zip_filter_strings_in_process(filter_string, expected):
    assert _in_process_filter_names(filter_string) == expected


def test_unsupported_zip_filter_string_is_not_implemented():
    with pytest.raises(FilterTranslationError) as excinfo:
        _in_process_filter_names('_httk_test_zip.labels:_httk_test_zip.labels HAS "a":"b"')
    assert excinfo.value.category == "not-implemented"


@pytest.mark.parametrize(("filter_string", "expected"), FILTER_CASES)
def test_zip_filter_strings_live(mongo_test_database, filter_string, expected):
    """The real plan, candidate stream and row verifier over hydrated records."""
    store = MongoStore(mongo_test_database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
    for record in RECORDS:
        store.save(record)
    searchers = store.stored_property_plan(ZipEntry).filter_searchers(filter_string)
    assert {row[0].name for searcher in searchers for row in searcher.results()} == expected
