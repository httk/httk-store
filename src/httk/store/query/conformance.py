"""State the neutral :class:`~httk.store.query.protocols.Store` query contract as data.

This module is the executable statement of the truth table every store that
implements :mod:`httk.store.query.protocols` must follow:

- Scalar comparisons (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``) and literal
  string matching (``contains``/``startswith``/``endswith``, in which ``%``
  and ``_`` match themselves) against a NULL value are *unknown*. ``~unknown``
  is unknown, unknown rows do not match, and ``&``/``|`` follow Kleene logic.
  ``field == None`` and ``field != None`` are ``IS [NOT] NULL`` and definite.
  (OPTIMADE: comparisons involving unknown values MUST NOT match.)
- ``is_in(...)`` is definite: NULL matches only an explicit ``None`` member,
  and ``~is_in(1)`` keeps NULL rows.
- Set operations treat a NULL list as the empty set: ``has``/``has_any`` are
  false, ``has_only`` is true, and all stay definite under ``~``.
- ``count()`` is exact and ignores limit and offset.
- Sorting places NULLs last in both directions.
- String-matching case sensitivity is backend-defined, so no case depends on it.

A third-party store loads :func:`conformance_rows` into one table or
collection (``rid`` is the row id, ``tags`` a list field) and then either calls
:func:`check_query_conformance` or parametrizes its own tests over
:func:`conformance_cases`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["ConformanceCase", "check_query_conformance", "conformance_cases", "conformance_rows"]


def conformance_rows() -> tuple[dict[str, Any], ...]:
    """Return fresh copies of the canonical conformance fixture rows.

    Each row has a non-null string id ``rid`` (``"a"`` to ``"f"``), nullable
    scalars ``name``, ``n`` and ``x``, and a nullable list ``tags``.

    :return: Six new row dictionaries.
    """
    return (
        {"rid": "a", "name": "alpha", "n": 1, "x": 0.5, "tags": ["p", "q"]},
        {"rid": "b", "name": "50%_off", "n": 2, "x": 1.5, "tags": ["p"]},
        {"rid": "c", "name": None, "n": None, "x": None, "tags": None},
        {"rid": "d", "name": "beta_gamma", "n": 3, "x": None, "tags": []},
        {"rid": "e", "name": "5000xoff", "n": None, "x": 2.5, "tags": ["q"]},
        {"rid": "f", "name": "gamma", "n": 2, "x": -1.0, "tags": ["r"]},
    )


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    """Describe one query over :func:`conformance_rows` and its expected ids.

    :param name: A unique case name.
    :param build: Build the filter expression from the search variable, or ``None`` for no filter.
    :param expected: The expected row ids; order matters only when ``ordered``.
    :param sort: ``(field, descending)`` sort keys, most significant first.
    :param limit: An optional result limit.
    :param offset: A result offset.
    :param ordered: Whether ``expected`` is compared in order.
    """

    name: str
    build: Callable[[Any], Any] | None
    expected: tuple[str, ...]
    sort: tuple[tuple[str, bool], ...] = ()
    limit: int | None = None
    offset: int = 0
    ordered: bool = False


def _case(name: str, build: Callable[[Any], Any], expected: str) -> ConformanceCase:
    return ConformanceCase(name, build, tuple(expected))


def conformance_cases() -> tuple[ConformanceCase, ...]:
    """Return the conformance cases over :func:`conformance_rows`.

    :return: The cases, with unique names.
    """
    # Rows: n = a1 b2 c- d3 e- f2; x = a.5 b1.5 c- d- e2.5 f-1; name c is NULL;
    # tags = a[p,q] b[p] c NULL d[] e[q] f[r].
    by_n = (("n", False), ("rid", False))
    return (
        # Scalar comparisons: NULL rows c and e never match.
        _case("n == 2", lambda v: v.n == 2, "bf"),
        _case("n != 2", lambda v: v.n != 2, "ad"),
        _case("n < 2", lambda v: v.n < 2, "a"),
        _case("n <= 2", lambda v: v.n <= 2, "abf"),
        _case("n > 2", lambda v: v.n > 2, "d"),
        _case("n >= 2", lambda v: v.n >= 2, "bdf"),
        _case("x > 0.0", lambda v: v.x > 0.0, "abe"),
        _case("name == 'alpha'", lambda v: v.name == "alpha", "a"),
        # Negation of an unknown comparison stays unknown.
        _case("~(n == 2)", lambda v: ~(v.n == 2), "ad"),
        _case("~(n != 2)", lambda v: ~(v.n != 2), "bf"),
        _case("~(n < 2)", lambda v: ~(v.n < 2), "bdf"),
        _case("~(n <= 2)", lambda v: ~(v.n <= 2), "d"),
        _case("~(n > 2)", lambda v: ~(v.n > 2), "abf"),
        _case("~(n >= 2)", lambda v: ~(v.n >= 2), "a"),
        _case("~(x > 0.0)", lambda v: ~(v.x > 0.0), "f"),
        _case("~(name == 'alpha')", lambda v: ~(v.name == "alpha"), "bdef"),
        # IS [NOT] NULL is definite.
        _case("n == None", lambda v: v.n == None, "ce"),
        _case("n != None", lambda v: v.n != None, "abdf"),
        _case("~(n == None)", lambda v: ~(v.n == None), "abdf"),
        _case("~(n != None)", lambda v: ~(v.n != None), "ce"),
        # Literal string matching: % and _ match themselves.
        _case("name.contains('%_')", lambda v: v.name.contains("%_"), "b"),
        _case("name.contains('_')", lambda v: v.name.contains("_"), "bd"),
        _case("name.startswith('50%')", lambda v: v.name.startswith("50%"), "b"),
        _case("name.endswith('_off')", lambda v: v.name.endswith("_off"), "b"),
        _case("name.contains('a')", lambda v: v.name.contains("a"), "adf"),
        _case("~name.contains('a')", lambda v: ~v.name.contains("a"), "be"),
        _case("~name.startswith('50%')", lambda v: ~v.name.startswith("50%"), "adef"),
        # is_in is definite.
        _case("n.is_in(1, 3)", lambda v: v.n.is_in(1, 3), "ad"),
        _case("~n.is_in(1, 3)", lambda v: ~v.n.is_in(1, 3), "bcef"),
        _case("n.is_in(None, 1)", lambda v: v.n.is_in(None, 1), "ace"),
        _case("~n.is_in(None, 1)", lambda v: ~v.n.is_in(None, 1), "bdf"),
        # Set operations: a NULL list is the empty set, definite under ~.
        _case("tags.has('p')", lambda v: v.tags.has("p"), "ab"),
        _case("~tags.has('p')", lambda v: ~v.tags.has("p"), "cdef"),
        _case("tags.has_any('q', 'r')", lambda v: v.tags.has_any("q", "r"), "aef"),
        _case("~tags.has_any('q', 'r')", lambda v: ~v.tags.has_any("q", "r"), "bcd"),
        _case("tags.has_only('p')", lambda v: v.tags.has_only("p"), "bcd"),
        _case("~tags.has_only('p')", lambda v: ~v.tags.has_only("p"), "aef"),
        # Kleene logic: c is (unknown, true), e is (unknown, false).
        _case("(n > 1) | (name == None)", lambda v: (v.n > 1) | (v.name == None), "bcdf"),
        _case("(n > 1) & (name == None)", lambda v: (v.n > 1) & (v.name == None), ""),
        _case("~((n > 1) & (name == None))", lambda v: ~((v.n > 1) & (v.name == None)), "abdef"),
        _case("~((n > 1) | (name == None))", lambda v: ~((v.n > 1) | (v.name == None)), "a"),
        _case("(n > 1) | always_true", lambda v: (v.n > 1) | v.always_true(), "abcdef"),
        # Constants.
        _case("always_true", lambda v: v.always_true(), "abcdef"),
        _case("always_false", lambda v: v.always_false(), ""),
        # Ordering and paging; ties broken by rid.
        ConformanceCase("sort n asc", None, tuple("abfdce"), sort=by_n, ordered=True),
        ConformanceCase("sort n desc", None, tuple("dbface"), sort=(("n", True), ("rid", False)), ordered=True),
        ConformanceCase(
            "sort rid limit 2 offset 1", None, ("b", "c"), sort=(("rid", False),), limit=2, offset=1, ordered=True
        ),
        ConformanceCase(
            "n != None sort n limit 2 offset 1",
            lambda v: v.n != None,
            ("b", "f"),
            sort=by_n,
            limit=2,
            offset=1,
            ordered=True,
        ),
    )


def check_query_conformance(store: Any, target: Any, *, id_field: str = "rid") -> None:
    """Run every conformance case against ``store`` holding :func:`conformance_rows`.

    Each case uses a fresh ``store.searcher()`` bound to ``target``. Besides the
    returned ids, ``count()`` must equal the unpaged match count, both before
    and after the case's limit and offset are applied.

    :param store: A :class:`~httk.store.query.protocols.Store` loaded with the conformance rows.
    :param target: The backend target the rows are stored under.
    :param id_field: The field or mapping key holding the row id.
    :raises AssertionError: If any case fails; the message lists every failing case.
    """
    failures: list[str] = []
    for case in conformance_cases():
        searcher = store.searcher()
        variable = searcher.variable(target)
        if case.build is not None:
            searcher.add(case.build(variable))
        unpaged = searcher.count()
        for field, descending in case.sort:
            searcher.add_sort(getattr(variable, field), descending)
        if case.limit is not None:
            searcher.set_limit(case.limit)
        if case.offset:
            searcher.add_offset(case.offset)
        ids = tuple(
            row.row[id_field] if isinstance(row.row, Mapping) else getattr(row.row, id_field)
            for row in searcher.results(row=variable)
        )
        expected = case.expected
        if not case.ordered:
            ids, expected = tuple(sorted(ids)), tuple(sorted(expected))
        if ids != expected:
            failures.append(f"{case.name}: expected {expected}, got {ids}")
        count = searcher.count()
        if count != unpaged:
            failures.append(f"{case.name}: count() changed from {unpaged} to {count} under limit/offset")
        if case.limit is None and not case.offset and unpaged != len(case.expected):
            failures.append(f"{case.name}: count() {unpaged} != {len(case.expected)} matches")
    if failures:
        raise AssertionError("query conformance failures:\n" + "\n".join(failures))
