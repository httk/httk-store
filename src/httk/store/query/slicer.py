"""A pandas-style ``[]`` indexing layer over the store search DSL.

The :class:`Slicer` compiles bracket indexing into the ordinary
:class:`~httk.store.query.protocols.Searcher` surface — ``variable``, ``add``,
``output``, ``results`` and ``count`` — and adds no query capability of its own.
It is a thin convenience: ``note = store.searcher().slicer(Note)`` gives an
object where ``note['title']`` iterates one field, ``note[note['value'] > 10]``
iterates the matching records, and ``len(note[mask])`` counts them.

Nothing here touches a searcher until it is iterated or measured. A slicer
holds only a zero-argument factory that mints a *fresh* searcher and the target
record class; a column holds a field path; a mask holds a small immutable op
tree (the private ``_Cmp``/``_And``/``_Or``/``_Not`` nodes). Backend search
expressions are bound to one searcher's variable and
:meth:`~httk.store.query.protocols.Searcher.add` mutates that searcher, so
every terminal operation runs against its own fresh
searcher and compiles the op tree against that searcher's variable. Two slicer
operations therefore never share filter state.

No sorting is offered: some conforming stores (the federation) reject it.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["Slicer", "SlicerColumn", "SlicerMask", "SlicerSelection"]

type _Op = Literal["eq", "ne", "lt", "le", "gt", "ge", "is_in", "contains", "startswith", "endswith"]
"""The comparison and matching operators a :class:`_Cmp` node can carry."""

_DUNDER: dict[str, str] = {
    "eq": "__eq__",
    "ne": "__ne__",
    "lt": "__lt__",
    "le": "__le__",
    "gt": "__gt__",
    "ge": "__ge__",
}
"""The rich-comparison operators, mapped to the field dunder that builds them."""


@dataclass(frozen=True, slots=True)
class _Cmp:
    """One leaf comparison against a field reached by ``path``.

    :param path: The attribute names walked from the query variable to the field.
    :param op: The comparison or matching operator to apply.
    :param args: The literal operands the operator consumes.
    """

    path: tuple[str, ...]
    op: _Op
    args: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _And:
    """A conjunction of two op-tree nodes.

    :param left: The left operand node.
    :param right: The right operand node.
    """

    left: "_Node"
    right: "_Node"


@dataclass(frozen=True, slots=True)
class _Or:
    """A disjunction of two op-tree nodes.

    :param left: The left operand node.
    :param right: The right operand node.
    """

    left: "_Node"
    right: "_Node"


@dataclass(frozen=True, slots=True)
class _Not:
    """A negation of one op-tree node.

    :param inner: The node to negate.
    """

    inner: "_Node"


type _Node = _Cmp | _And | _Or | _Not
"""One node of the slicer's tiny boolean op tree."""


def _resolve(variable: Any, path: tuple[str, ...]) -> Any:
    """Walk ``path`` from a query variable to the search field it names.

    An unknown name raises :class:`AttributeError` from the backend variable,
    which is left to propagate to the caller iterating the slicer.

    :param variable: The fresh searcher's query variable to walk from.
    :param path: The attribute names to follow to the target field.
    :return: The backend search field the path resolves to.
    """
    field = variable
    for name in path:
        field = getattr(field, name)
    return field


def _compile(node: _Node, variable: Any) -> Any:
    """Build a backend search expression for ``node`` against one variable.

    Comparison dunders cannot be typed as expression-returning, so they are
    invoked by name — the same idiom the neutral protocol documents.

    :param node: The op-tree node to compile.
    :param variable: The fresh searcher's query variable to bind against.
    :return: The backend search expression the node describes.
    """
    if isinstance(node, _Cmp):
        field = _resolve(variable, node.path)
        if node.op in _DUNDER:
            return getattr(field, _DUNDER[node.op])(node.args[0])
        if node.op == "is_in":
            return field.is_in(*node.args)
        return getattr(field, node.op)(node.args[0])
    if isinstance(node, _And):
        return _compile(node.left, variable) & _compile(node.right, variable)
    if isinstance(node, _Or):
        return _compile(node.left, variable) | _compile(node.right, variable)
    return ~_compile(node.inner, variable)


class Slicer:
    """A pandas-style indexing view over one record class in a store.

    Index it with a field-name string to iterate that field's values, or with a
    boolean :class:`SlicerMask` to iterate the matching records. Iterating the
    slicer itself yields every record; ``len()`` counts them. Each operation
    runs against its own fresh searcher, so operations never share filter state.

    :param make_searcher: A zero-argument callable returning a fresh searcher.
    :param target: The stored record class this slicer indexes.
    """

    def __init__(self, make_searcher: Callable[[], Any], target: Any) -> None:
        self._make = make_searcher
        self._target = target

    def __getitem__(self, key: "str | SlicerMask") -> "SlicerColumn | SlicerSelection":
        """Select a field column (string key) or a filtered record set (mask key).

        :param key: A field-name string, or a boolean mask from this slicer.
        :return: A column view for a string, or a filtered selection for a mask.
        :raises TypeError: If ``key`` is neither a field-name string nor a mask.
        """
        if isinstance(key, str):
            return SlicerColumn(self, (key,))
        if isinstance(key, SlicerMask):
            return SlicerSelection(self, key._ast)
        raise TypeError(
            "slicer indexing accepts a field-name string or a boolean mask only; "
            "column-lists, .loc/.iloc, and integer/slice indexing are not supported, "
            f"got {type(key).__name__}"
        )

    def __iter__(self) -> Iterator[Any]:
        """Iterate every record of the target class.

        :yield: Each reconstructed record instance.
        """
        searcher = self._make()
        variable = searcher.variable(self._target)
        yield from searcher.results(record=variable).scalars("record")

    def __len__(self) -> int:
        """Count every record of the target class.

        :return: The total number of records.
        """
        searcher = self._make()
        searcher.variable(self._target)
        return searcher.count()


class _SlicerStr:
    """The literal string-matching accessor reached through ``column.str``.

    Matching is literal — no regex or wildcard syntax — matching the httk field
    string-matching semantics (``%`` and ``_`` match themselves).

    :param column: The column whose values the matchers test.
    """

    def __init__(self, column: "SlicerColumn") -> None:
        self._column = column

    def contains(self, pat: str) -> "SlicerMask":
        """Match values containing ``pat`` as a literal substring.

        :param pat: The literal substring to find.
        :return: A mask matching values that contain it.
        """
        return SlicerMask(self._column._slicer, _Cmp(self._column._path, "contains", (pat,)))

    def startswith(self, pat: str) -> "SlicerMask":
        """Match values beginning with the literal ``pat``.

        :param pat: The literal prefix to find.
        :return: A mask matching values that start with it.
        """
        return SlicerMask(self._column._slicer, _Cmp(self._column._path, "startswith", (pat,)))

    def endswith(self, pat: str) -> "SlicerMask":
        """Match values ending with the literal ``pat``.

        :param pat: The literal suffix to find.
        :return: A mask matching values that end with it.
        """
        return SlicerMask(self._column._slicer, _Cmp(self._column._path, "endswith", (pat,)))


class SlicerColumn:
    """One field of a :class:`Slicer`, iterable and comparable.

    Iterating yields the field's decoded scalar values. Comparisons and the
    ``isin``/``isna``/``notna``/``between`` helpers, plus the ``.str`` literal
    matchers, build a :class:`SlicerMask` for use as a slicer index key.

    :param slicer: The owning slicer.
    :param path: The attribute names from the query variable to this field.
    """

    def __init__(self, slicer: Slicer, path: tuple[str, ...]) -> None:
        self._slicer = slicer
        self._path = path

    def __iter__(self) -> Iterator[Any]:
        """Iterate this field's decoded scalar values across every record.

        :yield: Each record's decoded field value.
        """
        slicer = self._slicer
        searcher = slicer._make()
        variable = searcher.variable(slicer._target)
        field = _resolve(variable, self._path)
        yield from searcher.results(col=field).scalars("col")

    def __eq__(self, value: object) -> "SlicerMask":  # type: ignore[override]
        return SlicerMask(self._slicer, _Cmp(self._path, "eq", (value,)))

    def __ne__(self, value: object) -> "SlicerMask":  # type: ignore[override]
        return SlicerMask(self._slicer, _Cmp(self._path, "ne", (value,)))

    def __lt__(self, value: Any) -> "SlicerMask":
        return SlicerMask(self._slicer, _Cmp(self._path, "lt", (value,)))

    def __le__(self, value: Any) -> "SlicerMask":
        return SlicerMask(self._slicer, _Cmp(self._path, "le", (value,)))

    def __gt__(self, value: Any) -> "SlicerMask":
        return SlicerMask(self._slicer, _Cmp(self._path, "gt", (value,)))

    def __ge__(self, value: Any) -> "SlicerMask":
        return SlicerMask(self._slicer, _Cmp(self._path, "ge", (value,)))

    __hash__ = object.__hash__

    def isin(self, values: Iterable[Any]) -> "SlicerMask":
        """Match records whose field value is one of ``values``.

        :param values: The membership set.
        :return: A mask matching records in the set.
        """
        return SlicerMask(self._slicer, _Cmp(self._path, "is_in", tuple(values)))

    def isna(self) -> "SlicerMask":
        """Match records whose field value is null.

        :return: A mask matching null field values.
        """
        return SlicerMask(self._slicer, _Cmp(self._path, "eq", (None,)))

    def notna(self) -> "SlicerMask":
        """Match records whose field value is not null.

        :return: A mask matching non-null field values.
        """
        return SlicerMask(self._slicer, _Cmp(self._path, "ne", (None,)))

    def between(self, low: Any, high: Any) -> "SlicerMask":
        """Match records whose field value lies in ``[low, high]`` inclusive.

        :param low: The inclusive lower bound.
        :param high: The inclusive upper bound.
        :return: A mask matching the closed interval.
        """
        return SlicerMask(self._slicer, _And(_Cmp(self._path, "ge", (low,)), _Cmp(self._path, "le", (high,))))

    @property
    def str(self) -> _SlicerStr:
        """The literal string-matching accessor for this column.

        :return: The ``contains``/``startswith``/``endswith`` accessor.
        """
        return _SlicerStr(self)


class SlicerMask:
    """A boolean predicate over a :class:`Slicer`, combinable with ``& | ^ ~``.

    A mask is not iterable and has no comparison operators; it is used only as a
    slicer index key or combined with another mask from the same slicer.

    :param slicer: The owning slicer.
    :param ast: The op tree the mask describes.
    """

    def __init__(self, slicer: Slicer, ast: _Node) -> None:
        self._slicer = slicer
        self._ast = ast

    def _require_sibling(self, other: object) -> "SlicerMask":
        """Return ``other`` when it is a mask from the same slicer.

        :param other: The right-hand operand of a boolean combination.
        :return: The validated sibling mask.
        :raises TypeError: If ``other`` is not a mask of the same slicer.
        """
        if not isinstance(other, SlicerMask) or other._slicer is not self._slicer:
            raise TypeError("boolean mask operators combine two masks from the same slicer")
        return other

    def __and__(self, other: object) -> "SlicerMask":
        return SlicerMask(self._slicer, _And(self._ast, self._require_sibling(other)._ast))

    def __or__(self, other: object) -> "SlicerMask":
        return SlicerMask(self._slicer, _Or(self._ast, self._require_sibling(other)._ast))

    def __xor__(self, other: object) -> "SlicerMask":
        right = self._require_sibling(other)._ast
        left = self._ast
        return SlicerMask(self._slicer, _And(_Or(left, right), _Not(_And(left, right))))

    def __invert__(self) -> "SlicerMask":
        return SlicerMask(self._slicer, _Not(self._ast))


class SlicerSelection:
    """The records of a :class:`Slicer` matching a :class:`SlicerMask`.

    Iterating yields the reconstructed records; ``len()`` counts them. Each runs
    against its own fresh searcher.

    :param slicer: The owning slicer.
    :param ast: The op tree selecting the records.
    """

    def __init__(self, slicer: Slicer, ast: _Node) -> None:
        self._slicer = slicer
        self._ast = ast

    def __iter__(self) -> Iterator[Any]:
        """Iterate the matching reconstructed records.

        :yield: Each matching record instance.
        """
        slicer = self._slicer
        searcher = slicer._make()
        variable = searcher.variable(slicer._target)
        searcher.add(_compile(self._ast, variable))
        yield from searcher.results(record=variable).scalars("record")

    def __len__(self) -> int:
        """Count the matching records.

        :return: The number of matching records.
        """
        slicer = self._slicer
        searcher = slicer._make()
        variable = searcher.variable(slicer._target)
        searcher.add(_compile(self._ast, variable))
        return searcher.count()
