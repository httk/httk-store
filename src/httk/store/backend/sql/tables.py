"""Query existing SQL tables in place through the neutral store protocol.

:class:`TableStore` implements :class:`~httk.store.query.Store` over tables that
*httk-store* did not create: each target names one existing parent table,
reflected when the store is constructed, optionally with list fields stored
as child rows (:class:`ListTable`) or as delimited text (:class:`DelimitedColumn`).
The store is read-only and follows the truth table of
:mod:`httk.store.query.conformance`:

- Search expressions are plain SQLAlchemy boolean clauses, so ``&``, ``|`` and
  ``~`` follow SQL three-valued logic. Comparisons and literal string matching
  against NULL are unknown; ``== None``/``!= None`` and :meth:`TableField.is_in`
  are definite.
- Set operations read a NULL or absent list as the empty set and are definite.
  :class:`ListTable` set operations assume a non-NULL key such as a primary
  key: ``has``/``has_any`` and their negations never match a NULL-key row.
  NULL child values are ignored.
- Result rows are read-only mappings of the parent row's column values, as
  returned by the driver through SQLAlchemy's column types, plus every
  declared list as a tuple ordered by value.
"""

import datetime
import operator
import re
import types
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import sqlalchemy
from sqlalchemy.exc import NoSuchTableError

from httk.store.query import MultipleResultsError, NoResultError, ResultRow, UnsupportedQueryError

__all__ = [
    "DelimitedColumn",
    "ListTable",
    "TableField",
    "TableResultSet",
    "TableSearcher",
    "TableSource",
    "TableStore",
    "TableVariable",
]

_HYDRATION_BATCH_SIZE = 500
"""The most parent keys bound into one :class:`ListTable` hydration query."""


def _escape_like(text: str) -> str:
    # "!" rather than backslash: duckdb_engine renders ESCAPE '\\' as a two-character escape.
    return text.replace("!", "!!").replace("%", "!%").replace("_", "!_")


@dataclass(frozen=True, slots=True)
class ListTable:
    """Declare a list field stored as child rows of another table.

    :param table: The child table name.
    :param parent_column: The child column holding the parent row's key.
    :param value_column: The child column holding one list value.
    :param schema: The child table's schema, or ``None`` for the default schema.
    """

    table: str
    parent_column: str
    value_column: str
    schema: str | None = None


@dataclass(frozen=True, slots=True)
class DelimitedColumn:
    """Declare a list field stored as delimited text in a parent column.

    Empty items are ignored, so ``''`` and NULL both read as the empty list in
    queries; a NULL column hydrates as ``None`` and ``''`` as ``()``.

    :param column: The parent column holding the delimited text.
    :param separator: The non-empty item separator.
    :raises ValueError: If ``separator`` is empty.
    """

    column: str
    separator: str = ","

    def __post_init__(self) -> None:
        if not self.separator:
            raise ValueError("DelimitedColumn separator must be non-empty")


@dataclass(frozen=True, slots=True)
class TableSource:
    """Describe one existing parent table served as a :class:`TableStore` target.

    :param name: The parent table name.
    :param schema: The parent table's schema, or ``None`` for the default schema.
    :param key: The column list tables join on; defaults to the single-column primary key.
    :param lists: Declared list fields by field name.
    """

    name: str
    schema: str | None = None
    key: str | None = None
    lists: Mapping[str, ListTable | DelimitedColumn] = field(default_factory=dict[str, ListTable | DelimitedColumn])


@dataclass(frozen=True, slots=True)
class _Target:
    """One reflected target: its parent table, key column and list declarations."""

    name: str
    table: sqlalchemy.Table
    key: sqlalchemy.Column[Any] | None
    lists: Mapping[str, tuple[ListTable, sqlalchemy.Table] | DelimitedColumn]


def _reflect(engine: sqlalchemy.Engine, target: str, name: str, schema: str | None) -> sqlalchemy.Table:
    try:
        return sqlalchemy.Table(name, sqlalchemy.MetaData(), autoload_with=engine, schema=schema)
    except NoSuchTableError:
        qualified = name if schema is None else f"{schema}.{name}"
        raise ValueError(f"TableStore target {target!r}: table {qualified!r} does not exist") from None


def _reflect_target(engine: sqlalchemy.Engine, target: str, source: TableSource) -> _Target:
    table = _reflect(engine, target, source.name, source.schema)
    if source.key is not None:
        if source.key not in table.c:
            raise ValueError(f"TableStore target {target!r}: key column {source.key!r} is not in {source.name!r}")
        key: sqlalchemy.Column[Any] | None = table.c[source.key]
    else:
        primary = list(table.primary_key.columns)
        key = primary[0] if len(primary) == 1 else None
    if source.lists and key is None:
        raise ValueError(
            f"TableStore target {target!r}: lists need a key, but {source.name!r} has no single-column "
            "primary key and no key was given"
        )
    lists: dict[str, tuple[ListTable, sqlalchemy.Table] | DelimitedColumn] = {}
    for list_name, spec in source.lists.items():
        if list_name in table.c:
            raise ValueError(f"TableStore target {target!r}: list {list_name!r} collides with a column of that name")
        if isinstance(spec, DelimitedColumn):
            if spec.column not in table.c:
                raise ValueError(
                    f"TableStore target {target!r}: delimited list {list_name!r} column {spec.column!r} "
                    f"is not in {source.name!r}"
                )
            lists[list_name] = spec
            continue
        child = _reflect(engine, target, spec.table, spec.schema)
        for column in (spec.parent_column, spec.value_column):
            if column not in child.c:
                raise ValueError(
                    f"TableStore target {target!r}: list {list_name!r} column {column!r} is not in {spec.table!r}"
                )
        lists[list_name] = (spec, child)
    return _Target(target, table, key, lists)


class TableStore:
    """Query existing SQL tables in place, read-only, through the neutral store protocol.

    Every table is reflected when the store is constructed.

    :param engine: A SQLAlchemy engine owned by the caller, or a SQLAlchemy URL
        from which the store creates, and on :meth:`close` disposes, its own engine.
    :param tables: The served tables by target name.
    :raises ValueError: If a table, key or list column does not exist, a list
        name collides with a column, or lists are declared without a usable key.
    """

    def __init__(self, engine: sqlalchemy.Engine | str, tables: Mapping[str, TableSource]) -> None:
        self._owned = isinstance(engine, str)
        self._engine = sqlalchemy.create_engine(engine) if isinstance(engine, str) else engine
        try:
            self._targets = {target: _reflect_target(self._engine, target, source) for target, source in tables.items()}
        except BaseException:
            self.close()
            raise

    def searcher(self, *, as_of: object = None) -> "TableSearcher":
        """Create an empty searcher.

        :param as_of: Must be ``None``; historic queries are not supported.
        :return: An empty searcher over this store's tables.
        :raises ValueError: If ``as_of`` is given.
        """
        if as_of is not None:
            raise ValueError("TableStore does not support historic queries")
        return TableSearcher._of(self._engine, self._targets)

    def close(self) -> None:
        """Dispose the engine if this store created it from a URL."""
        if self._owned:
            self._engine.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class TableVariable:
    """Bind a search to one :class:`TableStore` target; attributes yield its fields.

    Obtained from :meth:`TableSearcher.variable`.
    """

    __slots__ = ("_target",)
    _target: _Target

    @classmethod
    def _of(cls, target: _Target) -> Self:
        variable = object.__new__(cls)
        variable._target = target
        return variable

    def always_true(self) -> sqlalchemy.ColumnElement[bool]:
        """Return an expression that matches every row.

        :return: The constant true clause.
        """
        return sqlalchemy.true()

    def always_false(self) -> sqlalchemy.ColumnElement[bool]:
        """Return an expression that matches no row.

        :return: The constant false clause.
        """
        return sqlalchemy.false()

    def __getattr__(self, name: str) -> "TableField":
        if name.startswith("__") or name in TableVariable.__slots__:
            raise AttributeError(name)
        target = self._target
        if name in target.table.c or name in target.lists:
            return TableField._of(target, name)
        available = ", ".join([*target.table.c.keys(), *target.lists])
        raise AttributeError(f"table {target.table.name!r} has no field {name!r}; available fields: {available}")


def _literal(column: sqlalchemy.ColumnElement[Any], value: Any) -> Any:
    """Coerce a string literal to an integer or date/time column's type where it has that form."""
    if not isinstance(value, str):
        return value
    if isinstance(column.type, sqlalchemy.Integer):
        return int(value) if re.fullmatch(r"-?[0-9]+", value) else value
    if not isinstance(column.type, sqlalchemy.Date | sqlalchemy.DateTime):
        return value
    try:
        if not isinstance(column.type, sqlalchemy.DateTime):
            return datetime.date.fromisoformat(value)
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"column {column.name!r}: {value!r} is not an ISO-8601 timestamp") from None
    if parsed.tzinfo is not None and not column.type.timezone:
        parsed = parsed.astimezone(datetime.UTC).replace(tzinfo=None)
    return parsed


def _non_integer_text(column: sqlalchemy.ColumnElement[Any], value: Any) -> bool:
    """Whether a coerced literal is text that is not an integer, compared to an integer column."""
    return isinstance(value, str) and isinstance(column.type, sqlalchemy.Integer)


class TableField:
    """A queryable scalar column or declared list of a :class:`TableVariable`.

    Scalar columns support comparisons, literal string matching and
    :meth:`is_in`; lists support :meth:`has`, :meth:`has_any` and
    :meth:`has_only`. Using an operation of the other kind raises
    :class:`TypeError`.

    String literals are coerced to the column: an ISO-8601 string compared to a
    date/time column is parsed (an aware one converted to naive UTC for a naive
    column), and decimal integer text compared to an integer column is read as
    an integer. Other text compares against the column's text form and is
    dropped from :meth:`is_in`. String matching on a non-string column matches its text form.

    Obtained by attribute access on a :class:`TableVariable`.
    """

    __slots__ = ("_name", "_target")
    _name: str
    _target: _Target

    @classmethod
    def _of(cls, target: _Target, name: str) -> Self:
        field = object.__new__(cls)
        field._target = target
        field._name = name
        return field

    def _label(self) -> str:
        return f"{self._target.name}.{self._name}"

    def _column(self) -> sqlalchemy.Column[Any]:
        if self._name in self._target.lists:
            raise TypeError(f"field {self._label()!r} is a list; only has/has_any/has_only apply")
        return self._target.table.c[self._name]

    def _compare(self, operation: Callable[[Any, Any], Any], other: Any) -> sqlalchemy.ColumnElement[bool]:
        column = self._column()
        value = _literal(column, other)
        if not _non_integer_text(column, value):
            return operation(column, value)
        # Non-integer text compares against the column's text form, so NULL stays unknown.
        return operation(sqlalchemy.cast(column, sqlalchemy.String), value)

    def __eq__(self, other: object) -> sqlalchemy.ColumnElement[bool]:  # type: ignore[override]
        return self._column().is_(None) if other is None else self._compare(operator.eq, other)

    def __ne__(self, other: object) -> sqlalchemy.ColumnElement[bool]:  # type: ignore[override]
        return self._column().is_not(None) if other is None else self._compare(operator.ne, other)

    def __hash__(self) -> int:
        return id(self)

    def __lt__(self, other: Any) -> sqlalchemy.ColumnElement[bool]:
        return self._compare(operator.lt, other)

    def __le__(self, other: Any) -> sqlalchemy.ColumnElement[bool]:
        return self._compare(operator.le, other)

    def __gt__(self, other: Any) -> sqlalchemy.ColumnElement[bool]:
        return self._compare(operator.gt, other)

    def __ge__(self, other: Any) -> sqlalchemy.ColumnElement[bool]:
        return self._compare(operator.ge, other)

    def _like(self, pattern: str) -> sqlalchemy.ColumnElement[bool]:
        column = self._column()
        text = column if isinstance(column.type, sqlalchemy.String) else sqlalchemy.cast(column, sqlalchemy.String)
        return text.like(pattern, escape="!")

    def contains(self, text: str) -> sqlalchemy.ColumnElement[bool]:
        """Match values containing ``text`` as a literal substring.

        :param text: The literal text.
        :return: The match clause.
        """
        return self._like(f"%{_escape_like(text)}%")

    def startswith(self, prefix: str) -> sqlalchemy.ColumnElement[bool]:
        """Match values beginning with the literal ``prefix``.

        :param prefix: The literal prefix.
        :return: The match clause.
        """
        return self._like(f"{_escape_like(prefix)}%")

    def endswith(self, suffix: str) -> sqlalchemy.ColumnElement[bool]:
        """Match values ending with the literal ``suffix``.

        :param suffix: The literal suffix.
        :return: The match clause.
        """
        return self._like(f"%{_escape_like(suffix)}")

    def is_in(self, *values: Any) -> sqlalchemy.ColumnElement[bool]:
        r"""Match values equal to one of ``values``, definitely: NULL matches only a ``None`` member.

        :param \*values: The members.
        :return: The membership clause.
        """
        column = self._column()
        members = [_literal(column, value) for value in values if value is not None]
        members = [member for member in members if not _non_integer_text(column, member)]
        clause = sqlalchemy.and_(column.is_not(None), column.in_(members)) if members else sqlalchemy.false()
        return sqlalchemy.or_(column.is_(None), clause) if None in values else clause

    def _members(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        if self._name not in self._target.lists:
            raise TypeError(f"field {self._label()!r} is a scalar column; set operations need a declared list")
        if any(value is None for value in values):
            raise ValueError("None is not a valid member of a set operation")
        spec = self._target.lists[self._name]
        if isinstance(spec, DelimitedColumn):
            for value in values:
                if not isinstance(value, str):
                    raise TypeError(f"delimited list {self._label()!r} members must be str, got {value!r}")
                if not value or spec.separator in value:
                    raise ValueError(
                        f"delimited list {self._label()!r} members must be non-empty and not contain "
                        f"the separator {spec.separator!r}, got {value!r}"
                    )
        return values

    def has(self, value: Any) -> sqlalchemy.ColumnElement[bool]:
        """Match lists containing ``value``.

        :param value: The member.
        :return: The set clause.
        """
        return self.has_any(value)

    def has_any(self, *values: Any) -> sqlalchemy.ColumnElement[bool]:
        r"""Match lists containing any of ``values``.

        :param \*values: The members.
        :return: The set clause.
        """
        values = self._members(values)
        spec = self._target.lists[self._name]
        if isinstance(spec, DelimitedColumn):
            wrapped = self._wrapped(spec)
            patterns = [f"%{_escape_like(spec.separator + value + spec.separator)}%" for value in values]
            return sqlalchemy.or_(sqlalchemy.false(), *(wrapped.like(pattern, escape="!") for pattern in patterns))
        list_table, child = spec
        parent, value_column = child.c[list_table.parent_column], child.c[list_table.value_column]
        members = [_literal(value_column, value) for value in values]
        subquery = sqlalchemy.select(parent).where(value_column.in_(members), parent.is_not(None))
        return self._key().in_(subquery)

    def has_only(self, *values: Any) -> sqlalchemy.ColumnElement[bool]:
        r"""Match lists with no member outside ``values``.

        :param \*values: The allowed members.
        :return: The set clause.
        """
        values = self._members(values)
        spec = self._target.lists[self._name]
        if isinstance(spec, DelimitedColumn):
            separator = spec.separator
            # Doubling every separator gives each item its own pair, so removing
            # separator+value+separator strips every occurrence, duplicates included.
            remainder = sqlalchemy.func.replace(self._wrapped(spec), separator, separator + separator)
            for value in dict.fromkeys(values):
                remainder = sqlalchemy.func.replace(remainder, separator + value + separator, "")
            return sqlalchemy.func.replace(remainder, separator, "") == ""
        list_table, child = spec
        parent, value_column = child.c[list_table.parent_column], child.c[list_table.value_column]
        members = [_literal(value_column, value) for value in values]
        subquery = sqlalchemy.select(parent).where(
            value_column.is_not(None), value_column.not_in(members), parent.is_not(None)
        )
        return self._key().not_in(subquery)

    def _key(self) -> sqlalchemy.Column[Any]:
        key = self._target.key
        assert key is not None  # construction rejects lists without a key
        return key

    def _wrapped(self, spec: DelimitedColumn) -> sqlalchemy.ColumnElement[str]:
        separator = sqlalchemy.literal(spec.separator, sqlalchemy.String)
        column = self._target.table.c[spec.column]
        return separator + sqlalchemy.func.coalesce(column, "", type_=sqlalchemy.String) + separator

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in TableField.__slots__:
            raise AttributeError(name)
        raise AttributeError(f"field {self._label()!r} is not a reference; TableStore fields do not chain ({name!r})")


class TableResultSet:
    """A materialized page of :class:`TableSearcher` results.

    :param rows: The result rows.
    :param names: The declared output names.
    """

    def __init__(self, rows: tuple[ResultRow, ...], names: tuple[str, ...]) -> None:
        self._rows = rows
        self.names = names

    def __iter__(self) -> Iterator[ResultRow]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def first(self) -> ResultRow | None:
        """Return the first row, or ``None`` when no row matched.

        :return: The first row, if any.
        """
        return self._rows[0] if self._rows else None

    def one(self) -> ResultRow:
        """Return the only row.

        :return: The sole row.
        :raises ~httk.store.query.NoResultError: If no row matched.
        :raises ~httk.store.query.MultipleResultsError: If more than one row matched.
        """
        match self._rows:
            case ():
                raise NoResultError("expected exactly one result, found none")
            case (row,):
                return row
            case _:
                raise MultipleResultsError("expected exactly one result, found more than one")

    def scalars(self, name: str | None = None) -> Iterator[Any]:
        """Yield one output from each row.

        :param name: The output name, required when several outputs are declared.
        :return: An iterator over the output values.
        :raises ValueError: If ``name`` is omitted while several outputs are declared.
        :raises KeyError: If ``name`` is not a declared output.
        """
        if name is None:
            if len(self.names) != 1:
                raise ValueError(f"scalars() without a name requires exactly one output; declared: {self.names}")
            name = self.names[0]
        if name not in self.names:
            raise KeyError(f"unknown output {name!r}; declared: {self.names}")
        return (row[name] for row in self._rows)


class TableSearcher:
    """Build and run one read-only query over a :class:`TableStore` target.

    Obtained from :meth:`TableStore.searcher`.
    """

    offset: int
    """The number of leading result rows skipped; :meth:`add_offset` adds to it."""
    _engine: sqlalchemy.Engine
    _targets: Mapping[str, _Target]
    _target: _Target | None
    _filters: list[sqlalchemy.ColumnElement[bool]]
    _sorts: list[tuple[sqlalchemy.Column[Any], bool]]
    _limit: int

    @classmethod
    def _of(cls, engine: sqlalchemy.Engine, targets: Mapping[str, _Target]) -> Self:
        searcher = object.__new__(cls)
        searcher._engine = engine
        searcher._targets = targets
        searcher._target = None
        searcher._filters = []
        searcher._sorts = []
        searcher._limit = -1
        searcher.offset = 0
        return searcher

    def variable(self, target: str) -> TableVariable:
        """Bind the search variable to ``target``.

        :param target: A target name of the store.
        :return: The search variable.
        :raises ValueError: If ``target`` is not a target of the store.
        :raises ~httk.store.query.UnsupportedQueryError: If the searcher is already bound to another target.
        """
        if target not in self._targets:
            raise ValueError(f"unknown TableStore target {target!r}; targets: {', '.join(self._targets)}")
        if self._target is not None and self._target.name != target:
            raise UnsupportedQueryError("a TableStore searcher queries a single target")
        self._target = self._targets[target]
        return TableVariable._of(self._target)

    def add(self, expression: sqlalchemy.ColumnElement[bool]) -> None:
        """Add a filter expression.

        :param expression: An expression built from this searcher's variable.
        """
        self._filters.append(expression)

    def add_sort(self, field: TableField, descending: bool) -> None:
        """Sort by a scalar column, NULLs last in both directions.

        :param field: A scalar field of this searcher's variable.
        :param descending: Whether to sort descending.
        :raises ~httk.store.query.UnsupportedQueryError: If ``field`` is a list.
        """
        if field._name in field._target.lists:
            raise UnsupportedQueryError(f"cannot sort by list field {field._label()!r}")
        self._sorts.append((field._column(), descending))

    def set_limit(self, limit: int) -> None:
        """Set the maximum number of result rows.

        :param limit: The limit; negative means unbounded.
        """
        self._limit = limit

    def add_offset(self, offset: int) -> None:
        """Skip ``offset`` more result rows.

        :param offset: The rows to skip, added to :attr:`offset`.
        """
        self.offset += offset

    def _bound(self) -> _Target:
        if self._target is None:
            raise ValueError("no search variable is bound; call variable(target) first")
        return self._target

    def count(self) -> int:
        """Return the exact number of matching rows, ignoring limit and offset.

        :return: The match count.
        """
        statement = sqlalchemy.select(sqlalchemy.func.count()).select_from(self._bound().table).where(*self._filters)
        with self._engine.connect() as connection:
            return connection.execute(statement).scalar_one()

    def results(self, **outputs: TableVariable | TableField) -> TableResultSet:
        r"""Run the query and return the materialized page.

        :param \*\*outputs: Named outputs: the search variable (the matched row
            mapping) or one of its fields (that field's value).
        :return: The result page.
        :raises ValueError: If no output is given, or an output belongs to another target.
        """
        target = self._bound()
        if not outputs:
            raise ValueError("results() requires at least one output")
        for name, output in outputs.items():
            if not isinstance(output, TableVariable | TableField) or output._target is not target:
                raise ValueError(f"output {name!r} is not the search variable or one of its fields")
        order: list[Any] = []
        for column, descending in self._sorts:
            order += [sqlalchemy.case((column.is_(None), 1), else_=0), column.desc() if descending else column.asc()]
        if target.key is not None and (order or self._limit >= 0 or self.offset):
            order.append(target.key)
        statement = sqlalchemy.select(target.table).where(*self._filters).order_by(*order)
        if self._limit >= 0:
            statement = statement.limit(self._limit)
        if self.offset:
            statement = statement.offset(self.offset)
        with self._engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(statement).mappings()]
            self._hydrate(connection, target, rows)
        names = tuple(outputs)
        values = [
            types.MappingProxyType(row) if isinstance(output, TableVariable) else row[output._name]
            for row in rows
            for output in outputs.values()
        ]
        width = len(names)
        result_rows = tuple(ResultRow(tuple(values[i : i + width]), names) for i in range(0, len(values), width))
        return TableResultSet(result_rows, names)

    @staticmethod
    def _hydrate(connection: sqlalchemy.Connection, target: _Target, rows: list[dict[str, Any]]) -> None:
        """Add every declared list to ``rows``, with one child query per key batch per :class:`ListTable`."""
        for list_name, spec in target.lists.items():
            if isinstance(spec, DelimitedColumn):
                for row in rows:
                    text = row[spec.column]
                    row[list_name] = None if text is None else tuple(sorted(filter(None, text.split(spec.separator))))
                continue
            assert target.key is not None  # construction rejects lists without a key
            list_table, child = spec
            parent, value = child.c[list_table.parent_column], child.c[list_table.value_column]
            keys = list({row[target.key.name] for row in rows} - {None})
            members: dict[Any, list[Any]] = {}
            for start in range(0, len(keys), _HYDRATION_BATCH_SIZE):
                batch = keys[start : start + _HYDRATION_BATCH_SIZE]
                statement = (
                    sqlalchemy.select(parent, value)
                    .where(parent.in_(batch), value.is_not(None))
                    .order_by(parent, value)
                )
                for key, member in connection.execute(statement):
                    members.setdefault(key, []).append(member)
            for row in rows:
                row[list_name] = tuple(members.get(row[target.key.name], ()))
