"""Schema IR: resolve a storable dataclass into the relational schema that drives storage.

A *storable* class is a plain **frozen dataclass** declared with the stdlib-only
marker vocabulary from httk-core (:class:`~httk.core.storage.Indexed`,
:class:`~httk.core.storage.Unique`, :class:`~httk.core.storage.Skip`,
:class:`~httk.core.storage.markers.IdentitySkip`, :class:`~httk.core.storage.Shape`,
:class:`~httk.core.storage.StorageInfo`, :class:`~httk.core.storage.stored_property`).
:func:`resolve_schema` reads the class once — dataclass fields, ``Annotated``
markers, stored properties, and the optional class-level or externally
registered :class:`~httk.core.storage.StorageInfo` — and produces a
:class:`~httk.store.backend.schema.TableSchema`, the single source of truth from which the SQL layer
derives DDL, inserts, selects, and reconstruction alike.

Resolution rules (field annotation, then the resulting relational shape):

- ``int``/``str``/``bool``/``bytes`` — one scalar column named after the field
  (``bool`` is its own column kind, never folded into ``int``). ``float`` uses
  a query ``DOUBLE`` plus an exact ``*_exact`` hexadecimal text column so
  signed zero and every other finite binary64 value round-trip unchanged.
- ``X | None`` — the field is optional; all of its columns become nullable.
- a type with a registered :class:`~httk.store.backend.codecs.ValueCodec`
  (:class:`fractions.Fraction`, :class:`~httk.core.FracScalar`,
  :class:`~httk.core.SurdScalar`, :class:`datetime.datetime`, ...) — the
  codec's columns, named by appending each suffix to the field name.
- ``Annotated[FracVector, Shape(r, c)]`` with ``r >= 1`` — a fixed-shape tensor
  stored inline: ``r*c`` float columns ``name_0 .. name_{r*c-1}`` plus one
  ``name_exact`` text column holding the canonical exact tensor text.
- ``Annotated[FracVector, Shape(0, c)]`` — variable rows in a child table
  ``<parent>_<name>``, each row ``c`` float columns plus a ``name_exact`` text
  column with the same exact encoding per row.
- ``list[T]`` / homogeneous ``tuple[T, ...]`` — a child table: scalar or
  codec-typed elements store their columns per row; storable-dataclass elements
  store one ``name_sid`` foreign-key column per row.
- another storable frozen dataclass (optionally ``| None``) — a reference:
  one ``name_sid`` foreign-key column.
- ``Annotated[..., Skip()]`` — omitted from storage (the field must have a
  default so instances can be reconstructed without it).
- ``Annotated[..., IdentitySkip()]`` — stored normally but omitted from
  representation identity.
- ``Annotated[..., Related(...)]`` — relationship metadata carried on the
  resolved :class:`FieldSpec`; valid only on reference fields and on
  lists/tuples of storable classes.
- a :class:`~httk.core.storage.stored_property` — resolved like a field from its return
  annotation, flagged derived: stored and queryable, recomputed (not passed to
  ``__init__``) on reconstruction.

Optional child fields also receive a store-managed Boolean ``<field>_present``
parent column, preserving the distinction between ``None`` and an empty child
sequence.

The store layer additionally manages a ``sid`` integer primary key and a
``content_id`` text column on every table (and ``<parent>_sid`` /
``<name>_index`` columns on child tables); those never appear in
:attr:`TableSchema.fields`, and declaring a field named ``sid`` or
``content_id`` is an error.
"""

import collections.abc
import dataclasses
import re
import types
import typing
from typing import Annotated, Any, Final, Literal

from httk.core import FracVector
from httk.core.storage import (
    STORAGE_INFO_ATTRIBUTE,
    DedupPolicy,
    Indexed,
    Related,
    Shape,
    Skip,
    StorageInfo,
    StrongLink,
    Unique,
    WeakLink,
    stored_property,
)

from httk.store.backend.codecs import ScalarKind, codec_for, codec_named

__all__ = [
    "ChildTableSpec",
    "ColumnSpec",
    "FieldRole",
    "FieldSpec",
    "LinkSpec",
    "ScalarKind",
    "SchemaError",
    "TableSchema",
    "register_schema_override",
    "resolve_schema",
]

type FieldRole = Literal["scalar", "encoded", "fixed_array", "child", "reference"]
"""How a stored field maps onto the relational model."""


class SchemaError(Exception):
    """A class or field cannot be resolved into a storage schema; the message names both."""


@dataclasses.dataclass(frozen=True)
class ColumnSpec:
    """Describe one scalar column of a table.

    :param name: The column name.
    :param kind: The scalar storage kind.
    :param nullable: Whether the column accepts NULL.
    :param indexed: Whether a single-column index is requested.
    :param unique: Whether a unique index is requested.
    """

    name: str
    """The column name."""

    kind: ScalarKind
    """The scalar kind of the column."""

    nullable: bool = False
    """Whether the column accepts NULL (all columns of an optional field do)."""

    indexed: bool = False
    """Whether a single-column index is requested (:class:`~httk.core.storage.Indexed`)."""

    unique: bool = False
    """Whether a unique index is requested (:class:`~httk.core.storage.Unique`)."""


@dataclasses.dataclass(frozen=True)
class ChildTableSpec:
    """Describe the out-of-line child table backing a variable-length field.

    Only the per-element value columns are listed; the store layer adds the
    ``<parent>_sid`` foreign key and ``<name>_index`` ordering columns.

    :param table_name: The child table name.
    :param element_columns: The value columns of one element row.
    :param target: The storable element class for foreign-key rows, if any.
    """

    table_name: str
    """The child table name, ``<parent table>_<field>``."""

    element_columns: tuple[ColumnSpec, ...]
    """The value column(s) of one element row."""

    target: type | None = None
    """The storable element class when rows are foreign keys, else None."""


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """Describe the resolved storage shape of one stored field or property.

    :param field: The dataclass field or stored property name.
    :param python_type: The field value type after marker and optionality resolution.
    :param role: How the field maps onto the relational model.
    :param columns: The field columns in the parent table.
    :param codec_name: The codec name for the field or its child elements, if any.
    :param shape: The tensor shape marker, if any.
    :param child: The child table specification, if the field has a child role.
    :param target: The referenced storable class, if any.
    :param related: The relationship marker, if any.
    :param derived: Whether the value is a stored property recomputed on reconstruction.
    :param optional: Whether the annotation permits ``None``.
    """

    field: str
    """The dataclass field (or stored property) name."""

    python_type: Any
    """The field's value type with markers and optionality stripped."""

    role: FieldRole
    """How the field maps onto the relational model."""

    columns: tuple[ColumnSpec, ...] = ()
    """The field's columns in the parent table (empty for the child role)."""

    codec_name: str | None = None
    """The value codec encoding this field (or its child elements), if any."""

    shape: Shape | None = None
    """The :class:`~httk.core.storage.Shape` marker for tensor-valued fields, if any."""

    child: ChildTableSpec | None = None
    """The child table specification for the child role, else None."""

    target: type | None = None
    """The referenced storable class for reference (and child-of-storable) fields."""

    related: Related | None = None
    """The :class:`~httk.core.storage.Related` relationship marker of the field, if any.

    Only reference fields and child fields of storable elements can carry one;
    the marker's metadata flows into the relationships an entry provider emits
    for the field.
    """

    strong_link: StrongLink | None = None
    """The :class:`~httk.core.storage.StrongLink` edge-collection marker of the field, if any.

    Carried on a child field whose element class is an edge record with the
    string fields ``label``, ``entry_type``, and ``entry_id``. It is presentation
    metadata for the serving edge (which projects the field as forward and
    reverse OPTIMADE relationships); it is deliberately excluded from the schema
    fingerprint, exactly as :class:`~httk.core.storage.StrongLink` is excluded
    from content identity.
    """

    derived: bool = False
    """True for :class:`~httk.core.storage.stored_property` values: stored and queryable,
    recomputed rather than passed to ``__init__`` on reconstruction."""

    optional: bool = False
    """True when the annotation permits ``None``; child fields also use a managed presence column."""


@dataclasses.dataclass(frozen=True)
class LinkSpec:
    """Describe one resolved weak-link declaration of a storable class.

    A weak link is a store-managed, lineage-level association living in a
    dedicated append-only link table (never in a record field), declared as a
    ``WeakLink`` in :attr:`~httk.core.storage.StorageInfo.links`. Endpoints bind
    *lineages* and always resolve to the latest revision on either side.

    :param name: The link name; namespaces the link (e.g. ``record.links.<name>``).
    :param target: The storable class this link points at.
    :param exposed_relationship: Whether the link is served through the OPTIMADE relationship facility.
    :param role: The machine-readable relationship role, if any.
    :param description: The human-readable relationship description, if any.
    :param table_name: The dedicated link table name.
    """

    name: str
    """The link name, namespacing it under the source class's ``links``."""

    target: type
    """The storable class the link points at."""

    exposed_relationship: bool
    """Whether the link is served through the OPTIMADE relationship facility."""

    role: str | None
    """The machine-readable relationship role, if any."""

    description: str | None
    """The human-readable relationship description, if any."""

    table_name: str
    """The dedicated link table name, ``_httk_link_<source table>__<target table>__<name>``."""


@dataclasses.dataclass(frozen=True)
class TableSchema:
    """Describe the resolved relational schema of one storable class.

    :param cls: The storable dataclass resolved into this schema.
    :param table_name: The table name used for the class.
    :param fields: The stored fields and properties.
    :param composite_indexes: The resolved composite index declarations.
    :param dedup: The deduplication policy applied on save.
    :param links: The resolved weak-link declarations.
    """

    cls: type
    """The storable dataclass this schema was resolved from."""

    table_name: str
    """The table name (:attr:`~httk.core.storage.StorageInfo.storage_name` or the snake-cased class name)."""

    fields: tuple[FieldSpec, ...]
    """The stored fields, dataclass fields first (in declaration order), then stored properties.

    The store-managed ``sid`` and ``content_id`` columns are not fields and do
    not appear here.
    """

    composite_indexes: tuple[tuple[str, ...], ...]
    """The :attr:`~httk.core.storage.StorageInfo.indexes` declarations, resolved to column names."""

    dedup: DedupPolicy
    """The deduplication policy applied when instances are saved."""

    links: tuple[LinkSpec, ...] = ()
    """The class's resolved :attr:`~httk.core.storage.StorageInfo.links` weak-link declarations.

    Each declaration is resolved to a :class:`LinkSpec`: names are unique, valid
    identifiers that do not collide with field or reserved names, and each is
    assigned its dedicated link table name.
    """

    def field(self, name: str) -> FieldSpec:
        """Return the field specification named ``name``.

        :param name: The stored field or property name.
        :return: The matching field specification.
        :raises httk.store.backend.schema.SchemaError: If no stored field has that name.
        """
        for spec in self.fields:
            if spec.field == name:
                return spec
        raise SchemaError(f"{self.cls.__name__} has no stored field named {name!r}")

    def referenced_classes(self) -> tuple[type, ...]:
        """Return the distinct referenced storable classes in field order.

        :return: The referenced classes without duplicates.
        """
        seen: list[type] = []
        for spec in self.fields:
            if spec.target is not None and spec.target not in seen:
                seen.append(spec.target)
        return tuple(seen)


_RESERVED_FIELD_NAMES: Final = frozenset(
    {"sid", "content_id", "_httk_role", "store_timestamp", "logical_id", "alt_id", "alt_kind", "links"}
)

_SCALAR_KINDS: Final[dict[type, ScalarKind]] = {
    int: "int",
    str: "str",
    bool: "bool",
    bytes: "bytes",
}

_SNAKE_BOUNDARY_1: Final = re.compile(r"(.)([A-Z][a-z]+)")
_SNAKE_BOUNDARY_2: Final = re.compile(r"([a-z0-9])([A-Z])")

_schema_cache: dict[tuple[type, StorageInfo | None], TableSchema] = {}
_schema_overrides: dict[type, StorageInfo] = {}
_in_progress: set[type] = set()


def snake_case(name: str) -> str:
    """Convert a class name to its default acronym-aware table name.

    :param name: The class name to convert.
    :return: The lower-case snake-cased name.
    """
    return _SNAKE_BOUNDARY_2.sub(r"\1_\2", _SNAKE_BOUNDARY_1.sub(r"\1_\2", name)).lower()


def register_schema_override(cls: type, info: StorageInfo) -> None:
    """Register an external :class:`~httk.core.storage.StorageInfo` for a class that cannot declare one.

    The registered info is used by :func:`resolve_schema` whenever no explicit
    ``override`` argument is passed, and takes precedence over a
    ``__httk_storage__`` declaration on the class itself.

    :param cls: The storable class whose schema should be overridden.
    :param info: The external storage information to register.
    :return: None.
    """
    _schema_overrides[cls] = info


def resolve_schema(cls: type, *, override: StorageInfo | None = None) -> TableSchema:
    """Resolve (and cache) the :class:`~httk.store.backend.schema.TableSchema` of a storable dataclass.

    The effective :class:`~httk.core.storage.StorageInfo` is, in order of precedence:
    the explicit ``override`` argument, an info registered via
    :func:`register_schema_override`, the class's own ``__httk_storage__``
    attribute, or defaults. Results are cached per ``(class, effective
    override)``, so repeated calls return the same :class:`~httk.store.backend.schema.TableSchema` object.
    Reference cycles (a class referencing itself, or mutually referencing
    classes) are allowed and resolve without recursion loops.

    :param cls: The storable class to resolve.
    :param override: External storage information taking precedence over declarations.
    :return: The cached resolved table schema.
    :raises httk.store.backend.schema.SchemaError: If the class is not a frozen dataclass or one of its
        fields cannot be resolved; the diagnostic names the class and field.
    """
    row_base = getattr(cls, "__httk_row_base__", None)
    if row_base is not None:
        return resolve_schema(row_base, override=override)
    effective = override if override is not None else _schema_overrides.get(cls)
    key = (cls, effective)
    cached = _schema_cache.get(key)
    if cached is not None:
        return cached

    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise SchemaError(f"{getattr(cls, '__name__', cls)!r} is not a dataclass; storable classes are dataclasses")
    if not typing.cast(Any, cls).__dataclass_params__.frozen:
        raise SchemaError(
            f"{cls.__name__} is not a frozen dataclass; storable classes are immutable records "
            f"(declare it with @dataclass(frozen=True))"
        )

    _in_progress.add(cls)
    try:
        schema = _build_schema(cls, effective)
    finally:
        _in_progress.discard(cls)
    _schema_cache[key] = schema
    return schema


def _build_schema(cls: type, override: StorageInfo | None) -> TableSchema:
    info = override if override is not None else getattr(cls, STORAGE_INFO_ATTRIBUTE, None)
    if info is None:
        info = StorageInfo()
    table_name = info.storage_name if info.storage_name is not None else snake_case(cls.__name__)

    hints = typing.get_type_hints(cls, include_extras=True)
    specs: list[FieldSpec] = []
    for field in dataclasses.fields(cls):
        has_default = field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING
        spec = _resolve_field(cls, table_name, field.name, hints[field.name], derived=False, has_default=has_default)
        if spec is not None:
            specs.append(spec)

    field_names = {spec.field for spec in specs}
    for name, prop in _stored_properties(cls):
        if name in field_names:
            raise SchemaError(f"{cls.__name__}.{name} is both a dataclass field and a stored property")
        spec = _resolve_field(
            cls, table_name, name, _property_annotation(cls, name, prop), derived=True, has_default=True
        )
        if spec is not None:
            specs.append(spec)
            field_names.add(name)

    composite_indexes = _resolve_composite_indexes(cls, info, specs)
    declared_names = {field.name for field in dataclasses.fields(cls)}
    declared_names.update(name for name, _prop in _stored_properties(cls))
    for spec in specs:
        if spec.role == "child" and spec.optional and f"{spec.field}_present" in declared_names:
            raise SchemaError(
                f"{cls.__name__}.{spec.field}_present collides with the store-managed presence column "
                f"for optional child field {spec.field!r}"
            )
    links = _resolve_links(cls, table_name, info.links, specs)
    return TableSchema(
        cls=cls,
        table_name=table_name,
        fields=tuple(specs),
        composite_indexes=composite_indexes,
        dedup=info.dedup,
        links=links,
    )


def _storage_name_of(cls: type) -> str:
    """The target's table/collection storage name, derived without resolving its schema.

    This mirrors the storage-name derivation :func:`resolve_schema` applies to a
    class itself (registered override, then a ``__httk_storage__`` declaration,
    then the snake-cased class name), reading only ``storage_name`` so that
    self-referential and mutually-linked declarations cannot recurse. Deep
    target storability is validated at store-declaration time, not here.
    """
    row_base = getattr(cls, "__httk_row_base__", None)
    if row_base is not None:
        cls = row_base
    info = _schema_overrides.get(cls)
    if info is None:
        info = getattr(cls, STORAGE_INFO_ATTRIBUTE, None)
    if info is not None and info.storage_name is not None:
        return info.storage_name
    return snake_case(cls.__name__)


def _resolve_links(
    cls: type, table_name: str, links: tuple[WeakLink, ...], specs: list[FieldSpec]
) -> tuple[LinkSpec, ...]:
    """Resolve declared weak links: unique names not colliding with fields/reserved names, table names assigned."""
    field_names = {spec.field for spec in specs}
    seen: set[str] = set()
    resolved: list[LinkSpec] = []
    for link in links:
        name = link.name
        if name in seen:
            raise SchemaError(f"{cls.__name__}: weak-link name {name!r} is declared more than once")
        if name in field_names:
            raise SchemaError(f"{cls.__name__}: weak-link name {name!r} collides with a stored field name")
        if name in _RESERVED_FIELD_NAMES:
            raise SchemaError(f"{cls.__name__}: weak-link name {name!r} is reserved for store-managed use")
        if not isinstance(link.target, type) or not dataclasses.is_dataclass(link.target):
            raise SchemaError(f"{cls.__name__}: weak-link {name!r} target {link.target!r} is not a storable dataclass")
        seen.add(name)
        target_table = _storage_name_of(link.target)
        resolved.append(
            LinkSpec(
                name=name,
                target=link.target,
                exposed_relationship=link.exposed_relationship,
                role=link.role,
                description=link.description,
                table_name=f"_httk_link_{table_name}__{target_table}__{name}",
            )
        )
    return tuple(resolved)


def _stored_properties(cls: type) -> list[tuple[str, stored_property]]:
    """The class's stored properties in definition order, nearest override first per name."""
    found: dict[str, stored_property] = {}
    for klass in cls.__mro__:
        for name, attribute in vars(klass).items():
            if isinstance(attribute, stored_property) and name not in found:
                found[name] = attribute
    return list(found.items())


def _property_annotation(cls: type, name: str, prop: stored_property) -> Any:
    if prop.fget is None:
        raise SchemaError(f"{cls.__name__}.{name}: stored_property has no getter")
    hints = typing.get_type_hints(prop.fget, include_extras=True)
    if "return" not in hints:
        raise SchemaError(f"{cls.__name__}.{name}: stored_property getter needs a return annotation")
    return hints["return"]


def _resolve_field(
    cls: type, table_name: str, name: str, annotation: Any, *, derived: bool, has_default: bool
) -> FieldSpec | None:
    if name in _RESERVED_FIELD_NAMES:
        raise SchemaError(
            f"{cls.__name__}.{name}: the field names {sorted(_RESERVED_FIELD_NAMES)} are reserved for "
            f"store-managed columns"
        )

    base = annotation
    optional = False
    indexed = False
    unique = False
    shape: Shape | None = None
    skipped = False
    related: Related | None = None
    strong_link: StrongLink | None = None
    while True:
        origin = typing.get_origin(base)
        if origin is Annotated:
            arguments = typing.get_args(base)
            for marker in arguments[1:]:
                if isinstance(marker, Indexed):
                    indexed = True
                elif isinstance(marker, Unique):
                    unique = True
                elif isinstance(marker, Skip):
                    skipped = True
                elif isinstance(marker, Shape):
                    shape = marker
                elif isinstance(marker, Related):
                    related = marker
                elif isinstance(marker, StrongLink):
                    strong_link = marker
            base = arguments[0]
            continue
        if origin is typing.Union or origin is types.UnionType:
            arguments = typing.get_args(base)
            others = tuple(argument for argument in arguments if argument is not types.NoneType)
            if len(others) == len(arguments):
                raise SchemaError(f"{cls.__name__}.{name}: union types are not storable (only 'X | None')")
            if len(others) != 1:
                raise SchemaError(f"{cls.__name__}.{name}: only unions of one type with None are storable")
            optional = True
            base = others[0]
            continue
        break

    if skipped:
        if related is not None:
            raise SchemaError(f"{cls.__name__}.{name}: a Skip'd field is not stored and cannot carry a Related marker")
        if strong_link is not None:
            raise SchemaError(
                f"{cls.__name__}.{name}: a Skip'd field is not stored and cannot carry a StrongLink marker"
            )
        if not derived and not has_default:
            raise SchemaError(
                f"{cls.__name__}.{name}: a Skip'd field must have a default so instances can be "
                f"reconstructed without it"
            )
        return None

    spec = _resolve_unwrapped_field(cls, table_name, name, base, shape, optional, indexed, unique, derived)
    if related is not None:
        if spec.role != "reference" and not (spec.role == "child" and spec.target is not None):
            raise SchemaError(
                f"{cls.__name__}.{name}: a Related marker applies only to a storable-class reference "
                f"field or a list/tuple of storable classes, not to a {spec.role!r} field"
            )
        spec = dataclasses.replace(spec, related=related)
    if strong_link is not None:
        _validate_strong_link(cls, name, spec)
        spec = dataclasses.replace(spec, strong_link=strong_link)
    return spec


_STRONG_LINK_EDGE_FIELDS: Final = ("label", "entry_type", "entry_id")


def _validate_strong_link(cls: type, name: str, spec: FieldSpec) -> None:
    """Validate that a StrongLink-annotated field is a tuple of edge records.

    :param cls: The declaring storable class (named in error messages).
    :param name: The field name (named in error messages).
    :param spec: The resolved field spec of the annotated field.
    :raises SchemaError: If the field is not a child field of a storable element
        class exposing the string edge fields ``label``, ``entry_type``, and
        ``entry_id``.
    """
    element = spec.target
    if spec.role != "child" or element is None:
        raise SchemaError(
            f"{cls.__name__}.{name}: a StrongLink marker applies only to a list/tuple of edge records "
            f"(a storable element class), not to a {spec.role!r} field"
        )
    element_schema = resolve_schema(element)
    for edge_field in _STRONG_LINK_EDGE_FIELDS:
        try:
            edge_spec = element_schema.field(edge_field)
        except SchemaError:
            edge_spec = None
        if edge_spec is None or edge_spec.role != "scalar" or edge_spec.python_type is not str:
            raise SchemaError(
                f"{cls.__name__}.{name}: a StrongLink edge element class ({element.__name__}) must declare "
                f"string fields {list(_STRONG_LINK_EDGE_FIELDS)}; {edge_field!r} is missing or not a string"
            )


def _resolve_unwrapped_field(
    cls: type,
    table_name: str,
    name: str,
    base: Any,
    shape: Shape | None,
    optional: bool,
    indexed: bool,
    unique: bool,
    derived: bool,
) -> FieldSpec:
    if shape is not None:
        return _resolve_shape_field(cls, table_name, name, base, shape, optional, indexed, unique, derived)

    kind = _SCALAR_KINDS.get(base)
    if kind is not None:
        column = ColumnSpec(name, kind, nullable=optional, indexed=indexed, unique=unique)
        return FieldSpec(
            field=name, python_type=base, role="scalar", columns=(column,), derived=derived, optional=optional
        )

    codec = codec_for(base)
    if codec is not None:
        columns = tuple(
            ColumnSpec(
                f"{name}{suffix}",
                column_kind,
                nullable=optional,
                indexed=indexed and suffix == codec.query_suffix,
                unique=unique and suffix == codec.query_suffix,
            )
            for suffix, column_kind in codec.columns
        )
        return FieldSpec(
            field=name,
            python_type=base,
            role="encoded",
            columns=columns,
            codec_name=codec.name,
            derived=derived,
            optional=optional,
        )

    origin = typing.get_origin(base)
    if origin is list or origin is tuple:
        return _resolve_sequence_field(cls, table_name, name, base, origin, optional, derived)

    if isinstance(base, type) and issubclass(base, FracVector):
        raise SchemaError(
            f"{cls.__name__}.{name}: a FracVector field needs a Shape marker, e.g. Annotated[FracVector, Shape(3, 3)]"
        )

    if isinstance(base, type) and dataclasses.is_dataclass(base):
        _validate_target(cls, base)
        column = ColumnSpec(f"{name}_sid", "int", nullable=optional, indexed=indexed, unique=unique)
        return FieldSpec(
            field=name,
            python_type=base,
            role="reference",
            columns=(column,),
            target=base,
            derived=derived,
            optional=optional,
        )

    # A source-model dataclass may also expose a Mapping interface (for
    # example OptimadeResource).  Its concrete frozen-dataclass storage shape
    # is still the authoritative one; only non-dataclass mappings remain
    # unsupported values.
    if _is_mapping_type(base, origin):
        raise SchemaError(f"{cls.__name__}.{name}: mapping-typed fields are not storable (yet)")

    raise SchemaError(
        f"{cls.__name__}.{name}: type {base!r} is not storable; expected a scalar "
        f"(int/float/str/bool/bytes), a codec type, a Shape-annotated FracVector, a list/tuple, "
        f"or a storable frozen dataclass"
    )


def _resolve_shape_field(
    cls: type,
    table_name: str,
    name: str,
    base: Any,
    shape: Shape,
    optional: bool,
    indexed: bool,
    unique: bool,
    derived: bool,
) -> FieldSpec:
    if base is not FracVector:
        raise SchemaError(f"{cls.__name__}.{name}: Shape markers apply to FracVector fields only, got {base!r}")
    if shape.rows >= 1:
        size = shape.rows * shape.cols
        columns = tuple(
            ColumnSpec(f"{name}_{i}", "float", nullable=optional, indexed=indexed, unique=unique) for i in range(size)
        ) + (ColumnSpec(f"{name}_exact", "str", nullable=optional, indexed=indexed, unique=unique),)
        return FieldSpec(
            field=name,
            python_type=base,
            role="fixed_array",
            columns=columns,
            shape=shape,
            derived=derived,
            optional=optional,
        )
    element_columns = tuple(ColumnSpec(f"{name}_{i}", "float") for i in range(shape.cols)) + (
        ColumnSpec(f"{name}_exact", "str"),
    )
    child = ChildTableSpec(table_name=f"{table_name}_{name}", element_columns=element_columns)
    return FieldSpec(
        field=name, python_type=base, role="child", shape=shape, child=child, derived=derived, optional=optional
    )


def _resolve_sequence_field(
    cls: type, table_name: str, name: str, base: Any, origin: type, optional: bool, derived: bool
) -> FieldSpec:
    arguments = typing.get_args(base)
    if origin is tuple:
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise SchemaError(f"{cls.__name__}.{name}: only homogeneous variable tuples (tuple[T, ...]) are storable")
        element_type = arguments[0]
    else:
        if len(arguments) != 1:
            raise SchemaError(f"{cls.__name__}.{name}: a storable list needs exactly one element type")
        element_type = arguments[0]

    codec_name: str | None = None
    target: type | None = None
    kind = _SCALAR_KINDS.get(element_type)
    if kind is not None:
        element_columns: tuple[ColumnSpec, ...] = (ColumnSpec(name, kind),)
    else:
        codec = codec_for(element_type)
        if codec is not None:
            codec_name = codec.name
            element_columns = tuple(ColumnSpec(f"{name}{suffix}", column_kind) for suffix, column_kind in codec.columns)
        elif isinstance(element_type, type) and dataclasses.is_dataclass(element_type):
            _validate_target(cls, element_type)
            target = element_type
            element_columns = (ColumnSpec(f"{name}_sid", "int"),)
        else:
            raise SchemaError(
                f"{cls.__name__}.{name}: element type {element_type!r} is not storable; expected a "
                f"scalar, a codec type, or a storable frozen dataclass"
            )
    child = ChildTableSpec(table_name=f"{table_name}_{name}", element_columns=element_columns, target=target)
    return FieldSpec(
        field=name,
        python_type=base,
        role="child",
        codec_name=codec_name,
        child=child,
        target=target,
        derived=derived,
        optional=optional,
    )


def _is_mapping_type(base: Any, origin: Any) -> bool:
    candidate = origin if origin is not None else base
    return isinstance(candidate, type) and issubclass(candidate, collections.abc.Mapping)


def _validate_target(cls: type, target: type) -> None:
    """Resolve a referenced storable class, tolerating reference cycles."""
    if target is cls or target in _in_progress:
        return
    resolve_schema(target)


def _resolve_composite_indexes(cls: type, info: StorageInfo, specs: list[FieldSpec]) -> tuple[tuple[str, ...], ...]:
    by_name = {spec.field: spec for spec in specs}
    resolved: list[tuple[str, ...]] = []
    for index in info.indexes:
        columns: list[str] = []
        for field_name in index:
            spec = by_name.get(field_name)
            if spec is None:
                raise SchemaError(f"{cls.__name__}: composite index names unknown field {field_name!r}")
            if spec.role == "scalar" or spec.role == "reference":
                columns.append(spec.columns[0].name)
            elif spec.role == "encoded":
                assert spec.codec_name is not None
                columns.append(f"{spec.field}{codec_named(spec.codec_name).query_suffix}")
            else:
                raise SchemaError(
                    f"{cls.__name__}: composite index field {field_name!r} has role {spec.role!r}; only "
                    f"scalar, encoded, and reference fields can be composite-indexed"
                )
        resolved.append(tuple(columns))
    return tuple(resolved)
