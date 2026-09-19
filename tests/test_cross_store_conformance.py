"""Cross-store conformance checks for the portable OPTIMADE query surface.

The remote client is deliberately exercised through a real in-process ASGI
application, rather than Starlette's ``TestClient``: its blocking portal is
not reliable in this workspace.  ``AsgiSyncClient`` is a tiny synchronous
bridge over ``httpx2.ASGITransport`` and rejects every non-testserver URL, so
these tests cannot contact the network.
"""

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Annotated, Any, ClassVar, cast
from urllib.parse import quote, urlsplit

import httpx2
import pytest
from httk.atomistic import OptimadeStructure, Species, StructureEntryProvider, UnitcellStructure
from httk.core import (
    EntryProvider,
    EntryTypeDefinition,
    RelatedEntry,
    Run,
    RunEdge,
    RunEntry,
    load_entry_type_definition,
)
from httk.core.data_records import RECORDS_DEFINITION_ID
from httk.core.optimade import OptimadeResource
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, Related, StorageInfo, StoredPropertyProjection, Unique
from httk.serve.optimade import adapter_from_providers, adapter_from_stores, create_asgi_app
from httk.serve.optimade.backend.memory_store import InMemoryStore

from httk.store import Backend, EntryIdScheme, RunEntryProvider, SqlStore
from httk.store.backend.sql import StoredEntrySource
from httk.store.backend.sql.rows import is_lazy_row
from httk.store.optimade import OptimadeStore


class AsgiSyncClient:
    """Minimal synchronous httpx2 client adapter with no network escape hatch."""

    def __init__(self, app: Any, *, base_url: str) -> None:
        self.app = app
        self.base_url = base_url
        self.requests: list[str] = []
        self.closed = False

    def get(self, url: str) -> httpx2.Response:
        assert urlsplit(url).netloc == urlsplit(self.base_url).netloc
        self.requests.append(url)

        async def request() -> httpx2.Response:
            transport = httpx2.ASGITransport(app=self.app)
            async with httpx2.AsyncClient(transport=transport) as client:
                return await client.get(url)

        return asyncio.run(request())

    def close(self) -> None:
        self.closed = True


class _FlatStructureProvider(EntryProvider):
    """A standard (not extended) structure endpoint for exact describedby."""

    _definition = load_entry_type_definition("https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures")
    _keys: ClassVar[dict[str, str]] = {
        "id": "id",
        "type": "type",
        "immutable_id": "immutable_id",
        "last_modified": "last_modified",
        "elements": "elements",
        "nelements": "nelements",
        "elements_ratios": "elements_ratios",
        "chemical_formula_descriptive": "chemical_formula_descriptive",
        "chemical_formula_reduced": "chemical_formula_reduced",
        "chemical_formula_anonymous": "chemical_formula_anonymous",
        "nperiodic_dimensions": "nperiodic_dimensions",
        "nsites": "nsites",
        "structure_features": "structure_features",
    }

    def __init__(self) -> None:
        base = {
            "type": "structures",
            "immutable_id": "source",
            "last_modified": "2026-07-30T12:00:00+00:00",
            "nperiodic_dimensions": 3,
            "structure_features": [],
        }
        self._rows = (
            base
            | {
                "id": "nacl",
                "elements": ["Cl", "Na"],
                "nelements": 2,
                "elements_ratios": [0.5, 0.5],
                "chemical_formula_descriptive": "ClNa",
                "chemical_formula_reduced": "ClNa",
                "chemical_formula_anonymous": "AB",
                "nsites": 2,
            },
            base
            | {
                "id": "si",
                "elements": ["Si"],
                "nelements": 1,
                "elements_ratios": [1.0],
                "chemical_formula_descriptive": "Si",
                "chemical_formula_reduced": "Si",
                "chemical_formula_anonymous": "A",
                "nsites": 1,
            },
            base
            | {
                "id": "unknown",
                "elements": None,
                "nelements": None,
                "elements_ratios": None,
                "chemical_formula_descriptive": None,
                "chemical_formula_reduced": None,
                "chemical_formula_anonymous": None,
                "nsites": None,
            },
        )

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"structures": self._definition}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        assert entry_type == "structures"
        return self._keys

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        assert entry_type == "structures"
        return self._rows

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        assert entry_type == "structures"
        return {"nacl": (RelatedEntry("references", "r1"),)}


class _FlatReferenceProvider(EntryProvider):
    _definition = load_entry_type_definition("https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/references")

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"references": self._definition}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        assert entry_type == "references"
        return {"id": "id", "type": "type", "title": "title"}

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        assert entry_type == "references"
        return ({"id": "r1", "type": "references", "title": "Source"},)


def _provider() -> _FlatStructureProvider:
    return _FlatStructureProvider()


def _adapter(provider: _FlatStructureProvider):
    return adapter_from_providers(
        [provider, _FlatReferenceProvider()],
        sortable={"structures": ("id", "nelements")},
    )


def _extended_atomistic_provider() -> StructureEntryProvider:
    sodium = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    return StructureEntryProvider(
        {
            "sodium": UnitcellStructure(
                [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                [[0, 0, 0]],
                [sodium],
                ["Na"],
            )
        }
    )


@dataclass
class StoreTarget:
    name: str
    store: object
    target: object


def _ids(
    target: StoreTarget,
    build: Callable[[Any], Any],
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[str]:
    """Run one identical field-level query against one store implementation."""

    searcher = target.store.searcher()  # type: ignore[union-attr]
    variable = searcher.variable(target.target)
    searcher.add(build(variable))
    searcher.add_sort(variable.nelements, descending=True)
    searcher.add_sort(variable.id, descending=False)
    if offset:
        searcher.add_offset(offset)
    if limit is not None:
        searcher.set_limit(limit)
    results = searcher.results(record=variable, identifier=variable.id, formula=variable.chemical_formula_reduced)
    rows = list(results)
    for row in rows:
        assert row.names == ("record", "identifier", "formula")
        assert row.identifier == row["identifier"] == row[1]
        assert row.formula == row["formula"] == row[2]
    return [row.identifier for row in rows]


def _portable_expression(variable: Any) -> Any:
    """Use boolean, literal string/list, membership, and negation operations."""

    nacl = (
        (variable.nelements == 2)
        & variable.chemical_formula_descriptive.contains("Na")
        & variable.chemical_formula_descriptive.endswith("Na")
        & variable.elements.has("Na")
        & variable.elements.has_any("Cl")
        & variable.elements.has_only("Cl", "Na")
        & variable.id.is_in("nacl", "si")
    )
    silicon = (
        (variable.nelements == 1)
        & variable.chemical_formula_descriptive.startswith("Si")
        & variable.id.is_in("nacl", "si")
    )
    return (nacl | silicon) & ~(variable.id == "unknown")


def _all_remote_backends(store: OptimadeStore) -> list[Any]:
    searcher = store.searcher()
    variable = searcher.variable(store.entry_type("structures"))
    return [row.record for row in searcher.results(record=variable)]


def _database(dialect: str) -> Backend:
    if dialect == "duckdb":
        pytest.importorskip("duckdb_engine")
        return Backend.duckdb()
    assert dialect == "sqlite"
    return Backend.sqlite()


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_same_portable_structure_query_across_memory_sql_and_real_asgi_remote(dialect: str) -> None:
    """The normal portable profile has identical results through every store.

    It also verifies the client observes the server's filtered
    ``meta.data_returned`` count, independent of its page's length.
    """

    provider = _provider()
    adapter = _adapter(provider)
    base_url = "http://testserver"
    app = create_asgi_app(adapter, baseurl=base_url)
    client = AsgiSyncClient(app, base_url=base_url)
    remote = OptimadeStore(base_url, client=client, page_limit=1)

    # The memory store gets the same public record values the provider serves.
    memory_rows = [dict(record) for record in provider.records("structures")]
    memory = InMemoryStore({"structures": memory_rows})

    remote_backends = _all_remote_backends(remote)
    # A remote result is a thin bound subclass carrying a .links accessor
    # (see remote_query._bound_class), never the plain OptimadeStructure
    # class itself, though isinstance and equality against it still hold.
    assert all(isinstance(backend, OptimadeStructure) for backend in remote_backends)

    with _database(dialect) as database:
        sql = SqlStore(database, entry_records={})
        for backend in remote_backends:
            sql.save(backend)

        targets = (
            StoreTarget("memory", memory, "structures"),
            StoreTarget(dialect, sql, type(remote_backends[0])),
            StoreTarget("remote", remote, type(remote_backends[0])),
        )
        for target in targets:
            assert _ids(target, _portable_expression) == ["nacl", "si"], target.name
            assert _ids(target, lambda value: value.chemical_formula_reduced != None) == ["nacl", "si"], target.name
            assert _ids(target, lambda value: value.chemical_formula_reduced == None) == ["unknown"], target.name
            assert set(_ids(target, lambda value: value.chemical_formula_reduced.is_in(None, "Si"))) == {
                "unknown",
                "si",
            }, target.name
            assert set(_ids(target, lambda value: ~value.chemical_formula_reduced.is_in(None, "Si"))) == {"nacl"}, (
                target.name
            )
            assert _ids(target, _portable_expression, limit=1, offset=1) == ["si"], target.name

        # count is unpaged; len has the plan's offset/limit applied.
        remote_searcher = remote.searcher()
        remote_variable = remote_searcher.variable(type(remote_backends[0]))
        remote_searcher.add(_portable_expression(remote_variable))
        remote_searcher.add_offset(1)
        remote_searcher.set_limit(1)
        remote_results = remote_searcher.results(record=remote_variable)
        assert remote_searcher.count() == 2
        assert len(remote_results) == 1
        assert remote_results.first() is not None
        assert remote_results.one().record.id == "si"
        assert [cast(Any, record).id for record in remote_results.scalars("record")] == ["si"]
        assert [cast(Any, record).id for record in remote_results[0:1].scalars("record")] == ["si"]

        # A direct SearchResult retains declared values/names in every backend.
        for target in targets:
            searcher = target.store.searcher()  # type: ignore[union-attr]
            variable = searcher.variable(target.target)
            searcher.add(variable.id == "nacl")
            searcher._output(variable, "record")
            searcher._output(variable.id, "identifier")
            (result,) = list(searcher._matches())
            assert result.names == ("record", "identifier"), target.name
            assert result.values[1] == "nacl", target.name

        # The same multi-output shape, through the public results() form.
        for target in targets:
            searcher = target.store.searcher()  # type: ignore[union-attr]
            variable = searcher.variable(target.target)
            searcher.add(variable.id == "nacl")
            rows = list(searcher.results(record=variable, identifier=variable.id))
            assert rows[0].names == ("record", "identifier"), target.name
            assert [row.identifier for row in rows] == ["nacl"], target.name

    assert client.requests
    assert all(urlsplit(url).netloc == "testserver" for url in client.requests)
    assert not client.closed


def test_remote_typed_and_generic_resources_copy_to_sqlite_without_losing_source() -> None:
    """A remote typed backend and its generic source remain exact offline."""

    provider = _provider()
    base_url = "http://testserver"
    adapter = _adapter(provider)
    client = AsgiSyncClient(create_asgi_app(adapter, baseurl=base_url), base_url=base_url)
    remote = OptimadeStore(base_url, client=client)
    backends = _all_remote_backends(remote)
    nacl = next(value for value in backends if value.id == "nacl")
    assert nacl.elements_ratios == (Fraction(1, 2), Fraction(1, 2))
    assert nacl.resource.document.text
    assert nacl.resource.schema.info_document.text

    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        typed_sid = store.save(nacl)
        generic_sid = store.save(nacl.resource)
        # The lazy default returns a row subclass; only eager restores exact
        # type.  A fresh store on the same database gives a cold-cache proxy
        # (the live saved nacl otherwise wins the main store's cache slot).
        lazy_typed = SqlStore(database).fetch(type(nacl), typed_sid)
        assert is_lazy_row(lazy_typed) and lazy_typed.elements_ratios == nacl.elements_ratios
        typed = store.fetch(type(nacl), typed_sid, eager=True)
        generic = store.fetch(OptimadeResource, generic_sid)

        # nacl's own type is a thin bound-with-.links subclass (see
        # remote_query._bound_class); an eager fetch always reconstructs the
        # plain storage class, never that subclass, though they compare equal.
        assert type(typed) is OptimadeStructure
        assert typed == nacl
        assert typed.resource.document == nacl.resource.document
        assert typed.resource.schema == nacl.resource.schema
        assert typed.elements_ratios == nacl.elements_ratios
        assert generic == nacl.resource
        assert generic["relationships"] == nacl.resource.unwrap()["relationships"]
        assert generic.unwrap() == nacl.resource.unwrap()

        searcher = store.searcher()
        variable = searcher.variable(OptimadeResource)
        searcher.add((variable.id == "nacl") & (variable.type == "structures"))
        assert searcher.results(record=variable).one().record == nacl.resource


def test_extended_own_atomistic_structure_endpoint_is_typed_by_known_property_iris() -> None:
    """Ownerless extension IRIs do not contradict the standard structure binding."""

    base_url = "http://testserver"
    app = create_asgi_app(adapter_from_providers([_extended_atomistic_provider()]), baseurl=base_url)
    client = AsgiSyncClient(app, base_url=base_url)
    remote = OptimadeStore(base_url, client=client)

    descriptor = remote.entry_type("structures")
    assert descriptor.backend.__name__ == "OptimadeStructure"
    searcher = remote.searcher()
    variable = searcher.variable(descriptor.backend)
    row = searcher.results(record=variable, elements=variable.elements).one()
    assert row.record.id == "sodium"
    assert row.elements == ("Na",)


@dataclass(frozen=True)
class ConfRecordRow:
    """A minimal ``records`` backing for runs-relationship cross-store parity."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="conf_run_record")

    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class ConfRecordFamily:
    """The records family targeted by the conformance run's edges."""

    type = "records"
    definition_id = RECORDS_DEFINITION_ID


register_entry_family(
    name="conf-run-records", family=f"{__name__}:ConfRecordFamily", definition_id=RECORDS_DEFINITION_ID
)
register_entry_record(name="conf-run-records-rec", family="conf-run-records", record=f"{__name__}:ConfRecordRow")


class _FlatRecordsProvider(EntryProvider):
    """A minimal in-memory ``_httk_records`` provider (edge-target for reverse)."""

    _definition = load_entry_type_definition(RECORDS_DEFINITION_ID).served_form()

    def __init__(self, ids: Iterable[str]) -> None:
        self._ids = list(ids)

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"_httk_records": self._definition}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        assert entry_type == "_httk_records"
        return {"id": "id", "type": "type"}

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        assert entry_type == "_httk_records"
        return [{"id": record_id, "type": "_httk_records"} for record_id in self._ids]


def _served_blocks(app: Any, path: str, entry_id: str) -> dict[str, list[tuple[str, str]]]:
    """Return one served resource's relationship blocks as {key: [(type, id), ...]}."""
    client = AsgiSyncClient(app, base_url="http://testserver")
    payload = client.get(f"http://testserver{path}").json()
    resource = next(item for item in payload["data"] if item["id"] == entry_id)
    return {
        key: [(d["type"], d["id"]) for d in value["data"]] for key, value in resource.get("relationships", {}).items()
    }


def test_runs_relationships_forward_and_reverse_served_parity_across_stores() -> None:
    """Forward and reverse edge blocks are byte-identical through the SQL and in-memory routes.

    Both routes serve the identical run over the same ids and are queried through
    the served OPTIMADE/ASGI edge (not the provider hooks directly): the run's
    forward ``_httk_has_*`` blocks and the record's derived reverse ``_httk_is_*``
    blocks match, and the in-memory reverse hook actually reaches the wire.
    """
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={RunEntry: Run, ConfRecordFamily: ConfRecordRow},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        rec = store.fetch(ConfRecordRow, store.save(ConfRecordRow("r")), eager=True).id
        run_obj = Run(
            inputs=(RunEdge("in", "records", rec),),
            artifacts=(RunEdge("art", "records", rec),),
            source_id="ws:job",
        )
        run_id = store.fetch(Run, store.save(run_obj), eager=True).id

        sql_app = create_asgi_app(
            adapter_from_stores(
                (StoredEntrySource(store, RunEntry, "runs"), StoredEntrySource(store, ConfRecordFamily, "recs")),
            ),
            baseurl="http://testserver",
        )
        memory_app = create_asgi_app(
            adapter_from_providers([RunEntryProvider({run_id: run_obj}), _FlatRecordsProvider([rec])]),
            baseurl="http://testserver",
        )

        expected_forward = {
            "_httk_has_input": [("_httk_records", rec)],
            "_httk_has_artifact": [("_httk_records", rec)],
        }
        expected_reverse = {
            "_httk_is_input": [("_httk_runs", run_id)],
            "_httk_is_artifact": [("_httk_runs", run_id)],
        }
        for app in (sql_app, memory_app):
            run_blocks = _served_blocks(app, "/_httk_runs", run_id)
            assert {k: run_blocks[k] for k in expected_forward} == expected_forward
            record_blocks = _served_blocks(app, "/_httk_records", rec)
            assert {k: record_blocks[k] for k in expected_reverse} == expected_reverse

        # The `_httk_relationships.<semantic key>.id` filter extension gives the
        # SAME exact result set through both routes for keys served on both.
        for app in (sql_app, memory_app):
            assert _filtered_ids(app, "/_httk_runs", f'_httk_relationships._httk_has_input.id HAS "{rec}"') == [run_id]
            assert _filtered_ids(app, "/_httk_runs", f'_httk_relationships._httk_has_artifact.id HAS "{rec}"') == [
                run_id
            ]
            assert _filtered_ids(app, "/_httk_records", f'_httk_relationships._httk_is_input.id HAS "{run_id}"') == [
                rec
            ]
            assert _filtered_ids(app, "/_httk_records", f'_httk_relationships._httk_is_artifact.id HAS "{run_id}"') == [
                rec
            ]
            # A run that took no such input is filtered out on both routes.
            assert _filtered_ids(app, "/_httk_runs", '_httk_relationships._httk_has_input.id HAS "nonesuch"') == []


def _filtered_ids(app: Any, path: str, filter_string: str) -> list[str]:
    """Return the sorted data ids of a served filter request through the ASGI edge."""
    client = AsgiSyncClient(app, base_url="http://testserver")
    payload = client.get(f"http://testserver{path}?filter={quote(filter_string)}").json()
    return sorted(item["id"] for item in payload["data"])


# --- Related reference-field relationships: stored vs in-memory parity ---------

_REFERENCES = "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/references"


@dataclass(frozen=True)
class ConfPeerRow:
    """A ``references`` backing with a queryable ``doi`` (the relationship target)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="conf_related_peer")

    doi: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)

    __httk_stored_properties__: ClassVar = {
        "doi": StoredPropertyProjection(
            response=lambda record: record.doi,
            query=lambda context, operator, literal: context.compare(
                context.field("doi"), operator, context.constant(literal)
            ),
            sort=lambda context: context.field("doi"),
        )
    }


@dataclass(frozen=True)
class ConfWorkRow:
    """A ``_httk_records`` backing with one Related reference field to a peer."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="conf_related_work")

    name: str
    lead: Annotated[ConfPeerRow | None, Related(role="lead", description="Lead reference")] = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class ConfPeerFamily:
    type = "references"
    definition_id = _REFERENCES


class ConfWorkFamily:
    type = "records"
    definition_id = RECORDS_DEFINITION_ID


register_entry_family(name="conf-related-peer", family=f"{__name__}:ConfPeerFamily", definition_id=_REFERENCES)
register_entry_record(name="conf-related-peer-rec", family="conf-related-peer", record=f"{__name__}:ConfPeerRow")
register_entry_family(
    name="conf-related-work", family=f"{__name__}:ConfWorkFamily", definition_id=RECORDS_DEFINITION_ID
)
register_entry_record(name="conf-related-work-rec", family="conf-related-work", record=f"{__name__}:ConfWorkRow")


class _RelatedBlockProvider(EntryProvider):
    """An in-memory ``_httk_records`` + ``references`` provider mirroring the stored fixture.

    Serves the identical works, peers, reference-field relationship blocks, and
    ``doi`` property values, so the provider (in-memory) route is the byte-for-byte
    reference for the stored federation's Related reference-field serving and its
    depth-1 related-property filtering.
    """

    _work_def = load_entry_type_definition(RECORDS_DEFINITION_ID).served_form()
    _peer_def = load_entry_type_definition(_REFERENCES).served_form()

    def __init__(
        self,
        works: list[str],
        peers: list[tuple[str, str]],
        blocks: dict[str, tuple[tuple[str, str, str], ...]],
    ) -> None:
        self._works = works
        self._peers = peers
        self._blocks = blocks

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"_httk_records": self._work_def, "references": self._peer_def}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        if entry_type == "_httk_records":
            return {"id": "id", "type": "type"}
        return {"id": "id", "type": "type", "doi": "doi"}

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type == "_httk_records":
            return [{"id": work, "type": "_httk_records"} for work in self._works]
        return [{"id": peer, "type": "references", "doi": doi} for peer, doi in self._peers]

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        if entry_type != "_httk_records":
            return {}
        return {
            work: tuple(
                RelatedEntry("references", peer, role=role, description=description)
                for peer, role, description in entries
            )
            for work, entries in self._blocks.items()
        }


def test_related_reference_field_blocks_and_depth1_filter_parity() -> None:
    """Stored and in-memory routes agree on Related reference-field blocks and depth-1 filters."""
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={ConfWorkFamily: ConfWorkRow, ConfPeerFamily: ConfPeerRow},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        ada = store.fetch(ConfPeerRow, store.save(ConfPeerRow("10.1/ada")), eager=True)
        boole = store.fetch(ConfPeerRow, store.save(ConfPeerRow("10.2/boole")), eager=True)
        work_a = store.fetch(ConfWorkRow, store.save(ConfWorkRow("A", lead=ada)), eager=True)
        work_b = store.fetch(ConfWorkRow, store.save(ConfWorkRow("B", lead=boole)), eager=True)
        work_c = store.fetch(ConfWorkRow, store.save(ConfWorkRow("C")), eager=True)  # a work with no reference

        stored_app = create_asgi_app(
            adapter_from_stores(
                (StoredEntrySource(store, ConfWorkFamily, "work"), StoredEntrySource(store, ConfPeerFamily, "peer")),
            ),
            baseurl="http://testserver",
        )
        memory_app = create_asgi_app(
            adapter_from_providers(
                [
                    _RelatedBlockProvider(
                        works=[work_a.id, work_b.id, work_c.id],
                        peers=[(ada.id, "10.1/ada"), (boole.id, "10.2/boole")],
                        blocks={
                            work_a.id: ((ada.id, "lead", "Lead reference"),),
                            work_b.id: ((boole.id, "lead", "Lead reference"),),
                        },
                    )
                ]
            ),
            baseurl="http://testserver",
        )

        for app in (stored_app, memory_app):
            # Identical served relationship block on the referencing work.
            assert _served_blocks(app, "/_httk_records", work_a.id) == {"references": [("references", ada.id)]}, app
            # Identical depth-1 related-property filter results.
            assert _filtered_ids(app, "/_httk_records", 'references.doi CONTAINS "10.1"') == [work_a.id], app
            assert _filtered_ids(app, "/_httk_records", 'references.doi CONTAINS "10.2"') == [work_b.id], app
            assert _filtered_ids(app, "/_httk_records", 'references.doi CONTAINS "nomatch"') == [], app
            # The reference-less work_c matches the NOT complement on both routes.
            assert _filtered_ids(app, "/_httk_records", 'NOT (references.doi CONTAINS "10.1")') == sorted(
                [work_b.id, work_c.id]
            ), app
