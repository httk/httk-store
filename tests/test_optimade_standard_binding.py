"""No-network tests for standard-namespace schema completion in the OPTIMADE client.

The client binds a standard endpoint (such as ``structures``) to its typed
backend, and completes the identities of the endpoint's unprefixed standard
property names, from the specification version the service declares in
``/info`` -- even when the service publishes no property-definition ``$id``,
as every large public provider does. These tests drive that behaviour entirely
through the synchronous client against captured provider ``/info/structures``
documents, touching no network.
"""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from httk.core import load_entry_type_definition
from httk.core.optimade import OptimadeResource
from httk.core.register import optimade_entry_binding

from httk.store import UnsupportedQueryError
from httk.store.optimade import OptimadeStore

STRUCTURES = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
REFERENCES = "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/references"
FILES = "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/files"
CORE_ID = "https://schemas.optimade.org/defs/v1.2/properties/core/id"
CORE_TYPE = "https://schemas.optimade.org/defs/v1.2/properties/core/type"
FILE_URL = "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/url"
REFERENCE_ADDRESS = "https://schemas.optimade.org/defs/v1.2/properties/optimade/references/address"

_FIXTURES = Path(__file__).parent / "data" / "optimade_info"
_BASE = "https://example.test/v1"


def _require_structures_table() -> None:
    """Skip unless the installed httk-atomistic declares the structures version table.

    Standard-name completion of ``structures`` is gated by the
    ``standard_property_versions`` table that *httk-atomistic* registers on its
    binding (httk-atomistic >= 2.1.1); an older release leaves the endpoint
    generic, so these tests cannot run against it.
    """
    pytest.importorskip("httk.atomistic")
    binding = optimade_entry_binding("https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures")
    if binding is None or not binding.standard_property_versions:
        pytest.skip("installed httk-atomistic declares no standard_property_versions for structures")


def _structures_iri(name: str) -> str:
    """Return the vendored structures-definition IRI of one standard property."""

    return load_entry_type_definition(STRUCTURES).properties[name].definition_id


@dataclass
class FakeResponse:
    status_code: int
    text: str


class FakeClient:
    """A borrowed HTTP client answering only from a fixed URL map, no network.

    A request URL is matched exactly, then by its path (query string dropped),
    so a self-constructed ``/structures?...`` query resolves to the endpoint's
    prepared page regardless of the exact parameter ordering.
    """

    def __init__(self, responses: dict[str, list[FakeResponse]]) -> None:
        self.responses = {url: list(items) for url, items in responses.items()}
        self.requests: list[str] = []
        self.closed = False

    def get(self, url: str) -> FakeResponse:
        self.requests.append(url)
        for key in (url, url.split("?", 1)[0]):
            queue = self.responses.get(key)
            if queue:
                return queue.pop(0)
        raise AssertionError(f"unexpected external request: {url}")

    def close(self) -> None:
        self.closed = True


def _response(value: object) -> FakeResponse:
    return FakeResponse(200, json.dumps(value))


def _info(endpoint_names: list[str], api_version: str | None) -> dict[str, object]:
    attributes: dict[str, object] = {"available_endpoints": endpoint_names}
    if api_version is not None:
        attributes["api_version"] = api_version
    return {"data": {"type": "info", "attributes": attributes}}


def _entry(properties: dict[str, object], *, describedby: str | None = None) -> dict[str, object]:
    document: dict[str, object] = {"data": {"type": "info", "properties": properties}}
    if describedby is not None:
        document["links"] = {"describedby": describedby}
    return document


def _property(definition_id: str | None = None, *, ptype: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {}
    if definition_id is not None:
        value["$id"] = definition_id
    if ptype is not None:
        value["type"] = ptype
    return value


def _fixture_document(fixture_name: str, *, api_version: str | None = None) -> dict[str, object]:
    """Return a captured ``/info/structures`` fixture prepared for serving.

    The captured bodies trimmed the resource-identity ``type`` member that a
    1.2 service publishes, so it is restored here; ``api_version`` overrides the
    declared version for the version-gate test without touching the read-only
    fixture on disk.
    """

    document = json.loads((_FIXTURES / fixture_name).read_text())
    data = dict(document["data"])
    data["type"] = "info"
    document["data"] = data
    if api_version is not None:
        meta = dict(document.get("meta") or {})
        meta["api_version"] = api_version
        document["meta"] = meta
    return document


def _build_store(
    entries: dict[str, dict[str, object]],
    *,
    api_version: str | None,
    infer: bool = True,
    pages: dict[str, list[FakeResponse]] | None = None,
) -> tuple[OptimadeStore, FakeClient]:
    responses: dict[str, list[FakeResponse]] = {_BASE + "/info": [_response(_info(list(entries), api_version))]}
    for name, document in entries.items():
        # Standard-name completion reads the declared version from the entry
        # info document's own ``meta.api_version``, so mirror the service's
        # declared version there when the fixture/synthetic body has none.
        if api_version is not None and "meta" not in document:
            document = {**document, "meta": {"api_version": api_version}}
        responses[_BASE + "/info/" + name] = [_response(document)]
    for name, queue in (pages or {}).items():
        responses[_BASE + "/" + name] = list(queue)
    client = FakeClient(responses)
    return OptimadeStore(_BASE, client=client, infer_standard_definitions=infer), client


def _unprefixed(fixture_name: str) -> tuple[str, ...]:
    document = _fixture_document(fixture_name)
    properties = document["data"]["properties"]  # type: ignore[index,call-overload]
    return tuple(sorted(name for name in properties if not name.startswith("_")))


# --- Sub-package 1: discovery completion and binding -----------------------


def test_alexandria_structures_bind_by_standard_name_without_declared_ids() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    store, _client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
    )
    descriptor = store.entry_type("structures")

    assert descriptor.backend is OptimadeStructure
    assert descriptor.binding_evidence == "standard-name"
    assert descriptor.inferred_properties == _unprefixed("alexandria_pbe_info_structures.json")
    assert len(descriptor.inferred_properties) == 20
    # The three provider-prefixed names are advertised but never name-completed.
    for prefixed in ("_alexandria_band_gap", "_alexandria_band_gap_direct", "_alexandria_charges"):
        assert prefixed in descriptor.advertised_properties
        assert prefixed not in descriptor.inferred_properties
    species_iri = _structures_iri("species")
    assert descriptor.property_iris["species"] == species_iri
    assert descriptor.property_names[species_iri] == "species"


@pytest.mark.parametrize("fixture", ["materials_project_info_structures.json", "oqmd_info_structures.json"])
def test_v12_structures_infer_space_group_names(fixture: str) -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    store, _client = _build_store({"structures": _fixture_document(fixture)}, api_version="1.2.0")
    descriptor = store.entry_type("structures")

    assert descriptor.backend is OptimadeStructure
    assert descriptor.binding_evidence == "standard-name"
    assert descriptor.inferred_properties == _unprefixed(fixture)
    space_group_names = (
        "space_group_it_number",
        "space_group_symbol_hall",
        "space_group_symbol_hermann_mauguin",
        "space_group_symbol_hermann_mauguin_extended",
        "space_group_symmetry_operations_xyz",
    )
    for name in space_group_names:
        assert name in descriptor.inferred_properties


def test_version_gate_drops_later_names_through_the_client() -> None:
    _require_structures_table()

    store, _client = _build_store(
        {"structures": _fixture_document("oqmd_info_structures.json", api_version="1.1.0")},
        api_version="1.1.0",
    )
    descriptor = store.entry_type("structures")

    # ``space_group_*`` were introduced at 1.2; a 1.1 service declares no
    # standard meaning for them, so they stay unknown while 1.0 names remain.
    assert "elements" in descriptor.inferred_properties
    for name in (
        "space_group_it_number",
        "space_group_symbol_hall",
        "space_group_symbol_hermann_mauguin",
        "space_group_symbol_hermann_mauguin_extended",
        "space_group_symmetry_operations_xyz",
    ):
        assert name not in descriptor.inferred_properties


def test_inference_disabled_reproduces_generic_resource_behaviour() -> None:
    store, _client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
        infer=False,
    )
    descriptor = store.entry_type("structures")

    assert descriptor.backend is OptimadeResource
    assert descriptor.binding is None
    assert descriptor.binding_evidence is None
    assert descriptor.inferred_properties == ()
    assert dict(descriptor.property_iris) == {}


def test_declared_describedby_path_and_property_iris_unchanged() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    elements_iri = _structures_iri("elements")
    store, _client = _build_store(
        {"structures": _entry({"elements": _property(elements_iri)}, describedby=STRUCTURES)},
        api_version=None,
    )
    descriptor = store.entry_type("structures")

    assert descriptor.backend is OptimadeStructure
    assert descriptor.binding is not None
    assert descriptor.binding.definition_id == STRUCTURES
    assert descriptor.binding_evidence == "declared"
    assert dict(descriptor.property_iris) == {"elements": elements_iri}
    assert descriptor.inferred_properties == ()


def test_declared_property_ids_path_and_property_iris_unchanged() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    elements_iri = _structures_iri("elements")
    # A vendor endpoint name isolates the property-IRI path from the
    # standard-name path; the elements IRI is owned only by structures.
    store, _client = _build_store(
        {"structures-vendor": _entry({"elements": _property(elements_iri)})},
        api_version=None,
    )
    descriptor = store.entry_type("structures-vendor")

    assert descriptor.backend is OptimadeStructure
    assert descriptor.binding_evidence == "property-ids"
    assert dict(descriptor.property_iris) == {"elements": elements_iri}
    assert descriptor.inferred_properties == ()


def test_declared_id_beats_inference_per_property() -> None:
    _require_structures_table()

    nelements_iri = _structures_iri("nelements")
    elements_iri = _structures_iri("elements")
    store, _client = _build_store(
        {
            "structures": _entry(
                {
                    "nelements": _property(nelements_iri, ptype="integer"),
                    "elements": _property(ptype="list"),
                }
            )
        },
        api_version="1.2.0",
    )
    descriptor = store.entry_type("structures")

    # ``nelements`` carries a declared identity, so it is never inferred;
    # ``elements`` lacks one and is completed from the declared version.
    assert descriptor.property_iris["nelements"] == nelements_iri
    assert "nelements" not in descriptor.inferred_properties
    assert descriptor.property_iris["elements"] == elements_iri
    assert "elements" in descriptor.inferred_properties


def test_provider_prefixed_endpoint_name_never_binds_by_name() -> None:
    store, _client = _build_store(
        {"_exmpl_things": _entry({"nelements": _property(ptype="integer"), "elements": _property(ptype="list")})},
        api_version="1.2.0",
    )
    descriptor = store.entry_type("_exmpl_things")

    assert descriptor.backend is OptimadeResource
    assert descriptor.binding is None
    assert descriptor.binding_evidence is None
    assert descriptor.inferred_properties == ()


# --- Sub-package 2: query layer over a name-completed endpoint -------------

_NACL = {
    "id": "nacl",
    "type": "structures",
    "attributes": {
        "last_modified": "2026-07-30T12:00:00Z",
        "elements": ["Cl", "Na"],
        "nelements": 2,
        "elements_ratios": [0.5, 0.5],
        "chemical_formula_reduced": "ClNa",
        "chemical_formula_descriptive": "ClNa",
        "dimension_types": [1, 1, 1],
        "nperiodic_dimensions": 3,
        "lattice_vectors": [[0.0, 2.8, 2.8], [2.8, 0.0, 2.8], [2.8, 2.8, 0.0]],
        "cartesian_site_positions": [[0.0, 0.0, 0.0], [2.8, 2.8, 2.8]],
        "nsites": 2,
        "species": [
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
        ],
        "species_at_sites": ["Na", "Cl"],
        "structure_features": [],
        "_alexandria_band_gap": 1.25,
    },
}


def _nacl_page() -> FakeResponse:
    return _response({"data": [_NACL], "meta": {"more_data_available": False, "data_returned": 1}})


def test_query_layer_accepts_typed_standard_and_prefixed_fields() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    store, _client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
        pages={"structures": [_nacl_page()]},
    )
    searcher = store.searcher()
    variable = searcher.variable(OptimadeStructure)

    typed = (variable.nelements == 2) & variable.elements.has("Na") & (variable.chemical_formula_reduced == "ClNa")
    assert "nelements = 2" in typed.text
    assert 'elements HAS "Na"' in typed.text
    assert 'chemical_formula_reduced = "ClNa"' in typed.text
    # A provider-prefixed field stays queryable under its exact wire name.
    assert variable._alexandria_band_gap._remote_name == "_alexandria_band_gap"


def test_variable_resolves_by_backend_class_only_when_bound() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    inferred_store, _inferred_client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
    )
    assert inferred_store.searcher().variable(OptimadeStructure) is not None

    generic_store, _generic_client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
        infer=False,
    )
    with pytest.raises(UnsupportedQueryError):
        generic_store.searcher().variable(OptimadeStructure)


def test_typed_row_projects_prefixed_field_raw_and_standard_field_decoded() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    store, _client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
        pages={"structures": [_nacl_page()]},
    )
    searcher = store.searcher()
    variable = searcher.variable(OptimadeStructure)
    row = searcher.results(gap=variable._alexandria_band_gap, nsites=variable.nsites).one()

    # The schema-unknown provider field returns the raw served JSON value; the
    # identified standard field returns its decoded typed value.
    assert float(row.gap) == 1.25
    assert row.nsites == 2
    assert isinstance(row.nsites, int)


def test_generic_row_projection_of_prefixed_field_unchanged_when_inference_disabled() -> None:
    store, _client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
        infer=False,
        pages={"structures": [_nacl_page()]},
    )
    descriptor = store.entry_type("structures")
    assert descriptor.backend is OptimadeResource

    searcher = store.searcher()
    variable = searcher.variable(descriptor)
    row = searcher.results(gap=variable._alexandria_band_gap, nsites=variable.nsites).one()

    # The generic path is unchanged: every advertised field is read raw.
    assert float(row.gap) == 1.25
    assert row.nsites == 2


# --- Sub-package 3: end-to-end conversion ----------------------------------


def test_end_to_end_structure_conversion_and_timestamp() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure, UnitcellStructureView

    store, _client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
        pages={"structures": [_nacl_page()]},
    )
    searcher = store.searcher()
    variable = searcher.variable(OptimadeStructure)
    row = searcher.results(structure=variable).one()

    assert UnitcellStructureView(row.structure).formula == "ClNa"
    last_modified = row.structure.last_modified
    assert isinstance(last_modified, datetime.datetime)
    assert last_modified.tzinfo is not None
    assert last_modified.utcoffset() is not None


# --- Attribute access on typed rows (provider extensions never withdrawn) ---


def test_typed_row_attribute_access_exposes_extension_and_keeps_typed_field() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    store, _client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
        pages={"structures": [_nacl_page()]},
    )
    searcher = store.searcher()
    variable = searcher.variable(OptimadeStructure)
    row = searcher.results(structure=variable).one()

    # A provider extension is reachable as a raw attribute by exact wire name.
    assert float(row.structure._alexandria_band_gap) == 1.25
    # A decoded typed property still wins through normal attribute lookup.
    assert row.structure.nsites == 2
    assert isinstance(row.structure.nsites, int)
    # A name the endpoint does not advertise at all is a plain missing attribute.
    with pytest.raises(AttributeError):
        _ = row.structure.not_advertised


def test_slicer_iteration_reads_extension_and_id_on_typed_rows() -> None:
    _require_structures_table()

    store, _client = _build_store(
        {"structures": _fixture_document("alexandria_pbe_info_structures.json")},
        api_version="1.1.0",
        pages={"structures": [_nacl_page()]},
    )
    structures = store.slicer("structures")
    entry = next(iter(structures[structures["nsites"] == 2]))

    assert entry.id == "nacl"
    assert float(entry._alexandria_band_gap) == 1.25


def test_reference_bound_row_keeps_typed_field_and_exposes_extension() -> None:
    from httk.core.optimade import OptimadeReference

    reference = {
        "id": "ref-1",
        "type": "references",
        "attributes": {"last_modified": "2026-07-30T12:00:00Z", "_exmpl_note": "see figure 2"},
    }
    page = _response({"data": [reference], "meta": {"more_data_available": False, "data_returned": 1}})
    store, _client = _build_store(
        {
            "references": _entry(
                {
                    "id": _property(ptype="string"),
                    "type": _property(ptype="string"),
                    "last_modified": _property(ptype="timestamp"),
                    "_exmpl_note": _property(ptype="string"),
                }
            )
        },
        api_version="1.2.0",
        pages={"references": [page]},
    )
    descriptor = store.entry_type("references")
    assert descriptor.backend is OptimadeReference
    assert descriptor.binding_evidence == "standard-name"

    searcher = store.searcher()
    variable = searcher.variable(descriptor)
    row = searcher.results(reference=variable).one()

    # A name-completed standard field decodes to its typed value ...
    assert isinstance(row.reference.last_modified, datetime.datetime)
    # ... while the advertised provider extension is exposed raw by wire name.
    assert row.reference._exmpl_note == "see figure 2"


# --- Review findings: shadowing, contradiction, foreign describedby --------


def test_unidentified_advertised_name_never_shadows_a_renamed_standard_field() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    from httk.store.optimade.remote_query import RemoteSearcher

    species_iri = _structures_iri("species")
    # The service maps the standard ``species`` definition to a vendor wire name
    # and also advertises a same-spelled unprefixed ``species`` without a $id.
    store, _client = _build_store(
        {
            "structures": _entry(
                {
                    "renamed_species": _property(species_iri, ptype="list"),
                    "species": _property(ptype="list"),
                    "nelements": _property(_structures_iri("nelements"), ptype="integer"),
                }
            )
        },
        api_version="1.2.0",
    )
    descriptor = store.entry_type("structures")
    assert descriptor.backend is OptimadeStructure

    _query_fields, all_fields, _kinds, _capabilities = RemoteSearcher._typed_maps(descriptor)
    # The semantic ``species`` field keeps the declared wire name; the
    # same-spelled unidentified advertised field never overwrites it.
    assert all_fields["species"] == "renamed_species"
    # ``species`` is not a portable query field, so the unidentified advertised
    # copy is not resurrected as a spurious non-semantic query field either.
    assert "species" not in _query_fields
    with pytest.raises(UnsupportedQueryError):
        _ = store.searcher().variable(descriptor).species


def test_contradictory_declared_property_evidence_stays_generic() -> None:
    # Mutually exclusive declared IRIs (files-only and references-only) are a
    # contradiction, so a versioned standard endpoint name does not rescue it.
    store, _client = _build_store(
        {
            "structures": _entry(
                {
                    "fileish": _property(FILE_URL, ptype="string"),
                    "referenceish": _property(REFERENCE_ADDRESS, ptype="string"),
                }
            )
        },
        api_version="1.2.0",
    )
    descriptor = store.entry_type("structures")

    assert descriptor.backend is OptimadeResource
    assert descriptor.binding is None
    assert descriptor.binding_evidence is None


def test_consistent_ambiguous_evidence_falls_through_to_name_tier() -> None:
    _require_structures_table()
    from httk.atomistic import OptimadeStructure

    # Only a universal IRI (``id``) is declared: the candidate set is never
    # narrowed, so the endpoint is ambiguous-but-consistent and the name tier
    # binds it to the standard entry type of its name and declared version.
    store, _client = _build_store(
        {"structures": _entry({"id": _property(CORE_ID, ptype="string")})},
        api_version="1.2.0",
    )
    descriptor = store.entry_type("structures")

    assert descriptor.backend is OptimadeStructure
    assert descriptor.binding_evidence == "standard-name"


def _files_store(version: str, *, complete: bool) -> tuple[OptimadeStore, FakeClient]:
    attributes: dict[str, object] = {
        "last_modified": "2026-07-30T12:00:00Z",
        "url": "http://host.test/f",
        "size": 3,
        "_x_note": "n",
    }
    if complete:
        attributes["name"] = "f.txt"
    properties = {
        "id": _property(ptype="string"),
        "type": _property(ptype="string"),
        "last_modified": _property(ptype="timestamp"),
        "url": _property(ptype="string"),
        "size": _property(ptype="integer"),
        "name": _property(ptype="string"),
        "_x_note": _property(ptype="string"),
    }
    page = _response({"data": [{"id": "f1", "type": "files", "attributes": attributes}], "meta": {"data_returned": 1}})
    return _build_store({"files": _entry(properties)}, api_version=version, pages={"files": [page]})


def test_version_gated_standard_names_are_never_exposed_raw_on_a_typed_row() -> None:
    from httk.core.optimade import OptimadeFile

    # Every files property is introduced at 1.2, so a 1.1.0 files service binds
    # by name yet infers nothing; ``url``/``size`` carry no standard meaning and
    # must not leak a raw value under their standard spelling.
    store, _client = _files_store("1.1.0", complete=False)
    descriptor = store.entry_type("files")
    assert descriptor.backend is OptimadeFile
    assert descriptor.binding_evidence == "standard-name"
    assert descriptor.inferred_properties == ()

    searcher = store.searcher()
    variable = searcher.variable(descriptor)
    row = searcher.results(file=variable).one()

    with pytest.raises(AttributeError):
        _ = row.file.url
    with pytest.raises(AttributeError):
        _ = row.file.size
    # A genuine provider extension is still readable raw.
    assert row.file._x_note == "n"


def test_recognised_standard_names_resolve_typed_not_raw_on_a_typed_row() -> None:
    from httk.core.optimade import FileView, OptimadeFile

    # At 1.2.0 the same names are recognised (identified), so they are not raw
    # extensions on the row; their decoded values are reached through the
    # canonical view, and the provider extension remains a raw attribute.
    store, _client = _files_store("1.2.0", complete=True)
    descriptor = store.entry_type("files")
    assert descriptor.backend is OptimadeFile
    assert "url" in descriptor.inferred_properties
    assert "size" in descriptor.inferred_properties

    searcher = store.searcher()
    variable = searcher.variable(descriptor)
    row = searcher.results(file=variable).one()

    with pytest.raises(AttributeError):
        _ = row.file.url  # not a raw extension, and not a backend attribute
    assert FileView(row.file).url == "http://host.test/f"
    assert FileView(row.file).size == 3
    assert row.file._x_note == "n"


def test_unknown_declared_describedby_blocks_name_binding() -> None:
    # A declared but unrecognised links.describedby is a positive foreign claim:
    # the endpoint stays unbound even with a standard name and declared version.
    store, _client = _build_store(
        {
            "structures": _entry(
                {"nelements": _property(ptype="integer")},
                describedby="https://unknown.test/entrytypes/structures",
            )
        },
        api_version="1.2.0",
    )
    descriptor = store.entry_type("structures")

    assert descriptor.backend is OptimadeResource
    assert descriptor.binding is None
    assert descriptor.binding_evidence is None
