"""Coverage for provenance providers and their SQL storage backings."""

import datetime
import json
from dataclasses import replace

import pytest
import sqlalchemy
from httk.core import (
    DataRecord,
    DataRecordEntry,
    ProductLink,
    PropertyDefinition,
    RelatedEntry,
    Run,
    RunEdge,
    RunEntry,
    load_entry_type_definition,
)
from httk.core.provenance import RUNS_DEFINITION_ID
from httk.core.storage import content_id

from httk.store import DataRecordEntryProvider, EntryIdScheme, RunEntryProvider, product_relationships, validate_record
from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql import (
    Backend,
    EntryMetadataConflictError,
    SqlStore,
    StoredPropertySqlConfigurationError,
    stored_property_sql_plan,
)

UTC = datetime.UTC
ENERGY_ID = "https://schemas.example.org/properties/energy"
FORCE_ID = "https://schemas.example.org/properties/force"


def _definition(name: str, definition_id: str, *, required_response: bool = False) -> PropertyDefinition:
    return PropertyDefinition.from_simple(
        name,
        description=f"The {name} value.",
        fulltype="float",
        definition_id=definition_id,
        required_response=required_response,
    )


def _run() -> Run:
    return Run(
        # Run edges carry internal entry-type names; the provider translates
        # ``records`` -> ``_httk_records`` at the serving edge (standard names
        # such as ``structures``/``files`` pass through unchanged).
        inputs=(RunEdge("in-1", "records", "record-1"), RunEdge("in-2", "structures", "structure-1")),
        artifacts=(RunEdge("artifact", "files", "file-1"),),
        outputs=(RunEdge("out", "records", "record-2"),),
        source_id="ws:job",
        last_modified=datetime.datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
    )


def test_run_provider_serves_rows_and_relationships() -> None:
    run = _run()
    provider = RunEntryProvider({"run-key": run, "run-map": {"workflow_declaration_uri": None}})
    definition = provider.entry_types()['_httk_runs']
    assert set(definition.properties) == {
        "id",
        "type",
        "immutable_id",
        "_httk_source_id",
        "last_modified",
        "_httk_workflow_declaration_uri",
    }
    assert "workflow_declaration_uri" not in definition.properties

    rows = list(provider.records("_httk_runs"))
    assert json.dumps(rows)
    assert rows[0]["__id"] == "run-key"
    assert rows[0]["workflow_declaration_uri"] is None
    assert rows[0]["source_id"] == "ws:job"
    assert definition.properties["_httk_source_id"].nullable
    assert definition.properties["_httk_source_id"].requirements["query-support"] == "all mandatory"
    assert definition.properties["_httk_source_id"].requirements["sortable"] is True
    assert provider.relationships("_httk_runs")["run-key"] == (
        RelatedEntry("_httk_records", "record-1", role="input", label="in-1", relationship="_httk_has_input"),
        RelatedEntry("structures", "structure-1", role="input", label="in-2", relationship="_httk_has_input"),
        RelatedEntry("files", "file-1", role="artifact", label="artifact", relationship="_httk_has_artifact"),
        RelatedEntry("_httk_records", "record-2", role="output", label="out", relationship="_httk_has_output"),
    )
    # The reverse view is target-keyed (wire target type -> raw id -> runs) and
    # names this run under each field's wire reverse key.
    reverse = provider.reverse_relationships()
    assert reverse["_httk_records"]["record-1"] == (
        RelatedEntry("_httk_runs", "run-key", role="input", label="in-1", relationship="_httk_is_input"),
    )
    assert reverse["structures"]["structure-1"] == (
        RelatedEntry("_httk_runs", "run-key", role="input", label="in-2", relationship="_httk_is_input"),
    )
    assert reverse["files"]["file-1"] == (
        RelatedEntry("_httk_runs", "run-key", role="artifact", label="artifact", relationship="_httk_is_artifact"),
    )
    assert reverse["_httk_records"]["record-2"] == (
        RelatedEntry("_httk_runs", "run-key", role="output", label="out", relationship="_httk_is_output"),
    )
    # The empty-run entry contributes no reverse identifiers anywhere.
    assert all("run-map" not in {entry.id for entries in ids.values() for entry in entries} for ids in reverse.values())


def test_run_provider_rows_validate() -> None:
    provider = RunEntryProvider({"run-key": _run()})
    definition = provider.entry_types()['_httk_runs']
    keys = provider.property_keys("_httk_runs")
    for row in provider.records("_httk_runs"):
        validate_record(definition, {name: row[key] for name, key in keys.items()})


def test_data_record_provider_union_and_validation() -> None:
    definitions = {
        "_httk_energy": _definition("_httk_energy", ENERGY_ID),
        "_httk_force": _definition("_httk_force", FORCE_ID),
    }
    provider = DataRecordEntryProvider(
        {
            "energy-1": DataRecord.from_value(ENERGY_ID, "_httk_energy", 3.5),
            "force-1": DataRecord.from_value(FORCE_ID, "_httk_force", 2.0),
        },
        definitions=definitions,
    )
    keys = provider.property_keys("_httk_records")
    rows = list(provider.records("_httk_records"))
    assert rows[0]["_httk_energy"] == 3.5 and rows[0]["_httk_force"] is None
    assert rows[1]["_httk_energy"] is None and rows[1]["_httk_force"] == 2.0
    assert set(provider.entry_types()["_httk_records"].properties) == set(keys)
    for row in rows:
        validate_record(provider.entry_types()["_httk_records"], {name: row[key] for name, key in keys.items()})


def test_data_record_provider_fails_eagerly_on_definitions() -> None:
    record = DataRecord.from_value("https://unknown.example/energy", "_httk_energy", 1)
    with pytest.raises(ValueError, match="record 'r'.*_httk_energy.*https://unknown.example/energy"):
        DataRecordEntryProvider({"r": record})

    with pytest.raises(ValueError, match="record 'r'.*definition IRI"):
        DataRecordEntryProvider({"r": record}, definitions={"_httk_energy": _definition("_httk_energy", ENERGY_ID)})


def test_data_record_provider_rejects_non_nullable_union_nulls() -> None:
    energy = DataRecord.from_value(ENERGY_ID, "_httk_energy", 1.0)
    force = DataRecord.from_value(FORCE_ID, "_httk_force", 2.0)
    with pytest.raises(ValueError, match="_httk_energy.*force"):
        DataRecordEntryProvider(
            {"energy": energy, "force": force},
            definitions={
                "_httk_energy": _definition("_httk_energy", ENERGY_ID, required_response=True),
                "_httk_force": _definition("_httk_force", FORCE_ID),
            },
        )
    provider = DataRecordEntryProvider(
        {"energy-1": energy, "energy-2": DataRecord.from_value(ENERGY_ID, "_httk_energy", 3.0)},
        definitions={"_httk_energy": _definition("_httk_energy", ENERGY_ID, required_response=True)},
    )
    assert len(list(provider.records("_httk_records"))) == 2


def test_product_relationships_groups_in_input_order_and_checks_labels() -> None:
    links = [
        ProductLink("runs", "r-1", "records", "d-1", "energy"),
        ProductLink("runs", "r-1", "records", "d-2", "force"),
        ProductLink("runs", "r-2", "records", "d-3", "energy"),
    ]
    result = product_relationships(links)
    assert [entry.id for entry in result["runs"]["r-1"]] == ["d-1", "d-2"]
    assert result["runs"]["r-1"][0].role == "product"
    assert result["runs"]["r-2"][0].label == "energy"
    with pytest.raises(ValueError, match="runs.*r-1.*energy"):
        product_relationships([links[0], ProductLink("runs", "r-1", "records", "d-4", "energy")])


def test_data_record_provider_serves_product_relationships() -> None:
    records = {
        "record-1": DataRecord.from_value(ENERGY_ID, "_httk_energy", 1.0),
        "record-2": DataRecord.from_value(FORCE_ID, "_httk_force", 2.0),
    }
    links = product_relationships([ProductLink("_httk_records", "record-1", "_httk_records", "record-2", "derived")])
    provider = DataRecordEntryProvider(
        records,
        definitions={
            "_httk_energy": _definition("_httk_energy", ENERGY_ID),
            "_httk_force": _definition("_httk_force", FORCE_ID),
        },
        relationships=links["_httk_records"],
    )
    assert provider.relationships("_httk_records") == {
        "record-1": (
            RelatedEntry(
                "_httk_records", "record-2", role="product", label="derived", relationship="_httk_has_product"
            ),
        )
    }


def test_sql_store_round_trips_provenance_records_and_stored_number() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={RunEntry: Run, DataRecordEntry: DataRecord},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        run = _run()
        store.save(run)
        fetched = store.fetch_entry(RunEntry, content_id(run))
        assert fetched == replace(run, id="httk.test-1-1", immutable_id="httk.test-1-1~1")
        assert fetched.immutable_id == "httk.test-1-1~1" and fetched.last_modified == run.last_modified
        assert fetched.source_id == "ws:job"
        assert fetched.inputs == run.inputs

        conflicting = Run(
            inputs=run.inputs,
            artifacts=run.artifacts,
            outputs=run.outputs,
            source_id=run.source_id,
            immutable_id="different",
            last_modified=run.last_modified,
        )
        with pytest.raises(EntryMetadataConflictError):
            store.save(conflicting)

        record = DataRecord.from_value(ENERGY_ID, "_httk_energy", 3.5)
        store.save(record)
        assert store.fetch_entry(DataRecordEntry, content_id(record)).value == 3.5
        table_name = resolve_schema(DataRecord).table_name
        columns = {column["name"] for column in sqlalchemy.inspect(database.engine).get_columns(table_name)}
        assert "value_number" in columns
        searcher = store.searcher()
        variable = searcher.variable(DataRecord)
        searcher.add(variable.value_number == 3.5)
        searcher.output(variable, "record")
        assert [row[0][0].value for row in searcher] == [3.5]


def _plan_records(searchers) -> list:
    return [result[0][0] for searcher in searchers for result in searcher]


def test_run_stored_property_plan_serves_prefixed_properties() -> None:
    served = load_entry_type_definition(RUNS_DEFINITION_ID).served_form()
    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={RunEntry: Run},
            entry_ids=EntryIdScheme("httk.test", "1"),
        )
        store.save(Run(source_id="ws:a", workflow_declaration_uri="https://wf.example/a"))
        store.save(Run(source_id="ws:b"))

        plan = stored_property_sql_plan(store, RunEntry, served=served)

        # (a) rows serve the two properties' values (not null), under wire names.
        rows = {row["_httk_source_id"]: row for row in plan.records()}
        assert set(rows) == {"ws:a", "ws:b"}
        assert all(row["type"] == "_httk_runs" for row in rows.values())
        assert rows["ws:a"]["_httk_workflow_declaration_uri"] == "https://wf.example/a"
        assert rows["ws:b"]["_httk_workflow_declaration_uri"] is None

        # (b) a filter over _httk_source_id returns the right subset.
        filtered = _plan_records(plan.filter_searchers('_httk_source_id = "ws:a"'))
        assert [record.source_id for record in filtered] == ["ws:a"]

        # (c) a sort over _httk_source_id orders correctly.
        ordered = _plan_records(plan.filter_searchers('type = "_httk_runs"', sort=(("_httk_source_id", False),)))
        assert [record.source_id for record in ordered] == ["ws:a", "ws:b"]

        # (d) planning this prefixed family WITHOUT served= fails loudly: the
        # bare internal definition sees the served projection keys as unknown.
        with pytest.raises(StoredPropertySqlConfigurationError, match="_httk_source_id"):
            stored_property_sql_plan(store, RunEntry)


def test_product_link_storage_deduplicates_by_value() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        link = ProductLink("runs", "r-1", "records", "d-1", "energy")
        sid = store.save(link)
        assert store.save(link) == sid
        assert store.fetch(ProductLink, sid) == link


def test_run_provider_serves_through_optimade() -> None:
    pytest.importorskip("httk.serve.optimade")
    from httk.serve.optimade import adapter_from_providers, create_asgi_app
    from starlette.testclient import TestClient

    definition = _definition("_httk_energy", ENERGY_ID)
    record = DataRecord.from_value(ENERGY_ID, "_httk_energy", 3.5)
    product = DataRecord.from_value(FORCE_ID, "_httk_force", 2.0)
    run = Run(source_id="ws:job", inputs=(RunEdge("labeled-input", "records", "record-1"),))
    product_links = product_relationships(
        [ProductLink("_httk_records", "record-1", "_httk_records", "record-2", "derived")]
    )
    adapter = adapter_from_providers(
        [
            RunEntryProvider({"run-1": run, "run-2": Run(source_id="other-job")}),
            DataRecordEntryProvider(
                {"record-1": record, "record-2": product},
                definitions={"_httk_energy": definition, "_httk_force": _definition("_httk_force", FORCE_ID)},
                relationships=product_links["_httk_records"],
            ),
        ]
    )
    with TestClient(create_asgi_app(adapter, baseurl="http://testserver/")) as client:
        run_response = client.get("/_httk_runs")
        filtered_run_response = client.get("/_httk_runs", params={"filter": '_httk_source_id = "ws:job"'})
        records_response = client.get("/_httk_records")
    assert run_response.status_code == records_response.status_code == 200
    assert filtered_run_response.status_code == 200
    assert [item["id"] for item in filtered_run_response.json()["data"]] == ["run-1"]
    assert filtered_run_response.json()["data"][0]["attributes"]["_httk_source_id"] == "ws:job"
    run_relation = run_response.json()["data"][0]["relationships"]["_httk_has_input"]["data"][0]
    assert run_relation["type"] == "_httk_records"
    assert run_relation["meta"]["_httk_label"] == "labeled-input"
    record = next(item for item in records_response.json()["data"] if item["id"] == "record-1")
    product_relation = record["relationships"]["_httk_has_product"]["data"][0]
    assert product_relation["type"] == "_httk_records"
    assert product_relation["meta"] == {"role": "product", "_httk_label": "derived"}
