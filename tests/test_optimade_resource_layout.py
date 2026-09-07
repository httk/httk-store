"""Layout-drift rejection for ``OptimadeResource``'s persisted ``member`` field.

``OptimadeResource`` gained a fourth, identity-participating field (``member``)
so a resource can address ``included`` in place of ``data``.  Because the new
field is not ``IdentitySkip`` and has no nullable column, a store created
before the field existed must never be silently reused: reopening it has to
raise ``StorageLayoutUpgradeRequiredError`` rather than dedup or reconstruct
resources incorrectly against the new identity shape.
"""

import json

import pytest
import sqlalchemy
from httk.core.optimade import OptimadeDocument, OptimadeResource, OptimadeSchemaSnapshot

from httk.store import EntryFamilyDeclaration, EntryRecordDeclaration
from httk.store.backend.sql import Backend, SqlStore, StorageLayoutUpgradeRequiredError

PAGE_TEXT = '{"data": [{"id": "reference-1", "type": "references", "attributes": {}}]}'
INFO_TEXT = '{"data": {"id": "references", "type": "info", "properties": {}}}'


class OptimadeResourceLayoutFamily:
    """Test-only, application-owned family wrapping ``OptimadeResource`` directly."""


OPTIMADE_RESOURCE_LAYOUT = EntryFamilyDeclaration(
    name="test-optimade-resource-layout-family",
    family=OptimadeResourceLayoutFamily,
    records=(EntryRecordDeclaration(name="test-optimade-resource-layout-record", record=OptimadeResource),),
)


def _resource() -> OptimadeResource:
    page = OptimadeDocument(PAGE_TEXT, "https://example.invalid/v1/references")
    info = OptimadeDocument(INFO_TEXT, "https://example.invalid/v1/info/references")
    return OptimadeResource(page, 0, OptimadeSchemaSnapshot("references", info))


def _read_entry_schemas(database: Backend) -> str | None:
    with database.engine.connect() as connection:
        return connection.execute(
            sqlalchemy.text("SELECT value FROM _httk_store_metadata WHERE key = 'entry_schemas'")
        ).scalar_one_or_none()


def _write_entry_schemas(database: Backend, value: str) -> None:
    with database.engine.begin() as connection:
        connection.execute(
            sqlalchemy.text("UPDATE _httk_store_metadata SET value = :value WHERE key = 'entry_schemas'"),
            {"value": value},
        )


def test_reopen_rejects_optimade_resource_table_persisted_without_member_field() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_families=(OPTIMADE_RESOURCE_LAYOUT,))
        store.save(_resource())

        stored = json.loads(_read_entry_schemas(database) or "")
        # Simulate a store created before "member" existed: the old class's
        # fingerprint has no such field at all, not merely a differing column.
        del stored["tables"]["optimade_resource"]["fields"]["member"]
        _write_entry_schemas(database, json.dumps(stored, sort_keys=True, separators=(",", ":")))

        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            SqlStore(database, entry_families=(OPTIMADE_RESOURCE_LAYOUT,))
        assert "optimade_resource" in error.value.diff["schema"]
