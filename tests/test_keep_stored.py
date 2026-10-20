"""``KeepStored`` fields keep the first stored value on a deduplicating re-save (SQL single and bulk paths)."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import pytest
import sqlalchemy
from httk.core import FileRecord
from httk.core.storage import StorageInfo

from httk.store import EntryFamilyDeclaration, EntryRecordDeclaration
from httk.store.backend.sql import Backend, SqlStore
from httk.store.store_common import EntryMetadataConflictError


class KeepHolderFamily:
    """Test-only family whose record holds a child ``FileRecord``."""


@dataclass(frozen=True)
class KeepHolder:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="keep_holder", identity_name="keep-holder")

    label: str
    f: FileRecord


_FAMILIES = (
    EntryFamilyDeclaration(
        name="test-keep-stored",
        family=KeepHolderFamily,
        records=(EntryRecordDeclaration(name="test-keep-stored-holder", record=KeepHolder),),
    ),
)
_FIRST = FileRecord("data/ws/jobs/x/POSCAR", "POSCAR", size=10, sha256="ab" * 32)
_SECOND = replace(_FIRST, url="data/ws/jobs/y/POSCAR", size=11, description="other path")


@pytest.fixture(params=["sqlite", "duckdb"])
def store(request: pytest.FixtureRequest) -> Iterator[SqlStore]:
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Backend.duckdb() as database:
            yield SqlStore(database, entry_families=_FAMILIES)
        return
    with Backend.sqlite() as database:
        yield SqlStore(database, entry_families=_FAMILIES)


def _file_rows(store: SqlStore) -> list[Any]:
    with store._database.engine.connect() as connection:
        return list(connection.execute(sqlalchemy.text("SELECT url, size, description FROM core_file")))


def test_top_level_resave_keeps_first_location(store: SqlStore) -> None:
    sid = store.save(_FIRST)
    assert store.save(_SECOND) == sid
    assert _file_rows(store) == [(_FIRST.url, 10, None)]
    # A field without KeepStored still conflicts.
    with pytest.raises(EntryMetadataConflictError, match="immutable_id"):
        store.save(replace(_SECOND, immutable_id="different"))


def test_child_resave_keeps_first_location(store: SqlStore) -> None:
    store.save(KeepHolder("a", _FIRST))
    store.save(KeepHolder("b", _SECOND))
    assert _file_rows(store) == [(_FIRST.url, 10, None)]


# Deferred finalize and workers>1 accept only a physically empty store.
@pytest.mark.parametrize(
    ("options", "populated"),
    [({}, False), ({}, True), ({"finalize": "deferred"}, False), ({"workers": 2}, False)],
    ids=["serial-empty", "serial-populated", "deferred", "parallel"],
)
def test_bulk_resave_keeps_first_location(store: SqlStore, options: dict[str, Any], populated: bool) -> None:
    if populated:
        store.save(_FIRST)
    with store.bulk_ingest(**options) as bulk:
        if not populated:
            bulk.save(_FIRST)
        bulk.save(_SECOND)
        bulk.save(KeepHolder("a", _SECOND))
    assert _file_rows(store) == [(_FIRST.url, 10, None)]


def test_mongo_resave_keeps_first_location(mongo_test_database: Any) -> None:
    from httk.store.backend.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_families=_FAMILIES)
    sid = store.save(_FIRST)
    assert store.save(_SECOND) == sid
    store.save(KeepHolder("b", _SECOND))
    assert [document["f"]["url"] for document in mongo_test_database.database["core_file"].find({})] == [_FIRST.url]
    with pytest.raises(EntryMetadataConflictError, match="immutable_id"):
        store.save(replace(_SECOND, immutable_id="different"))
