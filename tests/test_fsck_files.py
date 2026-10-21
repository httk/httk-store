"""``fsck(data_dir=...)`` verification of stored ``FileRecord`` files (SQL and Mongo)."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from httk.core import FileRecord
from httk.core.storage import StorageInfo

from httk.store import EntryFamilyDeclaration, EntryRecordDeclaration
from httk.store.backend.sql import Backend, SqlStore


class FileHolderFamily:
    """Test-only family whose record holds a child ``FileRecord``."""


@dataclass(frozen=True)
class FileHolder:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="fsck_file_holder", identity_name="fsck-holder")

    label: str
    f: FileRecord


_FAMILIES = (
    EntryFamilyDeclaration(
        name="test-fsck-files",
        family=FileHolderFamily,
        records=(EntryRecordDeclaration(name="test-fsck-files-holder", record=FileHolder),),
    ),
)


def _file(root: Path, rest: str, content: bytes, *, write: bytes | None = None) -> FileRecord:
    """Return a ``data/<rest>`` record for ``content``; write ``write`` (default ``content``) unless ``b""``."""
    data = content if write is None else write
    if data:
        path = root / rest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return FileRecord(f"data/{rest}", Path(rest).name, sha256=hashlib.sha256(content).hexdigest())


def _populate(store: Any, root: Path) -> dict[str, int]:
    sids = {
        "ok": store.save(_file(root, "ok.txt", b"ok")),
        "missing": store.save(_file(root, "missing.txt", b"missing", write=b"")),
        "changed": store.save(_file(root, "changed.txt", b"original", write=b"edited")),
        "remote": store.save(FileRecord("https://example.org/x", "x", sha256="ab" * 32)),
        "escape": store.save(FileRecord("data/../secret", "secret", sha256="cd" * 32)),
    }
    store.save(FileHolder("holder", _file(root, "sub/child.txt", b"child", write=b"tampered")))
    (root.parent / "secret").write_bytes(b"outside")
    # Symlinks out of the data tree hold matching bytes but are never followed.
    (root.parent / "outside.txt").write_bytes(b"linked")
    (root / "link.txt").symlink_to(root.parent / "outside.txt")
    (root.parent / "outside_dir").mkdir()
    (root.parent / "outside_dir" / "inner.txt").write_bytes(b"inner")
    (root / "linkdir").symlink_to(root.parent / "outside_dir")
    sids["link"] = store.save(_file(root, "link.txt", b"linked", write=b""))
    sids["linkdir"] = store.save(_file(root, "linkdir/inner.txt", b"inner", write=b""))
    return sids


def _assert_summary(summary: Any, sids: dict[str, int], store: Any) -> None:
    assert summary.files_checked == 6  # ok, missing, changed, child, two symlinks
    assert summary.files_unchecked == 2  # https and the ``..`` escape
    assert summary.files_missing == (
        (sids["missing"], "data/missing.txt"),
        (sids["link"], "data/link.txt"),
        (sids["linkdir"], "data/linkdir/inner.txt"),
    )
    mismatched = dict(summary.files_mismatched)
    assert mismatched.pop(sids["changed"]) == "data/changed.txt"
    # The child FileRecord lives in the same table, reached through its holder.
    ((child_sid, child_url),) = mismatched.items()
    assert child_url == "data/sub/child.txt"
    assert store.fetch(FileRecord, child_sid).name == "child.txt"


@pytest.fixture(params=["sqlite", "duckdb"])
def sql_database(request: pytest.FixtureRequest) -> Iterator[Backend]:
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Backend.duckdb() as database:
            yield database
        return
    with Backend.sqlite() as database:
        yield database


def test_sql_fsck_verifies_file_records(sql_database: Backend, tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = SqlStore(sql_database, entry_families=_FAMILIES)
    sids = _populate(store, root)
    exclusive = sql_database.engine.dialect.name == "duckdb"

    plain = store.fsck(exclusive=exclusive)
    assert (plain.files_checked, plain.files_unchecked, plain.files_missing) == (0, 0, ())

    summary = store.fsck(data_dir=root, exclusive=exclusive)
    _assert_summary(summary, sids, store)
    # Read-only: a second pass sees the same rows and reports the same result.
    assert store.fsck(data_dir=str(root), exclusive=exclusive).files_mismatched == summary.files_mismatched


def test_mongo_fsck_verifies_file_records(mongo_test_database: Any, tmp_path: Path) -> None:
    from httk.store.backend.mongo import MongoStore

    root = tmp_path / "data"
    store = MongoStore(mongo_test_database, entry_families=_FAMILIES)
    sids = _populate(store, root)
    assert store.fsck().files_checked == 0
    _assert_summary(store.fsck(data_dir=root), sids, store)


def _relocate_and_check(store: Any, root: Path, **fsck: Any) -> None:
    """``relocate_file`` re-points a row checked by fsck, or marks it lost so fsck only counts it."""
    sid = store.save(FileRecord(None, "moved.txt", sha256=hashlib.sha256(b"moved").hexdigest()))
    assert store.fsck(data_dir=root, **fsck).files_lost == 1
    _file(root, "archive/moved.txt", b"moved")
    store.relocate_file(sid, "data/archive/moved.txt")
    assert store.fetch(FileRecord, sid).url == "data/archive/moved.txt"
    summary = store.fsck(data_dir=root, **fsck)
    assert (summary.files_checked, summary.files_lost, summary.files_missing, summary.files_mismatched) == (
        1,
        0,
        (),
        (),
    )
    store.relocate_file(sid, None)
    assert store.fetch(FileRecord, sid).url is None
    summary = store.fsck(data_dir=root, **fsck)
    assert (summary.files_checked, summary.files_lost, summary.files_missing) == (0, 1, ())
    with pytest.raises(KeyError):
        store.relocate_file(sid + 1000, None)
    with pytest.raises(TypeError, match="url must be a str or None"):
        store.relocate_file(sid, b"data/x")


def test_sql_relocate_file_and_lost_files(sql_database: Backend, tmp_path: Path) -> None:
    store = SqlStore(sql_database, entry_families=_FAMILIES)
    _relocate_and_check(store, tmp_path / "data", exclusive=sql_database.engine.dialect.name == "duckdb")


def test_sql_relocate_file_without_file_rows(sql_database: Backend) -> None:
    store = SqlStore(sql_database, entry_families=_FAMILIES)
    with pytest.raises(KeyError, match="no FileRecord table"):
        store.relocate_file(1, "data/x")
    store.save(FileRecord("data/x", "x", sha256="ab" * 32))
    with pytest.raises(KeyError) as missing:
        store.relocate_file(999, "data/x")
    assert missing.value.args == (FileRecord, 999)


def test_mongo_relocate_file_and_lost_files(mongo_test_database: Any, tmp_path: Path) -> None:
    from httk.store.backend.mongo import MongoStore

    _relocate_and_check(MongoStore(mongo_test_database, entry_families=_FAMILIES), tmp_path / "data")
