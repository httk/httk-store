"""Regression: stored ``files`` entries serve every stored OPTIMADE property."""

import datetime

import pytest
from httk.core import FileEntry, FileRecord

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore, StoredEntryFederation, StoredEntrySource
from httk.store.query.optimade_filters import FilterTranslationError

MTIME = datetime.datetime(2026, 10, 20, 12, 0, tzinfo=datetime.UTC)
DESCRIPTION = "VASP 6.6.0, PBEsol Static, 32 cores, 4.946 s"
SHA256 = "50fb3e0738b8f05204d5ebd2522779fe3fcd160c68683c056ad1494e22c3f824"
BARE_SHA256 = "0" * 64


def test_stored_files_serve_description_checksums_and_metadata() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={FileEntry: FileRecord}, entry_ids=EntryIdScheme("httk.test", "1"))
        store.save(FileRecord("data/run/OUTCAR", "OUTCAR", size=3, description=DESCRIPTION, sha256=SHA256, mtime=MTIME))
        store.save(FileRecord("data/run/bare", "bare", sha256=BARE_SHA256))
        federation = StoredEntryFederation((StoredEntrySource(store, FileEntry, "files"),))
        expected = {
            "description": DESCRIPTION,
            "checksums": {"sha256": SHA256},
            "mtime": MTIME.isoformat(),
            "version": None,
        }

        rows = {row["name"]: row for row in federation.query().rows}
        assert {name: rows["OUTCAR"][name] for name in expected} == expected
        assert rows["bare"]["description"] is None and rows["bare"]["checksums"] == {"sha256": BARE_SHA256}

        fetched = federation.fetch(rows["OUTCAR"]["id"], fields=tuple(expected))
        assert fetched is not None
        assert {name: fetched[0][name] for name in expected} == expected

        # Served but response-only, like the other ``files`` properties: an
        # explicit not-implemented filter, not a silent empty result.
        with pytest.raises(FilterTranslationError, match="not implemented"):
            federation.query('description CONTAINS "Static"')


def test_lost_file_serves_ni_content_uri() -> None:
    """A stored ``url=None`` (lost bytes) serves through core's projection as an RFC 6920 ``ni:`` URI."""
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={FileEntry: FileRecord}, entry_ids=EntryIdScheme("httk.test", "1"))
        sid = store.save(FileRecord(None, "lost", sha256=SHA256))
        federation = StoredEntryFederation((StoredEntrySource(store, FileEntry, "files"),))
        expected = "ni:///sha-256;UPs-Bzi48FIE1evSUid5_j_NFgxoaDwFatFJTiLD-CQ"

        (row,) = federation.query().rows
        assert (row["url"], row["checksums"]) == (expected, {"sha256": SHA256})
        fetched = federation.fetch(row["id"], fields=("url",))
        assert fetched is not None and fetched[0]["url"] == expected

        # Relocated, the stored url is served as is.
        store.relocate_file(sid, "data/run/OUTCAR")
        assert federation.query().rows[0]["url"] == "data/run/OUTCAR"
