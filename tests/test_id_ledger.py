"""Tests for the signed, append-only :class:`httk.store.IdLedger`."""

import json
import logging
import os
import sqlite3
import uuid
from pathlib import Path

import pytest
from httk.core._json import json_bytes
from httk.core.crypto import ed25519_public_key
from httk.core.entry_ids import format_entry_id, parse_entry_id
from httk.core.project import format_public_key, key_fingerprint
from httk.core.project.sealing import SealKey, build_seal_body, sign_seal_body

from httk.store import IdLedger, IdLedgerError, check_ledger_key
from httk.store.id_ledger import LEDGER_KIND, _initialize_schema

BASES = {"structures": "anyt.am.structure", "refs": "anyt.am.ref"}
SERIES = "a"


def _key(seed: bytes | None = None) -> SealKey:
    """Return one ``(role, seed)`` signing key with a random or given seed."""

    return ("identity", os.urandom(32) if seed is None else seed)


def _fingerprint(key: SealKey) -> str:
    """Return the trust-anchor fingerprint of a signing key."""

    return key_fingerprint(format_public_key(ed25519_public_key(key[1])))


def _create(path: Path, key: SealKey) -> IdLedger:
    """Create a ledger at *path* signed by *key*."""

    return IdLedger.create(path, bases=BASES, series=SERIES, keys=[key])


# -- assign ------------------------------------------------------------------


def test_assign_is_idempotent(tmp_path: Path) -> None:
    key = _key()
    with _create(tmp_path / "ids.sqlite", key) as ledger:
        first = ledger.assign("k1", "structures")
        assert ledger.assign("k1", "structures") == first


def test_assign_family_mismatch_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        ledger.assign("k1", "structures")
        with pytest.raises(IdLedgerError, match="family"):
            ledger.assign("k1", "refs")


def test_assign_of_aliased_key_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        with pytest.raises(IdLedgerError, match="alias"):
            ledger.assign("dup", "structures")


def test_assign_unknown_family_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger, pytest.raises(IdLedgerError, match="no id base"):
        ledger.assign("k1", "nope")


# -- alias -------------------------------------------------------------------


def test_alias_records_and_resolves(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        assert ledger.lookup("dup") == target


def test_alias_is_idempotent(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        ledger.alias("dup", target)  # no error
        assert ledger.lookup("dup") == target


def test_alias_conflict_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        one = ledger.assign("a", "structures")
        two = ledger.assign("b", "structures")
        ledger.alias("dup", one)
        with pytest.raises(IdLedgerError, match="already aliased"):
            ledger.alias("dup", two)


def test_alias_unknown_target_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        ledger.assign("owner", "structures")
        with pytest.raises(IdLedgerError, match="not assigned"):
            ledger.alias("dup", "anyt.am.structure-a-999")


def test_alias_of_assigned_key_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        one = ledger.assign("a", "structures")
        two = ledger.assign("b", "structures")
        with pytest.raises(IdLedgerError, match="already assigned"):
            ledger.alias("b", one)
        assert two  # b keeps its own id


# -- counter monotonicity ----------------------------------------------------


def test_counter_is_monotone_over_gaps_and_large_numbers(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    # Hand-seal a ledger whose structures family already holds a gap and a large
    # number, then confirm the next mint is strictly above the maximum.
    records = [
        {"key": "s1", "family": "structures", "id": "anyt.am.structure-a-3"},
        {"key": "s2", "family": "structures", "id": "anyt.am.structure-a-100"},
    ]
    _hand_seal(path, records, key)
    with IdLedger.open(path, keys=[key]) as ledger:
        assert ledger.assign("s3", "structures") == "anyt.am.structure-a-101"


# -- open verification -------------------------------------------------------


def test_open_catches_content_edit(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    with _create(path, _key()) as ledger:
        ledger.assign("k1", "structures")
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("UPDATE records SET id = 'anyt.am.structure-a-2' WHERE seq = 1")  # edit, no re-sign
    finally:
        connection.close()
    with pytest.raises(IdLedgerError, match="does not verify|restore|Restore"):
        IdLedger.open(path)


def test_open_catches_truncation(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    with _create(path, _key()) as ledger:
        ledger.assign("k1", "structures")
        ledger.assign("k2", "structures")
    # Drop the last record without touching the segment that still claims it.
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("DELETE FROM records WHERE seq = (SELECT MAX(seq) FROM records)")
    finally:
        connection.close()
    with pytest.raises(IdLedgerError, match="does not verify|partition|restore|Restore|git"):
        IdLedger.open(path)


def test_open_rejects_seal_swap_when_trusted_key_pinned(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    original = _key()
    pinned = _fingerprint(original)
    with _create(path, original):
        pass
    # Re-sign the same content with a DIFFERENT key: the signature is valid but
    # the signer is not the pinned trust anchor.
    _resign(path, _key())
    with pytest.raises(IdLedgerError, match="untrusted"):
        IdLedger.open(path, trusted_keys=[pinned])


def test_open_without_trusted_keys_logs_the_signer_as_an_audit_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "ids.sqlite"
    new_key = _key()
    with _create(path, _key()):
        pass
    _resign(path, new_key)  # a valid signature by an unpinned key: opens, logging the signer
    with caplog.at_level(logging.INFO, logger="httk.store.id_ledger"):
        IdLedger.open(path).close()
    message = next(record.getMessage() for record in caplog.records if "audit record" in record.getMessage())
    assert _fingerprint(new_key) in message


def test_open_rejects_base_map_and_series_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    # A properly-signed ledger whose entry id uses a base/series that disagree
    # with the declared subject: only open()'s own validation can catch this.
    records = [{"key": "k1", "family": "structures", "id": "other.base-z-1"}]
    _hand_seal(path, records, key)
    with pytest.raises(IdLedgerError, match="does not match the declared base"):
        IdLedger.open(path, keys=[key])


def test_open_rejects_non_sqlite_junk_file(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    path.write_text('{"not": "a sqlite database"}')
    with pytest.raises(IdLedgerError, match="not a readable sqlite database.*Restore it from git"):
        IdLedger.open(path, keys=[_key()])


def test_open_rejects_valid_sqlite_that_is_not_a_ledger(tmp_path: Path) -> None:
    # A readable sqlite file that is not a ledger (e.g. a store db pointed at by
    # --id-ledger) is diagnosed as the wrong container, not as corruption.
    path = tmp_path / "ids.sqlite"
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("CREATE TABLE something(x)")
    finally:
        connection.close()
    with pytest.raises(IdLedgerError, match="not an httk-idledger container"):
        IdLedger.open(path, keys=[_key()])


def test_open_missing_path_raises_without_creating_the_file(tmp_path: Path) -> None:
    path = tmp_path / "absent.sqlite"
    with pytest.raises(IdLedgerError, match="does not exist|Restore it from git"):
        IdLedger.open(path, keys=[_key()])
    assert not path.exists()  # sqlite3.connect must not have created it
    assert not (tmp_path / "absent.sqlite.lock").exists()  # lock released on refusal


# -- no-op close and round trip ----------------------------------------------


def test_noop_close_leaves_file_byte_identical(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    with _create(path, _key()) as ledger:
        ledger.assign("k1", "structures")
    before = path.read_bytes()
    # Reopen, assign the SAME key (idempotent, no state change), then close.
    with IdLedger.open(path, keys=[_key()]) as ledger:
        ledger.assign("k1", "structures")
    assert path.read_bytes() == before


def test_reopen_and_extend_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key) as ledger:
        first = ledger.assign("k1", "structures")
    with IdLedger.open(path, keys=[key]) as ledger:
        assert ledger.lookup("k1") == first  # first session's mint survives
        second = ledger.assign("k2", "structures")
        assert second != first
    with IdLedger.open(path, keys=[key]) as ledger:
        assert ledger.lookup("k2") == second


# -- lock --------------------------------------------------------------------


def test_lock_refusal_names_the_lock_path(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    lock = tmp_path / "ids.sqlite.lock"
    with _create(path, _key()):
        assert lock.exists()
        with pytest.raises(IdLedgerError) as excinfo:
            IdLedger.open(path)
    assert str(lock) in str(excinfo.value)
    assert not lock.exists()  # released on close


# -- key validator -----------------------------------------------------------


def test_check_ledger_key_warns_on_whitespace(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="httk.store.id_ledger"):
        assert check_ledger_key(" k ") == " k "
    assert any("whitespace" in record.message for record in caplog.records)


def test_check_ledger_key_warns_on_non_url_safe(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="httk.store.id_ledger"):
        assert check_ledger_key("a/b") == "a/b"
    assert any("URL-safe" in record.message for record in caplog.records)


def test_check_ledger_key_accepts_conventional_key(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="httk.store.id_ledger"):
        assert check_ledger_key("amdb:12:structure") == "amdb:12:structure"
    assert not caplog.records


# -- helpers -----------------------------------------------------------------


def _hand_seal(
    path: Path,
    records: list[dict[str, object]],
    key: SealKey,
    *,
    bases: dict[str, str] | None = None,
) -> None:
    """Build a fully-signed one-segment ledger database with the given records."""

    bases = BASES if bases is None else bases
    ledger_uuid = uuid.uuid4().hex
    subject = {
        "ledger_format_version": 1,
        "ledger": ledger_uuid,
        "series": SERIES,
        "bases": bases,
        "segment": 1,
        "first_record": 1,
        "record_count": len(records),
    }
    body = build_seal_body(LEDGER_KIND, subject, records)
    body_sha256, signatures = sign_seal_body(body, [key])
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        _initialize_schema(connection, SERIES, ledger_uuid)
        connection.execute("BEGIN")
        for offset, record in enumerate(records):
            connection.execute(
                "INSERT INTO records(seq, key, family, id, alias_of, supersedes) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    1 + offset,
                    record["key"],
                    record.get("family"),
                    record.get("id"),
                    record.get("alias_of"),
                    record.get("supersedes"),
                ),
            )
        connection.execute(
            "INSERT INTO segments"
            "(segment, first_seq, record_count, created_at, subject, body_sha256, signatures) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                1,
                len(records),
                body["created_at"],
                json_bytes(body["subject"]).decode("utf-8"),
                body_sha256,
                json_bytes(signatures).decode("utf-8"),
            ),
        )
        connection.execute("COMMIT")
    finally:
        connection.close()


def _resign(path: Path, key: SealKey) -> None:
    """Re-sign every stored segment under a different key, keeping content verbatim."""

    connection = sqlite3.connect(path, isolation_level=None)
    try:
        rows = connection.execute(
            "SELECT segment, first_seq, record_count, created_at, subject FROM segments ORDER BY segment"
        ).fetchall()
        for segment, first_seq, record_count, created_at, subject_json in rows:
            subject = json.loads(subject_json)
            records = [
                _record_dict(row)
                for row in connection.execute(
                    "SELECT key, family, id, alias_of, supersedes FROM records WHERE seq >= ? AND seq < ? ORDER BY seq",
                    (first_seq, first_seq + record_count),
                ).fetchall()
            ]
            body = build_seal_body(LEDGER_KIND, subject, records)
            body["created_at"] = created_at
            body_sha256, signatures = sign_seal_body(body, [key])
            connection.execute(
                "UPDATE segments SET body_sha256 = ?, signatures = ? WHERE segment = ?",
                (body_sha256, json_bytes(signatures).decode("utf-8"), segment),
            )
    finally:
        connection.close()


def _record_dict(row: tuple[object, ...]) -> dict[str, object]:
    """Rebuild the signed record dict from a raw ``records`` row."""

    key, family, entry_id, alias_of, supersedes = row
    record: dict[str, object] = (
        {"key": key, "alias_of": alias_of}
        if alias_of is not None
        else {
            "key": key,
            "family": family,
            "id": entry_id,
        }
    )
    if supersedes is not None:
        record["supersedes"] = supersedes
    return record


# -- supersession (split / merge regrouping) ---------------------------------


def test_split_transition_alias_supersede_assign(tmp_path: Path) -> None:
    key = _key()
    path = tmp_path / "ids.sqlite"
    with _create(path, key) as ledger:
        owner_id = ledger.assign("owner", "structures")
        ledger.alias("shared", owner_id)
        # The group splits: the aliased key becomes its own assignment.
        fresh = ledger.assign("shared", "structures", supersede=True)
        assert fresh != owner_id
        assert ledger.lookup("shared") == fresh  # newest binding wins
        assert ledger.lookup("owner") == owner_id  # old id still resolvable via owner
    with IdLedger.open(path, keys=[key]) as ledger:
        assert ledger.lookup("shared") == fresh
        assert ledger.lookup("owner") == owner_id


def test_merge_transition_assign_supersede_alias(tmp_path: Path) -> None:
    key = _key()
    path = tmp_path / "ids.sqlite"
    with _create(path, key) as ledger:
        owner_id = ledger.assign("owner", "structures")
        old_id = ledger.assign("moved", "structures")
        assert old_id != owner_id
        # The groups merge: the assigned key becomes an alias of the owner.
        ledger.alias("moved", owner_id, supersede=True)
        assert ledger.lookup("moved") == owner_id
        assert ledger.lookup("owner") == owner_id
    with IdLedger.open(path, keys=[key]) as ledger:
        assert ledger.lookup("moved") == owner_id
        # A later new key mints above the reserved (now-orphan) old id.
        nxt = ledger.assign("new", "structures")
        assert parse_entry_id(nxt)[2] > parse_entry_id(old_id)[2]  # type: ignore[index]


def test_supersede_of_assigned_key_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        ledger.assign("k1", "structures")
        with pytest.raises(IdLedgerError, match="never an assignment to another assignment"):
            ledger.assign("k1", "structures", supersede=True)


def test_default_false_errors_unchanged(tmp_path: Path) -> None:
    # supersede defaults to False: aliased-key assign and assigned-key alias
    # both raise exactly as before the feature existed.
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        with pytest.raises(IdLedgerError, match="is an alias of"):
            ledger.assign("dup", "structures")
        with pytest.raises(IdLedgerError, match="already assigned"):
            ledger.alias("owner", target + "x")


def test_open_rejects_forged_supersedes_chain(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    # "moved" supersedes an id it never actually resolved to.
    records = [
        {"key": "owner", "family": "structures", "id": "anyt.am.structure-a-1"},
        {"key": "moved", "family": "structures", "id": "anyt.am.structure-a-2"},
        {"key": "moved", "alias_of": "anyt.am.structure-a-1", "supersedes": "anyt.am.structure-a-99"},
    ]
    _hand_seal(path, records, key)
    with pytest.raises(IdLedgerError, match="prior binding was"):
        IdLedger.open(path, keys=[key])


def test_open_rejects_duplicate_live_binding(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    # Two records for "k1" with no supersession between them.
    records = [
        {"key": "k1", "family": "structures", "id": "anyt.am.structure-a-1"},
        {"key": "k1", "family": "structures", "id": "anyt.am.structure-a-2"},
    ]
    _hand_seal(path, records, key)
    with pytest.raises(IdLedgerError, match="duplicate live binding"):
        IdLedger.open(path, keys=[key])


def test_counter_unaffected_by_superseded_ids(tmp_path: Path) -> None:
    key = _key()
    path = tmp_path / "ids.sqlite"
    with _create(path, key) as ledger:
        owner_id = ledger.assign("owner", "structures")  # -1
        ledger.assign("moved", "structures")  # -2, will be orphaned
        ledger.alias("moved", owner_id, supersede=True)  # merge; -2 reserved
        nxt = ledger.assign("new", "structures")
        assert nxt == "anyt.am.structure-a-3"  # -2 still counted, not reused


def test_supersede_on_new_key_assign_errors(tmp_path: Path) -> None:
    with (
        _create(tmp_path / "ids.sqlite", _key()) as ledger,
        pytest.raises(IdLedgerError, match="broken regrouping upstream"),
    ):
        ledger.assign("brand_new", "structures", supersede=True)


def test_supersede_on_new_key_alias_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        with pytest.raises(IdLedgerError, match="broken regrouping upstream"):
            ledger.alias("brand_new", target, supersede=True)


# -- review-hardening guards -------------------------------------------------


def test_create_rechecks_existence_under_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a concurrent create finishing (and releasing its lock) in the
    # window between the existence check and lock acquisition: the racer writes
    # the ledger while we take the lock, and create must then refuse.
    import httk.store.id_ledger as mod

    path = tmp_path / "ids.sqlite"
    real_acquire = mod._acquire_lock

    def racing_acquire(lock_path: Path) -> None:
        real_acquire(lock_path)
        path.write_text("{}")  # a concurrent creator's ledger appears

    monkeypatch.setattr(mod, "_acquire_lock", racing_acquire)
    with pytest.raises(IdLedgerError, match="already exists"):
        IdLedger.create(path, bases=BASES, series=SERIES, keys=[_key()])
    assert not (tmp_path / "ids.sqlite.lock").exists()  # lock released on refusal


def test_create_rejects_duplicate_bases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shared by more than one family"):
        IdLedger.create(
            tmp_path / "ids.sqlite",
            bases={"a": "anyt.am.thing", "b": "anyt.am.thing"},
            series=SERIES,
            keys=[_key()],
        )


def test_open_rejects_duplicate_bases(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    _hand_seal(path, [], key, bases={"a": "anyt.am.thing", "b": "anyt.am.thing"})
    with pytest.raises(IdLedgerError, match="shares id base"):
        IdLedger.open(path, keys=[key])


def test_open_rejects_base_map_mismatch_vs_expected(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    with pytest.raises(IdLedgerError, match="not the expected"):
        IdLedger.open(path, keys=[key], bases={"structures": "anyt.am.other", "refs": "anyt.am.ref"})


def test_open_rejects_series_mismatch_vs_expected(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    with pytest.raises(IdLedgerError, match="not the expected"):
        IdLedger.open(path, keys=[key], series="z")


def test_open_accepts_matching_bases_and_series(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    with IdLedger.open(path, keys=[key], bases=BASES, series=SERIES) as ledger:
        assert ledger.assign("k1", "structures")


# -- superset-open base-map extension ----------------------------------------


def test_open_adds_missing_family_and_stamps_on_reseal(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    extended = {**BASES, "records": "anyt.am.rec"}
    with IdLedger.open(path, keys=[key], bases=extended) as ledger:
        assert ledger.assign("r1", "records") == format_entry_id("anyt.am.rec", SERIES, 1)
    # The added family is now stored: reopening with only the original bases is a
    # removal and errors, and reopening with the extended map keeps minting.
    with pytest.raises(IdLedgerError, match="removed or renamed"):
        IdLedger.open(path, keys=[key], bases=BASES)
    with IdLedger.open(path, keys=[key], bases=extended) as ledger:
        assert ledger.lookup("r1") == format_entry_id("anyt.am.rec", SERIES, 1)
        assert ledger.assign("r2", "records") == format_entry_id("anyt.am.rec", SERIES, 2)


def test_open_add_family_stamps_even_without_assign(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    before = path.read_bytes()
    extended = {**BASES, "records": "anyt.am.rec"}
    # No assign, but a family was added: close must reseal (subject stamping).
    with IdLedger.open(path, keys=[key], bases=extended):
        pass
    assert path.read_bytes() != before
    with pytest.raises(IdLedgerError, match="removed or renamed"):
        IdLedger.open(path, keys=[key], bases=BASES)


def test_open_matching_bases_noop_close_is_byte_identical(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    before = path.read_bytes()
    # Superset-open with an exactly-matching map adds nothing, so a no-op close
    # must not rewrite the file.
    with IdLedger.open(path, keys=[key], bases=BASES):
        pass
    assert path.read_bytes() == before


def test_open_rejects_removed_family(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    with pytest.raises(IdLedgerError, match="removed or renamed"):
        IdLedger.open(path, keys=[key], bases={"structures": "anyt.am.structure"})


def test_open_add_family_rejects_duplicate_base(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    # A new family reusing an existing family's base is a duplicate over the merged map.
    with pytest.raises(ValueError, match="shared by more than one family"):
        IdLedger.open(path, keys=[key], bases={**BASES, "clone": "anyt.am.structure"})


def test_open_add_family_rejects_malformed_base(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    with pytest.raises(ValueError):
        IdLedger.open(path, keys=[key], bases={**BASES, "bad": "not a base!"})


def test_alias_supersede_of_aliased_key_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        with pytest.raises(IdLedgerError, match="already an alias"):
            ledger.alias("dup", target, supersede=True)


def test_alias_supersede_self_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.sqlite", _key()) as ledger:
        own = ledger.assign("k1", "structures")
        with pytest.raises(IdLedgerError, match="cannot supersede its own id"):
            ledger.alias("k1", own, supersede=True)


def test_close_retries_after_failed_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key()
    path = tmp_path / "ids.sqlite"
    lock = tmp_path / "ids.sqlite.lock"
    ledger = _create(path, key)
    ledger.assign("k1", "structures")
    real_write = ledger._write
    calls = {"n": 0}

    def flaky_write() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        real_write()

    monkeypatch.setattr(ledger, "_write", flaky_write)
    with pytest.raises(OSError, match="disk full"):
        ledger.close()
    assert lock.exists()  # a failed reseal keeps the lock for a retry
    ledger.close()  # retry reattempts the write
    assert not lock.exists()
    with IdLedger.open(path, keys=[key]) as reopened:
        assert reopened.lookup("k1") is not None  # the mint was not silently dropped


# -- segment container semantics ---------------------------------------------


def _segment_count(path: Path) -> int:
    """Return how many segments the ledger database holds."""

    connection = sqlite3.connect(path, isolation_level=None)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM segments").fetchone()[0])
    finally:
        connection.close()


def test_two_sessions_write_two_segments_beyond_creation(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key) as ledger:
        first = ledger.assign("k1", "structures")  # session A -> segment 2
    with IdLedger.open(path, keys=[key]) as ledger:
        second = ledger.assign("k2", "structures")  # session B -> segment 3
    assert _segment_count(path) == 3  # creation segment plus one per appending session
    with IdLedger.open(path, keys=[key], trusted_keys=[_fingerprint(key)]) as ledger:
        assert ledger.lookup("k1") == first  # every segment verifies as trusted
        assert ledger.lookup("k2") == second


def test_tampering_an_old_segment_breaks_its_verification(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key) as ledger:
        ledger.assign("k1", "structures")  # segment 2, record seq 1
    with IdLedger.open(path, keys=[key]) as ledger:
        ledger.assign("k2", "structures")  # segment 3, record seq 2
    # Edit the OLD segment's record; its signature no longer matches.
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("UPDATE records SET id = 'anyt.am.structure-a-9' WHERE seq = 1")
    finally:
        connection.close()
    with pytest.raises(IdLedgerError, match="segment 2|does not verify"):
        IdLedger.open(path, keys=[key])


def test_deleting_a_middle_record_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key) as ledger:
        ledger.assign("k1", "structures")
        ledger.assign("k2", "structures")
        ledger.assign("k3", "structures")  # one segment covering three records
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("DELETE FROM records WHERE seq = 2")  # middle record
    finally:
        connection.close()
    with pytest.raises(IdLedgerError, match="git"):
        IdLedger.open(path, keys=[key])


def test_empty_base_extension_writes_a_segment(tmp_path: Path) -> None:
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key):
        pass
    assert _segment_count(path) == 1
    extended = {**BASES, "records": "anyt.am.rec"}
    with IdLedger.open(path, keys=[key], bases=extended):
        pass  # no assigns, but the grown scheme must be stamped
    assert _segment_count(path) == 2  # an empty segment carrying the grown bases
    with IdLedger.open(path, keys=[key], bases=extended) as ledger:
        assert ledger.assign("r1", "records") == format_entry_id("anyt.am.rec", SERIES, 1)


def test_dirty_close_without_keys_raises_and_keeps_lock(tmp_path: Path) -> None:
    from httk.core.project.sealing import SealError

    path = tmp_path / "ids.sqlite"
    lock = tmp_path / "ids.sqlite.lock"
    key = _key()
    with _create(path, key):
        pass
    ledger = IdLedger.open(path, keys=[])  # opened without a signing key
    ledger.assign("k1", "structures")
    with pytest.raises(SealError):
        ledger.close()  # a dirty close with no key cannot sign the new segment
    assert lock.exists()  # the ledger stays open and locked for a retry
    assert _segment_count(path) == 1  # nothing was written
    ledger._keys = (key,)  # supply a key and retry
    ledger.close()
    assert not lock.exists()
    with IdLedger.open(path, keys=[key]) as reopened:
        assert reopened.lookup("k1") is not None


def test_failed_create_leaves_no_partial_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import httk.store.id_ledger as mod

    path = tmp_path / "ids.sqlite"
    real_initialize = mod._initialize_schema

    def failing_initialize(connection: sqlite3.Connection, series: str, ledger_uuid: str) -> None:
        real_initialize(connection, series, ledger_uuid)  # lay down schema + meta, then fail
        raise OSError("disk full")

    monkeypatch.setattr(mod, "_initialize_schema", failing_initialize)
    with pytest.raises(OSError, match="disk full"):
        IdLedger.create(path, bases=BASES, series=SERIES, keys=[_key()])
    assert not path.exists()  # no schema-only/partial database left behind
    assert not (tmp_path / "ids.sqlite.lock").exists()  # lock released
    # A retried create at the same path succeeds: nothing claims it exists.
    monkeypatch.setattr(mod, "_initialize_schema", real_initialize)
    with IdLedger.create(path, bases=BASES, series=SERIES, keys=[_key()]) as ledger:
        assert ledger.assign("k1", "structures")


def test_segment_with_foreign_ledger_uuid_is_rejected(tmp_path: Path) -> None:
    # A segment validly re-signed (same key) but whose subject carries a different
    # ledger uuid than the meta must be rejected as a graft from another ledger.
    path = tmp_path / "ids.sqlite"
    key = _key()
    with _create(path, key) as ledger:
        ledger.assign("k1", "structures")  # segment 2, record seq 1
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        first_seq, record_count, created_at, subject_json = connection.execute(
            "SELECT first_seq, record_count, created_at, subject FROM segments WHERE segment = 2"
        ).fetchone()
        subject = json.loads(subject_json)
        subject["ledger"] = uuid.uuid4().hex  # stamp a foreign ledger identity
        records = [
            _record_dict(row)
            for row in connection.execute(
                "SELECT key, family, id, alias_of, supersedes FROM records WHERE seq >= ? AND seq < ? ORDER BY seq",
                (first_seq, first_seq + record_count),
            ).fetchall()
        ]
        body = build_seal_body(LEDGER_KIND, subject, records)
        body["created_at"] = created_at
        body_sha256, signatures = sign_seal_body(body, [key])  # a genuinely valid signature
        connection.execute(
            "UPDATE segments SET subject = ?, body_sha256 = ?, signatures = ? WHERE segment = 2",
            (json_bytes(subject).decode("utf-8"), body_sha256, json_bytes(signatures).decode("utf-8")),
        )
    finally:
        connection.close()
    with pytest.raises(IdLedgerError, match="different ledger|grafted"):
        IdLedger.open(path, keys=[key])
