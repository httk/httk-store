"""Tests for the sealed, append-only :class:`httk.store.IdLedger`."""

import json
import logging
import os
from pathlib import Path

import pytest
from httk.core.crypto import ed25519_public_key
from httk.core.entry_ids import parse_entry_id
from httk.core.project import format_public_key, key_fingerprint
from httk.core.project.sealing import SealKey, build_seal_body, write_seal

from httk.store import IdLedger, IdLedgerError, check_ledger_key
from httk.store.id_ledger import LEDGER_KIND

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
    with _create(tmp_path / "ids.json", key) as ledger:
        first = ledger.assign("k1", "structures")
        assert ledger.assign("k1", "structures") == first


def test_assign_family_mismatch_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        ledger.assign("k1", "structures")
        with pytest.raises(IdLedgerError, match="family"):
            ledger.assign("k1", "refs")


def test_assign_of_aliased_key_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        with pytest.raises(IdLedgerError, match="alias"):
            ledger.assign("dup", "structures")


def test_assign_unknown_family_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger, pytest.raises(IdLedgerError, match="no id base"):
        ledger.assign("k1", "nope")


# -- alias -------------------------------------------------------------------


def test_alias_records_and_resolves(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        assert ledger.lookup("dup") == target


def test_alias_is_idempotent(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        ledger.alias("dup", target)  # no error
        assert ledger.lookup("dup") == target


def test_alias_conflict_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        one = ledger.assign("a", "structures")
        two = ledger.assign("b", "structures")
        ledger.alias("dup", one)
        with pytest.raises(IdLedgerError, match="already aliased"):
            ledger.alias("dup", two)


def test_alias_unknown_target_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        ledger.assign("owner", "structures")
        with pytest.raises(IdLedgerError, match="not assigned"):
            ledger.alias("dup", "anyt.am.structure-a-999")


def test_alias_of_assigned_key_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        one = ledger.assign("a", "structures")
        two = ledger.assign("b", "structures")
        with pytest.raises(IdLedgerError, match="already assigned"):
            ledger.alias("b", one)
        assert two  # b keeps its own id


# -- counter monotonicity ----------------------------------------------------


def test_counter_is_monotone_over_gaps_and_large_numbers(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
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
    path = tmp_path / "ids.json"
    with _create(path, _key()) as ledger:
        ledger.assign("k1", "structures")
    document = json.loads(path.read_text())
    document["records"][0]["id"] = "anyt.am.structure-a-2"  # edit without re-signing
    path.write_text(json.dumps(document))
    with pytest.raises(IdLedgerError, match="signature does not verify|restore"):
        IdLedger.open(path)


def test_open_catches_truncation(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    with _create(path, _key()) as ledger:
        ledger.assign("k1", "structures")
        ledger.assign("k2", "structures")
    document = json.loads(path.read_text())
    document["records"] = document["records"][:1]  # drop a record without re-signing
    path.write_text(json.dumps(document))
    with pytest.raises(IdLedgerError, match="signature does not verify|restore"):
        IdLedger.open(path)


def test_open_rejects_seal_swap_when_trusted_key_pinned(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
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
    path = tmp_path / "ids.json"
    new_key = _key()
    with _create(path, _key()):
        pass
    _resign(path, new_key)  # a valid signature by an unpinned key: opens, logging the signer
    with caplog.at_level(logging.INFO, logger="httk.store.id_ledger"):
        IdLedger.open(path).close()
    message = next(record.getMessage() for record in caplog.records if "audit record" in record.getMessage())
    assert _fingerprint(new_key) in message


def test_open_rejects_base_map_and_series_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    key = _key()
    # A properly-signed ledger whose entry id uses a base/series that disagree
    # with the declared subject: only open()'s own validation can catch this.
    records = [{"key": "k1", "family": "structures", "id": "other.base-z-1"}]
    _hand_seal(path, records, key)
    with pytest.raises(IdLedgerError, match="does not match the declared base"):
        IdLedger.open(path, keys=[key])


def test_open_rejects_wrong_kind(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    key = _key()
    body = build_seal_body("project", {"project_id": "x"}, [])
    write_seal(path, body, [key])
    with pytest.raises(IdLedgerError, match="not 'httk-idledger'"):
        IdLedger.open(path, keys=[key])


# -- no-op close and round trip ----------------------------------------------


def test_noop_close_leaves_file_byte_identical(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    with _create(path, _key()) as ledger:
        ledger.assign("k1", "structures")
    before = path.read_bytes()
    # Reopen, assign the SAME key (idempotent, no state change), then close.
    with IdLedger.open(path, keys=[_key()]) as ledger:
        ledger.assign("k1", "structures")
    assert path.read_bytes() == before


def test_reopen_and_extend_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
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
    path = tmp_path / "ids.json"
    lock = tmp_path / "ids.json.lock"
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


def _hand_seal(path: Path, records: list[dict[str, object]], key: SealKey) -> None:
    """Write a fully-signed ledger with the given records and standard subject."""

    subject = {"ledger_format_version": 1, "bases": BASES, "series": SERIES}
    body = build_seal_body(LEDGER_KIND, subject, records)
    write_seal(path, body, [key])


def _resign(path: Path, key: SealKey) -> None:
    """Re-sign a ledger's existing body under a different key, keeping content."""

    seal = json.loads(path.read_text())
    body = build_seal_body(seal["kind"], seal["subject"], seal["records"])
    body["created_at"] = seal["created_at"]  # preserve content verbatim
    write_seal(path, body, [key])


# -- supersession (split / merge regrouping) ---------------------------------


def test_split_transition_alias_supersede_assign(tmp_path: Path) -> None:
    key = _key()
    path = tmp_path / "ids.json"
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
    path = tmp_path / "ids.json"
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
    with _create(tmp_path / "ids.json", _key()) as ledger:
        ledger.assign("k1", "structures")
        with pytest.raises(IdLedgerError, match="never an assignment to another assignment"):
            ledger.assign("k1", "structures", supersede=True)


def test_default_false_errors_unchanged(tmp_path: Path) -> None:
    # supersede defaults to False: aliased-key assign and assigned-key alias
    # both raise exactly as before the feature existed.
    with _create(tmp_path / "ids.json", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        with pytest.raises(IdLedgerError, match="is an alias of"):
            ledger.assign("dup", "structures")
        with pytest.raises(IdLedgerError, match="already assigned"):
            ledger.alias("owner", target + "x")


def test_open_rejects_forged_supersedes_chain(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
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
    path = tmp_path / "ids.json"
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
    path = tmp_path / "ids.json"
    with _create(path, key) as ledger:
        owner_id = ledger.assign("owner", "structures")  # -1
        ledger.assign("moved", "structures")  # -2, will be orphaned
        ledger.alias("moved", owner_id, supersede=True)  # merge; -2 reserved
        nxt = ledger.assign("new", "structures")
        assert nxt == "anyt.am.structure-a-3"  # -2 still counted, not reused


def test_supersede_on_new_key_assign_errors(tmp_path: Path) -> None:
    with (
        _create(tmp_path / "ids.json", _key()) as ledger,
        pytest.raises(IdLedgerError, match="broken regrouping upstream"),
    ):
        ledger.assign("brand_new", "structures", supersede=True)


def test_supersede_on_new_key_alias_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        with pytest.raises(IdLedgerError, match="broken regrouping upstream"):
            ledger.alias("brand_new", target, supersede=True)


# -- review-hardening guards -------------------------------------------------


def test_create_rechecks_existence_under_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a concurrent create finishing (and releasing its lock) in the
    # window between the existence check and lock acquisition: the racer writes
    # the ledger while we take the lock, and create must then refuse.
    import httk.store.id_ledger as mod

    path = tmp_path / "ids.json"
    real_acquire = mod._acquire_lock

    def racing_acquire(lock_path: Path) -> None:
        real_acquire(lock_path)
        path.write_text("{}")  # a concurrent creator's ledger appears

    monkeypatch.setattr(mod, "_acquire_lock", racing_acquire)
    with pytest.raises(IdLedgerError, match="already exists"):
        IdLedger.create(path, bases=BASES, series=SERIES, keys=[_key()])
    assert not (tmp_path / "ids.json.lock").exists()  # lock released on refusal


def test_create_rejects_duplicate_bases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shared by more than one family"):
        IdLedger.create(
            tmp_path / "ids.json",
            bases={"a": "anyt.am.thing", "b": "anyt.am.thing"},
            series=SERIES,
            keys=[_key()],
        )


def test_open_rejects_duplicate_bases(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    key = _key()
    subject = {"ledger_format_version": 1, "bases": {"a": "anyt.am.thing", "b": "anyt.am.thing"}, "series": SERIES}
    body = build_seal_body(LEDGER_KIND, subject, [])
    write_seal(path, body, [key])
    with pytest.raises(IdLedgerError, match="shares id base"):
        IdLedger.open(path, keys=[key])


def test_open_rejects_base_map_mismatch_vs_expected(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    key = _key()
    with _create(path, key):
        pass
    with pytest.raises(IdLedgerError, match="not the expected"):
        IdLedger.open(path, keys=[key], bases={"structures": "anyt.am.other", "refs": "anyt.am.ref"})


def test_open_rejects_series_mismatch_vs_expected(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    key = _key()
    with _create(path, key):
        pass
    with pytest.raises(IdLedgerError, match="not the expected"):
        IdLedger.open(path, keys=[key], series="z")


def test_open_accepts_matching_bases_and_series(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    key = _key()
    with _create(path, key):
        pass
    with IdLedger.open(path, keys=[key], bases=BASES, series=SERIES) as ledger:
        assert ledger.assign("k1", "structures")


def test_alias_supersede_of_aliased_key_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        target = ledger.assign("owner", "structures")
        ledger.alias("dup", target)
        with pytest.raises(IdLedgerError, match="already an alias"):
            ledger.alias("dup", target, supersede=True)


def test_alias_supersede_self_errors(tmp_path: Path) -> None:
    with _create(tmp_path / "ids.json", _key()) as ledger:
        own = ledger.assign("k1", "structures")
        with pytest.raises(IdLedgerError, match="cannot supersede its own id"):
            ledger.alias("k1", own, supersede=True)


def test_close_retries_after_failed_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key()
    path = tmp_path / "ids.json"
    lock = tmp_path / "ids.json.lock"
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
