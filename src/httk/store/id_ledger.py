"""A signed, append-only id ledger mapping stable source keys to entry ids.

The ledger is an *allocator*: it maps a stable, opaque **source key** to a
recommended entry id (``httk.core.entry_ids``) and hands the same id back for
that key forever, so a database rebuilt from the same sources keeps its ids and
content changes become revisions rather than fresh entries.

The on-disk container is a single stdlib-``sqlite3`` database file (not one of
httk-store's own store engines). Its ``meta`` table records the container format
and the id series; its ``records`` table holds the ordered ledger records, each a
``{key, family, id}`` assignment or a ``{key, alias_of}`` alias, either optionally
carrying ``supersedes`` (see below), keyed by a 1-based append sequence; and its
``segments`` table records, one row per append, the contiguous range of records
that append added together with the canonical seal-body signature over exactly
that segment (``httk.core.project.sealing``: ``kind="httk-idledger-segment"``,
its ``subject`` carrying the format version, this ledger's uuid, the id series,
the full per-family id bases at that point, and the segment's number and record
range). The uuid binds every segment to this ledger, so a segment cannot be
grafted from another ledger that happens to share a series and bases. Each close
that appends signs only the segment it added, so old segments are never rewritten.

Signatures attest to the bytes, never to their meaning, so :meth:`IdLedger.open`
validates the container's structure and every entry invariant itself. A segment
signature is an *audit record*, not a build gate: it is logged (naming the
signers) and inspected manually alongside git history, never demanded. The
integrity self-check is always on — an INVALID signature (content that no longer
matches its own segment: tamper, corruption, a hand-edit) always raises. Trust
enforcement is opt-in: with ``trusted_keys`` every segment's signer must be one
of them; without them a valid signature is accepted and the signers merely noted.
The segments partition the records exactly, so a middle-record deletion,
renumbering, or any record/segment range mismatch is caught in-file by the
partition check even without verification. What no in-file check can attest is
the *tip*: removing the newest segment(s) together with exactly the records they
cover leaves a state indistinguishable from an older valid ledger (as with any
whole-file rollback), so that class of loss is witnessed by git history alone.
The recovery for a corrupted ledger is to restore it from git; the errors say so.

No record is ever edited or removed. An entry for a source that later disappears
simply persists and keeps its number. A key is re-bound only by *supersession*:
appending a fresh record that carries ``supersedes=<the id the key resolved to>``.
Supersession exists to track source-data **regrouping** — when the store's
content deduplication splits one shared-content group into several, or merges
several into one — and is driven deliberately by build scripts. It is never a
recovery path for a lost or corrupted ledger; that is restore-from-git. The
newest record for a key wins; a superseded id stays reserved (counters only
grow) and stays resolvable through whatever key still points at it, becoming a
harmless orphan when none does.
"""

import json
import logging
import os
import socket
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import NamedTuple, Self, cast

from httk.core._json import json_bytes
from httk.core.entry_ids import ENTRY_ID_PATTERN, format_entry_id, is_url_safe_id, parse_entry_id
from httk.core.project.sealing import (
    INVALID,
    VALID_TRUSTED,
    SealError,
    build_seal_body,
    sign_seal_body,
    verify_signed_body,
)

__all__ = [
    "IdLedger",
    "IdLedgerError",
    "LedgerBinding",
    "check_ledger_key",
]

_LOGGER = logging.getLogger(__name__)

#: The seal ``kind`` every id ledger segment carries.
LEDGER_KIND = "httk-idledger-segment"

#: The sqlite container format the meta table records.
LEDGER_FORMAT = "httk-idledger-sqlite"

#: The ledger subject format the invariants below are written against.
LEDGER_FORMAT_VERSION = 1


class IdLedgerError(RuntimeError):
    """An id ledger cannot be opened, verified, locked, or extended.

    This covers both the *cannot proceed* cases — a held lock, a signature or
    invariant that does not verify — and the API-misuse cases — assigning an
    aliased key without superseding, or aliasing a key to an id absent from the
    ledger.
    """


def _live_id(record: Mapping[str, str]) -> str:
    """Return the id a record binds its key to: an assignment's id or an alias target."""

    return record["id"] if "id" in record else record["alias_of"]


class LedgerBinding(NamedTuple):
    """One key's current live binding, as returned by :meth:`IdLedger.bindings`.

    A plain, immutable tuple: enough to group by family and to tell an intrinsic
    assignment from a binding alias, without exposing the underlying record shape.

    :param id: The id this key currently resolves to (an alias's target, exactly
        as :meth:`IdLedger.lookup` would return).
    :param family: The id family the resolved id belongs to.
    :param is_alias: Whether this key is an alias of another key's id, as opposed
        to the assignment that established the id.
    """

    id: str
    family: str
    is_alias: bool


def check_ledger_key(key: str) -> str:
    """Check a ledger source key, warning for deviations but never rejecting.

    Ledger keys are opaque to the allocator: the standardized grammar is a
    convention enforced only by the workflow helpers, so this mirrors
    ``httk.core.entry_ids.check_entry_id`` in stance but, unlike it, never
    raises — it only warns on leading/trailing whitespace or non-URL-safe
    characters.

    :param key: The source key to check.
    :return: The unchanged key.
    """

    if key != key.strip():
        _LOGGER.warning("Ledger key %r has leading or trailing whitespace.", key, extra={"context": "store"})
    elif not is_url_safe_id(key):
        _LOGGER.warning(
            "Ledger key %r is not URL-safe (non-empty printable ASCII without '/').", key, extra={"context": "store"}
        )
    return key


class IdLedger:
    """A signed, append-only allocator of stable entry ids for source keys.

    Open one with :meth:`create` or :meth:`open` and use it as a context
    manager; the enclosing ``with`` holds an exclusive lock and, on exit, appends
    and signs one new segment only when something was assigned or aliased.
    Callers use those constructors rather than the initializer, which binds an
    already-validated state to its file and lock.

    :param path: The ledger sqlite database path.
    :param bases: The per-family id bases, keyed by family name.
    :param series: The id series every minted id carries.
    :param ledger_uuid: This ledger's identity, stamped into every segment subject.
    :param keys: The signing keys used to sign the next segment on close, each a
        ``(role, seed)`` pair.
    :param records: The ordered ledger records, each a ``{key, ...}`` mapping.
    :param live: The newest record per key, the key's live binding.
    :param persisted: How many records are already stored on disk.
    :param segments: How many segments are already stored on disk.
    :param read_only: Whether this ledger was opened read-only: mutators raise
        and :meth:`close` is a no-op that never writes and takes no lock.
    """

    def __init__(
        self,
        path: Path,
        *,
        bases: dict[str, str],
        series: str,
        ledger_uuid: str,
        keys: Sequence[tuple[str, bytes]],
        records: list[dict[str, str]],
        live: dict[str, dict[str, str]],
        persisted: int,
        segments: int,
        read_only: bool = False,
    ) -> None:
        self._path = path
        self._lock_path = path.with_name(path.name + ".lock")
        self._bases = bases
        self._series = series
        self._ledger_uuid = ledger_uuid
        self._keys = tuple(keys)
        self._records = records
        self._live = live
        self._persisted = persisted
        self._segments = segments
        self._read_only = read_only
        self._dirty = False
        self._closed = False

    # -- construction --------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        *,
        bases: Mapping[str, str],
        series: str,
        keys: Sequence[tuple[str, bytes]],
    ) -> "IdLedger":
        """Create and sign a fresh, empty ledger, then hold it open and locked.

        The fresh database carries segment 1: an empty segment (no records) whose
        signed subject stamps the initial bases, so the ledger is signed from
        birth.

        :param path: Where to write the ledger database; it must not exist.
        :param bases: The explicit per-family id bases, e.g.
            ``{"structures": "anyt.am.structure"}``.
        :param series: The id series token every minted id carries.
        :param keys: The signing keys, each a ``(role, seed)`` pair.
        :return: The open, locked ledger.
        :raises IdLedgerError: If the ledger already exists or the lock is held.
        :raises ValueError: If a base or the series is malformed.
        """

        location = Path(path)
        checked = _validate_bases(bases, series)
        _acquire_lock(location.with_name(location.name + ".lock"))
        try:
            # Re-check existence under the lock: checking before acquiring races a
            # concurrent create that could finish (and release its lock) in between,
            # letting this call overwrite its ledger with an empty one.
            if location.exists():
                raise IdLedgerError(f"id ledger already exists: {location}; open it instead of creating it")
            ledger = cls(
                location,
                bases=checked,
                series=series,
                ledger_uuid=uuid.uuid4().hex,
                keys=keys,
                records=[],
                live={},
                persisted=0,
                segments=0,
            )
            try:
                ledger._write()  # writes the schema, meta, and the signed empty segment 1
            except BaseException:
                # Only our own write can leave a partial file here (the existence
                # check above already refused a foreign ledger, which we must not
                # delete). A failed write — e.g. schema laid down before an
                # interrupted segment insert — must not leave a partial ledger, nor
                # a stale hot journal beside it; a hard power loss mid-create still
                # leaves the lock held for manual cleanup, which is accepted.
                for suffix in ("", "-journal", "-wal", "-shm"):
                    location.with_name(location.name + suffix).unlink(missing_ok=True)
                raise
        except BaseException:
            location.with_name(location.name + ".lock").unlink(missing_ok=True)
            raise
        return ledger

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        keys: Sequence[tuple[str, bytes]] = (),
        trusted_keys: Sequence[str] = (),
        verify: bool = True,
        bases: Mapping[str, str] | None = None,
        series: str | None = None,
        read_only: bool = False,
    ) -> "IdLedger":
        """Open an existing ledger, verifying its signatures and invariants.

        With *verify*, every segment's signature is checked: an INVALID signature
        (content that no longer matches its own segment) always raises, while a
        valid one is treated as an audit record — *trusted_keys* demand a trusted
        signer on every segment, and without them the valid signers are logged.
        The structure and every entry invariant are validated regardless (segment
        ranges must partition the records exactly, bases must grow monotonically,
        ids must conform, supersession must be sound), because a signature
        attests only to the bytes, not to their meaning. None of this is weakened
        by *read_only*.

        When *series* is given it must equal the stored series. When *bases* is
        given it is reconciled as a SUPERSET of the stored map: every stored
        family must appear in it with the same base (removing, renaming, or
        re-basing a stored family is an error), while families present only in the
        expectation are ADDED and stamped into the next segment's subject at close
        (so a build that assigns nothing but grows the scheme still writes on
        close). The merged map is revalidated for id shape and base uniqueness.
        With *read_only*, stamping that growth is impossible (there is no write),
        so an expectation that would add a family is refused outright rather than
        silently appearing to have taken effect; a stored map that is a subset of
        the expectation with no addition needed (i.e. an exact match) still opens.

        With *read_only*, no lock is taken or required — this succeeds even while
        another process holds the write lock, so the caller may observe a snapshot
        that a concurrent writer is about to extend — and the returned ledger
        permits no mutation: :meth:`assign`, :meth:`alias`, and any other mutator
        raise, and :meth:`close` is a no-op that neither writes nor requires a
        signing key.

        :param path: The ledger database to open.
        :param keys: The signing keys used to sign the next segment on close.
        :param trusted_keys: Trust anchors as ``ed25519:`` keys or ``sha256:``
            fingerprints; when given, every segment's signer must be one of them.
        :param verify: Whether to verify the segment signatures.
        :param bases: The per-family bases the caller expects, asserted when given.
        :param series: The id series the caller expects, asserted when given.
        :param read_only: Whether to open without a lock and without permitting
            mutation, for a concurrent-safe read path.
        :return: The open ledger, locked unless *read_only*.
        :raises IdLedgerError: If the ledger is missing, the lock is held (and
            *read_only* is false), verification fails, an invariant is violated,
            the expected bases/series disagree, or *read_only* is true and the
            expected bases would require growing the stored map.
        """

        location = Path(path)
        lock_path = location.with_name(location.name + ".lock")
        if not read_only:
            _acquire_lock(lock_path)
        try:
            # sqlite3.connect creates a missing file, so existence is checked here,
            # under the lock, before any connection is opened.
            if not location.exists():
                raise IdLedgerError(f"id ledger does not exist: {location}. Restore it from git.")
            stored_bases, stored_series, stored_uuid, records, live, segments = _read_and_validate(
                location, verify=verify, trusted_keys=trusted_keys
            )
            if series is not None and series != stored_series:
                raise IdLedgerError(f"id ledger {location} has series {stored_series!r}, not the expected {series!r}")
            effective_bases = stored_bases
            added_families = False
            if bases is not None:
                effective_bases, added_families = _extend_bases(stored_bases, dict(bases), stored_series, location)
                if added_families and read_only:
                    missing = sorted(set(bases) - set(stored_bases))
                    raise IdLedgerError(
                        f"id ledger {location} is missing families {missing} from the expected base map; opening "
                        "read-only cannot stamp that growth. Open for write instead, or drop them from bases="
                    )
            ledger = cls(
                location,
                bases=effective_bases,
                series=stored_series,
                ledger_uuid=stored_uuid,
                keys=keys,
                records=records,
                live=live,
                persisted=len(records),
                segments=segments,
                read_only=read_only,
            )
            # An added family is stamped into the next segment's subject at close; a
            # build that assigns nothing but extends the base map still writes.
            # (added_families is always false here when read_only, per the raise above.)
            ledger._dirty = added_families
            return ledger
        except BaseException:
            if not read_only:
                lock_path.unlink(missing_ok=True)
            raise

    # -- allocation ----------------------------------------------------------

    def assign(self, key: str, family: str, *, supersede: bool = False) -> str:
        """Return the id for a source key, minting one on first sight.

        The call is idempotent: a key already assigned in this family returns
        its existing id. A key assigned in another family is an error, and an
        aliased key is an error unless *supersede* re-binds it.

        With ``supersede=True`` on a currently-**aliased** key, a fresh
        assignment is appended carrying ``supersedes=<the alias target>`` and a
        newly minted id — the split half of source regrouping. ``supersede=True``
        on an already-assigned key is an error: an assignment is never re-bound
        to another assignment.

        :param key: The stable source key.
        :param family: The id family the key belongs to.
        :param supersede: Whether to re-bind an aliased key to a fresh assignment.
        :return: The assigned entry id.
        :raises IdLedgerError: If the ledger is read-only, the key is aliased and
            *supersede* is false, its family disagrees, the family has no base, or
            *supersede* is passed outside its one transition (a new key, or an
            already-assigned key).
        """

        self._require_open()
        live = self._live.get(key)
        if live is None:
            if supersede:
                raise IdLedgerError(
                    f"key {key!r} is new; supersede re-binds an existing alias, so passing it here signals a "
                    "broken regrouping upstream"
                )
            return self._mint(key, family)
        if "id" in live:
            if supersede:
                raise IdLedgerError(
                    f"key {key!r} is already assigned id {live['id']!r}; supersession re-binds an alias to a "
                    "fresh assignment, never an assignment to another assignment"
                )
            if live["family"] != family:
                raise IdLedgerError(f"key {key!r} is already assigned in family {live['family']!r}, not {family!r}")
            return live["id"]
        if not supersede:
            raise IdLedgerError(f"key {key!r} is an alias of {live['alias_of']!r} and cannot be assigned")
        return self._mint(key, family, supersedes=live["alias_of"])

    def alias(self, key: str, existing_id: str, *, supersede: bool = False) -> None:
        """Record a source key pointing at an id already in the ledger.

        Aliases exist because the store deduplicates content-identical rows
        family-wide: several keys can reach one row, which must keep one id. The
        call is idempotent for an identical ``(key, existing_id)`` pair; a
        conflicting re-alias, or targeting an id absent from the ledger, is an
        error, and so is aliasing an already-assigned key unless *supersede*
        re-binds it.

        With ``supersede=True`` on a currently-**assigned** key, an alias record
        is appended carrying ``supersedes=<its old id>`` — the merge half of
        source regrouping. The old id stays reserved and stays resolvable while
        any other key points at it, otherwise becoming a harmless orphan.

        :param key: The stable source key to record.
        :param existing_id: An id already assigned in the ledger.
        :param supersede: Whether to re-bind an assigned key to an alias.
        :raises IdLedgerError: If the ledger is read-only, on a conflict, an
            unknown target, an assigned key when *supersede* is false, or
            *supersede* passed for a new key.
        """

        self._require_open()
        live = self._live.get(key)
        if live is not None and "id" in live:
            if not supersede:
                raise IdLedgerError(f"key {key!r} is already assigned id {live['id']!r} and cannot be aliased")
            if existing_id == live["id"]:
                raise IdLedgerError(f"key {key!r} cannot supersede its own id {existing_id!r} with an alias to itself")
            self._require_target(existing_id, key)
            self._append({"key": key, "alias_of": existing_id, "supersedes": live["id"]})
            return
        if live is not None:
            if supersede:
                raise IdLedgerError(
                    f"key {key!r} is already an alias; supersede re-binds an assignment, so passing it here "
                    "signals a broken regrouping upstream"
                )
            if live["alias_of"] != existing_id:
                raise IdLedgerError(f"key {key!r} is already aliased to {live['alias_of']!r}, not {existing_id!r}")
            return
        if supersede:
            raise IdLedgerError(
                f"key {key!r} is new; supersede re-binds an existing assignment, so passing it here signals a "
                "broken regrouping upstream"
            )
        self._require_target(existing_id, key)
        self._append({"key": key, "alias_of": existing_id})

    def lookup(self, key: str) -> str | None:
        """Return the id a source key resolves to, following an alias.

        Resolution is to the key's newest record, so a superseded binding is
        never returned for the key that moved on from it.

        :param key: The source key to resolve.
        :return: The assigned id, or ``None`` when the key is unknown.
        """

        live = self._live.get(key)
        return None if live is None else _live_id(live)

    def bindings(self) -> Mapping[str, LedgerBinding]:
        """Return every key's current live binding, aliases resolved to their target id.

        Only live bindings appear here: a superseded record is as invisible to
        this as it is to :meth:`lookup` for the key that moved on from it. Each
        binding names the family its id belongs to (an alias's family is that of
        the id it resolves to, since a bare alias record carries no family of its
        own) and whether the key is an alias or the assignment that established
        the id — enough for a caller to group by family and separate intrinsic
        identity from binding, without exposing the record shape itself.

        The result is a snapshot, independent of the ledger's internals: it is a
        fresh mapping of immutable values, so neither mutating it nor a later
        :meth:`assign`/:meth:`alias` on this ledger can affect it, in either
        direction. Works the same whether the ledger was opened for write or
        *read_only*.

        :return: The live bindings, keyed by source key, in ascending key order
            (a fixed, deterministic order independent of append order).
        """

        id_families: dict[str, str] = {}
        for record in self._records:
            if "id" in record:
                id_families[record["id"]] = record["family"]
        out: dict[str, LedgerBinding] = {}
        for key in sorted(self._live):
            record = self._live[key]
            resolved = _live_id(record)
            is_alias = "alias_of" in record
            family = id_families[resolved] if is_alias else record["family"]
            out[key] = LedgerBinding(id=resolved, family=family, is_alias=is_alias)
        return MappingProxyType(out)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Append and sign a new segment when the ledger changed, then unlock.

        A ledger untouched since it was opened is left byte-identical: no
        connection is opened and nothing is written, so an idempotent rebuild
        produces no git churn. A dirty close with no signing key available raises
        from the sealing layer before anything is written; a failed close leaves
        the ledger unclosed and its lock held, so a retried close reattempts the
        write instead of silently skipping it (the manual remedy for an abandoned
        session is to delete the lock).

        A read-only ledger never took the lock, so this is purely a no-op: it
        neither writes nor touches any lock file (removing one here could delete
        a concurrent writer's *own* lock, since a read-only open may coexist with
        one).
        """

        if self._closed:
            return
        if self._read_only:
            self._closed = True
            return
        if self._dirty:
            self._write()  # a raise here keeps _closed False and the lock held for a retry
        self._closed = True
        self._lock_path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        """Enter the ledger context.

        :return: This ledger.
        """

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the ledger on context exit.

        :param exc_type: The exception type, if the block raised.
        :param exc: The exception, if the block raised.
        :param traceback: The traceback, if the block raised.
        """

        self.close()

    # -- internals -----------------------------------------------------------

    def _require_open(self) -> None:
        """Guard against mutating a closed or read-only ledger."""

        if self._closed:
            raise IdLedgerError(f"id ledger is closed: {self._path}")
        if self._read_only:
            raise IdLedgerError(f"id ledger is open read-only: {self._path}; assign/alias need a write-mode open")

    def _require_target(self, existing_id: str, key: str) -> None:
        """Guard that an alias target is an id the ledger has assigned."""

        if existing_id not in self._assigned_ids():
            raise IdLedgerError(f"cannot alias {key!r}: id {existing_id!r} is not assigned in the ledger")

    def _mint(self, key: str, family: str, *, supersedes: str | None = None) -> str:
        """Mint the next id in a family and append its assignment record."""

        base = self._bases.get(family)
        if base is None:
            raise IdLedgerError(
                f"no id base is configured for family {family!r}; known families: {sorted(self._bases)}"
            )
        entry_id = format_entry_id(base, self._series, self._next_number(family) + 1)
        record = {"key": key, "family": family, "id": entry_id}
        if supersedes is not None:
            record["supersedes"] = supersedes
        self._append(record)
        return entry_id

    def _append(self, record: dict[str, str]) -> None:
        """Append a record and make it the key's live binding."""

        self._records.append(record)
        self._live[record["key"]] = record
        self._dirty = True

    def _assigned_ids(self) -> set[str]:
        """Return every id ever assigned, superseded ones included (all reserved)."""

        return {record["id"] for record in self._records if "id" in record}

    def _next_number(self, family: str) -> int:
        """Return the largest assigned number in a family, or 0 when none.

        Every assignment record counts, superseded ones included, so a superseded
        id keeps its number reserved; the high-water mark is monotone and
        gap-tolerant.
        """

        highest = 0
        for record in self._records:
            if "id" not in record or record["family"] != family:
                continue
            parsed = parse_entry_id(record["id"])
            if parsed is not None and parsed[2] > highest:
                highest = parsed[2]
        return highest

    def _write(self) -> None:
        """Append and sign one segment covering records not yet persisted.

        The segment's signed subject carries the full current base map, so a
        base-extension-only close still writes an empty segment stamping the grown
        scheme. On the first write (creation) the schema and meta rows are laid
        down before segment 1. The signing key is required before any connection
        is opened, so a keyless dirty close leaves the file untouched.
        """

        if not self._keys:
            raise SealError("no signing key is available to sign the id ledger segment")
        new_records = self._records[self._persisted :]
        first_seq = self._persisted + 1
        segment = self._segments + 1
        subject: dict[str, object] = {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "ledger": self._ledger_uuid,
            "series": self._series,
            "bases": dict(self._bases),
            "segment": segment,
            "first_record": first_seq,
            "record_count": len(new_records),
        }
        signed_records = cast("list[dict[str, object]]", [dict(record) for record in new_records])
        body = build_seal_body(LEDGER_KIND, subject, signed_records)
        body_sha256, signatures = sign_seal_body(body, self._keys)
        connection = sqlite3.connect(self._path, isolation_level=None)
        try:
            if self._segments == 0:
                _initialize_schema(connection, self._series, self._ledger_uuid)
            connection.execute("BEGIN IMMEDIATE")
            for offset, record in enumerate(new_records):
                connection.execute(
                    "INSERT INTO records(seq, key, family, id, alias_of, supersedes) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        first_seq + offset,
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
                    segment,
                    first_seq,
                    len(new_records),
                    str(body["created_at"]),
                    json_bytes(body["subject"]).decode("utf-8"),
                    body_sha256,
                    json_bytes(signatures).decode("utf-8"),
                ),
            )
            connection.execute("COMMIT")
            # Advance bookkeeping the instant the data is durable: an interrupt
            # after COMMIT but before this leaves committed records that a
            # close-retry would re-insert (a raw records-PK IntegrityError with the
            # lock held forever), so it lives inside the try, right after COMMIT.
            self._persisted += len(new_records)
            self._segments = segment
            self._dirty = False
        finally:
            connection.close()


# -- schema ------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE records (
  seq INTEGER PRIMARY KEY,
  key TEXT NOT NULL,
  family TEXT,
  id TEXT,
  alias_of TEXT,
  supersedes TEXT,
  CHECK ((family IS NOT NULL AND id IS NOT NULL AND alias_of IS NULL)
      OR (family IS NULL AND id IS NULL AND alias_of IS NOT NULL))
);
CREATE TABLE segments (
  segment INTEGER PRIMARY KEY,
  first_seq INTEGER NOT NULL,
  record_count INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  subject TEXT NOT NULL,
  body_sha256 TEXT NOT NULL,
  signatures TEXT NOT NULL
);
"""


def _initialize_schema(connection: sqlite3.Connection, series: str, ledger_uuid: str) -> None:
    """Lay down the ledger schema and its meta rows on a fresh database."""

    connection.executescript(_SCHEMA)
    connection.executemany(
        "INSERT INTO meta(key, value) VALUES (?, ?)",
        [
            ("format", LEDGER_FORMAT),
            ("ledger_format_version", str(LEDGER_FORMAT_VERSION)),
            ("ledger", ledger_uuid),
            ("series", series),
        ],
    )


# -- validation helpers ------------------------------------------------------


def _validate_bases(bases: Mapping[str, str], series: str) -> dict[str, str]:
    """Validate that every family base plus the series forms a well-formed id.

    :param bases: The per-family bases to validate.
    :param series: The id series to validate.
    :return: The bases as a plain dict.
    :raises ValueError: If a base or the series is malformed.
    """

    if not bases:
        raise ValueError("an id ledger needs at least one family base")
    checked: dict[str, str] = {}
    for family, base in bases.items():
        # format_entry_id validates the base+series shape via ENTRY_ID_PATTERN.
        format_entry_id(base, series, 1)
        checked[str(family)] = str(base)
    duplicate = _duplicate_base(checked)
    if duplicate is not None:
        raise ValueError(f"id base {duplicate!r} is shared by more than one family; bases must be unique per family")
    return checked


def _extend_bases(
    stored: Mapping[str, str], expected: Mapping[str, str], series: str, location: Path
) -> tuple[dict[str, str], bool]:
    """Reconcile a stored base map against a caller's expectation (superset-open).

    Families the caller expects that are absent from the stored map are ADDED
    (stamped at the next segment); every stored family must appear in the
    expectation with the SAME base (a stored family is never removed, renamed, or
    re-based). The merged map is revalidated (id-shape + no duplicate base across
    families) before it is accepted.

    :param stored: The base map read from the ledger.
    :param expected: The base map the caller passed to :meth:`IdLedger.open`.
    :param series: The stored series, used to revalidate added bases' id shape.
    :param location: The ledger path, for error messages.
    :return: The merged base map and whether any family was added.
    :raises IdLedgerError: If a stored family is missing from or re-based by the
        expectation.
    :raises ValueError: If the merged map is malformed or shares a base.
    """

    for family, base in stored.items():
        if family not in expected:
            raise IdLedgerError(
                f"id ledger {location} has stored family {family!r} absent from the expected base map "
                f"{dict(expected)}, not the expected: a stored family is never removed or renamed"
            )
        if expected[family] != base:
            raise IdLedgerError(
                f"id ledger {location} has base {base!r} for family {family!r}, not the expected {expected[family]!r}"
            )
    added = {family: base for family, base in expected.items() if family not in stored}
    if not added:
        return dict(stored), False
    merged = {**stored, **added}
    _validate_bases(merged, series)
    return merged, True


def _duplicate_base(bases: Mapping[str, str]) -> str | None:
    """Return a base value used by more than one family, or ``None`` when all differ.

    Duplicate bases across families are the enabling condition for cross-family id
    collisions, so both create and open-validation reject them.

    :param bases: The per-family bases to check.
    :return: A duplicated base value, or ``None``.
    """

    seen: set[str] = set()
    for base in bases.values():
        if base in seen:
            return base
        seen.add(base)
    return None


def _acquire_lock(lock_path: Path) -> None:
    """Create the ledger lock exclusively, refusing when it is already held.

    :param lock_path: The lock file to create.
    :raises IdLedgerError: If the lock already exists (no auto-reclaim).
    """

    body = json_bytes({"created": _utc_now(), "hostname": socket.gethostname(), "pid": os.getpid()}) + b"\n"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        holder = _describe_lock(lock_path)
        raise IdLedgerError(
            f"id ledger lock is already held: {lock_path} ({holder}); another build may be running. "
            f"If none is, delete {lock_path} to release it (no automatic reclaim is done)."
        ) from None
    try:
        os.write(descriptor, body)
    finally:
        os.close(descriptor)


def _describe_lock(lock_path: Path) -> str:
    """Describe a held lock's recorded holder for the refusal message."""

    try:
        holder = json.loads(lock_path.read_bytes())
    except (OSError, ValueError):
        return "holder unknown"
    if not isinstance(holder, dict):
        return "holder unknown"
    return f"pid={holder.get('pid')} host={holder.get('hostname')!r} created={holder.get('created')!r}"


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z``."""

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_and_validate(
    location: Path, *, verify: bool, trusted_keys: Sequence[str]
) -> tuple[dict[str, str], str, str, list[dict[str, str]], dict[str, dict[str, str]], int]:
    """Read a ledger database and validate its structure and every entry invariant.

    :param location: The ledger database to read.
    :param verify: Whether to verify each segment's signature.
    :param trusted_keys: Trust anchors; when given, every segment must be trusted.
    :return: The bases, series, ledger uuid, ordered records, live binding per key,
        and the number of segments.
    :raises IdLedgerError: If the file is not an httk-idledger container, the meta,
        segment partition, base chain, or any record is malformed, an id does not
        parse under its declared base and series, a supersession chain is forged,
        or a signature does not verify (or is untrusted when trust is required).
    """

    try:
        connection = sqlite3.connect(location, isolation_level=None)
        try:
            _require_ledger_tables(connection, location)
            meta = _read_meta(connection)
            segment_rows = connection.execute(
                "SELECT segment, first_seq, record_count, created_at, subject, body_sha256, signatures "
                "FROM segments ORDER BY segment"
            ).fetchall()
            record_rows = connection.execute(
                "SELECT seq, key, family, id, alias_of, supersedes FROM records ORDER BY seq"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise IdLedgerError(
            f"id ledger is not a readable sqlite database: {location} ({exc}). Restore it from git."
        ) from exc

    series, ledger_uuid = _validate_meta(meta, location)
    bases = _validate_segments(
        segment_rows, record_rows, series, ledger_uuid, location, verify=verify, trusted_keys=trusted_keys
    )
    records = [_row_to_record(row) for row in record_rows]
    validated_records, live = _validate_entries(records, bases, series, location)
    return bases, series, ledger_uuid, validated_records, live, len(segment_rows)


def _require_ledger_tables(connection: sqlite3.Connection, location: Path) -> None:
    """Reject a readable sqlite file that is not an httk-idledger container.

    This distinguishes "someone pointed this at a store database" from a genuinely
    corrupt file: the tables are missing rather than the bytes unreadable, so no
    restore-from-git advice is given.

    :param connection: An open connection to the file.
    :param location: The ledger path, for the error message.
    :raises IdLedgerError: If any of the ledger's tables is absent.
    """

    present = {
        str(name) for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    if not {"meta", "records", "segments"} <= present:
        raise IdLedgerError(f"{location} is a sqlite file but not an httk-idledger container")


def _read_meta(connection: sqlite3.Connection) -> dict[str, str]:
    """Read the meta table into a plain dict."""

    return {str(key): str(value) for key, value in connection.execute("SELECT key, value FROM meta").fetchall()}


def _validate_meta(meta: Mapping[str, str], location: Path) -> tuple[str, str]:
    """Validate the container meta and return the stored id series and uuid.

    :param meta: The meta table rows.
    :param location: The ledger path, for error messages.
    :return: The stored id series and ledger uuid.
    :raises IdLedgerError: If the format, version, series, or uuid is missing or unsupported.
    """

    if meta.get("format") != LEDGER_FORMAT:
        raise IdLedgerError(f"{location} is not an {LEDGER_FORMAT} container (format {meta.get('format')!r})")
    if meta.get("ledger_format_version") != str(LEDGER_FORMAT_VERSION):
        raise IdLedgerError(
            f"id ledger {location} has unsupported format version {meta.get('ledger_format_version')!r}"
        )
    series = meta.get("series")
    if not isinstance(series, str) or not series:
        raise IdLedgerError(f"id ledger {location} has a malformed meta series")
    ledger_uuid = meta.get("ledger")
    if not isinstance(ledger_uuid, str) or not ledger_uuid:
        raise IdLedgerError(f"id ledger {location} has a malformed meta ledger uuid")
    return series, ledger_uuid


def _validate_segments(
    segment_rows: Sequence[tuple[object, ...]],
    record_rows: Sequence[tuple[object, ...]],
    series: str,
    ledger_uuid: str,
    location: Path,
    *,
    verify: bool,
    trusted_keys: Sequence[str],
) -> dict[str, str]:
    """Validate the segment partition and base chain, optionally verifying signatures.

    Segments must be numbered 1..M and their ranges must partition the records
    1..N in order (empty segments allowed); every segment subject must match its
    row and carry the meta series and format version; bases must grow
    monotonically (a later segment never removes, renames, or re-bases a family);
    and the last segment's bases are the live base map. With *verify*, each
    segment's signed body is rebuilt and checked, an INVALID verdict raising and,
    when *trusted_keys* is given, every segment being required to be trusted.

    :param segment_rows: The segment table rows, ordered by segment.
    :param record_rows: The record table rows, ordered by seq.
    :param series: The stored id series.
    :param ledger_uuid: The stored ledger uuid every segment subject must carry.
    :param location: The ledger path, for error messages.
    :param verify: Whether to verify each segment's signature.
    :param trusted_keys: Trust anchors; when given, every segment must be trusted.
    :return: The live per-family base map (the last segment's bases).
    :raises IdLedgerError: If the partition, chain, subjects, or signatures are unsound.
    """

    if not segment_rows:
        raise IdLedgerError(f"id ledger {location} has no segments. Restore it from git.")
    prev_bases: dict[str, str] | None = None
    expected_next = 1
    signers: list[str] = []
    for index, row in enumerate(segment_rows, start=1):
        raw_segment, raw_first, raw_count, created_at, subject_json, body_sha256, signatures_json = row
        if not isinstance(raw_segment, int) or not isinstance(raw_first, int) or not isinstance(raw_count, int):
            raise IdLedgerError(f"id ledger {location} has a malformed segment row. Restore it from git.")
        segment, first_seq, record_count = raw_segment, raw_first, raw_count
        if segment != index:
            raise IdLedgerError(
                f"id ledger {location} segment numbering is broken at {segment!r}. Restore it from git."
            )
        if first_seq != expected_next:
            raise IdLedgerError(
                f"id ledger {location} segment {segment} starts at record {first_seq!r}, not the expected "
                f"{expected_next}; segments must partition the records. Restore it from git."
            )
        subject = _parse_segment_subject(subject_json, segment, first_seq, record_count, series, ledger_uuid, location)
        seg_bases = _segment_bases(subject, segment, location)
        if prev_bases is not None:
            for family, base in prev_bases.items():
                if seg_bases.get(family) != base:
                    raise IdLedgerError(
                        f"id ledger {location} segment {segment} removes or re-bases family {family!r}; "
                        "a stored family is never removed, renamed, or re-based. Restore it from git."
                    )
        prev_bases = seg_bases
        if first_seq - 1 + record_count > len(record_rows):
            raise IdLedgerError(
                f"id ledger {location} segment {segment} covers records past the end; a record was deleted "
                "or truncated. Restore it from git."
            )
        if verify:
            segment_records = [_row_to_record(record_rows[first_seq - 1 + offset]) for offset in range(record_count)]
            signers.extend(
                _verify_segment(
                    subject, created_at, segment_records, body_sha256, signatures_json, segment, location, trusted_keys
                )
            )
        expected_next += record_count

    record_seqs = [row[0] for row in record_rows]
    if record_seqs != list(range(1, expected_next)):
        raise IdLedgerError(
            f"id ledger {location} records do not match the segment partition; a record was deleted, "
            "renumbered, or truncated. Restore it from git."
        )

    assert prev_bases is not None  # guarded by the empty-segments check above
    live_bases = _stored_bases(prev_bases, location)
    if verify and not trusted_keys:
        _LOGGER.info(
            "id ledger %s signed by %s (no trust anchor configured; signatures are an audit record).",
            location,
            ", ".join(dict.fromkeys(signers)) or "an unrecorded key",
            extra={"context": "store"},
        )
    return live_bases


def _parse_segment_subject(
    subject_json: object,
    segment: int,
    first_seq: int,
    record_count: int,
    series: str,
    ledger_uuid: str,
    location: Path,
) -> dict[str, object]:
    """Parse a segment subject and check it matches its row, ledger, series, and version."""

    if not isinstance(subject_json, str):
        raise IdLedgerError(f"id ledger {location} segment {segment} has a malformed subject. Restore it from git.")
    try:
        subject = json.loads(subject_json)
    except ValueError as exc:
        raise IdLedgerError(
            f"id ledger {location} segment {segment} subject is not valid JSON ({exc}). Restore it from git."
        ) from exc
    if not isinstance(subject, dict):
        raise IdLedgerError(f"id ledger {location} segment {segment} subject is not an object. Restore it from git.")
    if subject.get("ledger_format_version") != LEDGER_FORMAT_VERSION:
        raise IdLedgerError(
            f"id ledger {location} segment {segment} has unsupported format version "
            f"{subject.get('ledger_format_version')!r}"
        )
    if subject.get("ledger") != ledger_uuid:
        raise IdLedgerError(
            f"id ledger {location} segment {segment} subject is stamped for a different ledger "
            f"({subject.get('ledger')!r}, not {ledger_uuid!r}); a segment was grafted from another ledger. "
            "Restore it from git."
        )
    if subject.get("series") != series:
        raise IdLedgerError(
            f"id ledger {location} segment {segment} subject series {subject.get('series')!r} disagrees with "
            f"the meta series {series!r}. Restore it from git."
        )
    if (
        subject.get("segment") != segment
        or subject.get("first_record") != first_seq
        or subject.get("record_count") != record_count
    ):
        raise IdLedgerError(
            f"id ledger {location} segment {segment} subject range disagrees with its row. Restore it from git."
        )
    return subject


def _segment_bases(subject: Mapping[str, object], segment: int, location: Path) -> dict[str, str]:
    """Extract and type-check a segment subject's base map."""

    raw = subject.get("bases")
    if not isinstance(raw, dict) or not raw:
        raise IdLedgerError(f"id ledger {location} segment {segment} has a malformed base map. Restore it from git.")
    bases: dict[str, str] = {}
    for family, base in raw.items():
        if not isinstance(family, str) or not isinstance(base, str):
            raise IdLedgerError(
                f"id ledger {location} segment {segment} has a malformed base map. Restore it from git."
            )
        bases[family] = base
    return bases


def _stored_bases(bases: Mapping[str, str], location: Path) -> dict[str, str]:
    """Validate the live base map for base uniqueness across families."""

    duplicate = _duplicate_base(bases)
    if duplicate is not None:
        raise IdLedgerError(f"id ledger {location} shares id base {duplicate!r} across families")
    return dict(bases)


def _verify_segment(
    subject: Mapping[str, object],
    created_at: object,
    segment_records: Sequence[dict[str, str]],
    body_sha256: object,
    signatures_json: object,
    segment: int,
    location: Path,
    trusted_keys: Sequence[str],
) -> tuple[str, ...]:
    """Rebuild one segment's signed body and verify its signature.

    :param subject: The parsed segment subject.
    :param created_at: The segment's recorded ``created_at`` stamp.
    :param segment_records: The reconstructed records the segment covers.
    :param body_sha256: The segment's recorded body digest.
    :param signatures_json: The segment's recorded signatures, as JSON text.
    :param segment: The segment number, for error messages.
    :param location: The ledger path, for error messages.
    :param trusted_keys: Trust anchors; when given, the segment must be trusted.
    :return: The verifying signers' fingerprints.
    :raises IdLedgerError: If the signature is INVALID or untrusted when required.
    """

    if not isinstance(signatures_json, str) or not isinstance(body_sha256, str):
        raise IdLedgerError(
            f"id ledger {location} segment {segment} has malformed signature data. Restore it from git."
        )
    try:
        signatures = json.loads(signatures_json)
    except ValueError as exc:
        raise IdLedgerError(
            f"id ledger {location} segment {segment} signatures are not valid JSON ({exc}). Restore it from git."
        ) from exc
    if not isinstance(signatures, list):
        raise IdLedgerError(
            f"id ledger {location} segment {segment} has malformed signature data. Restore it from git."
        )
    body = build_seal_body(LEDGER_KIND, subject, cast("list[dict[str, object]]", [dict(r) for r in segment_records]))
    body["created_at"] = str(created_at)
    verification = verify_signed_body(json_bytes(body), body_sha256, signatures, trusted_keys=tuple(trusted_keys))
    if verification.verdict == INVALID:
        raise IdLedgerError(
            f"id ledger segment {segment} signature does not verify ({verification.reason}): {location}. "
            "The ledger may be corrupted or tampered; restore it from git."
        )
    if trusted_keys and verification.verdict != VALID_TRUSTED:
        raise IdLedgerError(
            f"id ledger segment {segment} is signed by an untrusted key ({verification.reason}): {location}. "
            "Pin the correct signer, or restore the ledger from git."
        )
    return verification.signers


def _row_to_record(row: tuple[object, ...]) -> dict[str, str]:
    """Reconstruct one record dict from a ``records`` table row."""

    _seq, key, family, entry_id, alias_of, supersedes = row
    if alias_of is not None:
        record: dict[str, str] = {"key": str(key), "alias_of": str(alias_of)}
    else:
        record = {"key": str(key), "family": str(family), "id": str(entry_id)}
    if supersedes is not None:
        record["supersedes"] = str(supersedes)
    return record


def _validate_entries(
    records: Sequence[Mapping[str, object]],
    bases: Mapping[str, str],
    series: str,
    location: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    """Validate ledger records in append order: ids conform, supersession is sound.

    Records are processed in order so each ``supersedes`` value can be checked
    against the key's binding at that position: a first record must not supersede,
    a repeat record must supersede exactly the key's prior id, and a repeat with
    no ``supersedes`` is a duplicate live binding. Alias targets are then checked
    against the full set of assigned ids.
    """

    out: list[dict[str, str]] = []
    live: dict[str, dict[str, str]] = {}
    assigned_ids: set[str] = set()
    prior_id: dict[str, str] = {}
    pending_targets: list[tuple[str, str]] = []
    for record in records:
        key = record.get("key")
        if not isinstance(key, str):
            raise IdLedgerError(f"id ledger {location} has a record without a string key")
        supersedes = record.get("supersedes")
        if supersedes is not None and not isinstance(supersedes, str):
            raise IdLedgerError(f"id ledger {location} has a malformed supersedes for key {key!r}")
        if "alias_of" in record:
            target = record.get("alias_of")
            if not isinstance(target, str):
                raise IdLedgerError(f"id ledger {location} has a malformed alias for key {key!r}")
            rec: dict[str, str] = {"key": key, "alias_of": target}
            resolved = target
            pending_targets.append((key, target))
        else:
            family = record.get("family")
            entry_id = record.get("id")
            if not isinstance(family, str) or not isinstance(entry_id, str):
                raise IdLedgerError(f"id ledger {location} has a malformed entry for key {key!r}")
            _validate_entry_id(entry_id, family, bases, series, location)
            if entry_id in assigned_ids:
                raise IdLedgerError(f"id ledger {location} reuses assigned id {entry_id!r}")
            assigned_ids.add(entry_id)
            rec = {"key": key, "family": family, "id": entry_id}
            resolved = entry_id
        _check_supersession(key, supersedes, prior_id.get(key), location)
        if supersedes is not None:
            rec["supersedes"] = supersedes
        prior_id[key] = resolved
        out.append(rec)
        live[key] = rec
    for key, target in pending_targets:
        if target not in assigned_ids:
            raise IdLedgerError(f"id ledger {location} aliases key {key!r} to unknown id {target!r}")
    return out, live


def _check_supersession(key: str, supersedes: str | None, prior: str | None, location: Path) -> None:
    """Validate one record's supersession against the key's binding at its position."""

    if prior is None:
        if supersedes is not None:
            raise IdLedgerError(
                f"id ledger {location} record for key {key!r} supersedes {supersedes!r} but the key had no "
                "prior binding"
            )
        return
    if supersedes is None:
        raise IdLedgerError(f"id ledger {location} has a duplicate live binding for key {key!r}")
    if supersedes != prior:
        raise IdLedgerError(
            f"id ledger {location} record for key {key!r} supersedes {supersedes!r} but its prior binding was {prior!r}"
        )


def _validate_entry_id(entry_id: str, family: str, bases: Mapping[str, str], series: str, location: Path) -> None:
    """Validate one assigned id parses under its family's declared base and series."""

    if ENTRY_ID_PATTERN.fullmatch(entry_id) is None:
        raise IdLedgerError(f"id ledger {location} has non-conforming id {entry_id!r} for key family {family!r}")
    base = bases.get(family)
    if base is None:
        raise IdLedgerError(f"id ledger {location} has entry in undeclared family {family!r}")
    parsed = parse_entry_id(entry_id)
    assert parsed is not None  # guarded by ENTRY_ID_PATTERN above
    if parsed[0] != base or parsed[1] != series:
        raise IdLedgerError(
            f"id ledger {location} id {entry_id!r} does not match the declared base {base!r} and series {series!r}"
        )
