"""A sealed, append-only id ledger mapping stable source keys to entry ids.

The ledger is an *allocator*: it maps a stable, opaque **source key** to a
recommended entry id (``httk.core.entry_ids``) and hands the same id back for
that key forever, so a database rebuilt from the same sources keeps its ids and
content changes become revisions rather than fresh entries.

The whole ledger is one signed seal document (httk-core
``core/project/sealing.py``): ``kind="httk-idledger"``, its ``subject`` carrying
the format version, the explicit per-family id bases, and the id series, and its
``records`` an ordered list carrying ``{key, family, id}`` assignments and
``{key, alias_of}`` aliases, either optionally carrying ``supersedes`` (see
below). It is replaced atomically and durably by ``write_seal`` and read back
through ``read_seal``.

``verify_seal`` checks signatures only, never content, so :meth:`IdLedger.open`
validates the document's structure and every entry invariant itself. The seal's
signature is an *audit record*, not a build gate: it is logged (naming the
signer) and inspected manually alongside git history, never demanded. The
integrity self-check is always on — an INVALID signature (content that no longer
matches its own seal: tamper, corruption, a hand-edit) always raises. Trust
enforcement is opt-in: with ``trusted_keys`` the signer must be one of them;
without them a valid signature is accepted and the signer merely noted. The
recovery for a corrupted ledger is to restore it from git; the verification
errors say so.

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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self, cast

from httk.core.entry_ids import ENTRY_ID_PATTERN, format_entry_id, is_url_safe_id, parse_entry_id
from httk.core.project.sealing import (
    INVALID,
    VALID_TRUSTED,
    build_seal_body,
    read_seal,
    verify_seal,
    write_seal,
)

__all__ = [
    "IdLedger",
    "IdLedgerError",
    "check_ledger_key",
]

_LOGGER = logging.getLogger(__name__)

#: The seal ``kind`` every id ledger carries.
LEDGER_KIND = "httk-idledger"

#: The ledger subject format the invariants below are written against.
LEDGER_FORMAT_VERSION = 1


class IdLedgerError(RuntimeError):
    """An id ledger cannot be opened, verified, locked, or extended.

    This covers both the *cannot proceed* cases — a held lock, a signature or
    invariant that does not verify — and the API-misuse cases — assigning an
    aliased key without superseding, or aliasing a key to an id absent from the
    ledger.
    """


def _json_bytes(value: object) -> bytes:
    """Encode a value as canonical, sorted-key, compact UTF-8 JSON."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _live_id(record: Mapping[str, str]) -> str:
    """Return the id a record binds its key to: an assignment's id or an alias target."""

    return record["id"] if "id" in record else record["alias_of"]


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
    """A sealed, append-only allocator of stable entry ids for source keys.

    Open one with :meth:`create` or :meth:`open` and use it as a context
    manager; the enclosing ``with`` holds an exclusive lock and reseals the
    document on exit only when something was assigned or aliased. Callers use
    those constructors rather than the initializer, which binds an
    already-validated state to its file and lock.

    :param path: The ledger seal document path.
    :param bases: The per-family id bases, keyed by family name.
    :param series: The id series every minted id carries.
    :param keys: The signing keys used to reseal on close, each a
        ``(role, seed)`` pair.
    :param records: The ordered ledger records, each a ``{key, ...}`` mapping.
    :param live: The newest record per key, the key's live binding.
    """

    def __init__(
        self,
        path: Path,
        *,
        bases: dict[str, str],
        series: str,
        keys: Sequence[tuple[str, bytes]],
        records: list[dict[str, str]],
        live: dict[str, dict[str, str]],
    ) -> None:
        self._path = path
        self._lock_path = path.with_name(path.name + ".lock")
        self._bases = bases
        self._series = series
        self._keys = tuple(keys)
        self._records = records
        self._live = live
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

        :param path: Where to write the ledger seal document; it must not exist.
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
            ledger = cls(location, bases=checked, series=series, keys=keys, records=[], live={})
            ledger._write()
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
    ) -> "IdLedger":
        """Open an existing ledger, verifying its signature and invariants.

        With *verify*, the signature is checked first: an INVALID signature
        (content that no longer matches its own seal) always raises, while a
        valid one is treated as an audit record — *trusted_keys* demand a trusted
        signer, and without them a valid signature is accepted and its signer
        logged. The structure and every entry invariant are then validated
        regardless, because the signature attests only to the bytes, not to
        their meaning.

        When *series* is given it must equal the stored series. When *bases* is
        given it is reconciled as a SUPERSET of the stored map: every stored
        family must appear in it with the same base (removing, renaming, or
        re-basing a stored family is an error), while families present only in the
        expectation are ADDED and stamped into the subject at the next reseal (so
        a build that assigns nothing but grows the scheme still writes on close).
        The merged map is revalidated for id shape and base uniqueness.

        :param path: The ledger seal document to open.
        :param keys: The signing keys used to reseal on close, each ``(role, seed)``.
        :param trusted_keys: Trust anchors as ``ed25519:`` keys or ``sha256:``
            fingerprints; when given, the signer must be one of them.
        :param verify: Whether to verify the seal signature.
        :param bases: The per-family bases the caller expects, asserted when given.
        :param series: The id series the caller expects, asserted when given.
        :return: The open, locked ledger.
        :raises IdLedgerError: If the lock is held, verification fails, an
            invariant is violated, or the expected bases/series disagree.
        """

        location = Path(path)
        _acquire_lock(location.with_name(location.name + ".lock"))
        try:
            if verify:
                _verify_signature(location, trusted_keys)
            stored_bases, stored_series, records, live = _read_and_validate(location)
            if series is not None and series != stored_series:
                raise IdLedgerError(f"id ledger {location} has series {stored_series!r}, not the expected {series!r}")
            effective_bases = stored_bases
            added_families = False
            if bases is not None:
                effective_bases, added_families = _extend_bases(stored_bases, dict(bases), stored_series, location)
            ledger = cls(location, bases=effective_bases, series=stored_series, keys=keys, records=records, live=live)
            # An added family is stamped into the subject at the next reseal; a build
            # that assigns nothing but extends the base map still writes on close.
            ledger._dirty = added_families
            return ledger
        except BaseException:
            location.with_name(location.name + ".lock").unlink(missing_ok=True)
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
        :raises IdLedgerError: If the key is aliased and *supersede* is false, its
            family disagrees, the family has no base, or *supersede* is passed
            outside its one transition (a new key, or an already-assigned key).
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
        :raises IdLedgerError: On a conflict, an unknown target, an assigned key
            when *supersede* is false, or *supersede* passed for a new key.
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

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Reseal the ledger when it changed, then release the lock.

        A ledger untouched since it was opened is left byte-identical: nothing
        is rewritten, so an idempotent rebuild produces no git churn. A reseal
        that finds no signing key available raises from the sealing layer; a
        failed reseal leaves the ledger unclosed and its lock held, so a retried
        close reattempts the write instead of silently skipping it (the manual
        remedy for an abandoned session is to delete the lock).
        """

        if self._closed:
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
        """Guard against mutating a closed ledger."""

        if self._closed:
            raise IdLedgerError(f"id ledger is closed: {self._path}")

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
        """Sign and atomically write the whole ledger as a seal document."""

        subject: dict[str, object] = {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "bases": dict(self._bases),
            "series": self._series,
        }
        records = cast("list[dict[str, object]]", [dict(record) for record in self._records])
        body = build_seal_body(LEDGER_KIND, subject, records)
        write_seal(self._path, body, self._keys)


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
    (stamped at the next reseal); every stored family must appear in the
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

    body = _json_bytes({"created": _utc_now(), "hostname": socket.gethostname(), "pid": os.getpid()}) + b"\n"
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


def _verify_signature(location: Path, trusted_keys: Sequence[str]) -> None:
    """Verify the ledger's seal signature and classify its signer.

    :param location: The ledger seal document to verify.
    :param trusted_keys: Trust anchors; when given, the signer must be trusted.
    :raises IdLedgerError: If no signature verifies, or a trusted one was required.
    """

    verification = verify_seal(location, trusted_keys=tuple(trusted_keys))
    if verification.verdict == INVALID:
        raise IdLedgerError(
            f"id ledger signature does not verify ({verification.reason}): {location}. "
            "The ledger may be corrupted or tampered; restore it from git."
        )
    if trusted_keys:
        if verification.verdict != VALID_TRUSTED:
            raise IdLedgerError(
                f"id ledger is signed by an untrusted key ({verification.reason}): {location}. "
                "Pin the correct signer, or restore the ledger from git."
            )
    else:
        _LOGGER.info(
            "id ledger %s signed by %s (no trust anchor configured; signature is an audit record).",
            location,
            ", ".join(verification.signers) or "an unrecorded key",
            extra={"context": "store"},
        )


def _read_and_validate(
    location: Path,
) -> tuple[dict[str, str], str, list[dict[str, str]], dict[str, dict[str, str]]]:
    """Read a ledger seal and validate its structure and every entry invariant.

    :param location: The ledger seal document to read.
    :return: The bases, series, ordered records, and live binding per key.
    :raises IdLedgerError: If the kind, subject, or any record is malformed, an id
        does not parse under its declared base and series, or a supersession chain
        is forged or leaves a key with duplicate live bindings.
    """

    try:
        seal = read_seal(location)
    except (OSError, ValueError) as exc:
        raise IdLedgerError(
            f"id ledger is not a readable seal document: {location} ({exc}). Restore it from git."
        ) from exc
    if seal.kind != LEDGER_KIND:
        raise IdLedgerError(f"seal at {location} is kind {seal.kind!r}, not {LEDGER_KIND!r}")
    bases, series = _validate_subject(seal.subject, location)
    records, live = _validate_entries(seal.records, bases, series, location)
    return bases, series, records, live


def _validate_subject(subject: Mapping[str, object], location: Path) -> tuple[dict[str, str], str]:
    """Validate a ledger subject and return its bases and series."""

    version = subject.get("ledger_format_version")
    if version != LEDGER_FORMAT_VERSION:
        raise IdLedgerError(f"id ledger {location} has unsupported format version {version!r}")
    raw_bases = subject.get("bases")
    series = subject.get("series")
    if not isinstance(raw_bases, dict) or not raw_bases or not isinstance(series, str):
        raise IdLedgerError(f"id ledger {location} has a malformed subject")
    bases: dict[str, str] = {}
    for family, base in raw_bases.items():
        if not isinstance(family, str) or not isinstance(base, str):
            raise IdLedgerError(f"id ledger {location} has a malformed base map")
        bases[family] = base
    duplicate = _duplicate_base(bases)
    if duplicate is not None:
        raise IdLedgerError(f"id ledger {location} shares id base {duplicate!r} across families")
    return bases, series


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
