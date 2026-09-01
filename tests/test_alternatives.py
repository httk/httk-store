"""Named alternative representations on the SQL backend (P2A)."""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique

from httk.store import EntryIdConflictError, EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql.mapping import ALT_ID_COLUMN, ALT_KIND_COLUMN, LOGICAL_ID_COLUMN
from httk.store.storage_layout import EntryFamilyDeclaration, EntryRecordDeclaration
from httk.store.store_common import SaveProjection


class AltFamily:
    type = "alts"


@dataclass(frozen=True)
class AltRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class AltMultiFamily:
    type = "altmulti"


@dataclass(frozen=True)
class AltFirst:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class AltSecond:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    value: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


def _single_family() -> EntryFamilyDeclaration:
    return EntryFamilyDeclaration(
        name="test-alts",
        family=AltFamily,
        records=(EntryRecordDeclaration(name="test-alt-record", record=AltRecord),),
        definition_id="urn:httk:test:alts",
    )


def _multi_family() -> EntryFamilyDeclaration:
    return EntryFamilyDeclaration(
        name="test-alts-multi",
        family=AltMultiFamily,
        records=(
            EntryRecordDeclaration(name="test-alt-first", record=AltFirst),
            EntryRecordDeclaration(name="test-alt-second", record=AltSecond),
        ),
        definition_id="urn:httk:test:alts:multi",
    )


@pytest.fixture(params=["sqlite", "duckdb"])
def backend(request: pytest.FixtureRequest):
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        database = Backend.duckdb()
    else:
        database = Backend.sqlite()
    with database:
        yield database


def _store(database: Backend, family: EntryFamilyDeclaration | None = None) -> SqlStore:
    return SqlStore(
        database,
        entry_families=(family or _single_family(),),
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


def _row(store: SqlStore, record_type: type, sid: int) -> dict[str, object]:
    from httk.store.backend.schema import resolve_schema

    table = store._table(resolve_schema(record_type).table_name)
    with store._database.engine.connect() as connection:
        row = connection.execute(
            sqlalchemy.select(
                table.c.id,
                table.c.immutable_id,
                table.c[LOGICAL_ID_COLUMN],
                table.c[ALT_ID_COLUMN],
                table.c[ALT_KIND_COLUMN],
            ).where(table.c.sid == sid)
        ).one()
    return {
        "id": row[0],
        "immutable_id": row[1],
        "logical_id": int(row[2]),
        "alt_id": int(row[3]),
        "alt_kind": row[4],
    }


def _fetch(store: SqlStore, record_type: type, sid: int):
    store._clear_identity_caches()
    return store.fetch(record_type, sid, eager=True)


def test_main_plus_two_alternatives(backend: Backend) -> None:
    store = _store(backend)
    main_sid = store.save(AltRecord(1))
    main = _fetch(store, AltRecord, main_sid)
    conv_sid = store.save(AltRecord(2), alternative_of=main.id, alternative_kind="conventional")
    prim_sid = store.save(AltRecord(3), alternative_of=main.id, alternative_kind="primitive")

    main_row = _row(store, AltRecord, main_sid)
    conv = _row(store, AltRecord, conv_sid)
    prim = _row(store, AltRecord, prim_sid)

    assert conv["id"] == main.id and prim["id"] == main.id
    assert conv["immutable_id"] == f"{main.id}~conventional~1"
    assert prim["immutable_id"] == f"{main.id}~primitive~1"
    assert conv["alt_id"] == main_row["logical_id"] == main_sid
    assert prim["alt_id"] == main_row["logical_id"]
    assert main_row["alt_kind"] is None
    assert conv["alt_kind"] == "conventional"
    assert prim["alt_kind"] == "primitive"
    # An alternative is its own lineage.
    assert conv["logical_id"] == conv_sid and prim["logical_id"] == prim_sid


def test_explicit_id_alternative(backend: Backend) -> None:
    store = _store(backend)
    main = _fetch(store, AltRecord, store.save(AltRecord(1)))
    other = _fetch(store, AltRecord, store.save(AltRecord(2)))

    accepted = store.save(AltRecord(3, id=main.id), alternative_of=main.id, alternative_kind="conventional")
    assert _row(store, AltRecord, accepted)["id"] == main.id

    with pytest.raises(EntryIdConflictError):
        store.save(AltRecord(4, id=other.id), alternative_of=main.id, alternative_kind="primitive")


def test_second_same_kind_conflicts(backend: Backend) -> None:
    store = _store(backend)
    main = _fetch(store, AltRecord, store.save(AltRecord(1)))
    store.save(AltRecord(2), alternative_of=main.id, alternative_kind="conventional")
    with pytest.raises(EntryIdConflictError):
        store.save(AltRecord(3), alternative_of=main.id, alternative_kind="conventional")


def test_wrong_alternative_of_resave_conflicts(backend: Backend) -> None:
    store = _store(backend)
    main_a = _fetch(store, AltRecord, store.save(AltRecord(1)))
    main_b = _fetch(store, AltRecord, store.save(AltRecord(2)))
    store.save(AltRecord(3), alternative_of=main_a.id, alternative_kind="conventional")
    # An existing conventional alternative carries group A's id.  Re-saving it
    # (explicit id A) while claiming group B mismatches the id B copied from B's
    # main.  (A bare re-save without an explicit id would instead be a NEW,
    # legitimate conventional alternative of group B -- see test 9.)
    with pytest.raises(EntryIdConflictError):
        store.save(AltRecord(3, id=main_a.id), alternative_of=main_b.id, alternative_kind="conventional")


def test_validation_errors(backend: Backend) -> None:
    store = _store(backend)
    main = _fetch(store, AltRecord, store.save(AltRecord(1)))
    conv_sid = store.save(AltRecord(2), alternative_of=main.id, alternative_kind="conventional")
    conv = _fetch(store, AltRecord, conv_sid)

    # An alternative shares its main's id, so passing a group member's id
    # resolves to the group main -- it never nests.  (The alternative-of-an-
    # alternative ValueError is a defensive guard for an orphan alternative,
    # which the public API cannot produce.)  A new kind is accepted.
    assert conv.id == main.id
    store.save(AltRecord(9), alternative_of=conv.id, alternative_kind="tetragonal")
    # Main not found.
    with pytest.raises(ValueError, match="no entry"):
        store.save(AltRecord(9), alternative_of="httk.test-does-not-exist-1", alternative_kind="foo")
    # Both-or-neither.
    with pytest.raises(ValueError, match="together"):
        store.save(AltRecord(9), alternative_of=main.id)
    with pytest.raises(ValueError, match="together"):
        store.save(AltRecord(9), alternative_kind="conventional")
    # Malformed kind.
    with pytest.raises(ValueError, match="invalid alternative_kind"):
        store.save(AltRecord(9), alternative_of=main.id, alternative_kind="Bad Kind")


def test_bulk_rejects_alternative() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        with store.bulk_ingest() as bulk, pytest.raises(ValueError, match="mains only"):
            bulk.save(AltRecord(1), alternative_of="httk.test-1-1", alternative_kind="conventional")


def test_degraded_profile_alternative() -> None:
    with Backend.sqlite(degraded=True) as database:
        store = _store(database)
        main_sid = store.save(AltRecord(1))
        main = _fetch(store, AltRecord, main_sid)
        conv_sid = store.save(AltRecord(2), alternative_of=main.id, alternative_kind="conventional")
        conv = _row(store, AltRecord, conv_sid)
        assert conv["id"] == main.id
        assert conv["immutable_id"] == f"{main.id}~conventional~1"
        assert conv["alt_kind"] == "conventional"
        assert conv["alt_id"] == _row(store, AltRecord, main_sid)["logical_id"] == main_sid


def test_replace_of_alternative(backend: Backend) -> None:
    store = _store(backend)
    main = _fetch(store, AltRecord, store.save(AltRecord(1)))
    conv_sid = store.save(AltRecord(2), alternative_of=main.id, alternative_kind="conventional")
    conv = _fetch(store, AltRecord, conv_sid)
    rev2_sid = store.replace(conv, AltRecord(99))
    rev2 = _row(store, AltRecord, rev2_sid)

    assert rev2["immutable_id"] == f"{main.id}~conventional~2"
    assert rev2["id"] == main.id
    assert rev2["alt_kind"] == "conventional"
    assert rev2["alt_id"] == _row(store, AltRecord, conv_sid)["alt_id"]
    assert rev2["logical_id"] == _row(store, AltRecord, conv_sid)["logical_id"]

    # A replacement whose content is byte-identical to the MAIN's must not dedup
    # onto the main: extras keep the two identities apart.
    main_sid = store.save(AltRecord(1))  # dedups back onto the existing main row
    rev3_sid = store.replace(_fetch(store, AltRecord, rev2_sid), AltRecord(1))
    assert rev3_sid != main_sid
    assert _row(store, AltRecord, rev3_sid)["immutable_id"] == f"{main.id}~conventional~3"


def test_idempotent_resave(backend: Backend) -> None:
    store = _store(backend)
    main = _fetch(store, AltRecord, store.save(AltRecord(1)))
    conv_sid = store.save(AltRecord(2), alternative_of=main.id, alternative_kind="conventional")
    again = store.save(AltRecord(2), alternative_of=main.id, alternative_kind="conventional")
    assert again == conv_sid


def test_content_id_separation(backend: Backend) -> None:
    store = _store(backend)
    main = _fetch(store, AltRecord, store.save(AltRecord(1)))
    projection = SaveProjection()
    main_cid = projection.content_id(AltRecord, AltRecord(2))
    alt_cid = projection.content_id(
        AltRecord, AltRecord(2), extras={"alternative_of": main.id, "alternative_kind": "conventional"}
    )
    assert main_cid != alt_cid

    # Two groups' same-kind alternatives with identical structural content both
    # insert cleanly (distinct content ids via distinct extras).
    other = _fetch(store, AltRecord, store.save(AltRecord(7)))
    a = store.save(AltRecord(5), alternative_of=main.id, alternative_kind="conventional")
    b = store.save(AltRecord(5), alternative_of=other.id, alternative_kind="conventional")
    assert a != b


def test_multi_backing_main_vs_alternative_dispatch() -> None:
    with Backend.sqlite() as database:
        store = _store(database, _multi_family())
        base = _fetch(store, AltFirst, store.save(AltFirst(1)))
        # An alternative whose structural content equals AltFirst(2).
        store.save(AltFirst(2), alternative_of=base.id, alternative_kind="conventional")
        # A plain main with the same structural content must not collide on the
        # per-family dispatch table (its key omits the extras).
        main2 = store.save(AltFirst(2))
        assert isinstance(main2, int)


def test_searcher_and_history(backend: Backend) -> None:
    store = _store(backend)
    main = _fetch(store, AltRecord, store.save(AltRecord(1)))
    conv_sid = store.save(AltRecord(2), alternative_of=main.id, alternative_kind="conventional")
    store.save(AltRecord(3), alternative_of=main.id, alternative_kind="primitive")

    def values(only_main_alt: bool) -> set[int]:
        searcher = store.searcher(only_main_alt=only_main_alt)
        searcher.output(searcher.variable(AltRecord), "rec")
        return {item[0][0].value for item in searcher}

    assert values(only_main_alt=True) == {1}
    assert values(only_main_alt=False) == {1, 2, 3}

    # history() of a main and of an alternative are each pure (no cross-mixing).
    assert [obj.value for obj in store.history(main)] == [1]
    assert [obj.value for obj in store.history(_fetch(store, AltRecord, conv_sid))] == [2]


def test_explicit_immutable_id_cannot_forge_second_lineage(backend: Backend) -> None:
    store = _store(backend)
    main = _fetch(store, AltRecord, store.save(AltRecord(1)))
    conv_sid = store.save(AltRecord(2), alternative_of=main.id, alternative_kind="conventional")
    # A fresh save carrying an explicit <id>~conventional~2 dodges the rev-1
    # immutable-id collision, but the (alt_id, alt_kind) lineage scan rejects it
    # as a second conventional lineage in the group.
    with pytest.raises(EntryIdConflictError):
        store.save(
            AltRecord(3, immutable_id=f"{main.id}~conventional~2"),
            alternative_of=main.id,
            alternative_kind="conventional",
        )
    # The legitimate revision path still works.
    rev2 = store.replace(_fetch(store, AltRecord, conv_sid), AltRecord(99))
    assert _row(store, AltRecord, rev2)["immutable_id"] == f"{main.id}~conventional~2"
