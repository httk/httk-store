"""Stored federation serving of Related-marked reference/child fields as relationships.

The fourth relationship collection in ``stored_federation._collect_relationships``:
reference and child-of-storable fields whose target class is a served family are
served as OPTIMADE relationship blocks, mirroring the in-memory provider path
(``StoreEntryProvider._relationship_specs`` / ``relationships``): ``Related``
metadata (role/description) rides the identifiers, ``Related(serve=False)``
suppresses a field, child fields contribute an ordered list, and the identifier
type is the target's served (wire) name. Reference/child values are record
content, so the blocks are revision-pinned (``~revs``) and — like forward
StrongLink edges — also present on named alternatives (``~alts``).
"""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar

from httk.core import RelatedEntry
from httk.core.data_records import RECORDS_DEFINITION_ID
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, Related, StorageInfo, Unique

from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore, StoredEntryFederation, StoredEntrySource

_REFERENCES = "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/references"


@dataclass(frozen=True)
class PersonRow:
    """The relationship target (served under the ``references`` wire type)."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="related_field_person")

    doi: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class ProjectRow:
    """A ``records`` backing with Related-marked reference and child fields."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="related_field_project")

    name: str
    lead: Annotated[PersonRow | None, Related(role="lead", description="Project lead")] = None
    backup: Annotated[PersonRow | None, Related(serve=False)] = None
    members: Annotated[tuple[PersonRow, ...], Related(role="member", description="A member")] = ()
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


class PersonFamily:
    type = "references"
    definition_id = _REFERENCES


class ProjectFamily:
    type = "records"
    definition_id = RECORDS_DEFINITION_ID


register_entry_family(name="related-field-person", family=f"{__name__}:PersonFamily", definition_id=_REFERENCES)
register_entry_record(name="related-field-person-rec", family="related-field-person", record=f"{__name__}:PersonRow")
register_entry_family(
    name="related-field-project", family=f"{__name__}:ProjectFamily", definition_id=RECORDS_DEFINITION_ID
)
register_entry_record(name="related-field-project-rec", family="related-field-project", record=f"{__name__}:ProjectRow")


def _store(database: Backend) -> SqlStore:
    return SqlStore(
        database,
        entry_records={PersonFamily: PersonRow, ProjectFamily: ProjectRow},
        entry_ids=EntryIdScheme("httk.test", "1"),
    )


def _save(store: SqlStore, record: object) -> object:
    return store.fetch(type(record), store.save(record), eager=True)


def _blocks(page: object, entry_id: str) -> dict[str, list[RelatedEntry]]:
    (row_index,) = [index for index, row in enumerate(page.rows) if row["id"] == entry_id]  # type: ignore[attr-defined]
    return {key: list(entries) for key, entries in dict(page.relationships[row_index]).items()}  # type: ignore[attr-defined]


def test_reference_and_child_fields_served_with_metadata() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ada = _save(store, PersonRow("10.1/ada"))
        boole = _save(store, PersonRow("10.2/boole"))
        cara = _save(store, PersonRow("10.3/cara"))
        project = _save(store, ProjectRow("Engine", lead=ada, backup=boole, members=(boole, cara)))

        federation = StoredEntryFederation((StoredEntrySource(store, ProjectFamily, "src"),))
        page = federation.query()
        blocks = _blocks(page, project.id)

        # The identifier type is the target's served wire name ("references"),
        # role/description ride the identifiers, and child order is preserved.
        assert blocks["references"] == [
            RelatedEntry("references", ada.id, description="Project lead", role="lead"),
            RelatedEntry("references", boole.id, description="A member", role="member"),
            RelatedEntry("references", cara.id, description="A member", role="member"),
        ]
        # Related(serve=False) suppresses the backup edge entirely: boole appears
        # only through the served member field, never as backup.
        assert all(entry.role != "backup" for entry in blocks["references"])


def test_serve_false_only_field_yields_no_block() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        boole = _save(store, PersonRow("10.2/boole"))
        # Only the suppressed field is populated: no relationship block at all.
        project = _save(store, ProjectRow("Quiet", backup=boole))
        federation = StoredEntryFederation((StoredEntrySource(store, ProjectFamily, "src"),))
        page = federation.query()
        assert _blocks(page, project.id) == {}


def test_reference_block_is_revision_pinned() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ada = _save(store, PersonRow("10.1/ada"))
        boole = _save(store, PersonRow("10.2/boole"))
        project = _save(store, ProjectRow("Engine", lead=ada))
        # A new revision retargets the lead reference; each revision carries its own FK.
        store.replace(project, ProjectRow("Engine", lead=boole))

        federation = StoredEntryFederation((StoredEntrySource(store, ProjectFamily, "src"),))
        revs = federation.query(revisions=True, sort=(("immutable_id", False),))
        leads_by_rev = {
            row["id"]: [entry.id for entry in dict(rel).get("references", ())]
            for row, rel in zip(revs.rows, revs.relationships, strict=True)
        }
        # Two revisions, each pinned to its own lead reference.
        assert sorted(leads_by_rev.values()) == sorted([[ada.id], [boole.id]])


def test_reference_block_present_on_alternatives() -> None:
    with Backend.sqlite() as database:
        store = _store(database)
        ada = _save(store, PersonRow("10.1/ada"))
        project = _save(store, ProjectRow("Engine", lead=ada))
        # A named alternative of the project (its own record content, its own FK):
        # forward record-content blocks appear on ~alts, like forward StrongLink edges.
        store.save(ProjectRow("Engine-alt", lead=ada), alternative_of=project.id, alternative_kind="conventional")

        federation = StoredEntryFederation((StoredEntrySource(store, ProjectFamily, "src"),))
        alts = federation.query(alternatives=True)
        assert len(alts.rows) == 1
        (rel,) = alts.relationships
        assert [entry.id for entry in dict(rel).get("references", ())] == [ada.id]
