"""Standard weak links (references, notes, tags, external) and the curation records.

Every entry record is linkable under the four ``httk.core.curation.STANDARD_LINKS``
names without declaring them, and a store declaring such a record gains the four
link-target families. Runs over the parametrized ``store_factory`` arms (Mongo
with transactions required, as its weak links need them); the OPTIMADE serving
checks are SQL-only, as in ``test_reference_links.py``.
"""

import datetime
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

import pytest
from httk.core import ReferenceRecord, Run, RunEntry
from httk.core.curation import (
    TAGS_DEFINITION_ID,
    ExternalDatabaseEntry,
    ExternalDatabaseEntryFamily,
    Note,
    NoteEntry,
    Tag,
    TagEntry,
)
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique, WeakLink

from httk.store import EntryIdScheme
from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql import SqliteStore, SqlStore, StoredEntrySource
from httk.store.backend.sql.layout import normalize_entry_declaration
from httk.store.backend.sql.stored_federation import StoredEntryFederation
from httk.store.storage_layout import (
    EntryFamilyDeclaration,
    EntryRecordDeclaration,
    declaration_json,
    normalize_entry_records,
)

_STANDARD = ("references", "notes", "tags", "external")
_TARGET_FAMILIES = {
    "references": ("core-reference-record",),
    "notes": ("core-note",),
    "tags": ("core-tag",),
    "external_database_entries": ("core-external-database-entry",),
}
_DATE = datetime.datetime(2026, 11, 25, 12, 0, tzinfo=datetime.UTC)


@dataclass(frozen=True)
class Cited:
    """An entry record declaring its own ``notes`` link to a different target."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(WeakLink("notes", Tag),))

    label: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class Tagged:
    """An entry record whose ``tags`` field wins over the standard ``tags`` link."""

    tags: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class Plain:
    """A storable dataclass without entry ids."""

    value: int


@dataclass(frozen=True)
class Label:
    """A second record kind of the tags family."""

    text: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


def _names(cls: type) -> tuple[str, ...]:
    return tuple(link.name for link in resolve_schema(cls).links)


def test_standard_links_resolve_at_schema_time() -> None:
    assert _names(Run) == _STANDARD
    links = {link.name: link for link in resolve_schema(Run).links}
    assert links["external"].target is ExternalDatabaseEntry
    assert all(link.exposed_relationship for link in links.values())
    # A declared link keeps its place and target; the same-named standard link is not added.
    assert _names(Cited) == ("notes", "references", "tags", "external")
    assert resolve_schema(Cited).links[0].target is Tag
    # A stored field of the link's name wins.
    assert _names(Tagged) == ("references", "notes", "external")
    # The link targets get no automatic links; ExternalDatabaseEntry keeps its declared one.
    assert _names(ReferenceRecord) == ()
    assert _names(Note) == ()
    assert _names(Tag) == ()
    assert _names(ExternalDatabaseEntry) == ("references",)
    assert _names(Plain) == ()


def test_layout_gains_link_target_families_once() -> None:
    layout = normalize_entry_records({RunEntry: Run})
    assert dict(layout.declaration) == {"runs": ("core-run",), **_TARGET_FAMILIES}
    # An empty declaration stays empty; a link-target-only declaration gains nothing.
    assert dict(normalize_entry_records({}).declaration) == {}
    assert dict(normalize_entry_records({TagEntry: Tag}).declaration) == {"tags": ("core-tag",)}
    # An already-declared target family is left as declared.
    both = normalize_entry_records({RunEntry: Run, NoteEntry: Note})
    assert declaration_json(both) == declaration_json(layout)


class _CitedFamily:
    type = "cited"


def test_merged_declarations_keep_one_copy_of_each_target_family() -> None:
    private = EntryFamilyDeclaration(
        name="test-cited", family=_CitedFamily, records=(EntryRecordDeclaration(name="test-cited", record=Cited),)
    )
    # Both parts gain the implicit target families; the merge keeps one of each.
    merged = normalize_entry_declaration({RunEntry: Run}, (private,))
    assert merged is not None
    assert dict(merged.declaration) == {"runs": ("core-run",), "test-cited": ("test-cited",), **_TARGET_FAMILIES}
    # A target family declared with more records wins over the implicit one of another part.
    tags = EntryFamilyDeclaration(
        name="tags",
        family=TagEntry,
        records=(
            EntryRecordDeclaration(name="core-tag", record=Tag, definition_id=TAGS_DEFINITION_ID),
            EntryRecordDeclaration(name="test-label", record=Label),
        ),
        definition_id=TAGS_DEFINITION_ID,
    )
    extended = normalize_entry_declaration({RunEntry: Run}, (tags,))
    assert extended is not None
    assert extended.declaration["tags"] == ("core-tag", "test-label")


def _linking_store(store_factory):
    return store_factory(
        transactions="require", entry_records={RunEntry: Run}, entry_ids=EntryIdScheme("httk.test", "1")
    )


def test_run_links_to_every_curation_record_and_reopens(store_factory) -> None:
    store = _linking_store(store_factory)
    run_sid = store.save(Run(source_id="ws:job"))
    run = store.fetch(Run, run_sid, eager=True)
    targets = {
        "references": ReferenceRecord(title="The external database"),  # type: ignore[arg-type]
        "notes": Note(_DATE, "httk-test 1.0", "Removed a stray loop."),
        "tags": Tag("source", "cod"),
        "external": ExternalDatabaseEntry("cod", "1000000", role="origin"),
    }
    ids = {}
    for name, target in targets.items():
        stored = store.fetch(type(target), store.save(target), eager=True)
        store.link(run, name, stored)
        ids[name] = stored.id
    for name in targets:
        assert [linked.id for linked in store.linked(run, name)] == [ids[name]]
    (note,) = store.linked(run, "notes")
    assert (note.date, note.author, note.text) == (_DATE, "httk-test 1.0", "Removed a stray loop.")

    reopened = store_factory.reopen(store)
    assert dict(reopened.layout.declaration) == {"runs": ("core-run",), **_TARGET_FAMILIES}
    again = reopened.fetch(Run, run_sid)
    assert [linked.id for linked in reopened.linked(again, "external")] == [ids["external"]]


def test_curation_records_deduplicate(store_factory) -> None:
    store = _linking_store(store_factory)
    assert store.save(Tag("source", "cod")) == store.save(Tag("source", "cod"))
    origin = ExternalDatabaseEntry("cod", "1", role="origin")
    origin_sid = store.save(origin)
    assert store.save(ExternalDatabaseEntry("cod", "1")) != origin_sid
    stored = store.fetch(ExternalDatabaseEntry, origin_sid, eager=True)
    assert stored.url is None
    # The url is outside the identity: a later save fills the stored NULL (KeepStored) in place.
    url = "https://www.crystallography.net/cod/1.html"
    assert store.save(ExternalDatabaseEntry("cod", "1", url=url, role="origin")) == origin_sid
    filled = store.fetch(ExternalDatabaseEntry, origin_sid, eager=True)
    assert filled.url == url
    assert (filled.id, filled.immutable_id) == (stored.id, stored.immutable_id)


def _labels(store, add) -> list[str | None]:
    searcher = store.searcher()
    v = searcher.variable(Run)
    add(searcher, v)
    return sorted(row.values[0].source_id for row in searcher.results(r=v))


def test_query_dsl_over_standard_links(store_factory) -> None:
    store = _linking_store(store_factory)
    first = store.fetch(Run, store.save(Run(source_id="first")), eager=True)
    second = store.fetch(Run, store.save(Run(source_id="second")), eager=True)
    for run, entries in (
        (first, (ExternalDatabaseEntry("cod", "1", role="origin"), ExternalDatabaseEntry("icsd", "2"))),
        (second, (ExternalDatabaseEntry("cod", "3"),)),
    ):
        for entry in entries:
            store.link(run, "external", store.fetch(ExternalDatabaseEntry, store.save(entry), eager=True))
    for run, tag in ((first, Tag("x", "y")), (second, Tag("x", "z"))):
        store.link(run, "tags", store.fetch(Tag, store.save(tag), eager=True))

    def same_external(searcher, v):
        lk = v.links.external
        searcher.add((lk.database == "cod") & (lk.role == "origin"))

    assert _labels(store, same_external) == ["first"]
    # Fresh accesses constrain independent linked entries: second has a cod entry, but no origin at all.
    assert _labels(store, lambda s, v: s.add(v.links.external.database == "cod")) == ["first", "second"]
    assert _labels(store, lambda s, v: s.add(v.links.tags.key == "x")) == ["first", "second"]

    def same_tag(searcher, v):
        t = v.links.tags
        searcher.add((t.key == "x") & (t.value == "y"))

    assert _labels(store, same_tag) == ["first"]


def _query(store: SqlStore, family: type, filter_string: str | None = None):
    return StoredEntryFederation((StoredEntrySource(store, family, "src"),)).query(filter_string)


def _ids(page) -> list[str]:
    return sorted(row["id"] for row in page.rows)


def test_curation_records_serve_and_filter(store_factory) -> None:
    store = _linking_store(store_factory)
    if not isinstance(store, SqlStore):
        pytest.skip("stored-federation serving and filtering are SQL-only")
    run = store.fetch(Run, store.save(Run(source_id="linked")), eager=True)
    other = store.fetch(Run, store.save(Run(source_id="unlinked")), eager=True)
    note = store.fetch(Note, store.save(Note(_DATE, "curator", "Checked.")), eager=True)
    tag = store.fetch(Tag, store.save(Tag("x", "y")), eager=True)
    origin = store.fetch(
        ExternalDatabaseEntry,
        store.save(ExternalDatabaseEntry("cod", "1", url="https://example.org/1", role="origin")),
        eager=True,
    )
    general = store.fetch(ExternalDatabaseEntry, store.save(ExternalDatabaseEntry("icsd", "2")), eager=True)
    store.link(run, "notes", note)
    store.link(run, "tags", tag)
    store.link(run, "external", origin)
    store.link(other, "external", general)

    (relationships,) = _query(store, RunEntry, f'id = "{run.id}"').relationships
    served = {key: [(entry.entry_type, entry.id) for entry in entries] for key, entries in relationships.items()}
    # The httk-defined entry types are served under the provider prefix.
    assert served == {
        "_httk_notes": [("_httk_notes", note.id)],
        "_httk_tags": [("_httk_tags", tag.id)],
        "_httk_external_database_entries": [("_httk_external_database_entries", origin.id)],
    }

    (note_row,) = _query(store, NoteEntry).rows
    assert (note_row["id"], note_row["_httk_author"], note_row["_httk_text"]) == (note.id, "curator", "Checked.")
    assert note_row["_httk_date"] is not None
    (tag_row,) = _query(store, TagEntry, '_httk_key = "x"').rows
    assert (tag_row["id"], tag_row["_httk_value"]) == (tag.id, "y")
    assert _ids(_query(store, TagEntry, '_httk_key = "other"')) == []
    rows = {row["id"]: row for row in _query(store, ExternalDatabaseEntryFamily).rows}
    assert rows[origin.id]["_httk_database"] == "cod"
    assert rows[origin.id]["_httk_entry_id"] == "1"
    assert rows[origin.id]["_httk_url"] == "https://example.org/1"
    assert rows[origin.id]["_httk_role"] == "origin"
    assert rows[general.id]["_httk_role"] is None
    assert _ids(_query(store, ExternalDatabaseEntryFamily, '_httk_database = "cod"')) == [origin.id]
    assert _ids(_query(store, ExternalDatabaseEntryFamily, '_httk_role = "origin"')) == [origin.id]

    assert _ids(_query(store, RunEntry, f'_httk_external_database_entries.id HAS "{origin.id}"')) == [run.id]
    assert _ids(_query(store, RunEntry, f'_httk_external_database_entries.id HAS "{general.id}"')) == [other.id]


def test_reopen_and_upgrade_see_the_same_layout(tmp_path) -> None:
    path = tmp_path / "curation.sqlite"
    scheme = EntryIdScheme("httk.test", "1")
    with SqliteStore(path, entry_records={RunEntry: Run}, entry_ids=scheme) as store:
        sid = store.save(Tag("x"))
        expected = dict(store.layout.declaration)
    for options in ({"entry_records": {RunEntry: Run}}, {}, {"entry_records": {RunEntry: Run}, "upgrade": True}):
        with SqliteStore(path, **options) as reopened:
            assert dict(reopened.layout.declaration) == expected
            assert reopened.fetch(Tag, sid).key == "x"
    # Declaring a target family explicitly is the same declaration.
    with SqliteStore(path, entry_records={RunEntry: Run, TagEntry: Tag}) as reopened:
        assert dict(reopened.layout.declaration) == expected
