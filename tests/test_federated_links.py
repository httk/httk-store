"""Federated coverage for weak-link-set search outputs across SQLite child stores.

Mirrors ``test_federation_sql.py``'s real-backend style: no existing federated
test file covers weak links, so this one does. ``FederatedVariable.links.<name>``
already replays as a ``_FieldOutput`` on each child (no dedicated federated
link-output type is needed); this proves the merged rows carry each child's
resolved tuple unchanged.
"""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from httk.core.storage import StorageInfo, WeakLink

from httk.store import FederatedStore
from httk.store.backend.sql import Backend, SqlStore


@dataclass(frozen=True)
class Project:
    name: str


@dataclass(frozen=True)
class Result:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(WeakLink("projects", Project),))

    label: str


def _store(database: Backend) -> SqlStore:
    return SqlStore(database, entry_records={})


def test_federated_link_set_output_merges_tuples_across_child_stores() -> None:
    with Backend.sqlite() as first_database, Backend.sqlite() as second_database:
        first_store = _store(first_database)
        p1 = Project("P1")
        first_store.save(p1)
        r1 = Result("R1")
        first_store.save(r1)
        first_store.link(r1, "projects", p1)

        second_store = _store(second_database)
        p2 = Project("P2")
        second_store.save(p2)
        r2 = Result("R2")
        second_store.save(r2)
        second_store.link(r2, "projects", p2)
        second_store.save(Result("R3"))  # unlinked: the empty-tuple case must merge too

        federation = FederatedStore({"first": first_store, "second": second_store})
        searcher = federation.searcher()
        record = searcher.variable(Result)
        result = searcher.results(label=record.label, projects=record.links.projects)

        by_label = {row.label: tuple(p.name for p in row.projects) for row in result}
        assert by_label == {"R1": ("P1",), "R2": ("P2",), "R3": ()}

        with pytest.raises(TypeError, match="weak-link-set output"):
            result.column("projects")
