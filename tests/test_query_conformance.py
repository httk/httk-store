"""Prove the neutral query conformance table against httk-store's own stores."""

from dataclasses import dataclass
from typing import ClassVar

from httk.core.storage import StorageInfo

from httk.store.query.conformance import check_query_conformance, conformance_cases, conformance_rows


@dataclass(frozen=True)
class ConformanceRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

    rid: str
    name: str | None
    n: int | None
    x: float | None
    tags: list[str] | None


def test_store_conforms_to_query_table(store_factory):
    store = store_factory()
    for row in conformance_rows():
        store.save(ConformanceRecord(**row))
    check_query_conformance(store, ConformanceRecord)


def test_rows_are_fresh_and_case_names_unique():
    first, second = conformance_rows(), conformance_rows()
    assert first == second
    first[0]["tags"].append("mutated")
    assert conformance_rows()[0]["tags"] == ["p", "q"]
    names = [case.name for case in conformance_cases()]
    assert len(names) == len(set(names))
