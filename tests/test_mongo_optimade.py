"""Live MongoDB semantic coverage for OPTIMADE-filter querying."""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Annotated, ClassVar

import pytest
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, Unique, WeakLink

from httk.store.backend.mongo import MongoStore, optimade_filter_searcher
from httk.store.query.optimade_filters import FilterTranslationError


@dataclass(frozen=True)
class Publication:
    doi: str
    year: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class Material:
    name: str
    x: Fraction
    symbols: list[str]
    ref: Publication | None = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class Part:
    label: str
    val: int
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class Assembly:
    name: str
    parts: list[Part]
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


PUB_A = Publication("10.1000/alpha", 1999, "publication-a", "publication-a~1")
PUB_B = Publication("10.2000/beta", 2005, "publication-b", "publication-b~1")

MAT_1 = Material("alpha oxide", Fraction(1, 2), ["O", "H"], PUB_A, "material-1", "material-1~1")
MAT_2 = Material("beta metal", Fraction(5, 2), ["Fe"], PUB_B, "material-2", "material-2~1")
MAT_3 = Material("gamma oxide", Fraction(7, 2), ["O"], None, "material-3", "material-3~1")


@pytest.fixture()
def store(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={})
    for material in (MAT_1, MAT_2, MAT_3):
        store.save(material)
    return store


def results(searcher):
    return [item.values[0] for item in searcher]


def test_numeric_comparison_on_fraction_field(store):
    searcher = optimade_filter_searcher(store, Material, "_httk_custom_x > 1")
    assert results(searcher) == [MAT_2, MAT_3]


def test_string_operations(store):
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_name STARTS WITH "beta"')
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_name CONTAINS "oxide"')
    assert results(searcher) == [MAT_1, MAT_3]
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_name = "gamma oxide"')
    assert results(searcher) == [MAT_3]


def test_has_over_list_field(store):
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_symbols HAS "O"')
    assert results(searcher) == [MAT_1, MAT_3]
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_symbols HAS ALL "O","H"')
    assert results(searcher) == [MAT_1]
    searcher = optimade_filter_searcher(store, Material, 'NOT _httk_custom_symbols HAS "O"')
    assert results(searcher) == [MAT_2]


def test_combined_scalar_and_related_property_semi_join(store):
    searcher = optimade_filter_searcher(
        store,
        Material,
        '_httk_custom_x > 1 AND refs._httk_custom_doi CONTAINS "10.2"',
        related_classes={"refs": Publication},
    )
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(
        store, Material, 'refs._httk_custom_doi CONTAINS "10."', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1, MAT_2]


def test_related_property_comparison_and_not_complement(store):
    searcher = optimade_filter_searcher(
        store, Material, "refs._httk_custom_year >= 2000", related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(
        store, Material, "NOT refs._httk_custom_year >= 2000", related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1, MAT_3]


def test_related_property_no_match_is_empty_not_an_error(store):
    searcher = optimade_filter_searcher(
        store, Material, 'refs._httk_custom_doi CONTAINS "nomatch"', related_classes={"refs": Publication}
    )
    assert results(searcher) == []


def test_relationship_id_has_over_foreign_key(store):
    searcher = optimade_filter_searcher(
        store, Material, f'refs.id HAS "{PUB_A.id}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1]


def test_relationship_id_has_only_preserves_empty_reference_vacuous_truth(store):
    searcher = optimade_filter_searcher(
        store, Material, f'refs.id HAS ONLY "{PUB_B.id}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_2, MAT_3]
    searcher = optimade_filter_searcher(
        store, Material, f'NOT refs.id HAS ONLY "{PUB_B.id}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1]

    searcher = optimade_filter_searcher(
        store, Material, f'refs.id HAS ANY "{PUB_B.id}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(
        store, Material, f'NOT refs.id HAS ANY "{PUB_B.id}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1, MAT_3]


def test_relationship_id_equality_routes_through_semi_join(store):
    searcher = optimade_filter_searcher(
        store, Material, f'refs.id = "{PUB_B.id}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(
        store, Material, f'refs.id != "{PUB_B.id}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1]


def test_invalid_or_foreign_id_formats_match_nothing(store):
    for bad_id in ("bogus", "refs-", "refs-abc", "other-1", "refs-1-2"):
        searcher = optimade_filter_searcher(
            store, Material, f'refs.id HAS "{bad_id}"', related_classes={"refs": Publication}
        )
        assert results(searcher) == []


def test_child_of_storable_target(store):
    part_a = Part("bolt", 1, "part-a", "part-a~1")
    part_b = Part("nut", 5, "part-b", "part-b~1")
    assembly_1 = Assembly("frame", [part_a, part_b], "assembly-1", "assembly-1~1")
    assembly_2 = Assembly("hinge", [part_a], "assembly-2", "assembly-2~1")
    store.save(assembly_1)
    store.save(assembly_2)
    searcher = optimade_filter_searcher(store, Assembly, "parts._httk_custom_val > 2", related_classes={"parts": Part})
    assert results(searcher) == [assembly_1]
    searcher = optimade_filter_searcher(
        store, Assembly, f'parts.id HAS "{part_a.id}"', related_classes={"parts": Part}
    )
    assert results(searcher) == [assembly_1, assembly_2]


def test_nested_dotted_path_not_implemented(store):
    with pytest.raises(FilterTranslationError) as excinfo:
        optimade_filter_searcher(
            store, Material, "refs.other._httk_custom_x = 1", related_classes={"refs": Publication}
        )
    assert excinfo.value.category == "not-implemented"


def test_dotted_filter_without_related_classes_matches_nothing(store):
    searcher = optimade_filter_searcher(store, Material, 'refs._httk_custom_doi CONTAINS "10."')
    assert results(searcher) == []


def test_id_filters_the_physical_entry_column_while_type_still_needs_a_handler(store):
    assert results(optimade_filter_searcher(store, Material, 'id = "material-1"')) == [MAT_1]
    with pytest.raises(FilterTranslationError) as excinfo:
        optimade_filter_searcher(store, Material, 'type = "materials"')
    assert excinfo.value.category == "not-implemented"


def test_unknown_prefixed_property_raises(store):
    with pytest.raises(FilterTranslationError) as excinfo:
        optimade_filter_searcher(store, Material, "_httk_bananas = 3")
    assert excinfo.value.category == "unrecognized-property"


def test_unknown_unprefixed_property_matches_nothing(store):
    searcher = optimade_filter_searcher(store, Material, "bananas = 3")
    assert results(searcher) == []


def test_unmatched_related_class_raises_value_error(store):
    with pytest.raises(ValueError):
        optimade_filter_searcher(store, Material, 'parts.id HAS "parts-1"', related_classes={"parts": Part})


# --------------------------------------------------------------------- exposed weak links


@dataclass(frozen=True)
class WAuthor:
    name: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class WArticle:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        links=(
            WeakLink("authors", WAuthor, exposed_relationship=True),
            WeakLink("editors", WAuthor, exposed_relationship=False),
        )
    )
    title: str
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


@dataclass(frozen=True)
class AmbiguousArticle:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        links=(WeakLink("authors", WAuthor, exposed_relationship=True),)
    )
    title: str
    lead: WAuthor | None = None
    id: Annotated[str | None, IdentitySkip(), Indexed()] = field(default=None, compare=False)
    immutable_id: Annotated[str | None, IdentitySkip(), Unique()] = field(default=None, compare=False)


def _weak_store(database):
    store = MongoStore(database, entry_records={})
    authors = (
        WAuthor("Ada", "author-1", "author-1~1"),
        WAuthor("Boole", "author-2", "author-2~1"),
        WAuthor("Cara", "author-3", "author-3~1"),
    )
    articles = (
        WArticle("Engines", "article-1", "article-1~1"),
        WArticle("Silence", "article-2", "article-2~1"),
        WArticle("Void", "article-3", "article-3~1"),
    )
    for obj in (*authors, *articles):
        store.save(obj)
    store.link(articles[0], "authors", authors[0])
    store.link(articles[0], "authors", authors[1])
    store.link(articles[1], "authors", authors[1])
    return store, authors, articles


def test_weak_link_id_has_matches_linked_sources(mongo_test_database):
    store, _authors, articles = _weak_store(mongo_test_database)
    searcher = optimade_filter_searcher(
        store, WArticle, 'authors.id HAS "author-2"', related_classes={"authors": WAuthor}
    )
    assert results(searcher) == [articles[0], articles[1]]


def test_weak_link_id_has_all_requires_every_target(mongo_test_database):
    store, _authors, articles = _weak_store(mongo_test_database)
    searcher = optimade_filter_searcher(
        store, WArticle, 'authors.id HAS ALL "author-1","author-2"', related_classes={"authors": WAuthor}
    )
    assert results(searcher) == [articles[0]]


def test_weak_link_id_has_only_matches_subset_and_no_links(mongo_test_database):
    store, _authors, articles = _weak_store(mongo_test_database)
    searcher = optimade_filter_searcher(
        store, WArticle, 'authors.id HAS ONLY "author-2"', related_classes={"authors": WAuthor}
    )
    # article-2 links only author-2; article-3 has no links (vacuous truth);
    # article-1 also links author-1 and so is excluded.
    assert results(searcher) == [articles[1], articles[2]]


def test_weak_link_id_unknown_matches_nothing(mongo_test_database):
    store, _authors, _articles = _weak_store(mongo_test_database)
    searcher = optimade_filter_searcher(
        store, WArticle, 'authors.id HAS "author-9"', related_classes={"authors": WAuthor}
    )
    assert results(searcher) == []
    searcher = optimade_filter_searcher(
        store, WArticle, 'authors.id HAS ALL "author-1","author-9"', related_classes={"authors": WAuthor}
    )
    assert results(searcher) == []


def test_weak_link_id_retracted_does_not_match(mongo_test_database):
    store, authors, articles = _weak_store(mongo_test_database)
    store.unlink(articles[1], "authors", authors[1])
    searcher = optimade_filter_searcher(
        store, WArticle, 'authors.id HAS "author-2"', related_classes={"authors": WAuthor}
    )
    assert results(searcher) == [articles[0]]


def test_weak_link_id_matches_revised_target_by_stable_id(mongo_test_database):
    store, authors, articles = _weak_store(mongo_test_database)
    store.replace(authors[0], WAuthor("Ada Lovelace", "author-1", "author-1~2"))
    searcher = optimade_filter_searcher(
        store, WArticle, 'authors.id HAS "author-1"', related_classes={"authors": WAuthor}
    )
    assert results(searcher) == [articles[0]]


def test_reference_field_and_exposed_weak_link_to_same_class_is_ambiguous(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={})
    with pytest.raises(ValueError, match="exactly one is required"):
        optimade_filter_searcher(
            store, AmbiguousArticle, 'authors.id HAS "author-1"', related_classes={"authors": WAuthor}
        )
