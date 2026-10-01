# Weak links and provenance

## Weak links

Reference and child fields are **sid-pinned**: they bind a specific revision, so
a record and its subrecords form a unit replaced together. A *weak link* is the
lineage-following alternative — a store-managed association that binds two
*lineages* and always resolves both endpoints to their latest revision, so a
`Result` linked to a `Project` sees the updated project after that project is
`replace()`d.

Weak links are declared on the **source** class in `StorageInfo.links`, not in a
field, and live in a dedicated `_httk_link_*` table:

```python
from dataclasses import dataclass
from typing import ClassVar

from httk.core.storage import StorageInfo, WeakLink


@dataclass(frozen=True)
class Project:
    name: str


@dataclass(frozen=True)
class Result:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        links=(
            WeakLink(
                "projects",
                target=Project,
                exposed_relationship=True,
                role="belongs-to",
                description="Owning project",
            ),
        )
    )
    value: int
```

Links are directed (declared on the source) but reverse-queryable through
forward filters. `name` namespaces the link and must be a valid identifier;
`target` is the linked storable class; `exposed_relationship` (default `False`)
plus `role`/`description` control OPTIMADE serving (below). Because links are not
part of a record's value, they never enter `content_id`: adding or retracting a
link leaves the record's content identity unchanged.

**Pair-lineage toggle model.** A link lineage is identified by the pair
`(source lineage, target lineage)`; endpoints never change within a lineage.
Revisions only toggle a `retracted` flag — re-pointing does not exist as an
operation (it is `unlink` + `link`, two lineages). `link()` is idempotent: a
live pair is a no-op, a retracted pair is revived, an absent pair founds a fresh
link lineage. `unlink()` retracts every live lineage of the pair; an absent or
already-retracted pair is a no-op. Nothing is deleted, so history and `as_of`
still see the earlier live rows. Duplicate pair lineages — two concurrent writers
minting the same pair, which no portable partial-unique index spans SQLite,
DuckDB, and Postgres to prevent — are **tolerated**: the pair is live if any
lineage is live, and `fsck` reports a multi-lineage pair as a repairable note,
not corruption.

**Store API.**

```python
store.link(result, "projects", project)     # idempotent assert
store.unlink(result, "projects", project)    # retract
targets = store.linked(result, "projects")   # latest target revisions
```

`linked()` returns the latest revision of each linked target lineage,
deduplicated by lineage and ordered by first-link order (stable across
retract+relink); `eager=True` materializes each instead of returning a lazy row.
Both endpoints must be stored in this store; the degraded write profile and an
open bulk-ingest context refuse `link`/`unlink`. Linking against an undeclared
target class raises `SchemaError`, and a target whose type does not match the
declaration raises `TypeError`.

`save()` and `replace()` accept a `links=` mapping of link name to a target or
iterable of targets; the save and every link commit in one atomic transaction:

```python
store.save(result, links={"projects": [project_a, project_b]})
```

Content dedup and links compose: saving duplicate content with `links=` reuses
the existing row and simply accumulates the associations, with no metadata
conflict.

**`.links` accessor.** A fetched (store-bound) record exposes
`record.links.<name>`, returning the same tuple as `store.linked()`, computed
lazily and memoized under the same staleness contract as reference-field
memoization (see [Lazy records](db-records.md#lazy-records)). It exists only on fetched rows:
a hand-constructed instance simply has no `links` attribute. One caveat — a
`save()` followed by a `fetch()` on the same handle hands back the
identity-cached *plain* instance (materialized-wins), which carries no `.links`;
`store.linked()` always works. Mongo reaches the same accessor through a thin
store-bound subclass.

**Query DSL.** Search variables expose a `v.links.<name>` namespace with
EXISTS/set semantics over the *latest* live-linked targets:

```python
r = search.variable(Result)
search.add(r.links.projects.name == "Ada")      # field chaining into the target, EXISTS
search.add(r.links.projects == stored_project)   # endpoint identity
search.add(r.links.projects.has_any(p1, p2))     # any live linked target among these
search.add(r.links.projects.has_only(p1, p2))    # every linked target among these (vacuously true with no links)
search.add(~r.links.projects.has_any(p1))        # set-wise negation
```

Each `v.links.<name>` access mints a *fresh* alias, so ANDed conditions express
HAS ALL: `(r.links.projects.name == "A") & (r.links.projects.name == "B")`
matches a source linked to both. Field chaining reaches scalar and encoded
fields of the target only; chaining deeper — into the target's references,
children, or its own links — raises `UnsupportedQueryError`. An identity RHS is a
stored object or a target search variable; a bare string raises `TypeError`
(pointing at `.id`/field chaining). Chained `== None` diverges by backend, the
same wrinkle child-field `== None` already has: on SQL a source with **no** live
links satisfies `v.links.<name>.<field> == None` (the LEFT-JOIN NULL row),
whereas on MongoDB it matches only a source with a live-linked target whose field
is null. Prefer `has_any()` or an explicit link-presence test when that
distinction matters. Link and target aliases are **always**
latest-filtered regardless of `only_latest` (that is what "weak" means), and
`as_of` is honored on both the link rows and the targets — a link or target
revision created after the cutoff is invisible. A bare `v.links.<name>` is also
usable as a `results()` **output**: it yields a tuple of the latest live-linked
targets per row, deduplicated by lineage and ordered by first-link order (the
same targets `store.linked()` returns), honoring `as_of` exactly like the
predicate form — a link or target revision created after the cutoff is absent
from the tuple. Declaring the output registers no join/lookup by itself (only
predicate use of a link set does); the resolution cost is one `linked()`-style
resolution per matched row, in addition to any query the link's own
predicates already run. Chaining into a target field before projecting it
(`results(x=v.links.<name>.<field>)`) stays rejected (`UnsupportedQueryError`),
like variable-length child projections — only the bare link set is a valid
output. Mongo limitation: `v.links.<name> == <target search variable>`
raises `UnsupportedQueryError`; a stored-object RHS works.

**Serving.** Only links declared `exposed_relationship=True` whose target class
is also served appear as OPTIMADE relationships, carrying `role`/`description`;
a retracted link disappears from them. The served id is lineage-level, so
revising a target keeps the same relationship id. `'<type>.id'` filters (HAS,
HAS ALL, HAS ANY, HAS ONLY) work over exposed weak links. This is route-aware:
in the library `optimade_filter_searcher` API these id-filters have always
worked; on the *served* OPTIMADE stored/federation route, `<type>.id`
relationship filtering only landed this series (before it, a bare
`references.id` there matched nothing). On the served route the same filter is
also reachable through the `_httk_relationships.<type>.id` alias. If a reference or child field
*and* an exposed weak link both target the same served class, id-filter binding
is ambiguous and raises `ValueError`. `StoredEntryFederation` collects these weak-link relationships for SQL-backed sources only; a Mongo-backed federation source does not yet serve link relationships (its per-row relationships channel is empty). Relationships always reflect the live link state regardless of a page's `as_of`: like the lineage-level in-store path, a retraction applies retroactively, so a historic page pairs its rows with the current link state. An unmapped target family falls back to its own served (wire) type name, never the internal one.

**Fingerprint.** The per-table schema fingerprint includes each link
declaration (name, target identity, `exposed_relationship`, role, description),
so any link change is non-additive: `upgrade=True` refuses it and the store must
be rebuilt (pre-release policy).

The `StrongLink` marker (see [Strong links](#strong-links-provenance-edges)) is
the opposite case for the marker itself: it is a code-only declaration, excluded
from both the schema fingerprint and the content identity, so declaring or
retyping an edge is invisible to storage. The one layout change it carries *is*
fingerprinted — `RunEdge` gained a composite `(entry_type, entry_id)` index — so,
exactly like the [`_httk_source_id` change](db-schema.md#vocabulary), an existing run-bearing store
opens against the new code as an incompatible layout and must be rebuilt (this
change is non-additive).

## Strong links (provenance edges)

`StrongLink(relationship, reverse=, role=, description=)` marks a run's
provenance edge fields (its inputs, artifacts, and outputs). Where a `WeakLink`
is mutable curation *outside* record identity (lineage-live), a `StrongLink` is
record content *inside* identity (revision-pinned): the edges are exactly the
ones a run's own revision declares. Unlike weak links the marker is code-only —
never persisted, excluded from the schema fingerprint and from content identity
(see [Fingerprint](#weak-links) above for the one indexed layout change).

**Serving.** The run family serves its edges as semantic OPTIMADE relationships
in both directions: forward keys (`_httk_has_input`/`_httk_has_artifact`/
`_httk_has_output`) on the run resource, and derived reverse keys
(`_httk_is_input`/...) on each targeted entry, with identical label/role payload.
Reverse blocks derive only from runs in the *same* source store, from the
latest-main revision per lineage; they are suppressed on a target's `~alts` and
carried lineage-level on `~revs`.

**Filtering.** The served relationships are filterable through the
`_httk_relationships.<key>.id HAS ...` extension (the semantic keys plus typed
aliases), part of the served-route relationship filtering that landed this
series.

**Mounted ids.** `StoredEntryFederation(source_inventory=...)` resolves relationship
ids using the target family's mount in the same store. `adapter_from_stores`
supplies this inventory automatically. Explicit `StoredEntrySource.relationship_sources`
selections (family class to source name) win, followed by the source itself for
same-family edges, a unique target mount, or a unique target mount with the same
prefix. Ambiguous mounts raise at configuration time. Forward and reverse
relationships, `include`, and relationship-id filters use the same selection;
unmounted loose targets retain their raw ids. An explicit override must name a
mounted source of the requested family in the same store.

Depth-1 related-property filters use these same target mounts. The factory
returned by `related_property_resolver_factory(plans)` accepts the source store
and an optional `RelationshipSourceMap`; federation supplies both. Each matching
candidate receives its concrete typed target backing's prefix before ids are
combined, so families sharing a wire type and raw id remain distinct. Unrelated
backings are excluded. The sibling search also applies the target prefix to
`id` equality and string matching; `immutable_id` stays intrinsic and unprefixed.
A direct `factory(store)` call without the map resolves
raw ids across the same-store plans. Both forms search latest main revisions;
named alternatives and stale revisions cannot satisfy the property filter.

**Accepted limitations.** The cross-store reverse gap (reverse derives only
within one source store); a backend with a custom `id_of` mapping
gets empty reverse blocks; `product_relationships()` emits a forward-only
`_httk_has_product` with no reverse.
