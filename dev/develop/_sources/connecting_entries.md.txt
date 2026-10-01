# Connecting entries

*httk-store* has three ways to connect stored entries. They can all appear as
OPTIMADE relationships, but they encode different facts:

- a **record-valued field** says that another exact record is part of this
  record's value;
- a **strong link** says that this revision makes an immutable, semantically
  named claim about another entry identity;
- a **weak link** is editable curation between two entry lineages.

Choose from the data semantics first. OPTIMADE serving follows from that
choice.

## At a glance

| | Record-valued field | `StrongLink` | `WeakLink` |
| --- | --- | --- | --- |
| Declared as | A dataclass field containing another storable class, or a list/tuple of one | An annotated field containing edge records with `label`, `entry_type`, and `entry_id` | A class-level `StorageInfo.links` entry; no record field |
| Stored endpoint | The exact target row (`sid`) | The target's entry type and public `id` | The source and target lineage ids (`logical_id`) |
| Part of source `content_id` | Yes, through the target's content id | Yes, through the stored edge values | No |
| When the target is revised | The source still reconstructs the exact old target revision | The edge still names the same public id, shared by the target lineage's revisions | The link resolves to the latest target revision |
| Changing the connection | Replace/save a new source record | Replace/save a new source record with different edges | Call `link()` or `unlink()` without replacing either endpoint |
| Query path | `result.structure.formula` | `run.links.has_input`; compare with a separate target variable | `result.links.projects`; scalar/encoded target fields can be chained |
| OPTIMADE form | Forward relationship grouped under the target's served type | Semantically named forward relationship and optional derived reverse relationship | Relationship grouped under the target's served type when explicitly exposed |

"Revision-pinned" therefore has two related meanings. A record-valued field
pins the exact target revision. A strong link pins the *edge set* to the source
revision, but its target is a public entry id, not an `immutable_id`: target
queries can match all revisions unless they request `only_latest=True`.

## Record-valued fields: exact composition

Any frozen dataclass with storable fields can be a target; it does not need a
storage base class. A singular target becomes a reference field, while a
`list` or homogeneous `tuple` of targets becomes an ordered child field. Saving
the source recursively saves its targets.

```python
from dataclasses import dataclass
from typing import Annotated

from httk.core.storage import Related


@dataclass(frozen=True)
class Structure:
    formula: str


@dataclass(frozen=True)
class Result:
    structure: Annotated[
        Structure,
        Related(role="subject", description="Structure evaluated by this result"),
    ]
    energy: float
```

`Result.structure` is a normal Python `Structure` object after reconstruction.
The parent stores the target `sid`, and its canonical identity contains the
target's content id. If `Structure("Si")` is later replaced by another
structure revision, an existing `Result` continues to reconstruct with the
original revision. A result for the replacement structure has different
content and must be saved as a new result or result revision.

When `Structure` and `Result` are both configured as served entry families,
the field automatically appears as an OPTIMADE relationship under the target
family's served type. `Related(...)` is optional: it adds `role` and
`description` metadata, while `Related(serve=False)` suppresses serving without
changing storage. Multiple record-valued fields targeting the same served
family share that relationship block.

Record references also give the richest target query path:

```python
search = store.searcher()
result = search.variable(Result)
search.add(result.structure.formula == "Si")
rows = search.results(result=result)
```

Use a record-valued field when the target is needed to reconstruct the source
value, or when the exact target revision matters.

## Strong links: immutable semantic edges

A `StrongLink` field contains edge records rather than target objects. Each
edge records a label, the target's internal entry-type name, and its raw public
id. The built-in `RunEdge` has exactly that shape:

```python
from dataclasses import dataclass
from typing import Annotated

from httk.core import RunEdge
from httk.core.storage import StrongLink


@dataclass(frozen=True)
class WorkflowRun:
    source_id: str
    inputs: Annotated[
        tuple[RunEdge, ...],
        StrongLink("has_input", reverse="is_input", role="input"),
    ] = ()
    outputs: Annotated[
        tuple[RunEdge, ...],
        StrongLink("has_output", reverse="is_output", role="output"),
    ] = ()


run = WorkflowRun(
    source_id="workspace-7:job-19",
    inputs=(RunEdge("structure", "structures", "example-structures-42"),),
    outputs=(RunEdge("relaxed", "structures", "example-structures-57"),),
)
```

The target is not recursively saved or reconstructed through the field:
`run.inputs[0]` is a `RunEdge`, not a structure. Strong-link targets use public
ids from the same store; cross-provider linking is not supported. Because the
edge values are record content, changing an id, type, or label changes the
source content identity. The `StrongLink` marker itself is code-only serving
metadata and does not enter content identity or the persisted schema
fingerprint.

The declaration gives the relationship semantic names. With the usual `httk`
provider prefix, the example serves `_httk_has_input` on the run and derives
`_httk_is_input` on each target. Reverse edges are computed from stored forward
edges; they are never stored separately. Labels and the marker's role and
description are included in the relationship identifiers.

Strong links use the shared `links` query namespace:

```python
search = store.searcher(only_latest=True)
run = search.variable(WorkflowRun)
structure = search.variable(Structure)
search.add(run.links.has_input == structure)
search.add(run.links.has_input.label == "structure")
rows = search.results(run=run, structure=structure)
```

An edge can target different configured entry types, so target fields cannot
be chained as `run.links.has_input.formula`. Bind a separate target variable as
above. Forward traversals can inspect the edge's own `label`, `entry_type`, and
`entry_id`; reverse traversals cannot chain edge fields.

Use a strong link for immutable provenance such as "this run consumed entry
X" or "this data record is a product of entry Y": the claim belongs to the
source revision, while the endpoint is an entry lineage identified by its
public id.

## Weak links: editable, lineage-following curation

A `WeakLink` is declared on the source class but is not one of its fields. It
lives in a store-managed link table and connects the two lineages. Adding,
retracting, or restoring it never changes either endpoint's `content_id`.

```python
from dataclasses import dataclass
from typing import ClassVar

from httk.core.storage import StorageInfo, WeakLink


@dataclass(frozen=True)
class Project:
    name: str


@dataclass(frozen=True)
class ResultIndex:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        links=(
            WeakLink(
                "projects",
                target=Project,
                exposed_relationship=True,
                role="belongs-to",
                description="Project containing this result",
            ),
        )
    )

    name: str
```

After both records have been saved, manage the connection independently:

```python
store.link(result, "projects", project)       # idempotent
projects = store.linked(result, "projects")   # latest Project revisions
store.unlink(result, "projects", project)     # retracts; does not delete history
store.link(result, "projects", project)       # restores the pair
```

`link()` and `unlink()` append link revisions. The pair is lineage-level, so
replacing `project` does not require relinking: `store.linked(...)` resolves
the new latest project revision. Use `as_of` on store queries when historic
link state is required.

Queries also use `links`, but weak links know one statically declared target
class and can chain its scalar fields:

```python
search = store.searcher(only_latest=True)
result = search.variable(ResultIndex)
search.add(result.links.projects.name == "screening-2026")
rows = search.results(result=result)
```

`exposed_relationship=False` is the default. Set it to `True` only when the
curation should be public over OPTIMADE. The served relationship is grouped
under the target family's type; the weak-link name is carried as the
relationship label. If a record field and an exposed weak link both target the
same served class, relationship filter binding is ambiguous and adapter
configuration raises `ValueError`.

Use a weak link for membership, tagging, ownership, or other curation that an
operator may change without asserting a new scientific record revision.

## Serving requirements and deeper details

The declarations above are sufficient for storage. To expose an endpoint over
OPTIMADE, both sides must also belong to configured served entry families and
carry public ids. Decorated application records can use `records=`; existing
domain records use `entry_records=` or `entry_families=`. See
{ref}`the application-record example <serving-application-records>` and
{doc}`details/db-serving`.

The detailed relationship reference covers link history, set semantics,
federation, mounted ids, relationship filters, reverse-edge limitations, and
backend-specific behavior: {doc}`details/db-relationships`.
