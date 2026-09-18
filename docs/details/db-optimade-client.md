# Reading a remote OPTIMADE service

`OptimadeStore` connects synchronously to a read-only remote (or federated)
OPTIMADE service and discovers it eagerly. It is a data-management capability
in *httk-store*: it reads other people's services, the mirror image of
[Serving through OPTIMADE](db-serving.md#serving-through-optimade), and it has
no dependency on *httk-serve*.

```python
from httk.store.optimade import OptimadeStore

with OptimadeStore("https://alexandria.icams.rub.de/pbe/v1") as store:
    print(store.api_version)                      # negotiated specification version
    structures = store.entry_type("structures")   # one discovered endpoint
    print(structures.backend.__name__)            # OptimadeStructure
```

Construction negotiates a supported API version (major 1) through the
service's `/versions` list when the base URL is unversioned, then reads
`/info` and each advertised `/info/<entry_type>` document to build one
immutable `RemoteEntryType` per endpoint. Each descriptor carries the
endpoint's transport name, its advertised property names, and — when the
endpoint is recognized — a semantic `binding` and the `backend` class that
`binding` resolves to (for example `OptimadeStructure`).

## Binding entry types

An endpoint is bound to a typed backend through three tiers, tried in
descending precedence. The descriptor records which tier succeeded in
`RemoteEntryType.binding_evidence`:

- `"declared"` — the endpoint's `/info` `links.describedby` names a known
  entry-type definition IRI.
- `"property-ids"` — the endpoint declares an unambiguous set of property
  definition `$id` IRIs owned by exactly one known entry type.
- `"standard-name"` — no `links.describedby` is declared and the declared
  property `$id` evidence is absent or merely ambiguous (not contradictory),
  but the endpoint's own name is a standard entry type (`structures`,
  `references`, `files`, `calculations`) and the service declares a
  specification version. This tier is the reason a plain `structures` endpoint
  binds to `OptimadeStructure` even when the service publishes no `$id` at all.
  A standard endpoint that advertises no properties at all still binds by name
  and then simply identifies nothing.

An unrecognized endpoint stays generic: its `binding` is `None`,
`binding_evidence` is `None`, and its `backend` is the source-exact
`OptimadeResource`. Provider-specific endpoints (`_exmpl_things`) always stay
generic. Two declared signals also keep an endpoint generic and are never
overridden by the name tier: mutually exclusive declared property `$id`s (a
contradiction, e.g. a files-only and a references-only IRI on one endpoint),
and a `links.describedby` naming an entry-type definition the client does not
recognize — a positive foreign claim, so a service declaring an unrecognized
`links.describedby` is not bound by name even under a declared version.

The transport names whose property identity came from the standard-name tier —
rather than a declared `$id` — are listed, sorted, in
`RemoteEntryType.inferred_properties`, so it is always visible exactly which
fields were name-completed:

```python
structures = store.entry_type("structures")
assert structures.binding_evidence == "standard-name"
assert "species" in structures.inferred_properties
assert "_alexandria_band_gap" not in structures.inferred_properties  # provider-prefixed
```

## The standard-namespace rule

Standard-name completion applies the OPTIMADE specification's own namespace
rule rather than guessing from spelling. On a standard endpoint, an unprefixed
property name is the standard property of that name as of the version the
service declares in the info document's own `meta.api_version`. A declared
`$id` always wins and is never overridden; a provider-prefixed name
(`_alexandria_band_gap`) carries no standard meaning and stays unknown; a name
introduced only in a later specification version than the one declared stays
unknown (the version gate); and a service that declares no version gets no name
completion at all. The version that governs completion is the entry info
document's own `meta.api_version`; its absence disables completion for that
endpoint even when the top-level `/info` declares a version. The completed
identities feed both typed querying and portable decoding, so the standard
properties of every conforming provider — not only *httk₂*-served ones —
become usable.

## Strict definition-only discovery

Pass `infer_standard_definitions=False` to switch the standard-name tier off
entirely. The store then recognizes a property only through a declared `$id`
and an endpoint only through `describedby` or property `$id`s — the strict
definition-only behaviour, useful when auditing what a federation actually
publishes. The flag governs discovery: binding, `inferred_properties`, and the
typed query fields a searcher exposes. It does not affect an entry backend
constructed directly over a raw `OptimadeResource` (for example
`UnitcellStructureView(resource)`), which always applies the standard-name
rule from the resource's own schema.

```python
store = OptimadeStore(base_url, infer_standard_definitions=False)
structures = store.entry_type("structures")
assert structures.binding is None            # no $id, so unrecognized
assert structures.inferred_properties == ()
```

## Querying a remote service

A `searcher()` builds one portable single-root query. Binding the query
variable to a name-completed endpoint exposes its typed standard fields, while
provider-prefixed properties remain queryable and readable under their exact
wire names. On a bound row a decoded typed property (`chemical_formula_reduced`)
and a raw provider extension (`_alexandria_band_gap`) are both reached by
attribute — standard-name completion adds identities but never withdraws a
provider field:

```python
search = store.searcher()
s = search.variable(structures.backend)       # or search.variable(structures)
search.add((s.nelements == 2) & s.elements.has("Na"))
for row in search.results(structure=s):
    print(row.structure.chemical_formula_reduced)  # decoded typed value
    print(row.structure._alexandria_band_gap)      # raw provider-extension value
```

Reaching a decoded typed value by attribute works where the backend exposes
that property, as `OptimadeStructure` does for its full quartet and scalars.
The entry backends (`OptimadeReference`, `OptimadeFile`, `OptimadeCalculation`)
expose only their portable fields as attributes; reach the rest through the
canonical view of the row (for example `FileView(row.file).url`). A recognized
standard name is never exposed as a raw attribute, so it cannot leak a raw
value under its standard spelling.
