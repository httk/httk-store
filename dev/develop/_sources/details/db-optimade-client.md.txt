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

The store creates its own HTTP client with a request timeout of 120 seconds
(public providers routinely take several seconds per filtered query);
`OptimadeStore(url, timeout=300)` changes it and `timeout=None` disables it.
A borrowed `client=` keeps its own timeout configuration.

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

## Non-conforming services

Some public providers deviate from the specification in ways that would
otherwise fail construction or paging. By default the client applies a
specification-anchored fallback for three such deviations — each a fallback
whose correctness follows from the specification itself, not a provider-specific
quirk table — and records every one it applied on `store.deviations`, a tuple of
frozen `ServiceDeviation(kind, url, detail)` records. Each newly recorded
deviation is also emitted once, per `(kind, url)`, through the report channel as
a `logging` warning tagged with the context `optimade`:

- `"versions-endpoint"` — the unversioned base returns HTTP 404 for
  `/versions` (which *Materials Project* does). The specification places major
  version 1 at `/v1`, so the client probes `<base>/v1/info`; when that is a
  valid `/info` declaring a major-1 service, it uses `/v1`. Any 404 with no
  such confirmation, and every non-404 status, re-raises unchanged.
- `"entry-info-identity"` — an `/info/<entry>` document omits the 1.2 resource
  `type` member but its `data.id` equals the endpoint name (again *Materials
  Project*). Identity is then established from `data.id`. An absent or
  mismatching `id`, or a present-but-wrong `type`, stays an error.
- `"continuation-scheme"` — a `links.next` continuation differs from the base
  origin only by using `http` where the service is `https`, on the same host
  and default ports (also *Materials Project*, whose `http` links answer a
  redirect). The link is upgraded to `https` (over the base's authority, so an
  explicit port on the `http` link cannot survive as a wrong one) and paging
  continues. Schemes are never downgraded, and any other origin difference
  still needs `allow_cross_origin_pagination`. This deviation is recorded once
  per service, its `url` the service base URL rather than any one continuation.

```python
store = OptimadeStore("https://optimade.materialsproject.org")
for deviation in store.deviations:
    print(deviation.kind, deviation.url, deviation.detail)
```

Pass `tolerate_deviations=False` to switch every fallback off and restore the
strict errors, for conformance auditing: the missing `/versions` raises
`OptimadeHTTPError`, the identity-less `/info/<entry>` raises
`OptimadeDiscoveryError`, and the `http` continuation raises
`OptimadePaginationError`.

```python
store = OptimadeStore(base_url, tolerate_deviations=False)  # fail strictly
```

A fourth provider deviation — a `last_modified` timestamp served without a UTC
offset — has no defined instant, so it is not tolerated here; the entry
backends in *httk-core* and *httk-atomistic* decide how such a naive value
decodes.
