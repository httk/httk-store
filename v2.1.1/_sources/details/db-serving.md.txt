# Serving through OPTIMADE

The example uses the `StructureRecord` and `Author` classes and `store` from
[Declaring, storing, and fetching records](db-records.md).

`StoreEntryProvider` bridges a store to the `httk.core.EntryProvider`
contract: it auto-generates an OPTIMADE entry-type definition per served class
from its schema (every schema-derived property named with a registered
database-specific prefix, `_httk_` by default), yields JSON-able records, and
declares relationships for reference fields — and for exposed weak links (see
[Weak links](db-relationships.md#weak-links)) — whose target class is also served. It also serves
the run family's `StrongLink` provenance edges (see
[Strong links](db-relationships.md#strong-links-provenance-edges)): the forward keys on the run
records, and the derived reverse keys on every served target family:

```python
from httk.store.backend.sql import StoreEntryProvider

provider = StoreEntryProvider(store, {"structures": StructureRecord, "authors": Author})
```

Handing the provider to *httk-serve*'s `adapter_from_providers` serves the
database as an OPTIMADE API. *httk-store* does not depend on *httk-serve*: the
provider handoff uses the httk-core contract, while *httk-serve* also consumes
*httk-store*'s neutral query and store APIs. Fields with no OPTIMADE value
representation (`bytes`, custom codecs) are not served, and rationals are
served as their nearest floats. The provider is also registered (as
`store-db-store`) for discovery through the `httk.core` registry.

Each served record also exposes its lineage identity as the integer property
`_httk_logical_id` (see [Record replacement and lineages](db-revisions.md#record-replacement-and-lineages)),
filterable like any other served field. Pass `only_latest=True` to
`StoreEntryProvider` to serve only the latest row of each lineage.
