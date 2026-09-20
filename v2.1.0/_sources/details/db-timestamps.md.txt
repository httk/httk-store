# Store timestamps

The examples use the `StructureRecord` class and database setup from
[Declaring, storing, and fetching records](db-records.md).

`SqlStore` enables store-managed timestamps by default:

```python
store = SqlStore(
    db,
    entry_records={},
    store_timestamps=True,
    store_timestamp_resolution=1_000,  # nanoseconds per stored unit; default: 1,000 (microseconds)
)
```

The stored value is `time.time_ns() // store_timestamp_resolution`. The searcher
API accepts a canonical nanosecond integer or an RFC3339/ISO-8601 timezone-aware
string and converts it to the store's units; the OPTIMADE filter form accepts
only the canonical nanosecond integer. For example, this historic query returns
rows present at `T`:

```python
searcher = store.searcher()
record = searcher.variable(StructureRecord)
searcher.output(record, "record")
searcher.add(record.store_timestamp <= "2026-01-01T00:00:00Z")
rows = searcher.results(record=record)
```

The equivalent OPTIMADE filter is:

```python
from httk.store.backend.sql import optimade_filter_searcher

rows = optimade_filter_searcher(
    store,
    StructureRecord,
    "_httk_store_timestamp <= 1767225600000000000",  # ns for 2026-01-01T00:00:00Z
)
```

`present at time T` means exactly `store_timestamp <= T`. FIRST-STORED-WINS
applies: a deduplication re-save does not replace the original timestamp, and
promoting a dependency to a main row does not replace it. One timestamp is
captured per save transaction and one per bulk batch, so all rows written by
that unit share its value.

Before capture, the writer checks a process-local high-water mark. A clock
regression smaller than 1 ms waits briefly when `clock_regression_grace=True`
(the default); larger regressions, or a failed grace wait, raise
`StoreClockRegressionError`. Set `clock_regression_grace=False` to skip the
wait, or `allow_clock_regression=True` to disable the guard. The mark is
per-process: reopening seeds it from stored rows, but it is not a cross-process
clock-coordination protocol.

`store.fsck()` checks for timestamps beyond the current clock plus the allowed
future slack. An administrative repair can clamp them:

```python
store.fsck(repair=True, clamp_future_timestamps=True, known_types=(StructureRecord,))
```

Clamping is destructive to historic-query fidelity; inspect a non-repair fsck
report and confirm the skew before using it.

The query stack also exposes `as_of=T` on stored-property federation
`query()`/`fetch()` and on the general `FederatedStore.searcher()`. The serving
layer accepts `_httk_as_of` and includes it in stable pagination plans. Stored
federation is availability-first: a source with `store_timestamps=False`
deliberately ignores the cutoff and serves that source's current state; sources
with timestamps enabled apply their own-resolution cutoff. Existing layouts do
not require an enable/disable migration for reading this capability.
