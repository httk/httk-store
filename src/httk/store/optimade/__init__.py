"""Public remote OPTIMADE client and query APIs.

The synchronous, read-only OPTIMADE client :class:`OptimadeStore` and its
portable :class:`RemoteSearcher` / :class:`RemoteResultSet` query layer for
accessing remote (federated) OPTIMADE services.
"""

from .client import (
    ALL_ADVERTISED,
    OptimadeClientError,
    OptimadeDiscoveryError,
    OptimadeErrorDocumentError,
    OptimadeHTTPError,
    OptimadeStore,
    OptimadeTransportError,
    OptimadeVersionNegotiationError,
    RemoteEntryType,
    ServiceDeviation,
)
from .remote_query import (
    CountUnavailableError,
    OptimadePaginationError,
    OptimadeResponseError,
    RemoteResultColumn,
    RemoteResultSet,
    RemoteSearcher,
)

__all__ = [
    "ALL_ADVERTISED",
    "CountUnavailableError",
    "OptimadeClientError",
    "OptimadeDiscoveryError",
    "OptimadeErrorDocumentError",
    "OptimadeHTTPError",
    "OptimadePaginationError",
    "OptimadeResponseError",
    "OptimadeStore",
    "OptimadeTransportError",
    "OptimadeVersionNegotiationError",
    "RemoteEntryType",
    "RemoteResultColumn",
    "RemoteResultSet",
    "RemoteSearcher",
    "ServiceDeviation",
]
