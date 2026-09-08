# httk-store

![Status: Early beta](https://img.shields.io/badge/status-early--beta-orange)

> **⚠️ EARLY BETA**
>
> This is an early beta release of *httk₂*. The organization of the packages
> and their APIs should not yet be regarded as stable, and may change between
> releases.

*httk-store* is a [*httk₂*](https://github.com/httk/httk2) module for data
management. Built on the stdlib-only contracts and models in *httk-core*, it
provides in-memory `httk.core.EntryProvider` implementations for the standard
OPTIMADE entry types (`references`, `files`, `calculations`),
property-definition validation on `jsonschema`, and a database storage layer
that stores plain frozen dataclasses in SQLite, DuckDB, PostgreSQL, or MongoDB,
makes them queryable through a backend-agnostic search DSL, and serves them
through the entry-provider contract. ClickHouse supports bulk ingestion and
read serving with its documented write restrictions.

Install the optional backend you need with `httk-store[db]` (SQLite),
`httk-store[duckdb]`, `httk-store[postgresql]`, `httk-store[mongodb]`, or
`httk-store[clickhouse,parallel]`. See the
[storage documentation](https://docs.httk.org/httk-store/db.html) for backend
capabilities and deployment requirements.
