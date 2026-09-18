# Captured OPTIMADE `/info/structures` documents

Real provider responses captured 2026-09-18, used as test fixtures for
standard-property-name schema completion. None of them carries a `$id`
property definition.

| file | source URL | `meta.api_version` | prefixed properties kept |
|---|---|---|---|
| `alexandria_pbe_info_structures.json` | https://alexandria.icams.rub.de/pbe/v1/info/structures | 1.1.0 | 3 |
| `materials_project_info_structures.json` | https://optimade.materialsproject.org/v1/info/structures | 1.2.0 | 2 |
| `oqmd_info_structures.json` | https://oqmd.org/optimade/v1/info/structures | 1.2.0 | 3 |

Trimming applied to keep fixtures small and reviewable, recorded here because
the files are therefore not byte-exact captures:

- `data.properties` keeps every unprefixed (standard-namespace) property plus
  at most the three alphabetically first provider-prefixed properties (a
  provider advertising fewer keeps all of them; Materials Project has two).
- `data.output_fields_by_format` removed.
- `meta` reduced to `api_version`.
- Every `description` string longer than 60 characters is truncated and marked
  `[truncated for test fixture]`.

Everything else in `data` is verbatim. In particular, the 1.1.0 documents
(Alexandria, COD) genuinely carry no `data.id`/`data.type` resource identity,
and Materials Project genuinely carries `data.id` but no `data.type`; tests
that need the 1.2 resource-identity check to pass add `type` when serving.
All machine-read property fields (`$id` where present, `x-optimade-type`,
`type`, `items`, `x-optimade-implementation`, `x-optimade-support`,
`x-optimade-unit`) are preserved verbatim.
