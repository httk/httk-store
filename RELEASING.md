# Releasing `httk-store`

Releases are built and published by GitHub Actions. PyPI authentication uses
Trusted Publishing, so the repository does not need a stored PyPI API token.

## One-time setup

1. Create accounts on [PyPI](https://pypi.org) and
   [TestPyPI](https://test.pypi.org), and enable two-factor authentication.
2. In the GitHub repository settings, create environments named `pypi` and
   `testpypi`. Configure a required reviewer for `pypi` (and optionally for
   `testpypi`); restricting the `pypi` environment to tags matching `v*` is
   also recommended.
3. On PyPI, add a pending GitHub Trusted Publisher with these values:

   - PyPI project name: `httk-store`
   - Owner: `httk`
   - Repository: `httk-store`
   - Workflow: `release.yml`
   - Environment: `pypi`

4. Add the corresponding pending publisher on TestPyPI, using the environment
   `testpypi` instead.

A pending publisher creates the project during its first upload. It does not
reserve the project name before then.

## Prepare and check a release

Set `project.version` in `pyproject.toml` to the version you intend to release,
then run the complete preparation from the working tree with Python 3.12+:

```console
make release-prepare VERSION=v2.1.0
```

`VERSION` is required and must equal `v` followed by `project.version`. A missing
or mismatched version stops preparation before any files are changed. The command does not
change the package version for you.

Preparation checks the current working tree in an isolated environment. It
ensures the documentation lock is current, refreshes the committed inventories
from published documentation, and runs CI, the normal tests on Python 3.12,
3.13, and 3.14, strict release documentation, distribution checks, a clean
locked documentation installation, and isolated package checks. Internal
`httk-*` dependencies and their versioned documentation
must already be published at the required versions; preparation uses the network
for dependency installation and inventory refreshes.

After preparation passes, review and commit the verified changes, including
any updated documentation lock and inventories. Follow the printed commands to
tag and push that commit and create its GitHub release. Preparation itself does
not commit, tag, push, or publish anything. If you change release inputs after a
successful run, run the preparation again before committing and tagging.

`make release-check` remains the local CI/documentation/distribution gate used
by the publication workflow. Its final output points to the full preparation
command; it does not replace the isolated and locked-installation checks in
`release-prepare`.

Versions on package indexes are immutable. Use a new release candidate version
when repeating an upload, for example `2.1.0rc1` followed by `2.1.0`.

## TestPyPI

Run the **Publish package** workflow manually in GitHub Actions. A manual run
publishes to TestPyPI only. To retry a TestPyPI upload without committing a version bump, pass the
optional `version_suffix` workflow input (e.g. `.post1` or `rc2`); it is
appended to `project.version` for that build only.
When the workflow run has completed (approving the
`testpypi` environment first, if it has a required reviewer), test the artifact
in a fresh environment:

```console
python -m venv /tmp/httk-store-test
/tmp/httk-store-test/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ httk-store==2.1.0
/tmp/httk-store-test/bin/python -c "import httk.store"
```

Replace `2.1.0` with the version being tested. Unlike `httk-core`, `httk-store`
has a runtime dependency (`httk-core`), so `--no-deps` is not appropriate here:
`import httk.store` pulls in `httk.core` at import time. The
`--extra-index-url` lets pip resolve that dependency (once it is published to the
real PyPI) while the package under test comes from TestPyPI.

## PyPI

1. Run `make release-prepare VERSION=v2.1.0`, then review and commit the verified
   working-tree changes.
2. Create the matching tag, `v2.1.0`, on that commit.
3. Push the commit and tag, then create and publish a GitHub release for that tag.
4. Approve the protected `pypi` environment.
5. Verify the release from a fresh environment with `pip install httk-store`.

The workflow rejects a Git tag that does not match `project.version`, rebuilds
the distributions from the tagged source, checks them, and publishes them via
PyPI Trusted Publishing.
