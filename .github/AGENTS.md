# Workflow and shell checks

Workflow orchestration and `.github/scripts/` shell boundaries are tested in
`.github/tests/`. Keep workflow-specific support beside those tests using the
`*.test-support.ts` suffix. Unit tests for metrics and automation implementation
remain under `tests/tools/`.

Run `CI=true npm run test:workflows` from the repository root. It strictly emits
the current source and runs the workflow tests without real Docker, browsers or
network access. CI includes these tests in its existing single tooling emit and
Node invocation. Calling syntax and isolation limits belong in the shell script
header; do not duplicate them here.
