#!/usr/bin/env bash
set -euo pipefail

# Keep workflow-specific environment and commands in the calling YAML.
exec docker run \
  --network none \
  --rm \
  -i \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  -u "$(id -u):$(id -g)" \
  -v "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE" \
  -v "$RUNNER_TEMP:$RUNNER_TEMP" \
  -w "$GITHUB_WORKSPACE" \
  -e HOME="$RUNNER_TEMP/bicore-home" \
  -e CI=true \
  -e NPM_CONFIG_OFFLINE=true \
  -e GIT_NO_LAZY_FETCH=1 \
  -e GITHUB_WORKSPACE \
  -e GITHUB_OUTPUT \
  -e GITHUB_ENV \
  -e GITHUB_STEP_SUMMARY \
  -e GITHUB_REPOSITORY \
  -e GITHUB_RUN_ID \
  -e GITHUB_RUN_ATTEMPT \
  -e GITHUB_JOB \
  -e RUNNER_TEMP \
  -e RUNNER_OS \
  -e RUNNER_ARCH \
  "$@"
