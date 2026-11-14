#!/usr/bin/env bash
set -euo pipefail

# Pass additional Docker options followed by the prepared image; quoted stdin
# runs in strict Bash.
# offline_docker_args=(
#   -e HOME=/tmp
# )
# .github/scripts/run-offline.sh "${offline_docker_args[@]}" bicore-offline:local <<'OFFLINE'
#   mkdir -p "$HOME"
#   node --version
# OFFLINE

# Isolation limits:
# --network none disables ordinary external IP connectivity. Bind-mounted Unix
# sockets may still provide access to host services or network proxies.
# Workspace and runner-temp files are writable and may affect later online steps.
# Additional Docker options come from trusted workflow code; this helper is not
# a security boundary against a malicious caller.
exec docker run \
  --pull=never \
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
  -e PLAYWRIGHT_BROWSERS_PATH="$GITHUB_WORKSPACE/.tools/browser-binaries" \
  -e LCB_SOURCE_COMMIT \
  -e BIC_METRICS_MODE \
  -e BIC_METRICS_RUNTIME \
  -e BIC_METRICS_PROFILE \
  -e BIC_METRICS_VARIANT \
  -e BIC_OFFLINE_ENVIRONMENT \
  -e BIC_CACHE_MODE \
  -e EM_CACHE \
  -e EM_COMPILER_WRAPPER \
  -e CCACHE_DIR \
  -e CCACHE_CONFIGPATH \
  -e BIC_CACHE_ID \
  "$@" bash -euo pipefail
