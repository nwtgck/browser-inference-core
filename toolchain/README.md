# Shared browser toolchain and cache contract

`config.json` pins Emscripten, the Dawn WebGPU port and the existing four-guard
Asyncify/BigInt correction. Both runtimes use these common pins. Their upstream
source revisions remain independent. Moving the setup and patch helper here does
not upgrade the compiler or expand the approved correction.

Acquisition is explicit in `.github/workflows/build.yml`: the compiler's fixed
GitHub repository, pinned release, Dawn download URL and archive checksum are
visible in the compile jobs. The same YAML shows the online Docker image build,
TypeScript acquisition and browser installation. Docker image build contexts
contain only their inline recipe, without repository source. Local preparation
never downloads missing inputs.
Use those acquisition commands when provisioning a development environment; an
offline environment must receive the same installed SDK and pinned Dawn archive.

After acquisition, from the repository root:

```sh
python3 -m build_tools.toolchain.prepare
source .tools/emsdk/emsdk_env.sh
```

In CI, local source preparation, cache admission, compilation, packaging, tests
and metric processing run through literal `docker run --network none` commands
in YAML. Each command uses the runner's user ID, explicit workspace and temporary
directory mounts, an explicit writable `HOME`, `--cap-drop=ALL` and
`--security-opt=no-new-privileges:true`. No offline wrapper or extra isolation
policy is involved. Git/npm/download and Actions cache transfers stay visible
outside those containers. Playwright browser binaries are acquired into
`.tools/browser-binaries`, so the same path is available in the offline container.

The host-code job acquires the pinned `typescriptVersion` into `.tools/typescript`
and passes its compiler path through `BICORE_TYPESCRIPT_COMPILER`; type tests never
install dependencies themselves.

Normal local builds do not require ccache. Cache configuration is an opt-in CI
step; without it, the same build scripts compile normally. The CI `--fresh` flag
removes only the chosen profile/variant build directory, before configuration.
It leaves the other pairs and source files alone.

## What is cached

| Data | Restore scope | Verification / use |
| --- | --- | --- |
| `.tools/downloads/emdawnwebgpu_pkg-<tag>.zip` | Exact pinned archive SHA-256 | Always hash before extraction; reconstruct the unpacked port |
| `.tools/emsdk/upstream/emscripten/cache` | Exact toolchain, runner image, runtime, profile/variant and build-input identity | Restored after a fresh SDK install, so a miss retains the populated SDK sysroot |
| `.cache/ccache/<runtime>/<profile>/<variant>` | Exact toolchain/config partition; previous source snapshots within that partition only | Emscripten wraps its actual Clang invocations with ccache; the driver always runs |

The SDK checkout/install, expanded Dawn directory, CMake/build trees, generated
bindings, final JavaScript/Wasm/types and model data are **not** cache paths.
Final build outputs pass through the existing Actions artifact and manifest
validation flow. A cache hit never changes `validation.realModelInference` or
skips build, source-cleanliness, patch, packaging or browser checks.

`ccache.conf` is recreated outside the restored cache: content-based compiler
identity, preprocessor mode (`direct_mode=false`), no sloppiness/path rewriting,
no hard-link/file-clone mode, and 256 MB maximum local cache size per build pair.
Compilation flags and preprocessed source/header contents are checked by ccache.
This conservative mode sacrifices some speed to avoid timestamp/direct-mode
header corner cases. The final link still runs in a freshly configured tree.

## Keys and concurrency

`build_tools/toolchain/cache.py` hashes common pins and setup/cache implementation files,
runner operating system, architecture and image version, Python and ccache
versions, and absolute workspace. It also hashes the offline execution
environment fingerprint: the public base image identities, exact Dockerfile
recipe, installed apt package versions and Node/npm/Python versions. The final
image build ID is not used because creation timestamps would defeat reuse even
when the execution environment is identical. Profile/variant settings form another partition.
Runtime CMake/build configuration, upstream pins and patch inventories are also
part of the Emscripten cache key. Object-cache restore scope deliberately survives
an upstream/source change; ccache decides which individual translations match.
Changing only the llama upstream pin does not invalidate the image cache.

GitHub cache entries are immutable. Each source commit gets its own ccache save
key; its restore prefix can reach only the same compiler/runtime/profile/variant
partition. Dawn and Emscripten have no broad fallback key. All concurrent workers
have separate local directories on separate runners. Shared Dawn keys can race
to save the same verified content; there is no shared writable CMake tree.
The `bic-v1` policy prefix is an explicit invalidation switch.

Caches are accelerators, not retained build evidence. Eviction, a cold start or
a missing entry must lead to an ordinary compilation, not a substituted old
runtime. No speedup is claimed without measuring a CI cache miss and hit. The
local 256 MB limit applies to each ccache shard, not to all remote snapshots;
repository cache storage and eviction policies still apply.

## Trust boundary

The restore and save steps are explicit in the workflow. Save steps run only after a successful
compile/upload on a push to the repository's current default branch, excluding
Dependabot. Each live-data check and each remote save has this guard.
PR, non-default source-branch and manual builds do not invoke the configured save
steps. They can read caches GitHub makes accessible to their ref. Neither the
publication job nor the privileged report workflow restores compiler caches.
No workflow is changed to `pull_request_target` to obtain broader cache access.

This is not a claim that arbitrary restored binaries are cryptographically
verified against source. Dawn is verified against an independently pinned hash;
Emscripten libraries and ccache objects rely on the trusted producer/ref boundary
as well as exact partitioning. Never place secrets in caches. GitHub's branch/ref
isolation remains part of this security assumption; do not grant lower-trust
code a cross-ref cache-write mechanism. A malicious default-branch change can
also change the build itself and is outside a cache-integrity guarantee.

A corrupt or unsafe Dawn archive fails before the existing unpacked port is replaced; it is not silently accepted or
allowed to publish. Delete that cache entry (or change the policy key after
review) and rerun from a cold cache. To troubleshoot other intermediates, select the explicit `cold` cache mode or bump the cache policy prefix.
Do not bypass provenance or license validation to accommodate a cache hit.

## References

- GitHub Actions dependency caching: https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching
- Emscripten compiler caching: https://emscripten.org/docs/compiling/Building-Projects.html
- Emscripten `EM_CACHE`: https://emscripten.org/docs/tools_reference/emcc.html
- ccache compiler checks and preprocessor mode: https://ccache.dev/manual/latest.html

These are implementation background, not evidence that this workflow has
completed a hosted CI run or real-model WebGPU inference.
