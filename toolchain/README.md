# Shared browser toolchain

`config.json` selects the Emscripten release and Dawn WebGPU archive shared by
both runtimes. Each runtime still owns its upstream Git links and produces its
own Wasm instance. The TypeScript version used by project tooling belongs to the
private root `package.json` and its lockfile, not this configuration. Tool metadata records
the version of the TypeScript module actually loaded; dependency installation
is responsible for keeping the installed packages consistent with the lockfile.

The Emscripten Asyncify/BigInt correction is the existing four-guard change in
`patches/emscripten-asyncify-bigint.patch`. It preserves genuine i64 arguments
across 32-bit Asyncify rewind. The actual SDK fixture described below exercises
sleep and rewind with a large bigint; patch application alone does not prove
this behavior.

The declaration correction in `patches/emscripten-typescript-signatures.patch`
connects the SDK's `--emit-tsd` output to its existing native signature
conversions. Under MEMORY64, exports wrapped by the SDK use Number for p/P
pointer and int53 slots; unrelated i64 exports use primitive `bigint`. The
existing JSPI export selection still determines Promise returns. The generated
JavaScript and native ABI are unchanged. No post-link declaration rewriting is
needed.

Both SDK corrections use forward/reverse patch-context checks on private file
copies, then atomically replace their owned SDK sources. Repeated preparation is a
no-op; unsupported or ambiguous context fails without changing the SDK source.
They do not use whole-file source acceptance hashes. Each patch file's observed
identity belongs to compiler-cache partitions and build provenance. Use an SDK
owned by the tools root, not a symlink to a shared installation. The llama link
depends on the actual SDK generator and native signature source, so preparing a
warm SDK also regenerates declarations on the next ordinary build. Both runtimes'
links also depend on the SDK's Asyncify implementation.

`tests/tools/toolchain/tsd-sdk.test.ts` compiles an actual SDK fixture and imports
the resulting declarations with strict TypeScript. It checks Number allocation,
p/P conversions, primitive i64, selected JSPI promises, and actual BigInt
Asyncify rewind in Node. It does not require a browser or GPU. On an SDK update,
review the wrapper/declaration generation boundary; remove the patch and its
preparation when the SDK supplies these declarations itself.

## Offline preparation

Acquire the pinned SDK and Dawn archive first. CI performs downloads, cache
transfers, and Git operations explicitly in workflow YAML. The local preparation
tool does not access the network. It requires Node.js, the private tool
packages from the root lockfile, `patch`, `unzip`, and `zipinfo`.

When preparing a llama Wasm build, also acquire the SDK's npm development
dependencies before the offline phase:

```sh
npm ci --prefix "$PWD/.tools/emsdk/upstream/emscripten" \
  --include=dev --ignore-scripts --no-audit --no-fund
```

Emscripten's `--emit-tsd` uses its own TypeScript dependency and SDK Node. The
SDK's `package-lock.json` owns that compiler version; the root lockfile owns the
compiler used by project tooling. The llama build checks the SDK compiler's
availability before compiling runtime sources. Offline builds never install or
download these dependencies.

After compiling the private tools with `npm run tooling:build`, run this from
the repository root:

```sh
node build/tooling/tools/toolchain/cli.js prepare \
  --checkout-root "$PWD" --tools-root "$PWD/.tools"
source .tools/emsdk/emsdk_env.sh
```

The archive is expected at
`.tools/downloads/emdawnwebgpu_pkg-<dawnTag>.zip`. Its SHA-256 is checked against
`config.json` before extraction into a fresh staging directory. A failed check
or extraction leaves the previous port in place. Emscripten's release metadata
must match the configured version and release before its correction is applied.
The prepared port is written to `.tools/emdawnwebgpu_pkg`.

In CI, offline preparation, building, and validation run in network-disabled
containers through `.github/scripts/run-offline.sh` without pulling an image.
Workflow YAML supplies additional Docker options, the explicit image argument
and quoted `OFFLINE` command bodies. A missing local input is an error;
preparation never downloads one implicitly.

## Caches

Caches are optional accelerators. Cold builds use the same compiler, source,
patches, and checks; a hit never substitutes a previously built runtime.

| Data | Restore scope | Use |
| --- | --- | --- |
| Dawn archive | Exact pinned archive SHA-256 | Verify on every use, then extract afresh |
| Emscripten cache | Exact compiler, environment, runtime, profile, and variant partition; source-specific save keys | Reuse earlier source snapshots inside that partition; let the SDK check and resolve required cache entries; a miss retains the fresh SDK cache |
| ccache | Exact compiler, environment, runtime, profile, and variant partition; source-specific save keys | Permit earlier source keys only inside that partition and let ccache check individual commands and dependencies |

Successful push, pull request and manual runs can save on every branch unless
the workflow's `cache_mode` input is `cold` or the key was an exact hit. The cold
mode disables both restore and save. Saving remains subject to GitHub's
[ref-based cache access rules](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#restrictions-for-accessing-a-cache):
pull request caches belong to `refs/pull/<number>/merge` and can be restored by
reruns of that pull request. The workflow does not expand cache write permissions.
A restore is adopted only after the restore step
succeeds and its matched key belongs to the requested partition; a miss or
failure keeps the fresh cache. Cache keys describe the selected compiler/environment and runtime profile
partition; a hit remains an accelerator, not a source or artifact receipt.

The ccache configuration is created outside restored data and uses a
content-based compiler check, preprocessor mode, no sloppiness, and a 256 MB
local limit for each runtime/profile/variant. Generated bindings, build trees,
final JavaScript, Wasm, types, and model data are not cache inputs. Their
validation and publication remain separate from cache use.

Dawn's archive digest checks the downloaded bytes. Emscripten and ccache
entries depend on GitHub's cache scope and the key partition; they are not a
cryptographic proof of the build. Do not put secrets in caches. Measure cold
and warm runs before claiming a performance improvement.
