# Optional WebGPU parameter upload batching

`LCB_WEBGPU_PARAM_UPLOAD_BATCHING` defaults to OFF. Enable it explicitly with `python3 scripts/build.py --profile webgpu-wasm64-jspi --variant browser --webgpu-param-upload-batching`, or the equivalent supported WebGPU profile. CPU profiles reject the option before invoking the toolchain. Direct CMake builds can set `-DLCB_WEBGPU_PARAM_UPLOAD_BATCHING=ON`.

The patch is source-bound to llama.cpp d81235049384534c167caea52b85a694f6103d14 and the pinned Dawn headers. `prepare_webgpu_source.py` composes this patch and the independently optional tensor-copy patch into exactly one private translation unit. Both flags can be OFF, either can be ON, or both can be ON. Each enabled output has a fixed reviewed hash. CMake substitutes the target source only once; MoE header overlay include priority is retained. The vendor checkout is not edited.

The batching guard is a private definition on the WebGPU target only when enabled. Reconfiguring OFF selects the pristine source or copy-only source and removes the definition. Command provenance records both flags explicitly. Source reports record the actual combined byte identity and compile definition per enabled profile/variant, so a copy-only identity cannot stand in for a combined build.

The build uses an aligned CPU parameter mirror and uploads it at the existing command-submission boundaries. No waits, readbacks, command cadence, buffer bindings, or slot resets are removed. Same-queue upload A / submit A / upload B / submit B order must be preserved. Extra CPU storage is 74 aligned parameter slots (18,944 bytes at 256-byte alignment); transferred padding increases upload bytes. Fewer API calls are not a measured speedup.

Manual CI has independent boolean experiment inputs. Either enabled experiment makes the run artifact-only, and the publisher separately rejects enabled or ambiguous experimental flags. Ordinary build defaults and consumer publication remain unchanged. Upstream updater preflight reports optional combination failures separately without preventing default builds.

## Artifact-only CI trial

1. Commit the reviewed candidate on a feature branch and manually run **Build and publish runtime** for that exact commit. Set `webgpu_param_upload_batching` to true; set `webgpu_tensor_copy` independently to false or true for the desired comparison. A normal push run leaves both options OFF.
2. After the selected run succeeds, download `llama-runtime-package`, `image-runtime-package`, and `webgpu-source-provenance` from that same run. Keep the run URL and immutable head commit with the files. No ordinary artifact publication or consumer lock fragment is produced for an enabled experiment.
3. Verify the package manifest's source commit, upstream/toolchain identities, payload hashes and per-profile CMake flags. Compare the report's compiled-copy hashes and definitions to the selected mode. Compare only runs with otherwise matching profile and variant settings.
4. The downloaded language runtime is an inner format-2 package, not the format-3 combined package expected by consumers of this repository's root package. The disabled publish job also skips root assembly. Extract the language and image artifacts into `build/experimental-package-inputs/llama-cpp/` and `build/experimental-package-inputs/stable-diffusion-cpp/` respectively, retaining the manifest at each directory root. From the exact candidate source checkout, assemble and validate without publishing:

   ```sh
   python3 scripts/package_runtime.py --inputs build/experimental-package-inputs --output dist/experimental-package
   npm pack ./dist/experimental-package --pack-destination dist
   ```

   Assembly rejects mismatched source commits and checks the full payloads; do not mix the image runtime from another run. The result is a format-3 manifest and a local npm tarball, with no published artifact commit.
5. If a consumer trial is separately approved, use that verified local tarball in an isolated consumer branch. Review exact generated-code compatibility and integrity mappings rather than bypassing guards or substituting a source commit for an unpublished artifact commit. The existing Git-pinned dependency cannot resolve this unpublished tarball automatically. A separate reviewed dependency/lock and exact-source adapter/integrity update is required for the local trial. Retain the ordinary published dependency for production.

## Checks and adoption limits

Run `LCB_TEST_LLAMA_SOURCE=/path/to/upstream-d812 LCB_TEST_DAWN_PACKAGE=/path/to/emdawnwebgpu_pkg python3 -m unittest discover -s llama-cpp/tests -p 'test_webgpu*overlay.py'` from the source repository root. These tests configure the actual upstream WebGPU target, verify all four source combinations, compile definitions, repeated reconfiguration, MoE coexistence, input/patch identities, and unchanged source bytes. They do not build or execute a WebGPU runtime.

The [standalone source checks](../experiments/webgpu-param-upload-batching/README.md) exercise the actual arena and graph loop with a CPU queue model, and can check actual Dawn API declarations. Tensor-copy tests separately exercise the actual copy hook.

Before adoption, complete the pinned Emscripten/Wasm build and browser/device GPU correctness, lifetime, numerical-equivalence and performance tests, including the combined mode and supported profiling configurations. Public API, model and checkpoint semantics are unchanged. No production-readiness or performance claim follows from host tests.

Upstream changes require review of source, parameter consumers, submit/reset boundaries and both patch compositions; textual application alone is insufficient. Keeping upstream behavior or waiting for upstream batching are valid no-patch alternatives. Remove the experiment and integration when upstream provides suitable support or the measured benefit no longer justifies maintenance.
