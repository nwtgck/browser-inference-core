# Native chat and multimodal bindings

This repository separates the heavy Wasm build and artifact publication from
application builds. It exposes upstream operations; applications own history, media input,
workers, generation, sampling, stopping, approval, tool execution, and cancellation.
Jinja and model-specific output parsing stay in upstream llama.cpp.

## Generated module

The Emscripten module works independently of the always-shipped `examples/runtime/`.
Applications can use that tested example to implement their own TypeScript host code:

```js
import createNative from './profiles/cpu-wasm32/browser/core.mjs';
const native = await createNative();
const inputs = new native.common_chat_templates_inputs();
const messagesJSON = native.common_json.parse('[{"role":"user","content":"Hello"}]');
const messages = native.common_chat_msgs_parse_oaicompat(messagesJSON);
inputs.messages = messages;
// Create native.common_chat_templates(modelPointer, templateOverride, bos, eos),
// then call templates.apply(inputs). Release every owned Embind object.
messages.delete();
messagesJSON.delete();
inputs.delete();
```

Emscripten's `--emit-tsd` writes each profile's `core.d.ts` alongside `core.mjs`;
the package's profile import resolves those generated types. Embind registers the upstream message, tool, input, result, diff, delimiter,
parser-parameter, and PEG-arena types, plus their vectors, maps, and enums.
`common_json.parse()` / `dump(-1)` and the upstream OpenAI-compatible helpers
handle JSON. There is no lcore JSON field mapper or alternate message schema.
Constructors retain upstream defaults, including empty message/tool vectors.
`chat_template_kwargs` is the upstream string map: its values are serialized JSON.
Clock duration/time-point types expose `now` without a custom timestamp format;
`system_clock_period_num()` / `system_clock_period_den()` return exact integer
duration ratios. Upstream Jinja currently
recomputes its clock and overrides `add_bos`/`add_eos`.

`common_chat_templates` is a small owner for upstream's opaque, custom-deleter
resource. Its methods directly call upstream apply/source/capability/formatting
operations. `delete()` invokes upstream cleanup. Supply every bound argument;
C++ default arguments do not become JavaScript defaults. Its model argument is a checked
`bigint` pointer from the C API; string overrides are ordinary JS strings.

Embind owns returned value objects and STL containers; call `delete()` on them.
Class-valued properties and vector elements use standard copy behavior: modifying
a retrieved copy does not update its parent until assigned back. String conversion
uses Embind's standard allocation handling. Enum values are standard Embind enum
objects, not lcore integers or booleans. Size-based values follow the selected ABI;
compare absent diff indices with `common_chat_no_tool_call_index()`, which returns
the native `size_t` sentinel in the selected ABI's integer representation.

Keep a `common_peg_arena` and call `load()` once, then reuse it with
`common_chat_peg_parse(arena, cumulativeText, isPartial, parserParams)`. This passes
the existing native arena by const reference. Assigning `parserParams.parser`
copies it once; subsequent `common_chat_parse()` calls reuse that member. The caller
controls the cumulative text and previous parsed message. Full apply results expose
grammar/lazy triggers, generation prefixes, stops, preserved tokens, reasoning
metadata, serialized parsers, and delimiters; retain the fields required by the loop.

JSON-schema grammar conversion and reasoning-budget sampler primitives are also
bound. Sampler pointers use the C API's destructor or transfer to a sampler chain.
`common_reasoning_budget_get_end_match_copy()` returns an owned token-vector copy,
empty when no end marker has matched. Call `.delete()` on this copy after use. It
remains valid after sampler mutation or release; changing it does not alter the
sampler. The old borrowed-reference getter is not exported. No reasoning policy or
generation loop is added by these bindings.

Native chat calls are synchronous and may throw. Raw C calls retain their own
signatures and profile-specific suspension behavior. For Asyncify, use
`ccall(..., { async: true })` to wait for completion; awaiting a raw export does
not wait for a suspended operation. The example loader configures this automatically.
Applications must sequence all access to one instance,
including Embind calls during pending GPU operations. The optional reference
`core.api` Promise wrapper and its busy guard apply only to calls through that wrapper.

## Explicit overload contracts

Every named C++ function pointer directly registered in `bridge/chat-embind.cpp`
uses Embind's standard `select_overload`, including currently non-overloaded free
functions, static methods, and instance methods. The signature is a handwritten
contract for the existing JavaScript surface, not a type deduced from whichever
upstream function happens to be available. For example:

```cpp
function("common_chat_verify_template",
    select_overload<bool(const std::string &, bool)>(&common_chat_verify_template));
// Constness and the owning class are explicit for member functions.
class_<common_json>("common_json")
    .function("dump", select_overload<std::string(int) const, common_json>(&common_json::dump));
```

The owning class is intentional. Specifying only `select_overload<Signature>` for
an instance method still asks the compiler to deduce `ClassType`. A future
same-name member function template can make that deduction ambiguous even when
the desired method remains. `select_overload<Signature, ClassType>` removes that
last inference. Static functions use the ordinary free-function-pointer form.
`select_const` alone does not specify argument/return types and is not a substitute.

No custom dispatcher, overload registry, name-based fallback, or generated
adapter is introduced. The implementation retains names, argument order, native
widths, const/reference qualifiers, return values and Embind ownership behavior.
For standard clock functions, the declared duration/representation aliases remain
platform-native; no assumption about their integer width or clock resolution is
added. Using a non-noexcept pointer signature also accepts the standard library's
noexcept methods without adding an exception-policy change at the JS boundary.

This prevents ambiguity when **the selected overload still exists** and other
overloads are added. A removed or changed selected signature is intentionally a
compile error; the bridge must not silently bind a different interface. Identical
types with changed behavior still need runtime tests. Field properties, enum
values and explicitly typed constructors are not unqualified function-pointer
registrations. Existing captureless pointer/ownership adapters retain their
ordinary typed call expressions; this policy does not attempt to freeze overload
resolution inside every adapter, upstream implementation, or generated C wrapper.
There is no change to the upstream source overlay or its application checks.

`tests/test_embind_overloads.py` guards the bridge's direct-registration style and
compiles every production signature against synthetic overload sets, including
extra arguments with defaults, function templates and opposite-const methods.
It covers both declaration orders, rejects bare-pointer registration, and checks
that removal of a selected signature fails instead of selecting a substitute.
A native execution fixture checks bool values, reference identity, mutable and
const methods, and native clock values. Another syntax test uses the real
checkout's headers (`LCB_TEST_LLAMA_SOURCE` can supply a local tree). These native
checks model only the two standard selector templates, not Embind marshalling.
They run through existing Python test discovery without new dependencies or a
workflow change. `LCB_TEST_CXX` optionally selects a native C++ compiler for them.

The real generated-module `tests/chat-surface.mjs` continues to cover JSON grammar
conversion with both boolean values and now also exercises static JSON factories,
const predicates, message methods, and parser JSON serialization/restoration. Only
the actual runtime build/smoke test verifies the complete Emscripten binding.

Reference: [Embind select_overload](https://emscripten.org/docs/api_reference/bind.h.html#select-overload-and-select-const).

## Multimodal C API and validation

The generator includes non-deprecated `mtmd.h` / `mtmd-helper.h` C declarations,
records, and constants for mmproj loading, capability queries, RGB/PCM and encoded
file input, chunks, batching, encoding, decoder positions, and audio generation.
Large media buffers remain in linear memory through C APIs. Follow the pinned
headers for borrowed buffer sizes/lifetimes; chunk save/load preserves placeholder
metadata, not encodable media data. OpenAI message helpers accept text/media markers,
not image URLs or audio payloads; upstream serialization may concatenate media parts.

Subprocess video APIs are excluded (`MTMD_VIDEO=OFF`); applications can supply RGB frames.
Profiles remain single-threaded. The build-tree [audio threading overlay](audio-single-thread.md)
serializes shared mel and Parakeet preprocessing in non-pthread Emscripten builds.
Qwen3-TTS speaker preprocessing otherwise requests four threads independently of
`n_threads=1`. This fix requires rebuilt runtime artifacts; it is not a new API.
Audio generation is experimental upstream. Exposed APIs do not certify model support.

Actions checks all five profiles in browser and test variants, standard type generation, package/schema consistency and size limits,
then exercises direct generated-module chat primitives and RGB/PCM allocation in
CPU Chromium. Trained tool-model inference, image/audio mmproj inference, and WebGPU
inference remain separate validations. Native host smoke covers C bindings only.


## WebGPU BF16 vision weights

WebGPU runtimes expand vision Brain Floating Point 16 (BF16) weights to F32 in
memory to avoid the pinned backend's BF16 matmul fallback. Model files and
language-model weights are unchanged. Converted resident weights use twice the
BF16 space; conversion scratch is bounded to 1.5 MiB. Existing WebGPU kernels may
still use F16 intermediate storage, so this is not full-F32 arithmetic. Native
`lcb_clip: bf16-f32` and `lcb_clip: matmul placement` summaries contain only counts
and byte totals, not paths or tensor values. Real-model speed and quality must be
validated separately; the workaround does not guarantee GPU placement for all
operations or sizes.

## Upstream audio generation boundary

The core exposes the pinned upstream audio helper without a downstream language
capability query or waveform-tail optimization. Manual language selection and the
upstream model default remain available through the existing input contract;
features in another reference implementation do not imply llama.cpp support.
The single-thread reference-audio preprocessing workaround is a separate retained
compatibility exception. See the [exception register](../upstream-patches-only-as-a-last-resort-with-explicit-user-approval/README.md).
