# Upstream patches are a last resort

Keeping this core close to upstream llama.cpp is a primary project goal.
Downstream patches are exceptions, not a feature-development mechanism.
A request to implement a feature or improve performance does not, by itself,
authorize adding or expanding a downstream patch.

## Stop and ask before introducing a new divergence

Before adding or expanding a downstream divergence:

1. Explain which current upstream interface is insufficient and why. Distinguish
   a model's official reference implementation from its support in llama.cpp.
2. Present a no-patch option, including omitting or deferring the feature, using
   an existing supported setting, or waiting for an upstream implementation.
3. Describe the proposed changes, affected files, internal assumptions, expected
   benefit, validation limits, and ongoing update/maintenance risks.
4. Obtain explicit user approval for that specific scope before implementation.

The user decides which capabilities are important enough to justify that cost.
Do not make this trade-off on the user's behalf. Approval of essential multimodal
performance work does not authorize automatic language selection, another model,
a speculative compatibility fix, or a separate optimization.

Treat source rewriting, build-tree overlays, toolchain patches, private APIs,
new public functions implemented by modifying upstream, copied implementations,
and replacement translation units as divergences regardless of their filename.
Moving equivalent logic into a wrapper or consumer does not avoid this rule.

## Prefer less coupling, not merely fewer changed lines

For approaches that actually meet the approved requirement, prefer:

1. Existing upstream functionality, settings, and public interfaces.
2. Independent new files that use those interfaces without reimplementing model
   internals or duplicating upstream logic.
3. Additions through supported extension points, with minimal connecting changes.
4. Narrow, conditional edits to existing upstream implementation files.
5. Changes to internal classes, graph construction, state layout, or multiple
   interdependent files only as a separately justified last resort.

File addition is a preference, not an exemption from approval. A copied pipeline
can be harder to maintain than one small local edit. Record internal dependencies
and semantic divergence even when text-level patch conflicts disappear.

## Keep accepted exceptions narrow and removable

Use the `README.md` register to record purpose, accepted scope, alternatives,
upstream revision, dependencies, tests and limitations, and a removal condition.
Keep unrelated changes separable. Do not bundle an optional feature into a
required compatibility fix or make one exception depend on an unrelated one.

Routine build application of an accepted patch needs no new approval. Mechanical
path/context maintenance within the accepted behavior needs no new approval,
but any change to the purpose, behavior, assumptions, or affected implementation
requires consultation. When upstream changes break an exception, report the
conflict; do not widen the patch, drop it silently, or use a stale patched copy.

Preserve upstream source bytes. Prepare isolated build copies, fail clearly on
application mismatch, retain regression tests and provenance, and verify actual
compiled behavior where applicable. Application checks alone do not validate
model quality, performance, or browser execution.

Remove an exception when upstream satisfies its requirement, or when the user
chooses to drop that requirement. Remove its preparation/build/schema plumbing,
consumer dependency, and obsolete claims together while preserving independent
fixes. Published artifacts remain immutable; build and publish a new artifact
and update consumers using real identities rather than fabricated hashes.
