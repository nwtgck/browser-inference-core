# Keep the core close to upstream

This repository is a thin browser build and binding layer for llama.cpp. Keep
application behavior out of the core and preserve the ability to update upstream
without maintaining a separate implementation of its model pipelines.

## Upstream divergences require explicit user approval

Downstream patches are a last resort, not the normal way to implement features
or improve performance. A feature request is **not** permission to add or expand
an upstream divergence. The user, not the implementing agent, decides whether
that feature or performance gain justifies the continuing maintenance cost.

Before proposing, adding, expanding, or repairing an upstream/toolchain
workaround, read
`upstream-patches-only-as-a-last-resort-with-explicit-user-approval/AGENTS.md`
and its `README.md` exception register. This requirement applies throughout the
repository, including `build_tools/`, `cmake/`, `native/`, generated bindings, copied
source files, and build-time transformations. It is not limited to `.patch` files.

When existing upstream interfaces cannot meet a request, stop before adding a
new divergence: explain the limitation, offer omission or deferral as a no-patch
option, describe the coupling and maintenance cost, and obtain explicit approval
for the specific scope. Do not silently substitute a downstream extension for
upstream support. Earlier approval of a different workaround is not precedent.

Prefer existing upstream functionality, then independent new files using public
interfaces or supported extension points, before edits to existing upstream
implementation files. A copied implementation in a new file is still a divergence;
do not relocate it into a consumer or wrapper to evade this policy.

Routine application and scope-preserving mechanical upkeep of an already
accepted patch do not require repeated approval. If an upstream update changes
the required behavior, assumptions, or affected implementation, explain the
conflict and ask before widening the exception. Passing `git apply --check` is
not evidence of semantic compatibility.

Do not modify the upstream checkout, silently drop an approved compatibility
fix, weaken regression checks, invent published artifact identities, or erase
existing deviations from provenance to make a build or update pass.
