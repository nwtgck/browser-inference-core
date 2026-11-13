# Local build tools

This is an ordinary Python package. Run its module entry points from the
repository root; modules do not download missing inputs. Network acquisition and
remote writes remain explicit operations in workflow YAML.

The image runtime currently exposes these module entry points:

```sh
python3 -m build_tools.runtimes.stable_diffusion_cpp.source --output runtimes/stable-diffusion-cpp/build/prepared
python3 -m build_tools.runtimes.stable_diffusion_cpp.build --profile webgpu-wasm32-jspi --variant browser
python3 -m build_tools.runtimes.stable_diffusion_cpp.bindings --help
```

The two runtimes own independent source pins, CMake graphs, and schemas. Selecting
one package adapter must not import or require the other adapter's configuration.

## Source identity and completion

The parent index selects the upstream gitlinks. A locally staged upstream update
is an explicit input selection, not proof that the parent tree is publishable.
Source preparation checks initialized checkouts, then reads the selected commits'
Git trees and blobs, including their nested gitlinks. It does not copy worktree
bytes, apply checkout filters, or honor `export-ignore` / `export-subst`. A clean
`git status` is not sufficient evidence of input contents.

Individual upstream `.patch` files remain the reviewable source of changes.
The source receipt records observed inputs and patch outputs, not a claim of
behavioral compatibility. Real type, numerical, and lifetime checks remain
necessary when updating an upstream.

Sources and their receipt are completed in a private directory before activation.
An unsuccessful rebuild does not discard the prior complete output. If both
activation and recovery fail, the exception identifies the retained recovery
directory; do not delete it before recovering any needed output. A prepared
directory belongs to one invocation and is not a concurrent mutable cache.

## Tests

```sh
python3 -m unittest discover -s tests/build_tools -t .
python3 -m unittest discover -s tests/runtimes/stable_diffusion_cpp -t .
```

Image binding and CMake tests require the image runtime's initialized upstreams.
The common build-tools suite does not require those checkouts. Test fixtures that
synthesize compiler outputs check driver behavior, not a successful Wasm build.
