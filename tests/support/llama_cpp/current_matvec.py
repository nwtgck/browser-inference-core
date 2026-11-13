"""Native current matvec recording with complete real shader headers.

Only low-level C WebGPU calls and resource boundaries are mocked. The dispatcher
is selected from a fully type-checked provider using Clang semantic ranges; it
is still a fragment test, not execution of the complete GPU backend.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess

from build_tools.runtimes.llama_cpp import webgpu_plan
from build_tools.toolchain.dawn import verify
from tests.support.llama_cpp.current_webgpu import dawn_package
from tests.support.llama_cpp.dispatch_ast import function_ast, _source_range, macro_definition


def build_complete_matvec_recorder(upstream: Path, prepared: Path, output: Path, *,
                                   case_directory: Path, dispatch_functions: tuple[str, ...],
                                   case_files: tuple[str, ...] = ('host-cases.cpp',)) -> Path:
    """Compile complete current/reference factories with owned boundary cases.

    The small runtime-specific arguments name actual C++ functions and owned
    case files. They do not describe source edits or replace factory behavior.
    """
    here = case_directory.resolve()
    upstream, prepared, output = upstream.resolve(), prepared.resolve(), output.resolve()
    for source_root in (upstream, prepared):
        if output == source_root or source_root in output.parents or output in source_root.parents:
            raise ValueError('Matvec recorder output must not overlap input trees')
    sdk = dawn_package()
    verify(sdk)
    receipt = json.loads((prepared / webgpu_plan.RECEIPT).read_text())
    source = prepared / 'ggml-webgpu.cpp'
    original = source.read_bytes()
    args = ['-std=c++20', '-D__wasi__', '-pthread', '-DGGML_SCHED_MAX_COPIES=4',
            '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS']
    if receipt['gpuProfile']:
        args += ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']
    args += ['-I' + str(path) for path in (
        prepared, upstream / webgpu_plan.BASE, upstream / 'ggml/include', upstream / 'ggml/src')]
    args += ['-isystem', str(sdk / 'webgpu/include'), '-isystem', str(sdk / 'webgpu_cpp/include')]
    chunks = []
    names = []
    for name in dispatch_functions:
        if name == 'ggml_webgpu_mul_mat' and receipt['gpuProfile']:
            # The profiled provider owns a context builder and a template,
            # then exposes the same ordinary entry point as a typed wrapper.
            names.extend(('ggml_webgpu_mat_context', 'ggml_webgpu_mul_mat_impl'))
        names.append(name)
    for name in names:
        node, data = function_ast(source, name, args, include_template=True)
        if data != original:
            raise ValueError('Provider changed between matvec dispatcher AST passes')
        chunks.append(_source_range(node, source, data))
    macro = macro_definition(source, 'CEIL_DIV', args)
    if source.read_bytes() != original:
        raise ValueError('Provider changed during matvec dispatcher preprocessing')
    output.mkdir(parents=True, exist_ok=True)
    fragment = output / 'dispatch.inc'
    fragment.write_text('#undef CEIL_DIV\n' + macro + '\n' + '\n'.join(chunks))
    # Both complete factories consume identical current common shader inputs.
    # Only the include path is adapted; the upstream factory stays unmodified.
    (output / 'ggml-wgsl-shaders.hpp').write_text(
        '#include "' + str(prepared / 'ggml-webgpu-aligned-word-load-shaders.hpp') + '"\n')
    code = output / 'host.cpp'
    code.write_text(
        '#include "' + str(here.parent / 'webgpu-native-recording/api.hpp') + '"\n'
        'namespace reference {\n'
        '#include "' + str(upstream / webgpu_plan.BASE / 'ggml-webgpu-shader-lib.hpp') + '"\n}\n'
        '#undef GGML_WEBGPU_SHADER_LIB_HPP\n'
        '#include "' + str(prepared / 'ggml-webgpu-shader-lib.hpp') + '"\n'
        '#include "' + str(here / 'current-prefix.cpp') + '"\n'
        '#include "' + str(fragment) + '"\n' +
        ''.join('#include "' + str(here / name) + '"\n' for name in case_files))
    executable = output / 'host'
    command = [os.environ.get('CXX', 'c++'), '-std=c++20', '-O0', '-Wall', '-Wextra', '-Werror',
               '-D__wasi__', '-pthread', '-I' + str(prepared), '-I' + str(output)]
    for path in (upstream / 'ggml/include', upstream / 'ggml/src', upstream / webgpu_plan.BASE,
                 sdk / 'webgpu/include', sdk / 'webgpu_cpp/include'):
        command += ['-isystem', str(path)]
    command += [str(code), '-o', str(executable)]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as error:
        error.add_note(error.stderr)
        raise
    if source.read_bytes() != original:
        raise ValueError('Provider changed during matvec recorder compilation')
    (output / 'host.json').write_text(json.dumps({
        'scope': 'Complete current factory and AST-selected dispatcher; native C WebGPU recorder only',
        'gpuProfile': receipt['gpuProfile'],
        'providerSha256': hashlib.sha256(original).hexdigest(),
        'dispatchSha256': hashlib.sha256(fragment.read_bytes()).hexdigest(),
        'sourcePreparation': receipt,
        'compileCommand': command,
    }, indent=2) + '\n')
    return executable
