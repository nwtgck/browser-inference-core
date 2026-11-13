"""Native current-library recording with the real Dawn declarations.

Only C WebGPU calls are mocked. This is neither a browser nor GPU execution.
"""
import os
from pathlib import Path
import shutil
import subprocess

from build_tools.project import REPOSITORY_ROOT, runtime_test_root
from build_tools.toolchain.dawn import verify


def dawn_package():
    return Path(os.environ.get('LCB_TEST_DAWN_PACKAGE',
                               REPOSITORY_ROOT / '.tools/emdawnwebgpu_pkg')).resolve()


def require_recording_inputs(test, *, clang_ast=False):
    missing_clang = clang_ast and not shutil.which(os.environ.get('CLANGXX', 'clang++'))
    if (missing_clang or not shutil.which(os.environ.get('CXX', 'c++'))
            or not (dawn_package() / 'webgpu/include/webgpu/webgpu.h').is_file()):
        message = 'Current shader recorder needs native C++, pinned Dawn headers (LCB_TEST_DAWN_PACKAGE)'
        if clang_ast:
            message += ', and Clang (CLANGXX) for dispatch AST checks'
        if os.environ.get('CI') == 'true':
            raise RuntimeError(message)
        test.skipTest(message)


def compile_recorder(upstream, prepared, output, cases, *, fragment=None):
    """Compile the complete current library; mock only the C WebGPU boundary."""
    sdk = dawn_package()
    verify(sdk)
    here = runtime_test_root('llama-cpp')
    output.mkdir(parents=True, exist_ok=True)
    code = output / 'host.cpp'
    # No reverse patches, golden hashes, factory extraction, or generated-text
    # substitutions. Declarations/inline definitions are the actual final input.
    includes = [here / 'webgpu-native-recording/api.hpp', prepared / 'ggml-webgpu-shader-lib.hpp']
    if fragment is not None:
        includes.append(fragment)
    includes.append(cases)
    code.write_text(''.join('#include "' + str(path.resolve()) + '"\n' for path in includes))
    executable = output / 'host'
    command = [os.environ.get('CXX', 'c++'), '-std=c++20', '-O1', '-Wall', '-Wextra', '-Werror',
               '-D__wasi__', '-pthread']
    command += ['-I' + str(prepared)]
    # Upstream utility headers contain intentionally unused static helpers.
    # Treat dependencies as system headers; keep -Werror on the actual generated
    # library and this recorder rather than disabling a warning globally.
    for path in (upstream / 'ggml/include', upstream / 'ggml/src',
                 upstream / 'ggml/src/ggml-webgpu'):
        command += ['-isystem', str(path)]
    command += ['-isystem', str(sdk / 'webgpu/include'), '-isystem', str(sdk / 'webgpu_cpp/include'),
                str(code), '-o', str(executable)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as error:
        error.add_note(error.stderr)
        raise
    return executable
