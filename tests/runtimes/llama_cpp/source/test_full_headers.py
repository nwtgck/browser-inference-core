"""Compile complete current headers; no historical hashes or extracted bodies.

The native compiler parses Dawn's actual C++ declarations. __wasi__ selects its
header-only binding mode, not a WebAssembly target. This checks types/declarations,
not linking, shader execution, Emscripten, or browser behavior.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from build_tools.project import REPOSITORY_ROOT, runtime_root
from build_tools.runtimes.llama_cpp import webgpu_plan as plan
from build_tools.toolchain.dawn import verify


class CompleteCurrentHeaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which(os.environ.get('CXX', 'c++'))
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE',
                                          runtime_root('llama-cpp') / 'vendor/llama.cpp'))
        cls.dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE',
                                      REPOSITORY_ROOT / '.tools/emdawnwebgpu_pkg'))
        missing = (not cls.compiler or not (cls.upstream / plan.BASE / 'ggml-webgpu-shader-lib.hpp').is_file()
                   or not (cls.dawn / 'webgpu/include/webgpu/webgpu.h').is_file())
        if missing:
            message = 'Complete-header checks need a native C++ compiler, llama source, and pinned Dawn headers'
            if os.environ.get('CI') == 'true':
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        verify(cls.dawn)
        temp = tempfile.TemporaryDirectory(prefix='complete-current-headers-')
        cls.addClassCleanup(temp.cleanup)
        cls.work = Path(temp.name)
        cls.outputs = {}
        for profiled in (False, True):
            output = cls.work / ('profile' if profiled else 'normal')
            plan.prepare(cls.upstream, output, gpu_profile=profiled)
            cls.outputs[profiled] = output

    def compile(self, output):
        # This is the same consumer preamble as the complete-header native
        # recorder. No provider function or declaration is sliced or replaced.
        unit = output.parent / (output.name + '-consumer.cpp')
        unit.write_text('#include <array>\n'
                        '#include "ggml-webgpu-shader-lib.hpp"\n'
                        '#include "ggml-webgpu-optional-pipeline.hpp"\n')
        command = [self.compiler, '-std=c++20', '-fsyntax-only', '-D__wasi__', '-pthread']
        command += ['-I' + str(path) for path in (
            output, self.upstream / plan.BASE, self.upstream / 'ggml/include', self.upstream / 'ggml/src')]
        command += ['-isystem', str(self.dawn / 'webgpu/include'),
                    '-isystem', str(self.dawn / 'webgpu_cpp/include'), str(unit)]
        return subprocess.run(command, capture_output=True, text=True, timeout=120)

    def private_source(self, destination):
        receipt = json.loads((self.outputs[False] / plan.RECEIPT).read_text())
        for name in receipt['inputs']:
            if name.startswith('runtime/'):
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.upstream / name, target)
        return destination

    def test_complete_normal_and_diagnostic_headers_compile(self):
        for profiled, output in self.outputs.items():
            with self.subTest(profiled=profiled):
                result = self.compile(output)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_complete_current_providers_exclude_the_old_canonical_header(self):
        poison = self.work / 'ggml-wgsl-shaders.hpp'
        poison.write_text('#error stale canonical shader embedding was included\n')
        header = 'ggml-webgpu-aligned-word-load-shaders.hpp'
        for profiled, output in self.outputs.items():
            with self.subTest(profiled=profiled):
                includes = (output, self.work, self.upstream / plan.BASE,
                            self.upstream / 'ggml/include', self.upstream / 'ggml/src')
                command = [self.compiler, '-std=c++20', '-fsyntax-only', '-H', '-D__wasi__',
                           '-DGGML_SCHED_MAX_COPIES=4', '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS']
                command += ['-I' + str(path) for path in includes]
                command += ['-isystem', str(self.dawn / 'webgpu/include'),
                            '-isystem', str(self.dawn / 'webgpu_cpp/include')]
                if profiled:
                    command += ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']
                result = subprocess.run(command + [str(output / 'ggml-webgpu.cpp')],
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stderr)
                traced = [line.lstrip('. ').strip() for line in result.stderr.splitlines()
                          if line.startswith('.')]
                self.assertEqual(traced.count(str(output / header)), 1)
                self.assertFalse(any(Path(name).name == 'ggml-wgsl-shaders.hpp' for name in traced))

    def test_unrelated_upstream_comment_does_not_need_a_golden_hash_update(self):
        with tempfile.TemporaryDirectory(prefix='current-comment-') as temp:
            root = Path(temp)
            source = self.private_source(root / 'source')
            header = source / plan.BASE / 'ggml-webgpu-shader-lib.hpp'
            header.write_bytes(header.read_bytes() + b'\n// unrelated upstream documentation\n')
            output = root / 'output'
            plan.prepare(source, output)
            self.assertIn('unrelated upstream documentation', (output / header.name).read_text())
            result = self.compile(output)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_unreferenced_declaration_is_not_hidden_by_function_extraction(self):
        with tempfile.TemporaryDirectory(prefix='current-broken-declaration-') as temp:
            root = Path(temp)
            source = self.private_source(root / 'source')
            header = source / plan.BASE / 'ggml-webgpu-shader-lib.hpp'
            header.write_bytes(header.read_bytes() + b'\nstruct bicore_unreferenced_contract { bicore_missing_type value; };\n')
            output = root / 'output'
            # A patch still applies; the semantic check must fail at compilation,
            # not merely because a whole-file digest changed.
            plan.prepare(source, output)
            result = self.compile(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('bicore_missing_type', result.stderr)
