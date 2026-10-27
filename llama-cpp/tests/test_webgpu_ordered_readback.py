"""Execute actual readback helpers with deferred CPU queues; no GPU speed claim."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tests/ordered_readback'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_ordered_readback as ordered_readback

IDENTITY = json.loads((HERE / 'inputs.json').read_text())
PATCH = ROOT / source.PATCH_DIRECTORY / 'ggml-webgpu-ordered-readback.patch'


def verify(path, expected):
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('ordered_readback needs semantic review: ' + str(path))


def function(text, name):
    """Extract a definition, skipping any matching forward declaration."""
    for match in re.finditer(r'\b' + re.escape(name) + r'\s*\(', text):
        start = text.rfind('\n', 0, match.start()) + 1
        brace = text.index('{', match.end())
        semicolon = text.find(';', match.end(), brace)
        if semicolon >= 0 or not text[start:match.start()].lstrip().startswith(('static ', 'void ')):
            continue
        depth, end = 1, brace + 1
        while depth:
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        previous = text.rfind('\n', 0, start - 1) + 1
        if text[previous:start].startswith('template <'):
            start = previous
        return text[start:end] + '\n'
    raise ValueError('Missing actual helper: ' + name)


class OrderedReadback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        for relative, expected in IDENTITY['upstream_sha256'].items():
            verify(cls.upstream / relative, expected)
        verify(PATCH, IDENTITY['patch_sha256'])

    def prepare(self, root):
        return source.prepare(self.upstream, root)

    def test_production_and_fixture_contracts_match(self):
        self.assertEqual(IDENTITY['upstream_sha256'], ordered_readback.CONTRACT_INPUTS)
        self.assertEqual(IDENTITY['patch_sha256'], ordered_readback.PATCH_SHA256)
        self.assertEqual(IDENTITY['dawn_header_sha256'], source.tensor_copy.DAWN_HEADERS)
        self.assertEqual(source.bind_group_cache.INPUT_SHA256, ordered_readback.OUTPUT_SHA256)
        self.assertEqual(source.scale_unary.INPUTS['ggml-webgpu.cpp'], source.bind_group_cache.OUTPUT_SHA256)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_order_call_counts_ranges_errors_and_destination_lifetime(self):
        before = (self.upstream / source.SOURCE_PATH).read_bytes()
        with tempfile.TemporaryDirectory(prefix='ordered-readback-') as temporary:
            root = Path(temporary)
            actual = self.prepare(root / 'overlay').read_text()
            generic = (self.upstream / 'ggml/src/ggml-backend.cpp').read_text()
            helpers = '\n'.join(function(actual, name) for name in (
                'ggml_webgpu_tensor_offset', 'ggml_webgpu_create_buffer',
                'ggml_backend_webgpu_check_wait_status', 'ggml_backend_webgpu_wait_queue',
                'ggml_backend_webgpu_map_buffer', 'ggml_backend_webgpu_buffer_get_tensor'))
            registration = re.search(r'/\* \.get_tensor_async\s+= \*/\s*(\w+)\s*,', actual)
            self.assertIsNotNone(registration)
            code = (HERE / 'mock-harness.cpp').read_text()
            for marker, contents in {
                '// @ACTUAL_HELPERS@': helpers,
                '// @ACTUAL_GENERIC_GET@': function(generic, 'ggml_backend_tensor_get'),
                '// @ACTUAL_SYNCHRONIZE@': function(actual, 'ggml_backend_webgpu_synchronize') + function(generic, 'ggml_backend_synchronize'),
                '// @ACTUAL_HOOK@': function(actual, 'ggml_backend_webgpu_get_tensor_async'),
                '// @ACTUAL_ASYNC_ENTRY@': function(generic, 'ggml_backend_tensor_get_async'),
                '// @REGISTERED_HOOK@': registration.group(1),
            }.items():
                code = code.replace(marker, contents)
            harness = root / 'ordered_readback.cpp'
            harness.write_text(code)
            binary = root / 'ordered_readback'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O0', str(harness), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
        self.assertEqual(before, (self.upstream / source.SOURCE_PATH).read_bytes())

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_complete_composed_source_syntax_with_pinned_ggml_and_dawn_headers(self):
        configured = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        dawn = Path(configured) if configured else ROOT.parent / '.tools/emdawnwebgpu_pkg'
        if not dawn.is_dir():
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE for the exact local Dawn package')
        for relative, expected in IDENTITY['dawn_header_sha256'].items():
            verify(dawn / relative, expected)
        with tempfile.TemporaryDirectory(prefix='ordered-readback-syntax-') as temporary:
            root = Path(temporary)
            target = self.prepare(root)
            upstream_webgpu = self.upstream / 'ggml/src/ggml-webgpu'
            subprocess.run([sys.executable, str(upstream_webgpu / 'wgsl-shaders/embed_wgsl.py'),
                            '--input_dir', str(upstream_webgpu / 'wgsl-shaders'),
                            '--output_file', str(root / 'ggml-wgsl-shaders.hpp')], check=True)
            includes = [self.upstream / path for path in ('ggml/include', 'ggml/src')]
            includes += [upstream_webgpu, root]
            # __wasi__ selects the same standard-WebGPU branches as Emscripten,
            # without requiring an unused Emscripten-only header on the host.
            command = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4']
            command += ['-I' + str(path) for path in includes]
            command += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
            # Compile the complete actual source, including the real registered interface.
            subprocess.run(command + [str(target)], check=True)
            subprocess.run(command + ['-DGGML_WEBGPU_GPU_PROFILE', '-DGGML_WEBGPU_CPU_PROFILE', str(target)], check=True)


if __name__ == '__main__':
    unittest.main()
