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

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
HERE = runtime_test_root('llama-cpp') / 'ordered_readback'
sys.path.insert(0, str(ROOT.parents[1]))
from build_tools.runtimes.llama_cpp import source as source
# Historical byte identities belong to the reference fixture, not production.
from tests.support import legacy_webgpu_source as reference_source
from tests.support.llama_cpp.webgpu_stages import ordered_readback as ordered_readback
IDENTITY = json.loads((HERE / 'inputs.json').read_text())
PATCH = ROOT / reference_source.PATCH_DIRECTORY / 'ggml-webgpu-ordered-readback.patch'


def verify(path, expected):
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('ordered_readback needs semantic review: ' + str(path))


def function(text, name):
    """Extract a pinned free function, skipping matching forward declarations.

    Its closing brace is at column zero. Raw brace counting is unsuitable for
    alternative preprocessor arms that open the same C++ scope twice in text.
    """
    for match in re.finditer(r'\b' + re.escape(name) + r'\s*\(', text):
        start = text.rfind('\n', 0, match.start()) + 1
        brace = text.index('{', match.end())
        semicolon = text.find(';', match.end(), brace)
        if semicolon >= 0 or not text[start:match.start()].lstrip().startswith(('static ', 'void ')):
            continue
        closing = re.search(r'^}[ \t]*$', text[brace:], re.MULTILINE)
        if closing is None:
            raise ValueError('Missing free-function closing brace: ' + name)
        end = brace + closing.end()
        previous = text.rfind('\n', 0, start - 1) + 1
        if text[previous:start].startswith('template <'):
            start = previous
        return text[start:end] + '\n'
    raise ValueError('Missing actual helper: ' + name)


class OrderedReadback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / reference_source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        for relative, expected in IDENTITY['upstream_sha256'].items():
            verify(cls.upstream / relative, expected)
        verify(PATCH, IDENTITY['patch_sha256'])

    def prepare(self, root):
        return source.prepare(self.upstream, root)

    def test_production_and_fixture_contracts_match(self):
        self.assertEqual(IDENTITY['upstream_sha256'], ordered_readback.CONTRACT_INPUTS)
        self.assertEqual(IDENTITY['patch_sha256'], ordered_readback.PATCH_SHA256)
        self.assertEqual(IDENTITY['dawn_header_sha256'], reference_source.tensor_copy.DAWN_HEADERS)
        self.assertEqual(reference_source.bind_group_cache.INPUT_SHA256, ordered_readback.OUTPUT_SHA256)
        self.assertEqual(reference_source.scale_unary.INPUTS['ggml-webgpu.cpp'], reference_source.bind_group_cache.OUTPUT_SHA256)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_order_call_counts_ranges_errors_and_destination_lifetime(self):
        before = (self.upstream / reference_source.SOURCE_PATH).read_bytes()
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
            for mode in ('native', 'emscripten'):
                binary = root / ('ordered_readback-' + mode)
                flags = ['-D__EMSCRIPTEN__=1'] if mode == 'emscripten' else []
                subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O0', *flags,
                                str(harness), '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)
        self.assertEqual(before, (self.upstream / reference_source.SOURCE_PATH).read_bytes())

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_complete_composed_source_syntax_with_pinned_ggml_and_dawn_headers(self):
        configured = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        dawn = Path(configured) if configured else ROOT.parents[1] / '.tools/emdawnwebgpu_pkg'
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
            # __wasi__ selects standard WebGPU without an Emscripten header.
            # This syntax check retains the native copy arm; the exact getter's
            # Emscripten arm is exercised by the queue model above.
            command = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4']
            command += ['-I' + str(path) for path in includes]
            command += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
            # Compile the complete actual source, including the real registered interface.
            subprocess.run(command + [str(target)], check=True)
            subprocess.run(command + ['-DGGML_WEBGPU_GPU_PROFILE', '-DGGML_WEBGPU_CPU_PROFILE', str(target)], check=True)


if __name__ == '__main__':
    unittest.main()
