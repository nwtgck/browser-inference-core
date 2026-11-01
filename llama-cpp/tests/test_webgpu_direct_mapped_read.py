"""Checked Emdawn aligned-copy composition; native/unaligned behavior stays."""
import copy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import prepare_webgpu_source as source
import prepare_webgpu_direct_mapped_read as direct
from test_webgpu_ordered_readback import function

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


class DirectMappedRead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='direct-mapped-read-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        wide = patch.object(source.quant_wide_m, 'prepare')
        wide.start()
        cls.addClassCleanup(wide.stop)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with patch.object(direct, 'prepare'):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile).parent

    def test_full_inverse_only_changes_the_copy_block(self):
        getters = []
        for profile in (False, True):
            old, new = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in old.iterdir()}, {p.name for p in new.iterdir()})
            self.assertEqual({p.name for p in old.iterdir() if p.read_bytes() != (new / p.name).read_bytes()},
                             {'ggml-webgpu.cpp'})
            for name, expected in direct.INPUTS[profile].items():
                self.assertEqual(sha(old / name), expected)
                self.assertEqual(sha(new / name), direct.OUTPUTS[profile][name])
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); target = work / direct.BASE / 'ggml-webgpu.cpp'; target.parent.mkdir(parents=True)
                shutil.copy2(new / target.name, target)
                subprocess.run(['git', 'apply', '--no-index', '-R', str(ROOT / source.PATCH_DIRECTORY / direct.PATCH_NAME)],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                self.assertEqual(target.read_bytes(), (old / target.name).read_bytes())
            before = (old / 'ggml-webgpu.cpp').read_text(); after = (new / 'ggml-webgpu.cpp').read_text()
            previous = function(before, 'ggml_backend_webgpu_buffer_get_tensor')
            getter = function(after, 'ggml_backend_webgpu_buffer_get_tensor')
            begin = getter.index('#ifdef __EMSCRIPTEN__')
            self.assertEqual(getter[:begin], previous[:begin])
            self.assertEqual(getter[getter.rindex('    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();'):],
                             previous[previous.rindex('    buf_ctx->global_ctx->get_tensor_staging_buf.Unmap();'):])
            self.assertIn('local_offset == 0 && size % 4 == 0', getter)
            getters.append(getter)
            for name in ('ggml_backend_webgpu_get_tensor_async', 'ggml_backend_webgpu_map_buffer',
                         'ggml_backend_webgpu_synchronize', 'ggml_backend_webgpu_wait_queue'):
                self.assertEqual(function(before, name), function(after, name))
        self.assertEqual(getters[0], getters[1])

    def test_source_patch_input_output_and_atomic_drift(self):
        for profile in (False, True):
            mutations = [('CONTRACT_INPUTS', name) for name in direct.CONTRACT_INPUTS]
            mutations += [(field, name) for field in ('INPUTS', 'OUTPUTS') for name in getattr(direct, field)[profile]]
            mutations += [('PATCH_SHA256', None)]
            for field, name in mutations:
                with self.subTest(profile=profile, field=field, name=name), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp); target = work / direct.BASE; target.mkdir(parents=True)
                    for filename in direct.INPUTS[profile]: shutil.copy2(self.before[profile] / filename, target / filename)
                    value = copy.deepcopy(getattr(direct, field))
                    if name is None: value = '0' * 64
                    elif field in ('INPUTS', 'OUTPUTS'): value[profile][name] = '0' * 64
                    else: value[name] = '0' * 64
                    with patch.object(direct, field, value), self.assertRaises(ValueError):
                        direct.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                       source.patch_environment(work), gpu_profile=profile)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); target = work / direct.BASE; target.mkdir(parents=True)
                for name in direct.INPUTS[profile]: shutil.copy2(self.final[profile] / name, target / name)
                with self.assertRaisesRegex(ValueError, 'composition input'):
                    direct.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                   source.patch_environment(work), gpu_profile=profile)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source.prepare(self.upstream, root, gpu_profile=True)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with patch.object(direct, 'PATCH_SHA256', '0' * 64), self.assertRaises(ValueError):
                source.prepare(self.upstream, root)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_final_identity_dependencies_and_idempotence(self):
        self.assertEqual(direct.INPUTS, source.quant_half_pairs.OUTPUTS)
        self.assertEqual(direct.OUTPUTS, source.quant_wide_m.INPUTS)
        self.assertEqual(direct.CONTRACT_INPUTS, source.ordered_readback.CONTRACT_INPUTS)
        self.assertEqual(direct.REVIEWED_DAWN_HEADERS, source.tensor_copy.DAWN_HEADERS)
        self.assertEqual(sha(ROOT / source.PATCH_DIRECTORY / direct.PATCH_NAME), direct.PATCH_SHA256)
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_direct_mapped_read.py', cmake)
        self.assertIn(direct.PATCH_NAME, cmake)
        for name in direct.REVIEWED_DAWN_HEADERS: self.assertIn('${EMDAWNWEBGPU_DIR}/' + name, cmake)
        for profile in (False, True):
            final = self.final[profile]; stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            source.prepare(self.upstream, final, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_pinned_dawn_exposes_the_checked_direct_copy_signature(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir(): self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        for name, expected in direct.REVIEWED_DAWN_HEADERS.items(): self.assertEqual(sha(dawn / name), expected)
        code = self.directory / 'dawn-api.cpp'
        code.write_text('''#include <webgpu/webgpu_cpp.h>
#include <type_traits>
#include <utility>
static_assert(std::is_same_v<decltype(std::declval<const wgpu::Buffer &>().ReadMappedRange(
    size_t(0), static_cast<void *>(nullptr), size_t(4))), wgpu::Status>);
''')
        subprocess.run(['c++', '-std=c++20', '-fsyntax-only', '-isystem', str(dawn / 'webgpu/include'),
                        '-isystem', str(dawn / 'webgpu_cpp/include'), str(code)], check=True, capture_output=True)


if __name__ == '__main__': unittest.main()
