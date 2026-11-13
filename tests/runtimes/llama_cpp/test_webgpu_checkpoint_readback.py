"""Checked checkpoint composition, source isolation and per-TU build routing."""
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

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from tests.support.llama_cpp.webgpu_stages import checkpoint_readback as checkpoint
# These tests isolate an accepted historical patch prefix.
from tests.support import legacy_webgpu_source as source
sys.path.insert(0, str(runtime_test_root('llama-cpp')))
from tests.runtimes.llama_cpp.webgpu_prefix_context import without_checkpoint_readback, without_subgroup_size_control


from tests.support import legacy_webgpu_source as legacy_source

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckpointReadback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='checkpoint-readback-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with without_checkpoint_readback(source):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            with without_subgroup_size_control(source):
                cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile).parent

    def seed(self, work, profile):
        target = work / checkpoint.BASE
        target.mkdir(parents=True)
        for name in checkpoint.INPUTS[profile]: shutil.copyfile(self.before[profile] / name, target / name)

    def prepare(self, work, profile):
        return checkpoint.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                  source.patch_environment(work), gpu_profile=profile)

    def test_exact_inverse_and_unchanged_surrounding_sources(self):
        for profile in (False, True):
            before, final = self.before[profile], self.final[profile]
            old_names, new_names = {p.name for p in before.iterdir()}, {p.name for p in final.iterdir()}
            self.assertEqual(new_names - old_names, set(checkpoint.GENERATED_NAMES))
            self.assertEqual(old_names - new_names, set())
            self.assertEqual({name for name in old_names if (before / name).read_bytes() != (final / name).read_bytes()},
                             {'ggml-webgpu.cpp'})
            for name, expected in checkpoint.OUTPUTS[profile].items(): self.assertEqual(digest(final / name), expected)
            for name, expected in checkpoint.GENERATED_OUTPUTS.items(): self.assertEqual(digest(final / Path(name).name), expected)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                for name in [str(checkpoint.BASE / 'ggml-webgpu.cpp'), *checkpoint.GENERATED_OUTPUTS]:
                    target = work / name; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(final / target.name, target)
                subprocess.run(['git', 'apply', '--no-index', '-R', str(ROOT / source.PATCH_DIRECTORY / checkpoint.PATCH_NAME)],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                self.assertEqual((work / checkpoint.BASE / 'ggml-webgpu.cpp').read_bytes(), (before / 'ggml-webgpu.cpp').read_bytes())
                self.assertEqual((work / checkpoint.CONTEXT_PATH).read_bytes(), (self.upstream / checkpoint.CONTEXT_PATH).read_bytes())
                self.assertTrue(all(not (work / name).exists() for name in checkpoint.GENERATED_OUTPUTS if name != checkpoint.CONTEXT_PATH))
        for name in checkpoint.GENERATED_NAMES:
            self.assertEqual((self.final[False] / name).read_bytes(), (self.final[True] / name).read_bytes())

    def test_source_patch_composition_and_output_drift_fail_closed(self):
        mutations = [('CONTRACT_INPUTS', name) for name in checkpoint.CONTRACT_INPUTS]
        mutations += [('GENERATED_OUTPUTS', name) for name in checkpoint.GENERATED_OUTPUTS]
        mutations += [('PATCH_SHA256', None)]
        for profile in (False, True):
            selected = mutations + [(field, name) for field in ['INPUTS', 'OUTPUTS'] for name in getattr(checkpoint, field)[profile]]
            for field, name in selected:
                with self.subTest(profile=profile, field=field, name=name), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp); self.seed(work, profile)
                    value = copy.deepcopy(getattr(checkpoint, field))
                    if name is None: value = '0' * 64
                    elif field in ['INPUTS', 'OUTPUTS']: value[profile][name] = '0' * 64
                    else: value[name] = '0' * 64
                    with patch.object(checkpoint, field, value), self.assertRaises(ValueError): self.prepare(work, profile)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); self.seed(work, profile); self.prepare(work, profile)
                with self.assertRaisesRegex(ValueError, 'composition input'): self.prepare(work, profile)
        # Preparation commits no changed files to a prior output on failure.
        final = self.final[True]; original = {p.name: p.read_bytes() for p in final.iterdir()}
        with patch.object(checkpoint, 'PATCH_SHA256', '0' * 64), self.assertRaises(ValueError):
            legacy_source.prepare(self.upstream, final, gpu_profile=True)
        self.assertEqual(original, {p.name: p.read_bytes() for p in final.iterdir()})

    def test_isolation_collisions_and_unchanged_input_tree(self):
        before = {name: (self.upstream / name).read_bytes() for name in checkpoint.CONTRACT_INPUTS}
        for forbidden in [self.upstream, self.upstream / 'output', self.upstream.parent]:
            with self.assertRaisesRegex(ValueError, 'outside the upstream'):
                checkpoint.prepare(forbidden, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                   source.patch_environment(forbidden), gpu_profile=False)
        for name in checkpoint.GENERATED_OUTPUTS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); self.seed(work, False)
                target = work / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text('unreviewed')
                with self.assertRaisesRegex(ValueError, 'unreviewed source'): self.prepare(work, False)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); self.seed(work, False); (work / 'src').symlink_to(self.upstream / 'src', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'inside the isolated'): self.prepare(work, False)
        self.assertEqual(before, {name: (self.upstream / name).read_bytes() for name in checkpoint.CONTRACT_INPUTS})

    def test_final_identities_dependencies_and_idempotence(self):
        self.assertEqual(checkpoint.INPUTS, source.aligned_word_load.OUTPUTS)
        self.assertEqual(checkpoint.OUTPUTS, source.subgroup_size_control.INPUTS)
        self.assertEqual(digest(ROOT / source.PATCH_DIRECTORY / checkpoint.PATCH_NAME), checkpoint.PATCH_SHA256)
        self.assertEqual(checkpoint.REVIEWED_DAWN_HEADERS, source.tensor_copy.DAWN_HEADERS)
        for profile in (False, True):
            final = self.final[profile]; stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            with without_subgroup_size_control(source):
                source.prepare(self.upstream, final, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_host_writer_offsets_fallback_and_failure_lifetime(self):
        text = (self.final[False] / 'llama-context.cpp').read_text()
        start = text.index('class llama_io_write_host :')
        writer = text[start:text.index('class llama_io_read_host :', start)]
        fixture = (runtime_test_root('llama-cpp') / 'webgpu-checkpoint-readback/host-writer.cpp').read_text()
        self.assertEqual(fixture.count('// PRODUCTION_WRITER'), 1)
        code = self.directory / 'host-writer.cpp'
        code.write_text(fixture.replace('// PRODUCTION_WRITER', writer))
        binary = self.directory / 'host-writer'
        command = ['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                   '-I' + str(self.final[False]), '-I' + str(self.upstream / 'src'),
                   '-I' + str(self.upstream / 'ggml/include'),
                   '-I' + str(self.upstream / 'include'), str(code), '-o', str(binary)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        subprocess.run([str(binary)], check=True, capture_output=True)

    @unittest.skipUnless(shutil.which('cmake'), 'CMake required; no compiler is used')
    def test_context_routing_preserves_order_and_is_emscripten_only(self):
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        block = cmake[cmake.index('# The provider is Emdawn-only.'):cmake.index('# The optional snapshot contract also changes')]
        for enabled, names, passes in [(False, ['a.cpp', 'llama-context.cpp', 'b.cpp'], True),
                                        (True, ['a.cpp', 'llama-context.cpp', 'b.cpp'], True),
                                        (True, ['a.cpp', 'b.cpp'], False),
                                        (True, ['llama-context.cpp', 'llama-context.cpp'], False)]:
            with self.subTest(enabled=enabled, names=names), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); (work / 'overlay').mkdir(); (work / 'upstream/src').mkdir(parents=True)
                for name in set(names): (work / name).write_text('')
                (work / 'overlay/llama-context.cpp').write_text('')
                code = 'cmake_minimum_required(VERSION 3.24)\nproject(checkpoint_routing NONE)\nadd_custom_target(llama)\n'
                code += 'set(EMSCRIPTEN ' + ('ON' if enabled else 'OFF') + ')\n'
                code += 'set(LCB_SOURCE_OVERLAY "${CMAKE_CURRENT_SOURCE_DIR}/overlay")\n'
                code += 'set(LCB_LLAMA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/upstream")\n'
                code += 'set_property(TARGET llama PROPERTY SOURCES "' + ';'.join(names) + '")\n'
                code += 'set_property(TARGET llama PROPERTY INCLUDE_DIRECTORIES "file-rows-first;upstream-second")\n' + block
                code += 'get_target_property(actual llama SOURCES)\nget_target_property(includes llama INCLUDE_DIRECTORIES)\n'
                code += 'get_source_file_property(options "${LCB_SOURCE_OVERLAY}/llama-context.cpp" TARGET_DIRECTORY llama COMPILE_OPTIONS)\n'
                code += 'get_source_file_property(unity "${LCB_SOURCE_OVERLAY}/llama-context.cpp" TARGET_DIRECTORY llama SKIP_UNITY_BUILD_INCLUSION)\n'
                code += 'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.txt" "${actual}\n${includes}\n${options}\n${unity}\n")\n'
                (work / 'CMakeLists.txt').write_text(code)
                result = subprocess.run(['cmake', '-S', str(work), '-B', str(work / 'build')], capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, passes, result.stdout + result.stderr)
                if not passes: self.assertIn('context source layout changed', result.stderr); continue
                actual, includes, options, unity = (work / 'build/result.txt').read_text().splitlines()
                expected = [str(work / 'overlay/llama-context.cpp') if name == 'llama-context.cpp' and enabled else name for name in names]
                self.assertEqual(actual.split(';'), expected)
                self.assertEqual(includes, 'file-rows-first;upstream-second')
                self.assertEqual(options, '-iquote;' + str(work / 'upstream/src') if enabled else 'NOTFOUND')
                self.assertEqual(unity, 'ON' if enabled else 'NOTFOUND')


if __name__ == '__main__': unittest.main()
