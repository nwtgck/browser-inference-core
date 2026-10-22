"""Review guards, coherent CMake selection, and the optional sync extension."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_file_backed_rows as rows
import generate_bindings
from test_callback_metadata_bindings import declarations

SOURCE = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))


class OptionalBindings(unittest.TestCase):
    def test_metadata_exports_are_sync_and_do_not_change_the_upstream_interface(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with patch.object(generate_bindings.subprocess, 'check_output',
                              return_value=json.dumps({'inner': declarations()})):
                schema = generate_bindings.generate(SOURCE, out, 'clang')
            exports = json.loads((out / 'exports.json').read_text())
            jspi = json.loads((out / 'jspi-exports.json').read_text())
            for name in ['lcb_file_backed_rows_version', 'lcb_model_file_backed_shard']:
                self.assertEqual(exports.count('_' + name), 1)
                self.assertNotIn(name, jspi)
                self.assertNotIn('_' + name, jspi)
                self.assertNotIn(name, (out / 'functions.d.ts').read_text())
            cpp = (out / 'bindings.cpp').read_text()
            self.assertIn('#ifdef LCB_FILE_BACKED_PLE', cpp)
            self.assertIn('if (!model) throw', cpp)
            self.assertIn('lcb_checked_pointer(model)', cpp)
            self.assertEqual(schema['abiVersion'], 1)


@unittest.skipUnless((SOURCE / 'src/models/qwen4exp.cpp').is_file(),
                     'Set LCB_TEST_LLAMA_SOURCE to the reviewed checkout')
class ReviewedOverlay(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)

    def test_preparation_is_pristine_idempotent_and_contains_a_coherent_definition(self):
        before = {p: (SOURCE / p).read_bytes() for p in json.loads(rows.INPUTS_PATH.read_text())}
        output = self.work / 'overlay'
        source = rows.prepare(SOURCE, output)
        first = {p: p.stat().st_mtime_ns for p in source.rglob('*') if p.is_file()}
        rows.prepare(SOURCE, output)
        self.assertEqual(first, {p: p.stat().st_mtime_ns for p in source.rglob('*') if p.is_file()})
        self.assertEqual(before, {p: (SOURCE / p).read_bytes() for p in before})
        self.assertEqual((source / 'models/file-backed-rows.h').read_bytes(),
                         (ROOT / 'bridge/file-backed-rows.h').read_bytes())
        self.assertIn('file_backed_ple', (source / 'models/models.h').read_text())
        self.assertIn('arch != LLM_ARCH_QWEN4EXP', (source / 'llama-model.cpp').read_text())
        self.assertIn('model export requires resident weights', (source / 'llama.cpp').read_text())
        for original in (SOURCE / 'src/models').glob('*.cpp'):
            if original.name != 'qwen4exp.cpp':
                self.assertEqual(original.read_bytes(), (source / 'models' / original.name).read_bytes())
        self.assertEqual((SOURCE / 'src/llama-model-loader.cpp').read_bytes(),
                         (source / 'llama-model-loader.cpp').read_bytes())

    def test_reviewed_input_drift_fails_before_publishing(self):
        pins = json.loads(rows.INPUTS_PATH.read_text())
        pins['src/models/qwen4exp.cpp'] = '0' * 64
        path = self.work / 'pins.json'
        path.write_text(json.dumps(pins))
        output = self.work / 'must-not-exist'
        with patch.object(rows, 'INPUTS_PATH', path), self.assertRaisesRegex(ValueError, 'upstream review'):
            rows.prepare(SOURCE, output)
        self.assertFalse(output.exists())

    def test_source_and_overlay_may_not_overlap(self):
        for output in [SOURCE, SOURCE / 'generated', SOURCE.parent]:
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'outside'):
                rows.prepare(SOURCE, output)

    def test_linked_overlay_files_are_rejected_without_following_them(self):
        output = self.work / 'overlay'
        output.mkdir()
        victim = self.work / 'victim'
        victim.write_text('untouched')
        (output / 'linked').symlink_to(victim)
        with self.assertRaisesRegex(ValueError, 'linked overlay'):
            rows.prepare(SOURCE, output)
        self.assertEqual(victim.read_text(), 'untouched')

    def test_patch_application_is_isolated_from_enclosing_git_and_inherited_environment(self):
        enclosing = self.work / 'enclosing'
        enclosing.mkdir()
        subprocess.run(['git', 'init', '-q', str(enclosing)], check=True)
        (enclosing / 'sentinel').write_text('do not change the caller index')
        subprocess.run(['git', '-C', str(enclosing), 'add', 'sentinel'], check=True)
        index = (enclosing / '.git/index').read_bytes()
        scratch = enclosing / 'temporary'
        scratch.mkdir()
        # Both a repository-local TMPDIR and Git hook/worktree environment used
        # to make git apply skip these paths or fail in an unrelated checkout.
        with patch.object(tempfile, 'tempdir', str(scratch)), patch.dict(os.environ, {
            'GIT_DIR': str(enclosing / '.git'), 'GIT_WORK_TREE': str(enclosing),
            'GIT_INDEX_FILE': str(enclosing / '.git/index'),
            'GIT_COMMON_DIR': str(enclosing / '.git'),
        }):
            prepared = rows.prepare(SOURCE, self.work / 'isolated-overlay')
        self.assertIn('file_backed_ple', (prepared / 'models/models.h').read_text())
        self.assertIn('lcb_qwen4exp_file_backed_shard', (prepared / 'models/qwen4exp.cpp').read_text())
        self.assertEqual((enclosing / '.git/index').read_bytes(), index)
        self.assertFalse((enclosing / 'src').exists())

    def test_symlinked_output_root_is_rejected_before_publishing(self):
        victim = self.work / 'victim'
        victim.mkdir()
        (victim / 'sentinel').write_text('untouched')
        output = self.work / 'linked-overlay'
        output.symlink_to(victim, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'linked overlay'):
            rows.prepare(SOURCE, output)
        self.assertEqual(list(victim.iterdir()), [victim / 'sentinel'])

    @unittest.skipUnless(shutil.which('cmake') and shutil.which('c++'), 'CMake and C++ required')
    def test_pointer_guard_uses_target_options_for_emscripten_and_abi_for_native(self):
        caller = self.work / 'guard-caller'
        caller.mkdir()
        (caller / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.24)
project(file_rows_guard LANGUAGES CXX)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(LCB_LLAMA_SOURCE "{SOURCE.as_posix()}")
file(GLOB model_sources "${{LCB_LLAMA_SOURCE}}/src/models/*.cpp")
add_library(llama OBJECT "${{LCB_LLAMA_SOURCE}}/src/llama.cpp"
    "${{LCB_LLAMA_SOURCE}}/src/llama-model.cpp" ${{model_sources}})
# Simulate the toolchain probe independently of the per-target memory64 flags.
set(EMSCRIPTEN "${{TEST_EMSCRIPTEN}}")
set(CMAKE_SIZEOF_VOID_P "${{TEST_POINTER_BYTES}}")
include("{(ROOT / 'cmake/FileBackedRowsOverlay.cmake').as_posix()}")
''')
        cases = [
            ('wasm64-default-probe', 'ON', 'ON', '4', True),
            ('wasm64-wide-probe', 'ON', 'ON', '8', True),
            ('wasm32-default-probe', 'ON', 'OFF', '4', False),
            ('wasm32-wide-probe', 'ON', 'OFF', '8', False),
            ('native64', 'OFF', 'OFF', '8', True),
            ('native32', 'OFF', 'OFF', '4', False),
            ('native32-cannot-spoof-with-option', 'OFF', 'ON', '4', False),
        ]
        for name, emscripten, memory64, pointer_bytes, accepted in cases:
            with self.subTest(name=name):
                result = subprocess.run(['cmake', '-S', str(caller), '-B', str(self.work / name),
                    f'-DTEST_EMSCRIPTEN={emscripten}', f'-DLCB_MEMORY64={memory64}',
                    f'-DTEST_POINTER_BYTES={pointer_bytes}'], capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, accepted, result.stdout + result.stderr)
                if not accepted:
                    self.assertIn('requires a 64-bit build', result.stderr)

    @unittest.skipUnless(shutil.which('cmake') and shutil.which('c++'), 'CMake and C++ required')
    def test_cmake_relocates_every_model_but_preserves_the_independent_loader(self):
        caller = self.work / 'caller'
        caller.mkdir()
        loader = caller / 'independent-loader.cpp'
        loader.write_text('// Simulated already selected WebGPU loader overlay.\n')
        (caller / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.24)
project(file_rows_selection LANGUAGES CXX)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(LCB_LLAMA_SOURCE "{SOURCE.as_posix()}")
file(GLOB model_sources "${{LCB_LLAMA_SOURCE}}/src/models/*.cpp")
add_library(llama OBJECT "${{LCB_LLAMA_SOURCE}}/src/llama.cpp"
    "${{LCB_LLAMA_SOURCE}}/src/llama-model.cpp" ${{model_sources}} "{loader.as_posix()}")
set_target_properties(llama PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 16)
set_source_files_properties("${{LCB_LLAMA_SOURCE}}/src/llama.cpp"
    "${{LCB_LLAMA_SOURCE}}/src/llama-model.cpp" PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
include("{(ROOT / 'cmake/FileBackedRowsOverlay.cmake').as_posix()}")
get_target_property(selected llama SOURCES)
get_target_property(includes llama INCLUDE_DIRECTORIES)
file(WRITE "${{CMAKE_BINARY_DIR}}/selected.txt" "${{selected}}")
file(WRITE "${{CMAKE_BINARY_DIR}}/includes.txt" "${{includes}}")
''')
        build = self.work / 'build'
        subprocess.run(['cmake', '-S', str(caller), '-B', str(build)],
                       check=True, capture_output=True)
        selected = (build / 'selected.txt').read_text().split(';')
        self.assertIn(str(loader), selected)
        self.assertFalse(any(p.startswith(str(SOURCE)) for p in selected))
        for name in ['llama.cpp', 'llama-model.cpp', 'models/qwen4exp.cpp']:
            self.assertIn(str(build / 'file-backed-rows-overlay/src' / name), selected)
        self.assertEqual(len(selected), len(list((SOURCE / 'src/models').glob('*.cpp'))) + 3)
        self.assertEqual((build / 'includes.txt').read_text().split(';')[0],
                         str(build / 'file-backed-rows-overlay/src'))
        # Source properties are keyed by path. Core TUs must not join the
        # model unity batches just because the overlay changed their path.
        unity_files = list((build / 'CMakeFiles/llama.dir/Unity').glob('*.cxx'))
        self.assertTrue(unity_files)
        unity = '\n'.join(path.read_text() for path in unity_files)
        for name in ('llama.cpp', 'llama-model.cpp'):
            self.assertNotIn(str(build / 'file-backed-rows-overlay/src' / name), unity)
        self.assertIn(str(build / 'file-backed-rows-overlay/src/models/clef.cpp'), unity)


if __name__ == '__main__':
    unittest.main()
