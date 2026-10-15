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


if __name__ == '__main__':
    unittest.main()
