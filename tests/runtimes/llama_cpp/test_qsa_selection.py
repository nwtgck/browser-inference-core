from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from build_tools.project import REPOSITORY_ROOT, runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from build_tools.runtimes.llama_cpp.sources import qsa_selection as qsa
from build_tools.runtimes.llama_cpp.sources import file_backed_rows as rows
SEMANTIC_DEPENDENCIES = ('src/models/qwen4exp.cpp', 'src/llama-memory-hybrid-idx.cpp', 'src/llama-memory-hybrid-idx.h', 'ggml/src/ggml.c', 'ggml/src/ggml-cpu/ops.cpp', 'ggml/src/ggml-backend.cpp', 'src/llama-context.cpp', 'src/llama-graph.cpp', 'src/llama-cparams.h', 'ggml/include/ggml.h')
ROWS_INPUTS = ('src/models/models.h', 'src/models/qwen4exp.cpp', 'src/llama-model.cpp', 'src/llama-model-loader.cpp', 'src/llama-model-loader.h', 'src/llama-mmap.cpp', 'src/llama-mmap.h', 'src/CMakeLists.txt', 'ggml/include/ggml.h', 'src/llama.cpp')

class QsaSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', str(ROOT/'vendor/llama.cpp')))
        if not (cls.source/'src/models/qwen4exp.cpp').is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE to the pinned source')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.output = self.work/'output'
        self.input = self.source/'src/models/qwen4exp.cpp'

    def test_pristine_composition_and_idempotence(self):
        original = self.input.read_bytes()
        out = qsa.prepare(self.source, self.input, self.output)
        first = out.read_bytes(), out.stat().st_mtime_ns
        qsa.prepare(self.source, self.input, self.output)
        self.assertEqual(first, (out.read_bytes(), out.stat().st_mtime_ns))
        self.assertEqual(original, self.input.read_bytes())
        self.assertIn(b'ggml_repeat_4d(ctx0, inp_kpool->pool_idxs', out.read_bytes())

    def test_composes_with_file_backed_header_layout(self):
        rows.prepare(self.source, self.work / 'file-rows')
        inp = self.work / 'file-rows/src/models/qwen4exp.cpp'
        out = qsa.prepare(self.source, inp, self.output)
        self.assertIn(b'lcb_qwen4exp_file_backed_shard', out.read_bytes())
        self.assertIn(b'ggml_repeat_4d(ctx0, inp_kpool->pool_idxs', out.read_bytes())
        record = json.loads((self.output / qsa.RECEIPT).read_text())['patches'][0]
        self.assertEqual(record['files'][0]['beforeSha256'], qsa.digest(inp))
        self.assertEqual(record['files'][0]['afterSha256'], qsa.digest(out))

    def test_harmless_input_comment_does_not_require_hash_refresh(self):
        baseline = qsa.prepare(self.source, self.input, self.output).read_bytes()
        inp = self.work / 'qwen4exp.cpp'
        comment = b'\n// unrelated upstream documentation\n'
        inp.write_bytes(self.input.read_bytes() + comment)
        out = qsa.prepare(self.source, inp, self.output)
        self.assertEqual(out.read_bytes(), baseline + comment)

    def test_linked_input_is_rejected(self):
        inp=self.work/'linked.cpp';inp.symlink_to(self.input)
        with self.assertRaisesRegex(ValueError,'Linked'):
            qsa.prepare(self.source,inp,self.output)

    def test_linked_output_root_is_rejected(self):
        target=self.work/'target';target.mkdir();self.output.symlink_to(target,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Linked'):
            qsa.prepare(self.source,self.input,self.output)
        self.assertEqual([],list(target.iterdir()))

    def test_linked_output_file_is_rejected(self):
        self.output.mkdir();target=self.work/'untouched';target.write_text('untouched')
        (self.output/'qwen4exp.cpp').symlink_to(target)
        with self.assertRaisesRegex(ValueError,'Linked'):
            qsa.prepare(self.source,self.input,self.output)
        self.assertEqual('untouched',target.read_text())

    def test_upstream_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            qsa.prepare(self.source,self.input,self.source/'qsa-output')

    def test_input_directory_output_is_rejected(self):
        inp=self.work/'qwen4exp.cpp';inp.write_bytes(self.input.read_bytes())
        with self.assertRaisesRegex(ValueError,'translation unit'):
            qsa.prepare(self.source,inp,self.work)

    def test_git_environment_and_parent_repository_do_not_skip_patch(self):
        subprocess.run(['git','init','-q',str(self.work)],check=True)
        foreign=self.work/'foreign';foreign.mkdir()
        subprocess.run(['git','init','-q',str(foreign)],check=True)
        env={'GIT_DIR':str(foreign/'.git'),'GIT_WORK_TREE':str(foreign),
             'GIT_INDEX_FILE':str(self.work/'foreign-index'),'GIT_PREFIX':'prefix/'}
        with mock.patch.dict(os.environ,env):
            out=qsa.prepare(self.source,self.input,self.output)
        self.assertIn(b'ggml_repeat_4d(ctx0, inp_kpool->pool_idxs',out.read_bytes())
        self.assertFalse((self.work/'foreign-index').exists())

    def test_receipt_records_the_actual_patch_and_output(self):
        out = qsa.prepare(self.source, self.input, self.output)
        info = json.loads((self.output / qsa.RECEIPT).read_text())
        self.assertEqual(info['output'], {'path': 'qwen4exp.cpp', 'sha256': qsa.digest(out)})
        self.assertEqual([r['file'] for r in info['patches']], [qsa.PATCH_NAME])
        self.assertEqual([f['path'] for f in info['patches'][0]['files']], [qsa.SOURCE_PATH])

    def test_preflight_prepares_pristine_and_composed_variants(self):
        before = self.input.read_bytes()
        qsa.preflight(self.source, self.work / 'preflight')
        for name, inp in (('pristine', self.input),
                          ('composed', self.work / 'preflight/file-rows' / qsa.SOURCE_PATH)):
            output = self.work / 'preflight' / name
            record = json.loads((output / qsa.RECEIPT).read_text())['patches'][0]
            self.assertEqual(record['files'][0]['beforeSha256'], qsa.digest(inp))
            self.assertEqual(record['files'][0]['afterSha256'], qsa.digest(output / 'qwen4exp.cpp'))
        self.assertEqual(self.input.read_bytes(), before)

    def test_incompatible_patch_context_fails_before_publishing(self):
        inp = self.work / 'qwen4exp.cpp'
        inp.write_text('// not a Qwen model translation unit\n')
        with self.assertRaisesRegex(ValueError, 'Patch application failed'):
            qsa.prepare(self.source, inp, self.output)
        self.assertFalse(self.output.exists())

    def test_cmake_composes_only_the_qsa_translation_unit_in_both_header_modes(self):
        if not shutil.which('cmake'):
            self.skipTest('CMake unavailable')
        for file_rows in (False, True):
            with self.subTest(file_rows=file_rows):
                fixture = self.work / ('rows' if file_rows else 'plain')
                fixture.mkdir()
                fixture.joinpath('CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.24)
project(qsa_overlay_check LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(LCB_LLAMA_SOURCE "{self.source.as_posix()}")
add_library(llama STATIC "${{LCB_LLAMA_SOURCE}}/src/llama.cpp"
  "${{LCB_LLAMA_SOURCE}}/src/llama-model.cpp"
  "${{LCB_LLAMA_SOURCE}}/src/models/qwen4exp.cpp"
  "${{LCB_LLAMA_SOURCE}}/src/models/llama.cpp")
{'include("' + (ROOT / 'cmake/FileBackedRowsOverlay.cmake').as_posix() + '")' if file_rows else ''}
get_target_property(before llama SOURCES)
include("{(ROOT / 'cmake/QsaSelectionOverlay.cmake').as_posix()}")
get_target_property(after llama SOURCES)
get_property(dependencies DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS)
get_source_file_property(headers "${{LCB_QSA_COPY}}" TARGET_DIRECTORY llama INCLUDE_DIRECTORIES)
get_source_file_property(unity "${{LCB_QSA_COPY}}" TARGET_DIRECTORY llama SKIP_UNITY_BUILD_INCLUSION)
file(WRITE "${{CMAKE_BINARY_DIR}}/dependencies.txt" "${{dependencies}}")
file(WRITE "${{CMAKE_BINARY_DIR}}/before.txt" "${{before}}")
file(WRITE "${{CMAKE_BINARY_DIR}}/after.txt" "${{after}}")
file(WRITE "${{CMAKE_BINARY_DIR}}/properties.txt" "${{headers}}\\n${{unity}}\\n")
''')
                build = fixture / 'build'
                subprocess.run(['cmake', '-S', str(fixture), '-B', str(build)], check=True, capture_output=True)
                before = (build / 'before.txt').read_text().split(';')
                after = (build / 'after.txt').read_text().split(';')
                self.assertEqual(len(before), len(after))
                self.assertEqual(len(set(before) - set(after)), 1)
                copies = set(after) - set(before)
                self.assertEqual(len(copies), 1)
                copy = Path(next(iter(copies)))
                variant = 'file-backed-rows' if file_rows else 'upstream'
                dependencies = set((build / 'dependencies.txt').read_text().split(';'))
                for relative in SEMANTIC_DEPENDENCIES:
                    self.assertIn(str(self.source / relative), dependencies, relative)
                for module in ['patches.py', 'files.py']:
                    self.assertIn(str(REPOSITORY_ROOT / 'build_tools/source' / module), dependencies)
                for module in ['output.py', 'project.py']:
                    self.assertIn(str(REPOSITORY_ROOT / 'build_tools' / module), dependencies)
                receipt = json.loads((copy.parent / qsa.RECEIPT).read_text())
                self.assertEqual(qsa.digest(copy), receipt['output']['sha256'])
                headers, unity = (build / 'properties.txt').read_text().splitlines()
                expected = build / 'file-backed-rows-overlay/src/models' if file_rows else self.source / 'src/models'
                self.assertEqual(Path(headers), expected)
                self.assertEqual(unity, 'ON')

    @unittest.skipUnless(shutil.which('cmake') and shutil.which('ninja'), 'CMake and Ninja required')
    def test_warm_build_rechecks_changed_fusion_dependencies(self):
        # Exercise Ninja's actual regeneration path, not just the property list.
        # A local copy ensures the corruption tests never edit the vendor tree.
        source = self.work / 'upstream'
        shutil.copytree(self.source / 'src', source / 'src')
        for relative in set(SEMANTIC_DEPENDENCIES) | set(ROWS_INPUTS):
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source / relative, target)
        for file_rows in (False, True):
            caller = self.work / ('warm-rows' if file_rows else 'warm-plain')
            caller.mkdir()
            caller.joinpath('CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.24)
project(qsa_warm_check LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(LCB_LLAMA_SOURCE "{source.as_posix()}")
add_library(llama STATIC "${{LCB_LLAMA_SOURCE}}/src/llama.cpp"
  "${{LCB_LLAMA_SOURCE}}/src/llama-model.cpp"
  "${{LCB_LLAMA_SOURCE}}/src/models/qwen4exp.cpp")
{'include("' + (ROOT / 'cmake/FileBackedRowsOverlay.cmake').as_posix() + '")' if file_rows else ''}
include("{(ROOT / 'cmake/QsaSelectionOverlay.cmake').as_posix()}")
add_custom_target(checked COMMAND "${{CMAKE_COMMAND}}" -E echo checked)
''')
            build = caller / 'build'
            subprocess.run(['cmake', '-S', str(caller), '-B', str(build), '-G', 'Ninja'],
                           check=True, capture_output=True)
            command = ['cmake', '--build', str(build), '--target', 'checked']
            subprocess.run(command, check=True, capture_output=True)
            for relative in ('src/llama-context.cpp', 'src/llama-graph.cpp',
                             'src/llama-graph.h', 'src/llama-model.cpp', 'src/llama-model.h',
                             'src/llama-cparams.h', 'ggml/include/ggml.h'):
                path = source / relative
                original = path.read_bytes()
                with self.subTest(file_rows=file_rows, dependency=relative):
                    path.write_bytes(original + b'\n// harmless comment\n')
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('Re-running CMake', result.stdout + result.stderr)
                    path.write_bytes(original)
                    # Recovery must also happen on the build entry point alone.
                    subprocess.run(command, check=True, capture_output=True)

if __name__=='__main__': unittest.main()
