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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_qsa_selection as qsa
import prepare_file_backed_rows as rows


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
        rows.prepare(self.source, self.work/'file-rows')
        inp=self.work/'file-rows/src/models/qwen4exp.cpp'
        out=qsa.prepare(self.source,inp,self.output)
        manifest=json.loads(qsa.INPUTS_PATH.read_text())
        self.assertEqual(manifest['variants']['file-backed-rows']['outputSha256'],qsa.digest(out))

    def test_input_drift_is_rejected(self):
        inp=self.work/'qwen4exp.cpp';inp.write_bytes(self.input.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'Unreviewed'):
            qsa.prepare(self.source,inp,self.output)

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

    def test_every_accepted_input_has_one_expected_output(self):
        info=json.loads(qsa.INPUTS_PATH.read_text())
        self.assertEqual({'upstream','file-backed-rows'},set(info['variants']))
        self.assertEqual(2,len({v['inputSha256'] for v in info['variants'].values()}))
        self.assertEqual(2,len({v['outputSha256'] for v in info['variants'].values()}))

    def test_preflight_prepares_pristine_and_composed_variants(self):
        before = self.input.read_bytes()
        qsa.preflight(self.source, self.work / 'preflight')
        info = json.loads(qsa.INPUTS_PATH.read_text())
        for name, variant in (('pristine', 'upstream'), ('composed', 'file-backed-rows')):
            self.assertEqual(qsa.digest(self.work / 'preflight' / name / 'qwen4exp.cpp'),
                             info['variants'][variant]['outputSha256'])
        self.assertEqual(self.input.read_bytes(), before)

    def test_each_semantic_dependency_rejects_drift_before_publishing(self):
        info = json.loads(qsa.INPUTS_PATH.read_text())
        source = self.work / 'source'
        for relative in info['upstream']:
            dest = source / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source / relative, dest)
        for relative in info['upstream']:
            path = source / relative
            original = path.read_bytes()
            with self.subTest(relative=relative):
                path.write_bytes(original + b' semantic drift')
                with self.assertRaisesRegex(ValueError, 'semantic review'):
                    qsa.prepare(source, source / qsa.SOURCE_PATH, self.output)
                self.assertFalse((self.output / 'qwen4exp.cpp').exists())
                path.write_bytes(original)

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
get_source_file_property(headers "${{LCB_QSA_COPY}}" TARGET_DIRECTORY llama INCLUDE_DIRECTORIES)
get_source_file_property(unity "${{LCB_QSA_COPY}}" TARGET_DIRECTORY llama SKIP_UNITY_BUILD_INCLUSION)
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
                info = json.loads(qsa.INPUTS_PATH.read_text())
                self.assertEqual(qsa.digest(copy), info['variants'][variant]['outputSha256'])
                headers, unity = (build / 'properties.txt').read_text().splitlines()
                expected = build / 'file-backed-rows-overlay/src/models' if file_rows else self.source / 'src/models'
                self.assertEqual(Path(headers), expected)
                self.assertEqual(unity, 'ON')

if __name__=='__main__': unittest.main()
