"""Retained-mode fail-closed configuration, compiler identity and packaging gates."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_retained_bf16_source import verify as verify_source
from verify_bf16_build import verify as verify_build
from verify_bf16_build import runtime_metadata, validate_metadata


class RetainedModeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='lcb-retained-')
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)

    def test_source_contract_matches_pinned_upstream(self):
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (source / 'tools/mtmd/clip.cpp').is_file():
            self.skipTest('Set LCB_TEST_LLAMA_SOURCE to the pinned checkout')
        verify_source(source)

    def test_source_contract_rejects_missing_or_modified_input(self):
        source = self.work / 'source'
        source.mkdir()
        p = source / 'file.cpp'
        p.write_bytes(b'original')
        contract = self.work / 'contract.json'
        contract.write_text(json.dumps({'files': {'file.cpp': hashlib.sha256(p.read_bytes()).hexdigest()}}))
        verify_source(source, contract)
        for data in (b'changed', None):
            if data is None:
                p.unlink()
            else:
                p.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'semantic review'):
                verify_source(source, contract)

    def test_empty_or_malformed_contract_is_not_a_success(self):
        contract = self.work / 'contract.json'
        for value in ({}, {'files': {}}, {'files': []}, [],
                      {'files': {'../outside': '0' * 64}},
                      {'files': {'/outside': '0' * 64}},
                      {'files': {'a.cpp': 'not-a-digest'}},
                      {'files': {'': '0' * 64}}):
            contract.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'semantic review'):
                verify_source(self.work, contract)

    def test_actual_compile_definition_is_required_and_unambiguous(self):
        db = self.work / 'commands.json'
        for webgpu in (False, True):
            expected = ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME=' + str(int(webgpu))]
            for key in ('arguments', 'command'):
                args = ['c++', '-c', 'mtmd.cpp', *expected]
                db.write_text(json.dumps([{'file': '/build/mtmd.cpp', key: args if key == 'arguments' else ' '.join(args)},
                                          {'file':'clip.cpp', 'command':'c++ -c clip.cpp '+expected[0]}]))
                verify_build(db, webgpu=webgpu)
        for args in ([], ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME=0'], ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME'],
                     ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME=1', '-DLCB_MTMD_WEBGPU_BF16_RUNTIME=0'],
                     ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME=1', '-DLCB_MTMD_WEBGPU_BF16_RUNTIME=1'],
                     ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME=1', '-U', 'LCB_MTMD_WEBGPU_BF16_RUNTIME'],
                     ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME=1', '-D', 'LCB_MTMD_WEBGPU_BF16_RUNTIME=0']):
            db.write_text(json.dumps([{'file':'mtmd.cpp','arguments':['c++',*args]},
                                      {'file':'clip.cpp','command':'c++ -c clip.cpp -DLCB_MTMD_WEBGPU_BF16_RUNTIME=1'}]))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                verify_build(db, webgpu=True)
        for entries in ([], [{'file':'clip.cpp','command':'c++'}] * 2):
            db.write_text(json.dumps(entries))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                verify_build(db, webgpu=False)
        db.write_text(json.dumps([{'file':'mtmd.cpp','command':'c++ -DLCB_MTMD_WEBGPU_BF16_RUNTIME=1'},
                                  {'file':'clip.cpp','command':'c++ -DLCB_MTMD_RETAIN_BF16=1'}]))
        with self.assertRaisesRegex(ValueError, 'Obsolete'):
            verify_build(db, webgpu=True)

    def test_distribution_preserves_shared_runtime_capability(self):
        validate_metadata({})  # Existing manifests retain their interpretation.
        validate_metadata({'visionBf16Storage':runtime_metadata()})
        for info in ({'visionBf16Storage':'retain-checked'}, {'experimentalVisionBf16':True},
                     {'cmakeCommand':['-DLCB_WEBGPU_BF16_STORAGE=retain-checked']},
                     {'visionBf16Storage':{**runtime_metadata(),'default':'retain-checked'}}):
            with self.subTest(info=info), self.assertRaises(ValueError):
                validate_metadata(info)

    @unittest.skipUnless(shutil.which('cmake') and shutil.which('ninja'), 'CMake and Ninja are required')
    def test_ordinary_build_rechecks_changed_reviewed_sources(self):
        upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (upstream / 'tools/mtmd/clip.cpp').is_file():
            self.skipTest('Set LCB_TEST_LLAMA_SOURCE to the pinned checkout')
        source = self.work / 'source'
        contract = json.loads((ROOT / 'config/retained-bf16-source.json').read_text())
        for name in contract['files']:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(upstream / name, target)
        # Configure the real hook on a minimal target, without compiling a copied
        # pipeline or modifying the pinned checkout. `help` triggers Ninja's
        # normal CMake regeneration, which must reject source drift first.
        project = self.work / 'project'
        project.mkdir()
        (project / 'clip.cpp').write_text('void fixture() {}\n')
        cmake = [
            'cmake_minimum_required(VERSION 3.24)',
            'project(retained_dependency LANGUAGES CXX)',
            'find_package(Python3 REQUIRED COMPONENTS Interpreter)',
            'add_library(mtmd STATIC clip.cpp)',
            f'set(LCB_LLAMA_SOURCE "{source}")',
            f'include("{ROOT / "cmake/MtmdOverlay.cmake"}")',
        ]
        (project / 'CMakeLists.txt').write_text('\n'.join(cmake) + '\n')
        build = self.work / 'build'
        configure = subprocess.run(['cmake', '-G', 'Ninja', '-S', str(project), '-B', str(build)],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(configure.returncode, 0, configure.stdout)
        before = (build / 'mtmd-overlay/clip.cpp').read_bytes()
        tracked = (build / 'build.ninja').read_text()
        for name in contract['files']:
            self.assertIn(str(source / name), tracked, name)
        changed = source / 'ggml/src/ggml.c'
        changed.write_text(changed.read_text() + '\n// semantic drift fixture\n')
        attempted = subprocess.run(['cmake', '--build', str(build), '--target', 'help'],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertNotEqual(attempted.returncode, 0, attempted.stdout)
        self.assertIn('semantic review', attempted.stdout)
        self.assertEqual((build / 'mtmd-overlay/clip.cpp').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
