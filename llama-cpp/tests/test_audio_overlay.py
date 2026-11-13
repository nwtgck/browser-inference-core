"""Track retired worker loops and the actual non-pthread pool compatibility seam."""
from pathlib import Path
import importlib.util
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_mtmd import prepare


class AudioOverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='lcb-audio-test-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))

    def test_only_known_translation_units_are_accepted(self):
        # Individual copied-TU entry points were retired. The coherent API only
        # accepts two reviewed compositions, never a caller-supplied filename.
        for mode in ('../webgpu', '/tmp/cpu', 'unknown'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'composition mode'):
                prepare(self.upstream, self.work / 'build', mode=mode)

    def test_real_audio_overlay_is_idempotent_and_preserves_input(self):
        before = {name: (self.upstream / 'tools/mtmd' / name).read_bytes()
                  for name in ('mtmd-audio.cpp', 'mtmd-subcontext.cpp')}
        for mode in ('cpu', 'webgpu'):
            target = prepare(self.upstream, self.work / mode, mode=mode)
            pool = target / 'mtmd-subcontext.cpp'
            stamp = pool.stat().st_mtime_ns
            prepare(self.upstream, target, mode=mode)
            self.assertEqual(pool.stat().st_mtime_ns, stamp)
            self.assertEqual((target / 'mtmd-audio.cpp').read_bytes(), before['mtmd-audio.cpp'])
            guard = '#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)'
            self.assertEqual(pool.read_text().count(guard), 1)
            self.assertNotIn('std::thread', (target / 'mtmd-audio.cpp').read_text())
            self.assertNotIn('std::thread', pool.read_text())
            self.assertIn('ggml_backend_sched_graph_compute', pool.read_text())
            manifest = json.loads((target / 'mtmd-source-manifest.json').read_text())
            self.assertEqual(manifest['stages'][-1]['patch'], 'mtmd-non-pthread-threadpool.patch')
        for name, raw in before.items():
            self.assertEqual((self.upstream / 'tools/mtmd' / name).read_bytes(), raw)
        self.assertFalse((ROOT / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-audio-single-thread.patch').exists())

    def test_actual_preprocessing_matches_without_creating_threads(self):
        # Upstream replaced FFT worker loops with DFT/F32 graph arithmetic.
        # Preserve that code; exercise the current compatibility helper instead
        # of treating the retired FFT's bitwise output as an audio reference.
        compiler = os.environ.get('LCB_TEST_CXX') or shutil.which('clang++') or shutil.which('g++')
        if not compiler:
            self.skipTest('A native C++ compiler is required')
        target = prepare(self.upstream, self.work / 'overlay', mode='webgpu')
        spec = importlib.util.spec_from_file_location('pool_fixture', ROOT / 'tests/mtmd-threadpool/emit_fixture.py')
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        source = self.work / 'threadpool.cpp'
        fixture.materialize(target / 'mtmd-subcontext.cpp', source)
        for name, defines in (('native', []), ('non-pthread', ['-D__EMSCRIPTEN__=1']),
                              ('pthread', ['-DLCB_SIMULATE_EMSCRIPTEN_PTHREADS=1'])):
            with self.subTest(configuration=name):
                binary = self.work / name
                subprocess.run([compiler, '-std=c++17', '-DLCB_EXPECT_POINTER_BYTES=' + str(ctypes.sizeof(ctypes.c_void_p)), *defines,
                                str(source), '-o', str(binary)], check=True, capture_output=True, text=True)
                result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
                rows = json.loads(result.stdout)
                self.assertEqual(len(rows['cases']), 8)
                self.assertEqual(rows['nonPthreadGuard'], name == 'non-pthread')
                self.assertTrue(all(row['live'] == 0 for row in rows['cases']))
                if name == 'non-pthread':
                    self.assertTrue(all(row['constructors'] == 0 and row['procQueries'] == 0 for row in rows['cases']))
