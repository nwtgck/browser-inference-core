"""Sampler build-copy contracts; not candidate Wasm or performance evidence."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_sampler as sampler


class SamplerOverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not all((upstream / p).is_file() for p in (sampler.SOURCE_PATH, sampler.CONTEXT_PATH)):
            self.skipTest('Initialize pinned submodule or set LCB_TEST_LLAMA_SOURCE')
        self.upstream = upstream
        self.source = self.work / 'source'
        (self.source / 'src').mkdir(parents=True)
        for name in ('llama-sampler.cpp', 'llama-context.cpp'):
            shutil.copyfile(upstream / 'src' / name, self.source / 'src' / name)
        self.output = self.work / 'output'

    def test_repeat_preserves_vendor_bytes_and_mtime(self):
        before = (self.source / sampler.SOURCE_PATH).read_bytes()
        target = sampler.prepare(self.source, self.output)
        stamp = target.stat().st_mtime_ns
        self.assertEqual(sampler.sampler_contract(target.read_text()), sampler.SAMPLER_OUTPUT_SHA256)
        self.assertEqual(sampler.prepare(self.source, self.output), target)
        self.assertEqual(target.stat().st_mtime_ns, stamp)
        self.assertEqual((self.source / sampler.SOURCE_PATH).read_bytes(), before)

    def test_reject_source_or_ancestor_output(self):
        for output in (self.source, self.source / 'build', self.work):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'outside'):
                sampler.prepare(self.source, output)

    def test_relevant_context_change_requires_review_preserves_old_output(self):
        target = sampler.prepare(self.source, self.output)
        before = target.read_bytes()
        p = self.source / sampler.CONTEXT_PATH
        p.write_text(p.read_text().replace('    output_swaps.clear();', '    output_swaps.clear(); // changed'))
        with self.assertRaisesRegex(ValueError, 'semantic review'):
            sampler.prepare(self.source, self.output)
        self.assertEqual(target.read_bytes(), before)

    def test_sampler_change_requires_review(self):
        p = self.source / sampler.SOURCE_PATH
        p.write_text(p.read_text().replace('    llama_sampler_apply(smpl, &cur_p);', '    llama_sampler_apply(smpl, &cur_p); // changed'))
        with self.assertRaisesRegex(ValueError, 'semantic review'):
            sampler.prepare(self.source, self.output)

    def test_unrelated_function_changes_do_not_require_hash_refresh(self):
        for path in (sampler.CONTEXT_PATH, sampler.SOURCE_PATH):
            p = self.source / path
            p.write_text(p.read_text() + '\n// unrelated source addition\n')
        sampler.prepare(self.source, self.output)

    def test_duplicate_context_signature_fails(self):
        p = self.source / sampler.CONTEXT_PATH
        p.write_text(p.read_text() + '\n' + sampler.CONTEXT_FUNCTIONS[0] + ' {\n}\n')
        with self.assertRaisesRegex(ValueError, 'review'):
            sampler.prepare(self.source, self.output)

    def test_bad_patch_fails_without_stale_success(self):
        target = sampler.prepare(self.source, self.output)
        before = target.read_bytes()
        bad = self.work / 'bad.patch'
        bad.write_text('not a patch\n')
        with self.assertRaisesRegex(ValueError, 'patch identity'):
            sampler.prepare(self.source, self.output, bad)
        self.assertEqual(target.read_bytes(), before)

    def test_nested_repository_does_not_silently_skip_patch(self):
        subprocess.run(['git', 'init', '-q', str(self.work)], check=True)
        target = sampler.prepare(self.source, self.output)
        self.assertEqual(sampler.sampler_contract(target.read_text()), sampler.SAMPLER_OUTPUT_SHA256)

    def test_real_headers_compile_complete_translation_unit(self):
        compiler = shutil.which('c++')
        if not compiler or not (self.upstream / 'include/llama.h').is_file():
            self.skipTest('Native compiler and full upstream headers required')
        target = sampler.prepare(self.source, self.output)
        subprocess.run([compiler, '-std=c++17', '-fsyntax-only',
                        *['-I' + str(self.upstream / d) for d in ('src', 'include', 'ggml/include', 'ggml/src')],
                        str(target)], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
