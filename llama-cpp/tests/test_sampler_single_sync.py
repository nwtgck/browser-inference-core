"""Validate exact sampler preparation and actual-source sampling semantics on CPU."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tests/sampler-single-sync'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_sampler_single_sync as sampler
import prepare_webgpu_source as webgpu_source
import prepare_webgpu_tensor_copy as identity


def definition(source, declaration):
    start = source.index(declaration)
    brace = source.index('{', start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'


class SamplerSingleSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.source / sampler.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')

    def prepare(self, directory, source=None, patches=None):
        return sampler.prepare(directory, source or self.source, patches or ROOT / identity.PATCH_DIRECTORY,
                               webgpu_source.patch_environment(directory))

    def test_exact_patch_and_public_getter_contracts_remain_pristine(self):
        before = {path: (self.source / path).read_bytes() for path in sampler.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory(prefix='sampler-single-sync-') as temporary:
            target = webgpu_source.prepare(self.source, Path(temporary)).parent / 'llama-sampler.cpp'
            self.assertEqual(identity.digest(target), sampler.OUTPUT_SHA256)
            actual = target.read_text()
            body = definition(actual, 'llama_token llama_sampler_sample(')
            self.assertEqual(body.count('ctx->synchronize();'), 1)
            self.assertLess(body.index('ctx->synchronize();'), body.index('ctx->get_sampled_token_ith'))
            self.assertNotIn('llama_get_sampled_', body)
            self.assertNotIn('llama_get_logits_ith(', body)
            self.assertIn('#include "llama-context.h"', actual)
        self.assertEqual(before, {path: (self.source / path).read_bytes() for path in sampler.CONTRACT_INPUTS})

    def test_changed_input_patch_output_and_in_tree_work_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='sampler-single-sync-guards-') as temporary:
            root = Path(temporary)
            source = root / 'upstream'
            for name in sampler.CONTRACT_INPUTS:
                target = source / name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.source / name, target)
            patches = root / 'patches'; patches.mkdir()
            patch_file = patches / sampler.PATCH_NAME
            shutil.copyfile(ROOT / identity.PATCH_DIRECTORY / sampler.PATCH_NAME, patch_file)
            for name in sampler.CONTRACT_INPUTS:
                dependency = source / name; original = dependency.read_bytes(); dependency.write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, 'contract changed'):
                    self.prepare(root / 'work', source, patches)
                dependency.write_bytes(original)
            original = patch_file.read_bytes(); patch_file.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'patch identity changed'):
                self.prepare(root / 'work', source, patches)
            patch_file.write_bytes(original)
            work = root / 'work'; target = work / sampler.SOURCE_PATH
            target.parent.mkdir(parents=True); target.write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                self.prepare(work, source, patches)
            shutil.copyfile(source / sampler.SOURCE_PATH, target)
            with patch.object(sampler, 'OUTPUT_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'output identity'):
                self.prepare(work, source, patches)
            for forbidden in (source, source / 'generated', source.parent):
                with self.assertRaisesRegex(ValueError, 'outside the upstream'):
                    self.prepare(forbidden, source, patches)
            linked = root / 'linked-work'; linked.mkdir()
            (linked / 'src').symlink_to(source / 'src', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'inside the isolated'):
                self.prepare(linked, source, patches)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_sampler_and_public_getters_against_two_backend_scheduler_model(self):
        with tempfile.TemporaryDirectory(prefix='sampler-single-sync-native-') as temporary:
            root = Path(temporary)
            actual = (webgpu_source.prepare(self.source, root / 'overlay').parent / 'llama-sampler.cpp').read_text()
            upstream = (self.source / sampler.SOURCE_PATH).read_text()
            context = (self.source / 'src/llama-context.cpp').read_text()
            code = (HERE / 'semantic-harness.cpp').read_text()
            getters = ['llama_token llama_get_sampled_token_ith(', 'float * llama_get_sampled_probs_ith(',
                       'float * llama_get_sampled_logits_ith(', 'llama_token * llama_get_sampled_candidates_ith(',
                       'uint32_t llama_get_sampled_probs_count_ith(', 'uint32_t llama_get_sampled_logits_count_ith(',
                       'float * llama_get_logits_ith(']
            code = code.replace('// @ACTUAL_CONTEXT_SYNCHRONIZE@', definition(context, 'void llama_context::synchronize('))
            code = code.replace('// @ACTUAL_PUBLIC_GETTERS@', '\n'.join(definition(context, name) for name in getters))
            for marker, source, name in [('ACTUAL_BASELINE_SAMPLE', upstream, 'baseline_sample'), ('ACTUAL_PREPARED_SAMPLE', actual, 'prepared_sample')]:
                body = definition(source, 'llama_token llama_sampler_sample(').replace('llama_sampler_sample(', name + '(', 1)
                code = code.replace('// @' + marker + '@', body)
            harness = root / 'sampling.cpp'; harness.write_text(code)
            binary = root / 'sampling'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O0', str(harness), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_entire_prepared_sampler_source_syntax_with_real_headers(self):
        with tempfile.TemporaryDirectory(prefix='sampler-single-sync-syntax-') as temporary:
            target = webgpu_source.prepare(self.source, Path(temporary)).parent / 'llama-sampler.cpp'
            command = ['c++', '-std=c++17', '-fsyntax-only', '-DGGML_SCHED_MAX_COPIES=4']
            command += ['-I' + str(self.source / relative) for relative in ['include', 'src', 'ggml/include', 'ggml/src']]
            subprocess.run(command + [str(target)], check=True)


if __name__ == '__main__':
    unittest.main()
