"""Validate exact sampler preparation and actual-source sampling semantics on CPU."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tests.support.local_source import source_repository

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
HERE = runtime_test_root('llama-cpp') / 'sampler-single-sync'
sys.path.insert(0, str(ROOT.parents[1]))
from tests.support.llama_cpp.webgpu_stages import sampler_single_sync as sampler
from build_tools.runtimes.llama_cpp import source as webgpu_source
# Historical byte identities belong to the reference fixture, not production.
from tests.support import legacy_webgpu_source as reference_source
from tests.support.llama_cpp.webgpu_stages import tensor_copy as identity
HISTORICAL = json.loads((HERE / 'inputs.json').read_text())
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
                               reference_source.patch_environment(directory))

    def test_exact_patch_and_public_getter_contracts_remain_pristine(self):
        before = {path: (self.source / path).read_bytes() for path in sampler.DEPENDENCIES}
        with tempfile.TemporaryDirectory(prefix='sampler-single-sync-') as temporary:
            target = webgpu_source.prepare(self.source, Path(temporary)).parent / 'llama-sampler.cpp'
            self.assertEqual(identity.digest(target), HISTORICAL['outputSha256'])
            actual = target.read_text()
            body = definition(actual, 'llama_token llama_sampler_sample(')
            self.assertEqual(body.count('ctx->synchronize();'), 1)
            self.assertLess(body.index('ctx->synchronize();'), body.index('ctx->get_sampled_token_ith'))
            self.assertNotIn('llama_get_sampled_', body)
            self.assertNotIn('llama_get_logits_ith(', body)
            self.assertIn('#include "llama-context.h"', actual)
        self.assertEqual(before, {path: (self.source / path).read_bytes() for path in sampler.DEPENDENCIES})

    def test_changed_input_patch_output_and_in_tree_work_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='sampler-single-sync-guards-') as temporary:
            root = Path(temporary)
            source = root / 'upstream'
            for name in sampler.DEPENDENCIES:
                target = source / name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.source / name, target)
            patches = root / 'patches'; patches.mkdir()
            patch_file = patches / sampler.PATCH_NAME
            shutil.copyfile(ROOT / identity.PATCH_DIRECTORY / sampler.PATCH_NAME, patch_file)
            for name in sampler.DEPENDENCIES:
                dependency = source / name
                original = dependency.read_bytes()
                dependency.unlink()
                with self.assertRaisesRegex(ValueError, 'Missing source file'):
                    self.prepare(root / 'work', source, patches)
                dependency.write_bytes(original)
            original = patch_file.read_bytes(); patch_file.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'patch identity changed'):
                self.prepare(root / 'work', source, patches)
            patch_file.write_bytes(original)
            work = root / 'work'; target = work / sampler.SOURCE_PATH
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                self.prepare(work, source, patches)
            for forbidden in (source, source / 'generated', source.parent):
                with self.assertRaisesRegex(ValueError, 'outside the upstream'):
                    self.prepare(forbidden, source, patches)
            linked = root / 'linked-work'; linked.mkdir()
            (linked / 'src').symlink_to(source / 'src', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Linked or non-directory'):
                self.prepare(linked, source, patches)

    def clone(self, root):
        destination = root / 'source'
        source_repository(self.source, destination)
        return destination

    def commit(self, source):
        subprocess.run(['git', '-C', str(source), 'add', '-A'], check=True)
        subprocess.run(['git', '-C', str(source), '-c', 'user.name=Local fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit', '--quiet', '-m',
                        'test: local synthetic source revision'], check=True)

    def test_new_clean_revision_comment_accepted_and_conflicting_patch_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix='sampler-commit-') as temporary:
            root = Path(temporary)
            source = self.clone(root)
            sampler_path = source / sampler.SOURCE_PATH
            sampler_path.write_text(sampler_path.read_text() + '\n// unrelated local fixture comment\n')
            self.commit(source)
            target = webgpu_source.prepare(source, root / 'output').parent / 'llama-sampler.cpp'
            self.assertIn('// unrelated local fixture comment', target.read_text())
            previous = {p.name: p.read_bytes() for p in target.parent.iterdir()}
            original = sampler_path.read_text()
            self.assertEqual(original.count('llama_get_sampled_token_ith     (ctx, idx)'), 1)
            sampler_path.write_text(original.replace('llama_get_sampled_token_ith     (ctx, idx)', 'incompatible_fixture_call(ctx, idx)', 1))
            self.commit(source)
            with self.assertRaisesRegex(ValueError, 'Patch application failed'):
                webgpu_source.prepare(source, target.parent)
            self.assertEqual(previous, {p.name: p.read_bytes() for p in target.parent.iterdir()})

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_dependency_type_break_is_rejected_by_complete_prepared_translation_unit(self):
        with tempfile.TemporaryDirectory(prefix='sampler-type-') as temporary:
            root = Path(temporary)
            source = self.clone(root)
            header = source / 'src/llama-vocab.h'
            value = header.read_text()
            self.assertEqual(value.count('llama_token token_eot() const;'), 1)
            header.write_text(value.replace('llama_token token_eot() const;', 'void        token_eot() const;'))
            self.commit(source)
            target = webgpu_source.prepare(source, root / 'output').parent / 'llama-sampler.cpp'
            command = ['c++', '-std=c++17', '-fsyntax-only', '-DGGML_SCHED_MAX_COPIES=4']
            command += ['-I' + str(source / relative) for relative in ['include', 'src', 'ggml/include', 'ggml/src']]
            result = subprocess.run(command + [str(target)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('void', result.stderr)

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
