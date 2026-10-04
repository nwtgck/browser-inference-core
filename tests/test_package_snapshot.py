"""Synthetic payloads and mocked smoke only; actual npm final packing is exercised."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_multi_runtime as fixtures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import validate_runtime_package as snapshot
from package_inputs import copy_regular, copy_regular_tree, tree_identity, regular_path, read_regular


class PackageInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'; self.source.mkdir()
        (self.source / 'file').write_bytes(b'abcdef')

    def test_regular_copy_keeps_tree_and_executable_bits(self):
        (self.source / 'file').chmod(0o755)
        copy_regular_tree(self.source, self.root / 'out')
        self.assertEqual(tree_identity(self.source), tree_identity(self.root / 'out'))

    def test_root_parent_file_directory_and_broken_links_are_rejected(self):
        for target in (self.source, self.source / 'file', self.root / 'missing'):
            link = self.source / 'link'; link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'Linked'):
                tree_identity(self.source)
            link.unlink()
        (self.root / 'parent-link').symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'parent'):
            read_regular(self.root / 'parent-link/file')
        with self.assertRaisesRegex(ValueError, 'Linked'):
            tree_identity(self.root / 'parent-link')

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires named pipes')
    def test_fifo_is_rejected_without_opening_or_blocking(self):
        os.mkfifo(self.source / 'pipe')
        with self.assertRaisesRegex(ValueError, 'non-regular'):
            tree_identity(self.source)
        with self.assertRaisesRegex(ValueError, 'non-regular'):
            read_regular(self.source / 'pipe')

    def test_case_collision_and_backslash_are_rejected(self):
        for name in ('FILE', 'bad\\path'):
            path = self.source / name; path.write_text('test')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                tree_identity(self.source)
            path.unlink()

    def test_copy_time_same_size_change_is_detected(self):
        original = shutil.copytree
        def changed(*args, **kwargs):
            value = original(*args, **kwargs)
            (args[1] / 'file').write_bytes(b'ghijkl')
            return value
        with patch('package_inputs.shutil.copytree', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'changed'):
                copy_regular_tree(self.source, self.root / 'out')

    def test_copy_time_link_is_rejected_not_followed(self):
        original = shutil.copytree
        def linked(source, target, **kwargs):
            (source / 'file').unlink()
            (source / 'file').symlink_to(self.root / 'secret')
            return original(source, target, **kwargs)
        (self.root / 'secret').write_text('never dereference')
        with patch('package_inputs.shutil.copytree', side_effect=linked):
            with self.assertRaisesRegex(ValueError, 'Linked'):
                copy_regular_tree(self.source, self.root / 'out')
        self.assertTrue((self.root / 'out/file').is_symlink())

    def test_copy_never_overwrites_existing_destination_link(self):
        target = self.root / 'target'; target.write_bytes(b'unchanged')
        link = self.root / 'destination'; link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            copy_regular(self.source / 'file', link)
        self.assertEqual(target.read_bytes(), b'unchanged')

    def test_linked_destination_parent_is_rejected_before_copy(self):
        target = self.root / 'target'; target.mkdir()
        link = self.root / 'linked-parent'; link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            copy_regular_tree(self.source, link / 'copy')
        self.assertEqual(list(target.iterdir()), [])

    def test_metadata_reads_are_bounded(self):
        with self.assertRaisesRegex(ValueError, 'Oversized'):
            read_regular(self.source / 'file', limit=5)


class ImageSmokeResults(unittest.TestCase):
    def setUp(self):
        runtime = ROOT / 'stable-diffusion-cpp'
        self.profiles = json.loads((runtime / 'config/profiles.json').read_text())
        self.variants = json.loads((runtime / 'config/variants.json').read_text())

    def results(self, test_webgpu):
        # Match the producer's emitted contract independently of image_scope().
        scope = ('real-Wasm Worker, public records/callbacks, sparse GGUF/safetensors/shard I/O; '
                 'test variants also check synthetic Qwen BF16 timestep and 3D convolution bias graph arithmetic on ' +
                 ('CPU and WebGPU' if test_webgpu else 'CPU (no GPU inference)') +
                 ', plus deep graph construction/selection; no trained-model image generation')
        return [{'profile': profile, 'variant': variant, 'passed': True, 'scope': scope,
                 **({'graphWalk': True, 'conv3dBias': [
                     {'backend': backend, 'passed': True}
                     for backend in (['CPU', 'WebGPU'] if test_webgpu else ['CPU'])
                 ]} if variant == 'test' else {})}
                for profile in self.profiles for variant in self.variants]

    def check(self, results, test_webgpu):
        return snapshot.checked_results('stable-diffusion-cpp', self.profiles, self.variants,
                                        results, test_webgpu=test_webgpu)

    def test_accepts_cpu_and_webgpu_scopes_with_test_only_graph_evidence(self):
        for test_webgpu in (False, True):
            with self.subTest(test_webgpu=test_webgpu):
                results = self.results(test_webgpu)
                self.assertEqual(self.check(results, test_webgpu), results)

    def test_rejects_old_scope_and_wrong_backend_scope(self):
        for test_webgpu in (False, True):
            results = self.results(test_webgpu)
            results[0]['scope'] = results[0]['scope'].replace(', plus deep graph construction/selection', '')
            with self.subTest(old_scope=test_webgpu), self.assertRaisesRegex(ValueError, 'Wrong image smoke scope'):
                self.check(results, test_webgpu)
            with self.subTest(wrong_backend=test_webgpu), self.assertRaisesRegex(ValueError, 'Wrong image smoke scope'):
                self.check(self.results(not test_webgpu), test_webgpu)

    def test_requires_true_graph_walk_for_every_test_profile(self):
        for profile in self.profiles:
            for value in (False, None, 0, 1, 'true'):
                results = self.results(False)
                result = next(item for item in results if item['profile'] == profile and item['variant'] == 'test')
                result['graphWalk'] = value
                with self.subTest(profile=profile, value=value), self.assertRaisesRegex(ValueError, 'graph walk evidence'):
                    self.check(results, False)
            results = self.results(False)
            result = next(item for item in results if item['profile'] == profile and item['variant'] == 'test')
            result.pop('graphWalk')
            with self.subTest(profile=profile, missing=True), self.assertRaisesRegex(ValueError, 'graph walk evidence'):
                self.check(results, False)

    def test_rejects_graph_walk_key_on_every_browser_profile(self):
        for profile in self.profiles:
            for value in (True, False, None):
                results = self.results(False)
                result = next(item for item in results if item['profile'] == profile and item['variant'] == 'browser')
                result['graphWalk'] = value
                with self.subTest(profile=profile, value=value), self.assertRaisesRegex(ValueError, 'graph walk evidence'):
                    self.check(results, False)

    def test_requires_convolution_evidence_from_every_expected_backend(self):
        for test_webgpu in (False, True):
            for value in (None, [], [{'backend': 'CPU', 'passed': False}],
                          [{'backend': 'CPU', 'passed': 1}], [{'backend': 'WebGPU', 'passed': True}]):
                results = self.results(test_webgpu)
                result = next(item for item in results if item['variant'] == 'test')
                result['conv3dBias'] = value
                with self.subTest(test_webgpu=test_webgpu, value=value), self.assertRaisesRegex(ValueError, 'convolution bias evidence'):
                    self.check(results, test_webgpu)
            results = self.results(test_webgpu)
            result = next(item for item in results if item['variant'] == 'test')
            result.pop('conv3dBias')
            with self.subTest(test_webgpu=test_webgpu, missing=True), self.assertRaisesRegex(ValueError, 'convolution bias evidence'):
                self.check(results, test_webgpu)
        results = self.results(True)
        result = next(item for item in results if item['variant'] == 'test')
        result['conv3dBias'] = [{'backend': 'CPU', 'passed': True}]
        with self.assertRaisesRegex(ValueError, 'convolution bias evidence'):
            self.check(results, True)

    def test_rejects_convolution_probe_in_browser_variants_and_old_scope(self):
        results = self.results(False)
        result = next(item for item in results if item['variant'] == 'browser')
        result['conv3dBias'] = None
        with self.assertRaisesRegex(ValueError, 'convolution bias evidence'):
            self.check(results, False)
        results = self.results(False)
        results[0]['scope'] = results[0]['scope'].replace(' and 3D convolution bias', '')
        with self.assertRaisesRegex(ValueError, 'Wrong image smoke scope'):
            self.check(results, False)


class RuntimeSnapshot(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MultiRuntime(); self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.repo = self.root / 'repo'
        for runtime in snapshot.RUNTIMES:
            rr = self.repo / runtime
            (rr / 'config').mkdir(parents=True)
            for name in ('profiles.json', 'variants.json'):
                shutil.copy2(ROOT / runtime / 'config' / name, rr / 'config' / name)
            for name in snapshot.TEST_INPUTS[runtime]:
                p = rr / name; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('Synthetic test-input identity fixture, NOT executable smoke code.\n')
        # Expand the root's intentionally tiny llama fixture to all configured profiles.
        llama = self.fixture.inputs / 'llama-cpp'
        manifest = json.loads((llama / 'manifest.json').read_text())
        profiles = json.loads((ROOT / 'llama-cpp/config/profiles.json').read_text())
        for profile in profiles:
            if profile == 'cpu-wasm32': continue
            shutil.copytree(llama / 'profiles/cpu-wasm32', llama / 'profiles' / profile)
            data = copy.deepcopy(manifest['profiles']['cpu-wasm32'])
            for item in data['variants'].values(): item['profile'] = profile
            manifest['profiles'][profile] = data
        fixtures.write_manifest(llama, manifest)
        self.env = patch.dict(os.environ, {'LCB_SOURCE_COMMIT': fixtures.SOURCE,
            'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2', 'GITHUB_JOB': 'fixture',
            'BIC_METRICS_MODE': 'off', 'SDCB_TEST_WEBGPU': '0'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.root_patch = patch.object(snapshot, 'ROOT', self.repo)
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        self.runtime = 'stable-diffusion-cpp'
        self.result_mutator = None; self.after_command = None; self.commands = []

    def package(self): return self.fixture.inputs / self.runtime
    def receipt(self): return self.root / (self.runtime + '-receipt.json')

    def results(self):
        rr = self.repo / self.runtime
        profiles = json.loads((rr / 'config/profiles.json').read_text())
        variants = json.loads((rr / 'config/variants.json').read_text())
        result = []
        for profile in profiles:
            for variant in variants:
                item = {'profile': profile, 'variant': variant, 'passed': True}
                if self.runtime == 'stable-diffusion-cpp':
                    item['scope'] = snapshot.image_scope(False)
                    if variant == 'test':
                        item['graphWalk'] = True
                        item['conv3dBias'] = [{'backend': 'CPU', 'passed': True}]
                elif profile.startswith('cpu-'): item['syntheticModel'] = True
                else: item.update(mockedAdapter=True, suspension=True)
                result.append(item)
        return result

    def fake_run(self, command, phase, cwd, env=None):
        self.commands.append((phase, command))
        if phase == 'test.generate_fixture':
            p = cwd / 'build/fixture.gguf'; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'GGUF synthetic identity only')
        if phase == 'test.chromium_smoke':
            envelope = {'session': json.loads(env['BIC_BROWSER_SESSION_JSON']), 'results': self.results()}
            if self.result_mutator: self.result_mutator(envelope)
            Path(env['BIC_BROWSER_RESULTS_FILE']).write_text(json.dumps(envelope))
        if self.after_command: self.after_command(phase)

    def validate(self):
        with patch.object(snapshot, '_run', side_effect=self.fake_run):
            return snapshot.validate_package(self.runtime, self.package(), self.receipt())

    def test_both_runtimes_pack_once_and_only_change_validation_fields(self):
        for runtime in snapshot.RUNTIMES:
            self.runtime = runtime
            before = tree_identity(self.package())
            manifest = json.loads((self.package() / 'manifest.json').read_text())
            original = subprocess.check_output
            with patch('subprocess.check_output', wraps=original) as calls:
                receipt = self.validate()
            packs = [call for call in calls.call_args_list if call.args[0][:2] == ['npm', 'pack']]
            self.assertEqual(len(packs), 1)
            after = tree_identity(self.package())
            self.assertEqual({k:v for k,v in before.items() if k != 'manifest.json'},
                             {k:v for k,v in after.items() if k != 'manifest.json'})
            final = json.loads((self.package() / 'manifest.json').read_text())
            self.assertEqual(final, snapshot.add_validation(manifest, runtime, self.results()))
            self.assertEqual(receipt['status'], 'complete')
            self.assertTrue(self.receipt().is_file())
            if runtime == 'llama-cpp':
                self.assertTrue(any(p == 'test.node_asyncify' for p, _ in self.commands))
            for info in final['profiles'].values():
                for data in info['variants'].values(): self.assertIs(data['validation']['realModelInference'], False)

    def prepare_nightly_subset(self):
        self.runtime='llama-cpp';rr=self.repo/self.runtime
        config=json.loads((ROOT/'llama-cpp/config/sources.json').read_text())
        (rr/'config/sources.json').write_text(json.dumps(config))
        nightly=config['sources']['upstream-nightly']
        p=rr/nightly['pinFile'];p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'llamaCommit':'b'*40}))
        (rr/nightly['patchSeries']).write_text('{"testIdentityOnly":true}')
        template=rr/nightly['vendorPath']/'models/templates/Qwen-Qwen3-0.6B.jinja'
        template.parent.mkdir(parents=True,exist_ok=True);template.write_text('nightly synthetic template')
        m=json.loads((self.package()/'manifest.json').read_text());m['sourceId']='upstream-nightly'
        for profile in list(m['profiles']):
            if profile not in nightly['profiles']:
                del m['profiles'][profile];shutil.rmtree(self.package()/'profiles'/profile)
            else:
                for record in m['profiles'][profile]['variants'].values():record['sourceId']='upstream-nightly'
        fixtures.write_manifest(self.package(),m)
        self.result_mutator=lambda envelope:envelope.__setitem__('results',[item for item in envelope['results'] if item['profile'] in nightly['profiles']])
        return template

    def test_nightly_smoke_checks_only_registered_subset_without_claiming_asyncify(self):
        template=self.prepare_nightly_subset();seen=[];original=self.fake_run
        def run(command,phase,cwd,env=None):
            if phase=='test.chromium_smoke':seen.append(env['BIC_LLAMA_CHAT_TEMPLATE'])
            return original(command,phase,cwd,env)
        with patch.object(snapshot,'_run',side_effect=run):
            receipt=snapshot.validate_package(self.runtime,self.package(),self.receipt())
        self.assertEqual(len(receipt['results']),4)
        self.assertIsNone(receipt['nodeAsyncifyPassed'])
        self.assertFalse(any(phase=='test.node_asyncify' for phase,_ in self.commands))
        self.assertEqual(seen,[str(template)])

    def test_nightly_template_change_and_wrong_pin_fail_closed(self):
        template=self.prepare_nightly_subset()
        self.after_command=lambda _:template.write_text('mutated template')
        with self.assertRaisesRegex(ValueError,'implementation'):self.validate()
        self.assertFalse(self.receipt().exists())
        self.after_command=None
        pin=self.repo/'llama-cpp/sources/upstream-nightly/pin.json';pin.write_text(json.dumps({'llamaCommit':'f'*40}))
        with self.assertRaisesRegex(ValueError,'pin mismatch'):self.validate()

    def test_owned_temporary_directory_resolves_system_alias_before_use(self):
        actual = self.root / 'temporary-root'; actual.mkdir()
        alias = self.root / 'system-temp-alias'; alias.symlink_to(actual, target_is_directory=True)
        with patch.object(tempfile, 'tempdir', str(alias)):
            self.validate()
        self.assertTrue(self.receipt().is_file())

    def test_source_build_mutation_is_not_reimported(self):
        def mutate(_):
            path = self.repo / self.runtime / 'build/unused/core.mjs'
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('changed build, must not be copied')
        self.after_command = mutate
        self.validate()
        self.assertNotIn('unused', ''.join(tree_identity(self.package())))

    def test_same_size_payload_change_after_smoke_blocks_final_pack(self):
        file = next(self.package().glob('profiles/*/*/core.mjs'))
        self.after_command = lambda phase: file.write_bytes(b'X' * file.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            self.validate()
        self.assertFalse(self.receipt().exists())

    def test_modified_manifest_with_recomputed_hashes_is_still_rejected(self):
        def mutate(_):
            file = next(self.package().glob('profiles/*/*/core.mjs')); file.write_text('changed')
            manifest = json.loads((self.package() / 'manifest.json').read_text())
            fixtures.write_manifest(self.package(), manifest)
        self.after_command = mutate
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            self.validate()

    def test_test_implementation_change_is_rejected(self):
        self.after_command = lambda _: (self.repo / self.runtime / 'tests/browser-smoke.mjs').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'implementation or fixture'):
            self.validate()

    def test_previous_run_attempt_and_context_types_cannot_be_replayed(self):
        for key, value in [('runId', '999'), ('runAttempt', '1'), ('sessionId', 'a' * 32),
                           ('sourceCommit', 'b'*40), ('testWebGpu', 0)]:
            self.result_mutator = lambda env, k=key, v=value: env['session'].__setitem__(k, v)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'not bound'):
                self.validate()
        self.assertFalse(self.receipt().exists())

    def test_string_or_integer_success_and_missing_duplicate_or_extra_results_fail(self):
        mutations = [lambda env, v=value: env['results'][0].__setitem__('passed', v)
                     for value in ('false', 'true', 1, 0, False, None)]
        mutations += [lambda env: env['results'].pop(),
                      lambda env: env['results'].append(env['results'][0]),
                      lambda env: env['results'].__setitem__(1, env['results'][0]),
                      lambda env: env['results'][0].__setitem__('scope', 'real model inference')]
        for change in mutations:
            self.result_mutator = change
            with self.assertRaises(ValueError): self.validate()
        self.assertFalse(self.receipt().exists())

    def test_wrong_llama_scope_fails(self):
        self.runtime = 'llama-cpp'
        self.result_mutator = lambda env: env['results'][0].__setitem__('syntheticModel', 'true')
        with self.assertRaisesRegex(ValueError, 'scope'): self.validate()

    def test_missing_graph_walk_evidence_cannot_finalize_or_leave_a_receipt(self):
        before = (self.package() / 'manifest.json').read_bytes()
        self.receipt().write_text('{"status":"complete"}')
        def mutate(envelope):
            next(item for item in envelope['results'] if item['variant'] == 'test').pop('graphWalk')
        self.result_mutator = mutate
        with self.assertRaisesRegex(ValueError, 'graph walk evidence'):
            self.validate()
        self.assertEqual((self.package() / 'manifest.json').read_bytes(), before)
        self.assertFalse(self.receipt().exists())

    def test_failed_child_preserves_failure_and_removes_old_success_receipt(self):
        self.receipt().write_text('{"status":"complete"}')
        with patch.object(snapshot, '_run', side_effect=subprocess.CalledProcessError(7, ['fixture'])):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                snapshot.validate_package(self.runtime, self.package(), self.receipt())
        self.assertEqual(error.exception.returncode, 7)
        self.assertFalse(self.receipt().exists())

    def test_missing_new_result_does_not_reuse_build_result(self):
        rr = self.repo / self.runtime / 'build'; rr.mkdir(exist_ok=True)
        (rr / 'browser-results.json').write_text(json.dumps(self.results()))
        with patch.object(snapshot, '_run'):
            with self.assertRaises(FileNotFoundError):
                snapshot.validate_package(self.runtime, self.package(), self.receipt())

    def test_npm_failure_restores_unvalidated_manifest(self):
        before = (self.package() / 'manifest.json').read_bytes()
        with patch('subprocess.check_output', side_effect=subprocess.CalledProcessError(6, ['npm'])):
            with self.assertRaises(subprocess.CalledProcessError): self.validate()
        self.assertEqual(before, (self.package() / 'manifest.json').read_bytes())
        self.assertFalse(self.receipt().exists())

    def test_modification_by_npm_blocks_success_receipt(self):
        original = subprocess.check_output
        def changed(*args, **kwargs):
            value = original(*args, **kwargs)
            if args[0][:2] == ['npm', 'pack']:
                file = next(self.package().glob('profiles/*/*/core.mjs'))
                file.write_bytes(b'Y' * file.stat().st_size)
            return value
        with patch('subprocess.check_output', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'snapshot changed'): self.validate()
        self.assertFalse(self.receipt().exists())

    def test_dirty_source_and_source_mismatch_fail_before_tests(self):
        with patch.dict(os.environ, LCB_SOURCE_COMMIT='b'*40):
            with self.assertRaisesRegex(ValueError, 'selected source'): self.validate()
        file = self.package() / 'manifest.json'
        data = json.loads(file.read_text())
        next(iter(next(iter(data['profiles'].values()))['variants'].values()))['sourceDirty'] = True
        file.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'dirty'): self.validate()
        self.assertEqual(self.commands, [])

    def test_receipt_cannot_delete_manifest(self):
        before = (self.package() / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'outside'):
            snapshot.validate_package(self.runtime, self.package(), self.package() / 'manifest.json')
        self.assertEqual(before, (self.package() / 'manifest.json').read_bytes())

    def test_json_rejects_duplicates_and_nonstandard_numbers(self):
        for raw in (b'{"passed":false,"passed":true}', b'{"value":NaN}'):
            with self.assertRaises(ValueError): snapshot.parse_json(raw)


if __name__ == '__main__': unittest.main()
