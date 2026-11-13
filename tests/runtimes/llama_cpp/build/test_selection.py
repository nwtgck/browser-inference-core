"""Exercise real input selection around a simulated CMake compiler boundary."""
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from build_tools.runtimes.llama_cpp import build, inputs
from tests.runtimes.llama_cpp.source.test_inputs import CaptureFixture
from tests.support.llama_toolchain import merged_toolchain
from tests.runtimes.llama_cpp.model_graph_test_support import seed_model_graph_outputs


class DriverSelection(CaptureFixture):
    # Inherit fixture setup only; input-capture tests run in their own module.
    def setUp(self):
        super().setUp()
        toolchain = merged_toolchain()
        toolchain['llamaCommit'] = self.upstream_commit
        config = self.runtime / 'config'
        config.mkdir()
        (config / 'profiles.json').write_text(json.dumps({'cpu': {'asyncify': False, 'memory64': False,
            'webgpu': False, 'jspi': False, 'maximumMemory': 4294967296}}))
        (config / 'variants.json').write_text(json.dumps({'test': {'assertions': 2, 'environment': 'web,node'}}))
        (config / 'toolchain.json').write_text(json.dumps({'llamaCommit': self.upstream_commit}))
        (self.repository / 'toolchain/config.json').write_text(json.dumps({k: v for k, v in toolchain.items() if k not in {'llamaCommit', 'artifactBranch'}}))
        self.revision = self.commit(self.repository)
        self.build = self.runtime / 'build/cpu/test'
        self.on_configure = lambda source: None
        self.on_build = lambda: None
        self.commands = []
        run = subprocess.run
        check_output = subprocess.check_output

        def simulate(command, **kwargs):
            if command[0] == 'emcmake':
                self.commands.append((command, kwargs))
                self.build = Path(command[command.index('-B') + 1])
                source = Path(command[command.index('-S') + 1])
                self.assertTrue(source.is_relative_to(self.build / 'source-inputs'))
                self.assertEqual(kwargs['env']['GIT_CEILING_DIRECTORIES'], str(self.build / 'source-inputs'))
                self.assertIn('-DGIT_EXE:FILEPATH=', command)
                self.assertIn('-DLLAMA_BUILD_NUMBER=1', command)
                self.assertIn('-DLLAMA_BUILD_COMMIT=' + self.upstream_commit[:7], command)
                self.assertEqual((source / inputs.PATCH_DIRECTORY / 'accepted.patch').read_text(), 'committed patch bytes\n')
                webgpu = '-DLCB_WEBGPU=ON' in command
                profiling = '-DGGML_WEBGPU_GPU_PROFILE=ON' in command
                entries = [{'file': name, 'command': 'c++ -DLCB_MTMD_WEBGPU_BF16_RUNTIME=' + str(int(webgpu)) + ' -c ' + name}
                           for name in ['mtmd-model.cpp', 'mtmd-subcontext.cpp', 'mtmd.cpp']]
                if webgpu:
                    entries.append({'file': 'ggml-webgpu.cpp', 'command': 'c++ -c ggml-webgpu.cpp' + (' -DGGML_WEBGPU_GPU_PROFILE=1' if profiling else '')})
                if profiling:
                    entries.extend({'file': name, 'command': 'c++ -DLCB_WEBGPU_Q8_DIAGNOSTIC=1 -c ' + name}
                                   for name in ['webgpu-q8-diagnostic.cpp', 'q8-diagnostic-metadata.cpp'])
                (self.build / 'compile_commands.json').write_text(json.dumps(entries))
                seed_model_graph_outputs(self.build)
                self.on_configure(source)
                return subprocess.CompletedProcess(command, 0)
            if command[:2] == ['cmake', '--build']:
                self.commands.append((command, kwargs))
                self.on_build()
                return subprocess.CompletedProcess(command, 0)
            if '-m' in command and 'build_tools.runtimes.llama_cpp.browser_module' in command:
                # Actual browser byte transformation has its own tests. This
                # fixture checks captured source selection and compiler flags.
                return subprocess.CompletedProcess(command, 0, stdout='{}', stderr='')
            return run(command, **kwargs)

        def version(command, **kwargs):
            if command == ['emcc', '--version']:
                return 'emcc (explicit test fixture) ' + toolchain['emsdkVersion'] + '\n'
            return check_output(command, **kwargs)

        for mock in [patch.object(build, 'ROOT', self.runtime), patch.object(subprocess, 'run', side_effect=simulate),
                     patch.object(subprocess, 'check_output', side_effect=version)]:
            mock.start()
            self.addCleanup(mock.stop)

    def execute(self, *extra):
        build.main(['--profile', 'cpu', '--variant', 'test', *extra])
        return json.loads((self.build / 'provenance.json').read_text())

    def test_configuration_and_compile_commands_use_captured_commits(self):
        result = self.execute()
        self.assertFalse(result['sourceDirty'])
        self.assertEqual(result['sourceCommit'], self.revision)
        self.assertEqual(result['sourceInputs']['upstream']['commit'], self.upstream_commit)
        self.assertEqual(result['sourceInputs']['parentInputMode'], 'commit-objects')
        self.assertEqual(len(self.commands), 2)

    def test_clean_head_advance_during_build_is_reported_and_cannot_change_capture(self):
        def advance(source):
            (self.runtime / inputs.PATCH_DIRECTORY / 'accepted.patch').write_text('later committed patch\n')
            self.commit(self.repository)
            self.assertEqual((source / inputs.PATCH_DIRECTORY / 'accepted.patch').read_text(), 'committed patch bytes\n')
        self.on_configure = advance
        result = self.execute()
        self.assertTrue(result['sourceDirty'])
        self.assertEqual(result['sourceStatusAfterBuild'], [])
        self.assertNotEqual(result['sourceCommitAfterBuild'], self.revision)

    def test_failed_compile_invalidates_previous_success_receipt(self):
        self.execute()
        self.assertTrue((self.build / 'provenance.json').exists())
        def fail():
            raise subprocess.CalledProcessError(2, ['cmake', '--build'])
        self.on_build = fail
        with self.assertRaises(subprocess.CalledProcessError):
            self.execute()
        self.assertFalse((self.build / 'provenance.json').exists())

    def test_captured_input_corruption_prevents_success_provenance(self):
        def corrupt(source):
            (source / inputs.PATCH_DIRECTORY / 'accepted.patch').write_text('corrupt private input\n')
        self.on_configure = corrupt
        with self.assertRaisesRegex(ValueError, 'Captured repository inputs changed'):
            self.execute()
        self.assertFalse((self.build / 'provenance.json').exists())

    def test_staged_new_gitlink_does_not_reselect_parent_committed_upstream(self):
        self.run_git(self.repository, 'update-index', '--cacheinfo', f'160000,{self.revision},{inputs.UPSTREAM}')
        result = self.execute()
        self.assertEqual(result['llamaCommit'], self.upstream_commit)
        self.assertTrue(result['sourceDirty'])
        self.assertEqual(result['sourceInputs']['parentInputMode'], 'dirty-local-worktree')

    def test_jobs_must_be_positive(self):
        for jobs in ['0', '-1']:
            with self.subTest(jobs=jobs), self.assertRaises(SystemExit):
                self.execute('--jobs', jobs)
        self.assertEqual(self.commands, [])

    def test_existing_profiles_select_backend_without_optimization_flags(self):
        actual = Path(__file__).resolve().parents[4] / inputs.RUNTIME / 'config'
        profiles = json.loads((actual / 'profiles.json').read_text())
        for name in ['profiles.json', 'variants.json']:
            (self.runtime / 'config' / name).write_bytes((actual / name).read_bytes())
        dawn = self.repository / '.tools/emdawnwebgpu_pkg'
        dawn.mkdir(parents=True)
        (dawn / 'emdawnwebgpu.port.py').touch()
        self.revision = self.commit(self.repository)
        with patch.object(build, 'verify_asyncify_bigint_patch') as verify_patch, patch.object(build.shutil, 'which', return_value='/fixture/emcc'):
            for profile, config in profiles.items():
                for variant in ['browser', 'test']:
                    with self.subTest(profile=profile, variant=variant):
                        verify_patch.reset_mock()
                        build.main(['--profile', profile, '--variant', variant])
                        command = self.commands[-2][0]
                        self.assertIn('-DLCB_VARIANT=' + variant, command)
                        for field, setting in [('MEMORY64', 'memory64'), ('WEBGPU', 'webgpu'), ('JSPI', 'jspi'), ('ASYNCIFY', 'asyncify')]:
                            self.assertIn('-DLCB_' + field + '=' + ('ON' if config[setting] else 'OFF'), command)
                        self.assertIn('-DLCB_MAXIMUM_MEMORY=' + str(config['maximumMemory']), command)
                        self.assertIn('-DGGML_WEBGPU_GPU_PROFILE=' + ('ON' if config.get('gpuProfile') else 'OFF'), command)
                        self.assertIn('-DLCB_WEBGPU_BF16_PROJECTOR=' + ('ON' if config['webgpu'] else 'OFF'), command)
                        self.assertEqual(any(arg.startswith('-DEMDAWNWEBGPU_DIR=') for arg in command), config['webgpu'])
                        self.assertIn('-DLCB_WEBGPU=' + ('ON' if config['webgpu'] else 'OFF'), command)
                        self.assertFalse(any('TENSOR_COPY' in arg or 'PARAM_UPLOAD_BATCHING' in arg or 'LCB_WEBGPU_MOE_DIRECT_SLOT' in arg for arg in command))
                        recorded = json.loads((self.build / 'provenance.json').read_text())
                        self.assertEqual(recorded['cmakeCommand'], command)
                        self.assertEqual(recorded['sourceInputs']['sourceCommit'], self.revision)
                        self.assertEqual(recorded['configuration'], config)
                        self.assertEqual(recorded['variant'], variant)
                        self.assertEqual(recorded['variantConfiguration'], json.loads((actual / 'variants.json').read_text())[variant])
                        self.assertEqual('browserModulePreparation' in recorded, variant == 'browser')
                        if config['asyncify']:
                            verify_patch.assert_called_once_with(Path('/fixture'), merged_toolchain()['emscriptenAsyncifyBigIntPatch'])
                        else:
                            verify_patch.assert_not_called()


    def test_committed_configuration_cannot_select_source_owned_output_paths(self):
        profile = self.runtime / 'config/profiles.json'
        value = json.loads(profile.read_text())
        value['../source-owned'] = value['cpu']
        profile.write_text(json.dumps(value))
        self.commit(self.repository)
        with self.assertRaises(SystemExit):
            build.main(['--profile', '../source-owned', '--variant', 'test', '--fresh'])
        self.assertEqual(self.commands, [])
        self.assertTrue((self.runtime / 'CMakeLists.txt').is_file())
