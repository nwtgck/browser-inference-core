"""Driver contracts with fake compiler outputs, not a Wasm build."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import REPOSITORY_ROOT, runtime_root
from build_tools.runtimes.stable_diffusion_cpp import build
from build_tools.source.inputs import file_identities, tree_identity


class ImageBuild(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.root = self.repo / 'runtimes/stable-diffusion-cpp'
        shutil.copytree(runtime_root('stable-diffusion-cpp') / 'config', self.root / 'config')
        self.profiles = json.loads((self.root / 'config/profiles.json').read_text())
        port = self.repo / '.tools/emdawnwebgpu_pkg/emdawnwebgpu.port.py'
        port.parent.mkdir(parents=True)
        port.write_text('# Test input; never executed\n')
        self.upstreams = {'stable-diffusion': {'commit': 'b' * 40},
                          'ggml': {'commit': 'c' * 40, 'sourceSubdirectory': 'ggml'}}
        self.commands = []
        self.selections = []
        self.resolutions = []
        self.git_calls = []
        self.config = {'emsdkVersion': '6.0.9', 'emscriptenAsyncifyBigIntPatch': {'fixture': True}}

    def invoke(self, profile='webgpu-wasm32-jspi', variant='browser', *, dirty_before=False, dirty_after=False,
               changed_head=False, switch_during_selection=False, extra=(), fail=False,
               mutate_captured=False, mutate_prepared=False):
        build_dir = self.root / 'build' / profile / variant
        head = 'a' * 40
        statuses = []
        committed = {str(path.relative_to(self.repo)): path.read_text() for path in (self.root / 'config').glob('*.json')}
        receipt = {'upstreams': self.upstreams, 'patches': []}
        def git(*args):
            self.git_calls.append(args)
            if args == ('rev-parse', 'HEAD'):
                return head
            self.assertEqual(args[0], 'status')
            dirty = dirty_after if statuses else dirty_before
            statuses.append(True)
            return ' M source.cpp' if dirty else ''
        def committed_file(repository, revision, relative):
            self.assertEqual(repository, self.repo)
            self.assertEqual(revision, 'a' * 40)
            self.assertGreaterEqual(len(self.git_calls), 2)
            return committed[relative]
        def resolve(repository, runtime, **options):
            nonlocal head
            self.resolutions.append(options)
            if switch_during_selection:
                head = 'd' * 40
            return {}, self.upstreams
        def prepare(sources, output, **options):
            self.selections.append(options)
            output.mkdir(parents=True, exist_ok=True)
            (output / 'patch-provenance.json').write_text(json.dumps(receipt))
            return receipt
        def capture(repository, revision, output, *, dirty=False):
            self.assertEqual(repository, self.repo)
            self.assertEqual(revision, 'a' * 40)
            self.assertEqual(dirty, dirty_before)
            exported = output / 'repository'
            for name, content in committed.items():
                path = exported / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            (exported / 'toolchain').mkdir(exist_ok=True)
            (exported / 'toolchain/config.json').write_text(json.dumps(self.config))
            record = {'repositoryTree': tree_identity(file_identities(exported)), 'sourceCommit': revision}
            return build.inputs.Captured(output, record)
        def run(command, **kwargs):
            nonlocal head
            self.commands.append(command)
            self.assertEqual(kwargs['cwd'], build_dir)
            captured_root = build_dir / 'source-inputs/repository'
            self.assertEqual(kwargs['env']['PYTHONPATH'], str(captured_root))
            self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
            if command[:2] == ['emcmake', 'cmake']:
                self.assertEqual(command[command.index('-S') + 1], str(captured_root / 'runtimes/stable-diffusion-cpp'))
            if command[:2] == ['cmake', '--build']:
                if fail:
                    raise subprocess.CalledProcessError(1, command)
                if changed_head:
                    head = 'd' * 40
                if mutate_captured:
                    (captured_root / 'toolchain/config.json').write_text('changed during compilation')
                if mutate_prepared:
                    (build_dir / 'prepared/changed.cpp').write_text('changed during compilation')
                for part, names in (('runtime', ('core.mjs', 'core.wasm', 'core.d.ts')),
                                    ('generated', ('schema.json', 'schema.mjs', 'functions.d.ts', 'exports.json'))):
                    (build_dir / part).mkdir(exist_ok=True)
                    for name in names:
                        (build_dir / part / name).write_text('synthetic driver output\n')
        with ExitStack() as stack:
            stack.enter_context(patch.object(build, 'ROOT', self.root))
            stack.enter_context(patch.object(build, 'REPO', self.repo))
            stack.enter_context(patch.object(build, 'resolve_sources', side_effect=resolve))
            toolchain = stack.enter_context(patch.object(build, 'load_toolchain', return_value=self.config))
            stack.enter_context(patch.object(build, 'commit_file', side_effect=committed_file))
            stack.enter_context(patch.object(build, 'prepare', side_effect=prepare))
            stack.enter_context(patch.object(build.inputs, 'capture', side_effect=capture))
            stack.enter_context(patch.object(build, 'git', side_effect=git))
            stack.enter_context(patch.object(build.subprocess, 'check_output', return_value='emcc 6.0.9\n'))
            stack.enter_context(patch.object(build.subprocess, 'run', side_effect=run))
            stack.enter_context(patch.object(build.shutil, 'which', return_value='/fixture/emcc'))
            checked = stack.enter_context(patch.object(build, 'verify_asyncify_bigint_patch'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            build.main(['--profile', profile, '--variant', variant, '--jobs', '2', *extra])
            toolchain.assert_called_once_with(self.repo, revision=None if dirty_before else 'a' * 40)
        return build_dir, checked

    def test_all_profile_variants_have_distinct_commands_receipts_and_staging(self):
        outputs = []
        for profile, flags in self.profiles.items():
            for variant in ('browser', 'test'):
                with self.subTest(profile=profile, variant=variant):
                    directory, checked = self.invoke(profile, variant)
                    outputs.append(directory)
                    command = self.commands[-2]
                    self.assertIn('-DSDCB_VARIANT=' + variant, command)
                    for key, flag in (('JSPI', 'jspi'), ('MEMORY64', 'memory64')):
                        self.assertIn(f'-DSDCB_{key}=' + ('ON' if flags[flag] else 'OFF'), command)
                    receipt_path = directory / 'prepared/patch-provenance.json'
                    self.assertIn('-DSDCB_PROVENANCE=' + str(receipt_path), command)
                    self.assertEqual(json.loads(receipt_path.read_text())['upstreams'], self.upstreams)
                    self.assertEqual(checked.call_count, int(flags['asyncify']))
                    self.assertEqual(self.selections[-1], {
                        'patches': directory / 'source-inputs/repository/runtimes/stable-diffusion-cpp/upstream-patches',
                        'upstreams': self.upstreams, 'replace': True})
                    self.assertEqual(self.resolutions[-1], {'revision': 'a' * 40})
                    stage = self.root / 'build/ci-upload' / profile / variant
                    self.assertTrue((stage / 'runtime/core.wasm').is_file())
                    self.assertFalse((stage / 'prepared').exists())
                    self.assertFalse(json.loads((stage / 'provenance.json').read_text())['sourceDirty'])
                    data = json.loads((stage / 'provenance.json').read_text())
                    self.assertEqual(data['sourceInputs']['sourceCommit'], 'a' * 40)
                    self.assertEqual(data['preparedSources'], tree_identity(file_identities(directory / 'prepared')))
        self.assertEqual(len(outputs), 6)
        self.assertEqual(len(set(outputs)), 6)

    def test_dirty_result_does_not_replace_an_earlier_staged_package(self):
        stage = self.root / 'build/ci-upload/webgpu-wasm32-jspi/browser'
        stage.mkdir(parents=True)
        (stage / 'keep').write_text('previous validated candidate')
        with self.assertRaisesRegex(RuntimeError, 'Source changed'):
            self.invoke(dirty_after=True)
        self.assertEqual([p.name for p in stage.iterdir()], ['keep'])
        self.assertEqual((stage / 'keep').read_text(), 'previous validated candidate')

    def test_clean_commit_change_is_not_mistaken_for_unchanged_source(self):
        stage = self.root / 'build/ci-upload/webgpu-wasm32-jspi/browser'
        stage.mkdir(parents=True)
        (stage / 'keep').write_text('previous validated candidate')
        with self.assertRaisesRegex(RuntimeError, 'Source changed'):
            self.invoke(changed_head=True)
        provenance = json.loads((self.root / 'build/webgpu-wasm32-jspi/browser/provenance.json').read_text())
        self.assertEqual(provenance['sourceCommit'], 'a' * 40)
        self.assertEqual(provenance['sourceCommitAfterBuild'], 'd' * 40)
        self.assertEqual(provenance['sourceStatusAfterBuild'], [])
        self.assertTrue(provenance['sourceDirty'])
        self.assertEqual((stage / 'keep').read_text(), 'previous validated candidate')

    def test_compiler_failure_does_not_stage_success(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(fail=True)
        self.assertFalse((self.root / 'build/ci-upload').exists())

    def test_commit_switch_during_selection_cannot_become_the_baseline(self):
        with self.assertRaisesRegex(RuntimeError, 'Source changed'):
            self.invoke(switch_during_selection=True)
        self.assertEqual(self.git_calls[0], ('rev-parse', 'HEAD'))
        self.assertEqual(self.git_calls[1][0], 'status')
        self.assertEqual(self.resolutions, [{'revision': 'a' * 40}])
        provenance = json.loads((self.root / 'build/webgpu-wasm32-jspi/browser/provenance.json').read_text())
        self.assertEqual(provenance['sourceCommit'], 'a' * 40)
        self.assertEqual(provenance['sourceCommitAfterBuild'], 'd' * 40)
        self.assertFalse((self.root / 'build/ci-upload').exists())

    def test_dirty_development_selection_does_not_stage_publishable_output(self):
        with self.assertRaisesRegex(RuntimeError, 'Source changed'):
            self.invoke(dirty_before=True)
        self.assertEqual(self.resolutions, [{'revision': None}])
        self.assertFalse((self.root / 'build/ci-upload').exists())

    def test_failed_rebuild_invalidates_previous_success_receipt(self):
        receipt = self.root / 'build/webgpu-wasm32-jspi/browser/provenance.json'
        receipt.parent.mkdir(parents=True)
        receipt.write_text('{"validation":{"compiled":true}}')
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(fail=True)
        self.assertFalse(receipt.exists())

    def test_changed_captured_or_prepared_inputs_cannot_replace_an_earlier_stage(self):
        stage = self.root / 'build/ci-upload/webgpu-wasm32-jspi/browser'
        stage.mkdir(parents=True)
        (stage / 'keep').write_text('previous valid output')
        for option, message in [('mutate_captured', 'Captured image-runtime'),
                                ('mutate_prepared', 'Prepared image sources')]:
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, message):
                self.invoke(**{option: True})
            self.assertEqual([path.name for path in stage.iterdir()], ['keep'])
            self.assertEqual((stage / 'keep').read_text(), 'previous valid output')
            self.assertFalse((self.root / 'build/webgpu-wasm32-jspi/browser/provenance.json').exists())

    def test_fresh_build_does_not_delete_a_sibling_profile(self):
        sibling = self.root / 'build/webgpu-wasm64-jspi/browser/keep'
        sibling.parent.mkdir(parents=True)
        sibling.write_text('independent profile')
        stale = self.root / 'build/webgpu-wasm32-jspi/browser/stale'
        stale.parent.mkdir(parents=True)
        stale.write_text('old build')
        self.invoke(extra=['--fresh'])
        self.assertFalse(stale.exists())
        self.assertEqual(sibling.read_text(), 'independent profile')

    def test_module_cli_is_available_without_path_bootstrapping(self):
        for module in (build.__name__, 'build_tools.runtimes.stable_diffusion_cpp.bindings'):
            result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=REPOSITORY_ROOT,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_or_unsafe_selection_fails_before_preparing(self):
        for args in (['--profile', '../escape', '--variant', 'browser'],
                     ['--profile', 'webgpu-wasm32-jspi', '--variant', '../escape'],
                     ['--profile', 'webgpu-wasm32-jspi', '--variant', 'browser', '--jobs', '0']):
            with self.subTest(args=args), patch.object(build, 'resolve_sources') as resolve:
                with self.assertRaises(SystemExit):
                    build.main(args)
                resolve.assert_not_called()
