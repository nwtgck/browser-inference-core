"""Exercise publication with synthetic packages, real npm packing and local Git.

No compiler, model inference, GitHub push or network lock resolution is claimed.
"""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_multi_runtime as fixtures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import package_runtime as package
import publish_artifacts as publisher
import consumer_metadata as reporter


def git(*args, cwd):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def tree(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob('*') if p.is_file()}


def pack_calls(mock):
    return [call for call in mock.call_args_list
            if call.args[0][:3] == ['npm', 'pack', '--dry-run']]


class PublicationOptimization(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MultiRuntime()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root, self.inputs, self.out = self.fixture.root, self.fixture.inputs, self.fixture.out
        self.remote = self.root / 'remote.git'
        git('init', '--bare', '--quiet', str(self.remote), cwd=self.root)
        self.env = {'GITHUB_REPOSITORY': 'example/lcore', 'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2',
                    'GITHUB_STEP_SUMMARY': str(self.root / 'test-step-summary')}

    def assemble(self):
        package.assemble(self.inputs, self.out, check_npm_pack=False)

    def test_workflow_summary_adds_details_without_repeating_install_command(self):
        self.assemble()
        summary = Path(self.env['GITHUB_STEP_SUMMARY'])
        summary.write_text('## Runtime artifact commit\n\n```sh\nnpm install github:example/lcore#' + 'd' * 40 + '\n```\n')
        self.report_main(digest=package.identity(self.out / 'manifest.json')['sha256'])
        text = summary.read_text()
        self.assertEqual(text.count('npm install github:example/lcore#'), 1)
        self.assertEqual(text.count('## Runtime artifact commit'), 1)
        self.assertNotIn('## Runtime artifact published', text)
        self.assertIn('## Consumer integration details', text)
        self.assertIn('stableDiffusion:', text)
        self.assertIn('consumerIntegration:', text)
        self.assertIn('npm install', (self.root / 'report/consumer-update.md').read_text())

    def report_main(self, *, digest=None, lock_effect=None, commit='d' * 40):
        args = ['consumer_metadata.py', '--package', str(self.out), '--commit', commit,
                '--output', str(self.root / 'report')]
        if digest is not None:
            args += ['--published-manifest-sha256', digest]
        lock = {'specifier': 'github:example/lcore#' + commit}
        with patch.object(sys, 'argv', args), patch.dict(os.environ, self.env), \
             patch.object(reporter.legacy, 'generate_lock', side_effect=lock_effect, return_value=lock) as resolve, \
             patch.object(reporter, 'collect', return_value={'baseCommit': 'b' * 40}), redirect_stdout(io.StringIO()):
            reporter.main()
        return resolve

    def test_same_job_handoff_runs_exactly_three_packs_end_to_end(self):
        output = self.root / 'github-output'
        assemble_args = ['package_runtime.py', '--inputs', str(self.inputs), '--output', str(self.out), '--defer-npm-pack']
        publish_args = ['publish_artifacts.py', '--package', str(self.out), '--remote', str(self.remote)]
        with patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls:
            with patch.object(sys, 'argv', assemble_args), redirect_stdout(io.StringIO()):
                package.main()
            with patch.object(sys, 'argv', publish_args), \
                 patch.dict(os.environ, {**self.env, 'GITHUB_OUTPUT': str(output)}), redirect_stdout(io.StringIO()):
                publisher.main()
            values = dict(line.split('=', 1) for line in output.read_text().splitlines())
            self.report_main(digest=values['manifest-sha256'], commit=values['commit'])
        self.assertEqual(len(pack_calls(calls)), 3)
        envelope = json.loads((self.root / 'report/report.json').read_text())
        self.assertEqual(envelope['artifactCommit'], values['commit'])
        self.assertEqual(envelope['sourceCommit'], fixtures.SOURCE)
        self.assertEqual(envelope['runAttempt'], 2)

    def test_workflow_assembly_runs_zero_packs_and_keeps_identical_bytes(self):
        args = ['package_runtime.py', '--inputs', str(self.inputs), '--output', str(self.out), '--defer-npm-pack']
        with patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls, \
             patch.object(sys, 'argv', args), redirect_stdout(io.StringIO()):
            package.main()
        self.assertEqual(len(pack_calls(calls)), 0)
        normal = self.root / 'normal'
        package.assemble(self.inputs, normal)
        self.assertEqual(tree(self.out), tree(normal))

    def test_standalone_validators_still_pack_each_runtime_and_root(self):
        self.assemble()
        with patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls:
            package.validate(self.out)
        self.assertEqual(len(pack_calls(calls)), 3)
        for runtime in package.RUNTIMES:
            with self.subTest(runtime=runtime), \
                 patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls:
                package.runtime_module(runtime).validate(self.out / package.ARTIFACT_DIRS[runtime])
            self.assertEqual(len(pack_calls(calls)), 1)

    def test_standalone_assembly_and_verify_cli_do_not_defer_by_default(self):
        args = ['package_runtime.py', '--inputs', str(self.inputs), '--output', str(self.out)]
        for extra, count in (([], 8), (['--verify-only'], 3)):
            with self.subTest(extra=extra), patch.object(sys, 'argv', args + extra), \
                 patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls, \
                 redirect_stdout(io.StringIO()):
                package.main()
            self.assertEqual(len(pack_calls(calls)), count)

    def test_publication_packs_only_the_private_snapshot_and_commits_identical_tree(self):
        self.assemble()
        before = tree(self.out)
        with patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls:
            commit = publisher.publish(self.out, str(self.remote))
        packs = pack_calls(calls)
        self.assertEqual(len(packs), 3)
        for call in packs:
            self.assertFalse(Path(call.kwargs['cwd']).is_relative_to(self.out))
        paths = git('--git-dir=' + str(self.remote), 'ls-tree', '-r', '--name-only', commit, cwd=self.root).splitlines()
        self.assertEqual(set(paths), set(before))
        for path in paths:
            data = subprocess.check_output(['git', '--git-dir=' + str(self.remote), 'show', commit + ':' + path])
            self.assertEqual(data, before[path], path)
        self.assertEqual(tree(self.out), before)

    def test_publish_output_binds_commit_and_manifest_without_changing_stdout(self):
        self.assemble()
        output = self.root / 'github-output'
        summary = self.root / 'summary.md'
        args = ['publish_artifacts.py', '--package', str(self.out), '--remote', str(self.remote)]
        stdout = io.StringIO()
        with patch.object(sys, 'argv', args), \
             patch.dict(os.environ, {**self.env, 'GITHUB_OUTPUT': str(output), 'GITHUB_STEP_SUMMARY': str(summary)}), \
             redirect_stdout(stdout):
            publisher.main()
        values = dict(line.split('=', 1) for line in output.read_text().splitlines())
        self.assertEqual(set(values), {'commit', 'manifest-sha256'})
        self.assertEqual(stdout.getvalue(), values['commit'] + '\n')
        committed = subprocess.check_output(['git', '--git-dir=' + str(self.remote), 'show', values['commit'] + ':manifest.json'])
        self.assertEqual(values['manifest-sha256'], hashlib.sha256(committed).hexdigest())
        self.assertIn(values['commit'], summary.read_text())

    def test_early_summary_shows_escaped_source_title_without_another_install(self):
        self.assemble()
        with self.assertRaisesRegex(ValueError, 'Invalid source commit'):
            publisher.source_commit_subject('--upload-pack=unexpected', self.root)
        source_repo = self.root / 'source'
        source_repo.mkdir()
        git('init', '-q', cwd=source_repo)
        git('config', 'user.name', 'Test', cwd=source_repo)
        git('config', 'user.email', 'test@example.com', cwd=source_repo)
        sentinel = self.root / 'unexpected'
        title = f'Improve <img src=x> & `GPU` $(touch {sentinel})'
        git('commit', '--allow-empty', '-q', '-m', title, cwd=source_repo)
        source = git('rev-parse', 'HEAD', cwd=source_repo)
        manifest_path = self.out / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['sourceCommit'] = source
        manifest_path.write_text(json.dumps(manifest))
        summary = self.root / 'source-summary.md'
        args = ['publish_artifacts.py', '--package', str(self.out), '--remote', str(self.remote),
                '--source-repo', str(source_repo)]
        with patch.object(sys, 'argv', args), \
             patch.dict(os.environ, {**self.env, 'GITHUB_STEP_SUMMARY': str(summary)}), \
             patch.object(publisher, 'publish', return_value='d' * 40), redirect_stdout(io.StringIO()):
            publisher.main()
        text = summary.read_text()
        self.assertIn(f'**Source commit:** `{source}`', text)
        self.assertIn('**Source commit title:** <code>Improve &lt;img src=x&gt; &amp; `GPU` $(touch ', text)
        self.assertNotIn('<img', text)
        self.assertFalse(sentinel.exists())
        self.assertEqual(text.count('npm install github:example/lcore#'), 1)
        self.assertIn('`' + 'd' * 40 + '`', text)

    def test_unavailable_source_title_preserves_early_publication_receipt(self):
        self.assemble()
        summary = self.root / 'missing-title-summary.md'
        args = ['publish_artifacts.py', '--package', str(self.out), '--remote', str(self.remote),
                '--source-repo', str(self.root / 'missing-repo')]
        with patch.object(sys, 'argv', args), \
             patch.dict(os.environ, {**self.env, 'GITHUB_STEP_SUMMARY': str(summary)}), \
             patch.object(publisher, 'publish', return_value='d' * 40), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            publisher.main()
        text = summary.read_text()
        self.assertIn('`' + 'd' * 40 + '`', text)
        self.assertEqual(text.count('npm install github:example/lcore#'), 1)

    def test_reporting_with_publisher_digest_avoids_packing_but_matches_full_report(self):
        self.assemble()
        digest = package.identity(self.out / 'manifest.json')['sha256']
        with patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls:
            resolve = self.report_main(digest=digest)
        self.assertEqual(len(pack_calls(calls)), 0)
        resolve.assert_called_once_with('example/lcore', 'd' * 40, '0.1.0')
        fast = tree(self.root / 'report')
        with patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls:
            self.report_main()
        # Standalone reporting packs once before lock resolution, not twice.
        self.assertEqual(len(pack_calls(calls)), 3)
        self.assertEqual(tree(self.root / 'report'), fast)

    def test_direct_metadata_call_still_performs_full_validation_by_default(self):
        self.assemble()
        with patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls:
            reporter.metadata(self.out, 'example/lcore', 'd' * 40, {}, {'baseCommit': 'b' * 40})
        self.assertEqual(len(pack_calls(calls)), 3)

    def test_wrong_or_empty_digest_fails_before_lock_resolution(self):
        self.assemble()
        for digest in ('', 'not-a-hash', 'A' * 64, '0' * 64):
            with self.subTest(digest=digest):
                resolutions = []
                with self.assertRaisesRegex(ValueError, 'manifest'):
                    self.report_main(digest=digest, lock_effect=lambda *args: resolutions.append(args))
                self.assertEqual(resolutions, [])

    def test_payload_tampering_cannot_reuse_a_matching_manifest_digest(self):
        self.assemble()
        digest = package.identity(self.out / 'manifest.json')['sha256']
        target = self.out / 'llama-cpp-browser-core/profiles/cpu-wasm32/browser/core.mjs'
        target.write_bytes(b'X' * target.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'hash/size'):
            reporter.validate_for_report(self.out, digest)

    def test_new_consistent_manifest_cannot_reuse_old_publisher_digest(self):
        self.assemble()
        digest = package.identity(self.out / 'manifest.json')['sha256']
        (self.out / 'README.md').write_text('different but internally consistent package')
        fixtures.write_manifest(self.out, json.loads((self.out / 'manifest.json').read_text()))
        with self.assertRaisesRegex(ValueError, 'published manifest'):
            reporter.validate_for_report(self.out, digest)

    def test_modification_during_lock_resolution_is_detected_with_or_without_receipt(self):
        for receipt in (True, False):
            for rewrite_manifest in (True, False):
                with self.subTest(receipt=receipt, rewrite_manifest=rewrite_manifest):
                    self.assemble()
                    shutil.rmtree(self.root / 'report', ignore_errors=True)
                    digest = package.identity(self.out / 'manifest.json')['sha256'] if receipt else None
                    def change(*args):
                        (self.out / 'README.md').write_text('changed during network access')
                        if rewrite_manifest:
                            fixtures.write_manifest(self.out, json.loads((self.out / 'manifest.json').read_text()))
                        return {}
                    with self.assertRaisesRegex(ValueError, 'published manifest|hash/size'):
                        self.report_main(digest=digest, lock_effect=change)
                    self.assertFalse((self.root / 'report/report.json').exists())

    def test_deferred_validation_keeps_semantic_guards_even_with_rehashed_payloads(self):
        cases = ('dirty', 'source', 'variant', 'license', 'wasm', 'schema', 'hooks', 'extra', 'symlink')
        for case in cases:
            with self.subTest(case=case):
                self.assemble()
                image = self.out / 'stable-diffusion-cpp-browser-core'
                manifest = json.loads((image / 'manifest.json').read_text())
                profile = next(iter(manifest['profiles']))
                item = manifest['profiles'][profile]['variants']['browser']
                target = image / f'profiles/{profile}/browser/core.wasm'
                if case == 'dirty': item['sourceDirty'] = True
                elif case == 'source': item['sourceCommit'] = 'f' * 40
                elif case == 'variant': del manifest['profiles'][profile]['variants']['test']
                elif case == 'license': (image / 'licenses/ggml/LICENSE').unlink()
                elif case == 'wasm': target.write_bytes(b'not wasm')
                elif case == 'schema': manifest['schemaSha256'] = '0' * 64
                elif case == 'hooks':
                    pkg = json.loads((image / 'package.json').read_text()); pkg['scripts'] = {'prepare': 'false'}
                    (image / 'package.json').write_text(json.dumps(pkg))
                elif case == 'extra': (image / 'unlisted').write_text('extra')
                elif case == 'symlink':
                    data = self.root / 'external.wasm'; data.write_bytes(target.read_bytes())
                    target.unlink(); target.symlink_to(data)
                if case != 'extra': fixtures.write_manifest(image, manifest)
                fixtures.write_manifest(self.out, json.loads((self.out / 'manifest.json').read_text()))
                digest = package.identity(self.out / 'manifest.json')['sha256']
                with self.assertRaises(ValueError): reporter.validate_for_report(self.out, digest)

    def test_each_runtime_still_rejects_invalid_wasm_header_without_read_bytes(self):
        for runtime in package.RUNTIMES:
            with self.subTest(runtime=runtime):
                self.assemble()
                inner = self.out / package.ARTIFACT_DIRS[runtime]
                target = next(inner.glob('profiles/*/browser/core.wasm'))
                target.write_bytes(b'not wasm')
                fixtures.write_manifest(inner, json.loads((inner / 'manifest.json').read_text()))
                validator = package.runtime_module(runtime)
                with patch.object(Path, 'read_bytes', side_effect=AssertionError('Unbounded payload read')), \
                     self.assertRaisesRegex(ValueError, 'WebAssembly'):
                    validator.validate(inner, check_npm_pack=False)

    def test_wasm_header_check_never_loads_entire_module_into_memory(self):
        self.assemble()
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('Unbounded payload read')):
            package.validate(self.out, check_npm_pack=False)

    def test_publisher_rejects_npm_omissions_for_each_package_before_remote_access(self):
        for runtime in (*package.RUNTIMES, ''):
            with self.subTest(runtime=runtime):
                self.assemble()
                directory = self.out / package.ARTIFACT_DIRS[runtime] if runtime else self.out
                pkg = json.loads((directory / 'package.json').read_text()); pkg['files'] = ['LICENSE']
                (directory / 'package.json').write_text(json.dumps(pkg))
                if runtime:
                    fixtures.write_manifest(directory, json.loads((directory / 'manifest.json').read_text()))
                fixtures.write_manifest(self.out, json.loads((self.out / 'manifest.json').read_text()))
                # Payload checks alone pass, but cannot authorize publication.
                package.validate(self.out, check_npm_pack=False)
                with patch.object(publisher, 'run', wraps=publisher.run) as commands, \
                     self.assertRaisesRegex(ValueError, 'npm'):
                    publisher.publish(self.out, str(self.remote))
                self.assertFalse(any(c.args[1] in ('ls-remote', 'fetch', 'push') for c in commands.call_args_list))
                self.assertEqual(git('--git-dir=' + str(self.remote), 'for-each-ref', cwd=self.root), '')

    def test_copy_corruption_is_detected_on_the_snapshot_before_any_push(self):
        self.assemble()
        copytree = shutil.copytree
        def corrupt(source, destination, *args, **kwargs):
            result = copytree(source, destination, *args, **kwargs)
            if Path(source) == self.out / 'llama-cpp-browser-core':
                (Path(destination) / 'profiles/cpu-wasm32/browser/core.wasm').write_bytes(b'not wasm')
            return result
        with patch.object(shutil, 'copytree', side_effect=corrupt), \
             self.assertRaisesRegex(ValueError, 'hash/size'):
            publisher.publish(self.out, str(self.remote))
        self.assertEqual(git('--git-dir=' + str(self.remote), 'for-each-ref', cwd=self.root), '')

    def test_wrong_publication_digest_fails_without_changing_remote(self):
        self.assemble()
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            publisher.publish(self.out, str(self.remote), expected_manifest_sha256='0' * 64)
        self.assertEqual(git('--git-dir=' + str(self.remote), 'for-each-ref', cwd=self.root), '')

    def test_failed_publication_emits_no_commit_or_validation_receipt(self):
        self.assemble()
        (self.out / 'README.md').write_text('corrupt')
        output = self.root / 'github-output'
        args = ['publish_artifacts.py', '--package', str(self.out), '--remote', str(self.remote)]
        with patch.object(sys, 'argv', args), patch.dict(os.environ, {'GITHUB_OUTPUT': str(output)}), \
             self.assertRaisesRegex(ValueError, 'hash/size'):
            publisher.main()
        self.assertFalse(output.exists())

    def test_two_publications_keep_append_only_history_and_old_cold_npm_install(self):
        self.assemble()
        first = publisher.publish(self.out, str(self.remote))
        runtime = self.inputs / 'llama-cpp'
        target = runtime / 'profiles/cpu-wasm32/browser/core.mjs'
        target.write_text('export default () => "updated-fixture";')
        fixtures.write_manifest(runtime, json.loads((runtime / 'manifest.json').read_text()))
        self.assemble()
        second = publisher.publish(self.out, str(self.remote))
        self.assertEqual(git('--git-dir=' + str(self.remote), 'rev-parse', second + '^', cwd=self.root), first)
        consumer = self.root / 'consumer'; consumer.mkdir()
        (consumer / 'package.json').write_text('{"private":true}')
        subprocess.run(['npm', 'install', '--ignore-scripts', '--no-audit', '--no-fund',
                        '--cache', str(self.root / 'install-cache'), f'git+file://{self.remote}#{first}'],
                       cwd=consumer, check=True, capture_output=True)
        shutil.rmtree(consumer / 'node_modules')
        subprocess.run(['npm', 'ci', '--ignore-scripts', '--no-audit', '--no-fund',
                        '--cache', str(self.root / 'empty-ci-cache')], cwd=consumer, check=True, capture_output=True)
        subprocess.run(['node', '--input-type=module', '-e', '''
          const legacy = await import('llama-cpp-browser-core/profiles/cpu-wasm32/browser/core.mjs');
          const named = await import('llama-cpp-browser-core/llama-cpp/profiles/cpu-wasm32/browser/core.mjs');
          const image = await import('llama-cpp-browser-core/stable-diffusion-cpp/profiles/webgpu-wasm32-jspi/browser/core.mjs');
          if (legacy.default !== named.default || legacy.default() !== 'fixture' || image.default() !== 'fixture')
            throw Error('Old artifact content or export mapping changed');
        '''], cwd=consumer, check=True, capture_output=True)


class PublicationWorkflow(unittest.TestCase):
    def test_workflow_defers_only_assembly_and_uses_paired_publisher_outputs(self):
        workflow = (ROOT / '.github/workflows/build.yml').read_text()
        publish = workflow.split('\n  publish:\n', 1)[1]
        self.assertIn('scripts/package_runtime.py --defer-npm-pack', publish)
        self.assertEqual(publish.count('--defer-npm-pack'), 1)
        self.assertIn('ARTIFACT_COMMIT: ${{ steps.publish.outputs.commit }}', publish)
        self.assertIn('PUBLISHED_MANIFEST_SHA256: ${{ steps.publish.outputs.manifest-sha256 }}', publish)
        self.assertIn('--published-manifest-sha256 "$PUBLISHED_MANIFEST_SHA256"', publish)
        self.assertLess(publish.index('Verify package source before publication'), publish.index('scripts/publish_artifacts.py'))
        self.assertLess(publish.index('scripts/publish_artifacts.py'), publish.index('scripts/consumer_metadata.py'))
        self.assertIn("failure() && steps.publish.outputs.commit != ''", publish)
        self.assertEqual(workflow.count('submodules: false'), 7)
        self.assertEqual(workflow.count('submodule update --init --recursive --depth=1'), 6)
        self.assertIn('submodules: false', publish)
        self.assertIn('persist-credentials: false', publish)

    def test_selective_checkout_uses_the_gitlink_not_remote_tip_or_nested_submodules(self):
        publish = (ROOT / '.github/workflows/build.yml').read_text().split('\n  publish:\n', 1)[1]
        command = re.search(r'(git submodule update[^\n]+)', publish)[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / 'upstream'; upstream.mkdir()
            source = root / 'source'; source.mkdir()
            for path in (upstream, source):
                git('init', '-q', cwd=path)
                git('config', 'user.name', 'Fixture', cwd=path)
                git('config', 'user.email', 'fixture@example.invalid', cwd=path)
            (upstream / 'payload').write_text('pinned')
            (upstream / '.gitmodules').write_text('[submodule "unused"]\n path = nested\n url = file:///not-a-repository\n')
            git('add', '.', cwd=upstream)
            git('update-index', '--add', '--cacheinfo', '160000,' + 'a' * 40 + ',nested', cwd=upstream)
            git('commit', '-qm', 'pinned fixture', cwd=upstream)
            pinned = git('rev-parse', 'HEAD', cwd=upstream)
            (upstream / 'payload').write_text('newer remote tip')
            git('commit', '-qam', 'tip fixture', cwd=upstream)
            modules = []
            for path in ('llama-cpp/vendor/llama.cpp', 'stable-diffusion-cpp/vendor/stable-diffusion.cpp',
                         'stable-diffusion-cpp/vendor/ggml-webgpu-source'):
                url = upstream.as_uri() if path.startswith('llama-cpp/') else 'file:///not-a-repository'
                modules.append(f'[submodule "{path}"]\n path = {path}\n url = {url}\n')
                git('update-index', '--add', '--cacheinfo', f'160000,{pinned},{path}', cwd=source)
            (source / '.gitmodules').write_text(''.join(modules))
            git('add', '.gitmodules', cwd=source)
            git('commit', '-qm', 'fixture source gitlinks', cwd=source)
            env = {**os.environ, 'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'protocol.file.allow',
                   'GIT_CONFIG_VALUE_0': 'always', 'GIT_TERMINAL_PROMPT': '0'}
            subprocess.run(['bash', '-e', '-c', command], cwd=source, env=env, check=True, capture_output=True)
            checkout = source / 'llama-cpp/vendor/llama.cpp'
            self.assertEqual(git('rev-parse', 'HEAD', cwd=checkout), pinned)
            self.assertEqual((checkout / 'payload').read_text(), 'pinned')
            self.assertFalse((checkout / 'nested/.git').exists())
            self.assertFalse((source / 'stable-diffusion-cpp/vendor/ggml-webgpu-source/.git').exists())


if __name__ == '__main__': unittest.main()
