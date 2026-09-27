"""Fault injection across validation boundaries; real npm/Git, no remote network.

These are corruption/race regression tests, not a claim to sandbox a compromised
runner or to exercise compiled Wasm. File mutation is injected at exact handoffs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_multi_runtime as fixtures
import test_publish_optimization as helpers

package = helpers.package
publisher = helpers.publisher
reporter = helpers.reporter


class AdversarialPublication(unittest.TestCase):
    def setUp(self):
        self.fixture = helpers.PublicationOptimization()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.out, self.remote = self.fixture.root, self.fixture.out, self.fixture.remote
        self.fixture.assemble()

    def assert_no_remote(self, calls=None):
        self.assertEqual(helpers.git('--git-dir=' + str(self.remote), 'for-each-ref', cwd=self.root), '')
        if calls is not None:
            self.assertFalse(any(c.args[1] in ('ls-remote', 'fetch', 'push') for c in calls.call_args_list))

    def rewrite_root_manifest(self):
        fixtures.write_manifest(self.out, json.loads((self.out / 'manifest.json').read_text()))

    def test_git_global_exclusion_cannot_publish_a_missing_wasm(self):
        exclude = self.root / 'git-excludes'; exclude.write_text('*.wasm\n')
        env = {'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.excludesFile',
               'GIT_CONFIG_VALUE_0': str(exclude)}
        with patch.dict(os.environ, env), patch.object(publisher, 'run', wraps=publisher.run) as calls, \
             self.assertRaisesRegex(ValueError, 'Git.*tree'):
            publisher.publish(self.out, str(self.remote))
        self.assert_no_remote(calls)

    def test_git_line_ending_conversion_cannot_publish_different_bytes(self):
        (self.out / 'README.md').write_bytes(b'line one\r\nline two\r\n')
        self.rewrite_root_manifest()
        env = {'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.autocrlf', 'GIT_CONFIG_VALUE_0': 'true'}
        with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, 'Git.*payload'):
            publisher.publish(self.out, str(self.remote))
        self.assert_no_remote()

    def test_git_clean_filter_cannot_publish_different_same_size_bytes(self):
        attributes = self.root / 'git-attributes'; attributes.write_text('README.md filter=fixture\n')
        env = {'GIT_CONFIG_COUNT': '2', 'GIT_CONFIG_KEY_0': 'core.attributesFile',
               'GIT_CONFIG_VALUE_0': str(attributes), 'GIT_CONFIG_KEY_1': 'filter.fixture.clean',
               'GIT_CONFIG_VALUE_1': 'tr a-z A-Z'}
        with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, 'Git.*payload'):
            publisher.publish(self.out, str(self.remote))
        self.assert_no_remote()

    def test_mutation_after_last_pack_is_rejected_before_remote_or_receipt(self):
        output = self.root / 'github-output'
        args = ['publish_artifacts.py', '--package', str(self.out), '--remote', str(self.remote)]
        real_output = subprocess.check_output
        def change(command, *args, **kwargs):
            result = real_output(command, *args, **kwargs)
            if command[:3] == ['npm', 'pack', '--dry-run'] and (Path(kwargs['cwd']) / 'llama-cpp-browser-core').is_dir():
                target = Path(kwargs['cwd']) / 'README.md'
                target.write_bytes(b'X' * target.stat().st_size)
            return result
        with patch.object(sys, 'argv', args), patch.dict(os.environ, {'GITHUB_OUTPUT': str(output)}), \
             patch.object(subprocess, 'check_output', side_effect=change), \
             patch.object(publisher, 'run', wraps=publisher.run) as calls, \
             self.assertRaisesRegex(ValueError, 'Git.*payload'):
            publisher.main()
        self.assertFalse(output.exists())
        self.assert_no_remote(calls)

    def test_root_manifest_mutation_after_packing_is_rejected(self):
        real = publisher.run
        def change(*args, **kwargs):
            if args[1:3] == ('add', '--all'):
                path = Path(kwargs['cwd']) / 'manifest.json'
                m = json.loads(path.read_text()); m['sourceCommit'] = 'f' * 40
                path.write_text(json.dumps(m))
            return real(*args, **kwargs)
        with patch.object(publisher, 'run', side_effect=change), \
             self.assertRaisesRegex(ValueError, 'Git.*payload'):
            publisher.publish(self.out, str(self.remote))
        self.assert_no_remote()

    def test_staged_extra_file_and_nonregular_modes_are_rejected(self):
        for case in ('extra', 'symlink', 'gitlink'):
            with self.subTest(case=case):
                real = publisher.run
                def change(*args, **kwargs):
                    result = real(*args, **kwargs)
                    if args[1:3] == ('add', '--all'):
                        work = Path(kwargs['cwd'])
                        if case == 'extra':
                            (work / 'unvalidated.txt').write_text('not part of the manifest')
                            real('git', 'add', 'unvalidated.txt', cwd=work)
                        else:
                            blob = real('git', 'rev-parse', ':README.md', cwd=work).stdout.strip()
                            mode = '120000' if case == 'symlink' else '160000'
                            real('git', 'update-index', '--cacheinfo', mode + ',' + blob + ',README.md', cwd=work)
                    return result
                with patch.object(publisher, 'run', side_effect=change), \
                     self.assertRaisesRegex(ValueError, 'Git.*tree'):
                    publisher.publish(self.out, str(self.remote))
                self.assert_no_remote()

    def test_manifest_and_payload_may_contain_unusual_filenames(self):
        # ls-tree must use NUL framing, not line splitting or quoted paths.
        name = 'unicode-資料 space\ttab\nline.txt'
        (self.out / 'llama-cpp-browser-core/licenses' / name).write_bytes(b'notice\x00binary\xff')
        inner = self.out / 'llama-cpp-browser-core'
        fixtures.write_manifest(inner, json.loads((inner / 'manifest.json').read_text()))
        self.rewrite_root_manifest()
        commit = publisher.publish(self.out, str(self.remote))
        data = subprocess.check_output(['git', '--git-dir=' + str(self.remote), 'show', commit + ':llama-cpp-browser-core/licenses/' + name])
        self.assertEqual(data, b'notice\x00binary\xff')

    def test_mutation_during_report_construction_never_writes_report(self):
        for receipt in (True, False):
            with self.subTest(receipt=receipt):
                self.fixture.assemble()
                digest = package.identity(self.out / 'manifest.json')['sha256'] if receipt else None
                original = reporter.legacy.metadata
                def change(*args, **kwargs):
                    path = self.out / 'llama-cpp-browser-core/manifest.json'
                    m = json.loads(path.read_text()); m['sourceCommit'] = 'e' * 40
                    path.write_text(json.dumps(m))
                    return original(*args, **kwargs)
                with patch.object(reporter.legacy, 'metadata', side_effect=change), \
                     self.assertRaisesRegex(ValueError, 'hash/size|manifest'):
                    self.fixture.report_main(digest=digest)
                self.assertFalse((self.root / 'report/report.json').exists())

    def test_temporarily_replaced_manifest_cannot_poison_in_memory_metadata(self):
        digest = package.identity(self.out / 'manifest.json')['sha256']
        original = reporter.legacy.metadata
        path = self.out / 'llama-cpp-browser-core/manifest.json'; saved = path.read_bytes()
        def change(*args, **kwargs):
            m = json.loads(saved); m['sourceCommit'] = 'e' * 40
            path.write_text(json.dumps(m))
            try: return original(*args, **kwargs)
            finally: path.write_bytes(saved)
        with patch.object(reporter.legacy, 'metadata', side_effect=change):
            self.fixture.report_main(digest=digest)
        data = json.loads((self.root / 'report/report.json').read_text())
        self.assertEqual(data['sourceCommit'], fixtures.SOURCE)

    def test_report_rejects_manifest_change_between_validation_and_snapshot_read(self):
        original = reporter.validate_for_report
        def change(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.out / 'llama-cpp-browser-core/manifest.json'
            m = json.loads(path.read_text()); m['sourceCommit'] = 'e' * 40
            path.write_text(json.dumps(m))
            return result
        digest = package.identity(self.out / 'manifest.json')['sha256']
        with patch.object(reporter, 'validate_for_report', side_effect=change), \
             self.assertRaisesRegex(ValueError, 'manifest|snapshot'):
            reporter.metadata(self.out, 'example/lcore', 'd' * 40, {}, {'baseCommit': 'b' * 40},
                              published_manifest_sha256=digest)

    def test_full_and_deferred_validators_reject_special_tree_entries(self):
        for runtime in ('', *package.RUNTIMES):
            for kind in ('directory-link', 'dangling-link', 'fifo'):
                for packing in (True, False):
                    with self.subTest(runtime=runtime, kind=kind, packing=packing):
                        directory = self.out / package.ARTIFACT_DIRS[runtime] if runtime else self.out
                        target = directory / 'unexpected'
                        if kind == 'fifo': os.mkfifo(target)
                        else:
                            outside = self.root / 'outside'
                            outside.mkdir(exist_ok=True)
                            target.symlink_to(outside if kind == 'directory-link' else self.root / 'absent',
                                              target_is_directory=kind == 'directory-link')
                        validator = package if not runtime else package.runtime_module(runtime)
                        try:
                            with patch.object(subprocess, 'check_output', side_effect=AssertionError('Must fail before npm')), \
                                 self.assertRaisesRegex(ValueError, 'linked|symlink|regular|special'):
                                validator.validate(directory, check_npm_pack=packing)
                        finally: target.unlink()

    def test_manifest_symlink_is_rejected_before_reading_in_all_validators(self):
        for runtime in ('', *package.RUNTIMES):
            with self.subTest(runtime=runtime):
                directory = self.out / package.ARTIFACT_DIRS[runtime] if runtime else self.out
                path = directory / 'manifest.json'; saved = path.read_bytes()
                outside = self.root / 'outside-manifest'; outside.write_bytes(saved)
                path.unlink(); path.symlink_to(outside)
                validator = package if not runtime else package.runtime_module(runtime)
                try:
                    with patch.object(Path, 'read_text', side_effect=AssertionError('Must reject the link before read')), \
                         self.assertRaisesRegex(ValueError, 'linked|symlink|regular|special'):
                        validator.validate(directory, check_npm_pack=False)
                finally: path.unlink(); path.write_bytes(saved)

    def test_no_validator_accepts_peer_or_bundled_dependencies(self):
        fields = {'peerDependencies': {'fixture-peer': '1.0.0'},
                  'peerDependenciesMeta': {'fixture-peer': {'optional': True}},
                  'bundleDependencies': ['fixture-peer'], 'bundledDependencies': ['fixture-peer']}
        for runtime in ('', *package.RUNTIMES):
            for key, value in fields.items():
                with self.subTest(runtime=runtime, field=key):
                    self.fixture.assemble()
                    directory = self.out / package.ARTIFACT_DIRS[runtime] if runtime else self.out
                    path = directory / 'package.json'; data = json.loads(path.read_text()); data[key] = value
                    path.write_text(json.dumps(data))
                    if runtime: fixtures.write_manifest(directory, json.loads((directory / 'manifest.json').read_text()))
                    self.rewrite_root_manifest()
                    with self.assertRaises(ValueError): package.validate(self.out, check_npm_pack=False)

    def test_failed_push_does_not_emit_receipt_or_retry_permission_failure(self):
        hook = self.remote / 'hooks/pre-receive'
        hook.write_text('#!/bin/sh\necho denied-by-test >&2\nexit 1\n'); hook.chmod(0o755)
        output = self.root / 'github-output'
        args = ['publish_artifacts.py', '--package', str(self.out), '--remote', str(self.remote)]
        with patch.object(sys, 'argv', args), patch.dict(os.environ, {'GITHUB_OUTPUT': str(output)}), \
             patch.object(publisher.time, 'sleep', side_effect=AssertionError('Do not retry permission errors')), \
             self.assertRaisesRegex(RuntimeError, 'denied-by-test'):
            publisher.main()
        self.assertFalse(output.exists())
        self.assert_no_remote()

    def test_later_worktree_changes_cannot_change_the_verified_git_tree(self):
        before = helpers.tree(self.out)
        verify = publisher.verify_git_tree
        def change(work, *args, **kwargs):
            verify(work, *args, **kwargs)
            (work / 'README.md').write_text('changed after immutable tree verification')
            (work / 'manifest.json').write_text('{}')
        with patch.object(publisher, 'verify_git_tree', side_effect=change):
            commit = publisher.publish(self.out, str(self.remote))
        for name in ('README.md', 'manifest.json'):
            blob = subprocess.check_output(['git', '--git-dir=' + str(self.remote), 'show', commit + ':' + name])
            self.assertEqual(blob, before[name])

    def test_streaming_git_verification_accepts_empty_and_multichunk_blobs(self):
        # Synthetic data beyond the verifier's one-MiB read window, not Wasm.
        notices = self.out / 'llama-cpp-browser-core/licenses'
        (notices / 'empty.txt').write_bytes(b'')
        (notices / 'large.txt').write_bytes(b'x' * (2 * 1024 * 1024 + 7))
        inner = self.out / 'llama-cpp-browser-core'
        fixtures.write_manifest(inner, json.loads((inner / 'manifest.json').read_text()))
        self.rewrite_root_manifest()
        publisher.publish(self.out, str(self.remote))

    def test_standalone_metadata_hashes_the_exact_bytes_it_parses(self):
        inner = self.out / 'llama-cpp-browser-core'
        original = Path.read_bytes; reads = []
        def read(path):
            if path == inner / 'manifest.json': reads.append(path)
            return original(path)
        with patch.object(Path, 'read_bytes', read):
            result = reporter.legacy.metadata(inner, 'example/lcore', 'd' * 40, {}, {'baseCommit': 'b' * 40})
        self.assertEqual(len(reads), 1)
        self.assertEqual(result['retrieval']['manifest'],
                         {'path': 'manifest.json', **package.identity(inner / 'manifest.json')})

    def test_npm_failure_never_reaches_git_remote(self):
        real = subprocess.check_output
        def reject(command, *args, **kwargs):
            if command[:3] == ['npm', 'pack', '--dry-run']:
                raise subprocess.CalledProcessError(1, command, stderr='synthetic packing failure')
            return real(command, *args, **kwargs)
        with patch.object(subprocess, 'check_output', side_effect=reject), \
             patch.object(publisher, 'run', wraps=publisher.run) as calls, \
             self.assertRaises(subprocess.CalledProcessError):
            publisher.publish(self.out, str(self.remote))
        self.assert_no_remote(calls)

    def test_exhausted_push_conflicts_do_not_repack_or_force_push(self):
        real = publisher.run
        def conflict(*args, **kwargs):
            if args[1] == 'push':
                self.assertNotIn('--force', args)
                return subprocess.CompletedProcess(args, 1, '', 'non-fast-forward')
            return real(*args, **kwargs)
        with patch.object(publisher, 'run', side_effect=conflict), patch.object(publisher.time, 'sleep'), \
             patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls, \
             self.assertRaisesRegex(RuntimeError, 'conflicted repeatedly'):
            publisher.publish(self.out, str(self.remote), attempts=2)
        self.assertEqual(len(helpers.pack_calls(calls)), 3)
        self.assert_no_remote()

    def test_true_push_conflict_retries_with_the_new_parent_without_repacking(self):
        real = publisher.run; injected = []
        def compete(*args, **kwargs):
            if args[1] == 'push' and not injected:
                candidate = args[-1].split(':', 1)[0]
                tree = real('git', 'rev-parse', candidate + '^{tree}', cwd=kwargs['cwd']).stdout.strip()
                rival = real('git', 'commit-tree', tree, '-m', 'competing fixture publication', cwd=kwargs['cwd']).stdout.strip()
                real('git', 'push', 'origin', rival + ':refs/heads/artifacts', cwd=kwargs['cwd'])
                injected.append(rival)
            return real(*args, **kwargs)
        with patch.object(publisher, 'run', side_effect=compete), patch.object(publisher.time, 'sleep'), \
             patch.object(subprocess, 'check_output', wraps=subprocess.check_output) as calls:
            commit = publisher.publish(self.out, str(self.remote))
        self.assertEqual(len(helpers.pack_calls(calls)), 3)
        self.assertEqual(helpers.git('--git-dir=' + str(self.remote), 'rev-parse', commit + '^', cwd=self.root), injected[0])


if __name__ == '__main__': unittest.main()
