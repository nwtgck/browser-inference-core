"""Validate local npm receipts and execute the workflow boundary with a stand-in.

The shell test records the exact npm invocation without contacting a registry or
GitHub. Complete package fixtures exercise report rejection, not native builds.
"""
from contextlib import redirect_stderr
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.packaging import assemble
from build_tools.packaging.layout import PACKAGE_NAME
from build_tools.project import REPOSITORY_ROOT
from build_tools.publication import consumer, npm_lock
from tests.support.runtime_package import RuntimePackageFixture
from tests.support.workflow import jobs, shell_body, step_name


COMMIT = 'c' * 40
REPOSITORY = 'example/core'
SPECIFIER = f'github:{REPOSITORY}#{COMMIT}'
ENTRY = {
    'version': '0.1.0',
    'resolved': f'git+ssh://git@github.com/{REPOSITORY}.git#{COMMIT}',
    'integrity': 'sha512-fixture-not-a-real-package-hash',
    'license': 'MIT',
}


def fixture_lock():
    return {'lockfileVersion': 3, 'packages': {
        '': {'dependencies': {PACKAGE_NAME: SPECIFIER}},
        'node_modules/' + PACKAGE_NAME: copy.deepcopy(ENTRY),
    }}


class NpmLock(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'build/consumer-lock'

    def prepare(self):
        return npm_lock.prepare(self.directory, REPOSITORY, COMMIT, '0.1.0')

    def write_lock(self, value=None):
        (self.directory / 'package-lock.json').write_text(json.dumps(fixture_lock() if value is None else value))

    def read(self):
        # Node/npm version reporting is local. Only the two version commands
        # are substituted; all receipt parsing and file validation are real.
        def version(command, **kwargs):
            self.assertIn(command, (['node', '--version'], ['npm', '--version']))
            return 'v22.16.0\n' if command[0] == 'node' else '10.9.2\n'
        with patch.object(npm_lock.subprocess, 'check_output', side_effect=version):
            return npm_lock.read(self.directory)

    def test_prepare_creates_only_a_fresh_direct_dependency_fixture(self):
        request = self.prepare()
        self.assertEqual(request, {'repository': REPOSITORY, 'commit': COMMIT,
                                  'version': '0.1.0', 'package': PACKAGE_NAME, 'specifier': SPECIFIER})
        self.assertEqual({p.name for p in self.directory.iterdir()},
                         {'package.json', 'request.json', 'empty.npmrc', 'empty-global.npmrc'})
        self.assertEqual(json.loads((self.directory / 'package.json').read_text()), {
            'name': 'runtime-consumer-lock-fixture', 'version': '0.0.0', 'private': True,
            'dependencies': {PACKAGE_NAME: SPECIFIER},
        })
        self.assertEqual((self.directory / 'empty.npmrc').read_bytes(), b'')
        self.assertEqual((self.directory / 'empty-global.npmrc').read_bytes(), b'')

    def test_existing_or_linked_resolution_directory_is_preserved_and_rejected(self):
        self.prepare()
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        with self.assertRaisesRegex(ValueError, 'fresh directory'):
            self.prepare()
        self.assertEqual({p.name: p.read_bytes() for p in self.directory.iterdir()}, before)
        linked = self.root / 'linked'
        linked.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'fresh directory'):
            npm_lock.prepare(linked, REPOSITORY, COMMIT, '0.1.0')
        with self.assertRaisesRegex(ValueError, 'must not be linked'):
            npm_lock.read(linked)

    def test_prepare_rejects_mutable_refs_wrong_names_and_invalid_repositories(self):
        cases = [
            (REPOSITORY, 'main', '0.1.0', PACKAGE_NAME),
            (REPOSITORY, COMMIT[:12], '0.1.0', PACKAGE_NAME),
            (REPOSITORY, COMMIT.upper(), '0.1.0', PACKAGE_NAME),
            ('example/../other', COMMIT, '0.1.0', PACKAGE_NAME),
            ('https://github.com/example/core', COMMIT, '0.1.0', PACKAGE_NAME),
            (REPOSITORY, COMMIT, '', PACKAGE_NAME),
            (REPOSITORY, COMMIT, '0.1.0', 'llama-cpp-browser-core'),
        ]
        for repository, commit, version, package in cases:
            with self.subTest(repository=repository, commit=commit, package=package), self.assertRaises(ValueError):
                npm_lock.prepare(self.directory, repository, commit, version, package)
            self.assertFalse(self.directory.exists())

    def test_read_preserves_npm_integrity_and_unknown_fields_without_reconstruction(self):
        self.prepare()
        lock = fixture_lock()
        lock['packages']['node_modules/' + PACKAGE_NAME]['futureNpmField'] = {'preserved': True}
        self.write_lock(lock)
        result = self.read()
        self.assertEqual(result['packageEntry'], lock['packages']['node_modules/' + PACKAGE_NAME])
        self.assertEqual(result['specifier'], SPECIFIER)
        self.assertEqual(result['rootDependencies'], {PACKAGE_NAME: SPECIFIER})
        self.assertEqual((result['nodeVersion'], result['npmVersion']), ('v22.16.0', '10.9.2'))

    def test_absent_integrity_is_not_invented(self):
        lock = fixture_lock()
        del lock['packages']['node_modules/' + PACKAGE_NAME]['integrity']
        result = npm_lock.extract_lock(lock, SPECIFIER, COMMIT, '0.1.0')
        self.assertNotIn('integrity', result['packageEntry'])

    def test_wrong_commit_version_specifier_or_lock_schema_is_rejected(self):
        changes = [
            lambda value: value.update(lockfileVersion=2),
            lambda value: value['packages']['']['dependencies'].update({PACKAGE_NAME: 'latest'}),
            lambda value: value['packages']['node_modules/' + PACKAGE_NAME].update(version='0.2.0'),
            lambda value: value['packages']['node_modules/' + PACKAGE_NAME].update(
                resolved=f'git+https://github.com/{REPOSITORY}.git#' + 'd' * 40),
        ]
        for change in changes:
            value = fixture_lock()
            change(value)
            with self.subTest(lock=value), self.assertRaises(ValueError):
                npm_lock.extract_lock(value, SPECIFIER, COMMIT, '0.1.0')

    def test_transitive_dependencies_and_install_time_behavior_are_rejected(self):
        for key, value in [
            ('dependencies', {'another': '1'}), ('optionalDependencies', {'another': '1'}),
            ('peerDependencies', {'another': '1'}), ('peerDependenciesMeta', {'another': {}}),
            ('bundleDependencies', ['another']), ('bundledDependencies', ['another']),
            ('hasInstallScript', True),
        ]:
            lock = fixture_lock()
            lock['packages']['node_modules/' + PACKAGE_NAME][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'install-time'):
                npm_lock.extract_lock(lock, SPECIFIER, COMMIT, '0.1.0')
        lock = fixture_lock()
        lock['packages']['node_modules/another'] = {}
        with self.assertRaisesRegex(ValueError, 'transitive dependency'):
            npm_lock.extract_lock(lock, SPECIFIER, COMMIT, '0.1.0')

    def test_resolved_url_must_identify_exact_credential_free_github_repository(self):
        urls = [
            f'git+https://example.invalid/{REPOSITORY}.git#{COMMIT}',
            f'git+https://github.com/example/other.git#{COMMIT}',
            f'git+https://github.com/{REPOSITORY}.git?query=1#{COMMIT}',
            f'git+https://git:secret@github.com/{REPOSITORY}.git#{COMMIT}',
            f'git+https://other@github.com/{REPOSITORY}.git#{COMMIT}',
            f'git+https://github.com:443/{REPOSITORY}.git#{COMMIT}',
            f'https://github.com/{REPOSITORY}/archive/{COMMIT}.tar.gz#{COMMIT}',
            f'git+file:///tmp/{REPOSITORY}.git#{COMMIT}',
        ]
        for url in urls:
            lock = fixture_lock()
            lock['packages']['node_modules/' + PACKAGE_NAME]['resolved'] = url
            with self.subTest(url=url), self.assertRaises(ValueError):
                npm_lock.extract_lock(lock, SPECIFIER, COMMIT, '0.1.0')
        for protocol in ('git+ssh://git@', 'git+https://'):
            lock = fixture_lock()
            lock['packages']['node_modules/' + PACKAGE_NAME]['resolved'] = protocol + f'github.com/{REPOSITORY}.git#{COMMIT}'
            self.assertEqual(npm_lock.extract_lock(lock, SPECIFIER, COMMIT, '0.1.0')['packageEntry'],
                             lock['packages']['node_modules/' + PACKAGE_NAME])

    def test_changed_request_or_resolution_fixture_is_rejected(self):
        original = self.prepare()
        self.write_lock()
        for key, value in [('package', 'another'), ('repository', 'example/other'),
                           ('commit', 'd' * 40), ('specifier', 'github:example/core#main'),
                           ('version', '0.2.0')]:
            request = {**original, key: value}
            (self.directory / 'request.json').write_text(json.dumps(request))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read()
        (self.directory / 'request.json').write_text(json.dumps(original))
        package_path = self.directory / 'package.json'
        package = json.loads(package_path.read_text())
        package['scripts'] = {'preinstall': 'unexpected'}
        package_path.write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, 'fixture changed'):
            self.read()

    def test_missing_or_linked_receipt_input_is_rejected_before_version_commands(self):
        self.prepare()
        self.write_lock()
        for name in ('request.json', 'package.json', 'package-lock.json'):
            path = self.directory / name
            saved = self.root / (name + '.saved')
            path.rename(saved)
            with self.subTest(name=name, kind='missing'), \
                 patch.object(npm_lock.subprocess, 'check_output') as versions, \
                 self.assertRaisesRegex(ValueError, 'Missing or linked'):
                npm_lock.read(self.directory)
            versions.assert_not_called()
            path.symlink_to(saved)
            with self.subTest(name=name, kind='linked'), \
                 patch.object(npm_lock.subprocess, 'check_output') as versions, \
                 self.assertRaisesRegex(ValueError, 'Missing or linked'):
                npm_lock.read(self.directory)
            versions.assert_not_called()
            path.unlink()
            saved.rename(path)

    def test_actual_online_shell_uses_the_fresh_fixture_and_disables_lifecycle_hooks(self):
        self.prepare()
        script = shell_body(step_name(jobs((REPOSITORY_ROOT / '.github/workflows/build.yml').read_text())['publish'],
                                      'Resolve the published npm dependency (online)'))
        bindir = self.root / 'bin'
        bindir.mkdir()
        npm = bindir / 'npm'
        npm.write_text('#!' + sys.executable + '\n' + '''
import json, os, pathlib, sys
root=pathlib.Path.cwd()
value={'argv':sys.argv[1:],'cwd':str(root),'cache':os.environ.get('NPM_CONFIG_CACHE'),
       'userconfig':os.environ.get('NPM_CONFIG_USERCONFIG'),
       'globalconfig':os.environ.get('NPM_CONFIG_GLOBALCONFIG'),
       'gitPrompt':os.environ.get('GIT_TERMINAL_PROMPT')}
(root/'invocation.json').write_text(json.dumps(value))
''')
        npm.chmod(0o755)
        subprocess.run(['bash', '-euo', 'pipefail', '-c', script], cwd=self.root,
                       env={**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH']},
                       check=True, capture_output=True, text=True, timeout=20)
        invocation = json.loads((self.directory / 'invocation.json').read_text())
        self.assertEqual(invocation['argv'], ['install', '--package-lock-only', '--ignore-scripts',
                                             '--no-audit', '--no-fund', '--lockfile-version=3'])
        self.assertEqual(invocation['cwd'], str(self.directory))
        self.assertEqual(invocation['cache'], str(self.directory / 'cache'))
        self.assertEqual(invocation['userconfig'], str(self.directory / 'empty.npmrc'))
        self.assertEqual(invocation['globalconfig'], str(self.directory / 'empty-global.npmrc'))
        self.assertEqual(invocation['gitPrompt'], '0')


class PublishedReceipt(RuntimePackageFixture):
    def test_report_rejects_lock_receipt_from_different_repository_commit_or_version(self):
        assemble.assemble(self.inputs, self.out, check_npm_pack=False)
        digest = assemble.identity(self.out / 'manifest.json')['sha256']
        arguments = ['consumer', '--package', str(self.out), '--commit', COMMIT,
                     '--lock-directory', str(self.root / 'lock'), '--output', str(self.root / 'report'),
                     '--published-manifest-sha256', digest]
        for specifier, version in [(f'github:example/other#{COMMIT}', '0.1.0'),
                                   (f'github:{REPOSITORY}#' + 'd' * 40, '0.1.0'),
                                   (SPECIFIER, '0.2.0')]:
            lock = {'specifier': specifier, 'packageEntry': {**ENTRY, 'version': version}}
            with self.subTest(specifier=specifier, version=version), \
                 patch.object(sys, 'argv', arguments), \
                 patch.dict(os.environ, {'GITHUB_REPOSITORY': REPOSITORY}), \
                 patch.object(npm_lock, 'read', return_value=lock), \
                 patch.object(consumer, 'collect') as collect, redirect_stderr(io.StringIO()), \
                 self.assertRaisesRegex(ValueError, 'receipt differs from the published package'):
                consumer.main()
            collect.assert_not_called()
            self.assertFalse((self.root / 'report').exists())


if __name__ == '__main__':
    unittest.main()
