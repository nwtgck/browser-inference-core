"""Scoped notice paths through Git transport, catalog validation and raw reuse.

The artifacts are local fixtures: no network, upstream compiler or model inference.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import package_sources as source
import pipeline_plan as pipeline
from package_inputs import file_identity
from package_notices import collect_notices
import test_pipeline_plan as fixtures
from test_multi_runtime import write_manifest

SCOPED_NOTICE = 'licenses/1/node_modules/@jridgewell/gen-mapping/LICENSE'
NOTICE = b'Fixture notice, copied byte-for-byte.\n'


def git(root, *args):
    return subprocess.check_output(
        ['git', *map(str, args)], cwd=root, text=True, stderr=subprocess.STDOUT,
        timeout=30,
    ).strip()


def repository(root):
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'core.autocrlf', 'false')
    return root


def publish(root):
    # Notices intentionally preserve node_modules in the original relative path.
    git(root, 'add', '--force', '.')
    git(root, 'commit', '-qm', 'Local package fixture')
    git(root, 'branch', '-f', 'artifacts')
    return git(root, 'rev-parse', 'HEAD')


class RelativePackagePaths(unittest.TestCase):
    def test_scoped_names_are_payload_paths_not_source_identifiers(self):
        for value in (
            SCOPED_NOTICE,
            'llama-cpp-browser-core/' + SCOPED_NOTICE,
            'stable-diffusion-cpp-browser-core/licenses/toolchain/emscripten/'
            'node_modules/@jridgewell/sourcemap-codec/LICENSE',
            'runtimes/llama-cpp/sources/upstream-stable/' + SCOPED_NOTICE,
            'licenses/node_modules/@jridgewell',
        ):
            with self.subTest(value=value):
                self.assertEqual(source.safe_relative(value), value)

    def test_allowing_at_sign_does_not_allow_traversal_or_path_reinterpretation(self):
        for value in (
            '', '.', '..', '/licenses/@scope/file',
            'licenses//file', 'licenses/@scope/../file',
            'licenses/@scope/./file', 'licenses/@scope/file/',
            r'licenses\@scope\file', 'C:/licenses/file',
            'licenses/file:stream', '//server/share',
            'licenses/%2e%2e/file', 'licenses/file?query',
            'licenses/file#fragment', 'licenses/file\x00',
            'licenses/file\n', 'licenses/file\r', 'licenses/file\t',
            None, 1, [], {},
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'Unsafe package path'):
                    source.safe_relative(value)

    def test_rejection_identifies_the_path_without_literal_control_characters(self):
        value = 'licenses/@scope/file\nnext'
        with self.assertRaises(ValueError) as failure:
            source.safe_relative(value)
        self.assertIn(repr(value), str(failure.exception))
        self.assertNotIn('\n', str(failure.exception))


class ScopedArchivePaths(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = repository(self.root / 'repo')
        (self.repo / 'manifest.json').write_text('{"formatVersion":3}')
        path = self.repo / 'llama-cpp-browser-core' / SCOPED_NOTICE
        path.parent.mkdir(parents=True)
        path.write_bytes(NOTICE)
        self.commit = publish(self.repo)

    def test_actual_git_archive_preserves_notices_and_legacy_plan_builds_cold(self):
        out = self.root / 'snapshot'
        receipt = pipeline.fetch_snapshot(str(self.repo), out)
        self.assertEqual(receipt['artifactCommit'], self.commit)
        self.assertEqual(
            (out / 'llama-cpp-browser-core' / SCOPED_NOTICE).read_bytes(), NOTICE,
        )
        result = pipeline.create_plan(out, receipt, self.root / 'reuse')
        self.assertEqual(result['reuse'], {})
        self.assertTrue(result['compile']['include'])
        self.assertIsNone(result['imageReuse'])
        self.assertIn('Legacy', result['imageReason'])

    def test_invalid_filename_is_reported_not_silently_skipped(self):
        name = 'licenses/@scope/LICENSE:stream'
        target = self.repo / name
        target.parent.mkdir(parents=True)
        target.write_bytes(b'invalid path')
        commit = publish(self.repo)
        with self.assertRaises(ValueError) as failure:
            pipeline.extract_snapshot(self.repo, commit, self.root / 'bad')
        self.assertIn(repr(name), str(failure.exception))
        self.assertFalse((self.root / 'bad' / name).exists())

    def test_hidden_control_file_under_scope_is_still_rejected(self):
        target = self.repo / 'licenses/@scope/.gitmodules'
        target.parent.mkdir(parents=True)
        target.write_text('forbidden control file')
        commit = publish(self.repo)
        with self.assertRaisesRegex(ValueError, 'Hidden'):
            pipeline.extract_snapshot(self.repo, commit, self.root / 'bad')

    def test_scoped_symlink_is_still_rejected_before_reading_target(self):
        private = self.root / 'private'
        private.write_bytes(b'must not be copied')
        target = self.repo / 'licenses/@scope/LICENSE'
        target.parent.mkdir(parents=True)
        target.symlink_to(private)
        commit = publish(self.repo)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            pipeline.extract_snapshot(self.repo, commit, self.root / 'bad')
        self.assertFalse((self.root / 'bad/licenses/@scope/LICENSE').exists())
        self.assertEqual(private.read_bytes(), b'must not be copied')

    def test_case_colliding_scopes_are_still_rejected(self):
        for scope in ('@Scope', '@scope'):
            path = self.repo / 'licenses' / scope / 'LICENSE'
            path.parent.mkdir(parents=True)
            path.write_bytes(NOTICE)
        commit = publish(self.repo)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pipeline.extract_snapshot(self.repo, commit, self.root / 'bad')

    def test_archive_entry_and_byte_budgets_still_apply_to_scoped_notices(self):
        for name, value in (('MAX_FILES', 1), ('MAX_FILE', 1), ('MAX_TOTAL', 1)):
            with self.subTest(budget=name), patch.object(pipeline, name, value):
                with self.assertRaises(ValueError):
                    pipeline.extract_snapshot(self.repo, self.commit, self.root / name)


class ScopedCatalogPaths(unittest.TestCase):
    def setUp(self):
        # Reuse a fixture that carries full current reuse fingerprints and smoke
        # metadata. This metadata is synthetic, not actual inference evidence.
        self.fixture = fixtures.PipelinePlan(
            methodName='test_unchanged_packages_keep_original_bytes_and_need_no_compilation',
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        upstream = self.root / 'toolchain'
        notice = upstream / 'node_modules/@jridgewell/gen-mapping/LICENSE'
        notice.parent.mkdir(parents=True)
        notice.write_bytes(NOTICE)
        packages = [*self.fixture.fixture.inputs.values(), self.fixture.fixture.image]
        for package in packages:
            collect_notices(upstream, package / 'licenses/1')
            write_manifest(package, json.loads((package / 'manifest.json').read_text()))
        shutil.rmtree(self.fixture.previous)
        self.fixture.fixture.build()
        self.previous = self.fixture.previous

    def test_notice_collection_assembly_and_npm_inventory_preserve_scope_and_bytes(self):
        manifest = source.validate(self.previous, check_npm_pack=True)
        self.assertEqual(manifest['formatVersion'], 4)
        names = [entry['path'] for entry in manifest['files'] if '@jridgewell' in entry['path']]
        self.assertEqual(len(names), 3)
        for name in names:
            self.assertEqual((self.previous / name).read_bytes(), NOTICE)

    def test_v4_real_git_fetch_and_reuse_preserve_source_notices(self):
        remote = repository(self.root / 'git-remote')
        shutil.copytree(self.previous, remote, dirs_exist_ok=True)
        publish(remote)
        out = self.root / 'snapshot'
        receipt = pipeline.fetch_snapshot(str(remote), out)
        result = pipeline.create_plan(out, receipt, self.root / 'scoped-reuse')
        self.assertEqual(result['compile']['include'], [])
        self.assertIsNotNone(result['imageReuse'])
        self.assertEqual(set(result['reuse']), set(self.fixture.fixture.inputs))
        for name, original in self.fixture.fixture.inputs.items():
            copied = self.root / 'scoped-reuse/llama-cpp' / name / SCOPED_NOTICE
            self.assertEqual(file_identity(copied), file_identity(original / SCOPED_NOTICE))

    def test_license_tampering_is_not_masked_by_accepting_at_sign(self):
        path = next(self.previous.glob('runtimes/*/sources/*/' + SCOPED_NOTICE))
        path.write_bytes(b'corrupt notice')
        with self.assertRaisesRegex(ValueError, 'identity'):
            source.validate(self.previous, check_npm_pack=False)
        with self.assertRaisesRegex(ValueError, 'hash/size'):
            pipeline.validate_raw_snapshot(self.previous)


if __name__ == '__main__':
    unittest.main()
