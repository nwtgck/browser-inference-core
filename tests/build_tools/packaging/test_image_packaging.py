"""Reproduce the failed ggml-subtree package without needing a GPU or compiler."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import REPOSITORY_ROOT as ROOT
from tests.support import runtime_package as fixtures
from build_tools.packaging.notices import collect_notices, collect_subtree_notices, stage_toolchain_notices
from build_tools.packaging.files import tree_identity
from tests.support.packaging_cli import run_cli

spec = importlib.util.spec_from_file_location('image_package_regression', ROOT/'build_tools/runtimes/stable_diffusion_cpp/package.py')
image = importlib.util.module_from_spec(spec); spec.loader.exec_module(image)


def put(root, relative, content):
    path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


def refresh_manifest(root, manifest):
    manifest['files'] = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    (root/'manifest.json').write_text(json.dumps(manifest))


class ImagePackaging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)/'stable-diffusion-cpp'
        self.build, self.out = self.root/'build', self.root/'out'
        self.ggml = self.root/'vendor/ggml-webgpu-source'
        put(self.root, 'LICENSE', 'fixture core notice')
        put(self.root, 'README.md', 'Fixture packaging ONLY, not a runtime build')
        put(self.root, 'vendor/stable-diffusion.cpp/LICENSE', 'fixture image notice')
        for name in ('json.hpp', 'stb_image.h', 'stb_image_resize.h', 'stb_image_write.h'):
            put(self.root, 'vendor/stable-diffusion.cpp/thirdparty/'+name, 'fixture embedded notice')
        put(self.ggml, 'LICENSE', 'fixture ggml ROOT notice')
        put(self.ggml, 'ggml/src/fixture.cpp', '// no LICENSE under ggml, matching the CI failure')
        put(self.ggml, 'tools/unrelated/LICENSE', 'must not be collected')
        self.roots = [self.root.parent/'.tools'/name for name in ('emscripten', 'emdawnwebgpu_pkg')]
        for root in self.roots: put(root, 'LICENSE', 'fixture toolchain notice')
        for name in image.HELPERS: put(self.root, 'host/'+name, 'fixture helper')
        for profile, cfg in image.PROFILES.items():
            for variant, vcfg in image.VARIANTS.items():
                b = self.build/profile/variant
                for name in image.API_FILES: put(b, 'generated/'+name, fixtures.schema_file(name))
                put(b, 'runtime/core.wasm', b'\0asm\1\0\0\0')  # Only a synthetic package fixture.
                put(b, 'runtime/core.mjs', 'export default function fixtureOnly() {}')
                put(b, 'runtime/core.d.ts', 'export default function fixtureOnly(): void;')
                put(b, 'provenance.json', json.dumps({
                    'profile': profile, 'variant': variant, 'configuration': cfg, 'variantConfiguration': vcfg,
                    'sourceCommit': 'a'*40, 'upstreams': {'fixtureOnly': True}, 'sourceDirty': False,
                    'patches': fixtures.image_patch_provenance(),
                    'validation': {'compiled': True, 'fixtureOnly': True, 'realModelInference': False}}))

    def tearDown(self): self.tmp.cleanup()

    def package(self):
        with patch.object(image, 'ROOT', self.root): image.package(self.build, self.out, self.roots)

    def test_repository_license_is_shipped_when_compiled_subtree_has_no_notice(self):
        self.package()
        self.assertEqual((self.out/'licenses/ggml/LICENSE').read_bytes(), (self.ggml/'LICENSE').read_bytes())
        self.assertFalse((self.out/'licenses/ggml/tools').exists())
        result = image.validate(self.out)
        self.assertEqual(result['sourceCommit'], 'a'*40)
        for info in result['profiles'].values():
            for v in info['variants'].values(): self.assertFalse(v['validation']['realModelInference'])

    def test_subtree_notices_cannot_overwrite_repository_license(self):
        put(self.ggml, 'ggml/LICENSE', 'subtree specific notice')
        collect_subtree_notices(self.ggml, 'ggml', self.out)
        self.assertEqual((self.out/'LICENSE').read_text(), 'fixture ggml ROOT notice')
        self.assertEqual((self.out/'ggml/LICENSE').read_text(), 'subtree specific notice')

    def test_missing_or_empty_root_notice_still_fails_closed(self):
        (self.ggml/'LICENSE').unlink()
        with self.assertRaisesRegex(ValueError, 'notice'): self.package()
        (self.ggml/'LICENSE').touch()
        with self.assertRaisesRegex(ValueError, 'notice'): self.package()

    def test_licensing_policy_cannot_be_bypassed_by_rebuilding_manifest(self):
        self.package()
        manifest = image.validate(self.out)
        (self.out/'licenses/ggml/LICENSE').unlink()
        refresh_manifest(self.out, manifest)
        with self.assertRaisesRegex(ValueError, 'ggml repository license'): image.validate(self.out)

    def test_each_runtime_stages_its_own_complete_toolchain_notices(self):
        stage_toolchain_notices(self.roots, self.build/'ci-upload')
        staged = self.build/'ci-upload/toolchain-licenses'
        for root in self.roots:
            self.assertEqual((staged/root.name/'LICENSE').read_bytes(), (root/'LICENSE').read_bytes())
        with self.assertRaisesRegex(ValueError, 'already staged'): stage_toolchain_notices(self.roots, self.build/'ci-upload')
        with self.assertRaisesRegex(ValueError, 'distinct'): stage_toolchain_notices([self.roots[0]]*2, self.build/'different')

    def test_missing_or_duplicate_toolchain_notices_are_rejected(self):
        with patch.object(image, 'ROOT', self.root):
            for roots in [[], self.roots[:1], [self.roots[0]]*2]:
                with self.assertRaisesRegex(ValueError, 'exactly'): image.package(self.build, self.out, roots)
        (self.roots[1]/'LICENSE').unlink()
        with self.assertRaisesRegex(ValueError, 'No license notices'): self.package()

    def test_notice_symlinks_are_rejected_and_unrelated_symlinks_are_not_followed(self):
        directory = self.root/'notices'; directory.mkdir()
        put(directory, 'LICENSE', 'valid fixture notice')
        (directory/'bin').symlink_to(self.ggml, target_is_directory=True)
        count = collect_notices(directory, self.out)
        self.assertEqual(count, 1)
        self.assertFalse((self.out/'bin').exists())
        (directory/'COPYING').symlink_to(self.ggml/'LICENSE')
        with self.assertRaisesRegex(ValueError, 'Linked'): collect_notices(directory, self.out)

    def test_removing_toolchain_notice_even_with_consistent_manifest_fails(self):
        self.package(); manifest = image.validate(self.out)
        (self.out/'licenses/toolchain/emdawnwebgpu_pkg/LICENSE').unlink()
        refresh_manifest(self.out, manifest)
        with self.assertRaisesRegex(ValueError, 'toolchain notices'): image.validate(self.out)

    def cli(self, *, fail_pack=None, defer=False):
        arguments = ['--build-root', str(self.build), '--output', str(self.out)]
        for root in self.roots:
            arguments += ['--license-root', str(root)]
        if defer:
            arguments.append('--defer-npm-pack')
        return run_cli(self.root, 'build_tools.runtimes.stable_diffusion_cpp.package', arguments,
                       source_root=self.root, fail_pack=fail_pack)

    def test_cli_npm_failure_keeps_the_previous_valid_package(self):
        self.package()
        before, inode = tree_identity(self.out), self.out.stat().st_ino
        result, calls = self.cli(fail_pack=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('injected npm pack failure', result.stderr)
        self.assertEqual(tree_identity(self.out), before)
        self.assertEqual(self.out.stat().st_ino, inode)
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(Path(calls[0]['cwd']), self.out)
        self.assertEqual(list(self.out.parent.glob('bicore-output-*')), [])

    def test_cli_validates_once_before_activation_and_defers_only_when_requested(self):
        result, calls = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(Path(calls[0]['cwd']).name, 'prepared')
        self.assertNotEqual(Path(calls[0]['cwd']), self.out)
        result, calls = self.cli(defer=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_repackage_input_copy_schema_and_notice_failures_keep_old_package(self):
        self.package()
        before = tree_identity(self.out)
        for relative in ('runtime/core.wasm', 'generated/schema.mjs', 'provenance.json'):
            path = self.build / next(iter(image.PROFILES)) / 'browser' / relative
            value = path.read_bytes(); path.unlink()
            with self.subTest(relative=relative), self.assertRaises((ValueError, FileNotFoundError)):
                self.package()
            self.assertEqual(tree_identity(self.out), before)
            path.write_bytes(value)
        with patch.object(image, 'write_index', side_effect=OSError('notice index failure')):
            with self.assertRaisesRegex(OSError, 'notice index failure'):
                self.package()
        self.assertEqual(tree_identity(self.out), before)


if __name__ == '__main__': unittest.main()
