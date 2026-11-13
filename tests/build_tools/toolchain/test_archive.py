"""Offline archive validation; acquisition is an explicit workflow operation."""
import hashlib
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import warnings

from build_tools.toolchain.archive import prepare_dawn


class DawnArchive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tools = Path(self.tmp.name)
        self.config = {'dawnTag': 'fixture', 'dawnSha256': ''}
        self.archive = self.tools / 'downloads/emdawnwebgpu_pkg-fixture.zip'
        self.archive.parent.mkdir()

    def archive_bytes(self, name='emdawnwebgpu_pkg/emdawnwebgpu.port.py', symlink=False):
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as archive:
            info = zipfile.ZipInfo(name)
            if symlink:
                info.external_attr = 0o120777 << 16
            archive.writestr(info, 'fixture data, not a real port')
        value = out.getvalue()
        self.config['dawnSha256'] = hashlib.sha256(value).hexdigest()
        return value

    def old_port(self):
        old = self.tools / 'emdawnwebgpu_pkg/old'
        old.parent.mkdir()
        old.write_text('preserve until complete validation')
        return old

    def test_cached_download_is_verified_and_port_reconstructed_without_network(self):
        self.archive.write_bytes(self.archive_bytes())
        stale = self.old_port()
        prepare_dawn(self.config, self.tools)
        self.assertFalse(stale.exists())
        self.assertTrue((stale.parent / 'emdawnwebgpu.port.py').is_file())

    def test_corrupt_cached_archive_fails_before_existing_port_is_removed(self):
        self.archive.write_bytes(self.archive_bytes() + b'changed')
        old = self.old_port()
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            prepare_dawn(self.config, self.tools)
        self.assertTrue(old.is_file())

    def test_missing_archive_is_an_explicit_input_error_not_a_download(self):
        old = self.old_port()
        with self.assertRaisesRegex(RuntimeError, 'Missing pinned Dawn archive'):
            prepare_dawn(self.config, self.tools)
        self.assertTrue(old.is_file())
        self.assertFalse(self.archive.exists())
        self.assertEqual(list(self.archive.parent.iterdir()), [])

    def test_bad_input_hash_does_not_extract(self):
        self.archive_bytes()
        self.archive.write_bytes(b'bad input')
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            prepare_dawn(self.config, self.tools)
        self.assertFalse((self.tools / 'emdawnwebgpu_pkg').exists())

    def test_archive_path_traversal_and_symlinks_are_rejected_before_replacement(self):
        old = self.old_port()
        for name, linked in [('../escape', False), ('/tmp/escape', False),
                             ('emdawnwebgpu_pkg/link', True), ('other/root', False),
                             ('emdawnwebgpu_pkg/../escape', False)]:
            with self.subTest(name=name):
                self.archive.write_bytes(self.archive_bytes(name, linked))
                with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                    prepare_dawn(self.config, self.tools)
                self.assertTrue(old.is_file())

    def test_cached_archive_and_extracted_port_cannot_be_symlinks(self):
        target = self.tools / 'target'
        target.write_bytes(self.archive_bytes())
        self.archive.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, 'linked'):
            prepare_dawn(self.config, self.tools)
        self.archive.unlink()
        self.archive.write_bytes(target.read_bytes())
        (self.tools / 'emdawnwebgpu_pkg').symlink_to(self.tools, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'linked'):
            prepare_dawn(self.config, self.tools)

    def test_missing_port_and_bad_archive_preserve_previous_port(self):
        old = self.old_port()
        for data in (self.archive_bytes('emdawnwebgpu_pkg/not-the-port.py'), b'not zip'):
            self.config['dawnSha256'] = hashlib.sha256(data).hexdigest()
            self.archive.write_bytes(data)
            with self.assertRaises((RuntimeError, zipfile.BadZipFile)):
                prepare_dawn(self.config, self.tools)
            self.assertTrue(old.is_file())

    def test_duplicate_names_and_file_directory_collisions_are_rejected(self):
        old = self.old_port()
        for names in [('emdawnwebgpu_pkg/emdawnwebgpu.port.py',) * 2,
                      ('emdawnwebgpu_pkg/emdawnwebgpu.port.py', 'emdawnwebgpu_pkg/emdawnwebgpu.port.py/child')]:
            with self.subTest(names=names):
                out = io.BytesIO()
                with zipfile.ZipFile(out, 'w') as archive:
                    for name in names:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', UserWarning)
                            archive.writestr(name, 'fixture')
                data = out.getvalue()
                self.config['dawnSha256'] = hashlib.sha256(data).hexdigest()
                self.archive.write_bytes(data)
                with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                    prepare_dawn(self.config, self.tools)
                self.assertTrue(old.is_file())

    def test_executable_bit_is_preserved_but_privileged_bits_are_not(self):
        with zipfile.ZipFile(self.archive, 'w') as archive:
            info = zipfile.ZipInfo('emdawnwebgpu_pkg/emdawnwebgpu.port.py')
            info.external_attr = (stat.S_IFREG | 0o6777) << 16
            archive.writestr(info, 'fixture')
        self.config['dawnSha256'] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        prepare_dawn(self.config, self.tools)
        self.assertEqual(stat.S_IMODE((self.tools / 'emdawnwebgpu_pkg/emdawnwebgpu.port.py').stat().st_mode), 0o755)

    def test_failed_activation_rolls_back_previous_port(self):
        self.archive.write_bytes(self.archive_bytes())
        old = self.old_port()
        import os
        original = os.replace
        def replace(source, destination):
            if Path(source).parent.name == 'unpacked':
                raise OSError('fixture activation failure')
            return original(source, destination)
        with patch('build_tools.toolchain.archive.os.replace', side_effect=replace):
            with self.assertRaisesRegex(OSError, 'activation failure'):
                prepare_dawn(self.config, self.tools)
        self.assertTrue(old.is_file())
        self.assertFalse(list(self.tools.glob('dawn-prepare-*')))


    def test_failed_recovery_does_not_delete_the_only_previous_package(self):
        self.archive.write_bytes(self.archive_bytes())
        self.old_port()
        import os
        original = os.replace
        def replace(source, destination):
            if Path(source).parent.name == 'unpacked' or Path(source).name == 'previous':
                raise OSError('fixture activation and recovery failure')
            return original(source, destination)
        with patch('build_tools.toolchain.archive.os.replace', side_effect=replace):
            with self.assertRaisesRegex(RuntimeError, 'previous package retained at') as caught:
                prepare_dawn(self.config, self.tools)
        recovery, = self.tools.glob('dawn-prepare-*/previous')
        self.assertIn(str(recovery), str(caught.exception))
        self.assertEqual((recovery / 'old').read_text(), 'preserve until complete validation')


if __name__ == '__main__':
    unittest.main()
