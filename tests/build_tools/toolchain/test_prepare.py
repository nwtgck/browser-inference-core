"""Local SDK identity checks are independent from online acquisition."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_tools.toolchain.prepare import prepare, verify_sdk


class PrepareToolchain(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sdk = self.root / '.tools/emsdk'
        self.sdk.mkdir(parents=True)
        self.config = {'emsdkVersion': '1.2.3', 'emscriptenRelease': 'a' * 40,
                       'dawnTag': 'fixture', 'dawnSha256': 'b' * 64}
        (self.sdk / 'emscripten-releases-tags.json').write_text(json.dumps({'releases': {'1.2.3': 'a' * 40}}))
        (self.root / 'toolchain').mkdir()
        (self.root / 'toolchain/config.json').write_text(json.dumps(self.config))

    def test_release_matches_declared_version(self):
        verify_sdk(self.config, self.sdk)

    def test_wrong_release_is_rejected_before_running_sdk_code(self):
        with self.assertRaisesRegex(RuntimeError, 'release identity'):
            verify_sdk({**self.config, 'emscriptenRelease': 'c' * 40}, self.sdk)

    def test_linked_release_file_is_rejected(self):
        tags = self.sdk / 'emscripten-releases-tags.json'
        actual = self.sdk / 'actual'
        tags.rename(actual)
        tags.symlink_to(actual)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            verify_sdk(self.config, self.sdk)

    def test_missing_sdk_is_an_input_error(self):
        with self.assertRaisesRegex(ValueError, 'Missing'):
            verify_sdk(self.config, self.root / 'missing')

    def test_dawn_only_does_not_require_or_activate_sdk(self):
        with patch('build_tools.toolchain.prepare.prepare_dawn') as dawn, \
                patch('build_tools.toolchain.prepare.verify_sdk', side_effect=AssertionError('SDK not requested')):
            prepare(self.root, dawn_only=True)
        dawn.assert_called_once_with(self.config, self.root / '.tools')

    def test_invalid_dawn_tag_is_rejected_before_any_preparation(self):
        (self.root / 'toolchain/config.json').write_text(json.dumps({**self.config, 'dawnTag': '../outside'}))
        with patch('build_tools.toolchain.prepare.prepare_dawn', side_effect=AssertionError('Unsafe tag')):
            with self.assertRaisesRegex(ValueError, 'Dawn tag'):
                prepare(self.root, dawn_only=True)


if __name__ == '__main__':
    unittest.main()
