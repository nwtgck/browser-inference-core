"""Pinned header integrity is separate from source preparation semantics."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import REPOSITORY_ROOT
from build_tools.toolchain import dawn


class DawnIntegrity(unittest.TestCase):
    def test_every_declared_header_is_required_and_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            originals = {}
            for index, name in enumerate(dawn.PINNED_HEADERS):
                data = f'header fixture {index}\n'.encode()
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                originals[name] = data
            expected = {name: hashlib.sha256(data).hexdigest() for name, data in originals.items()}
            with patch.object(dawn, 'PINNED_HEADERS', expected):
                dawn.verify(root)
                for name, data in originals.items():
                    with self.subTest(header=name):
                        target = root / name
                        target.write_bytes(data + b'changed\n')
                        with self.assertRaisesRegex(ValueError, 'Dawn header integrity'):
                            dawn.verify(root)
                        target.unlink()
                        with self.assertRaises(ValueError):
                            dawn.verify(root)
                        target.write_bytes(data)
                dawn.verify(root)

    def test_linked_header_is_not_an_admitted_sdk_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / 'real.h'
            real.write_bytes(b'header\n')
            (root / 'linked.h').symlink_to(real)
            with patch.object(dawn, 'PINNED_HEADERS', {'linked.h': hashlib.sha256(real.read_bytes()).hexdigest()}):
                with self.assertRaisesRegex(ValueError, 'Linked'):
                    dawn.verify(root)

    def test_release_pin_matches_acquisition_configuration(self):
        config = json.loads((REPOSITORY_ROOT / 'toolchain/config.json').read_text())
        self.assertEqual(dawn.PINNED_RELEASE, config['dawnTag'])
        self.assertEqual(len(dawn.PINNED_HEADERS), 4)

    def test_production_source_import_loads_no_historical_stages(self):
        command = ('import sys,json; import build_tools.runtimes.llama_cpp.source; '
                   'print(json.dumps(sorted(name for name in sys.modules '
                   'if name.startswith(("build_tools.runtimes.llama_cpp.sources.", "tests.")))))')
        loaded = json.loads(subprocess.check_output([sys.executable, '-S', '-c', command],
                                                     cwd=REPOSITORY_ROOT, text=True))
        self.assertEqual(loaded, [])


if __name__ == '__main__':
    unittest.main()
