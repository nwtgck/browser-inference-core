"""Run the real root package boundary from the image-native CI suite as well.

The browser producer scope function is invoked through Node. Package payloads and
browser execution are synthetic in these host checks; final npm pack is real.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ImagePackageBoundary(unittest.TestCase):
    def test_image_producer_and_publication_validator_share_a_contract(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                                 '-s', str(root / 'tests'), '-p', 'test_package_snapshot.py'],
                                cwd=root, capture_output=True, text=True, timeout=120,
                                env={**os.environ, 'BIC_METRICS_MODE': 'off'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
