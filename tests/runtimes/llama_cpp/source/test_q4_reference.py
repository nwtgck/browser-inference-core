"""Known Q4_K blocks through the complete current CPU quantization unit."""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from tests.support.llama_cpp.native_quantization import build_reference

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1] / 'webgpu-q4-short'


class CurrentQ4Reference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE',
            REPOSITORY_ROOT / 'runtimes/llama-cpp/vendor/llama.cpp')).resolve()

    @unittest.skipUnless(shutil.which('cc') and shutil.which('c++'), 'Native C/C++ compilers required')
    def test_actual_upstream_dequantizer_reference_known_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            exe = build_reference(self.upstream, out, HERE / 'reference-main.cpp', name='q4-reference')
            params = [0,0,0,1,1,256,1,256,1,256,1,256,1,1,1,1]
            header = struct.pack('<5I', 0,144,256,0,16)
            for scales, qs, d, dmin, expected in [
                ([1]*4+[0]*4+[1]*4, 0x21, 0x3c00, 0, 384.0),
                ([255]*12, 255, 0x2c00, 0x2800, 14616.0),
                ([0]*12, 0, 0x3c00, 0, 0.0),
            ]:
                weights = struct.pack('<2H', d,dmin) + bytes(scales) + bytes([qs]*128)
                payload = header + weights + struct.pack('<256f', *([1.0]*256)) + struct.pack('<16I', *params)
                result = subprocess.run([str(exe)], input=payload, check=True, capture_output=True)
                self.assertEqual(struct.unpack('<2d', result.stdout), (expected, expected))
            result = subprocess.run([str(exe)], input=header, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'truncated fixture', result.stderr)



if __name__ == "__main__":
    unittest.main()
