"""Independent IQ3_S fixtures and full current CPU dequantization reference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPOSITORY_ROOT / 'runtimes/llama-cpp'
HERE = Path(__file__).resolve().parents[1] / 'webgpu-iq3s-subgroup'


class CurrentIq3CpuFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()

    @unittest.skipUnless(shutil.which('cc'), 'C compiler needed')
    def test_deterministic_synthetic_fixtures_use_actual_cpu_dequantization(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            subprocess.run([sys.executable, str(HERE / 'make_fixtures.py'), '--llama-source', str(self.upstream),
                            '--work', str(out)], check=True, capture_output=True)
            metadata = json.loads((out / 'fixture-source.json').read_text())
            self.assertIn('Unmodified complete current ggml-quants.c', metadata['cpuOracle'])
            self.assertEqual(metadata['sourceSha256']['ggml/src/ggml-common.h'],
                             hashlib.sha256((self.upstream / 'ggml/src/ggml-common.h').read_bytes()).hexdigest())
            rows = json.loads((out / 'fixtures/manifest.json').read_text())
            self.assertTrue(rows)
            self.assertTrue(any(row['skippedSlots'] for row in rows))
            for row in rows:
                self.assertEqual(len(row['params']), 13)
                self.assertEqual((out / 'fixtures' / row['referenceFile']).stat().st_size, row['m'] * row['slots'] * 16)
                for key in ('weights', 'input', 'ids', 'reference'):
                    self.assertEqual(hashlib.sha256((out / 'fixtures' / row[key + 'File']).read_bytes()).hexdigest(), row['hashes'][key])
                if row['baselineCase'] is not None:
                    baseline = next(item for item in rows if item['name'] == row['baselineCase'])
                    self.assertEqual(row['params'], baseline['params'])
                    self.assertEqual(row['hashes']['weights'], baseline['hashes']['weights'])
                    self.assertEqual(row['hashes']['input'], baseline['hashes']['input'])
                    reference = (out / 'fixtures' / row['referenceFile']).read_bytes()
                    previous = (out / 'fixtures' / baseline['referenceFile']).read_bytes()
                    for slot in range(row['slots']):
                        start, end = slot * row['m'] * 16, (slot + 1) * row['m'] * 16
                        expected = bytes(row['m'] * 16) if slot in row['skippedSlots'] else previous[start:end]
                        self.assertEqual(reference[start:end], expected)
            baseline = rows[0]
            for invalid_id in (baseline['storedExperts'], 0xfffffffe):
                with self.subTest(invalid_id=invalid_id):
                    ids = bytearray((out / 'fixtures' / baseline['idsFile']).read_bytes())
                    struct.pack_into('<I', ids, baseline['params'][2] * 4, invalid_id)
                    invalid_path = out / 'invalid-ids.bin'
                    invalid_path.write_bytes(ids)
                    result = subprocess.run([str(out / 'oracle'), str(out / 'fixtures' / baseline['weightsFile']),
                        str(out / 'fixtures' / baseline['inputFile']), str(invalid_path),
                        str(out / 'fixtures' / baseline['name'] / 'params.bin'), str(out / 'invalid-reference.bin')],
                        capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('selected expert outside stored payload', result.stderr)



if __name__ == "__main__":
    unittest.main()
