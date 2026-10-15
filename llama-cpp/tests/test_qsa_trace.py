"""The comparison must reject a no-op optimization and misleading traces."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parent / 'qsa-selection'
spec = importlib.util.spec_from_file_location('qsa_trace_compare', DIRECTORY / 'compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

# Include the independent algebra checks in the normal Python discovery job.
spec = importlib.util.spec_from_file_location('qsa_mask_equivalence', DIRECTORY / 'test_equivalence.py')
algebra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(algebra)
AllPoolsEquivalence = algebra.AllPoolsEquivalence


class TraceOracle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reference = self.root / 'reference'
        self.optimized = self.root / 'optimized'
        self.rows = {}
        for mode, directory in (('reference', self.reference), ('optimized', self.optimized)):
            directory.mkdir()
            (directory / 'logits.f32').write_bytes(struct.pack('<3f', 0.0, 1.0, -2.0))
            rows = []
            for phase in sorted(compare.PHASES):
                count = 0 if phase in ('reserve', 'parallel-reserve') else 4
                selected = count
                if mode == 'optimized' and phase in compare.SHORT:
                    selected = 0
                elif mode == 'optimized' and phase in compare.CROSS:
                    selected = 2
                rows.append({'phase': phase, 'indexer': selected, 'top_k': selected,
                             'query': selected, 'raw_key': count,
                             'selection_mask': count//2 if mode == 'optimized' and phase in compare.SHORT else count,
                             'pools64': 0, 'pools128': 0})
            self.rows[mode] = rows
            self.write(directory, rows)

    def write(self, directory, rows, fields=compare.FIELDS):
        with (directory / 'nodes.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def assert_invalid(self):
        with self.assertRaises((AssertionError, ValueError)):
            compare.compare(self.reference, self.optimized)

    def test_valid_trace_is_accepted(self):
        self.assertTrue(compare.compare(self.reference, self.optimized)['identical'])

    def test_reference_compared_to_itself_is_not_optimization_coverage(self):
        with self.assertRaisesRegex(AssertionError, 'bypass'):
            compare.compare(self.reference, self.reference)

    def test_nonfinite_identical_outputs_are_not_parity(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                for directory in (self.reference, self.optimized):
                    (directory / 'logits.f32').write_bytes(struct.pack('<f', value))
                self.assert_invalid()

    def test_equal_constant_output_is_not_a_useful_fixture(self):
        for value in (0.0, 1.0):
            with self.subTest(value=value):
                for directory in (self.reference, self.optimized):
                    (directory / 'logits.f32').write_bytes(struct.pack('<3f', value, value, value))
                self.assert_invalid()

    def test_malformed_empty_and_changed_outputs_are_rejected(self):
        for value in (b'', b'abc', struct.pack('<3f', 0, 1, -3), struct.pack('<4f', 0, 1, -2, 3)):
            with self.subTest(value=value):
                (self.optimized / 'logits.f32').write_bytes(value)
                self.assert_invalid()

    def test_missing_phase_is_rejected_even_when_both_runs_omit_it(self):
        for mode, directory in (('reference', self.reference), ('optimized', self.optimized)):
            self.write(directory, self.rows[mode][1:])
        self.assert_invalid()

    def test_changed_observation_order_is_rejected(self):
        self.write(self.optimized, list(reversed(self.rows['optimized'])))
        self.assert_invalid()

    def test_missing_top_k_column_is_rejected(self):
        rows = [{key: value for key, value in row.items() if key != 'top_k'} for row in self.rows['optimized']]
        self.write(self.optimized, rows, [key for key in compare.FIELDS if key != 'top_k'])
        self.assert_invalid()

    def test_negative_counts_are_rejected(self):
        self.rows['optimized'][0]['query'] = -1
        self.write(self.optimized, self.rows['optimized'])
        self.assert_invalid()

    def test_mask_sharing_must_execute_not_only_score_bypass(self):
        for row in self.rows['optimized']:
            if row['phase'] in compare.SHORT:
                row['selection_mask'] *= 2
        self.write(self.optimized, self.rows['optimized'])
        self.assert_invalid()

    def test_key_writes_and_ranked_path_are_not_optional(self):
        for phase, key, value in (('short-reused', 'raw_key', 0),
                                  ('cross-threshold', 'indexer', 0),
                                  ('restore-cross-threshold', 'indexer', 1),
                                  ('ranked-prefill', 'top_k', 0)):
            with self.subTest(phase=phase, key=key):
                rows = [dict(row) for row in self.rows['optimized']]
                next(row for row in rows if row['phase'] == phase)[key] = value
                self.write(self.optimized, rows)
                self.assert_invalid()


if __name__ == '__main__':
    unittest.main()
