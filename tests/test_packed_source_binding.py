"""Raw source provenance and decoder manifests must agree, not just round-trip.

Tiny empty/custom-section Wasm is a packaging fixture, not a model test.
"""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_source_package as fixtures
source = fixtures.source
write_manifest = fixtures.write_manifest


class PackedSourceBinding(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.SourcePackage(methodName='test_missing_source_not_silently_replaced')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # Different valid bytes for nightly: correct hashes alone must not allow
        # assigning the nightly binary to the stable identity.
        path = fixture.inputs['upstream-nightly']
        raw = path / 'profiles/cpu-wasm32/browser/core.wasm'
        raw.write_bytes(b'\0asm\x01\0\0\0\0\x04\x01xNN')
        write_manifest(path, json.loads((path / 'manifest.json').read_text()))
        fixture.build(packing={'codecs': ['gzip'], 'pairs': []})
        self.root = fixture.out
        self.catalog_path = self.root / 'packed/llama-cpp/catalog.json'
        self.catalog = json.loads(self.catalog_path.read_text())
        self.stable = 'upstream-stable--cpu-wasm32'
        self.nightly = 'upstream-nightly--cpu-wasm32'

    def write_catalog(self, catalog):
        self.catalog_path.write_text(json.dumps(catalog))
        write_manifest(self.root, json.loads((self.root / 'manifest.json').read_text()))

    def test_other_source_binary_cannot_be_labelled_as_stable(self):
        c = copy.deepcopy(self.catalog)
        c['targets'][self.stable]['raw'] = c['targets'][self.nightly]['raw']
        c['targets'][self.stable]['representations'] = c['targets'][self.nightly]['representations']
        self.write_catalog(c)
        # The old verifier reconstructed the nightly bytes successfully and
        # accepted them under the stable coordinate.
        with self.assertRaisesRegex(ValueError, 'Packed target'):
            source.validate(self.root, check_npm_pack=False)

    def test_missing_extra_or_relabelled_targets_rejected_before_decoder(self):
        cases = {
            'missing': lambda c: c['targets'].pop(self.nightly),
            'extra': lambda c: c['targets'].update(unregistered=copy.deepcopy(c['targets'][self.stable])),
            'source': lambda c: c['targets'][self.stable]['identity'].update(source='upstream-nightly'),
            'profile': lambda c: c['targets'][self.stable]['identity'].update(profile='cpu-wasm64'),
            'variant': lambda c: c['targets'][self.stable]['identity'].update(variant='test'),
            'path': lambda c: c['targets'][self.stable]['raw'].update(path=c['targets'][self.nightly]['raw']['path']),
            'digest': lambda c: c['targets'][self.stable]['raw'].update(sha256='f' * 64),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                c = copy.deepcopy(self.catalog)
                mutate(c)
                self.write_catalog(c)
                with patch.object(source.subprocess, 'run', side_effect=AssertionError('Decoder ran before binding')):
                    with self.assertRaisesRegex(ValueError, 'Packed target'):
                        source.validate(self.root, check_npm_pack=False)

    def test_raw_binding_not_disabled_with_decoder_verification(self):
        c = copy.deepcopy(self.catalog)
        c['targets'][self.stable]['raw'] = c['targets'][self.nightly]['raw']
        self.write_catalog(c)
        with self.assertRaisesRegex(ValueError, 'Packed target'):
            source.validate(self.root, check_npm_pack=False, verify_packed=False)

    def test_missing_catalog_file_rejected_even_when_decoder_not_run(self):
        self.catalog_path.unlink()
        write_manifest(self.root, json.loads((self.root / 'manifest.json').read_text()))
        with self.assertRaisesRegex(ValueError, 'packed catalog'):
            source.validate(self.root, check_npm_pack=False, verify_packed=False)

    def test_duplicate_catalog_key_is_not_accepted_as_last_value(self):
        self.catalog_path.write_text(self.catalog_path.read_text().rstrip()[:-1] + ',"targets":{}}')
        write_manifest(self.root, json.loads((self.root / 'manifest.json').read_text()))
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            source.validate(self.root, check_npm_pack=False, verify_packed=False)

    def test_catalog_replaced_after_inventory_is_not_accepted(self):
        read = source.read_regular
        raw = self.catalog_path.read_bytes()
        def changed(path, **kwargs):
            if Path(path) == self.catalog_path:
                return raw + b' '
            return read(path, **kwargs)
        with patch.object(source, 'read_regular', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'Changed packed catalog identity'):
                source.validate(self.root, check_npm_pack=False, verify_packed=False)

    def test_non_json_numbers_in_catalog_are_rejected(self):
        self.catalog_path.write_text(self.catalog_path.read_text().rstrip()[:-1] + ',"bad":NaN}')
        write_manifest(self.root, json.loads((self.root / 'manifest.json').read_text()))
        with self.assertRaisesRegex(ValueError, 'Invalid JSON number'):
            source.validate(self.root, check_npm_pack=False, verify_packed=False)

    def test_valid_catalog_roundtrips_and_preserves_all_raw_inputs(self):
        source.validate(self.root, check_npm_pack=False)
        self.assertEqual(len(self.catalog['targets']), 7)
        for target in self.catalog['targets'].values():
            self.assertTrue((self.root / target['raw']['path']).is_file())


if __name__ == '__main__':
    unittest.main()
