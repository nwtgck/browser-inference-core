"""Model graph contract/overlay validation; not a browser inference test."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_model_graph as generator
import model_graph_contract as contract
import prepare_model_graph_chat as overlay
import upstream_provenance as provenance


class ModelGraphContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exports = self.root / 'base.json'
        self.exports.write_text('["_malloc", "_free"]')
        self.clang = shutil.which('clang')
        if not self.clang:
            self.skipTest('clang AST tooling required')
        self.schema = generator.generate(ROOT / 'bridge/model-graph.h', self.root, self.exports, self.clang)
        self.record = {'modelGraphBoundary': {'schemaSha256': self.schema['schemaSha256']}}

    def test_generated_schema_and_types_validate_and_preserve_base_exports(self):
        self.assertEqual(contract.validate(self.root, self.record), self.schema['schemaSha256'])
        exports = json.loads((self.root / 'model-graph-exports.json').read_text())
        self.assertEqual(exports[:2], ['_malloc', '_free'])
        self.assertEqual(len(set(exports)), len(exports))
        self.assertNotIn('Promise', (self.root / contract.TYPES_FILE).read_text())

    def test_legacy_package_without_extension_remains_valid(self):
        (self.root / contract.SCHEMA_FILE).unlink()
        (self.root / contract.TYPES_FILE).unlink()
        self.assertIsNone(contract.validate(self.root, {}))

    def test_missing_provenance_schema_or_types_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            contract.validate(self.root, {})
        for name in [contract.SCHEMA_FILE, contract.TYPES_FILE]:
            file = self.root / name; before = file.read_bytes(); file.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Incomplete'):
                contract.validate(self.root, self.record)
            file.write_bytes(before)

    def test_mismatched_runtime_provenance_or_types_is_rejected(self):
        record = copy.deepcopy(self.record); record['modelGraphBoundary']['schemaSha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            contract.validate(self.root, record)
        types = self.root / contract.TYPES_FILE
        types.write_text(types.read_text().replace('bigint', 'number', 1))
        with self.assertRaisesRegex(ValueError, 'TypeScript'):
            contract.validate(self.root, self.record)

    def test_tampered_schema_rejected_even_when_self_hash_is_updated(self):
        for mutation in ['signature', 'duplicate', 'missing', 'pointer', 'async', 'export']:
            schema = copy.deepcopy(self.schema)
            if mutation == 'signature': schema['functions'][0]['returnType'] = 'uint64_t'
            elif mutation == 'duplicate': schema['functions'][-1] = schema['functions'][0]
            elif mutation == 'missing': schema['functions'].pop()
            elif mutation == 'pointer': schema['pointerRepresentation'] = 'number'
            elif mutation == 'async': schema['suspension'] = 'promise'
            else: schema['functions'][0]['export'] = '_unknown'
            schema.pop('schemaSha256')
            schema['schemaSha256'] = hashlib.sha256((json.dumps(schema, indent=2) + '\n').encode()).hexdigest()
            (self.root / contract.SCHEMA_FILE).write_text(json.dumps(schema))
            record = {'modelGraphBoundary': {'schemaSha256': schema['schemaSha256']}}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                contract.validate(self.root, record)

    def test_generator_rejects_missing_changed_or_unreviewed_exports(self):
        original = (ROOT / 'bridge/model-graph.h').read_text()
        for change in ['missing', 'type', 'extra']:
            text = original
            if change == 'missing': text = text.replace('lcb_model_graph_version(void)', 'not_an_export(void)')
            elif change == 'type': text = text.replace('uint32_t lcb_model_graph_version', 'uint64_t lcb_model_graph_version')
            else: text += '\nuint32_t lcb_model_graph_unknown(void);\n'
            header = self.root / 'modified.h'; header.write_text(text)
            with self.subTest(change=change), self.assertRaises(ValueError):
                generator.generate(header, self.root, self.exports, self.clang)

    def test_duplicate_base_export_is_rejected(self):
        self.exports.write_text('["_lcb_model_graph_version"]')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            generator.generate(ROOT / 'bridge/model-graph.h', self.root, self.exports, self.clang)


class ModelGraphOverlay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'pristine'
        self.output = self.root / 'overlay'
        upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        self.inputs = json.loads((ROOT / 'tests/model-graph/inputs.json').read_text())
        for relative in self.inputs['upstream']:
            src = upstream / relative
            if not src.is_file(): self.skipTest('Reviewed upstream inputs required')
            target = self.source / relative; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(src.read_bytes())

    def test_preparation_is_idempotent_checked_and_does_not_modify_upstream(self):
        before = {key: (self.source / key).read_bytes() for key in self.inputs['upstream']}
        result = overlay.prepare(self.source, self.output)
        self.assertEqual(result['outputs'], self.inputs['outputs'])
        self.assertEqual(result, overlay.prepare(self.source, self.output))
        self.assertFalse((self.output / 'common/chat.h').exists())
        self.assertEqual(before, {key: (self.source / key).read_bytes() for key in before})
        for relative, sha in result['outputs'].items():
            self.assertEqual(hashlib.sha256((self.output / relative).read_bytes()).hexdigest(), sha)

    def reporting_fixture(self):
        root = self.root / 'report'
        shutil.copytree(self.source, root / 'vendor/llama.cpp')
        paths = ['bridge/model-graph.h', 'bridge/model-graph.cpp', 'bridge/model-graph-internal.h',
                 'scripts/generate_model_graph.py', 'tests/model-graph/inputs.json',
                 str(overlay.PATCH.relative_to(ROOT))]
        for relative in paths:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        prepared = overlay.prepare(self.source, self.output)
        record = {**prepared, 'schemaSha256': 'd' * 64}
        manifest = {'profiles': {'cpu': {'variants': {'browser': {'modelGraphBoundary': record},
                                                     'test': {'modelGraphBoundary': copy.deepcopy(record)}}}}}
        return root, manifest

    def test_report_describes_the_verified_overlay_not_an_unapplied_patch(self):
        root, manifest = self.reporting_fixture()
        report = provenance.model_graph_preparations(root, manifest)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['schemaSha256'], 'd' * 64)
        self.assertEqual(report[0]['application']['enabledProfileVariants'], ['cpu/browser', 'cpu/test'])
        self.assertEqual(set(report[0]['compiledCopies']), set(self.inputs['outputs']))
        for relative, identity in report[0]['compiledCopies'].items():
            self.assertEqual(identity['sha256'], self.inputs['outputs'][relative])

    def test_report_rejects_partial_activation_or_forged_overlay_provenance(self):
        root, original = self.reporting_fixture()
        for mutation in ['missing', 'schema', 'output', 'invalid-record']:
            manifest = copy.deepcopy(original)
            target = manifest['profiles']['cpu']['variants']['test']
            if mutation == 'missing': target.pop('modelGraphBoundary')
            elif mutation == 'schema': target['modelGraphBoundary']['schemaSha256'] = 'e' * 64
            elif mutation == 'output': target['modelGraphBoundary']['outputs']['common/chat.cpp'] = '0' * 64
            else: target['modelGraphBoundary'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                provenance.model_graph_preparations(root, manifest)
        self.assertEqual(provenance.model_graph_preparations(root, {'profiles': {'cpu': {'variants': {'browser': {}}}}}), [])

    def test_changed_upstream_is_not_silently_accepted(self):
        path = self.source / 'common/chat.cpp'; path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'semantic review'):
            overlay.prepare(self.source, self.output)

    def test_output_inside_pristine_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            overlay.prepare(self.source, self.source / 'new')

    def test_stale_header_cannot_shadow_pristine_header(self):
        header = self.output / 'common/chat.h'; header.parent.mkdir(parents=True)
        header.write_text('stale')
        overlay.prepare(self.source, self.output)
        self.assertFalse(header.exists())

    def test_linked_source_and_output_are_rejected(self):
        alias = self.root / 'alias'; alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            overlay.prepare(alias, self.output)
        with self.assertRaisesRegex(ValueError, 'Linked'):
            overlay.prepare(self.source, alias)


if __name__ == '__main__':
    unittest.main()
