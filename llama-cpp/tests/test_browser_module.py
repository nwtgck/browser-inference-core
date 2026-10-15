"""Check byte-preserving URL annotation with synthetic JavaScript, not compilation."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_browser_module as browser

SOURCE = (b'// Synthetic module, not an inference runtime.\r\n'
          + b'function findWasmBinary(){return new URL("core.wasm",import.meta.url).href}'
          + b'\nexport default async () => ({ fixture: true });\n')


class BrowserModulePreparation(unittest.TestCase):
    def test_adds_only_one_annotation_and_is_idempotent(self):
        annotated = browser.annotate(SOURCE)
        self.assertEqual(annotated.replace(browser.ANNOTATION, b'', 1), SOURCE)
        self.assertEqual(annotated.count(browser.ANNOTATION), 1)
        self.assertEqual(browser.annotate(annotated), annotated)

    def test_unknown_or_multiple_boundaries_fail_before_writing(self):
        annotated = browser.annotate(SOURCE)
        cases = [b'export default () => {};', SOURCE + SOURCE, SOURCE + annotated,
                 SOURCE.replace(b'"core.wasm"', b'"./core.wasm"'),
                 SOURCE.replace(b'import.meta.url', b'otherBase'),
                 annotated.replace(browser.ANNOTATION, browser.ANNOTATION * 2),
                 SOURCE + b'new URL( "core.wasm", import.meta.url);',
                 SOURCE + b'new URL( "other.wasm", import.meta.url);']
        with tempfile.TemporaryDirectory() as temporary:
            module = Path(temporary) / 'core.mjs'
            for source in cases:
                with self.subTest(source=source):
                    module.write_bytes(source)
                    with self.assertRaisesRegex(ValueError, 'Unknown generated browser Wasm URL boundary'):
                        browser.prepare(module)
                    self.assertEqual(module.read_bytes(), source)

    def test_unrelated_functions_quotes_and_whitespace_are_preserved(self):
        source = (b'const other = () => new  URL (\n\t\'core.wasm\' , import . meta . url , ).href;\r\n'
                  b'function findWasmBinary(path){return other()}')
        result = browser.annotate(source)
        self.assertIn(browser.ANNOTATION + b'\'core.wasm\'', result)
        self.assertEqual(result.replace(browser.ANNOTATION, b'', 1), source)
        self.assertEqual(browser.annotate(result), result)

    def test_provenance_records_actual_input_output_and_script_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = Path(temporary) / 'core.mjs'
            module.write_bytes(SOURCE)
            first = browser.prepare(module)
            result = module.read_bytes()
            self.assertEqual(first['input'], {'bytes': len(SOURCE), 'sha256': hashlib.sha256(SOURCE).hexdigest()})
            self.assertEqual(first['output'], browser.identity(result))
            self.assertEqual(first['implementation'], {
                'path': browser.SCRIPT_PATH, **browser.identity(Path(browser.__file__).read_bytes())})
            self.assertFalse(first['alreadyAnnotated'])
            second = browser.prepare(module)
            self.assertTrue(second['alreadyAnnotated'])
            self.assertEqual(second['input'], first['output'])
            self.assertEqual(second['output'], first['output'])
            self.assertEqual(module.read_bytes(), result)


if __name__ == '__main__':
    unittest.main()
