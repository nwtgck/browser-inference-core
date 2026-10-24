"""No compiler/native build: structured notices over tiny original documents."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_license_index import INDEX, embedded_comments, make_index, owner, safe_path, validate_index, write_index


def put(root, name, text):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


class NativeLicenseIndex(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        put(self.root, 'LICENSE', 'Core notice\r\n')
        put(self.root, 'licenses/0/LICENSE', 'llama.cpp notice')
        put(self.root, 'licenses/0/licenses/LICENSE-jsonhpp', 'JSON complete license')
        put(self.root, 'licenses/embedded/vendor/nlohmann/json.hpp.txt', '// SPDX-License-Identifier: MIT\n/* Attribution without keywords */\nint implementation;')

    def write(self):
        write_index(self.root, runtime='llama-cpp', license_roots=['llama.cpp'])
        return json.loads((self.root / INDEX).read_text())

    def validate(self):
        validate_index(self.root, runtime='llama-cpp', license_roots=['llama.cpp'])

    def test_own_package_metadata_does_not_leak_into_upstream_components(self):
        put(self.root, 'package.json', json.dumps({'version': '0.1.0', 'license': 'MIT'}))
        index, _ = make_index(self.root, runtime='llama-cpp', license_roots=['llama.cpp'])
        values = {c['id']: c for c in index['components']}
        self.assertEqual(values['llama-cpp-browser-core']['version'], '0.1.0')
        self.assertEqual(values['llama-cpp-browser-core']['license'], 'MIT')
        self.assertIsNone(values['llama.cpp']['version'])
        self.assertIsNone(values['llama.cpp']['license'])

    def test_character_prefixes_do_not_hide_keywordless_attribution(self):
        for literal in ("L'a'", "u'a'", "U'a'", "u8'a'", "'a'"):
            source = literal + "; // By A. Contributor\nchar c='x'; /* Terms unchanged */"
            result = embedded_comments(source)
            self.assertIn('// By A. Contributor', result)
            self.assertIn('/* Terms unchanged */', result)
        source = "// By Contributor\nint n=0xdead'beef; /* trailing notice */"
        self.assertIn('/* trailing notice */', embedded_comments(source))
        legacy = '??/\n* legal text */'
        self.assertEqual(embedded_comments(legacy), legacy)

    def test_groups_documents_by_software_without_version_or_hash_gates(self):
        value = self.write()
        self.assertEqual(set(value), {'components'})
        components = {item['id']: item for item in value['components']}
        self.assertEqual(set(components), {'llama-cpp-browser-core', 'llama.cpp', 'nlohmann-json'})
        self.assertEqual(len(components['nlohmann-json']['notices']), 2)
        self.assertIsNone(components['nlohmann-json']['version'])
        self.assertIn('sourcePath', components['nlohmann-json']['notices'][1])
        original = (self.root / 'licenses/embedded/vendor/nlohmann/json.hpp.txt').read_text()
        self.assertIn('int implementation', original)
        self.assertNotIn('int implementation', (self.root / components['nlohmann-json']['notices'][1]['path']).read_text())
        self.validate()

    def test_is_byte_reproducible_and_supports_more_software_without_a_fixed_count(self):
        self.write(); before = (self.root / INDEX).read_bytes()
        self.write(); self.assertEqual(before, (self.root / INDEX).read_bytes())
        put(self.root, 'licenses/0/vendor/new-library/LICENSE', 'New library notice')
        self.write(); self.validate()
        self.assertNotEqual(before, (self.root / INDEX).read_bytes())

    def test_unknown_hierarchies_need_producer_review(self):
        put(self.root, 'licenses/0/unexpected-tree/library/LICENSE', 'Notice')
        with self.assertRaisesRegex(ValueError, 'Unclassified'): self.write()

    def test_missing_duplicate_and_reassigned_documents_fail_even_without_file_hashes(self):
        value = self.write()
        for mutation in ('missing', 'duplicate', 'owner'):
            data = json.loads(json.dumps(value))
            if mutation == 'missing': data['components'].pop()
            elif mutation == 'duplicate': data['components'].append(data['components'][0])
            else: data['components'][0]['name'] = 'Wrong owner'
            put(self.root, INDEX, json.dumps(data))
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'differs'): self.validate()

    def test_changed_extracted_text_and_unindexed_files_fail(self):
        value = self.write()
        extracted = next(n['path'] for c in value['components'] for n in c['notices'] if 'sourcePath' in n)
        put(self.root, extracted, 'Missing attribution')
        with self.assertRaisesRegex(ValueError, 'differs'): self.validate()
        self.write()
        put(self.root, 'licenses/0/NOTICE', 'New notice')
        with self.assertRaisesRegex(ValueError, 'differs'): self.validate()

    def test_empty_unknown_embedded_and_linked_originals_fail(self):
        for text in ('', '  \n'):
            put(self.root, 'licenses/0/LICENSE', text)
            with self.assertRaisesRegex(ValueError, 'Empty'): self.write()
        put(self.root, 'licenses/0/LICENSE', 'Notice')
        put(self.root, 'licenses/embedded/unreviewed.h.txt', '// Unknown owner')
        with self.assertRaisesRegex(ValueError, 'Unclassified'): self.write()
        (self.root / 'licenses/embedded/unreviewed.h.txt').unlink()
        (self.root / 'licenses/0/linked').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Linked'): self.write()

    def test_refuses_linked_output_index(self):
        target = put(self.root, 'untouched', 'original')
        (self.root / INDEX).symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Linked'): self.write()
        self.assertEqual(target.read_text(), 'original')

    def test_both_runtime_roots_and_nested_package_copies_are_independent(self):
        self.assertNotEqual(owner('emscripten', 'node_modules/source-map/LICENSE')[0], owner('emscripten', 'node_modules/tool/node_modules/source-map/LICENSE')[0])
        self.assertEqual(owner('emscripten', 'node_modules/@scope/lib/LICENSE')[1], '@scope/lib')
        self.assertNotEqual(owner('ggml', 'LICENSE')[0], owner('stable-diffusion.cpp', 'ggml/LICENSE')[0])
        image = self.root / 'image'; put(image, 'LICENSE', 'Image core'); put(image, 'licenses/stable-diffusion/LICENSE', 'SD')
        put(image, 'licenses/ggml/LICENSE', 'ggml'); put(image, 'licenses/embedded/stb_image.h.txt', '/* SPDX-License-Identifier: MIT */\nint image;')
        write_index(image, runtime='stable-diffusion-cpp')
        validate_index(image, runtime='stable-diffusion-cpp')
        self.assertEqual(set(json.loads((image / INDEX).read_text())), {'components'})

    def test_paths_reject_traversal_absolute_windows_and_control_forms(self):
        for value in ('../x', '/x', 'a//b', 'a/./b', 'C:/x', 'a\\b', 'a\0b', ''):
            with self.subTest(path=value), self.assertRaisesRegex(ValueError, 'Unsafe'): safe_path(value)


class EmbeddedComments(unittest.TestCase):
    def test_all_comments_not_just_ones_with_license_keywords_are_preserved(self):
        source = '/* Author A */\n// Attribution continuation\n\nint code;\n/* Permission */'
        self.assertEqual(embedded_comments(source), '/* Author A */\n// Attribution continuation\n\n\n/* Permission */')

    def test_quotes_raw_strings_and_character_literals_cannot_hide_comments(self):
        source = 'const char *s = "/* not a comment */"; auto r=R"tag(// not a comment)tag"; char c=\'/\';\n/* Actual notice */'
        self.assertEqual(embedded_comments(source), '/* Actual notice */')

    def test_line_splicing_preserves_exact_original_comment_text(self):
        source = '/\\\n* Author *\\\n/\nint x;\n// Continued \\\nwithout keywords\nint y;'
        result = embedded_comments(source)
        self.assertIn('/\\\n* Author *\\\n/', result)
        self.assertIn('// Continued \\\nwithout keywords\n', result)
        self.assertNotIn('int x', result)
        self.assertNotIn('int y', result)

    def test_ambiguous_syntax_missing_comments_and_string_notices_use_full_source(self):
        for source in ['int x;', '/* unterminated', '"unterminated\n/* notice */', 'R"oops(unterminated', '/* A */ char *s="Copyright 2026 B";']:
            with self.subTest(source=source): self.assertEqual(embedded_comments(source), source)

    def test_crlf_and_no_final_newline_survive(self):
        source = '// A\r\n// B\r\nint x;\r\n/* C */'
        self.assertEqual(embedded_comments(source), '// A\r\n// B\r\n\n\n/* C */')


if __name__ == '__main__': unittest.main()
