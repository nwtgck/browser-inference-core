"""Production classification coverage and typed ABI contract tests, without model/GPU I/O."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReviewCoverageTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory(prefix='lcb-bf16-coverage-')
        self.addCleanup(work.cleanup)
        self.work = Path(work.name)
        self.source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (self.source / 'tools/mtmd/clip-impl.h').is_file():
            self.skipTest('Initialize the pinned upstream checkout')
        self.original = (self.source / 'tools/mtmd/clip-impl.h').read_text()
        self.classification = (ROOT / 'bridge/mtmd-vision-bf16-classification.h').read_text()
        (self.work / 'probe.cpp').write_text('#include "mtmd-vision-bf16-classification.h"\nint main() { return lcb_mtmd_review_projector(PROJECTOR_TYPE_GEMMA4UV); }\n')

    def test_each_supported_compiler_rejects_unreviewed_enum_drift_and_missing_cases(self):
        compilers = [compiler for compiler in ('g++', 'clang++') if shutil.which(compiler)]
        if not compilers:
            self.skipTest('GCC or Clang required')
        variants = {
            'normal': (self.original, self.classification, True),
            'added': (self.original.replace('enum projector_type {', 'enum projector_type {\n    PROJECTOR_TYPE_FUTURE = 9917,'), self.classification, False),
            'missing-case': (self.original, self.classification.replace('    case PROJECTOR_TYPE_GEMMA4UV:\n', ''), False),
            'renamed': (self.original.replace('PROJECTOR_TYPE_GEMMA4UV', 'PROJECTOR_TYPE_FUTURE'), self.classification, False),
            'duplicate-case': (self.original, self.classification.replace('    case PROJECTOR_TYPE_GEMMA4UV:', '    case PROJECTOR_TYPE_GEMMA4UV:\n    case PROJECTOR_TYPE_GEMMA4UV:'), False),
        }
        for compiler in compilers:
            for name, (header, classification, should_pass) in variants.items():
                with self.subTest(compiler=compiler, mutation=name):
                    (self.work / 'clip-impl.h').write_text(header)
                    (self.work / 'mtmd-vision-bf16-classification.h').write_text(classification)
                    result = subprocess.run([compiler, '-std=c++17', '-fsyntax-only',
                        '-I' + str(self.work), '-I' + str(ROOT / 'bridge'),
                        '-I' + str(self.source / 'tools/mtmd'), '-I' + str(self.source / 'include'),
                        '-I' + str(self.source / 'ggml/include'), str(self.work / 'probe.cpp')],
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
                    self.assertEqual(result.returncode == 0, should_pass, result.stdout)
                    if name in ('added', 'missing-case'):
                        self.assertIn('switch', result.stdout)

    def test_real_classification_and_diagnostics_execute_without_gpu_or_model(self):
        if not shutil.which('g++'):
            self.skipTest('GCC required')
        binary = self.work / 'diagnostics'
        subprocess.run(['g++', '-std=c++17', '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections', '-fsanitize=undefined', '-fno-sanitize-recover=all',
            '-I' + str(ROOT / 'bridge'), '-I' + str(self.source / 'tools/mtmd'),
            '-I' + str(self.source / 'include'), '-I' + str(self.source / 'ggml/include'),
            str(ROOT / 'tests/mtmd-vision-bf16-diagnostics.cpp'), '-o', str(binary)], check=True, timeout=30)
        subprocess.run([str(binary)], check=True, timeout=30)

    def test_public_header_is_c_compatible_and_is_an_additive_record_boundary(self):
        if not shutil.which('cc'):
            self.skipTest('C compiler required')
        code = self.work / 'public.c'
        code.write_text('#include "mtmd-vision-bf16-diagnostics.h"\n'
                        '_Static_assert(sizeof(((struct mtmd_vision_bf16_statistics *)0)->source_bf16_bytes) == 8, "u64");\n'
                        'int main(void) { return sizeof(struct mtmd_vision_bf16_load_report) != 24; }\n')
        subprocess.run(['cc', '-std=c11', '-fsyntax-only', '-I' + str(ROOT / 'bridge'),
            '-I' + str(self.source / 'tools/mtmd'), '-I' + str(self.source / 'include'),
            '-I' + str(self.source / 'ggml/include'), str(code)], check=True, timeout=30)
        public = (ROOT / 'bridge/mtmd-vision-bf16.h').read_text()
        self.assertNotIn('STORAGE_AUTO', public)
        self.assertNotIn('diagnostics', (self.source / 'tools/mtmd/mtmd.h').read_text())


if __name__ == '__main__':
    unittest.main()
