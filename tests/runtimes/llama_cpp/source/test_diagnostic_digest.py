"""Compile the complete owned SHA-256 header; no upstream text extraction."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from build_tools.project import runtime_root


@unittest.skipUnless(shutil.which('c++'), 'A native C++ compiler is required')
class DiagnosticDigest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix='owned-diagnostic-digest-')
        cls.addClassCleanup(temporary.cleanup)
        cls.work = Path(temporary.name)
        source = cls.work / 'digest.cpp'
        # Include the production header itself. Test chunk boundaries in the
        # actual functions, rather than copying a digest implementation here.
        source.write_text(r'''#include "ggml-webgpu-q8-diagnostic-sha256.hpp"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const size_t stride = std::strtoull(argv[1], nullptr, 10);
    if (!stride) return 2;
    const std::string input{std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
    ggml_webgpu_q8_digest::sha256_ctx context;
    ggml_webgpu_q8_digest::sha256_init(context);
    ggml_webgpu_q8_digest::sha256_update(context, input.data(), 0);
    for (size_t offset = 0; offset < input.size();) {
        const size_t count = std::min(stride, input.size() - offset);
        ggml_webgpu_q8_digest::sha256_update(context, input.data() + offset, count);
        ggml_webgpu_q8_digest::sha256_update(context, input.data() + offset, 0);
        offset += count;
    }
    uint8_t digest[32];
    ggml_webgpu_q8_digest::sha256_final(context, digest);
    for (uint8_t value : digest) std::cout << std::hex << std::setfill('0') << std::setw(2) << unsigned(value);
    return 0;
}
''')
        cls.executables = []
        compiler = shutil.which('c++')
        for name, options in [('normal', ['-O2']), ('undefined', ['-O1', '-fsanitize=undefined', '-fno-sanitize-recover=all'])]:
            executable = cls.work / name
            subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', *options,
                            '-I' + str(runtime_root('llama-cpp') / 'native/diagnostics'),
                            str(source), '-o', str(executable)], check=True, capture_output=True)
            cls.executables.append(executable)

    def test_known_vectors_padding_boundaries_and_incremental_updates(self):
        cases = [b'', b'abc', b'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq', b'a' * 1_000_000]
        cases += [bytes((i * 37 + 11) % 256 for i in range(length))
                  for length in [1, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 4097]]
        for executable in self.executables:
            for content in cases:
                for stride in [1, 7, 64, max(1, len(content))]:
                    with self.subTest(compiler=executable.name, bytes=len(content), stride=stride):
                        result = subprocess.run([str(executable), str(stride)], input=content,
                                                capture_output=True, check=True)
                        self.assertEqual(result.stdout.decode(), hashlib.sha256(content).hexdigest())


if __name__ == '__main__':
    unittest.main()
