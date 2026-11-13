"""Updated source identity regression retained separately from frozen comparisons."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import runtime_root
from tests.support import legacy_webgpu_source as source
from tests.support.llama_cpp.webgpu_stages import aligned_word_load as word
from tests.runtimes.llama_cpp.webgpu_prefix_context import without_checkpoint_readback, without_shared_word_load
ROOT = runtime_root('llama-cpp')

class UpdatedAlignedWordProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = without_checkpoint_readback(source)
        prefix.__enter__()
        cls.addClassCleanup(prefix.__exit__, None, None, None)
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / word.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE to the updated pinned source')
        cls.temp = tempfile.TemporaryDirectory(prefix='aligned-word-load-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with without_shared_word_load(source):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile).parent
        cls.original = (cls.upstream / word.SOURCE_PATH).read_text()
        cls.candidate = (cls.final[False] / word.HEADER_NAME).read_text()
        shaders = cls.upstream / word.BASE / 'wgsl-shaders'
        cls.canonical = cls.directory / 'canonical-original.hpp'
        subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                        '--output_file', str(cls.canonical)], check=True, capture_output=True)

    def test_provenance_input_identities_match_actual_generated_bytes(self):
        inputs = [('canonical', self.canonical, word.CANONICAL_INPUT_IDENTITY)]
        for profile in (False, True):
            identities = word.CONSUMER_INPUT_IDENTITIES[profile]
            self.assertEqual(set(identities),
                             {word.HEADER_NAME, *word.EMBEDDING_INPUTS, 'ggml-webgpu-shader-lib.hpp'})
            for name, expected in identities.items():
                path = self.canonical if name == word.HEADER_NAME else self.before[profile] / name
                inputs.append((f'{profile}/{name}', path, expected))
        for label, path, expected in inputs:
            with self.subTest(input=label):
                raw = path.read_bytes()
                self.assertEqual({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}, expected)
