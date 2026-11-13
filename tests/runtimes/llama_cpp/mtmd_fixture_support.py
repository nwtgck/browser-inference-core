"""Small exact contracts for testing the real coherent mtmd preparer."""
import difflib
import hashlib
from pathlib import Path
from unittest.mock import patch
from build_tools.runtimes.llama_cpp.sources import mtmd as mtmd

ORIGINAL = 'before\noriginal\nafter\n'
CHANGED = 'before\npatched\nafter\n'


class CheckedMtmdFixture:
    def __init__(self, source: Path, patch_root: Path):
        self.source, self.patch_root = source, patch_root
        tree = source / 'tools/mtmd'
        tree.mkdir(parents=True, exist_ok=True)
        for name in ('mtmd-model.cpp', 'mtmd-model.h', 'mtmd-subcontext.cpp', 'mtmd.cpp', 'mtmd-audio.cpp'):
            (tree / name).write_text(ORIGINAL)
        (tree / 'models').mkdir()
        (tree / 'models/model.cpp').write_text('unchanged model builder\n')
        (tree / 'CMakeLists.txt').write_text(
            'set(MTMD_CORE_SOURCES mtmd.cpp mtmd-model.cpp mtmd-subcontext.cpp mtmd-audio.cpp)\n')
        self.contracts = {'tools/mtmd/' + p.name: self.digest(p.read_bytes()) for p in tree.glob('*') if p.is_file()}
        patch_root.mkdir(parents=True, exist_ok=True)
        self.stages = {}
        for name, files in (
            ('mtmd-vision-bf16-runtime.patch', ('mtmd.cpp', 'mtmd-model.h')),
            ('mtmd-webgpu-bf16.patch', ('mtmd-model.cpp',)),
            ('mtmd-non-pthread-threadpool.patch', ('mtmd-subcontext.cpp',)),
        ):
            raw = ''.join(''.join(difflib.unified_diff(ORIGINAL.splitlines(True), CHANGED.splitlines(True),
                               fromfile='a/' + f, tofile='b/' + f)) for f in files).encode()
            (patch_root / name).write_bytes(raw)
            self.stages[name] = {'sha256': self.digest(raw), 'outputs': {f: self.digest(CHANGED.encode()) for f in files}}
        self.outputs = {mode: {name: self.digest((ORIGINAL if name == 'mtmd-model.cpp' and mode == 'cpu'
                                                 else CHANGED).encode())
                              for name in ('mtmd.cpp', 'mtmd-model.h', 'mtmd-model.cpp', 'mtmd-subcontext.cpp')}
                        for mode in ('cpu', 'webgpu')}

    @staticmethod
    def digest(raw):
        return hashlib.sha256(raw).hexdigest()

    def install(self, test):
        for name, values in (('CONTRACT_INPUTS', self.contracts), ('STAGES', self.stages), ('OUTPUTS', self.outputs)):
            context = patch.object(mtmd, name, values)
            context.start()
            test.addCleanup(context.stop)
