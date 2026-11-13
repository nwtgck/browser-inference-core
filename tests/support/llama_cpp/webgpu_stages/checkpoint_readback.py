"""Compose bounded blocking checkpoint reads through an optional registry hook."""
from pathlib import Path
import shutil
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
CONTEXT_PATH = 'src/llama-context.cpp'
PATCH_NAME = 'ggml-webgpu-checkpoint-batch-readback.patch'
PATCH_SHA256 = 'b5c8d9695c098f6fa4262233976df6468f3a0e195f137f81f4be6f709324c994'
CONTRACT_INPUTS = {'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
 'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
 'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa',
 'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
 'ggml/src/ggml-backend-reg.cpp': '0a23a627cec97926fb1fbb2b1bf4fa7fed4900aa21b0215302eced797231578b',
 'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8',
 'ggml/src/ggml-webgpu/CMakeLists.txt': 'aa19125326050c3e0ae768f31f43a61b8150805a4b330853662b955e8ef34b2a',
 'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
 'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc',
 'include/llama.h': '19ad09e336dd61317ff816bbf448e9b76c1d925d430191429c17df4679f9a1a4',
 'src/CMakeLists.txt': '3ad358fcd78137c551bdcd0b79698881d1515026ff2585693e898e3b85b3bec3',
 'src/llama-context.cpp': '6b2a143eee8d53cdb559eb18b5612028bdf8545ab832cfdca2c2fd87c9c00309',
 'src/llama-context.h': '287d337d23520c56d3d8bcc31087b34d55b066d3a5fec574b7f44f9f082d7a5e',
 'src/llama-io.h': 'cc5bd88ceeb9c859384d1cece26fce2ed73bcf16eb6a5fa4b571dfa2c06613fe'}
REVIEWED_DAWN_HEADERS = dict(tensor_copy.DAWN_HEADERS)
INPUTS = {False: {'ggml-webgpu-shader-lib.hpp': 'c88729ac0b397d8c6980a4e31d5fa50348332f09264512bf5ab14438f8b0a220',
         'ggml-webgpu.cpp': '23873d267578c6bce435eff0f7643f2853f705ad0b46b3ed8e93cbaada2bb8a2'},
 True: {'ggml-webgpu-shader-lib.hpp': 'dfc693145993e435cbec068b121580ae7e0a816ab652bfb4daf605ab1be58d16',
        'ggml-webgpu.cpp': 'de513ccb952c3cc1da2e42f45f4ed1acd9433f6bdea3e38a61f951f437b9c79c'}}
OUTPUTS = {False: {'ggml-webgpu-shader-lib.hpp': 'c88729ac0b397d8c6980a4e31d5fa50348332f09264512bf5ab14438f8b0a220',
         'ggml-webgpu.cpp': '24e0e208dbc2d444899ea908e226faa4e5e97010bd062d6eaf8d4f285ce3b4c4'},
 True: {'ggml-webgpu-shader-lib.hpp': 'dfc693145993e435cbec068b121580ae7e0a816ab652bfb4daf605ab1be58d16',
        'ggml-webgpu.cpp': '0094c6b3d16d14a9f9b393ec169fab90d2adab22f2fe2d732af020f9761e91f3'}}
GENERATED_OUTPUTS = {'ggml/src/ggml-backend-tensor-get-batch.h': '01c4b471ade51ddb6fd91e1d474e00943813f2d598e0192ab7d8caf571c3978a',
 'ggml/src/ggml-webgpu/ggml-webgpu-batch-readback-geometry.hpp': 'f6e9cd0d46a7c32a112f870a705c735be070224f586ae97a2d513a20e75e7f0a',
 'ggml/src/ggml-webgpu/ggml-webgpu-batch-readback.hpp': '62a542146fb2e57cb1b9539d4fd0cde2a3f73c00ca7893600fbc6401430bf24a',
 'src/llama-context.cpp': '1dfe5e2cfc32369e17484ce7efba5926be9f72779c7d6a72f1c5291ef39e1c6d'}
GENERATED_NAMES = tuple(Path(name).name for name in GENERATED_OUTPUTS)


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> dict[str, Path]:
    source, work = source.resolve(), work.resolve()
    if source == work or source in work.parents or work in source.parents:
        raise ValueError('Checkpoint readback overlay must be outside the upstream source tree')
    for relative, expected in CONTRACT_INPUTS.items():
        original = source / relative
        if original.is_symlink() or tensor_copy.digest(original) != expected:
            raise ValueError('Checkpoint readback contract changed: ' + relative)
    for name, expected in INPUTS[gpu_profile].items():
        target = work / BASE / name
        if not target.resolve().is_relative_to(work) or target.is_symlink():
            raise ValueError('Checkpoint readback input must remain inside the isolated work directory')
        if tensor_copy.digest(target) != expected:
            raise ValueError('Checkpoint readback composition input changed: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Checkpoint readback patch identity changed')
    for name in GENERATED_OUTPUTS:
        target = work / name
        if not target.resolve().is_relative_to(work) or target.is_symlink():
            raise ValueError('Checkpoint readback output must remain inside the isolated work directory')
        if target.exists() or (name != CONTEXT_PATH and (source / name).exists()):
            raise ValueError('Checkpoint readback would replace an unreviewed source: ' + name)
    context = work / CONTEXT_PATH
    context.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / CONTEXT_PATH, context)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected checkpoint readback output: ' + name)
    for name, expected in GENERATED_OUTPUTS.items():
        if tensor_copy.digest(work / name) != expected:
            raise ValueError('Unexpected checkpoint readback generated output: ' + name)
    return {Path(name).name: work / name for name in GENERATED_OUTPUTS}
