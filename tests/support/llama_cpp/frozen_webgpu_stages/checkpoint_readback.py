"""Compose bounded blocking checkpoint reads through an optional registry hook."""
from pathlib import Path
import shutil
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
CONTEXT_PATH = 'src/llama-context.cpp'
PATCH_NAME = 'ggml-webgpu-checkpoint-batch-readback.patch'
PATCH_SHA256 = '1a6975a2e4fa7b1c88dbf111c127fe00a9b98c5b084309a84086d6bcf7c7a753'
CONTRACT_INPUTS = {'ggml/include/ggml-backend.h': '34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a',
 'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
 'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
 'ggml/src/ggml-backend-reg.cpp': '1436b46bd1e8355345d454baa4b40069be73e9cb09a3a188d32426ea1e1037fe',
 'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
 'ggml/src/ggml-webgpu/CMakeLists.txt': 'aa19125326050c3e0ae768f31f43a61b8150805a4b330853662b955e8ef34b2a',
 'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
 'ggml/src/ggml.c': 'c0e81f0ba55ad76020ada29b01385a7a742c800971e7c974c425a08839750b49',
 'include/llama.h': 'f07ffbabd3fbe9ef21d97b4d4ce8328d9c2a123ccbf5266f04a3146d70ce4f91',
 'src/CMakeLists.txt': '802f9f3eced47f2396f2e8b7534050361e72882cffe2619fed6653d11b1f474d',
 'src/llama-context.cpp': '4d137ded28efc44c8b471f6c056bee9444a24deb7b42d1e04c64728c25e5c822',
 'src/llama-context.h': 'c0ed1f0e8b47d2f001a4a8ba61d01f05b16c2972d693780ac1cc09a6d9a05d88',
 'src/llama-io.h': 'cc5bd88ceeb9c859384d1cece26fce2ed73bcf16eb6a5fa4b571dfa2c06613fe'}
REVIEWED_DAWN_HEADERS = dict(tensor_copy.DAWN_HEADERS)
INPUTS = {False: {'ggml-webgpu-shader-lib.hpp': 'e90bac8b868cd3674c69903cc9bf5bac1327a236fb8984d50c340f2c6d73a67d',
         'ggml-webgpu.cpp': 'c977f258c8ec2b98e07f69388e1bc22a00646f76c559ce5afee533d7ff6a0c76'},
 True: {'ggml-webgpu-shader-lib.hpp': '9959ab22cb286927cbc3c289914cbfc0c81047324d5a7d17f37cb3c9114e9371',
        'ggml-webgpu.cpp': '62e142cdac82c751f3c758e264c3750ea420429f58a0fa883414322756aa7268'}}
OUTPUTS = {False: {'ggml-webgpu-shader-lib.hpp': 'e90bac8b868cd3674c69903cc9bf5bac1327a236fb8984d50c340f2c6d73a67d',
         'ggml-webgpu.cpp': '296f4068e349aaa2fae83bc64a2357497ffe83e62b919a7416469c0cfaa4684a'},
 True: {'ggml-webgpu-shader-lib.hpp': '9959ab22cb286927cbc3c289914cbfc0c81047324d5a7d17f37cb3c9114e9371',
        'ggml-webgpu.cpp': 'b08dffb6fe1450aaf7cec68203a7ca22f082380a6488ab9195ee26ed407b9f6c'}}
GENERATED_OUTPUTS = {'ggml/src/ggml-backend-tensor-get-batch.h': '01c4b471ade51ddb6fd91e1d474e00943813f2d598e0192ab7d8caf571c3978a',
 'ggml/src/ggml-webgpu/ggml-webgpu-batch-readback-geometry.hpp': 'f6e9cd0d46a7c32a112f870a705c735be070224f586ae97a2d513a20e75e7f0a',
 'ggml/src/ggml-webgpu/ggml-webgpu-batch-readback.hpp': '62a542146fb2e57cb1b9539d4fd0cde2a3f73c00ca7893600fbc6401430bf24a',
 'src/llama-context.cpp': '6dd7a06d3eb939618d08d50054c6963e7c4806828e7c686bf4ab3daa43ccbd2c'}
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
