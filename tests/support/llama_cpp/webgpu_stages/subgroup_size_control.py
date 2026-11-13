"""Request portable S16 for guarded ordinary Q8/Q4 matvecs with scoped fallback."""
from pathlib import Path
import shutil
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
from tests.support.llama_cpp.webgpu_stages import q8_subgroup
from tests.support.llama_cpp.webgpu_stages import aligned_word_load
BASE = Path('ggml/src/ggml-webgpu')
CONTRACT_INPUTS = dict(q8_subgroup.CONTRACT_INPUTS)
REVIEWED_DAWN_HEADERS = dict(tensor_copy.DAWN_HEADERS)
PATCHES = {False: ('ggml-webgpu-subgroup-size-control.patch',
         '2a75159bc6f6afca14a216e089616f2f81aa37295f0da49a8ae92121de1b5bf0'),
 True: ('ggml-webgpu-subgroup-size-control-profile.patch',
        '08d8d4cde097fc4ee418e8ed1ae59825198f51ba1750b166ebe06a44dd806859')}
SHARED_PATCH = ('ggml-webgpu-subgroup-size-control-shared.patch',
 'bceb8c48605ffd4a9da75536fffffb1595d9314e0330edecf5aaabaf09663bdb')
INPUTS = {False: {'ggml-webgpu.cpp': '24e0e208dbc2d444899ea908e226faa4e5e97010bd062d6eaf8d4f285ce3b4c4',
         'ggml-webgpu-shader-lib.hpp': 'c88729ac0b397d8c6980a4e31d5fa50348332f09264512bf5ab14438f8b0a220'},
 True: {'ggml-webgpu.cpp': '0094c6b3d16d14a9f9b393ec169fab90d2adab22f2fe2d732af020f9761e91f3',
        'ggml-webgpu-shader-lib.hpp': 'dfc693145993e435cbec068b121580ae7e0a816ab652bfb4daf605ab1be58d16'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'd586843f24c47a12a29357c30b35b4d3f7b186f18af17bab222a52c5612a10c4',
         'ggml-webgpu-shader-lib.hpp': '87370c7273e37342a4e6ec580b3395c1a4c4e8c9e7e50d2a0e3b7fd1a4d2cb58'},
 True: {'ggml-webgpu.cpp': 'a62afc88802e9a84d80a8f03ac11ed50133f05841c25542d57672250f28474ad',
        'ggml-webgpu-shader-lib.hpp': '8543ff1b77cd69af4c0e8416ec5e89561b754f65a0b278df023cb20ee06cb1cc'}}
SHARED_INPUTS = {'ggml-webgpu-q8-subgroup-shaders.hpp': 'b1b86048fbb6b9a58834a4b5ee5552ac2704641cbf4db3c996c1a85674ea302b',
 'ggml-webgpu-batch-readback.hpp': '62a542146fb2e57cb1b9539d4fd0cde2a3f73c00ca7893600fbc6401430bf24a'}
GENERATED_OUTPUTS = {'ggml-webgpu-optional-pipeline.hpp': 'f2645457e2dc70b592a34b5d35262c26a60d3f8b0c7d2e3570eb1ee53a201f5e',
 'ggml-webgpu-batch-readback.hpp': 'a2e07803f52f53f935f7ac33337e59abbe970d991babbe026b0ea00fd3cc0e1a'}
HELPER_NAME = 'ggml-webgpu-optional-pipeline.hpp'
EMBEDDED_HEADER_NAME = q8_subgroup.HEADER_NAME
GENERATED_NAMES = (*GENERATED_OUTPUTS, EMBEDDED_HEADER_NAME)
SHADER_PATCH = ('ggml-webgpu-subgroup-size-control-wgsl.patch',
                '8691427c9cc0a962d76a46d3e2400efb5a88858a0b8d82ee394291dfaf9decc8')
# Keep the historical audit patch intact. Its C++ shader chunk hunks are
# superseded by the equivalent source patch and the unchanged generator.
SUPERSEDED_SHARED_PATH = str(BASE / EMBEDDED_HEADER_NAME)


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *,
            gpu_profile: bool, headers: dict[str, Path]) -> dict[str, Path]:
    work, source = work.resolve(), source.resolve()
    if source == work or source in work.parents or work in source.parents:
        raise ValueError('Subgroup-size-control overlay must be outside the upstream source tree')
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Subgroup-size-control upstream contract changed: ' + relative)
    for name, expected in INPUTS[gpu_profile].items():
        target = work / BASE / name
        if target.is_symlink() or not target.resolve().is_relative_to(work):
            raise ValueError('Subgroup-size-control input must remain inside the isolated work directory')
        if tensor_copy.digest(target) != expected:
            raise ValueError('Subgroup-size-control composition input changed: ' + name)
    if (work / BASE / HELPER_NAME).exists() or (source / BASE / HELPER_NAME).exists():
        raise ValueError('Subgroup-size-control helper would replace unreviewed source')
    for name, expected in SHARED_INPUTS.items():
        if name in headers and (headers[name].is_symlink() or not headers[name].resolve().is_relative_to(work)):
            raise ValueError('Subgroup-size-control shared input must remain inside the isolated work directory')
        if name not in headers or tensor_copy.digest(headers[name]) != expected:
            raise ValueError('Subgroup-size-control shared input changed: ' + name)
        target = work / BASE / name
        if headers[name] != target:
            if target.exists():
                raise ValueError('Subgroup-size-control generated header collision: ' + name)
            shutil.copyfile(headers[name], target)
    shader_inputs = work / aligned_word_load.DEDICATED_SHADERS[EMBEDDED_HEADER_NAME].directory
    patched_template = work / 'aligned-word-load-template' / aligned_word_load.SOURCE_PATH
    for name, expected in (SHARED_PATCH, PATCHES[gpu_profile], SHADER_PATCH):
        path = patch_root / name
        if tensor_copy.digest(path) != expected:
            raise ValueError('Subgroup-size-control patch identity changed: ' + name)
        exclude = ['--exclude=' + SUPERSEDED_SHARED_PATH] if name == SHARED_PATCH[0] else []
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *exclude, *flags, str(path.resolve())],
                           cwd=work, env=env, check=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected subgroup-size-control source: ' + name)
    for name, expected in GENERATED_OUTPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected subgroup-size-control generated output: ' + name)
    shutil.copyfile(work / BASE / 'wgsl-shaders' / q8_subgroup.SHADER_NAME, shader_inputs / q8_subgroup.SHADER_NAME)
    aligned_word_load.generate_canonical(source, patched_template, work / BASE / EMBEDDED_HEADER_NAME, inputs=shader_inputs)
    return {name: work / BASE / name for name in GENERATED_NAMES}
