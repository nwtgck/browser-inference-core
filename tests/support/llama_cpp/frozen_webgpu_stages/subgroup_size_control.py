"""Request portable S16 for guarded ordinary Q8/Q4 matvecs with scoped fallback."""
from pathlib import Path
import shutil
import subprocess
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
from tests.support.llama_cpp.frozen_webgpu_stages import q8_subgroup
from tests.support.llama_cpp.frozen_webgpu_stages import aligned_word_load
BASE = Path('ggml/src/ggml-webgpu')
CONTRACT_INPUTS = dict(q8_subgroup.CONTRACT_INPUTS)
REVIEWED_DAWN_HEADERS = dict(tensor_copy.DAWN_HEADERS)
PATCHES = {False: ('ggml-webgpu-subgroup-size-control.patch',
         '11adb69a127b41fd6d22aea4cf2c63af0b5f55d40072b55c1c80fb3144a49c7e'),
 True: ('ggml-webgpu-subgroup-size-control-profile.patch',
        '79c6cad72257e05ae1c00bbdfaebbb678eee9057a932dc3b2db76b640b2fa85b')}
SHARED_PATCH = ('ggml-webgpu-subgroup-size-control-shared.patch',
 'e3020bf2c241b92a9b63669ebde89590319585dde6c2dc9d4b53bdc6de8b01d8')
INPUTS = {False: {'ggml-webgpu.cpp': '296f4068e349aaa2fae83bc64a2357497ffe83e62b919a7416469c0cfaa4684a',
         'ggml-webgpu-shader-lib.hpp': 'e90bac8b868cd3674c69903cc9bf5bac1327a236fb8984d50c340f2c6d73a67d'},
 True: {'ggml-webgpu.cpp': 'b08dffb6fe1450aaf7cec68203a7ca22f082380a6488ab9195ee26ed407b9f6c',
        'ggml-webgpu-shader-lib.hpp': '9959ab22cb286927cbc3c289914cbfc0c81047324d5a7d17f37cb3c9114e9371'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': 'ea4f876b27b7a1664364312374b41acf637cf16ba5267b651cfa7cde93665b57',
         'ggml-webgpu-shader-lib.hpp': 'cc2447023811811fd492d99e9a90b9e83c3a974a2e604a644b77e9ea30177a6c'},
 True: {'ggml-webgpu.cpp': 'ae59490ebba9058d3000e5a9bd57854b0319f3d453e499ddbdd2adcbd9080ad9',
        'ggml-webgpu-shader-lib.hpp': 'f5a07a04ad663e748e25cfa5dde4a97b298609ce08b07b4ceaacd60618f0490e'}}
SHARED_INPUTS = {'ggml-webgpu-batch-readback.hpp': '62a542146fb2e57cb1b9539d4fd0cde2a3f73c00ca7893600fbc6401430bf24a'}
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
