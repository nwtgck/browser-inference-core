"""Request portable S16 for guarded ordinary Q8/Q4 matvecs with scoped fallback."""
from pathlib import Path
import shutil
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy
import prepare_webgpu_q8_subgroup as q8_subgroup

BASE = Path('ggml/src/ggml-webgpu')
CONTRACT_INPUTS = dict(q8_subgroup.CONTRACT_INPUTS)
REVIEWED_DAWN_HEADERS = dict(tensor_copy.DAWN_HEADERS)
PATCHES = {False: ('ggml-webgpu-subgroup-size-control.patch',
         '11adb69a127b41fd6d22aea4cf2c63af0b5f55d40072b55c1c80fb3144a49c7e'),
 True: ('ggml-webgpu-subgroup-size-control-profile.patch',
        '79c6cad72257e05ae1c00bbdfaebbb678eee9057a932dc3b2db76b640b2fa85b')}
SHARED_PATCH = ('ggml-webgpu-subgroup-size-control-shared.patch',
 'bceb8c48605ffd4a9da75536fffffb1595d9314e0330edecf5aaabaf09663bdb')
INPUTS = {False: {'ggml-webgpu.cpp': '57d577c17ef1b87c088005defa103906ec124ce93b2429ace86e88028b638cbf',
         'ggml-webgpu-shader-lib.hpp': 'e0b874393679689e988471b83d7dff7926b9b28e68a3dc71c46d06df178fe1b9'},
 True: {'ggml-webgpu.cpp': 'e3bce48df987e216db5094bfb9eaf56f53fdaabb75f06cb9ad2cabbda24e000d',
        'ggml-webgpu-shader-lib.hpp': 'f83bd98f1a0b31caa5b6bb1fdbb46d44dc44066e98d367306feaf3005ee01b23'}}
OUTPUTS = {False: {'ggml-webgpu.cpp': '708de1eb3c267527d65cdb097a06d2ae71437f2ce4880071de42acbe098ffc55',
         'ggml-webgpu-shader-lib.hpp': '4947500704450e61424926a34ea69346b70900944896f5e7495eeed0b4d69725'},
 True: {'ggml-webgpu.cpp': '0d7f1dcb2ad06124a83fa6c33f2b388418268ba0d26d6d12262ccb299b860445',
        'ggml-webgpu-shader-lib.hpp': '91eb008a1ca86b10143027e5080b4ca0469512d3625d4028984e11e89e3cf716'}}
SHARED_INPUTS = {'ggml-webgpu-q8-subgroup-shaders.hpp': 'b1b86048fbb6b9a58834a4b5ee5552ac2704641cbf4db3c996c1a85674ea302b',
 'ggml-webgpu-batch-readback.hpp': '62a542146fb2e57cb1b9539d4fd0cde2a3f73c00ca7893600fbc6401430bf24a'}
GENERATED_OUTPUTS = {'ggml-webgpu-optional-pipeline.hpp': 'f2645457e2dc70b592a34b5d35262c26a60d3f8b0c7d2e3570eb1ee53a201f5e',
 'ggml-webgpu-q8-subgroup-shaders.hpp': '3db91d6d9aeef1df96cb9d55e8109001eb585c172b94d80d7556baaeae395a40',
 'ggml-webgpu-batch-readback.hpp': 'a2e07803f52f53f935f7ac33337e59abbe970d991babbe026b0ea00fd3cc0e1a'}
HELPER_NAME = 'ggml-webgpu-optional-pipeline.hpp'
EMBEDDED_HEADER_NAME = q8_subgroup.HEADER_NAME


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
    for name, expected in (SHARED_PATCH, PATCHES[gpu_profile]):
        path = patch_root / name
        if tensor_copy.digest(path) != expected:
            raise ValueError('Subgroup-size-control patch identity changed: ' + name)
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(path.resolve())],
                           cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected subgroup-size-control source: ' + name)
    for name, expected in GENERATED_OUTPUTS.items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected subgroup-size-control generated output: ' + name)
    return {name: work / BASE / name for name in GENERATED_OUTPUTS}
