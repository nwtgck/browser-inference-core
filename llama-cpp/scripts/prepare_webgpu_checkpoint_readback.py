"""Compose bounded blocking checkpoint reads through an optional registry hook."""
from pathlib import Path
import shutil
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
CONTEXT_PATH = 'src/llama-context.cpp'
PATCH_NAME = 'ggml-webgpu-checkpoint-batch-readback.patch'
PATCH_SHA256 = 'f9ea45e231da708d8e02c7bb4107bc35e1d2454cab37f97303d676b618ace5d1'
CONTRACT_INPUTS = {'ggml/include/ggml-backend.h': '0e5e8e1f16d11ebbf501e38bc6705d79eee819b5f0b17476533f2100ed9674e2',
 'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
 'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
 'ggml/src/ggml-backend-reg.cpp': '1436b46bd1e8355345d454baa4b40069be73e9cb09a3a188d32426ea1e1037fe',
 'ggml/src/ggml-backend.cpp': '79ade6b83b3d042e43e035fa81d2c672211a552af9af38e410a047e4ba2723a0',
 'ggml/src/ggml-webgpu/CMakeLists.txt': 'aa19125326050c3e0ae768f31f43a61b8150805a4b330853662b955e8ef34b2a',
 'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
 'ggml/src/ggml.c': '45e90696c4a766d5747f3443f817cd3be11f0110cf93cd16926698c0f49ed083',
 'include/llama.h': 'c0854771d3760a41642c4e59429c5b3b9fc774887c59d37eeadc4527681732cd',
 'src/CMakeLists.txt': '802f9f3eced47f2396f2e8b7534050361e72882cffe2619fed6653d11b1f474d',
 'src/llama-context.cpp': 'f76532e7d7c510426bb4f96f1c1bcf4ca5321790b07b56ce9446d793feef8d35',
 'src/llama-context.h': 'c5123eb0b1b4426737a70a3a616d8ba48b646143b596fa3f42d20ab57b2fb9e5',
 'src/llama-io.h': 'cc5bd88ceeb9c859384d1cece26fce2ed73bcf16eb6a5fa4b571dfa2c06613fe'}
REVIEWED_DAWN_HEADERS = dict(tensor_copy.DAWN_HEADERS)
INPUTS = {False: {'ggml-webgpu-shader-lib.hpp': '87e20119da7064dda8210c6130b1e43a4aaf83c644ae5e58a691009f709405a8',
         'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335'},
 True: {'ggml-webgpu-shader-lib.hpp': '01f9e59fe52a3f1865398d96478a7ce9bafb174c320809ede04384ba269acd0a',
        'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791'}}
OUTPUTS = {False: {'ggml-webgpu-shader-lib.hpp': '87e20119da7064dda8210c6130b1e43a4aaf83c644ae5e58a691009f709405a8',
         'ggml-webgpu.cpp': '57d577c17ef1b87c088005defa103906ec124ce93b2429ace86e88028b638cbf'},
 True: {'ggml-webgpu-shader-lib.hpp': '01f9e59fe52a3f1865398d96478a7ce9bafb174c320809ede04384ba269acd0a',
        'ggml-webgpu.cpp': 'e3bce48df987e216db5094bfb9eaf56f53fdaabb75f06cb9ad2cabbda24e000d'}}
GENERATED_OUTPUTS = {'ggml/src/ggml-backend-tensor-get-batch.h': '01c4b471ade51ddb6fd91e1d474e00943813f2d598e0192ab7d8caf571c3978a',
 'ggml/src/ggml-webgpu/ggml-webgpu-batch-readback-geometry.hpp': 'f6e9cd0d46a7c32a112f870a705c735be070224f586ae97a2d513a20e75e7f0a',
 'ggml/src/ggml-webgpu/ggml-webgpu-batch-readback.hpp': '62a542146fb2e57cb1b9539d4fd0cde2a3f73c00ca7893600fbc6401430bf24a',
 'src/llama-context.cpp': 'fd7523d1d0900505f12489604dcf9671bac9e2195f6f61b56ba2a61b3478370d'}
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
                       cwd=work, env=env, check=True, capture_output=True)
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected checkpoint readback output: ' + name)
    for name, expected in GENERATED_OUTPUTS.items():
        if tensor_copy.digest(work / name) != expected:
            raise ValueError('Unexpected checkpoint readback generated output: ' + name)
    return {Path(name).name: work / name for name in GENERATED_OUTPUTS}
