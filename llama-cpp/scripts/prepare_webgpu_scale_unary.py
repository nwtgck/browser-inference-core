"""Source-bound contiguous F32 SCALE/UNARY fusion; upstream remains read-only."""
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-scale-unary.patch'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-scale-unary-shaders.hpp'
PATCH_SHA256 = '1c03ca5a6ce80bc01efd87d2f286430e5c1e648116328fb86046f227e58f7caa'
INPUTS = {'ggml-webgpu.cpp': 'b5ac93bbf834718ef60002d71b2a60d52dc605a3b64d5c90305599cc07f7d19e', 'ggml-webgpu-shader-lib.hpp': 'd0045d4cc1f4fc88a8ed6a24edbe64c6148f5d3eb2e70b7fcabb9346e17900a8'}
OUTPUTS = {'ggml-webgpu.cpp': 'df43feda75cf9f1485f177d88d94bd6a436e7d562df42ee941d97c2c8b0c5ed8', 'ggml-webgpu-shader-lib.hpp': '1ef8ab81da16dcbdf9c505142b33557cbba4b38f38f524cc37419119303e4dd1', 'ggml-webgpu-scale-unary.hpp': '78efb6f8ff9b31813cad1da78d5d97e3d38f2e44ba63dc7165d01aebdebe2898', 'wgsl-shaders/scale_unary.wgsl': '0039234c6e562a502569314c216622df26a9b6b04fc24a897c85b7c87bc97ca2'}
HEADER_SHA256 = '54e8ffbef83b37881f7561cd65c4295c71a8806d157ddf4345367e9cce9d8c9b'
CONTRACT_INPUTS = {'ggml/include/ggml.h': '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3', 'ggml/src/ggml.c': '42b238412cda218552812a37329fab95a2f94be6d8bff5139083450d432f796d', 'ggml/src/ggml-impl.h': '43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3', 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39', 'ggml/src/ggml-backend.cpp': '43dea13b13c2a15afd83f53515f3488c17072fc107a6ba4a5e0370f90b4fe7ff', 'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7', 'ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl': 'cee80d0720353f4ad24c8845db54000e6f9b1dd1d1c3585acce3f99b693b7a45', 'ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl': '7d4d66e2b954b5d1067447bb24082cf19f5fba7e021d7101193e3edc73cefbc1', 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5', 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea'}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Scale/unary contract changed: ' + relative)
    for relative, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Scale/unary composition input changed: ' + relative)
    for relative in ('ggml-webgpu-scale-unary.hpp', 'wgsl-shaders/scale_unary.wgsl'):
        if (source / BASE / relative).exists():
            raise ValueError('Upstream scale/unary source now exists: ' + relative)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Scale/unary patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for relative, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected scale/unary output: ' + relative)
    shaders = work / 'scale-unary-only-wgsl'
    shaders.mkdir()
    shutil.copyfile(work / BASE / 'wgsl-shaders/scale_unary.wgsl', shaders / 'scale_unary.wgsl')
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected scale/unary embedding identity')
    return {HEADER_NAME: header, 'ggml-webgpu-scale-unary.hpp': work / BASE / 'ggml-webgpu-scale-unary.hpp'}
