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
INPUTS = {'ggml-webgpu.cpp': 'ac00909a1706e65dcc11e18e6074dc05c7a9263fd17cdf48884732035de41fe1', 'ggml-webgpu-shader-lib.hpp': 'ca9e2701c3cab09631aafdb6cf8cf631742d87cc9eb02106941ca3ab031843f5'}
OUTPUTS = {'ggml-webgpu.cpp': 'd7363c0026d0dec265679d3cb1dddeafcf3f5049d5846eca124ec2c70615d17a', 'ggml-webgpu-shader-lib.hpp': '24fd12bea29e2bd0590ea80984cfb8c73553df1a4a9b7810edb5d191f0931ab3', 'ggml-webgpu-scale-unary.hpp': '78efb6f8ff9b31813cad1da78d5d97e3d38f2e44ba63dc7165d01aebdebe2898', 'wgsl-shaders/scale_unary.wgsl': '0039234c6e562a502569314c216622df26a9b6b04fc24a897c85b7c87bc97ca2'}
HEADER_SHA256 = '54e8ffbef83b37881f7561cd65c4295c71a8806d157ddf4345367e9cce9d8c9b'
CONTRACT_INPUTS = {'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101', 'ggml/src/ggml.c': 'c0e81f0ba55ad76020ada29b01385a7a742c800971e7c974c425a08839750b49', 'ggml/src/ggml-impl.h': 'c84c8f9e3c246504602f5cd388ba54e36691e46bf150e4d70ae857bb823c985f', 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39', 'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e', 'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3', 'ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl': 'cee80d0720353f4ad24c8845db54000e6f9b1dd1d1c3585acce3f99b693b7a45', 'ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl': '7d4d66e2b954b5d1067447bb24082cf19f5fba7e021d7101193e3edc73cefbc1', 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5', 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea'}


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
