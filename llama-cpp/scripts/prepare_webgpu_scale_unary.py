"""Source-bound contiguous F32 SCALE/UNARY fusion; upstream remains read-only."""
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-scale-unary.patch'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-scale-unary-shaders.hpp'
PATCH_SHA256 = 'ffc2f1f1db33d0d350ae2fad89a341debd93d6880909feba7f69d897dcb44602'
INPUTS = {'ggml-webgpu.cpp': '95ab51d247fde6b917adfdbdd469572884952221051b0a79744bb177c902075d', 'ggml-webgpu-shader-lib.hpp': '16b06f059ab3c70b24c62e2d92bd1ec8f69f56af675e27b28d879da553af44c0'}
OUTPUTS = {'ggml-webgpu.cpp': '0e001ef27537170355fecd57b664a5a5fb2ecbb225f570533a9618c899841d16', 'ggml-webgpu-shader-lib.hpp': '498e7a201e64bcf2be3f4480b018141c0e79e0633b4dc0fccaa89bd6c6b07fb2', 'ggml-webgpu-scale-unary.hpp': '78efb6f8ff9b31813cad1da78d5d97e3d38f2e44ba63dc7165d01aebdebe2898', 'wgsl-shaders/scale_unary.wgsl': '0039234c6e562a502569314c216622df26a9b6b04fc24a897c85b7c87bc97ca2'}
HEADER_SHA256 = '54e8ffbef83b37881f7561cd65c4295c71a8806d157ddf4345367e9cce9d8c9b'
CONTRACT_INPUTS = {'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b', 'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc', 'ggml/src/ggml-impl.h': '75465b01eba2b1ecae8634898508e71294fbff0f7f96ee9ad2caf5bae5b6c5fd', 'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa', 'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8', 'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3', 'ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl': 'cee80d0720353f4ad24c8845db54000e6f9b1dd1d1c3585acce3f99b693b7a45', 'ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl': '7d4d66e2b954b5d1067447bb24082cf19f5fba7e021d7101193e3edc73cefbc1', 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5', 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea'}


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
