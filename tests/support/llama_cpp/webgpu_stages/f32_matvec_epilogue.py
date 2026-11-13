"""Checked F32 matvec/scale/sigmoid/scale composition with canonical fallback."""
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import shutil
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-f32-matvec-epilogue-shaders.hpp'
HELPER_NAME = 'ggml-webgpu-f32-matvec-epilogue.hpp'
# Normal and optional diagnostic builds have distinct, checked host prefixes.
PATCHES = {False: ('ggml-webgpu-f32-matvec-epilogue.patch',
         'b66f272698c0c9efd0320acfa3a09512773799f5454ef3f3656771fb0087929a'),
 True: ('ggml-webgpu-f32-matvec-epilogue-profile.patch',
        'fa16ff6c519412233a6740e4f67fa99f580bbf7a4bf422e4e7136cf929d3d8b6')}
INPUTS = {False: {'ggml-webgpu.cpp': 'c42e621e1318a2c70e6a6e651d6abdfac0197415fcdf8f0745c26fc082888232',
         'ggml-webgpu-shader-lib.hpp': '29e31f71751110558ce8718c6f421b3673a6cadc512de0c05817aa166ec1492f'},
 True: {'ggml-webgpu.cpp': '75f2fcdec2e53eade6c9a008c9e1b86e707a0a37d0064e38e3d513ab07c5a92c',
        'ggml-webgpu-shader-lib.hpp': 'f9f39dfcd8b94c44effbf09451af951dc4cc4cd5504ceb1a7b20cb42e348a6b7'}}
OUTPUTS = {False: {'ggml-webgpu-shader-lib.hpp': '1c03edf405f8c645a4ef834a0bf5da41ec868037461092765bd33a8d1baf76f0',
         'ggml-webgpu-f32-matvec-epilogue.hpp': '17ded3f99cc7f4ff63996b6cf71a201d903eb89c85a89ee3e686f5600b322232',
         'ggml-webgpu.cpp': 'c50aeea2606f5da330e6b5f2d26115b94579acd87bb90ae3497415879b90a100',
         'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl': 'b6a0d9fb8a193bfe9c716e3bd191c0af176af839a722bf1b0e4fe0ccb214ddb8'},
 True: {'ggml-webgpu-shader-lib.hpp': 'ab81a43f65058fd9ae02963d98c42da8a08fd1b7483d6f82fc89c2d0c34d6bc6',
        'ggml-webgpu-f32-matvec-epilogue.hpp': '17ded3f99cc7f4ff63996b6cf71a201d903eb89c85a89ee3e686f5600b322232',
        'ggml-webgpu.cpp': 'f7c1281458464c910ebbbbb9f7e679a20580498fdc7ce9cff5000f1f6dde1607',
        'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl': 'b6a0d9fb8a193bfe9c716e3bd191c0af176af839a722bf1b0e4fe0ccb214ddb8'}}
HEADER_SHA256 = 'aa35891dcf4f46072e4501bd58f54b3e931444fbb3b1872bf45fba3dd7c982c0'
CONTRACT_INPUTS = {'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
 'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc',
 'ggml/src/ggml-impl.h': '75465b01eba2b1ecae8634898508e71294fbff0f7f96ee9ad2caf5bae5b6c5fd',
 'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa',
 'ggml/src/ggml-backend.cpp': '2f724930ea53dd8f50f26706ecf10a177d56601e668f109cf1cd95052b1b9ad8',
 'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
 'ggml/src/ggml-webgpu/ggml-webgpu.cpp': 'c3e627af0d18c596dd45f051c7eba612a6f8f53f458bceecd557d8704f52b8ee',
 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': 'a110a08b87a2b2768c991281e450cc0e89b5c34f67856b58138b7ca9e9aa5f17',
 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl': '4fdbbaf3b3d3b7c965786b62a58b6b35ceb6d2f80b29e53f4a027904d3243f18',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_q_acc.tmpl': 'f0353ca818b8922ec11f120e78fefa9d9a4695796663a29aeddc133cce98cf4e',
 'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
 'ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl': 'cee80d0720353f4ad24c8845db54000e6f9b1dd1d1c3585acce3f99b693b7a45',
 'ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl': '7d4d66e2b954b5d1067447bb24082cf19f5fba7e021d7101193e3edc73cefbc1',
 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea'}


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *, gpu_profile: bool) -> dict[str, Path]:
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('F32 matvec epilogue contract changed: ' + relative)
    for relative, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('F32 matvec epilogue composition input changed: ' + relative)
    for relative in (HELPER_NAME, HEADER_NAME, 'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl'):
        if (source / BASE / relative).exists() or (work / BASE / relative).exists():
            raise ValueError('F32 matvec epilogue would replace an unreviewed source: ' + relative)
    name, expected = PATCHES[gpu_profile]
    patch = patch_root / name
    if tensor_copy.digest(patch) != expected:
        raise ValueError('F32 matvec epilogue patch identity changed: ' + name)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True)
    for relative, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected F32 matvec epilogue output: ' + relative)
    shaders = work / 'f32-matvec-epilogue-only-wgsl'
    shaders.mkdir()
    for name in ('common_decls.tmpl', 'mul_mat_vec_acc.tmpl', 'mul_mat_vec_q_acc.tmpl'):
        shutil.copyfile(source / BASE / 'wgsl-shaders' / name, shaders / name)
    shutil.copyfile(work / BASE / 'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl', shaders / 'mul_mat_vec_f32_epilogue.wgsl')
    header = work / HEADER_NAME
    generate_wgsl(shaders, None, header, generator=source / BASE / 'wgsl-shaders/embed_wgsl.py')
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected F32 matvec epilogue embedding identity')
    return {HEADER_NAME: header, HELPER_NAME: work / BASE / HELPER_NAME}
