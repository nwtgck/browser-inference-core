"""Checked F32 matvec/scale/sigmoid/scale composition with canonical fallback."""
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-f32-matvec-epilogue-shaders.hpp'
HELPER_NAME = 'ggml-webgpu-f32-matvec-epilogue.hpp'
# Normal and optional diagnostic builds have distinct, checked host prefixes.
PATCHES = {False: ('ggml-webgpu-f32-matvec-epilogue.patch',
         'b66f272698c0c9efd0320acfa3a09512773799f5454ef3f3656771fb0087929a'),
 True: ('ggml-webgpu-f32-matvec-epilogue-profile.patch',
        'fa16ff6c519412233a6740e4f67fa99f580bbf7a4bf422e4e7136cf929d3d8b6')}
INPUTS = {False: {'ggml-webgpu.cpp': '8e5058d5729d74d4fa6b871b7e5bdbb5abfbeef51234df157064059561b5039a',
         'ggml-webgpu-shader-lib.hpp': 'f8bb0bd001ff26226393d6118ebc5666978e0247a266427a967050d89e9f7399'},
 True: {'ggml-webgpu.cpp': '9f60948e2e9cf75ea538f9907a389a00a7d1ee100559daae3e1465108d9971e0',
        'ggml-webgpu-shader-lib.hpp': 'e5a3811a965acf9b681efc6af3f06183ec9c9a90c0d6abb48a743dd3493d74a3'}}
OUTPUTS = {False: {'ggml-webgpu-shader-lib.hpp': 'f6e586c2bcfc312dcc1cd953e3f0244e4f7cd8696f399e3dfb600cd3c492a5b8',
         'ggml-webgpu-f32-matvec-epilogue.hpp': '17ded3f99cc7f4ff63996b6cf71a201d903eb89c85a89ee3e686f5600b322232',
         'ggml-webgpu.cpp': '6a99af824c58601ccc633e84c266f2efc7091f3616d2552cfe0cf91c312c9318',
         'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl': 'b6a0d9fb8a193bfe9c716e3bd191c0af176af839a722bf1b0e4fe0ccb214ddb8'},
 True: {'ggml-webgpu-shader-lib.hpp': 'a1e963e47e7c424783c955ac75b13f6757c8a574797fb0da0cec2a7601019ef3',
        'ggml-webgpu-f32-matvec-epilogue.hpp': '17ded3f99cc7f4ff63996b6cf71a201d903eb89c85a89ee3e686f5600b322232',
        'ggml-webgpu.cpp': 'e55179d4198ebd17d3ab0ec20d0736211a823649be2959e63934f6e3670ed451',
        'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl': 'b6a0d9fb8a193bfe9c716e3bd191c0af176af839a722bf1b0e4fe0ccb214ddb8'}}
HEADER_SHA256 = 'aa35891dcf4f46072e4501bd58f54b3e931444fbb3b1872bf45fba3dd7c982c0'
CONTRACT_INPUTS = {'ggml/include/ggml.h': '1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101',
 'ggml/src/ggml.c': '4b141cb8e781090b9a4c453781a31c9436052cced6018b080774650b7b57ed4c',
 'ggml/src/ggml-impl.h': 'c84c8f9e3c246504602f5cd388ba54e36691e46bf150e4d70ae857bb823c985f',
 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
 'ggml/src/ggml-backend.cpp': '7ea26359e7eda6957227d6e4c2a24a96aed622cbbca300b066580f7f3d95df5e',
 'ggml/src/ggml-backend-impl.h': '7693def12dcc79a10dd2152e7cb659278756dba7622ebe41937f951dc94944e3',
 'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '7fa5eca8734b7d31bf467281c698b3356c5750c2f923008c179a4dae426cffe5',
 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '34522d61990406ddc1e46c55da4531e6aa2ad1561bf5bf512ebbb38843dbd49d',
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
                       cwd=work, env=env, check=True, capture_output=True)
    for relative, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected F32 matvec epilogue output: ' + relative)
    shaders = work / 'f32-matvec-epilogue-only-wgsl'
    shaders.mkdir()
    for name in ('common_decls.tmpl', 'mul_mat_vec_acc.tmpl', 'mul_mat_vec_q_acc.tmpl'):
        shutil.copyfile(source / BASE / 'wgsl-shaders' / name, shaders / name)
    shutil.copyfile(work / BASE / 'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl', shaders / 'mul_mat_vec_f32_epilogue.wgsl')
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(source / BASE / 'wgsl-shaders/embed_wgsl.py'),
                    '--input_dir', str(shaders), '--output_file', str(header)], check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected F32 matvec epilogue embedding identity')
    return {HEADER_NAME: header, HELPER_NAME: work / BASE / HELPER_NAME}
