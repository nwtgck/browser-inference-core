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
         '1f9ca041784b2665f87c95d0c2623904b139747f59178413f7134f5c695a126e'),
 True: ('ggml-webgpu-f32-matvec-epilogue-profile.patch',
        'cff31629dc1b71d0943af0700c0f0bc414c4d4527b34d22e0f6cc02e2a3bc18c')}
INPUTS = {False: {'ggml-webgpu.cpp': '1ad7f5a177473d4e35f79f0463decc485ed5b9aaf0a3b5a87338ec01dbd757c8',
         'ggml-webgpu-shader-lib.hpp': '5bc3e50a164d6a7fed6980c21a208575972ebce118f8b37787ea2fc8fc856e3e'},
 True: {'ggml-webgpu.cpp': 'c4b8dee4e691a7e10a52412a3dcb4478d3992a2946c4011f4b23850e80332e1f',
        'ggml-webgpu-shader-lib.hpp': 'fabc949c00c6196cd660f6f611ed47ce4ee2c34f54e09109ba9feac545c37880'}}
OUTPUTS = {False: {'ggml-webgpu-shader-lib.hpp': '2fddc2634df24710adb36a8a159d706a7da83468c306df96d6a287b3c6703b46',
         'ggml-webgpu-f32-matvec-epilogue.hpp': '17ded3f99cc7f4ff63996b6cf71a201d903eb89c85a89ee3e686f5600b322232',
         'ggml-webgpu.cpp': 'b16a18a3d544577fbfbc3279475743a5c4cd8c4074f7325d7e337f69cd616708',
         'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl': 'b6a0d9fb8a193bfe9c716e3bd191c0af176af839a722bf1b0e4fe0ccb214ddb8'},
 True: {'ggml-webgpu-shader-lib.hpp': 'e70e89859a4a853670155218adbaf8c7f946b5e07795859ec240b559f4ee5158',
        'ggml-webgpu-f32-matvec-epilogue.hpp': '17ded3f99cc7f4ff63996b6cf71a201d903eb89c85a89ee3e686f5600b322232',
        'ggml-webgpu.cpp': 'a9b0c18384ba901fc38b73652772f608b7ef221edd9d9f908116097255404e5b',
        'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl': 'b6a0d9fb8a193bfe9c716e3bd191c0af176af839a722bf1b0e4fe0ccb214ddb8'}}
HEADER_SHA256 = 'aa35891dcf4f46072e4501bd58f54b3e931444fbb3b1872bf45fba3dd7c982c0'
CONTRACT_INPUTS = {'ggml/include/ggml.h': '2b4dd92fa992cb4ef484917a6c62fef5097698744160ccc09214fd0c459c92c7',
 'ggml/src/ggml.c': '45e90696c4a766d5747f3443f817cd3be11f0110cf93cd16926698c0f49ed083',
 'ggml/src/ggml-impl.h': '43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3',
 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
 'ggml/src/ggml-backend.cpp': '79ade6b83b3d042e43e035fa81d2c672211a552af9af38e410a047e4ba2723a0',
 'ggml/src/ggml-backend-impl.h': '4e861c8e23d3d7b0d99d8e0836e68564af58f8597d75d7bbb732857d7a632fa7',
 'ggml/src/ggml-webgpu/ggml-webgpu.cpp': '67e0c1b6ffeec0d943eb63aced20c00951d565cf2bbb964e90a99265259dca8e',
 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp': '52d39024d24f723dc1b88dc8bf8b4ed18f1e50846010c35065cebc364a789e2d',
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
