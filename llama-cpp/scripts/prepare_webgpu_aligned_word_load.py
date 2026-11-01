"""Avoid the unused following-word read in aligned shared byte loaders."""
from pathlib import Path
import io
import re
import runpy
import subprocess
import prepare_webgpu_tensor_copy as tensor_copy

BASE = Path('ggml/src/ggml-webgpu')
PATCH_NAME = 'ggml-webgpu-aligned-word-load.patch'
PATCH_SHA256 = '33283e68a5bdffeb616597a5c19d973c03528bea8ac85d36f8e6259c705686bb'
SOURCE_PATH = 'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl'
TEMPLATE_INPUT_SHA256 = '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555'
TEMPLATE_OUTPUT_SHA256 = 'cf3d4d411353796bfb1b100ccb9822839f070c705e0f4430ebd78a9c7db6d85e'
HEADER_NAME = 'ggml-webgpu-aligned-word-load-shaders.hpp'
CANONICAL_INPUT_SHA256 = 'fc870a2d2b5a14a7614859b00f3d2b97de2780ae37b183951b543c85ad77c8e7'
HEADER_SHA256 = '6074e74ee8dde4dcb0d6738d4a18568aaa015bdd788889df643d0a6fe000cb93'
CONTRACT_INPUTS = {'ggml/src/ggml-webgpu/wgsl-shaders/add_id.wgsl': 'c7947fbbd1b63985a34d006d40d55a57eb62e66b5f1bbc32764e537bffb2c13c',
 'ggml/src/ggml-webgpu/wgsl-shaders/argmax.wgsl': '6d23941f4d12015bd7c0b08217109673b39b431bb0d29de25aa4121eb33eb472',
 'ggml/src/ggml-webgpu/wgsl-shaders/argsort.wgsl': 'c95cf77d2f466a7c013c933cc07395a6869f1c73ff24476d3097c19336d98e2a',
 'ggml/src/ggml-webgpu/wgsl-shaders/argsort_merge.wgsl': 'd4b4ad56a3ad14f84b58a9d98fb3e8a9040345df05e299bad06e9ae6fa8f8c0b',
 'ggml/src/ggml-webgpu/wgsl-shaders/binary.wgsl': 'ec26216dfd3881e6ee118103cf9bc8b2f16c22310dbd8d451442f93e7d70d726',
 'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl': '1cfe6cae3ab4a4ccac963a1a68ea9ddf373f4b4beefe7c933807f37969bf5555',
 'ggml/src/ggml-webgpu/wgsl-shaders/concat.wgsl': 'e6c603258460546c87f11a05cb91a881abda42fed71c8ff5d627dbb0ab777cc5',
 'ggml/src/ggml-webgpu/wgsl-shaders/conv2d.wgsl': 'f256a9000d171a0966095ea1f27b3aa3b405fb4ccff5556fa60e9c47b945f84d',
 'ggml/src/ggml-webgpu/wgsl-shaders/conv2d_dw.wgsl': '7d69fe0ff4696875bab6673f31f8695fa5a78db56d30e2a5c48cf2ff8d1fc338',
 'ggml/src/ggml-webgpu/wgsl-shaders/cpy.wgsl': '8ecd80450f0bee89e2673fa0a0594cc4646c888d68010db74121bb3da9e2f0f3',
 'ggml/src/ggml-webgpu/wgsl-shaders/cumsum.wgsl': 'e226b8276251dfd9ad7d8d62004e48fbd718e806b64ff83a187f060ff2c4f462',
 'ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py': '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea',
 'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn.wgsl': '20b2379ea3e9b1552cfa9371b99eb281389c9299b57783c5194056efb9bc6100',
 'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_decls.tmpl': 'd5785764c269ecaa063b4971d8715866a7b201a758edeb61787c118f270aff3d',
 'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_staging.tmpl': '4531292a3288b1064ed8c6c4caf48299e4dd86caa5e4dc48abad73da36cbf691',
 'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_tile.wgsl': 'ac2186476a9426ac525e2ace891c2add10cfc1b6404e1600495f9348fb51ab7b',
 'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_vec_blk.wgsl': '5157da8a01ae95e1102c91c11b98835c445b6c4a7c376fdfc51672b9d9bab9b1',
 'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_vec_reduce.wgsl': 'af1cb00b68724eb2c6e0fb5bd669b4a9ad4f2881bbbdfbbc1e903696e8f141e2',
 'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_vec_split.wgsl': '04f56b75dcdd60c868ee4cbe6992a7d82b7879276af43b762c02301b5c875a21',
 'ggml/src/ggml-webgpu/wgsl-shaders/gated_delta_net.wgsl': 'ec7cd6052b2e7aae6aebd55c41c6d4f7519f4109d1512d658b072a32dd01e77d',
 'ggml/src/ggml-webgpu/wgsl-shaders/get_rows.wgsl': '89d5cef5182e5ce308d4a0bc77c232301836e7d7bfb0d61e46f0ca775b43640c',
 'ggml/src/ggml-webgpu/wgsl-shaders/glu.wgsl': '2cffc91905d5dc5ac783c4bdeb07a39ae2d4e43e5a917907db9e12d1a43476e1',
 'ggml/src/ggml-webgpu/wgsl-shaders/im2col.wgsl': 'f87634613ce8269fe99d0e754e9ff177cb23231f80395b2ec895d614a7f43fcc',
 'ggml/src/ggml-webgpu/wgsl-shaders/memset.wgsl': 'b2dbb3b3d8a466b6eb4d2aa9d357db16bedbd3eaa04e251e8deba99417a4a9b3',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl': 'c461aa2108b752504f47338e41b14ee6570abdd831cd83652e052147c89d69ef',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id.wgsl': '667d3a0f8c0e8a8b033d1b3cf861b69d299dee276e76caddaa0ecccbe885a683',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_gather.wgsl': '433c7b72fb9382e6e452004bdb602d7be8f2654543de680a28d31d8a263298e7',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl': 'cbe87da8b6eb80fcb7905855fe1d212dc5a0afac55b77600c507fbbd2fd7c31d',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl': '2d1cb611caf10b72d20b021772806ca4b3ea4f40c9d7a93941e135dacf859f81',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_subgroup_matrix.wgsl': 'cab4d5d7d034923d9bc092dcf02f92e0d8c9d28d05d29378515dfef82a4e15f0',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec.wgsl': '4fdbbaf3b3d3b7c965786b62a58b6b35ceb6d2f80b29e53f4a027904d3243f18',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl': '9421bdb7ac569e03e8483365653c28b95e659702f8ff849310b2c077ee64290f',
 'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_q_acc.tmpl': 'f0353ca818b8922ec11f120e78fefa9d9a4695796663a29aeddc133cce98cf4e',
 'ggml/src/ggml-webgpu/wgsl-shaders/pad.wgsl': 'a0d4d71d174e52c64e0c3ddf833f277b3208d6c63117201dce3538db9f4061bd',
 'ggml/src/ggml-webgpu/wgsl-shaders/quant_inner_loops.tmpl': '7f023ec774a11265985d92fb630fb94cd8e4599499f941983d1b23b37dacf5ad',
 'ggml/src/ggml-webgpu/wgsl-shaders/quantize_q8.wgsl': 'd466b420d78b33137df123742567551f5c3cd3438905f3254addf9898e4faac2',
 'ggml/src/ggml-webgpu/wgsl-shaders/repeat.wgsl': '909b8c274feeb4f1b4fcf907325b9ba025d956c20ba2078dc7b37b452344fd6f',
 'ggml/src/ggml-webgpu/wgsl-shaders/rms_norm_mul.wgsl': '5e250825d669a99b0c5c894eddebe3307521f89673987f070c8a8d10e808a8f4',
 'ggml/src/ggml-webgpu/wgsl-shaders/rope.wgsl': '2497707c6b6a6d2fb44bc36229d7c2eb8cdbe71620fa572f1cfe9d180d4c3994',
 'ggml/src/ggml-webgpu/wgsl-shaders/row_norm.wgsl': '1676d1c2d90f44b83584d175078654812d2c8a36dd3a261a055e265f267f1a41',
 'ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl': 'cee80d0720353f4ad24c8845db54000e6f9b1dd1d1c3585acce3f99b693b7a45',
 'ggml/src/ggml-webgpu/wgsl-shaders/set.wgsl': 'a16785cbe0faf352371ec764626f48c6f0d3baecff6b6014bc96a9aff0139175',
 'ggml/src/ggml-webgpu/wgsl-shaders/set_rows.wgsl': '074501568a5380778d5a09dcc87379038c1c8e0d4b92e2ec2dd526a5072d210f',
 'ggml/src/ggml-webgpu/wgsl-shaders/set_rows_quant.wgsl': '9f8500cc80b76674e6790d6700fb62dccaf07576b5b8712ae678014b793b1a7c',
 'ggml/src/ggml-webgpu/wgsl-shaders/soft_max.wgsl': 'b9a4714b586400e5ad9cf87d45f6615c41e1b004bfa22f296e6e20af899b35a8',
 'ggml/src/ggml-webgpu/wgsl-shaders/solve_tri.wgsl': '1b4bcbe26ab4c4741487786a8d81f2b0f9cf480955ea7babea0902438ac939e4',
 'ggml/src/ggml-webgpu/wgsl-shaders/ssm_conv.wgsl': '868a7284cb858811b5ac70f466550340dba0df356b3b9ac1e7568053b7b9c809',
 'ggml/src/ggml-webgpu/wgsl-shaders/ssm_scan.wgsl': '78e5f808dbeeebf56c4d33c588f5a8f2499152334c1d98d0f5578a2f561c0253',
 'ggml/src/ggml-webgpu/wgsl-shaders/sum_rows.wgsl': 'e2e478cdddec955c3abab6de99afc4b012f47c0b5ae8f828e41bc8a606c88a35',
 'ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl': '7d4d66e2b954b5d1067447bb24082cf19f5fba7e021d7101193e3edc73cefbc1',
 'ggml/src/ggml-webgpu/wgsl-shaders/upscale.wgsl': 'd2a309d31c875ffb2ccf2a3bf34f54296ce08e12851a6ba90c34e20f1a95e5d6'}
INPUTS = {False: {'ggml-webgpu-shader-lib.hpp': 'd05738a29d0629d60be93224b1ba69c16d67d97300864c058730bc64c4c73297',
         'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335'},
 True: {'ggml-webgpu-shader-lib.hpp': 'c9a8bc051f56126606327e8fd137a8a2e7374daa0490c893a3036182b6b98462',
        'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791'}}
OUTPUTS = {False: {'ggml-webgpu-shader-lib.hpp': '87e20119da7064dda8210c6130b1e43a4aaf83c644ae5e58a691009f709405a8',
         'ggml-webgpu.cpp': 'ef3e19338638464151e5416591e435deb62cb0d4974be45e61e4b8ec7bba0335'},
 True: {'ggml-webgpu-shader-lib.hpp': '01f9e59fe52a3f1865398d96478a7ce9bafb174c320809ede04384ba269acd0a',
        'ggml-webgpu.cpp': '9a6c93ab8c9bfb1736224e5943e3eb899e83aa4cbdaeacd9ff6f533158527791'}}
EMBEDDING_INPUTS = {'ggml-webgpu-f32-matvec-epilogue-shaders.hpp': 'aa35891dcf4f46072e4501bd58f54b3e931444fbb3b1872bf45fba3dd7c982c0',
 'ggml-webgpu-iq3s-id-subgroup-shaders.hpp': '6427de75e644afb55e31cc2c8779004087087b60e846dfbdba11f3fecc88acbb',
 'ggml-webgpu-mul-mat-id-group-shaders.hpp': '8b3dabdd9356b5a5e020d84c354faa68a70b8634c81793060d53fb58bd55bc10',
 'ggml-webgpu-q8-subgroup-shaders.hpp': '7d151de06e9a3cb5366b29e043052b3b949c4c1948e7b3c2e80e99a944b95cab',
 'ggml-webgpu-quant-half-pairs-shaders.hpp': '693ede233b36b81d0360bfc212a5ecb6c31cf10aae06afa6788682fbf317e19a'}
EMBEDDING_OUTPUTS = {'ggml-webgpu-f32-matvec-epilogue-shaders.hpp': '352b801deed74bce60bf88b5455a2663eaaa0f71fd6a778bbfbf434627b6ad31',
 'ggml-webgpu-iq3s-id-subgroup-shaders.hpp': '9df296bb9d18cb3377ff161143a1d34678342c2c5de6ea962f6e19f185636de3',
 'ggml-webgpu-mul-mat-id-group-shaders.hpp': '16b9db3c2287111de4f073eb0f62d8377ff00368b3b9a3897b6fca3209405877',
 'ggml-webgpu-q8-subgroup-shaders.hpp': 'b1b86048fbb6b9a58834a4b5ee5552ac2704641cbf4db3c996c1a85674ea302b',
 'ggml-webgpu-quant-half-pairs-shaders.hpp': '8bd7b49bb141aded20c0c5ea1277376825e16a46d71f88f2372e954589b6f676'}
HELPERS = ('load_u32_at_src', 'load_u32_at_src0')
TEMPLATE_OUTPUT_BYTES = 74306
CANONICAL_INPUT_IDENTITY = {'bytes': 1471238, 'sha256': 'fc870a2d2b5a14a7614859b00f3d2b97de2780ae37b183951b543c85ad77c8e7'}
CONSUMER_INPUT_IDENTITIES = {False: {'ggml-webgpu-aligned-word-load-shaders.hpp': {'bytes': 1471238,
                                                       'sha256': 'fc870a2d2b5a14a7614859b00f3d2b97de2780ae37b183951b543c85ad77c8e7'},
         'ggml-webgpu-f32-matvec-epilogue-shaders.hpp': {'bytes': 174683,
                                                         'sha256': 'aa35891dcf4f46072e4501bd58f54b3e931444fbb3b1872bf45fba3dd7c982c0'},
         'ggml-webgpu-iq3s-id-subgroup-shaders.hpp': {'bytes': 155961,
                                                      'sha256': '6427de75e644afb55e31cc2c8779004087087b60e846dfbdba11f3fecc88acbb'},
         'ggml-webgpu-mul-mat-id-group-shaders.hpp': {'bytes': 130579,
                                                      'sha256': '8b3dabdd9356b5a5e020d84c354faa68a70b8634c81793060d53fb58bd55bc10'},
         'ggml-webgpu-q8-subgroup-shaders.hpp': {'bytes': 153198,
                                                 'sha256': '7d151de06e9a3cb5366b29e043052b3b949c4c1948e7b3c2e80e99a944b95cab'},
         'ggml-webgpu-quant-half-pairs-shaders.hpp': {'bytes': 129649,
                                                      'sha256': '693ede233b36b81d0360bfc212a5ecb6c31cf10aae06afa6788682fbf317e19a'},
         'ggml-webgpu-shader-lib.hpp': {'bytes': 171887,
                                        'sha256': 'd05738a29d0629d60be93224b1ba69c16d67d97300864c058730bc64c4c73297'}},
 True: {'ggml-webgpu-aligned-word-load-shaders.hpp': {'bytes': 1471238,
                                                      'sha256': 'fc870a2d2b5a14a7614859b00f3d2b97de2780ae37b183951b543c85ad77c8e7'},
        'ggml-webgpu-f32-matvec-epilogue-shaders.hpp': {'bytes': 174683,
                                                        'sha256': 'aa35891dcf4f46072e4501bd58f54b3e931444fbb3b1872bf45fba3dd7c982c0'},
        'ggml-webgpu-iq3s-id-subgroup-shaders.hpp': {'bytes': 155961,
                                                     'sha256': '6427de75e644afb55e31cc2c8779004087087b60e846dfbdba11f3fecc88acbb'},
        'ggml-webgpu-mul-mat-id-group-shaders.hpp': {'bytes': 130579,
                                                     'sha256': '8b3dabdd9356b5a5e020d84c354faa68a70b8634c81793060d53fb58bd55bc10'},
        'ggml-webgpu-q8-subgroup-shaders.hpp': {'bytes': 153198,
                                                'sha256': '7d151de06e9a3cb5366b29e043052b3b949c4c1948e7b3c2e80e99a944b95cab'},
        'ggml-webgpu-quant-half-pairs-shaders.hpp': {'bytes': 129649,
                                                     'sha256': '693ede233b36b81d0360bfc212a5ecb6c31cf10aae06afa6788682fbf317e19a'},
        'ggml-webgpu-shader-lib.hpp': {'bytes': 174249,
                                       'sha256': 'c9a8bc051f56126606327e8fd137a8a2e7374daa0490c893a3036182b6b98462'}}}

def definition(text: str, name: str) -> str:
    start = text.index('fn ' + name + '(')
    at = text.index('{', start) + 1
    depth = 1
    while depth:
        if at >= len(text):
            raise ValueError('Unterminated shared word helper: ' + name)
        depth += (text[at] == '{') - (text[at] == '}')
        at += 1
    return text[start:at]


def replace_embedding(raw: bytes, original: str, candidate: str, *, count: int) -> bytes:
    before = raw.decode('utf-8')
    after = before
    ranges = [(m.start(2), m.end(2)) for m in re.finditer(r'R"([A-Za-z0-9_]*)\((.*?)\)\1"', before, re.DOTALL)]
    for name in HELPERS:
        old, new = definition(original, name), definition(candidate, name)
        if before.count(old) != count or before.count(new):
            raise ValueError('Shared word-helper embedding inventory changed: ' + name)
        for match in re.finditer(re.escape(old), before):
            if not any(a <= match.start() and match.end() <= b for a, b in ranges):
                raise ValueError('Shared word helper crosses a raw-string chunk boundary')
        after = after.replace(old, new)
    inverse = after
    for name in HELPERS:
        inverse = inverse.replace(definition(candidate, name), definition(original, name))
    if inverse != before:
        raise ValueError('Shared word-loader changed unrelated embedding bytes')
    # Keep the original safe reserve capacity and chunk layout. Only WGSL bodies change.
    return after.encode('utf-8')


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str], *,
            gpu_profile: bool, headers: dict[str, Path]) -> dict[str, Path]:
    if tensor_copy.digest(source / SOURCE_PATH) != TEMPLATE_INPUT_SHA256:
        raise ValueError('Shared word-loader template identity changed')
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Shared word-loader upstream contract changed: ' + relative)
    for name, expected in INPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Shared word-loader composition input changed: ' + name)
    if (work / HEADER_NAME).exists() or (source / BASE / HEADER_NAME).exists():
        raise ValueError('Shared word-loader canonical embedding already exists')
    for name, expected in EMBEDDING_INPUTS.items():
        if name not in headers or tensor_copy.digest(headers[name]) != expected:
            raise ValueError('Shared word-loader dedicated embedding changed: ' + name)
    for name, path in headers.items():
        if name not in EMBEDDING_INPUTS and any('fn ' + helper + '(' in path.read_text() for helper in HELPERS):
            raise ValueError('Unclassified shared word-loader consumer: ' + name)
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('Shared word-loader patch identity changed')
    isolated = work / 'aligned-word-load-template'
    target = isolated / SOURCE_PATH
    target.parent.mkdir(parents=True)
    original = (source / SOURCE_PATH).read_text()
    target.write_text(original)
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=isolated, env=env, check=True, capture_output=True)
    if tensor_copy.digest(target) != TEMPLATE_OUTPUT_SHA256:
        raise ValueError('Unexpected patched shared word-loader template')
    candidate = target.read_text()
    inverse = candidate
    for name in HELPERS:
        inverse = inverse.replace(definition(candidate, name), definition(original, name))
    if inverse != original:
        raise ValueError('Shared template changed outside the two helpers')
    embed = runpy.run_path(str(source / BASE / 'wgsl-shaders/embed_wgsl.py'))
    out = io.StringIO()
    out.write('// Auto-generated shader embedding\n#include <string>\n\n')
    for path in sorted((source / BASE / 'wgsl-shaders').glob('*.wgsl')):
        embed['write_shader'](path.stem, path.read_text(), None, out, str(path.parent))
    canonical = work / HEADER_NAME
    canonical.write_text(out.getvalue())
    if tensor_copy.digest(canonical) != CANONICAL_INPUT_SHA256:
        raise ValueError('Canonical upstream embedding changed')
    canonical.write_bytes(replace_embedding(canonical.read_bytes(), original, candidate, count=12))
    if tensor_copy.digest(canonical) != HEADER_SHA256:
        raise ValueError('Unexpected shared word-loader canonical embedding')
    generated = {HEADER_NAME: canonical}
    for name, expected in EMBEDDING_OUTPUTS.items():
        target_header = headers[name]
        target_header.write_bytes(replace_embedding(target_header.read_bytes(), original, candidate, count=1))
        if tensor_copy.digest(target_header) != expected:
            raise ValueError('Unexpected shared word-loader dedicated output: ' + name)
        generated[name] = target_header
    library = work / BASE / 'ggml-webgpu-shader-lib.hpp'
    text = library.read_text()
    old_include = '#include "ggml-wgsl-shaders.hpp"'
    new_include = '#include "' + HEADER_NAME + '"'
    if text.count(old_include) != 1 or HEADER_NAME in text:
        raise ValueError('Shared word-loader canonical include ownership changed')
    library.write_text(text.replace(old_include, new_include))
    for name, expected in OUTPUTS[gpu_profile].items():
        if tensor_copy.digest(work / BASE / name) != expected:
            raise ValueError('Unexpected shared word-loader final source: ' + name)
    return generated
