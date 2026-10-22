"""Compose narrow, checked WebGPU HC support after the existing overlays."""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import sys
import prepare_webgpu_tensor_copy as tensor_copy

PATCH_NAME = 'ggml-webgpu-hc-pre-post.patch'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-hc-shaders.hpp'
PATCH_SHA256 = '97d142652b31d41836f6046e0fcaa1ea922583bbcd161f5c6903495c2a5afd3a'
INPUTS = {'ggml-webgpu-shader-lib.hpp': '8d6c8d311106eaf3d97b57b278c6264d806876242ce3b5f4f72573c69221b85f',
 'ggml-webgpu.cpp': 'bed54cb4d4328531b65006e41262043f41632ea050a6af31ecc560a24b284509'}
OUTPUTS = {'ggml-webgpu-hc.hpp': '24abc54f87a00ff2113d74480268b9f15a9bd5c2fbe01ac88eee3ea5f36351cc',
 'ggml-webgpu-shader-lib.hpp': '544c99d99ab4e986989f3220b0327f787ab200c4cbd23e3f6a5ff8901d7ec42c',
 'ggml-webgpu.cpp': 'c0ecb208561874a1c0e1f584604e289cd0a2c6673563fa4357982912043b0d75',
 'wgsl-shaders/hc_post.wgsl': '6170c98f654bcb6230b2f5512865b175fcb2c7dfa60715b4eaf161579c1c1a9d',
 'wgsl-shaders/hc_pre.wgsl': '18c8b9b4a4a0c0b7c6a86f0b2672eb05db8393ccbf295f611c204afd4bf3740b'}
CONTRACT_INPUTS = {'ggml/include/ggml.h': '349e1ea995dffbf900ce4345e0b2786c8a9b778035a66d6118b702f52b3af13f',
 'ggml/src/ggml-alloc.c': '9664bda3e9c4223f5c94ebfeb3c0381f0065a5091ddae707c09a800f7e419a39',
 'ggml/src/ggml-cpu/ops.cpp': '05ee55649f47fad58791436401444720a110d941f314a5ad623bb191d49d823a',
 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
 'ggml/src/ggml-webgpu/wgsl-shaders/binary.wgsl': 'ec26216dfd3881e6ee118103cf9bc8b2f16c22310dbd8d451442f93e7d70d726',
 'ggml/src/ggml-webgpu/wgsl-shaders/cpy.wgsl': '8ecd80450f0bee89e2673fa0a0594cc4646c888d68010db74121bb3da9e2f0f3',
 'ggml/src/ggml-webgpu/wgsl-shaders/repeat.wgsl': '909b8c274feeb4f1b4fcf907325b9ba025d956c20ba2078dc7b37b452344fd6f',
 'ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl': 'cee80d0720353f4ad24c8845db54000e6f9b1dd1d1c3585acce3f99b693b7a45',
 'ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl': '7d4d66e2b954b5d1067447bb24082cf19f5fba7e021d7101193e3edc73cefbc1',
 'ggml/src/ggml.c': '6f839423326ca75319ab0301b4b242e94974b48808888042813f199df98e7482',
 'src/llama-context.cpp': '07109d97009e8aae60c7a4b685246d9ffde3d9eadb1606364a1e5f6a56553502',
 'src/models/qwen4exp.cpp': '8b723892b9f75ce6ce2c9f77d3e517c9b8032beb549f487857fb1c524f0efbe7'}
HEADER_SHA256 = 'e6cbf0800041206f1ba27aac5ad23b2a02701753b5869060a4c3d66c5288e8c2'
EMBEDDER_SHA256 = '37d4a93bb46b0ff417c31a0c1d3fa4b9db52556556138cb2eaedae9daa8dcbea'


def prepare(work: Path, source: Path, patch_root: Path, env: dict[str, str]) -> dict[str, Path]:
    # All work remains temporary until the caller verifies every combined output.
    for relative, expected in CONTRACT_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('HC upstream contract changed: ' + relative)
    for relative, expected in INPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('HC composition input changed: ' + relative)
    for relative in ('ggml-webgpu-hc.hpp', 'wgsl-shaders/hc_pre.wgsl', 'wgsl-shaders/hc_post.wgsl'):
        if (source / BASE / relative).exists():
            raise ValueError('Upstream HC source now exists: ' + relative)
    embedder = source / BASE / 'wgsl-shaders/embed_wgsl.py'
    patch = patch_root / PATCH_NAME
    if tensor_copy.digest(embedder) != EMBEDDER_SHA256:
        raise ValueError('HC embedder identity changed')
    if tensor_copy.digest(patch) != PATCH_SHA256:
        raise ValueError('HC patch identity changed')
    for flags in (['--check'], []):
        subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                       cwd=work, env=env, check=True, capture_output=True)
    for relative, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected HC output: ' + relative)
    shaders = work / 'hc-only-wgsl'
    shaders.mkdir()
    for name in ('hc_pre.wgsl', 'hc_post.wgsl'):
        shutil.copyfile(work / BASE / 'wgsl-shaders' / name, shaders / name)
    header = work / HEADER_NAME
    subprocess.run([sys.executable, str(embedder), '--input_dir', str(shaders), '--output_file', str(header)],
                   check=True, capture_output=True)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected HC shader header identity')
    return {HEADER_NAME: header, 'ggml-webgpu-hc.hpp': work / BASE / 'ggml-webgpu-hc.hpp'}
