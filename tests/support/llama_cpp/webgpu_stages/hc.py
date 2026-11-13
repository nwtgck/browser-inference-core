"""Compose narrow, checked WebGPU HC support after the existing overlays."""
from __future__ import annotations
from pathlib import Path
from build_tools.source.wgsl import generate as generate_wgsl
import shutil
import subprocess
from tests.support.llama_cpp.webgpu_stages import tensor_copy
PATCH_NAME = 'ggml-webgpu-hc-pre-post.patch'
BASE = Path('ggml/src/ggml-webgpu')
HEADER_NAME = 'ggml-webgpu-hc-shaders.hpp'
PATCH_SHA256 = 'e91713461447e5921bec09a788095179d98611167eb7eac7fc3b63004d1e42db'
INPUTS = {'ggml-webgpu-shader-lib.hpp': '7abf67de6bff8c22c3131cf01e29ca06c04adadbcd7c58fce5e820007b56117d',
 'ggml-webgpu.cpp': '94302059f96b905538fdb4fba9f9631c2688bbdbf2ca656f8ea88f5843a58cb4'}
OUTPUTS = {'ggml-webgpu-hc.hpp': '24abc54f87a00ff2113d74480268b9f15a9bd5c2fbe01ac88eee3ea5f36351cc',
 'ggml-webgpu-shader-lib.hpp': '921a3626246150081a28b4e9662cb516b80da26cdc73291dccd30b9a4f456806',
 'ggml-webgpu.cpp': 'ca3f9148143e9ed1629678dc6310fe44f86227d54767bb2e892113fd8cf539b6',
 'wgsl-shaders/hc_post.wgsl': '6170c98f654bcb6230b2f5512865b175fcb2c7dfa60715b4eaf161579c1c1a9d',
 'wgsl-shaders/hc_pre.wgsl': '18c8b9b4a4a0c0b7c6a86f0b2672eb05db8393ccbf295f611c204afd4bf3740b'}
CONTRACT_INPUTS = {'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
 'ggml/src/ggml-alloc.c': 'c4001b08581b87b80283953acc67d0d2ff13ef0a425179e2bf621f45333bf0fa',
 'ggml/src/ggml-cpu/ops.cpp': '289be21fbf5736b0deb282652b89078cb5833ec081ebd368ebaa1d9a5d503684',
 'ggml/src/ggml-webgpu/pre_wgsl.hpp': '0d6f781eb5723311cf856e585566c70ddcc779f0e97fcfe6d0e3f636c35688e5',
 'ggml/src/ggml-webgpu/wgsl-shaders/binary.wgsl': 'ec26216dfd3881e6ee118103cf9bc8b2f16c22310dbd8d451442f93e7d70d726',
 'ggml/src/ggml-webgpu/wgsl-shaders/cpy.wgsl': '8ecd80450f0bee89e2673fa0a0594cc4646c888d68010db74121bb3da9e2f0f3',
 'ggml/src/ggml-webgpu/wgsl-shaders/repeat.wgsl': '909b8c274feeb4f1b4fcf907325b9ba025d956c20ba2078dc7b37b452344fd6f',
 'ggml/src/ggml-webgpu/wgsl-shaders/scale.wgsl': 'cee80d0720353f4ad24c8845db54000e6f9b1dd1d1c3585acce3f99b693b7a45',
 'ggml/src/ggml-webgpu/wgsl-shaders/unary.wgsl': '7d4d66e2b954b5d1067447bb24082cf19f5fba7e021d7101193e3edc73cefbc1',
 'ggml/src/ggml.c': '1008000571b896a22dd3861731ca29e6ec3615898cdc15d9cb529fb171815dfc',
 'src/llama-context.cpp': '6b2a143eee8d53cdb559eb18b5612028bdf8545ab832cfdca2c2fd87c9c00309',
 'src/models/qwen4exp.cpp': 'cd855a0d5356de677b0224658ccd75a0a2d52873ea4019536dbc688600ca48e0'}
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
                       cwd=work, env=env, check=True)
    for relative, expected in OUTPUTS.items():
        if tensor_copy.digest(work / BASE / relative) != expected:
            raise ValueError('Unexpected HC output: ' + relative)
    shaders = work / 'hc-only-wgsl'
    shaders.mkdir()
    for name in ('hc_pre.wgsl', 'hc_post.wgsl'):
        shutil.copyfile(work / BASE / 'wgsl-shaders' / name, shaders / name)
    header = work / HEADER_NAME
    generate_wgsl(shaders, None, header, generator=embedder)
    if tensor_copy.digest(header) != HEADER_SHA256:
        raise ValueError('Unexpected HC shader header identity')
    return {HEADER_NAME: header, 'ggml-webgpu-hc.hpp': work / BASE / 'ggml-webgpu-hc.hpp'}
