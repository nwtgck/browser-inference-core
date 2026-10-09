#!/usr/bin/env python3
"""Prepare one common sampler translation unit; never edit vendor/."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from prepare_mtmd import PATCH_DIRECTORY

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = 'src/llama-sampler.cpp'
CONTEXT_PATH = 'src/llama-context.cpp'
PATCH_NAME = 'llama-sampler-single-sync.patch'
PATCH_SHA256 = '416644819ba169f342db263f1d2d370ff0cfa456149c3370426e4800b9522650'
SAMPLER_SIGNATURE = 'llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx)'
SAMPLER_INPUT_SHA256 = 'a80ba30da083f637b63b75639e6ab56859d12062b148963fd15cf0f7ada7618a'
SAMPLER_OUTPUT_SHA256 = '6cfed41ba4e8b83748646d68c0ce8d777690dfeed11d8953b6d6a4e54abc0391'
# Record full-file identities in provenance, but gate only the relevant context
# functions: unrelated architecture changes must not require a hash refresh.
CONTEXT_FUNCTIONS = (
    'void llama_context::synchronize()',
    'int64_t llama_context::output_resolve_row(int32_t i) const',
    'float * llama_context::get_logits_ith(int32_t i)',
    'llama_token llama_context::get_sampled_token_ith(int32_t idx)',
    'float * llama_context::get_sampled_probs_ith(int32_t idx)',
    'float * llama_context::get_sampled_logits_ith(int32_t idx)',
    'const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx)',
    'size_t llama_context::get_sampled_logits_count(int32_t idx)',
    'size_t llama_context::get_sampled_probs_count(int32_t idx)',
    'void llama_context::output_reorder()',
    'float * llama_get_logits_ith(llama_context * ctx, int32_t i)',
    'llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i)',
    'float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i)',
    'float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i)',
    'llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i)',
    'uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i)',
    'uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i)',
)
CONTEXT_CONTRACT_SHA256 = 'f5efd596911a0b5e0ea35726df598b7dc795d8dce696f1bcf20ae29773f74f59'


def sampler_contract(text: str) -> str:
    marker = SAMPLER_SIGNATURE + ' {'
    if text.count(marker) != 1:
        raise ValueError('Sampler definition needs semantic review')
    start = text.index(marker)
    end = text.index('\n}', start) + 2
    return hashlib.sha256(text[start:end].encode()).hexdigest()


def context_contract(source: Path) -> str:
    text = (source / CONTEXT_PATH).read_text()
    blocks = []
    for signature in CONTEXT_FUNCTIONS:
        marker = signature + ' {'
        if text.count(marker) != 1:
            raise ValueError('Sampler output contract needs review: ' + signature)
        start = text.index(marker)
        # Pinned top-level definitions end at an unindented closing brace.
        end = text.index('\n}', start) + 2
        blocks.append(text[start:end])
    digest = hashlib.sha256('\n'.join(blocks).encode()).hexdigest()
    if digest != CONTEXT_CONTRACT_SHA256:
        raise ValueError('Sampler synchronization/output contract needs semantic review')
    return digest


def prepare(source: Path, output: Path, patch: Path = ROOT / PATCH_DIRECTORY / PATCH_NAME) -> Path:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Sampler overlay must be outside the upstream source tree')
    if hashlib.sha256(patch.read_bytes()).hexdigest() != PATCH_SHA256:
        raise ValueError('Sampler patch identity changed; review before building')
    context_contract(source)
    if sampler_contract((source / SOURCE_PATH).read_text()) != SAMPLER_INPUT_SHA256:
        raise ValueError('Sampler input needs semantic review')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sampler-overlay-', dir=output) as temporary:
        work = Path(temporary)
        (work / 'src').mkdir()
        shutil.copyfile(source / SOURCE_PATH, work / SOURCE_PATH)
        # Prevent discovery of an enclosing consumer/build repository. Git may
        # otherwise silently skip paths even when --no-index was requested.
        env = os.environ.copy()
        for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE'):
            env.pop(key, None)
        env['GIT_CEILING_DIRECTORIES'] = str(work.parent)
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags,
                            str(patch.resolve())], cwd=work, env=env, check=True, capture_output=True)
        patched = work / SOURCE_PATH
        text = patched.read_text()
        # Fail closed if a skipped or changed patch did not establish the boundary.
        if sampler_contract(text) != SAMPLER_OUTPUT_SHA256 or '#include "llama-context.h"' not in text:
            raise ValueError('Unexpected sampler overlay output')
        destination = output / 'llama-sampler.cpp'
        if not destination.exists() or destination.read_bytes() != patched.read_bytes():
            patched.replace(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
