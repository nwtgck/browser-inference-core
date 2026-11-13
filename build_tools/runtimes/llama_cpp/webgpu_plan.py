"""Prepare the approved WebGPU source set before generating any shader headers.

Patch scope is declared in config/webgpu-patches.json. Hashes in the receipt
observe the bytes used; they do not stand in for type or numerical contracts.
The caller supplies a captured upstream tree. Neither that tree nor a previous
successful output is modified when preparation fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from build_tools.output import output_directories, preserve_unchanged
from build_tools.project import runtime_root
from build_tools.source.files import absolute, directory, regular_file
from build_tools.source.patches import apply_patch, read_series
from build_tools.source.wgsl import generate, input_files

ROOT = runtime_root('llama-cpp')
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
BASE = Path('ggml/src/ggml-webgpu')
WGSL = BASE / 'wgsl-shaders'
INVENTORY = ROOT / 'config/webgpu-patches.json'
RECEIPT = 'source-preparation.json'

# This is a concrete build recipe for the two accepted compositions, not a
# plugin registry or executable configuration language. Shared patch order is
# visible here; the inventory separately limits the files each patch may edit.
PREFIX = (
    'ggml-webgpu-same-device-tensor-copy.patch',
    'ggml-webgpu-batch-param-uploads.patch',
    'llama-model-loader-webgpu-chunked-upload.patch',
    'ggml-webgpu-ssm-conv-single-token.patch',
    'ggml-webgpu-hc-pre-post.patch',
    'ggml-webgpu-q8-short-workgroup.patch',
    'ggml-webgpu-moe-expert-sum.patch',
    'ggml-webgpu-expert-short-workgroup.patch',
    'ggml-webgpu-ordered-readback.patch',
    'ggml-webgpu-bind-group-cache.patch',
    'ggml-webgpu-q4-short-workgroup.patch',
    'ggml-webgpu-scale-unary.patch',
    'ggml-webgpu-q8-subgroup-matvec.patch',
    'ggml-webgpu-rms-scale.patch',
    'ggml-webgpu-async-tensor-copy.patch',
    'ggml-webgpu-iq3s-id-subgroup.patch',
    'ggml-webgpu-host-input-snapshot.patch',
    'ggml-webgpu-ssm-conv-silu.patch',
    'ggml-webgpu-moe-group-skip.patch',
    'ggml-webgpu-router-tail.patch',
    'ggml-webgpu-host-encode-assembly.patch',
    'ggml-webgpu-f32-small-m.patch',
    'ggml-webgpu-moe-gather-logical32.patch',
    'llama-sampler-single-sync.patch',
)


def patch_names(gpu_profile: bool) -> tuple[str, ...]:
    suffix = '-profile' if gpu_profile else ''
    diagnostic = ('ggml-webgpu-profile-query-budget.patch',
                  'ggml-webgpu-q8-diagnostic.patch') if gpu_profile else ()
    return (*PREFIX, *diagnostic,
            f'ggml-webgpu-matmul-vec4-layout{suffix}.patch',
            'ggml-webgpu-moe-tight-tile-bound.patch',
            f'ggml-webgpu-f32-matvec-epilogue{suffix}.patch',
            'ggml-webgpu-q6-short-workgroup.patch',
            'ggml-webgpu-quant-half-pairs.patch',
            'ggml-webgpu-direct-mapped-read.patch',
            'ggml-webgpu-quant-wide-m.patch',
            'ggml-webgpu-q4-subgroup-matvec.patch',
            'ggml-webgpu-aligned-word-load-integration.patch',
            'ggml-webgpu-checkpoint-batch-readback.patch',
            'ggml-webgpu-subgroup-size-control-shared.patch',
            f'ggml-webgpu-subgroup-size-control{suffix}.patch',
            'ggml-webgpu-subgroup-size-control-wgsl.patch',
            'ggml-webgpu-q6-subgroup-matvec.patch',
            'ggml-webgpu-buffer-size-limit.patch',
            'ggml-webgpu-emscripten-scalar-tensor-writes.patch')


# Dedicated variants start from private copies, never from generated C++.
DERIVED = {
    'ssm_conv_silu.wgsl': 'ssm_conv.wgsl',
    'mul_mat_id_group.wgsl': 'mul_mat_id.wgsl',
    'mul_mat_id_gather_logical32.wgsl': 'mul_mat_id_gather.wgsl',
    'mul_mat_reg_tile_half_pairs.wgsl': 'mul_mat_reg_tile.wgsl',
    'mul_mat_half_pairs_decls.tmpl': 'mul_mat_decls.tmpl',
    'quant_half_pairs_inner_loops.tmpl': 'quant_inner_loops.tmpl',
}
HOST_FILES = (
    'ggml/src/ggml-backend.cpp',
    'ggml/src/ggml-backend-host-snapshot.h',
    'ggml/src/ggml-backend-tensor-get-batch.h',
    'src/llama-model-loader.cpp', 'src/llama-sampler.cpp', 'src/llama-context.cpp',
    *('ggml/src/ggml-webgpu/' + name for name in (
        'ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp', 'ggml-webgpu-hc.hpp',
        'ggml-webgpu-moe-sum.hpp', 'ggml-webgpu-scale-unary.hpp',
        'ggml-webgpu-rms-scale.hpp', 'ggml-webgpu-ssm-conv-silu.hpp',
        'ggml-webgpu-router-tail.hpp', 'ggml-webgpu-f32-matvec-epilogue.hpp',
        'ggml-webgpu-batch-readback-geometry.hpp', 'ggml-webgpu-batch-readback.hpp',
        'ggml-webgpu-optional-pipeline.hpp')),
)


def identity(path: Path) -> dict:
    raw = regular_file(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def copy(source: Path, target: Path) -> None:
    regular_file(source)
    directory(target.parent, create=True)
    if target.exists() or target.is_symlink():
        raise ValueError('Source preparation would overwrite a private input: ' + str(target))
    shutil.copyfile(source, target)


def _shader_inputs(source: Path, work: Path, iq3s: Path, template: Path) -> dict[str, dict[str, Path]]:
    original, patched = source / WGSL, work / WGSL
    common = {'common_decls.tmpl': template}
    accumulator = {'mul_mat_vec_acc.tmpl': original / 'mul_mat_vec_acc.tmpl'}
    def edited(*names: str) -> dict[str, Path]:
        return {name: patched / name for name in names}
    def upstream(*names: str) -> dict[str, Path]:
        return {name: original / name for name in names}
    return {
        'ggml-webgpu-hc-shaders.hpp': edited('hc_pre.wgsl', 'hc_post.wgsl'),
        'ggml-webgpu-moe-sum-shaders.hpp': edited('moe_expert_sum.wgsl'),
        'ggml-webgpu-scale-unary-shaders.hpp': edited('scale_unary.wgsl'),
        'ggml-webgpu-rms-scale-shaders.hpp': {'rms_norm_scale.wgsl': patched / 'row_norm.wgsl'},
        'ggml-webgpu-ssm-conv-silu-shaders.hpp': edited('ssm_conv_silu.wgsl'),
        'ggml-webgpu-router-tail-shaders.hpp': edited('router_tail.wgsl', 'router_tail_merged.wgsl'),
        'ggml-webgpu-mul-mat-id-gather-logical32-shaders.hpp': edited('mul_mat_id_gather_logical32.wgsl'),
        'ggml-webgpu-q8-subgroup-shaders.hpp': {**common, **accumulator, **edited('q8_subgroup_matvec.wgsl')},
        'ggml-webgpu-iq3s-id-subgroup-shaders.hpp': {
            **common, **edited('iq3s_id_subgroup.wgsl'),
            'mul_mat_vec_acc.tmpl': iq3s / WGSL / 'mul_mat_vec_acc.tmpl'},
        'ggml-webgpu-mul-mat-id-group-shaders.hpp': {
            **common, **edited('mul_mat_id_group.wgsl'),
            **upstream('mul_mat_decls.tmpl', 'quant_inner_loops.tmpl')},
        'ggml-webgpu-f32-matvec-epilogue-shaders.hpp': {
            **common, **accumulator, **upstream('mul_mat_vec_q_acc.tmpl'),
            **edited('mul_mat_vec_f32_epilogue.wgsl')},
        'ggml-webgpu-quant-half-pairs-shaders.hpp': {
            **common, **edited('mul_mat_reg_tile_half_pairs.wgsl',
                               'mul_mat_half_pairs_decls.tmpl', 'quant_half_pairs_inner_loops.tmpl')},
    }


def prepare(source: Path, output: Path, *, patch_root: Path = ROOT / PATCH_DIRECTORY,
            gpu_profile: bool = False, inventory: Path = INVENTORY) -> Path:
    source, output = directory(source), absolute(output)
    patch_root = directory(patch_root)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('WebGPU preparation must be outside the upstream source tree')
    inventory = regular_file(inventory)
    owned_inputs = patch_root.parent / 'native/diagnostics'
    for input_root in (patch_root, owned_inputs):
        if output == input_root or output in input_root.parents or input_root in output.parents:
            raise ValueError('WebGPU output must not overlap preparation inputs')
    if inventory == output or output in inventory.parents:
        raise ValueError('WebGPU output must not replace its patch inventory')
    if (output / 'ggml-wgsl-shaders.hpp').exists() or (output / 'ggml-wgsl-shaders.hpp').is_symlink():
        raise ValueError('Unexpected canonical WGSL header would shadow upstream shaders')
    series = read_series(inventory, targets=('llama-webgpu', 'iq3s', 'word-template'))
    definitions = {patch.file: patch for patch in series.patches}
    names = patch_names(gpu_profile)
    extra = ('ggml-webgpu-iq3s-field-loads.patch', 'ggml-webgpu-iq3s-row-split.patch',
             'ggml-webgpu-aligned-word-load.patch')
    expected = set(patch_names(False)) | set(patch_names(True)) | set(extra)
    if set(definitions) != expected:
        raise ValueError('WebGPU patch inventory differs from the explicit preparation recipe')
    if any(definitions[name].target != 'llama-webgpu' for name in names):
        raise ValueError('WebGPU host patch has the wrong source scope')
    if any(definitions[name].target != 'iq3s' for name in extra[:2]) or definitions[extra[2]].target != 'word-template':
        raise ValueError('Dedicated shader patch has the wrong source scope')

    with output_directories(output, replace=True, reuse_if=preserve_unchanged) as (prepared,), \
            tempfile.TemporaryDirectory(prefix='webgpu-plan-', dir=prepared.parent) as temporary:
        private = Path(temporary)
        work = private / 'host'
        work.mkdir()
        # Capture patches before any inspection/application. Mid-preparation
        # edits to the producer cannot mix two patch revisions in one output.
        patches = private / 'patches'
        patches.mkdir()
        for name in (*names, *extra):
            copy(patch_root / name, patches / name)
        inputs: dict[str, dict] = {}
        selected = [definitions[name] for name in names]
        touched = {name for patch in selected for name in patch.files}
        originals = {name for name in touched
                     if (source / name).exists() or (source / name).is_symlink()}
        originals.update(str(path.relative_to(source)) for path in input_files(source / WGSL))
        # The upstream tree is already pinned by build input capture. Also take
        # one private copy here so all shader families use the same input bytes.
        captured = private / 'upstream'
        for name in sorted(originals):
            copy(source / name, captured / name)
            inputs[name] = identity(captured / name)
        for name in DERIVED:
            if (source / WGSL / name).exists() or (source / WGSL / name).is_symlink():
                raise ValueError('Dedicated shader now exists upstream: ' + name)
        owned = private / 'diagnostics'
        if gpu_profile:
            for name in ('webgpu-q8-diagnostic.h', 'ggml-webgpu-q8-diagnostic-sha256.hpp'):
                copy(patch_root.parent / 'native/diagnostics' / name, owned / name)
        source = captured
        for name in sorted(touched & originals):
            copy(source / name, work / name)
        # The RMS variant changes a private copy of row_norm in its accepted
        # patch. The canonical header still uses the original upstream shader.
        for name, original in DERIVED.items():
            if (source / WGSL / name).exists():
                raise ValueError('Dedicated shader now exists upstream: ' + name)
            copy(source / WGSL / original, work / WGSL / name)
            inputs[str(WGSL / original)] = identity(work / WGSL / name)
        records = [apply_patch(work, patches, patch) for patch in selected]

        iq3s = private / 'iq3s'
        copy(source / WGSL / 'mul_mat_vec_acc.tmpl', iq3s / WGSL / 'mul_mat_vec_acc.tmpl')
        inputs[str(WGSL / 'mul_mat_vec_acc.tmpl')] = identity(iq3s / WGSL / 'mul_mat_vec_acc.tmpl')
        for name in extra[:2]:
            records.append(apply_patch(iq3s, patches, definitions[name]))
        words = private / 'word-template'
        copy(source / WGSL / 'common_decls.tmpl', words / WGSL / 'common_decls.tmpl')
        inputs[str(WGSL / 'common_decls.tmpl')] = identity(words / WGSL / 'common_decls.tmpl')
        records.append(apply_patch(words, patches, definitions[extra[2]]))
        template = words / WGSL / 'common_decls.tmpl'
        generator = source / WGSL / 'embed_wgsl.py'
        inputs[str(WGSL / 'embed_wgsl.py')] = identity(generator)

        shaders = _shader_inputs(source, work, iq3s, template)
        canonical = {path.name: path for path in input_files(source / WGSL) if path != generator}
        canonical['common_decls.tmpl'] = template
        shaders['ggml-webgpu-aligned-word-load-shaders.hpp'] = canonical
        shader_records = {}
        for header, files in shaders.items():
            shader_root = private / 'shaders' / header.removesuffix('.hpp')
            shader_root.mkdir(parents=True)
            for name, path in files.items():
                copy(path, shader_root / name)
            shader_records[header] = {name: identity(shader_root / name) for name in sorted(files)}
            generate(shader_root, None, prepared / header, generator=generator)
        for name in HOST_FILES:
            copy(work / name, prepared / Path(name).name)
        if gpu_profile:
            for name in ('webgpu-q8-diagnostic.h', 'ggml-webgpu-q8-diagnostic-sha256.hpp'):
                original = owned / name
                final = 'ggml-webgpu-q8-diagnostic.h' if name == 'webgpu-q8-diagnostic.h' else name
                copy(original, prepared / final)
                inputs['runtime/native/diagnostics/' + name] = identity(prepared / final)
        receipt = {'formatVersion': 1, 'gpuProfile': gpu_profile,
                   'inventorySha256': series.sha256, 'inputs': inputs,
                   'patches': records, 'shaders': shader_records,
                   'outputs': {path.name: identity(path) for path in sorted(prepared.iterdir())}}
        (prepared / RECEIPT).write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return output / 'ggml-webgpu.cpp'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-profile', action='store_true')
    args = parser.parse_args()
    prepare(args.source, args.output, gpu_profile=args.gpu_profile)


if __name__ == '__main__':
    main()
