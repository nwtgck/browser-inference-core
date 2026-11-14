import { cp, copyFile, lstat, mkdir, readdir, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import type { PreparationOptions, PreparationReceipt } from '#dto/llama-cpp';
import type { FileIdentity } from '#dto/shared';

import { directory, regularFileIdentities, fileIdentity, isContained, optionalStat, regularFile, treeIdentity } from '#tools/shared/files';
import { withOutputDirectories } from '#tools/shared/output';
import { temporaryDirectory } from '#tools/shared/temporary-directory';
import { applyPatch, type PatchReceipt } from '#tools/source/patches';
import { generateWgsl } from '#tools/source/wgsl';
import { loadConfiguration, runtimePath } from '#runtimes/llama-cpp/tools/config';
import { modelPatches, mtmdPatches, patchDirectory, webgpuPatches, type SourceMode } from '#runtimes/llama-cpp/tools/recipe';

const webgpuBase = 'ggml/src/ggml-webgpu';
const shaderBase = `${webgpuBase}/wgsl-shaders`;
const derivedShaders = {
  'ssm_conv_silu.wgsl': 'ssm_conv.wgsl',
  'mul_mat_id_group.wgsl': 'mul_mat_id.wgsl',
  'mul_mat_id_gather_logical32.wgsl': 'mul_mat_id_gather.wgsl',
  'mul_mat_reg_tile_half_pairs.wgsl': 'mul_mat_reg_tile.wgsl',
  'mul_mat_half_pairs_decls.tmpl': 'mul_mat_decls.tmpl',
  'quant_half_pairs_inner_loops.tmpl': 'quant_inner_loops.tmpl',
} as const;

export type CompileInputs = { compileRoot: string; upstreamRoots: Readonly<Record<string, string>> };
export type PreparedSources = { root: string; receipt: PreparationReceipt };

async function copyInput({ source, destination }: { source: string; destination: string }): Promise<void> {
  await regularFile({ path: source });
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
}

async function copyCompanion({ source, destination }: { source: string; destination: string }): Promise<void> {
  if (await optionalStat({ path: destination })) {
    throw new Error(`Companion header now exists upstream: ${destination}`);
  }
  await copyInput({ source, destination });
}

async function copySourceTree({ source, destination }: { source: string; destination: string }): Promise<void> {
  await directory({ path: source });
  await cp(source, destination, {
    recursive: true,
    filter: async path => {
      if (basename(path).toLowerCase() === '.git') {
        return false;
      }
      if ((await lstat(path)).isSymbolicLink()) {
        throw new Error(`Linked upstream source is not accepted: ${path}`);
      }
      return true;
    },
  });
}

function requireSeparate({ source, output }: { source: string; output: string }): void {
  if (isContained({ root: source, path: output }) || isContained({ root: output, path: source })) {
    throw new Error(`Preparation output overlaps its inputs: ${output}`);
  }
}

function shaderInputs({ original, patched, iq3s, words }: { original: string; patched: string; iq3s: string; words: string }): Record<string, Record<string, string>> {
  const common = { 'common_decls.tmpl': join(words, shaderBase, 'common_decls.tmpl') };
  const accumulator = { 'mul_mat_vec_acc.tmpl': join(original, shaderBase, 'mul_mat_vec_acc.tmpl') };
  const edited = ({ names }: { names: readonly string[] }): Record<string, string> => Object.fromEntries(names.map(name => [name, join(patched, shaderBase, name)]));
  const upstream = ({ names }: { names: readonly string[] }): Record<string, string> => Object.fromEntries(names.map(name => [name, join(original, shaderBase, name)]));
  return {
    'ggml-webgpu-hc-shaders.hpp': edited({ names: ['hc_pre.wgsl', 'hc_post.wgsl'] }),
    'ggml-webgpu-moe-sum-shaders.hpp': edited({ names: ['moe_expert_sum.wgsl'] }),
    'ggml-webgpu-scale-unary-shaders.hpp': edited({ names: ['scale_unary.wgsl'] }),
    'ggml-webgpu-rms-scale-shaders.hpp': { 'rms_norm_scale.wgsl': join(patched, shaderBase, 'row_norm.wgsl') },
    'ggml-webgpu-ssm-conv-silu-shaders.hpp': edited({ names: ['ssm_conv_silu.wgsl'] }),
    'ggml-webgpu-router-tail-shaders.hpp': edited({ names: ['router_tail.wgsl', 'router_tail_merged.wgsl'] }),
    'ggml-webgpu-mul-mat-id-gather-logical32-shaders.hpp': edited({ names: ['mul_mat_id_gather_logical32.wgsl'] }),
    'ggml-webgpu-q8-subgroup-shaders.hpp': { ...common, ...accumulator, ...edited({ names: ['q8_subgroup_matvec.wgsl'] }) },
    'ggml-webgpu-iq3s-id-subgroup-shaders.hpp': { ...common, ...edited({ names: ['iq3s_id_subgroup.wgsl'] }), 'mul_mat_vec_acc.tmpl': join(iq3s, shaderBase, 'mul_mat_vec_acc.tmpl') },
    'ggml-webgpu-mul-mat-id-group-shaders.hpp': { ...common, ...edited({ names: ['mul_mat_id_group.wgsl'] }), ...upstream({ names: ['mul_mat_decls.tmpl', 'quant_inner_loops.tmpl'] }) },
    'ggml-webgpu-f32-matvec-epilogue-shaders.hpp': { ...common, ...accumulator, ...upstream({ names: ['mul_mat_vec_q_acc.tmpl'] }), ...edited({ names: ['mul_mat_vec_f32_epilogue.wgsl'] }) },
    'ggml-webgpu-quant-half-pairs-shaders.hpp': { ...common, ...edited({ names: ['mul_mat_reg_tile_half_pairs.wgsl', 'mul_mat_half_pairs_decls.tmpl', 'quant_half_pairs_inner_loops.tmpl'] }) },
  };
}

async function prepareWebgpu({ source, original, scratch, patchRoot, runtime, mode, python }: { source: string; original: string; scratch: string; patchRoot: string; runtime: string; mode: SourceMode; python: string }): Promise<{ patches: PatchReceipt[]; shaders: Record<string, Record<string, FileIdentity>> }> {
  for (const [name, originalName] of Object.entries(derivedShaders)) {
    try {
      await lstat(join(original, shaderBase, name));
      throw new Error(`Dedicated shader now exists upstream: ${name}`);
    } catch (error) {
      if (!(error instanceof Error && 'code' in error && error.code === 'ENOENT')) {
        throw error;
      }
    }
    await copyInput({ source: join(original, shaderBase, originalName), destination: join(source, shaderBase, name) });
  }
  const iq3s = join(scratch, 'iq3s');
  const words = join(scratch, 'word-template');
  await copyInput({ source: join(original, shaderBase, 'mul_mat_vec_acc.tmpl'), destination: join(iq3s, shaderBase, 'mul_mat_vec_acc.tmpl') });
  await copyInput({ source: join(original, shaderBase, 'common_decls.tmpl'), destination: join(words, shaderBase, 'common_decls.tmpl') });
  const patches: PatchReceipt[] = [];
  for (const patch of webgpuPatches({ mode })) {
    const sourceRoot = patch.target === 'iq3s' ? iq3s : patch.target === 'word-template' ? words : source;
    patches.push(await applyPatch({ sourceRoot, patchRoot, patch }));
  }
  const headers = shaderInputs({ original, patched: source, iq3s, words });
  const canonical: Record<string, string> = {};
  for (const name of (await readdir(join(original, shaderBase))).sort()) {
    if (name.endsWith('.wgsl') || name.endsWith('.tmpl')) {
      canonical[name] = join(original, shaderBase, name);
    }
  }
  canonical['common_decls.tmpl'] = join(words, shaderBase, 'common_decls.tmpl');
  headers['ggml-webgpu-aligned-word-load-shaders.hpp'] = canonical;
  const shaders: Record<string, Record<string, FileIdentity>> = {};
  const generator = join(original, shaderBase, 'embed_wgsl.py');
  // Each dedicated header is emitted once by the unmodified upstream generator.
  // The canonical family takes upstream shaders and the accepted word template;
  // e.g. the dedicated RMS shader must not replace upstream row_norm here.
  for (const [header, files] of Object.entries(headers)) {
    const shaderRoot = join(scratch, 'shaders', header.slice(0, -4));
    for (const [name, path] of Object.entries(files)) {
      await copyInput({ source: path, destination: join(shaderRoot, name) });
    }
    shaders[header] = await regularFileIdentities({ root: shaderRoot });
    await generateWgsl({ sourceRoot: shaderRoot, output: join(source, webgpuBase, header), generator, python });
  }
  switch (mode) {
  case 'profile': {
    for (const [name, output] of [
      ['webgpu-q8-diagnostic.h', 'ggml-webgpu-q8-diagnostic.h'],
      ['ggml-webgpu-q8-diagnostic-sha256.hpp', 'ggml-webgpu-q8-diagnostic-sha256.hpp'],
    ] as const) {
      await copyCompanion({ source: join(runtime, 'native/diagnostics', name), destination: join(source, webgpuBase, output) });
    }
    break;
  }
  case 'normal': break;
  default: {
    const _ex: never = mode;
    throw new Error(`Unknown source mode: ${String(_ex)}`);
  }
  }
  return { patches, shaders };
}

/** Prepare one coherent complete upstream tree; checkout bytes are never edited. */
export async function prepareSources({ inputs, output, options, python }: { inputs: CompileInputs; output: string; options: PreparationOptions; python: string }): Promise<PreparedSources> {
  const { target, webgpu, gpuProfile, fileRows, qsa, mtmdMode } = options;
  const upstream = inputs.upstreamRoots['llama'];
  if (upstream === undefined) {
    throw new Error('Source preparation requires the llama upstream root');
  }
  if (gpuProfile && !webgpu) {
    throw new Error('GPU profile source requires WebGPU');
  }
  if (mtmdMode === 'webgpu' && !webgpu) {
    throw new Error('WebGPU mtmd source requires WebGPU');
  }
  const original = await directory({ path: upstream });
  const destination = resolve(output);
  const runtime = await directory({ path: join(inputs.compileRoot, runtimePath) });
  requireSeparate({ source: original, output: destination });
  requireSeparate({ source: join(runtime, 'native'), output: destination });
  requireSeparate({ source: join(runtime, patchDirectory), output: destination });
  await directory({ path: dirname(destination), create: true });
  let receipt: PreparationReceipt | undefined;
  await withOutputDirectories({
    destinations: [destination],
    replace: true,
    preserveUnchanged: true,
    prepare: async ({ paths }) => {
      const prepared = paths[0];
      if (prepared === undefined) {
        throw new Error('Missing preparation output');
      }
      await using scratch = await temporaryDirectory({ prefix: join(dirname(prepared), '.llama-source-') });
      const root = join(prepared, 'llama.cpp');
      await copySourceTree({ source: original, destination: root });
      const inputs = await regularFileIdentities({ root });
      const owned = join(scratch.root, 'owned');
      await cp(join(runtime, 'native'), join(owned, 'native'), { recursive: true });
      const runtimeInputs = await regularFileIdentities({ root: owned });
      const patchRoot = join(scratch.root, 'patches');
      await cp(join(runtime, patchDirectory), patchRoot, { recursive: true });
      const patches: PatchReceipt[] = [];
      const selections: PreparationReceipt['selections'] = [];
      let shaders: Record<string, Record<string, FileIdentity>> = {};
      if (webgpu) {
        const nativeContext = join(scratch.root, 'native-context.cpp');
        switch (target) {
        case 'native': await copyInput({ source: join(root, 'src/llama-context.cpp'), destination: nativeContext });
          break;
        case 'wasm': break;
        default: {
          const _ex: never = target;
          throw new Error(`Unknown preparation target: ${String(_ex)}`);
        }
        }
        const canonicalRoot = join(scratch.root, 'original');
        await cp(join(root, shaderBase), join(canonicalRoot, shaderBase), { recursive: true });
        const result = await prepareWebgpu({ source: root, original: canonicalRoot, scratch: scratch.root, patchRoot, runtime: owned, mode: gpuProfile ? 'profile' : 'normal', python });
        patches.push(...result.patches);
        shaders = result.shaders;
        switch (target) {
        case 'native': {
        // The accepted checkpoint provider belongs to Emdawn. Preserve the
        // pre-existing native selection within the same coherent tree.
          const context = join(root, 'src/llama-context.cpp');
          const before = await fileIdentity({ path: context });
          await copyInput({ source: nativeContext, destination: context });
          selections.push({ file: 'src/llama-context.cpp', mode: 'upstream_native_context', reason: 'Native WebGPU retains the upstream context; browser checkpoint batching uses the Emdawn-only provider.', before, after: await fileIdentity({ path: context }) });
          break;
        }
        case 'wasm': break;
        default: {
          const _ex: never = target;
          throw new Error(`Unknown preparation target: ${String(_ex)}`);
        }
        }
      }
      if (fileRows) {
        patches.push(await applyPatch({ sourceRoot: root, patchRoot, patch: modelPatches.fileRows }));
        for (const name of ['file-backed-rows.h', 'file-backed-identity.h']) {
          await copyCompanion({ source: join(owned, 'native/io', name), destination: join(root, 'src/models', name) });
        }
      }
      if (qsa) {
        patches.push(await applyPatch({ sourceRoot: root, patchRoot, patch: modelPatches.qsa }));
      }
      patches.push(await applyPatch({ sourceRoot: root, patchRoot, patch: modelPatches.chat }));
      for (const patch of mtmdPatches({ mode: mtmdMode })) {
        patches.push(await applyPatch({ sourceRoot: join(root, 'tools/mtmd'), patchRoot, patch }));
      }
      const outputs = await regularFileIdentities({ root });
      receipt = { formatVersion: 1, options, upstream: treeIdentity({ files: inputs }), inputs, runtimeInputs, patches, selections, shaders, outputs };
      await writeFile(join(prepared, 'source-preparation.json'), `${JSON.stringify(receipt, undefined, 2)}\n`);
    },
  });
  if (receipt === undefined) {
    throw new Error('Source preparation did not produce a receipt');
  }
  return { root: join(destination, 'llama.cpp'), receipt };
}

/** Verify the existing prepared bytes; reporting never runs another composer. */
export async function verifyPreparedSources({ prepared }: { prepared: PreparedSources }): Promise<void> {
  const { root, receipt } = prepared;
  const files = await regularFileIdentities({ root });
  const expected = Object.entries(receipt.outputs);
  if (Object.keys(files).length !== expected.length || expected.some(([name, identity]) => files[name]?.bytes !== identity.bytes || files[name]?.sha256 !== identity.sha256)) {
    throw new Error('Prepared source bytes changed after preparation');
  }
}

export async function collectSourceFacts({ checkoutRoot, inputs, output, python }: { checkoutRoot: string; inputs: CompileInputs; output: string; python: string }): Promise<Record<string, PreparationReceipt>> {
  const result: Record<string, PreparationReceipt> = {};
  const preparedByOptions = new Map<string, PreparationReceipt>();
  const configuration = await loadConfiguration({ checkoutRoot });
  for (const [profile, options] of Object.entries(configuration.profiles)) {
    const preparation = { target: 'wasm' as const, webgpu: options.webgpu, gpuProfile: options.gpuProfile === true, fileRows: options.memory64, qsa: true, mtmdMode: options.webgpu ? 'webgpu' as const : 'cpu' as const };
    const key = JSON.stringify(preparation);
    let receipt = preparedByOptions.get(key);
    if (receipt === undefined) {
      const prepared = await prepareSources({ inputs, output: join(output, profile), python, options: preparation });
      await verifyPreparedSources({ prepared });
      receipt = prepared.receipt;
      preparedByOptions.set(key, receipt);
    }
    result[profile] = receipt;
  }
  return result;
}
