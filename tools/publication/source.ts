import { join } from 'node:path';
import type { CompilerInputs, PatchReceipt, SourceIdentity, SourceInputs, SourceFacts, TreeIdentity } from '#dto/shared';
import type { PreparationReceipt } from '#dto/llama-cpp';
import type { PreparedSources } from '#dto/stable-diffusion-cpp';
import { observeSource, snapshotCompileInputs, exportCommit } from '#tools/source/capture';
import { treeIdentity } from '#tools/shared/files';
import { assertSame } from '#tools/package/files';
import type { RuntimeManifests } from '#tools/package/aggregate';
import { collectSourceFacts } from '#runtimes/llama-cpp/tools/prepare';
import { loadConfiguration as loadLlamaConfiguration, sourceCaptureSelection as llamaSelection, upstreamPath, upstreamCommit } from '#runtimes/llama-cpp/tools/config';
import { prepareRuntimeSources } from '#runtimes/stable-diffusion-cpp/tools/prepare';
import { loadConfiguration as loadImageConfiguration, upstreamFacts } from '#runtimes/stable-diffusion-cpp/tools/config';
import { prepareRuntimeSources as prepareSpeechRuntimeSources } from '#runtimes/speech-cpp/tools/prepare';
import { loadConfiguration as loadSpeechConfiguration, upstreamFacts as speechUpstreamFacts } from '#runtimes/speech-cpp/tools/config';
import { sourceCaptureSelection as speechSelection } from '#runtimes/speech-cpp/tools/build';
import { sourceCaptureSelection as imageSelection } from '#runtimes/stable-diffusion-cpp/tools/build';
import { captureTreeFingerprint } from '#tools/source/fingerprint';

function identityFacts({ identity }: { identity: SourceIdentity }) {
  return typeof identity.bytes === 'number' && typeof identity.sha256 === 'string' ? { bytes: identity.bytes, sha256: identity.sha256 } : { symlink: identity.symlink };
}

function fileFacts({ files }: { files: Readonly<Record<string, SourceIdentity>> }) {
  return Object.fromEntries(Object.entries(files).map(([name, identity]) => [name, identityFacts({ identity })]));
}

function treeFacts({ tree }: { tree: TreeIdentity }) {
  return { files: tree.files, sha256: tree.sha256 };
}

function patchFacts({ patches }: { patches: readonly PatchReceipt[] }) {
  return patches.map(({ file, target, mode, reason, sha256, files }) => ({ file, target, mode, reason, sha256, files: files.map(({ path, beforeSha256, afterSha256 }) => ({ path, beforeSha256, afterSha256 })) }));
}

function llamaPreparationFacts({ receipt }: { receipt: PreparationReceipt }) {
  const { target, webgpu, gpuProfile, fileRows, qsa, mtmdMode } = receipt.options;
  return { formatVersion: receipt.formatVersion, options: { target, webgpu, gpuProfile, fileRows, qsa, mtmdMode }, upstream: treeFacts({ tree: receipt.upstream }), inputs: fileFacts({ files: receipt.inputs }), runtimeInputs: fileFacts({ files: receipt.runtimeInputs }), patches: patchFacts({ patches: receipt.patches }), selections: receipt.selections.map(({ file, mode, reason, before, after }) => ({ file, mode, reason, before: identityFacts({ identity: before }), after: identityFacts({ identity: after }) })), shaders: Object.fromEntries(Object.entries(receipt.shaders).map(([name, files]) => [name, fileFacts({ files })])), outputs: fileFacts({ files: receipt.outputs }) };
}

/** Compare source identities and preparation decisions; unrelated saved metadata is retained. */
export function verifyLlamaPreparationReceipt({ recorded, expected }: { recorded: PreparationReceipt; expected: PreparationReceipt }): void {
  assertSame({ actual: llamaPreparationFacts({ receipt: recorded }), expected: llamaPreparationFacts({ receipt: expected }), message: 'Prepared llama sources differ from selected inputs' });
}

function imagePreparationFacts({ receipt }: { receipt: PreparedSources }) {
  return { runtime: receipt.runtime, upstreams: upstreamFacts({ upstreams: receipt.upstreams }), patches: patchFacts({ patches: receipt.patches }), vendorCheckoutModified: receipt.vendorCheckoutModified };
}

export function verifyCompiledInputs({ compiled, source, prepared }: { compiled: CompilerInputs; source: Readonly<Record<string, SourceIdentity>>; prepared: Readonly<Record<string, SourceIdentity>> }): void {
  let preparedUnits = 0;
  for (const group of [compiled.translationUnits, ...Object.values(compiled.targetSources)]) {
    for (const [name, identity] of Object.entries(group)) {
      if (name.startsWith('prepared/')) {
        const expected = prepared[name.slice('prepared/'.length)];
        if (!expected) {
          throw new Error(`Compiled source is absent from prepared receipt: ${name}`);
        }
        assertSame({ actual: identityFacts({ identity }), expected: identityFacts({ identity: expected }), message: `Actual compiled input differs from prepared source: ${name}` });
        preparedUnits++;
      } else if (name.startsWith('compile/')) {
        const expected = source[name.slice('compile/'.length)];
        assertSame({ actual: identityFacts({ identity }), expected: expected && identityFacts({ identity: expected }), message: `Actual compiled input differs from source snapshot: ${name}` });
      }
    }
  }
  if (!preparedUnits) {
    throw new Error('Missing actual prepared compilation inputs');
  }
}

function verifySourceInputs({ recorded, expected, sourceCommit }: { recorded: SourceInputs; expected: Readonly<Record<string, SourceIdentity>>; sourceCommit: string }): void {
  if (recorded.mode !== 'release' || !recorded.publishable || recorded.sourceCommit !== sourceCommit) {
    throw new Error('Build captured another source or a nonpublishable local input');
  }
  assertSame({ actual: fileFacts({ files: recorded.files }), expected: fileFacts({ files: expected }), message: 'Compiler input snapshot differs from selected Git inputs' });
}

/** One release-source verification at freeze. It never creates a second Node environment. */
export async function verifyPublicationSources({ sourceRoot, workRoot, sourceCommit, manifests, python }: { sourceRoot: string; workRoot: string; sourceCommit: string; manifests: RuntimeManifests; python: string }): Promise<SourceFacts> {
  // These loaders verify that the supplied checkout owns the executing modules
  // and their static configuration. The snapshot is only compiler input.
  await loadLlamaConfiguration({ checkoutRoot: sourceRoot });
  const imageConfiguration = await loadImageConfiguration({ checkoutRoot: sourceRoot });
  const speechConfiguration = await loadSpeechConfiguration({ checkoutRoot: sourceRoot });
  const source = await observeSource({ checkoutRoot: sourceRoot, mode: { kind: 'release', expectedCommit: sourceCommit } });
  const llamaSnapshot = await snapshotCompileInputs({ source, destination: join(workRoot, 'llama-cpp/compile-inputs'), ...llamaSelection({ kind: 'wasm' }), replace: false });
  const imageSnapshot = await snapshotCompileInputs({ source, destination: join(workRoot, 'stable-diffusion-cpp/compile-inputs'), ...imageSelection(), replace: false });
  const speechSnapshot = await snapshotCompileInputs({ source, destination: join(workRoot, 'speech-cpp/compile-inputs'), ...speechSelection(), replace: false });
  const llamaCommit = await upstreamCommit({ checkoutRoot: sourceRoot, revision: sourceCommit });
  if (manifests['llama-cpp'].llamaCommit !== llamaCommit) {
    throw new Error('Llama package upstream differs from selected gitlink');
  }
  const upstream = join(workRoot, 'llama-cpp/upstream');
  await exportCommit({ sourceRoot: join(sourceRoot, upstreamPath), revision: llamaCommit, destination: upstream });
  const prepared = await collectSourceFacts({ checkoutRoot: sourceRoot, inputs: { compileRoot: llamaSnapshot.compileRoot, upstreamRoots: { llama: upstream } }, output: join(workRoot, 'llama-cpp/prepared'), python });
  const llamaVariants: string[] = [];
  for (const [profile, data] of Object.entries(manifests['llama-cpp'].profiles)) {
    for (const [variant, information] of Object.entries(data.variants)) {
      const expected = prepared[profile];
      if (!expected) {
        throw new Error(`Missing source preparation for ${profile}`);
      }
      verifyLlamaPreparationReceipt({ recorded: information.preparedSources, expected });
      verifySourceInputs({ recorded: information.sourceInputs, expected: llamaSnapshot.files, sourceCommit });
      verifyCompiledInputs({ compiled: information.compiledInputs, source: llamaSnapshot.files, prepared: expected.outputs });
      llamaVariants.push(`${profile}/${variant}`);
    }
  }
  const image = (await prepareRuntimeSources({ snapshot: imageSnapshot, declarations: imageConfiguration.upstreams, output: join(workRoot, 'stable-diffusion-cpp/prepared'), replace: false })).facts;
  const imageVariants: string[] = [];
  for (const [profile, data] of Object.entries(manifests['stable-diffusion-cpp'].profiles)) {
    for (const [variant, information] of Object.entries(data.variants)) {
      verifySourceInputs({ recorded: information.sourceInputs, expected: imageSnapshot.files, sourceCommit });
      assertSame({ actual: treeFacts({ tree: information.preparedSources }), expected: treeFacts({ tree: image.identity }), message: 'Prepared image sources differ from selected inputs' });
      assertSame({ actual: imagePreparationFacts({ receipt: information.patches }), expected: imagePreparationFacts({ receipt: image.receipt }), message: 'Image patch receipt differs from selected sources' });
      assertSame({ actual: upstreamFacts({ upstreams: information.upstreams }), expected: upstreamFacts({ upstreams: image.receipt.upstreams }), message: 'Image upstreams differ from selected gitlinks' });
      verifyCompiledInputs({ compiled: information.compiledInputs, source: imageSnapshot.files, prepared: image.files });
      imageVariants.push(`${profile}/${variant}`);
    }
  }
  const speech = (await prepareSpeechRuntimeSources({ snapshot: speechSnapshot, declarations: speechConfiguration.upstreams, output: join(workRoot, 'speech-cpp/prepared'), replace: false })).facts;
  const speechVariants: string[] = [];
  for (const [profile, data] of Object.entries(manifests['speech-cpp'].profiles)) {
    for (const [variant, information] of Object.entries(data.variants)) {
      verifySourceInputs({ recorded: information.sourceInputs, expected: speechSnapshot.files, sourceCommit });
      assertSame({ actual: treeFacts({ tree: information.preparedSources }), expected: treeFacts({ tree: speech.identity }), message: 'Prepared speech sources differ from selected inputs' });
      assertSame({ actual: { runtime: information.patches.runtime, upstreams: speechUpstreamFacts({ upstreams: information.patches.upstreams }), patches: patchFacts({ patches: information.patches.patches }), vendorCheckoutModified: information.patches.vendorCheckoutModified }, expected: { runtime: speech.receipt.runtime, upstreams: speechUpstreamFacts({ upstreams: speech.receipt.upstreams }), patches: patchFacts({ patches: speech.receipt.patches }), vendorCheckoutModified: speech.receipt.vendorCheckoutModified }, message: 'Speech patch receipt differs from selected sources' });
      assertSame({ actual: speechUpstreamFacts({ upstreams: information.upstreams }), expected: speechUpstreamFacts({ upstreams: speech.receipt.upstreams }), message: 'Speech upstreams differ from selected gitlinks' });
      verifyCompiledInputs({ compiled: information.compiledInputs, source: speechSnapshot.files, prepared: speech.files });
      speechVariants.push(`${profile}/${variant}`);
    }
  }
  return {
    sourceCommit,
    treeFingerprint: await captureTreeFingerprint({ sourceRoot, sourceCommit }),
    runtimes: {
      'llama-cpp': { upstream: llamaCommit, preparations: Object.fromEntries(Object.entries(prepared).map(([name, receipt]) => [name, { upstream: receipt.upstream, options: receipt.options, patches: receipt.patches.map(({ file, target, mode, sha256, reason }) => ({ file, target, mode, sha256, reason })), prepared: treeIdentity({ files: receipt.outputs }) }])), verifiedVariants: llamaVariants },
      'stable-diffusion-cpp': { upstreams: image.receipt.upstreams, preparation: image.identity, patches: image.receipt.patches.map(({ file, target, mode, sha256, reason }) => ({ file, target, mode, sha256, reason })), verifiedVariants: imageVariants },
      'speech-cpp': { upstreams: speech.receipt.upstreams, preparation: speech.identity, patches: speech.receipt.patches.map(({ file, target, mode, sha256, reason }) => ({ file, target, mode, sha256, reason })), verifiedVariants: speechVariants },
    },
  };
}
