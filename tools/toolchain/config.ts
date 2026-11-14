import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { containedPath, fileIdentity, type FileIdentity } from '#tools/shared/files';
import config from '#source/toolchain.json' with { type: 'json' };

export type SdkPatch = FileIdentity & { file: string };
export type ToolchainConfiguration = Omit<typeof config, 'emscriptenTypeScriptPatch' | 'emscriptenAsyncifyBigIntPatch'> & { typescriptVersion: string; emscriptenTypeScriptPatch: SdkPatch; emscriptenAsyncifyBigIntPatch: SdkPatch };

async function observePatch({ checkoutRoot, file }: { checkoutRoot: string; file: string }): Promise<SdkPatch> {
  return { file, ...await fileIdentity({ path: containedPath({ root: checkoutRoot, relative: file }) }) };
}

/** Configuration is owned by the checkout that supplied the running tool. */
export async function loadToolchainConfiguration({ checkoutRoot }: { checkoutRoot: string }): Promise<ToolchainConfiguration> {
  const imported = fileURLToPath(import.meta.resolve('#source/toolchain.json'));
  if (resolve(imported) !== resolve(checkoutRoot, 'toolchain/config.json')) {
    throw new Error('Toolchain configuration must belong to the executing checkout');
  }
  const emscriptenTypeScriptPatch = await observePatch({ checkoutRoot, file: config.emscriptenTypeScriptPatch });
  const emscriptenAsyncifyBigIntPatch = await observePatch({ checkoutRoot, file: config.emscriptenAsyncifyBigIntPatch });
  return { ...config, typescriptVersion: ts.version, emscriptenTypeScriptPatch, emscriptenAsyncifyBigIntPatch };
}
