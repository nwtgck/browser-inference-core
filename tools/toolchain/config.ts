import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import config from '#source/toolchain.json' with { type: 'json' };

export type AsyncifyPatchConfig = typeof config.emscriptenAsyncifyBigIntPatch;
export type ToolchainConfiguration = typeof config & { typescriptVersion: string };

/** Configuration is owned by the checkout that supplied the running tool. */
export async function loadToolchainConfiguration({ checkoutRoot }: { checkoutRoot: string }): Promise<ToolchainConfiguration> {
  const imported = fileURLToPath(import.meta.resolve('#source/toolchain.json'));
  if (resolve(imported) !== resolve(checkoutRoot, 'toolchain/config.json')) {
    throw new Error('Toolchain configuration must belong to the executing checkout');
  }
  return { ...config, typescriptVersion: ts.version };
}
