import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import type { Options } from 'yargs';
import type { RuntimeName } from '#dto/shared';

export const requiredString = {
  type: 'string',
  demandOption: true,
  requiresArg: true,
  coerce: (value: unknown) => {
    if (typeof value !== 'string' || value.length === 0) {
      throw new Error('Expected one nonempty string option value');
    }
    return value;
  },
} as const satisfies Options;

/** Identical cache options for the aggregate and independently selected runtime CLIs. */
export const cachePlanOptions = {
  'checkout-root': requiredString,
  'work-root': requiredString,
  profile: requiredString,
  variant: requiredString,
  'source-commit': requiredString,
  'ccache-version': requiredString,
  environment: requiredString,
  os: requiredString,
  arch: requiredString,
  output: requiredString,
} as const;

/** The caller selects and validates its own runtime configuration first. */
export async function runCachePlan({ runtime, checkoutRoot, workRoot, profile, variant, profileConfiguration, variantConfiguration, sourceCommit, ccacheVersion, environment, os, arch }: {
  runtime: RuntimeName; checkoutRoot: string; workRoot: string; profile: string; variant: string;
  profileConfiguration: object; variantConfiguration: object; sourceCommit: string; ccacheVersion: string;
  environment: string; os: string; arch: string;
}) {
  const selectedCheckout = resolve(checkoutRoot);
  const workspace = resolve(workRoot);
  const [{ loadToolchainConfiguration }, { cachePlan, configureCache }] = await Promise.all([
    import('#tools/toolchain/config'), import('#tools/toolchain/cache'),
  ]);
  const toolchain = await loadToolchainConfiguration({ checkoutRoot: selectedCheckout });
  const plan = cachePlan({
    runtime,
    profile,
    variant,
    profileConfiguration,
    variantConfiguration,
    sourceCommit,
    toolchain,
    ccacheVersion,
    platform: { os, arch, environment },
    workspace,
  });
  return { ...plan, environment: await configureCache({ workRoot: workspace, plan }) };
}

export async function writeCachePlan({ value, output }: { value: Awaited<ReturnType<typeof runCachePlan>>; output: string }): Promise<void> {
  const target = resolve(output);
  const text = `${JSON.stringify(value, undefined, 2)}\n`;
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, text);
  process.stdout.write(text);
}
