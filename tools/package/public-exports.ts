import { mkdir, realpath, symlink, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
import ts from 'typescript';
import { artifactDirectories, packageName, runtimeNames } from '#tools/shared/contracts';
import { runProcess } from '#tools/shared/process';
import { temporaryDirectory } from '#tools/shared/temporary-directory';
import type { RuntimeManifests } from './aggregate.js';

const commonImports = [
  ['host', 'host/index.mjs'], ['host/abi', 'host/abi.mjs'],
  ['api/functions', 'api/functions.mjs'], ['api/schema.mjs', 'api/schema.mjs'],
] as const;

/** Check the assembled package from the perspective of TypeScript and Node ESM consumers. */
export async function checkPublicExports({ directory, runtimes }: { directory: string; runtimes: RuntimeManifests }): Promise<void> {
  const root = await realpath(directory);
  const imports: string[] = [];
  const runtimeImports: { specifier: string; url: string }[] = [];
  for (const runtime of runtimeNames) {
    const prefix = `${packageName}/${runtime}/`;
    const runtimeRoot = join(root, artifactDirectories[runtime]);
    const manifest = runtimes[runtime];
    // Select public entry points independently of the export map being checked.
    for (const [entry, target] of commonImports) {
      const specifier = prefix + entry;
      imports.push(specifier);
      runtimeImports.push({ specifier, url: pathToFileURL(join(runtimeRoot, target)).href });
    }
    switch (runtime) {
    case 'llama-cpp':
      imports.push(prefix + 'api/extensions');
      break;
    case 'stable-diffusion-cpp':
    case 'speech-cpp':
      break;
    default: {
      const exhaustive: never = runtime;
      throw new Error(String(exhaustive));
    }
    }
    for (const [profile, definition] of Object.entries(manifest.profiles)) {
      for (const variant of Object.keys(definition.variants)) {
        const core = `profiles/${profile}/${variant}/core`;
        imports.push(`${prefix}${core}.mjs`);
        for (const extension of ['mjs', 'wasm']) {
          const target = `${core}.${extension}`;
          runtimeImports.push({ specifier: prefix + target, url: pathToFileURL(join(runtimeRoot, target)).href });
        }
      }
    }
    for (const { path } of manifest.files) {
      if (path.startsWith('profiles/') && path.endsWith('.functions.d.ts')) {
        imports.push(prefix + path.slice(0, -'.d.ts'.length));
      }
    }
  }

  await using consumer = await temporaryDirectory({ prefix: join(tmpdir(), 'bicore-public-exports-') });
  const installed = join(consumer.root, 'node_modules', packageName);
  await mkdir(dirname(installed));
  await symlink(root, installed, 'junction');

  // TypeScript can succeed through the "types" condition even when "import"
  // targets a different payload. Node's resolver may return a missing file URL.
  await runProcess({
    command: process.execPath,
    args: ['--input-type=module', '--eval', `\
import { lstatSync } from 'node:fs';
for (const { specifier, url } of ${JSON.stringify(runtimeImports)}) {
  if (!lstatSync(new URL(url), { throwIfNoEntry: false })?.isFile()) {
    throw new Error('Public runtime import ' + specifier + ' is missing its packaged file: ' + url);
  }
  const actual = import.meta.resolve(specifier);
  if (actual !== url) {
    throw new Error('Public runtime import ' + specifier + ' resolved to ' + actual + ', expected ' + url);
  }
}
`],
    cwd: consumer.root,
    env: { kind: 'inherit', overrides: { NODE_OPTIONS: undefined }, removePrefixes: [] },
  });

  const file = join(consumer.root, 'consumer.mts');
  const source = imports.map((specifier, index) => `import type * as Api${index} from ${JSON.stringify(specifier)};`).join('\n');
  await writeFile(file, source);

  const program = ts.createProgram([file], {
    target: ts.ScriptTarget.ES2023,
    module: ts.ModuleKind.NodeNext,
    strict: true,
    noEmit: true,
    preserveSymlinks: true,
    types: [],
    lib: ['lib.es2023.d.ts', 'lib.dom.d.ts', 'lib.dom.iterable.d.ts'],
  });
  const diagnostics = ts.getPreEmitDiagnostics(program);
  if (diagnostics.length > 0) {
    const details = ts.formatDiagnostics(diagnostics, {
      getCanonicalFileName: path => path,
      getCurrentDirectory: () => consumer.root,
      getNewLine: () => '\n',
    });
    throw new Error(`Public package declarations do not resolve:\n${details}`);
  }
}
