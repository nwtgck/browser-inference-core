import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { test } from 'node:test';
import { ESLint } from 'eslint';
import ts from 'typescript';

const root = resolve(import.meta.dirname, '../../..');
const eslint = new ESLint({
  cwd: root,
  overrideConfigFile: resolve(root, 'eslint.config.js'),
  // Policy snippets deliberately need no repository type project. Keep every
  // real boundary/layout/resource rule, disabling the type-service rules.
  overrideConfig: [{
    files: ['**/*.{ts,tsx,mts,cts,js,jsx,mjs,cjs}'],
    languageOptions: { parserOptions: { project: false, projectService: false } },
    rules: {
      '@typescript-eslint/restrict-template-expressions': 'off',
      'local-rules-switch/force-switch-for-union': 'off',
    },
  }],
});
const zodMessage = 'Zod belongs only in 00-dto/';
const privateMessage = 'Import DTO types and decoders through #dto/shared';
const parentMessage = 'instead of a parent-relative';

async function messages({ filePath, code }) {
  const [result] = await eslint.lintText(code, { filePath });
  assert.equal(result.messages.some(message => message.fatal), false, JSON.stringify(result.messages));
  return result.messages.filter(message => ['no-restricted-imports', 'no-restricted-syntax'].includes(message.ruleId));
}

async function assertBlocked({ filePath, code, reason }) {
  assert.ok((await messages({ filePath, code })).some(message => message.message.includes(reason)), `${filePath}: ${code}`);
}

const importForms = [
  value => `import { z } from ${JSON.stringify(value)};`,
  value => `import type { z } from ${JSON.stringify(value)};`,
  value => `import ${JSON.stringify(value)};`,
  value => `export { z } from ${JSON.stringify(value)};`,
  value => `export type { z } from ${JSON.stringify(value)};`,
  value => `export * from ${JSON.stringify(value)};`,
  value => `void import(${JSON.stringify(value)});`,
  value => `void import(\`${value}\`);`,
  value => `type Value = import(${JSON.stringify(value)}).Value;`,
  value => `require(${JSON.stringify(value)});`,
  value => `require(\`${value}\`);`,
  value => `import dependency = require(${JSON.stringify(value)});`,
];

for (const filePath of ['tools/dto-memory-fixture.ts', 'runtimes/example/tools/dto-memory-fixture.ts', 'runtimes/example/host/dto-memory-fixture.mjs', 'tests/tools/dto-memory-fixture.test.ts', '00-dto-extra/dto-memory-fixture.ts', 'tools/00-dto/dto-memory-fixture.ts']) {
  test(`Zod and nested subpaths stay outside sources beyond 00-dto/: ${filePath}`, async () => {
    for (const value of ['zod', 'zod/v4', 'zod/v4/core']) {
      for (const form of importForms) {
        await assertBlocked({ filePath, code: form(value), reason: zodMessage });
      }
    }
  });
}

for (const filePath of ['00-dto/index.ts', '00-dto/shared/future-structure.ts', '00-dto/scripts/inspect.mjs', '00-dto/new-runtime/nested/entry.js']) {
  test(`new DTO definitions, scripts and indexes may use Zod without registration: ${filePath}`, async () => {
    for (const value of ['zod', 'zod/v4', 'zod/v4/core']) {
      for (const form of importForms) {
        assert.deepEqual(await messages({ filePath, code: form(value) }), []);
      }
    }
  });
}

for (const filePath of ['tools/dto-memory-fixture.ts', 'runtimes/example/host/dto-memory-fixture.mjs']) {
  test(`private DTO aliases and raw paths stay private: ${filePath}`, async () => {
    for (const value of ['#dto-internal/shared/artifacts', '#dto/shared/artifacts', './00-dto/shared/artifacts', '../00-dto/shared/artifacts', '/source/00-dto/shared/artifacts']) {
      for (const form of importForms) {
        await assertBlocked({ filePath, code: form(value), reason: privateMessage });
      }
    }
  });
}

test('DTO implementation files may share private schemas inside the boundary', async () => {
  for (const filePath of ['00-dto/shared/abi.ts', '00-dto/shared/index.ts', '00-dto/llama-cpp/artifacts.ts']) {
    for (const form of importForms) {
      assert.deepEqual(await messages({ filePath, code: form('#dto-internal/shared/artifacts') }), []);
    }
  }
});

test('all three public DTO aliases stay usable without exposing raw modules', async () => {
  for (const filePath of ['tools/dto-memory-fixture.ts', 'runtimes/example/host/dto-memory-fixture.mjs']) {
    for (const value of ['#dto/shared', '#dto/llama-cpp', '#dto/stable-diffusion-cpp']) {
      for (const form of importForms) {
        assert.deepEqual(await messages({ filePath, code: form(value) }), []);
      }
    }
  }
});

test('similar ordinary package and directory names are not DTO or Zod imports', async () => {
  for (const value of ['zodiac', '@vendor/zod', './not-00-dto/helpers', '#dto-other/shared', './shared/00-dto-helper']) {
    for (const form of importForms) {
      assert.deepEqual(await messages({ filePath: 'tools/dto-memory-fixture.ts', code: form(value) }), []);
    }
  }
});

for (const filePath of ['tools/dto-memory-fixture.ts', 'runtimes/example/tools/dto-memory-fixture.ts', '00-dto/shared/artifacts.ts', '00-dto/shared/index.ts']) {
  test(`parent-relative imports remain forbidden in typed tooling and DTOs: ${filePath}`, async () => {
    for (const code of ['import { value } from "../shared/value";', 'import type { Value } from "../shared/value";', 'export { value } from "../shared/value";', 'void import("../shared/value");', 'void import(`../shared/value`);', 'void import("..");', 'type Value = import("../shared/value").Value;']) {
      await assertBlocked({ filePath, code, reason: parentMessage });
    }
  });
}

test('host declarations retain their necessary relative artifact imports', async () => {
  for (const filePath of ['runtimes/llama-cpp/host/index.d.ts', 'runtimes/example/host/runtime.mjs']) {
    for (const form of importForms) {
      assert.deepEqual(await messages({ filePath, code: form('../../shared/host/abi.js') }), []);
    }
  }
});

test('DTO restrictions keep layout, resource and named-argument policies active', async () => {
  for (const filePath of ['00-dto/shared/artifacts.ts', '00-dto/shared/index.ts', 'tools/dto-memory-fixture.ts']) {
    const config = await eslint.calculateConfigForFile(filePath);
    for (const name of ['local-rules-object-layout/object-layout', 'local-rules-function-spacing/function-spacing', 'local-rules-statement-layout/statement-layout', 'local-rules-test-resource/test-resource-management', 'local-rules-test-imports/no-test-support-imports', 'local-rules-named-args/require-named-args']) {
      assert.equal(config.rules[name][0], 2, `${filePath}: ${name}`);
    }
  }
});

function publicValueViolations({ files, fixture }) {
  const options = {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext,
    strict: true,
    skipLibCheck: true,
    noEmit: true,
    types: [],
  };
  const host = ts.createCompilerHost(options);
  const readFile = host.readFile.bind(host);
  const fileExists = host.fileExists.bind(host);
  if (fixture) {
    host.readFile = file => file === fixture.path ? fixture.code : readFile(file);
    host.fileExists = file => file === fixture.path || fileExists(file);
  }
  const program = ts.createProgram(files, options, host);
  const checker = program.getTypeChecker();
  const violations = [];
  for (const file of files) {
    const source = program.getSourceFile(file);
    assert.ok(source, file);
    const module = checker.getSymbolAtLocation(source);
    assert.ok(module, file);
    for (const exported of checker.getExportsOfModule(module)) {
      const value = exported.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(exported) : exported;
      if (!(value.flags & ts.SymbolFlags.Value)) {
        continue;
      }
      const type = checker.getTypeOfSymbolAtLocation(value, source);
      const signatures = type.getCallSignatures();
      if (type.getProperty('_zod') || signatures.length === 0 || signatures.some(signature => {
        const parameters = signature.getParameters();
        const declaration = signature.getDeclaration();
        if (parameters.length !== 1 || declaration?.parameters.length !== 1
          || declaration.parameters[0].questionToken || declaration.parameters[0].initializer) {
          return true;
        }
        const parameterType = checker.getTypeOfSymbolAtLocation(parameters[0], source);
        const input = parameterType.getProperty('input');
        return !input || Boolean(input.flags & ts.SymbolFlags.Optional)
          || checker.getTypeOfSymbolAtLocation(input, source).flags !== ts.TypeFlags.Unknown;
      })) {
        violations.push(`${file}: ${exported.name}`);
      }
    }
  }
  return violations;
}

test('actual public DTO exports have decoder types rather than raw schema values', () => {
  const files = ['shared', 'llama-cpp', 'stable-diffusion-cpp'].map(name => resolve(root, `00-dto/${name}/index.ts`));
  assert.deepEqual(publicValueViolations({ files }), []);
});

test('renaming a raw schema to a decoder-like name does not pass the public contract', () => {
  const path = resolve(root, '00-dto/public-memory-fixture.ts');
  const fixture = { path, code: 'export { fileIdentitySchema as decodeIdentity } from "#dto-internal/shared/artifacts";' };
  assert.deepEqual(publicValueViolations({ files: [path], fixture }), [`${path}: decodeIdentity`]);
});
