import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import test from 'node:test';
import { ESLint, Linter } from 'eslint';
import * as parser from '@typescript-eslint/parser';
import ts from 'typescript';
import policy from '#lint/node-env-policy';
import { repoRoot } from '#lint/node-env-ast';

function lint({ code, filename = 'tools/example.ts', program }) {
  const messages = new Linter({ cwd: repoRoot }).verify(code, [{
    ...policy,
    files: ['**'],
    languageOptions: { parser, sourceType: 'module', parserOptions: program ? { programs: [program] } : { project: false, projectService: false } },
  }], { filename: resolve(repoRoot, filename) });
  assert.deepEqual(messages.filter(message => message.fatal), []);
  assert.ok(messages.every(message => message.fix === undefined && message.suggestions === undefined));
  return messages.map(message => message.messageId);
}

test('the reader source permits only lazy, identity-preserving named declarations', () => {
  for (const code of [
    'export const READ_ENV_PATH = () => process.env.PATH;',
    '// A harmless comment\nexport const READ_ENV_PATH =\n  () => process.env.PATH;\n',
    'export const READ_ENV_X0_B_2 = (() => (process.env.X0_B_2));',
  ]) {
    assert.deepEqual(lint({ code, filename: 'tools/shared/node-env.ts' }), [], code);
  }
  for (const code of [
    'export const READ_ENV_PATH = process.env.PATH;',
    'export const READ_ENV_PATH = () => process.env.HOME;',
    'export const READ_ENV_PATH = () => process.env.PATH ?? "fallback";',
    'export const READ_ENV_PATH = () => process.env["PATH"];',
    'export const READ_ENV_PATH = async () => process.env.PATH;',
    'export const READ_ENV_PATH = () => { return process.env.PATH; };',
    'export const READ_ENV_PATH = () => process.env.PATH as string;',
    'export const READ_ENV_PATH = (): string | undefined => process.env.PATH;',
    'export const READ_ENV_PATH: () => string = () => process.env.PATH;',
    'export let READ_ENV_PATH = () => process.env.PATH;',
    'export const GET_PATH = () => process.env.PATH;',
    'export const READ_ENV_ALL = () => process.env;',
    'export const READ_ENV_SNAPSHOT = () => ({ ...process.env });',
    'export const READ_ENV_PATH = () => process.env.PATH = "changed";',
    'export const READ_ENV_PATH = () => delete process.env.PATH;',
    'import process from "node:process"; export const READ_ENV_PATH = () => process.env.PATH;',
    'export type Environment = NodeJS.ProcessEnv;',
    'export default () => process.env.PATH;',
  ]) {
    assert.ok(lint({ code, filename: 'tools/shared/node-env.ts' }).includes('registryShape'), code);
  }
  for (const filename of ['tools/other/node-env.ts', 'tools/shared/node-env.ts?raw', 'tests/tools/shared/node-env.ts']) {
    assert.ok(lint({ code: 'export const READ_ENV_PATH = () => process.env.PATH;', filename }).includes('directEnvironment'), filename);
  }
});

test('owned Node TS, JS and MJS tests and build tools all reject direct environment reads and writes', () => {
  for (const filename of ['tools/run.ts', 'tools/run.js', 'tests/tools/env.test.ts', 'tests/tools/env.test-support.ts', 'tests/tools/env.test.mjs', 'tests/runtimes/llama_cpp/browser-smoke.mjs', 'eslint-local-rules/example.js', '.github/tests/env.test.ts', '.github/tests/env.test-support.ts']) {
    for (const code of ['process.env.X;', 'process.env.X = "value";', 'delete process.env.X;', 'const copy = { ...process.env };']) {
      assert.ok(lint({ code, filename }).includes('directEnvironment'), `${filename}: ${code}`);
    }
  }
});

test('aliases, destructuring, global roots and typed process parameters cannot hide an environment', () => {
  for (const code of [
    'process.env;', 'process.env.X++;', 'process.env.X ??= "value";', 'process.env = {};',
    'const p = process; const q = p; q.env.X;',
    'let p = {}; p = process; p.env.X;',
    'const p = condition ? {} : process; p.env.X;',
    'const p = (0, process); p.env.X;',
    'globalThis.process.env.X;', 'global.process.env.X;', 'self.process.env.X;',
    'const key = "en" + "v"; process[key].X;',
    'const { process: { env } } = globalThis;', 'const { env: e = {} } = process;',
    'let e; ({ env: e } = process);',
    'const holder = { p: process }; holder.p.env.X;',
    'const { runtime: { env } } = { runtime: process }; env.X;',
    'import p from "node:process"; p.env.X;',
    'import * as p from "process"; p.default.env.X;',
    'import { env } from "node:process";',
    'function read(p: NodeJS.Process) { return p.env.X; }',
    'type P = typeof process; function read({ env }: P) { return env; }',
    'process.loadEnvFile();',
  ]) {
    assert.ok(lint({ code }).includes('directEnvironment'), code);
  }
  for (const code of ['process[key];', 'const p = process; p[key] = {};', 'const { [key]: value } = process;']) {
    assert.ok(lint({ code }).includes('dynamicProcessKey'), code);
  }
});

test('reflection, opaque process loads and environment-property mocks are rejected', () => {
  for (const code of [
    'Reflect.get(process, "env");', 'const get = Reflect.get; get.call(null, process, "env");',
    'Object.defineProperty(process, "env", { value: {} });', 'Object.assign(process, arbitrary);',
    'Object.entries(process);', 'const { ...rest } = process;', 'JSON.stringify(process);',
    'context.mock.property(process, "env", {});', 'context.mock.method(process, "loadEnvFile");',
  ]) {
    assert.ok(lint({ code, filename: 'tests/tools/env.test.ts' }).includes('processReflection'), code);
  }
  for (const code of [
    'require("node:process");', 'module.require("process");', '(await import("node:process")).env;',
    'process.getBuiltinModule("process");',
    'import { createRequire } from "node:module"; const load = createRequire(import.meta.url); load("process");',
    'export * from "node:process";', 'export { env } from "process";', 'export default process;',
  ]) {
    assert.ok(lint({ code }).includes('processModule'), code);
  }
  assert.ok(lint({ code: 'vi.stubEnv("X", "value");', filename: 'tests/tools/env.test.ts' }).includes('testStubOnly'));
});

test('unrelated environment objects, Node process members and virtual processes remain ordinary values', () => {
  for (const code of [
    'process.argv; process.stdout.write("text"); process.cwd(); process.exitCode = 1;',
    'import { argv } from "node:process"; export { argv };',
    'type P = typeof process; type Environment = typeof process.env;',
    'const process = { env: new Map() }; process.env.get("USER");',
    'function update(process: VirtualProcess) { process.env.set("X", "value"); }',
    'const environment: NodeJS.ProcessEnv = {}; environment.X = "value"; delete environment.X;',
    'const copy = { ...environment, GIT_DIR: directory }; delete copy.GIT_INDEX_FILE;',
    'import.meta.env.DEV;',
  ]) {
    assert.deepEqual(lint({ code }), [], code);
  }
});

test('reader imports use Node directory ownership rather than individual caller registrations', () => {
  const code = 'import { READ_ENV_PATH } from "#tools/shared/node-env"; READ_ENV_PATH();';
  for (const filename of ['tools/new-feature.ts', 'runtimes/future/tools/entry.ts', 'tests/tools/new.test.mjs', 'tests/runtimes/llama_cpp/browser-smoke.mjs', 'scripts/future.mjs', '.github/scripts/future.js', '.github/tests/new.test.ts', '.github/tests/helper.test-support.ts', 'eslint.config.js']) {
    assert.deepEqual(lint({ code, filename }), [], filename);
  }
  for (const filename of ['runtimes/llama-cpp/host/index.mjs', 'runtimes/example/host/tools/runtime.ts', '00-dto/shared/value.ts', 'src/browser.ts']) {
    assert.ok(lint({ code, filename }).includes('registryImport'), filename);
  }
  for (const code of [
    'import { READ_ENV_PATH as read } from "#tools/shared/node-env";',
    'import * as environment from "#tools/shared/node-env";',
    'import environment from "#tools/shared/node-env";', 'import "#tools/shared/node-env";',
    'export { READ_ENV_PATH } from "#tools/shared/node-env";', 'export * from "#tools/shared/node-env";',
    'import("#tools/shared/node-env");', 'require("#tools/shared/node-env");',
  ]) {
    assert.ok(lint({ code }).includes('registryImport'), code);
  }
  for (const source of ['./shared/node-env.js', './shared/node-env.ts?raw', '../build/tooling/tools/shared/node-env.js']) {
    assert.ok(lint({ code: `import { READ_ENV_PATH as read } from ${JSON.stringify(source)};` }).includes('registryImport'), source);
  }
});

test('ordinary aliases cannot re-export imported readers while local reader tables remain valid', () => {
  const imported = 'import { READ_ENV_PATH } from "#tools/shared/node-env"; ';
  for (const code of [
    'export { READ_ENV_PATH };', 'export { READ_ENV_PATH as readPath };',
    'export const readPath = READ_ENV_PATH;', 'const read = READ_ENV_PATH; export default read;',
    'const read = READ_ENV_PATH; export { read };',
    'const read = condition ? globalThis : READ_ENV_PATH; export { read };',
  ]) {
    assert.ok(lint({ code: imported + code }).includes('registryImport'), code);
  }
  for (const code of ['const table = [{ read: READ_ENV_PATH }]; table[0].read();', 'export const selectedPath = READ_ENV_PATH();']) {
    assert.deepEqual(lint({ code: imported + code }), [], code);
  }
});

test('browser and data modules cannot import any Node tooling barrel', () => {
  for (const source of ['#tools/shared/environment-barrel', '#runtimes/llama-cpp/tools/environment-barrel', '../../../tools/shared/environment-barrel.js']) {
    assert.ok(lint({ code: `import { value } from ${JSON.stringify(source)};`, filename: 'runtimes/llama-cpp/host/index.mjs' }).includes('nodeToolImport'), source);
  }
  assert.ok(lint({ code: 'import { value } from "#tools/shared/environment-barrel";', filename: '00-dto/shared/value.ts' }).includes('nodeToolImport'));
  assert.deepEqual(lint({ code: 'import { value } from "#tools/shared/environment-barrel";' }), []);
  assert.deepEqual(lint({ code: 'import { value } from "#runtimes/llama-cpp/host/index.mjs";', filename: 'runtimes/stable-diffusion-cpp/host/index.mjs' }), []);
});

test('actual TypeScript Node Process provenance is distinct from plain and virtual environments', () => {
  const filename = resolve(repoRoot, 'tools/env-provenance-fixture.ts');
  const source = `\
declare function obtain(): NodeJS.Process;
obtain().env.X;
const processes = [obtain()];
processes.map(value => value.env.X);
interface VirtualProcess { env: Map<string, string> }
const virtual: VirtualProcess[] = [];
virtual.map(process => process.env.get('USER'));
const environment: NodeJS.ProcessEnv = {};
environment.X = 'value';
function nested({ runtime: { env } }: { runtime: NodeJS.Process }) { return env.X; }
function virtualNested({ runtime: { env } }: { runtime: VirtualProcess }) { return env.get('USER'); }
function plainNested({ runtime: { env } }: { runtime: { env: NodeJS.ProcessEnv } }) { return env.X; }
`;
  const options = { target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext, types: ['node'], typeRoots: [resolve(repoRoot, 'node_modules/@types')], strict: true, skipLibCheck: true, noEmit: true };
  const host = ts.createCompilerHost(options);
  const getSourceFile = host.getSourceFile.bind(host);
  host.getSourceFile = (file, languageVersion, onError, shouldCreateNewSourceFile) => file === filename ? ts.createSourceFile(file, source, languageVersion, true) : getSourceFile(file, languageVersion, onError, shouldCreateNewSourceFile);
  const program = ts.createProgram([filename], options, host);
  assert.deepEqual(ts.getPreEmitDiagnostics(program).map(diagnostic => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')), []);
  assert.deepEqual(lint({ code: source, filename, program }), ['directEnvironment', 'directEnvironment', 'directEnvironment']);
});

test('the real ESLint configuration enforces Node JS/MJS and preserves generated exclusions', async () => {
  const eslint = new ESLint({ cwd: repoRoot, overrideConfigFile: resolve(repoRoot, 'eslint.config.js') });
  for (const filePath of ['tests/tools/cli/cli.test.mjs', 'tests/runtimes/llama_cpp/browser-smoke.mjs', 'eslint-local-rules/example.js']) {
    const [result] = await eslint.lintText('process.env.X;', { filePath });
    assert.deepEqual(result.messages.filter(message => message.fatal), []);
    assert.ok(result.messages.some(message => message.ruleId === 'local-rules-node-env/node-env-policy'));
  }
  const reader = await eslint.calculateConfigForFile('tools/shared/node-env.ts');
  assert.equal(reader.rules['local-rules-function-spacing/function-spacing'][0], 0);
  assert.equal(reader.rules['local-rules-node-env/node-env-policy'][0], 2);
  for (const filename of ['node_modules/library/index.js', 'build/tooling/tools/shared/node-env.js', 'runtimes/llama-cpp/vendor/llama.cpp/tool.js']) {
    assert.equal(await eslint.isPathIgnored(filename), true, filename);
  }
});
