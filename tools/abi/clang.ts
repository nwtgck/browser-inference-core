import path from 'node:path';
import { decodeClangAst, decodeCompilationDatabase, type AstNode } from '#dto/shared';
import { readJson } from '#tools/shared/json';
import { runProcess, splitCommandLine } from '#tools/shared/process';
import { readClangJson } from './clang-json.js';
export type { AstNode } from '#dto/shared';
export type Compilation = { directory: string; command: string; args: string[]; file: string };

export async function readCompilation({ database, translationUnit }: { database: string; translationUnit: string }): Promise<Compilation> {
  const entries = await readJson({ path: database, decode: decodeCompilationDatabase });
  const matches = entries.filter((entry) => path.resolve(entry.directory, entry.file) === path.resolve(translationUnit));
  if (matches.length !== 1) {
    throw new Error(`Expected one actual compile command for ${translationUnit}, found ${matches.length}`);
  }
  const entry = matches[0];
  if (!entry) {
    throw new Error('Missing compile command');
  }
  const argv = entry.arguments ?? (entry.command === undefined ? undefined : splitCommandLine({ command: entry.command }));
  if (!argv?.length) {
    throw new Error(`Missing compiler argv for ${translationUnit}`);
  }
  let executable = argv[0];
  let args = argv.slice(1);
  if (executable && /(?:^|[/\\])(?:ccache|sccache)$/u.test(executable)) {
    executable = args[0];
    args = args.slice(1);
  }
  if (!executable) {
    throw new Error('Missing compiler executable');
  }
  return { directory: entry.directory, command: executable, args, file: path.resolve(translationUnit) };
}

/** Keep the real target's language, sysroot, defines, include order and ABI flags. */
export function analysisArguments({ compilation }: { compilation: Compilation }): string[] {
  const result: string[] = [];
  for (let index = 0; index < compilation.args.length; index++) {
    const argument = compilation.args[index];
    if (!argument) {
      throw new Error('Empty compiler argument');
    }
    if (['-o', '-MF', '-MT', '-MQ', '-MJ'].includes(argument)) {
      index++;
      continue;
    }
    if (['-c', '-MD', '-MMD', '-MP', '-save-temps'].includes(argument)) {
      continue;
    }
    if (/^-(?:o|MF|MT|MQ|MJ).+/u.test(argument)) {
      continue;
    }
    if (argument === '-fshort-enums' || argument.startsWith('-fpack-struct')) {
      throw new Error(`Unsupported public ABI option: ${argument}`);
    }
    if (argument.startsWith('@')) {
      throw new Error('Compile response files must be expanded in the actual compilation database');
    }
    result.push(argument);
  }
  return [...result, '-DBICORE_ABI_EXTRACT=1', '-fsyntax-only'];
}

export type AstEntry = { node: AstNode; namespaces: string[]; parent: AstNode | undefined };

export function astNodes({ ast }: { ast: AstNode }): AstEntry[] {
  const entries: AstEntry[] = [];
  const visit = ({ node, namespaces, parent }: { node: AstNode; namespaces: string[]; parent: AstNode | undefined }): void => {
    entries.push({ node, namespaces, parent });
    const nested = node.kind === 'NamespaceDecl' ? [...namespaces, node.name ?? '(anonymous)'] : namespaces;
    for (const child of node.inner ?? []) {
      visit({ node: child, namespaces: nested, parent: node });
    }
  };
  visit({ node: ast, namespaces: [], parent: undefined });
  return entries;
}

export function childNodes({ node }: { node: AstNode }): AstNode[] {
  return node.inner ?? [];
}

export type AstScope = { filters: readonly string[]; prefixes: readonly string[]; aliases: readonly string[] };

export async function inspectAst({ compilation, scope }: { compilation: Compilation; scope: AstScope | undefined }): Promise<AstEntry[]> {
  const args = analysisArguments({ compilation });
  if (scope === undefined) {
    const result = await runProcess({ command: compilation.command, args: [...args, '-Xclang', '-ast-dump=json'], cwd: compilation.directory });
    const value: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(result.stdout));
    return astNodes({ ast: decodeClangAst({ input: value }) });
  }
  const listed = await runProcess({ command: compilation.command, args: [...args, '-Xclang', '-ast-list'], cwd: compilation.directory });
  const names = new Set(new TextDecoder('utf-8', { fatal: true }).decode(listed.stdout).split(/\r?\n/u).filter(Boolean));
  const aliases = new Set(scope.aliases);
  const types = new Map<string, string>();
  const declarations: AstNode[] = [];
  const namespaces = new Set<string>();
  for (const filter of scope.filters) {
    const anonymous = scope.prefixes.some(prefix => prefix.startsWith(filter) || filter.startsWith(prefix));
    await readClangJson({
      command: compilation.command,
      args: [...args, '-Xclang', '-ast-dump=json', '-Xclang', `-ast-dump-filter=${filter}`],
      cwd: compilation.directory,
      onDocument: ({ input }) => {
        if (!input || typeof input !== 'object' || Array.isArray(input)) {
          throw new TypeError('Clang declaration document must be an object');
        }
        const name = 'name' in input ? input.name : undefined;
        const kind = 'kind' in input ? input.kind : undefined;
        // A matching namespace is emitted as one document. Its children must not
        // become globals. The ABI consumes only global C declarations.
        if (kind === 'NamespaceDecl') {
          if (typeof name === 'string') {
            namespaces.add(name);
          }
          return;
        }
        const selected = typeof name === 'string' && names.has(name) && (scope.prefixes.some(prefix => name.startsWith(prefix)) || aliases.has(name));
        const unnamedRecord = anonymous && name === undefined && (kind === 'RecordDecl' || kind === 'CXXRecordDecl');
        if (!selected && !unnamedRecord) {
          return;
        }
        const node = decodeClangAst({ input });
        if ((node.kind === 'TypedefDecl' || node.kind === 'TypeAliasDecl') && node.name && aliases.has(node.name) && node.type) {
          const type = node.type.desugaredQualType ?? node.type.qualType;
          const previous = types.get(node.name);
          if (previous !== undefined && previous !== type) {
            throw new Error(`Ambiguous compiler typedef ${node.name}: ${previous} versus ${type}; preserve its namespace context`);
          }
          types.set(node.name, type);
        }
        declarations.push(node);
      },
    });
  }
  const entries = astNodes({ ast: { kind: 'TranslationUnitDecl', inner: declarations } });
  const observed = new Set([...namespaces, ...entries.flatMap(({ node }) => node.name ? [node.name] : [])]);
  const missing = [...names].filter(name => !name.includes('::') && scope.prefixes.some(prefix => name.startsWith(prefix)) && !observed.has(name));
  if (missing.length) {
    throw new Error(`Clang filtered declarations are incomplete: ${missing.join(', ')}`);
  }
  return entries;
}

export type RecordLayout = { type: string; size: number; alignment: number; fields: { name: string; offset: number; depth: number; anonymous: boolean }[] };

/** A bounded decoder of Clang's record-layout diagnostic, not a C++ parser. */
export function decodeRecordLayouts({ text }: { text: string }): RecordLayout[] {
  const layouts: RecordLayout[] = [];
  for (const block of text.split('*** Dumping AST Record Layout').slice(1)) {
    const lines = block.split(/\r?\n/u);
    // Named anonymous typedefs are printed as `0 | sd_image_t`, while tagged
    // records use `0 | struct ggml_tensor`. Root indentation is one space;
    // members start at three, including members at byte zero.
    const typeLine = lines.find((line) => /^\s*0 \| \S/u.test(line));
    const summary = /\[sizeof=(\d+),[^\]]*?align=(\d+)/u.exec(block.replaceAll('\n', ' '));
    if (!typeLine || !summary) {
      throw new Error('Unsupported Clang record-layout diagnostic');
    }
    const type = typeLine.slice(typeLine.indexOf('|') + 2).trim();
    const fields: RecordLayout['fields'] = [];
    for (const line of lines) {
      if (line === typeLine) {
        continue;
      }
      const match = /^\s*(\d+) \| ( +)(.+)$/u.exec(line);
      if (!match) {
        continue;
      }
      const offset = Number(match[1]);
      const spaces = match[2];
      const declaration = match[3]?.trimEnd();
      if (!spaces || !declaration) {
        throw new Error('Invalid Clang field layout');
      }
      const name = /(?:^|\s)([A-Za-z_][A-Za-z_0-9]*)$/u.exec(declaration)?.[1];
      const anonymous = /\((?:unnamed|anonymous) at .+\)$/u.test(declaration);
      fields.push({ name: name ?? '', offset, depth: spaces.length, anonymous });
    }
    layouts.push({ type, size: Number(summary[1]), alignment: Number(summary[2]), fields });
  }
  return layouts;
}

export async function inspectLayouts({ compilation }: { compilation: Compilation }): Promise<RecordLayout[]> {
  const result = await runProcess({ command: compilation.command, args: [...analysisArguments({ compilation }), '-Xclang', '-fdump-record-layouts'], cwd: compilation.directory });
  return decodeRecordLayouts({ text: result.stdout.toString('utf8') });
}

export function promotedFields({ layout }: { layout: RecordLayout }): Map<string, number> {
  const result = new Map<string, number>();
  const parents: RecordLayout['fields'] = [];
  for (const field of layout.fields) {
    while (parents.length && (parents.at(-1)?.depth ?? 0) >= field.depth) {
      parents.pop();
    }
    if (field.name && parents.every((parent) => parent.anonymous)) {
      if (result.has(field.name)) {
        throw new Error(`Ambiguous promoted field ${layout.type}.${field.name}`);
      }
      result.set(field.name, field.offset);
    }
    parents.push(field);
  }
  return result;
}

export async function compilerMacros({ compilation }: { compilation: Compilation }): Promise<Map<string, string>> {
  const args = analysisArguments({ compilation }).filter((argument) => argument !== '-fsyntax-only');
  const result = await runProcess({ command: compilation.command, args: [...args, '-E', '-dM'], cwd: compilation.directory });
  const macros = new Map<string, string>();
  for (const line of result.stdout.toString('utf8').split(/\r?\n/u)) {
    const match = /^#define ([A-Za-z_][A-Za-z_0-9]*)\s+(.*)$/u.exec(line);
    if (match?.[1] && match[2] !== undefined) {
      macros.set(match[1], match[2]);
    }
  }
  return macros;
}
