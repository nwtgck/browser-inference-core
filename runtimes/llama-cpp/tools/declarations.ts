import { resolve } from 'node:path';
import ts from 'typescript';

/** Emscripten's native i64 values are JavaScript primitives under WASM_BIGINT. */
export function prepareModuleDeclarations({ source }: { source: string }): { source: string; correctedBigIntReferences: number } {
  const filename = resolve('emscripten-module.d.ts');
  const options: ts.CompilerOptions = { target: ts.ScriptTarget.ES2023, lib: ['lib.es2023.d.ts'], types: [], noEmit: true };
  const host = ts.createCompilerHost(options);
  const readSource = host.getSourceFile;
  host.getSourceFile = (path, languageVersion, onError, shouldCreateNewSourceFile) => resolve(path) === filename
    ? ts.createSourceFile(filename, source, languageVersion, true, ts.ScriptKind.TS)
    : readSource(path, languageVersion, onError, shouldCreateNewSourceFile);
  const program = ts.createProgram({ rootNames: [filename], options, host });
  const file = program.getSourceFile(filename);
  if (file === undefined) {
    throw new Error('Missing generated Emscripten declaration source');
  }
  const diagnostics = program.getSyntacticDiagnostics(file);
  if (diagnostics.length !== 0) {
    throw new Error(`Invalid generated Emscripten declarations: ${diagnostics.map(diagnostic => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')).join('\n')}`);
  }
  const modules = file.statements.filter(ts.isInterfaceDeclaration).filter(node => node.name.text === 'WasmModule');
  if (modules.length === 0) {
    throw new Error('Missing generated Emscripten WasmModule interface');
  }
  const checker = program.getTypeChecker();
  const edits: { start: number; end: number }[] = [];
  function visit({ node }: { node: ts.Node }): void {
    if (ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName) && node.typeName.text === 'BigInt' && node.typeArguments === undefined) {
      const symbol = checker.getSymbolAtLocation(node.typeName);
      const declarations = symbol?.declarations;
      if (declarations?.every(declaration => program.isSourceFileDefaultLibrary(declaration.getSourceFile())) && declarations.some(declaration => ts.isInterfaceDeclaration(declaration) && declaration.name.text === 'BigInt')) {
        edits.push({ start: node.typeName.getStart(file), end: node.typeName.end });
      }
    }
    ts.forEachChild(node, child => visit({ node: child }));
  }
  for (const module of modules) {
    for (const member of module.members) {
      if (ts.isMethodSignature(member) && (ts.isIdentifier(member.name) || ts.isStringLiteral(member.name)) && member.name.text.startsWith('_')) {
        visit({ node: member });
      }
    }
  }
  let result = source;
  // AST-selected token edits preserve unrelated SDK declarations and comments.
  for (const { start, end } of edits.sort((a, b) => b.start - a.start)) {
    result = `${result.slice(0, start)}bigint${result.slice(end)}`;
  }
  return { source: result, correctedBigIntReferences: edits.length };
}
