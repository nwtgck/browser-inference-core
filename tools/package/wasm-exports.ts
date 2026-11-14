import ts from 'typescript';

/** Resolve the generated factory's public export assignments to actual Wasm symbols.
 * Emscripten may minify the Wasm names while retaining Module's public API.
 * This inspects a distribution boundary; it does not extract/rebuild source code.
 */
export function factoryExportMap({ source, wasmNames }: { source: string; wasmNames: ReadonlySet<string> }): Map<string, string> {
  const file = ts.createSourceFile('core.mjs', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const mappings = new Map<string, string>();
  function property({ node }: { node: ts.Node }): string | undefined {
    if (ts.isElementAccessExpression(node) && node.argumentExpression && ts.isStringLiteralLike(node.argumentExpression)) {
      return node.argumentExpression.text;
    }
    if (ts.isPropertyAccessExpression(node)) {
      return node.name.text;
    }
    return undefined;
  }
  function visit({ node }: { node: ts.Node }): void {
    if (ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.EqualsToken) {
      const publicName = property({ node: node.left });
      if (publicName?.startsWith('_')) {
        const candidates = new Set<string>();
        function find({ node: value }: { node: ts.Node }): void {
          const name = property({ node: value });
          if (name && wasmNames.has(name)) {
            candidates.add(name);
          }
          ts.forEachChild(value, child => find({ node: child }));
        }
        find({ node: node.right });
        if (candidates.size > 1) {
          throw new Error(`Ambiguous generated Wasm export mapping: ${publicName}`);
        }
        const symbol = [...candidates][0];
        if (symbol) {
          const previous = mappings.get(publicName);
          if (previous !== undefined && previous !== symbol) {
            throw new Error(`Conflicting generated Wasm export mapping: ${publicName}`);
          }
          mappings.set(publicName, symbol);
        }
      }
    }
    ts.forEachChild(node, child => visit({ node: child }));
  }
  visit({ node: file });
  return mappings;
}

export function assertWasmExports({ source, wasmNames, expected }: { source: string; wasmNames: ReadonlySet<string>; expected: readonly string[] }): void {
  const mapping = factoryExportMap({ source, wasmNames });
  for (const name of expected) {
    if (!mapping.has(name)) {
      throw new Error(`Missing actual Wasm export or generated factory mapping: ${name}`);
    }
  }
}
