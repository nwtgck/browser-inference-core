import { readFile } from 'node:fs/promises';
import ts from 'typescript';
import { regularFile } from './files.js';

/** JSON.parse accepts duplicate object keys. Reject those before schema validation. */
function rejectDuplicateKeys({ text }: { text: string }): void {
  const tree = ts.parseJsonText('input.json', text);
  function visit({ node }: { node: ts.Node }): void {
    if (ts.isObjectLiteralExpression(node)) {
      const keys = new Set<string>();
      for (const property of node.properties) {
        if (!ts.isPropertyAssignment(property) || !ts.isStringLiteral(property.name)) {
          throw new SyntaxError('Expected a JSON object key');
        }
        const key = property.name.text;
        if (keys.has(key)) {
          throw new SyntaxError(`Duplicate JSON key: ${key}`);
        }
        keys.add(key);
      }
    }
    ts.forEachChild(node, (child) => visit({ node: child }));
  }
  visit({ node: tree });
}

export function parseJson<T>({ text, decode }: { text: string; decode: ({ input }: { input: unknown }) => T }): T {
  const value: unknown = JSON.parse(text);
  rejectDuplicateKeys({ text });
  return decode({ input: value });
}

export async function readJson<T>({ path, decode }: { path: string; decode: ({ input }: { input: unknown }) => T }): Promise<T> {
  return parseJson({ text: await readFile(await regularFile({ path }), 'utf8'), decode });
}
