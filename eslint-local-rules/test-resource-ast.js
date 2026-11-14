// Small, non-evaluating helpers shared by the resource and import boundaries.
// Only lexical constants are followed; arbitrary JavaScript is never executed.
export function unwrap({ node }) {
  while (node && ['TSAsExpression', 'TSTypeAssertion', 'TSNonNullExpression', 'TSSatisfiesExpression', 'TSInstantiationExpression', 'ChainExpression'].includes(node.type)) {
    node = node.expression;
  }
  return node;
}

export function findVariable({ sourceCode, node, name }) {
  for (let scope = sourceCode.getScope(node); scope; scope = scope.upper) {
    const variable = scope.set.get(name);
    if (variable) {
      return variable;
    }
  }
  return undefined;
}

export function staticString({ node, sourceCode }) {
  return resolveStaticString({ node, sourceCode, seen: new Set(), depth: 0 });
}

function resolveStaticString({ node, sourceCode, seen, depth }) {
  node = unwrap({ node });
  if (!node || depth > 16) {
    return undefined;
  }
  const next = value => resolveStaticString({ node: value, sourceCode, seen: new Set(seen), depth: depth + 1 });
  if (node.type === 'Literal' && typeof node.value === 'string') {
    return node.value;
  }
  if (node.type === 'BinaryExpression' && node.operator === '+') {
    const left = next(node.left), right = next(node.right);
    return typeof left === 'string' && typeof right === 'string' ? left + right : undefined;
  }
  if (node.type === 'TemplateLiteral') {
    let value = '';
    for (let index = 0; index < node.quasis.length; index++) {
      const part = node.quasis[index].value.cooked;
      if (part === null) {
        return undefined;
      }
      value += part;
      if (index < node.expressions.length) {
        const expression = next(node.expressions[index]);
        if (expression === undefined) {
          return undefined;
        }
        value += expression;
      }
    }
    return value;
  }
  if (node.type === 'Identifier') {
    const variable = findVariable({ sourceCode, node, name: node.name });
    if (!variable || variable.defs.length !== 1 || seen.has(variable)) {
      return undefined;
    }
    const definition = variable.defs[0];
    if (definition.type !== 'Variable' || definition.parent?.kind !== 'const' || definition.node.id.type !== 'Identifier') {
      return undefined;
    }
    seen.add(variable);
    return next(definition.node.init);
  }
  return undefined;
}

export function propertyName({ node, sourceCode }) {
  const key = node?.property ?? node?.key;
  return !node?.computed && key?.type === 'Identifier' ? key.name : staticString({ node: key, sourceCode });
}

export function builtinName({ node, sourceCode }) {
  return resolveBuiltinName({ node, sourceCode, seen: new Set(), depth: 0 });
}

function resolveBuiltinName({ node, sourceCode, seen, depth }) {
  node = unwrap({ node });
  if (!node || depth > 16) {
    return undefined;
  }
  const next = value => resolveBuiltinName({ node: value, sourceCode, seen: new Set(seen), depth: depth + 1 });
  const globals = new Set(['globalThis', 'window', 'self', 'global']);
  if (node.type === 'MemberExpression' || node.type === 'TSQualifiedName') {
    const object = node.object ?? node.left;
    const owner = next(object);
    const property = node.type === 'TSQualifiedName' ? node.right.name : propertyName({ node, sourceCode });
    if (globals.has(owner)) {
      return property;
    }
    if (['Symbol', 'Object', 'Reflect', 'require'].includes(owner) && property !== undefined) {
      return `${owner}.${property}`;
    }
  }
  if (node.type !== 'Identifier') {
    return undefined;
  }
  const variable = findVariable({ sourceCode, node, name: node.name });
  if (!variable || variable.defs.length === 0) {
    return node.name;
  }
  if (variable.defs.length !== 1 || seen.has(variable)) {
    return undefined;
  }
  seen.add(variable);
  const definition = variable.defs[0];
  if (definition.type !== 'Variable') {
    return undefined;
  }
  // A mutable alias initialized with a builtin is conservatively treated as that
  // builtin. Reassigning it does not grant permission to access its protocols.
  if (definition.node.id.type === 'Identifier') {
    return next(definition.node.init);
  }
  const owner = next(definition.node.init);
  if (definition.node.id.type === 'ObjectPattern' && (globals.has(owner) || ['Symbol', 'Object', 'Reflect', 'require'].includes(owner))) {
    const property = definition.node.id.properties.find(value => value.type === 'Property' && value.value.type === 'Identifier' && value.value.name === node.name);
    const key = property && propertyName({ node: property, sourceCode });
    return key === undefined ? undefined : globals.has(owner) ? key : `${owner}.${key}`;
  }
  return undefined;
}

export function cleanModulePath({ value }) {
  const normalized = value.replaceAll('\\', '/');
  // A leading # belongs to a Node package-import specifier, not a URL fragment.
  const prefix = normalized.startsWith('#') ? '#' : '';
  return prefix + normalized.slice(prefix.length).split(/[?#]/u, 1)[0];
}

export function isTestFile({ filename }) {
  return filename.replaceAll('\\', '/').endsWith('.test.ts');
}

export function isTestSupportFile({ filename }) {
  return filename.replaceAll('\\', '/').endsWith('.test-support.ts');
}

export function isTestOnlyModule({ value }) {
  const clean = cleanModulePath({ value });
  // Legacy spellings are DENIED as import targets, never granted permissions.
  return /(?:^#tests\/|(?:^|\/)tests\/|eslint-local-rules\/fixtures(?:\/|$)|\.test(?:\.|$)|\.spec(?:\.|$)|(?:^|[-/.])(?:test-support|test-helpers|test-utils|test-mocks|test-setup|test-tmp|lint-rule-tmp)(?:[/.]|$))/u.test(clean);
}
