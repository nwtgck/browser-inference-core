import ts from 'typescript';

export const rule = {
  meta: {
    type: 'suggestion',
    docs: {
      description: 'Prefer switch statements for direct string literal union comparisons',
    },
    messages: {
      preferSwitch: 'Use a switch statement for union types instead of `{{condition}}`. Add a default never check when handling every variant; this rule does not verify switch exhaustiveness.',
    },
    schema: [],
  },
  create(context) {
    const sourceCode = context.sourceCode;
    const services = sourceCode.parserServices;
    if (!services?.program || !services.esTreeNodeToTSNodeMap) {
      throw new Error('force-switch-for-union requires TypeScript parser services with type information.');
    }
    const checker = services.program.getTypeChecker();

    function check({ test }) {
      if (test.type !== 'BinaryExpression' || !['===', '==', '!==', '!='].includes(test.operator)) {
        return;
      }
      const { left, right } = test;
      const isLeftStringLiteral = left.type === 'Literal' && typeof left.value === 'string' ||
        left.type === 'TemplateLiteral' && left.quasis.length === 1 && left.expressions.length === 0;
      const isRightStringLiteral = right.type === 'Literal' && typeof right.value === 'string' ||
        right.type === 'TemplateLiteral' && right.quasis.length === 1 && right.expressions.length === 0;
      if (isLeftStringLiteral === isRightStringLiteral) {
        return;
      }
      const target = isLeftStringLiteral ? right : left;
      if (target.type === 'UnaryExpression' && target.operator === 'typeof') {
        return;
      }
      const type = checker.getTypeAtLocation(services.esTreeNodeToTSNodeMap.get(target));
      if (!type.isUnion()) {
        return;
      }
      let hasStringLiteral = false;
      for (const member of type.types) {
        const flags = member.getFlags();
        const isStringLiteral = member.isStringLiteral() || Boolean(flags & ts.TypeFlags.StringLiteral);
        hasStringLiteral ||= isStringLiteral;
        if (!isStringLiteral && !(flags & (ts.TypeFlags.UniqueESSymbol | ts.TypeFlags.ESSymbol | ts.TypeFlags.Null | ts.TypeFlags.Undefined))) {
          return;
        }
      }
      if (hasStringLiteral) {
        context.report({ node: test, messageId: 'preferSwitch', data: { condition: sourceCode.getText(test) } });
      }
    }

    return {
      IfStatement(node) {
        check({ test: node.test });
      },
      ConditionalExpression(node) {
        check({ test: node.test });
      },
    };
  },
};

export default {
  files: ['00-dto/**/*.ts', 'tools/**/*.ts', 'runtimes/*/tools/**/*.ts'],
  ignores: ['**/*.test.ts', '**/*.spec.ts'],
  plugins: { 'local-rules-switch': { rules: { 'force-switch-for-union': rule } } },
  rules: { 'local-rules-switch/force-switch-for-union': 'error' },
};
