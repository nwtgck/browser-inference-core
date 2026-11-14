import eslint from '@eslint/js';
import { defineConfig } from 'eslint/config';
import tseslint from 'typescript-eslint';
import resourceManagement from '#lint/test-resource-management';
import testSupportImports from '#lint/no-test-support-imports';
import namedArguments from '#lint/require-named-args';
import objectLayout from '#lint/object-layout';
import functionSpacing from '#lint/function-spacing';
import statementLayout from '#lint/statement-layout';
import forceSwitchForUnion from '#lint/force-switch-for-union';

const typedSources = ['00-dto/**/*.ts', 'tools/**/*.ts', 'runtimes/*/tools/**/*.ts', 'tests/tools/**/*.ts', 'tests/runtimes/*/**/*.ts'];
const parentImports = {
  group: ['..', '../**'],
  message: 'Use the package imports (#tools, #runtimes, #tests or #dto) instead of a parent-relative import.',
};
const parentImportSyntax = {
  selector: [
    'ImportExpression[source.value=/^\\.\\.(\\/|$)/]',
    'TSImportType[source.value=/^\\.\\.(\\/|$)/]',
    'ImportExpression > TemplateLiteral[expressions.length=0] > TemplateElement[value.cooked=/^\\.\\.(\\/|$)/]',
  ].join(', '),
  message: 'Use a package import instead of a parent-relative dynamic or type import.',
};
const privateDtoImports = {
  // Glob patterns treat a leading # as a comment. Match module specifiers
  // directly, with complete path segments rather than directory substrings.
  regex: '^#dto-internal(?:/|$)|^#dto/[^/]+/|(?:^|/)00-dto(?:/|$)',
  caseSensitive: true,
  message: 'Import DTO types and decoders through #dto/shared, #dto/llama-cpp or #dto/stable-diffusion-cpp.',
};
const privateDtoSyntax = {
  selector: [
    'ImportExpression[source.value=/^#dto-internal(\\/|$)/]',
    'TSImportType[source.value=/^#dto-internal(\\/|$)/]',
    'CallExpression[callee.name="require"][arguments.0.value=/^#dto-internal(\\/|$)/]',
    'ImportExpression[source.value=/^#dto\\/.*\\//]',
    'TSImportType[source.value=/^#dto\\/.*\\//]',
    'CallExpression[callee.name="require"][arguments.0.value=/^#dto\\/.*\\//]',
    'ImportExpression[source.value=/(^|\\/)00-dto(\\/|$)/]',
    'TSImportType[source.value=/(^|\\/)00-dto(\\/|$)/]',
    'CallExpression[callee.name="require"][arguments.0.value=/(^|\\/)00-dto(\\/|$)/]',
    ':matches(ImportExpression, CallExpression[callee.name="require"]) > TemplateLiteral[expressions.length=0] > TemplateElement[value.cooked=/^#dto-internal(\\/|$)/]',
    ':matches(ImportExpression, CallExpression[callee.name="require"]) > TemplateLiteral[expressions.length=0] > TemplateElement[value.cooked=/^#dto\\/.*\\//]',
    ':matches(ImportExpression, CallExpression[callee.name="require"]) > TemplateLiteral[expressions.length=0] > TemplateElement[value.cooked=/(^|\\/)00-dto(\\/|$)/]',
  ].join(', '),
  message: privateDtoImports.message,
};
const zodImports = {
  regex: '^zod(?:/|$)',
  caseSensitive: true,
  message: 'Zod belongs only in 00-dto/; use public DTO types and decoders elsewhere.',
};
const zodSyntax = {
  selector: [
    'ImportExpression[source.value=/^zod(\\/|$)/]',
    'TSImportType[source.value=/^zod(\\/|$)/]',
    'CallExpression[callee.name="require"][arguments.0.value=/^zod(\\/|$)/]',
    ':matches(ImportExpression, CallExpression[callee.name="require"]) > TemplateLiteral[expressions.length=0] > TemplateElement[value.cooked=/^zod(\\/|$)/]',
  ].join(', '),
  message: zodImports.message,
};

export default defineConfig([
  { ignores: ['build/**', 'dist/**', '.tools/**', 'node_modules/**', '**/vendor/**', '**/build/**', '**/dist/**', '**/node_modules/**'] },
  {
    files: ['**/*.{ts,tsx,mts,cts,js,jsx,mjs,cjs}'],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: { ecmaVersion: 'latest', sourceType: 'module', project: false, projectService: false },
    },
  },
  resourceManagement,
  testSupportImports,
  {
    files: typedSources,
    extends: [eslint.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      parserOptions: { project: './tsconfig.tools.json', tsconfigRootDir: import.meta.dirname },
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': 'off',
      '@typescript-eslint/consistent-type-imports': 'off',
      '@typescript-eslint/restrict-template-expressions': ['error', {
        allowNever: true,
        allowAny: false,
        allowBoolean: true,
        allowNullish: true,
        allowNumber: true,
        allowRegExp: false,
      }],
      'no-useless-assignment': 'off',
      'preserve-caught-error': 'off',
      'quotes': 'off',
      'comma-dangle': 'off',
      'max-len': 'off',
      'indent': ['warn', 2],
      'brace-style': ['warn', '1tbs', { allowSingleLine: false }],
      'no-trailing-spaces': 'error',
      'no-restricted-imports': ['error', {
        patterns: [parentImports],
      }],
      'no-restricted-syntax': ['error', parentImportSyntax],
    },
  },
  {
    files: ['**/*.{ts,tsx,mts,cts,js,jsx,mjs,cjs}'],
    ignores: ['00-dto/**'],
    rules: {
      'no-restricted-imports': ['error', { patterns: [privateDtoImports, zodImports] }],
      'no-restricted-syntax': ['error', privateDtoSyntax, zodSyntax],
    },
  },
  {
    files: typedSources,
    ignores: ['00-dto/**'],
    rules: {
      'no-restricted-imports': ['error', { patterns: [parentImports, privateDtoImports, zodImports] }],
      'no-restricted-syntax': ['error', parentImportSyntax, privateDtoSyntax, zodSyntax],
    },
  },
  {
    files: ['00-dto/**/*.{ts,tsx,mts,cts,js,jsx,mjs,cjs}'],
    rules: {
      'no-restricted-imports': ['error', { patterns: [parentImports] }],
      'no-restricted-syntax': ['error', parentImportSyntax],
    },
  },
  namedArguments,
  forceSwitchForUnion,
  objectLayout,
  functionSpacing,
  statementLayout,
]);
