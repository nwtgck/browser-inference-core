import assert from 'node:assert/strict';
import test from 'node:test';
import { decodeClangAst } from '#dto/shared';
import { astNodes, decodeRecordLayouts, promotedFields } from '#tools/abi/clang';
import { Declarations, enumValues } from '#tools/abi/declarations';
import { renderFunctionModule, renderFunctionTypes } from '#tools/abi/schema';

test('Clang anonymous typedef root and promoted union offsets use actual indentation', () => {
  const layouts = decodeRecordLayouts({
    text: `\
*** Dumping AST Record Layout
         0 | sd_image_t
         0 |   uint32_t width
         8 |   uint8_t * data
           | [sizeof=16, dsize=16, align=8,
           |  nvsize=16, nvalign=8]
*** Dumping AST Record Layout
         0 | struct ggml_fixture
         0 |   union ggml_fixture::(anonymous at api.h:2:3)
         0 |     int32_t signed_value
         0 |     uint32_t unsigned_value
         4 |   struct nested_t nested
         4 |     int x
           | [sizeof=8, dsize=8, align=4,
           |  nvsize=8, nvalign=4]
`,
  });
  assert.equal(layouts[0]?.type, 'sd_image_t');
  const record = layouts[1];
  assert.ok(record);
  assert.deepEqual([...promotedFields({ layout: record })], [['signed_value', 0], ['unsigned_value', 0], ['nested', 4]]);
  assert.throws(() => decodeRecordLayouts({ text: '*** Dumping AST Record Layout\n0 | sd_image_t\n' }), /Unsupported/u);
});

test('external enum values preserve evaluated expressions and implicit increments', () => {
  const node = decodeClangAst({
    input: {
      kind: 'EnumDecl',
      inner: [
        { kind: 'EnumConstantDecl', name: 'A', inner: [{ kind: 'ConstantExpr', value: '-7' }] },
        { kind: 'EnumConstantDecl', name: 'B' },
        { kind: 'EnumConstantDecl', name: 'C', inner: [{ kind: 'ConstantExpr', value: '4294967295' }] },
      ],
    },
  });
  assert.deepEqual([...enumValues({ node })], [['A', '-7'], ['B', '-6'], ['C', '4294967295']]);
});

test('declaration facts distinguish pointer-to-array, arrays, and callback record references', () => {
  const record = { kind: 'CXXRecordDecl', id: 'record', name: 'sd_progress', completeDefinition: true, tagUsed: 'struct', inner: [] };
  const ast = decodeClangAst({
    input: {
      kind: 'TranslationUnitDecl',
      inner: [record,
        { kind: 'TypedefDecl', name: 'sd_callback_t', type: { qualType: 'void (*)(struct sd_progress *)' }, inner: [{ kind: 'ElaboratedType', ownedTagDecl: { id: 'record' } }] },
        { kind: 'TypedefDecl', name: 'sd_direct_t', type: { qualType: 'struct sd_progress' }, inner: [{ kind: 'RecordType', decl: { id: 'record' } }] },
        { kind: 'EnumDecl', name: 'ggml_short', fixedUnderlyingType: { qualType: 'unsigned short' }, inner: [{ kind: 'EnumConstantDecl', name: 'V' }] },
        { kind: 'EnumDecl', name: 'ggml_wide', inner: [{ kind: 'EnumConstantDecl', name: 'W', inner: [{ kind: 'ConstantExpr', value: '18446744073709551615' }] }] },
      ],
    },
  });
  const declarations = new Declarations({ entries: astNodes({ ast }), layouts: [], macros: new Map([['__SIZEOF_POINTER__', '4'], ['__SIZEOF_INT__', '4'], ['__SIZEOF_LONG__', '4']]), runtime: 'stable-diffusion-cpp', translationUnit: 'metadata.cpp' });
  assert.equal(declarations.recordDeclarations.has('sd_callback_t'), false);
  assert.equal(declarations.recordDeclarations.has('sd_direct_t'), true);
  assert.equal(declarations.describe({ type: 'int (*)[4]' }).kind, 'pointer');
  assert.deepEqual(declarations.describe({ type: 'int[3][4]' }).abiType.dimensions, [3, 4]);
  assert.equal(declarations.describe({ type: 'enum ggml_short' }).abiType.bits, 16);
  assert.equal(declarations.describe({ type: 'enum ggml_wide' }).kind, 'u64');
  assert.throws(() => declarations.describe({ type: '__int128' }), /Unsupported public C type/u);
  const factory = decodeClangAst({ input: { kind: 'FunctionDecl', name: 'callback_factory', type: { qualType: 'void (*(void))(int)' } } });
  assert.throws(() => declarations.function({ node: factory, bridge: false }), /Unsupported function return declarator.*callback_factory/u);
});

test('typed facade returns the checked caller Promise without an extra async frame', () => {
  const functions = [{ name: 'token', export: '_token', returnType: 'int32_t', returnKind: 'signed' as const, abiType: { kind: 'integer' as const, bits: 32, signed: true }, parameters: [], bridge: false }];
  const source = renderFunctionModule({ schema: { functions } });
  assert.match(source, /invoker\.number\(\{ name: "token", args: \[\] \}\)/u);
  assert.doesNotMatch(source, /\basync\b|\bawait\b/u);
  assert.match(renderFunctionTypes({ schema: { functions } }), /token\(\): Promise<number>/u);
});

test('a standard-layout C++ convenience record still needs trivial copying', () => {
  for (const isTriviallyCopyable of [false, undefined]) {
    const ast = decodeClangAst({ input: { kind: 'TranslationUnitDecl', inner: [{ kind: 'CXXRecordDecl', name: 'mtmd_cpp_owner', completeDefinition: true, definitionData: { isStandardLayout: true, ...(isTriviallyCopyable === undefined ? {} : { isTriviallyCopyable }) } }] } });
    const declarations = new Declarations({ entries: astNodes({ ast }), layouts: [], macros: new Map([['__SIZEOF_POINTER__', '4']]), runtime: 'llama-cpp', translationUnit: 'metadata.cpp' });
    assert.ok(declarations.opaqueRecords.get('mtmd_cpp_owner')?.includes('trivially copyable'));
    assert.equal(declarations.describe({ type: 'struct mtmd_cpp_owner *' }).kind, 'pointer');
    assert.throws(() => declarations.describe({ type: 'struct mtmd_cpp_owner' }), /Unsupported by-value/u);
  }
});
