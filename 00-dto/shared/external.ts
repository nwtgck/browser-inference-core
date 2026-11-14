import { z } from 'zod';
import { runtimeNameSchema } from '#dto-internal/shared/artifacts';

export const browserSessionSchema = z.object({
  schemaVersion: z.literal(1),
  runtime: runtimeNameSchema,
  sourceCommit: z.string(),
  runId: z.string(),
  runAttempt: z.string(),
  job: z.string(),
  sessionId: z.string(),
  snapshotSha256: z.string(),
  testInputsSha256: z.string(),
  testWebGpu: z.boolean(),
});
export type BrowserSession = z.infer<typeof browserSessionSchema>;

const compileEntrySchema = z.object({ directory: z.string(), file: z.string(), command: z.string().optional(), arguments: z.array(z.string()).optional() });
export type CompileEntry = z.infer<typeof compileEntrySchema>;

export function decodeCompilationDatabase({ input }: { input: unknown }): CompileEntry[] {
  return z.array(compileEntrySchema).parse(input);
}

// Clang owns this open format. Only attributes read by the adapter are decoded.
// Children become typed at this boundary, so traversal never reparses them.
const typeSchema = z.object({ qualType: z.string(), desugaredQualType: z.string().optional() });
const locationSchema = z.object({
  file: z.string().optional(),
  line: z.number().optional(),
  col: z.number().optional(),
  offset: z.number().optional(),
  includedFrom: z.object({ file: z.string() }).optional(),
});
const referenceSchema = z.object({ id: z.string(), kind: z.string().optional(), name: z.string().optional() });
const astNodeSchema = z.object({
  kind: z.string(),
  id: z.string().optional(),
  name: z.string().optional(),
  mangledName: z.string().optional(),
  type: typeSchema.optional(),
  fixedUnderlyingType: typeSchema.optional(),
  value: z.union([z.string(), z.number(), z.boolean()]).optional(),
  get inner() {
    return z.array(astNodeSchema).optional();
  },
  loc: locationSchema.optional(),
  range: z.object({ begin: locationSchema, end: locationSchema }).optional(),
  ownedTagDecl: referenceSchema.optional(),
  decl: referenceSchema.optional(),
  completeDefinition: z.boolean().optional(),
  isImplicit: z.boolean().optional(),
  isBitfield: z.boolean().optional(),
  variadic: z.boolean().optional(),
  tagUsed: z.string().optional(),
  language: z.string().optional(),
  storageClass: z.string().optional(),
  bases: z.array(z.object({ type: typeSchema, access: z.string().optional(), isVirtual: z.boolean().optional() })).optional(),
  definitionData: z.object({ isStandardLayout: z.boolean().optional(), isTriviallyCopyable: z.boolean().optional() }).optional(),
});
export type AstNode = z.infer<typeof astNodeSchema>;

// ABI extraction consumes global declarations. Namespace/template definitions
// and function bodies can contain large implementation-only trees; leave them
// out before structural decoding instead of cloning facts no consumer reads.
function globalDeclarationProjection({ input }: { input: unknown }): unknown {
  if (typeof input !== 'object' || input === null || !('inner' in input)) {
    return input;
  }
  if ('kind' in input && typeof input.kind === 'string' && input.kind.endsWith('Attr')) {
    // Attribute identity (for example DeprecatedAttr) is consumed, its payload
    // is not. Clang may emit empty placeholder objects inside that payload.
    const { inner: _inner, ...attribute } = input;
    return attribute;
  }
  if (!Array.isArray(input.inner)) {
    return input;
  }
  const children: readonly unknown[] = input.inner;
  const inner: unknown[] = [];
  for (const child of children) {
    if (typeof child === 'object' && child !== null && 'kind' in child) {
      if (child.kind === 'NamespaceDecl' || child.kind === 'FunctionTemplateDecl' || child.kind === 'ClassTemplateDecl') {
        continue;
      }
      if ('kind' in input && input.kind === 'FunctionDecl' && child.kind === 'CompoundStmt') {
        continue;
      }
    }
    inner.push(globalDeclarationProjection({ input: child }));
  }
  return { ...input, inner };
}

export function decodeClangAst({ input }: { input: unknown }): AstNode {
  return astNodeSchema.parse(globalDeclarationProjection({ input }));
}

const emscriptenReleasesSchema = z.object({ releases: z.record(z.string(), z.string()) });
export type EmscriptenReleases = z.infer<typeof emscriptenReleasesSchema>;

export function decodeEmscriptenReleases({ input }: { input: unknown }): EmscriptenReleases {
  return emscriptenReleasesSchema.parse(input);
}
