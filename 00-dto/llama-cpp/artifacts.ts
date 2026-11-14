import { z } from 'zod';
import { buildCacheSchema, compilerInputsSchema, fileEntrySchema, fileIdentitiesSchema, fileIdentitySchema, jsonObjectSchema, jsonSchema, patchReceiptSchema, sourceInputsSchema, treeIdentitySchema, validationSchema } from '#dto-internal/shared/artifacts';
import { browserSessionSchema } from '#dto-internal/shared/external';

// Browser/worker output is untyped input. Named evidence controls admission;
// other harness statistics remain opaque report data and are preserved.
const llamaBrowserResultSchema = z.object({
  profile: z.string(),
  variant: z.string(),
  passed: z.boolean(),
  syntheticModel: z.boolean().optional(),
  mockedAdapter: z.boolean().optional(),
  suspension: z.boolean().optional(),
  defaultNavigatorFallback: z.boolean().optional(),
  hostNavigatorInjection: z.boolean().optional(),
  independentFactories: z.boolean().optional(),
}).catchall(jsonSchema);
export type LlamaBrowserResult = z.infer<typeof llamaBrowserResultSchema>;
const llamaBrowserEnvelopeSchema = z.object({ session: browserSessionSchema.extend({ runtime: z.literal('llama-cpp') }), results: z.array(llamaBrowserResultSchema) }).catchall(jsonSchema);
export type LlamaBrowserEnvelope = z.infer<typeof llamaBrowserEnvelopeSchema>;
export const llamaValidationSchema = validationSchema.extend({ syntheticUntrainedModel: llamaBrowserResultSchema.optional(), mockedAdapterSuspension: llamaBrowserResultSchema.optional() });
export type LlamaValidation = z.infer<typeof llamaValidationSchema>;

export function decodeLlamaBrowserEnvelope({ input }: { input: unknown }): LlamaBrowserEnvelope {
  return llamaBrowserEnvelopeSchema.parse(input);
}

export const preparationOptionsSchema = z.object({
  target: z.enum(['wasm', 'native']),
  webgpu: z.boolean(),
  gpuProfile: z.boolean(),
  fileRows: z.boolean(),
  qsa: z.boolean(),
  mtmdMode: z.enum(['cpu', 'webgpu']),
}).catchall(jsonSchema);
export type PreparationOptions = z.infer<typeof preparationOptionsSchema>;
export const preparationReceiptSchema = z.object({
  formatVersion: z.literal(1),
  options: preparationOptionsSchema,
  upstream: treeIdentitySchema,
  inputs: fileIdentitiesSchema,
  runtimeInputs: fileIdentitiesSchema,
  patches: z.array(patchReceiptSchema),
  selections: z.array(z.object({ file: z.string(), mode: z.literal('upstream_native_context'), reason: z.string(), before: fileIdentitySchema, after: fileIdentitySchema }).catchall(jsonSchema)),
  shaders: z.record(z.string(), fileIdentitiesSchema),
  outputs: fileIdentitiesSchema,
}).catchall(jsonSchema);
export type PreparationReceipt = z.infer<typeof preparationReceiptSchema>;

export function decodePreparationReceipt({ input }: { input: unknown }): PreparationReceipt {
  return preparationReceiptSchema.parse(input);
}

// Configuration and toolchain data are retained only for equality comparisons
// and reports. Build decisions use the checkout's statically typed configuration.
const llamaProvenanceSchema = z.object({
  runtime: z.literal('llama-cpp').optional(),
  profile: z.string(),
  variant: z.string(),
  sourceCommit: z.string(),
  sourceDirty: z.boolean(),
  configuration: jsonObjectSchema,
  variantConfiguration: jsonObjectSchema,
  toolchain: jsonObjectSchema,
  llamaCommit: z.string(),
  emccVersion: z.string(),
  nodeVersion: z.string(),
  sourceInputs: sourceInputsSchema,
  upstreamBuildInfo: z.object({ commit: z.string(), number: z.number() }).catchall(jsonSchema),
  preparedSources: preparationReceiptSchema,
  compiledInputs: compilerInputsSchema,
  cmakeCommand: z.array(z.string()),
  builtAtUnix: z.number().optional(),
  buildCache: buildCacheSchema,
  visionBf16Storage: z.object({ version: z.number(), modes: z.array(z.string()), default: z.string() }).catchall(jsonSchema).optional(),
  experimentalVisionBf16: z.boolean().optional(),
  browserModulePreparation: z.object({ input: fileIdentitySchema, output: fileIdentitySchema, alreadyAnnotated: z.boolean() }).catchall(jsonSchema).optional(),
  moduleDeclarationPreparation: z.object({ input: fileIdentitySchema, output: fileIdentitySchema, correctedBigIntReferences: z.number() }).catchall(jsonSchema).optional(),
  validation: llamaValidationSchema,
}).catchall(jsonSchema);
export type LlamaProvenance = z.infer<typeof llamaProvenanceSchema>;
const llamaManifestSchema = z.object({
  formatVersion: z.literal(2),
  runtime: z.literal('llama-cpp'),
  abiVersion: z.literal(3),
  sourceCommit: z.string(),
  schemaSha256: z.string(),
  llamaCommit: z.string(),
  licenseRoots: z.array(z.string()),
  files: z.array(fileEntrySchema),
  profiles: z.record(z.string(), z.object({ variants: z.record(z.string(), llamaProvenanceSchema) }).catchall(jsonSchema)),
}).catchall(jsonSchema);
export type LlamaManifest = z.infer<typeof llamaManifestSchema>;

export function decodeLlamaProvenance({ input }: { input: unknown }): LlamaProvenance {
  return llamaProvenanceSchema.parse(input);
}

export function decodeLlamaManifest({ input }: { input: unknown }): LlamaManifest {
  return llamaManifestSchema.parse(input);
}
