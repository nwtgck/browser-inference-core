import { z } from 'zod';
import { buildCacheSchema, compilerInputsSchema, fileEntrySchema, jsonObjectSchema, jsonSchema, patchReceiptSchema, sourceInputsSchema, treeIdentitySchema, validationSchema } from '#dto-internal/shared/artifacts';
import { browserSessionSchema } from '#dto-internal/shared/external';

// Backend names are the actual native backend_by_name identifiers. Other probe
// statistics are opaque report data; named evidence is checked by validation.
const browserProbeSchema = z.object({ backend: z.enum(['CPU', 'WebGPU']), passed: z.boolean() }).catchall(jsonSchema);
export type StableDiffusionBrowserProbe = z.infer<typeof browserProbeSchema>;
const browserEvidenceSchema = z.object({
  graphWalk: z.boolean().optional(),
  timestep: z.array(browserProbeSchema),
  bf16Weights: z.array(browserProbeSchema),
  conv3dBias: z.array(browserProbeSchema).optional(),
  webgpuPerformance: z.array(browserProbeSchema).optional(),
}).catchall(jsonSchema);
export type StableDiffusionBrowserEvidence = z.infer<typeof browserEvidenceSchema>;
const stableDiffusionBrowserResultSchema = browserEvidenceSchema.extend({ profile: z.string(), variant: z.string(), passed: z.boolean(), scope: z.string() }).catchall(jsonSchema);
export type StableDiffusionBrowserResult = z.infer<typeof stableDiffusionBrowserResultSchema>;
const stableDiffusionBrowserEnvelopeSchema = z.object({ session: browserSessionSchema.extend({ runtime: z.literal('stable-diffusion-cpp') }), results: z.array(stableDiffusionBrowserResultSchema) }).catchall(jsonSchema);
export type StableDiffusionBrowserEnvelope = z.infer<typeof stableDiffusionBrowserEnvelopeSchema>;
export const stableDiffusionValidationSchema = validationSchema.extend({ browserSmokeScope: z.string().optional(), browserSmokeEvidence: browserEvidenceSchema.optional() });
export type StableDiffusionValidation = z.infer<typeof stableDiffusionValidationSchema>;

export function decodeStableDiffusionBrowserEnvelope({ input }: { input: unknown }): StableDiffusionBrowserEnvelope {
  return stableDiffusionBrowserEnvelopeSchema.parse(input);
}

export const upstreamsSchema = z.object({
  'stable-diffusion': z.object({ path: z.string(), repository: z.string(), commit: z.string() }).catchall(jsonSchema),
  ggml: z.object({ path: z.string(), repository: z.string(), sourceSubdirectory: z.string(), commit: z.string() }).catchall(jsonSchema),
}).catchall(jsonSchema);
export type Upstreams = z.infer<typeof upstreamsSchema>;
export const preparedSourcesSchema = z.object({ runtime: z.literal('stable-diffusion-cpp'), upstreams: upstreamsSchema, patches: z.array(patchReceiptSchema), vendorCheckoutModified: z.boolean() }).catchall(jsonSchema);
export type PreparedSources = z.infer<typeof preparedSourcesSchema>;
const imageCapabilitiesSchema = z.object({
  ggufFileOffsetBits: z.number(),
  safetensorsFileOffsetBits: z.number(),
  callerOwnedRandomAccess: z.boolean(),
  upstreamApi: z.boolean(),
  ggufShards: z.boolean(),
  ggufOriginalShapes: z.boolean(),
  bf16WeightRetention: z.boolean(),
}).catchall(jsonSchema);
export type ImageCapabilities = z.infer<typeof imageCapabilitiesSchema>;

// Configuration and toolchain fields are comparison/report metadata. Build
// decisions use the checkout's statically typed configuration instead.
const stableDiffusionProvenanceSchema = z.object({
  runtime: z.literal('stable-diffusion-cpp'),
  profile: z.string(),
  variant: z.string(),
  sourceCommit: z.string(),
  sourceDirty: z.boolean(),
  upstreams: upstreamsSchema,
  configuration: jsonObjectSchema,
  variantConfiguration: jsonObjectSchema,
  cmakeCommand: z.array(z.string()),
  patches: preparedSourcesSchema,
  sourceInputs: sourceInputsSchema,
  preparedSources: treeIdentitySchema,
  compiledInputs: compilerInputsSchema,
  nativeCheck: z.boolean(),
  toolchain: jsonObjectSchema,
  emccVersion: z.string().optional(),
  buildCache: buildCacheSchema,
  validation: stableDiffusionValidationSchema,
}).catchall(jsonSchema);
export type StableDiffusionProvenance = z.infer<typeof stableDiffusionProvenanceSchema>;
const stableDiffusionManifestSchema = z.object({
  formatVersion: z.literal(2),
  runtime: z.literal('stable-diffusion-cpp'),
  abiVersion: z.literal(3),
  sourceCommit: z.string(),
  schemaSha256: z.string(),
  upstreams: upstreamsSchema,
  experimental: z.boolean(),
  capabilities: imageCapabilitiesSchema,
  files: z.array(fileEntrySchema),
  profiles: z.record(z.string(), z.object({ variants: z.record(z.string(), stableDiffusionProvenanceSchema) }).catchall(jsonSchema)),
}).catchall(jsonSchema);
export type StableDiffusionManifest = z.infer<typeof stableDiffusionManifestSchema>;

export function decodeStableDiffusionProvenance({ input }: { input: unknown }): StableDiffusionProvenance {
  return stableDiffusionProvenanceSchema.parse(input);
}

export function decodeStableDiffusionManifest({ input }: { input: unknown }): StableDiffusionManifest {
  return stableDiffusionManifestSchema.parse(input);
}
