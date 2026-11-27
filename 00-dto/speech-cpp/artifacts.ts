import { z } from 'zod';
import { buildCacheSchema, compilerInputsSchema, fileEntrySchema, jsonObjectSchema, jsonSchema, patchReceiptSchema, sourceInputsSchema, treeIdentitySchema, validationSchema } from '#dto-internal/shared/artifacts';
import { browserSessionSchema } from '#dto-internal/shared/external';

const speechBrowserResultSchema = z.object({
  profile: z.string(),
  variant: z.literal('browser'),
  passed: z.boolean(),
  pointerBytes: z.number(),
  apiVersionMajor: z.number(),
  apiVersionMinor: z.number(),
  version: z.string(),
});
export type SpeechBrowserResult = z.infer<typeof speechBrowserResultSchema>;
const speechBrowserEnvelopeSchema = z.object({ session: browserSessionSchema.extend({ runtime: z.literal('speech-cpp') }), results: z.array(speechBrowserResultSchema) });
export type SpeechBrowserEnvelope = z.infer<typeof speechBrowserEnvelopeSchema>;

export function decodeSpeechBrowserEnvelope({ input }: { input: unknown }): SpeechBrowserEnvelope {
  return speechBrowserEnvelopeSchema.parse(input);
}

export const upstreamsSchema = z.object({
  speech: z.object({ path: z.string(), repository: z.string(), commit: z.string() }).catchall(jsonSchema),
  ggml: z.object({ path: z.string(), repository: z.string(), commit: z.string() }).catchall(jsonSchema),
}).catchall(jsonSchema);
export type Upstreams = z.infer<typeof upstreamsSchema>;
export const preparedSourcesSchema = z.object({ runtime: z.literal('speech-cpp'), upstreams: upstreamsSchema, patches: z.array(patchReceiptSchema), vendorCheckoutModified: z.boolean() }).catchall(jsonSchema);
export type PreparedSources = z.infer<typeof preparedSourcesSchema>;
const speechCapabilitiesSchema = z.object({ ggufFileOffsetBits: z.number(), callerOwnedRandomAccess: z.boolean(), upstreamApi: z.boolean() }).catchall(jsonSchema);
export type SpeechCapabilities = z.infer<typeof speechCapabilitiesSchema>;

// Saved build metadata is decoded at the artifact boundary; build decisions use
// this checkout's statically typed configuration.
const speechProvenanceSchema = z.object({
  runtime: z.literal('speech-cpp'),
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
  validation: validationSchema,
}).catchall(jsonSchema);
export type SpeechProvenance = z.infer<typeof speechProvenanceSchema>;
const speechManifestSchema = z.object({
  formatVersion: z.literal(2),
  runtime: z.literal('speech-cpp'),
  abiVersion: z.literal(3),
  sourceCommit: z.string(),
  schemaSha256: z.string(),
  upstreams: upstreamsSchema,
  experimental: z.boolean(),
  capabilities: speechCapabilitiesSchema,
  files: z.array(fileEntrySchema),
  profiles: z.record(z.string(), z.object({ variants: z.record(z.string(), speechProvenanceSchema) }).catchall(jsonSchema)),
}).catchall(jsonSchema);
export type SpeechManifest = z.infer<typeof speechManifestSchema>;

export function decodeSpeechProvenance({ input }: { input: unknown }): SpeechProvenance {
  return speechProvenanceSchema.parse(input);
}

export function decodeSpeechManifest({ input }: { input: unknown }): SpeechManifest {
  return speechManifestSchema.parse(input);
}
