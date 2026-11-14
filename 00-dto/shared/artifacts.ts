import { z } from 'zod';

export const jsonSchema = z.json();
export type JsonValue = z.infer<typeof jsonSchema>;
export const jsonObjectSchema = z.record(z.string(), jsonSchema);
export type JsonObject = z.infer<typeof jsonObjectSchema>;

export const runtimeNameSchema = z.enum(['llama-cpp', 'stable-diffusion-cpp']);
export type RuntimeName = z.infer<typeof runtimeNameSchema>;

// DTOs describe wire structure. Paths, hashes, sizes and source identities are
// checked by the operation that uses them against the actual files or Git tree.
export const fileIdentitySchema = z.object({ bytes: z.number(), sha256: z.string() }).catchall(jsonSchema);
export type FileIdentity = z.infer<typeof fileIdentitySchema>;
export const sourceIdentitySchema = z.union([fileIdentitySchema, z.object({ symlink: z.string() }).catchall(jsonSchema)]);
export type SourceIdentity = z.infer<typeof sourceIdentitySchema>;
export const fileEntrySchema = fileIdentitySchema.extend({ path: z.string() });
export type FileEntry = z.infer<typeof fileEntrySchema>;
export const treeIdentitySchema = z.object({ files: z.number(), sha256: z.string() }).catchall(jsonSchema);
export type TreeIdentity = z.infer<typeof treeIdentitySchema>;
export const fileIdentitiesSchema = z.record(z.string(), fileIdentitySchema);
export const sourceIdentitiesSchema = z.record(z.string(), sourceIdentitySchema);
export const sourceInputsSchema = z.object({ mode: z.enum(['release', 'local']), publishable: z.boolean(), sourceCommit: z.string(), files: sourceIdentitiesSchema }).catchall(jsonSchema);
export type SourceInputs = z.infer<typeof sourceInputsSchema>;

export const patchReceiptSchema = z.object({
  file: z.string(),
  target: z.string(),
  reason: z.string(),
  mode: z.enum(['apply', 'reverse']),
  sha256: z.string(),
  files: z.array(z.object({ path: z.string(), beforeSha256: z.string().optional(), afterSha256: z.string().optional() }).catchall(jsonSchema)),
}).catchall(jsonSchema);
export type PatchReceipt = z.infer<typeof patchReceiptSchema>;
export const compilerInputsSchema = z.object({
  translationUnits: fileIdentitiesSchema,
  targetSources: z.record(z.string(), fileIdentitiesSchema),
}).catchall(jsonSchema);
export type CompilerInputs = z.infer<typeof compilerInputsSchema>;

export const validationSchema = z.object({ compiled: z.boolean(), browserSmoke: z.boolean().optional(), realModelInference: z.boolean().optional() }).catchall(jsonSchema);
export type Validation = z.infer<typeof validationSchema>;
export const buildCacheSchema = z.object({ identity: z.string().optional(), compilerWrapper: z.string().optional() }).catchall(jsonSchema);
export type BuildCache = z.infer<typeof buildCacheSchema>;

export const rootManifestSchema = z.object({
  formatVersion: z.literal(5),
  sourceCommit: z.string(),
  runtimes: z.object({
    'llama-cpp': z.object({ manifest: z.string(), manifestFormatVersion: z.literal(2) }).catchall(jsonSchema),
    'stable-diffusion-cpp': z.object({ manifest: z.string(), manifestFormatVersion: z.literal(2) }).catchall(jsonSchema),
  }).catchall(jsonSchema),
  files: z.array(fileEntrySchema),
}).catchall(jsonSchema);
export type RootManifest = z.infer<typeof rootManifestSchema>;

export function decodeRootManifest({ input }: { input: unknown }): RootManifest {
  return rootManifestSchema.parse(input);
}

export function decodeStringArray({ input }: { input: unknown }): string[] {
  return z.array(z.string()).parse(input);
}
