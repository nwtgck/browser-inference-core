import { z } from 'zod';
import { fileIdentitySchema, jsonObjectSchema, jsonSchema, patchReceiptSchema, treeIdentitySchema } from '#dto-internal/shared/artifacts';
import { llamaValidationSchema, preparationOptionsSchema } from '#dto-internal/llama-cpp/artifacts';
import { stableDiffusionValidationSchema, upstreamsSchema } from '#dto-internal/stable-diffusion-cpp/artifacts';

const patchSummarySchema = patchReceiptSchema.pick({ file: true, target: true, mode: true, sha256: true, reason: true });
export type PatchSummary = z.infer<typeof patchSummarySchema>;
const sourceFactsSchema = z.object({
  sourceCommit: z.string(),
  runtimes: z.object({
    'llama-cpp': z.object({
      upstream: z.string(),
      preparations: z.record(z.string(), z.object({ upstream: treeIdentitySchema, options: preparationOptionsSchema, patches: z.array(patchSummarySchema), prepared: treeIdentitySchema }).catchall(jsonSchema)),
      verifiedVariants: z.array(z.string()),
    }).catchall(jsonSchema),
    'stable-diffusion-cpp': z.object({ upstreams: upstreamsSchema, preparation: treeIdentitySchema, patches: z.array(patchSummarySchema), verifiedVariants: z.array(z.string()) }).catchall(jsonSchema),
  }).catchall(jsonSchema),
}).catchall(jsonSchema);
export type SourceFacts = z.infer<typeof sourceFactsSchema>;
const llamaReportProfilesSchema = z.record(z.string(), z.object({ variants: z.record(z.string(), z.object({ configuration: jsonObjectSchema, variantConfiguration: jsonObjectSchema, validation: llamaValidationSchema }).catchall(jsonSchema)) }).catchall(jsonSchema));
const imageReportProfilesSchema = z.record(z.string(), z.object({ variants: z.record(z.string(), z.object({ configuration: jsonObjectSchema, variantConfiguration: jsonObjectSchema, validation: stableDiffusionValidationSchema }).catchall(jsonSchema)) }).catchall(jsonSchema));
const reportDataSchema = z.object({
  sourceCommit: z.string(),
  package: z.object({ name: z.literal('browser-inference-core'), version: z.string() }).catchall(jsonSchema),
  source: sourceFactsSchema,
  runtimes: z.object({
    'llama-cpp': z.object({ manifest: z.string(), abiVersion: z.literal(3), schemaSha256: z.string(), upstreams: z.object({ llama: z.string() }).catchall(jsonSchema), profiles: llamaReportProfilesSchema }).catchall(jsonSchema),
    'stable-diffusion-cpp': z.object({ manifest: z.string(), abiVersion: z.literal(3), schemaSha256: z.string(), upstreams: upstreamsSchema, profiles: imageReportProfilesSchema }).catchall(jsonSchema),
  }).catchall(jsonSchema),
}).catchall(jsonSchema);
export type ReportData = z.infer<typeof reportDataSchema>;
const frozenPublicationSchema = z.object({
  schemaVersion: z.literal(2),
  sourceCommit: z.string(),
  tree: z.string(),
  branch: z.string(),
  manifest: fileIdentitySchema,
  runtimeManifests: z.object({ 'llama-cpp': fileIdentitySchema, 'stable-diffusion-cpp': fileIdentitySchema }).catchall(jsonSchema),
  report: z.object({ sha256: z.string(), data: reportDataSchema }).catchall(jsonSchema),
}).catchall(jsonSchema);
export type FrozenPublication = z.infer<typeof frozenPublicationSchema>;

export function decodeFrozenPublication({ input }: { input: unknown }): FrozenPublication {
  return frozenPublicationSchema.parse(input);
}

const licenseNoticeSchema = z.object({ path: z.string(), sourcePath: z.string().optional() });
export type LicenseNotice = z.infer<typeof licenseNoticeSchema>;
const licenseComponentSchema = z.object({ id: z.string(), name: z.string(), version: z.string().nullable(), license: z.string().nullable(), notices: z.array(licenseNoticeSchema) });
export type LicenseComponent = z.infer<typeof licenseComponentSchema>;
const licenseIndexSchema = z.object({ components: z.array(licenseComponentSchema) });
export type LicenseIndex = z.infer<typeof licenseIndexSchema>;

export function decodeLicenseIndex({ input }: { input: unknown }): LicenseIndex {
  return licenseIndexSchema.parse(input);
}

const packageMetadataSchema = z.object({
  name: z.string(),
  version: z.string(),
  type: z.string(),
  exports: jsonSchema,
  license: z.string().optional(),
  scripts: jsonSchema.optional(),
  dependencies: jsonSchema.optional(),
  devDependencies: jsonSchema.optional(),
  optionalDependencies: jsonSchema.optional(),
  workspaces: jsonSchema.optional(),
  peerDependencies: jsonSchema.optional(),
  peerDependenciesMeta: jsonSchema.optional(),
  bundleDependencies: jsonSchema.optional(),
  bundledDependencies: jsonSchema.optional(),
});
export type PackageMetadata = z.infer<typeof packageMetadataSchema>;

export function decodePackageMetadata({ input }: { input: unknown }): PackageMetadata {
  return packageMetadataSchema.parse(input);
}

const npmPackResultSchema = z.array(z.object({ files: z.array(z.object({ path: z.string(), size: z.number() })) }));
export type NpmPackResult = z.infer<typeof npmPackResultSchema>;

export function decodeNpmPack({ input }: { input: unknown }): NpmPackResult {
  return npmPackResultSchema.parse(input);
}

// These named fields describe npm-owned install behavior. Their contents are
// compared or rejected by the consumer. Other npm fields are opaque report data
// (for example integrity); they never decide which files or commands are used.
const npmPackageEntrySchema = z.object({
  version: z.string().optional(),
  resolved: z.string().optional(),
  dependencies: z.record(z.string(), z.string()).optional(),
  hasInstallScript: jsonSchema.optional(),
  optionalDependencies: jsonSchema.optional(),
  peerDependencies: jsonSchema.optional(),
  peerDependenciesMeta: jsonSchema.optional(),
  bundleDependencies: jsonSchema.optional(),
  bundledDependencies: jsonSchema.optional(),
}).catchall(jsonSchema);
export type NpmPackageEntry = z.infer<typeof npmPackageEntrySchema>;
const npmLockSchema = z.object({ lockfileVersion: z.number(), packages: z.record(z.string(), npmPackageEntrySchema) });
export type NpmLock = z.infer<typeof npmLockSchema>;

export function decodeNpmLock({ input }: { input: unknown }): NpmLock {
  return npmLockSchema.parse(input);
}
