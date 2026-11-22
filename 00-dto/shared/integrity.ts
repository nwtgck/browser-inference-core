import { z } from 'zod';

// This scheme is shared with Naidan; its spelling is part of the external
// fingerprint format, not a project-owned discriminant.
export const treeFingerprintSchema = z.object({
  algorithm: z.literal('sha256'),
  scheme: z.literal('git-committed-tree-sha256-v1'),
  value: z.string(),
  reproduce: z.string(),
}).strict();
export type TreeFingerprint = z.infer<typeof treeFingerprintSchema>;

const artifactIntegritySchema = z.object({
  schema_version: z.literal(1),
  kind: z.literal('browser_inference_core_artifact_integrity'),
  source: z.object({
    repository: z.string(),
    commit: z.string(),
    tree_fingerprint: treeFingerprintSchema,
  }).strict(),
  manifest: z.object({
    path: z.literal('manifest.json'),
    size: z.number(),
    sha256: z.string(),
    contents: z.string(),
  }).strict(),
}).strict();
export type ArtifactIntegrity = z.infer<typeof artifactIntegritySchema>;

/** Independently downloaded YAML is an external, untrusted input. */
export function decodeArtifactIntegrity({ input }: { input: unknown }): ArtifactIntegrity {
  return artifactIntegritySchema.parse(input);
}
