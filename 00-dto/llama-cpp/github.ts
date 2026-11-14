import { z } from 'zod';

// GitHub owns these response fields and their null conventions. Decode the
// projection used by automation; URLs, identifiers and status policy belong to
// the operation that selects an upstream revision or updates a comment.
const releaseSchema = z.object({ draft: z.boolean(), prerelease: z.boolean(), published_at: z.string().nullable(), tag_name: z.string(), html_url: z.string() });
export type GitHubRelease = z.infer<typeof releaseSchema>;

export function decodeGitHubRelease({ input }: { input: unknown }): GitHubRelease {
  return releaseSchema.parse(input);
}

export function decodeGitHubReleases({ input }: { input: unknown }): GitHubRelease[] {
  return z.array(releaseSchema).parse(input);
}

const compareSchema = z.object({ status: z.string() });
export type GitHubCompare = z.infer<typeof compareSchema>;

export function decodeGitHubCompare({ input }: { input: unknown }): GitHubCompare {
  return compareSchema.parse(input);
}

const repositorySchema = z.object({ full_name: z.string() });
const pullRequestSchema = z.object({ number: z.number(), state: z.string(), head: z.object({ sha: z.string(), ref: z.string(), repo: repositorySchema.nullable() }) });
export type GitHubPullRequest = z.infer<typeof pullRequestSchema>;

export function decodeGitHubPullRequest({ input }: { input: unknown }): GitHubPullRequest {
  return pullRequestSchema.parse(input);
}

export function decodeGitHubPullRequests({ input }: { input: unknown }): GitHubPullRequest[] {
  return z.array(pullRequestSchema).parse(input);
}

const workflowRunSchema = z.object({
  id: z.number(),
  run_attempt: z.number(),
  head_sha: z.string(),
  head_branch: z.string().nullable(),
  head_repository: repositorySchema.nullable(),
  event: z.string(),
  status: z.string(),
  conclusion: z.string().nullable(),
  html_url: z.string(),
  run_started_at: z.string().nullable().optional(),
  created_at: z.string().optional(),
});
export type GitHubWorkflowRun = z.infer<typeof workflowRunSchema>;

export function decodeGitHubWorkflowRuns({ input }: { input: unknown }): GitHubWorkflowRun[] {
  return z.array(workflowRunSchema).parse(input);
}

const artifactSchema = z.object({ id: z.number(), name: z.string(), expired: z.boolean() });
export type GitHubArtifact = z.infer<typeof artifactSchema>;

export function decodeGitHubArtifacts({ input }: { input: unknown }): GitHubArtifact[] {
  return z.array(artifactSchema).parse(input);
}

const issueCommentSchema = z.object({ id: z.number(), user: z.object({ login: z.string(), type: z.string().optional() }), body: z.string().nullable().optional() });
export type GitHubIssueComment = z.infer<typeof issueCommentSchema>;

export function decodeGitHubIssueComments({ input }: { input: unknown }): GitHubIssueComment[] {
  return z.array(issueCommentSchema).parse(input);
}

const consumerReportEnvelopeSchema = z.object({ schemaVersion: z.literal(1), repository: z.string(), sourceCommit: z.string(), artifactCommit: z.string(), runId: z.number(), runAttempt: z.number(), markdownSha256: z.string() });
export type ConsumerReportEnvelope = z.infer<typeof consumerReportEnvelopeSchema>;

export function decodeConsumerReportEnvelope({ input }: { input: unknown }): ConsumerReportEnvelope {
  return consumerReportEnvelopeSchema.parse(input);
}
