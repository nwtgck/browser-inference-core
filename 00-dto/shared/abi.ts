import { z } from 'zod';
import { runtimeNameSchema } from '#dto-internal/shared/artifacts';

export const kindSchema = z.enum(['void', 'pointer', 'record', 'array', 'boolean', 'signed', 'unsigned', 'u64', 'i64', 'float']);
export type Kind = z.infer<typeof kindSchema>;
export const abiTypeSchema = z.object({
  kind: z.enum(['void', 'pointer', 'integer', 'float', 'record', 'array']),
  bits: z.number().optional(),
  signed: z.boolean().optional(),
  pointerSized: z.boolean().optional(),
  record: z.string().optional(),
  callback: z.string().optional(),
  dimensions: z.array(z.number()).optional(),
  get element() {
    return z.object({ kind: kindSchema, abiType: abiTypeSchema }).optional();
  },
});
export type AbiType = z.infer<typeof abiTypeSchema>;
const parameterSchema = z.object({ name: z.string(), cType: z.string(), kind: kindSchema, abiType: abiTypeSchema });
export type AbiParameter = z.infer<typeof parameterSchema>;
const functionSchema = z.object({
  name: z.string(),
  export: z.string(),
  returnType: z.string(),
  returnKind: kindSchema,
  abiType: abiTypeSchema,
  parameters: z.array(parameterSchema),
  bridge: z.boolean(),
});
export type AbiFunction = z.infer<typeof functionSchema>;
const fieldSchema = parameterSchema.extend({ id: z.number(), offset: z.number(), size: z.number() });
export type AbiField = z.infer<typeof fieldSchema>;
const recordSchema = z.object({ name: z.string(), cType: z.string(), id: z.number(), size: z.number(), alignment: z.number(), fields: z.array(fieldSchema) });
export type AbiRecord = z.infer<typeof recordSchema>;
const callbackSchema = z.object({
  name: z.string(),
  cType: z.string(),
  returnType: z.string(),
  returnKind: kindSchema,
  abiType: abiTypeSchema,
  parameters: z.array(parameterSchema),
});
export type AbiCallback = z.infer<typeof callbackSchema>;
export const abiExtensionSchema = z.object({
  extensionId: z.string(),
  version: z.literal(1),
  abiVersion: z.literal(1),
  baseAbiVersion: z.literal(3),
  pointerRepresentation: z.literal('bigint_u64'),
  suspension: z.enum(['none', 'jspi']),
  versionExport: z.string(),
  schemaHashExport: z.string(),
  functions: z.array(functionSchema),
  records: z.array(recordSchema),
  constants: z.array(z.string()),
  constantValues: z.record(z.string(), z.string()),
});
export type AbiExtension = z.infer<typeof abiExtensionSchema>;
const extensionDeclarationSchema = abiExtensionSchema.pick({ extensionId: true, functions: true });
export type AbiExtensionDeclaration = z.infer<typeof extensionDeclarationSchema>;
export const abiSchemaSchema = z.object({
  abiVersion: z.literal(3),
  runtime: runtimeNameSchema,
  pointerRepresentation: z.literal('bigint_native'),
  pointerBytes: z.union([z.literal(4), z.literal(8)]),
  schemaSha256: z.string(),
  functions: z.array(functionSchema),
  records: z.array(recordSchema),
  callbacks: z.array(callbackSchema),
  testFunctions: z.array(functionSchema).optional(),
  constants: z.array(z.string()),
  constantValues: z.record(z.string(), z.string()),
  excluded: z.array(z.object({ name: z.string(), reason: z.string() })),
  exports: z.array(z.string()),
  suspendingExports: z.array(z.string()),
  callbackMetadata: z.object({ version: z.literal(2), getters: z.array(z.object({ name: z.string(), export: z.string() })) }),
  capabilities: z.record(z.string(), z.number()).optional(),
  extensions: z.record(z.string(), abiExtensionSchema).optional(),
  extensionDeclarations: z.array(extensionDeclarationSchema).optional(),
});
export type AbiSchema = z.infer<typeof abiSchemaSchema>;
const abiCatalogSchema = abiSchemaSchema.extend({ targets: z.record(z.string(), abiSchemaSchema) });
export type AbiCatalog = z.infer<typeof abiCatalogSchema>;
const abiExtensionSidecarSchema = abiExtensionSchema.extend({ schemaSha256: z.string() });
export type AbiExtensionSidecar = z.infer<typeof abiExtensionSidecarSchema>;

export function decodeAbiSchema({ input }: { input: unknown }): AbiSchema {
  return abiSchemaSchema.parse(input);
}

export function decodeAbiCatalog({ input }: { input: unknown }): AbiCatalog {
  return abiCatalogSchema.parse(input);
}

export function decodeAbiExtensionSidecar({ input }: { input: unknown }): AbiExtensionSidecar {
  return abiExtensionSidecarSchema.parse(input);
}
