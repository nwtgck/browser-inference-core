/** Synchronous native extension, derived from its real declarations. */
export interface ModelGraphNativeExports {
  _lcb_model_graph_architecture_known(name: bigint): number;
  _lcb_model_graph_last_error(): bigint;
  _lcb_model_graph_sample(workspace: bigint, sampler: bigint, logits: bigint, count: number): number;
  _lcb_model_graph_sampler_workspace_free(workspace: bigint): void;
  _lcb_model_graph_sampler_workspace_new(vocabulary_size: number): bigint;
  _lcb_model_graph_schema_hash(): bigint;
  _lcb_model_graph_version(): number;
  _lcb_model_graph_vocab_free(owner: bigint): void;
  _lcb_model_graph_vocab_get(owner: bigint): bigint;
  _lcb_model_graph_vocab_new(metadata: bigint): bigint;
}

/** C diagnostic contract; implementation availability is recorded separately. */
export interface Q8DiagnosticFunctions {
  version(backend: bigint): number;
  runV1(backend: bigint, dst: bigint, request: bigint, report: bigint, report_capacity: number): Promise<number>;
}
