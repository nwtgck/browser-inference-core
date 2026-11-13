/** Emscripten host services. Native callable exports are generated separately. */
export interface EmscriptenRuntime {
  HEAPU8: Uint8Array;
  /** Emscripten's JS filesystem. Storage and mounts belong to the caller. */
  FS: object;
  addFunction<Args extends (number | bigint)[]>(callback: (...args: Args) => number | bigint | void, signature: string): number | bigint;
  removeFunction(pointer: number | bigint): void;
  ccall(name: string, result: 'number' | 'bigint' | null, types: string[], args: unknown[], options?: { async: boolean }): unknown;
  _sdc_abi_version(): number;
  /** bit 0: 64-bit safetensors offsets; bit 1: complete GGUF shard groups. */
  _sdc_model_io_capabilities(): number;
  _sdc_pointer_bytes(): number;
  _sdc_schema_hash(): bigint;
  _sdc_malloc(bytes: bigint): bigint;
  _sdc_free(pointer: bigint): void;
  _sdc_sizeof_record(record: number): bigint;
  _sdc_alignof_record(record: number): bigint;
  _sdc_offsetof_field(record: number, field: number): bigint;
  _sdc_sizeof_field(record: number, field: number): bigint;
  _sdc_constant(id: number): bigint;
}
export interface CoreOptions {
  wasmBinary?: Uint8Array;
  /** Call onSuccess to complete initialization. The hook's return value is ignored. */
  instantiateWasm?(imports: WebAssembly.Imports, onSuccess: (instance: WebAssembly.Instance, module?: WebAssembly.Module) => void): void;
  locateFile?(path: string): string;
  print?(text: string): void;
  printErr?(text: string): void;
  onAbort?(reason: unknown): void;
}
