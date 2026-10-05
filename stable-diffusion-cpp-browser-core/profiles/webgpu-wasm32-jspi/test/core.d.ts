/** Policy-free native module ABI 2. Use the optional examples/runtime helpers. */
export interface StableDiffusionModule {
  HEAPU8: Uint8Array;
  /** Emscripten's JS filesystem. Storage and mounts belong to the caller. */
  FS: object;
  addFunction(callback: (...args: (number | bigint)[]) => number | void, signature: string): number | bigint;
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
  [exportName: string]: unknown;
}
export interface CoreOptions {
  wasmBinary?: Uint8Array;
  locateFile?(path: string): string;
  print?(text: string): void;
  printErr?(text: string): void;
  onAbort?(reason: unknown): void;
}
export default function createStableDiffusionCore(options?: CoreOptions): Promise<StableDiffusionModule>;
