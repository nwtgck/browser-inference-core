/** Runtime-only ABI marshaling. No model policy, tools, compiler, or Node imports. */
const publicKinds = new Set(['void', 'pointer', 'record', 'array', 'boolean', 'signed', 'unsigned', 'u64', 'i64', 'float']);
const nativeKinds = new Set(['void', 'pointer', 'integer', 'float', 'record', 'array']);

function object(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`Invalid ${label}`);
  }
  return value;
}

function array(value, label) {
  if (!Array.isArray(value)) {
    throw new TypeError(`Invalid ${label}`);
  }
  return value;
}

function integer(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TypeError(`Invalid ${label}`);
  }
  return value;
}

function nativeFunction(module, name) {
  const value = module[name];
  if (typeof value !== 'function') {
    throw new TypeError(`Missing native export: ${name}`);
  }
  return value;
}

export function toNativePointer({ value, pointerBytes }) {
  if (typeof value !== 'bigint') {
    throw new TypeError('Native pointer arguments use bigint');
  }
  if (![4, 8].includes(pointerBytes) || value < 0n || value >= 1n << BigInt(pointerBytes * 8)) {
    throw new RangeError('Invalid native pointer');
  }
  return pointerBytes === 4 ? Number(value) : value;
}

export function fromNativePointer({ value, pointerBytes }) {
  if (![4, 8].includes(pointerBytes)) {
    throw new RangeError('Invalid native pointer width');
  }
  if (pointerBytes === 4) {
    if (typeof value !== 'number' || !Number.isInteger(value) || value < -2147483648 || value > 4294967295) {
      throw new RangeError('Invalid wasm32 pointer result');
    }
    return BigInt(value >>> 0);
  }
  if (typeof value !== 'bigint' || value < -(1n << 63n) || value >= 1n << 64n) {
    throw new RangeError('Invalid wasm64 pointer result');
  }
  return BigInt.asUintN(64, value);
}

/** Emscripten's malloc/free exports convert JS addresses even with MEMORY64.
 * Ordinary C pointer exports keep their native Wasm ABI. */
export function toEmscriptenPointer({ value, pointerBytes }) {
  if (typeof value !== 'bigint' || ![4, 8].includes(pointerBytes) || value < 0n || value > BigInt(pointerBytes === 4 ? 4294967295 : Number.MAX_SAFE_INTEGER)) {
    throw new RangeError('Emscripten address exceeds exact JS indexing');
  }
  return Number(value);
}

export function fromEmscriptenPointer({ value, pointerBytes }) {
  if (pointerBytes === 4) {
    return fromNativePointer({ value, pointerBytes });
  }
  if (pointerBytes !== 8 || typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new RangeError('Invalid Emscripten memory64 address');
  }
  return BigInt(value);
}

function fingerprint(module) {
  if (nativeFunction(module, '_bicore_abi_version')() !== 3) {
    throw new Error('Unsupported native ABI version');
  }
  const pointerBytes = nativeFunction(module, '_bicore_pointer_bytes')();
  if (pointerBytes !== 4 && pointerBytes !== 8) {
    throw new Error('Unsupported native pointer width');
  }
  const address = fromNativePointer({ value: nativeFunction(module, '_bicore_schema_hash')(), pointerBytes });
  const start = Number(address);
  const heap = module.HEAPU8;
  if (!Number.isSafeInteger(start) || start <= 0 || !(heap instanceof Uint8Array) || start + 65 > heap.byteLength || heap[start + 64] !== 0) {
    throw new Error('Invalid native ABI fingerprint storage');
  }
  const hash = new TextDecoder().decode(heap.subarray(start, start + 64));
  if (!/^[0-9a-f]{64}$/u.test(hash)) {
    throw new Error('Invalid native ABI fingerprint');
  }
  return { hash, pointerBytes };
}

function checkType(value) {
  const type = object(value, 'native type');
  if (!nativeKinds.has(type.kind)) {
    throw new TypeError('Unsupported native type');
  }
  if (['pointer', 'integer', 'float'].includes(type.kind) && ![8, 16, 32, 64].includes(type.bits)) {
    throw new TypeError('Invalid native scalar width');
  }
  if (type.kind === 'integer' && typeof type.signed !== 'boolean') {
    throw new TypeError('Missing native integer signedness');
  }
  if (type.kind === 'record' && typeof type.record !== 'string') {
    throw new TypeError('Missing native record type');
  }
  return type;
}

/** Select by the compiled fingerprint, rather than guessing from a profile name. */
export function selectAbiSchema({ module, schema }) {
  const native = fingerprint(module);
  const input = object(schema, 'ABI schema');
  const selected = input.targets === undefined ? input : object(object(input.targets, 'ABI targets')[native.hash], 'compiled ABI target');
  if (selected.abiVersion !== 3 || selected.pointerRepresentation !== 'bigint_native' || selected.pointerBytes !== native.pointerBytes || selected.schemaSha256 !== native.hash) {
    throw new Error('The compiled core and ABI metadata do not match');
  }
  const names = new Set();
  for (const fn of array(selected.functions, 'ABI functions')) {
    object(fn, 'ABI function');
    if (typeof fn.name !== 'string' || typeof fn.export !== 'string' || names.has(fn.name) || !publicKinds.has(fn.returnKind) || typeof fn.bridge !== 'boolean') {
      throw new TypeError('Invalid or duplicate ABI function');
    }
    names.add(fn.name);
    checkType(fn.abiType);
    for (const parameter of array(fn.parameters, 'ABI parameters')) {
      object(parameter, 'ABI parameter');
      if (!publicKinds.has(parameter.kind) || ['void', 'array'].includes(parameter.kind)) {
        throw new TypeError('Invalid function parameter kind');
      }
      checkType(parameter.abiType);
    }
    nativeFunction(module, fn.export);
  }
  const records = new Set();
  for (const record of array(selected.records, 'ABI records')) {
    object(record, 'ABI record');
    if (typeof record.name !== 'string' || records.has(record.name)) {
      throw new TypeError('Invalid or duplicate ABI record');
    }
    records.add(record.name);
    integer(record.size, 'record size');
    if (!integer(record.alignment, 'record alignment')) {
      throw new TypeError('Invalid record alignment');
    }
    const fields = new Set();
    for (const field of array(record.fields, 'ABI fields')) {
      object(field, 'ABI field');
      if (typeof field.name !== 'string' || fields.has(field.name) || !publicKinds.has(field.kind)) {
        throw new TypeError('Invalid or duplicate ABI field');
      }
      fields.add(field.name);
      checkType(field.abiType);
      if (integer(field.offset, 'field offset') + integer(field.size, 'field size') > record.size) {
        throw new RangeError('Field exceeds native record');
      }
    }
  }
  const constants = object(selected.constantValues, 'ABI constants');
  for (const name of array(selected.constants, 'constant names')) {
    if (typeof name !== 'string' || typeof constants[name] !== 'string' || !/^-?[0-9]+$/u.test(constants[name])) {
      throw new TypeError('Invalid native constant');
    }
  }
  const callback = object(selected.callbackMetadata, 'synchronous metadata surface');
  if (callback.version !== 2) {
    throw new Error('Unsupported synchronous metadata contract');
  }
  for (const getter of array(callback.getters, 'synchronous getters')) {
    if (!getter || typeof getter.name !== 'string' || typeof getter.export !== 'string' || !selected.functions.some(fn => fn.name === getter.name && fn.export === getter.export)) {
      throw new Error('Synchronous getter does not belong to the compiled API');
    }
  }
  array(selected.suspendingExports, 'suspending native exports');
  return selected;
}

function encode(value, kind, type, pointerBytes) {
  if (kind === 'pointer' || kind === 'record') {
    return toNativePointer({ value, pointerBytes });
  }
  if (type.kind === 'float') {
    if (typeof value !== 'number') {
      throw new TypeError('Invalid native floating-point argument');
    }
    return value;
  }
  const wide = kind === 'u64' || kind === 'i64';
  if (!wide && type.bits <= 32) {
    const minimum = type.signed ? -(2 ** (type.bits - 1)) : 0;
    const maximum = kind === 'boolean' ? 1 : type.signed ? 2 ** (type.bits - 1) - 1 : 2 ** type.bits - 1;
    if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum || value > maximum) {
      throw new RangeError('Native integer exceeds its declared width');
    }
    return value;
  }
  if (wide ? typeof value !== 'bigint' : typeof value !== 'number' || !Number.isInteger(value)) {
    throw new TypeError('Invalid native integer representation');
  }
  const number = BigInt(value);
  const bits = BigInt(type.bits);
  const minimum = type.signed ? -(1n << (bits - 1n)) : 0n;
  const maximum = kind === 'boolean' ? 1n : type.signed ? (1n << (bits - 1n)) - 1n : (1n << bits) - 1n;
  if (number < minimum || number > maximum) {
    throw new RangeError('Native integer exceeds its declared width');
  }
  return type.bits === 64 ? number : Number(number);
}

function decode(value, kind, type, pointerBytes) {
  if (kind === 'void' || kind === 'record') {
    if (value !== undefined) {
      throw new TypeError('Expected a void native result');
    }
    return;
  }
  if (kind === 'pointer') {
    return fromNativePointer({ value, pointerBytes });
  }
  if (type.kind === 'float') {
    if (typeof value !== 'number') {
      throw new TypeError('Invalid native floating-point result');
    }
    return value;
  }
  if (type.bits <= 32) {
    if (typeof value !== 'number' || !Number.isInteger(value) || value < -2147483648 || value > 4294967295) {
      throw new TypeError('Invalid native integer result representation');
    }
    const shift = 32 - type.bits;
    const result = type.signed ? (value << shift) >> shift : type.bits === 32 ? value >>> 0 : value & (2 ** type.bits - 1);
    if (kind === 'boolean' && result !== 0 && result !== 1) {
      throw new RangeError('Invalid native boolean result');
    }
    return kind === 'u64' || kind === 'i64' ? BigInt(result) : result;
  }
  if (typeof value !== 'bigint') {
    throw new TypeError('Invalid native integer result representation');
  }
  const raw = BigInt(value);
  const result = type.signed ? BigInt.asIntN(type.bits, raw) : BigInt.asUintN(type.bits, raw);
  if (kind === 'boolean' && result !== 0n && result !== 1n) {
    throw new RangeError('Invalid native boolean result');
  }
  return kind === 'u64' || kind === 'i64' ? result : Number(result);
}

// Compiler facts are invariant for the lifetime of this caller. Precompute the
// common token/enum/bool bounds instead of exponentiating and dispatching on
// their public/native kinds for every inference call.
function argumentEncoder(parameter, pointerBytes) {
  const { kind, abiType: type } = parameter;
  if (type.kind === 'integer' && type.bits <= 32 && kind !== 'u64' && kind !== 'i64') {
    const minimum = type.signed ? -(2 ** (type.bits - 1)) : 0;
    const maximum = kind === 'boolean' ? 1 : type.signed ? 2 ** (type.bits - 1) - 1 : 2 ** type.bits - 1;
    return value => {
      if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum || value > maximum) {
        throw new RangeError('Native integer exceeds its declared width');
      }
      return value;
    };
  }
  return value => encode(value, kind, type, pointerBytes);
}

function resultDecoder(fn, pointerBytes) {
  const { returnKind: kind, abiType: type } = fn;
  if (type.kind === 'integer' && type.bits <= 32 && kind !== 'u64' && kind !== 'i64') {
    const shift = 32 - type.bits;
    const mask = 2 ** type.bits - 1;
    return value => {
      if (typeof value !== 'number' || !Number.isInteger(value) || value < -2147483648 || value > 4294967295) {
        throw new TypeError('Invalid native integer result representation');
      }
      const result = type.signed ? (value << shift) >> shift : type.bits === 32 ? value >>> 0 : value & mask;
      if (kind === 'boolean' && result !== 0 && result !== 1) {
        throw new RangeError('Invalid native boolean result');
      }
      return result;
    };
  }
  return value => decode(value, kind, type, pointerBytes);
}

/** The owner supplies its gate so ordinary calls and optional extensions share
 * one lifetime. Conversion, suspension, result checking and release use one
 * async frame; the generated facade returns that Promise directly. */
export function createAbiCaller({ module, schema: input, suspension, gate }) {
  if (!['direct', 'jspi', 'asyncify'].includes(suspension)) {
    throw new TypeError('Unknown native suspension mode');
  }
  if (suspension === 'asyncify' && typeof module.ccall !== 'function') {
    throw new TypeError('Asyncify requires the Emscripten ccall helper');
  }
  const schema = selectAbiSchema({ module, schema: input });
  const records = new Map(schema.records.map(record => [record.name, record]));
  const synchronous = new Set(schema.callbackMetadata.getters.map(getter => getter.name));
  const scalarType = (parameter) => parameter.kind === 'record' || parameter.kind === 'pointer' ? schema.pointerBytes === 8 ? 'bigint' : 'number' : parameter.abiType.bits === 64 && parameter.abiType.kind === 'integer' ? 'bigint' : 'number';
  const functions = new Map(schema.functions.map(fn => {
    const parameters = fn.returnKind === 'record' ? [{ kind: 'record', abiType: fn.abiType }, ...fn.parameters] : fn.parameters;
    return [fn.name, { fn, parameters, encode: parameters.map(parameter => argumentEncoder(parameter, schema.pointerBytes)), decode: resultDecoder(fn, schema.pointerBytes), native: nativeFunction(module, fn.export), kinds: parameters.map(scalarType), returns: ['record', 'void'].includes(fn.returnKind) ? undefined : scalarType({ kind: fn.returnKind, abiType: fn.abiType }) }];
  }));
  const argumentsFor = (call, args) => {
    if (call.parameters.length !== args.length) {
      throw new TypeError(`Incorrect native argument count: ${call.fn.name}`);
    }
    let converted;
    for (let index = 0; index < call.parameters.length; index++) {
      const parameter = call.parameters[index];
      const value = args[index];
      if (parameter.kind === 'record') {
        const record = records.get(parameter.abiType.record);
        if (!record || typeof value !== 'bigint' || value <= 0n || value % BigInt(record.alignment) !== 0n || value + BigInt(record.size) > BigInt(module.HEAPU8.byteLength)) {
          throw new RangeError('Invalid native record storage');
        }
      }
      const encoded = call.encode[index](value);
      if (encoded !== value) {
        converted ??= [...args];
      }
      if (converted) {
        converted[index] = encoded;
      }
    }
    return converted ?? args;
  };
  const invoke = async ({ name, args, resultType }) => {
    gate?.enter();
    try {
      const call = functions.get(name);
      if (!call) {
        throw new TypeError(`Unknown native function: ${name}`);
      }
      const nativeArgs = argumentsFor(call, args);
      const value = await (suspension === 'asyncify'
        ? module.ccall(call.fn.export.slice(1), call.returns, call.kinds, nativeArgs, { async: true })
        : call.native(...nativeArgs));
      const result = call.decode(value);
      if (resultType !== undefined && typeof result !== resultType) {
        throw new TypeError(`Expected a ${resultType} native result: ${name}`);
      }
      return result;
    } finally {
      gate?.leave();
    }
  };
  return {
    schema,
    call: ({ name, args }) => invoke({ name, args, resultType: undefined }),
    invoker: {
      number: ({ name, args }) => invoke({ name, args, resultType: 'number' }),
      bigint: ({ name, args }) => invoke({ name, args, resultType: 'bigint' }),
      void: ({ name, args }) => invoke({ name, args, resultType: 'undefined' }),
    },
    callSync({ name, args }) {
      if (!synchronous.has(name)) {
        throw new Error('Only declared synchronous leaf functions may bypass the host gate');
      }
      const call = functions.get(name);
      if (!call) {
        throw new TypeError(`Unknown native function: ${name}`);
      }
      const value = call.native(...argumentsFor(call, args));
      if (value && typeof value.then === 'function') {
        throw new Error('A synchronous leaf returned a Promise');
      }
      return call.decode(value);
    },
  };
}

/** Shared memory helpers. The injected generated facade gives callers the full
 * static API without casting a dynamically populated object to an interface. */
export function attachAbiCore({ module, schema, suspension, createApi }) {
  let busy = false;
  const assertIdle = () => {
    if (busy) {
      throw new Error('A native call is pending; serialize access to this module');
    }
  };
  const caller = createAbiCaller({
    module,
    schema,
    suspension,
    gate: {
      enter() {
        assertIdle();
        busy = true;
      },
      leave() {
        busy = false;
      },
    },
  });
  const selected = caller.schema;
  const records = new Map(selected.records.map(record => [record.name, record]));
  const api = createApi ? createApi({ invoker: caller.invoker }) : Object.fromEntries(selected.functions.map(fn => [fn.name, (...args) => caller.call({ name: fn.name, args })]));
  const index = value => {
    const number = Number(value);
    if (!Number.isSafeInteger(number) || number < 0) {
      throw new RangeError('Unsafe native memory index');
    }
    return number;
  };
  const recordInfo = name => {
    const record = records.get(name);
    if (!record) {
      throw new TypeError(`Unknown native record: ${name}`);
    }
    return record;
  };
  const fieldInfo = (name, field) => {
    const result = recordInfo(name).fields.find(entry => entry.name === field);
    if (!result) {
      throw new TypeError(`Unknown native field: ${name}.${field}`);
    }
    return result;
  };
  const span = (pointer, length) => {
    const start = index(pointer), size = index(length), heap = module.HEAPU8;
    if (start > heap.byteLength || size > heap.byteLength - start) {
      throw new RangeError('Native memory access out of bounds');
    }
    return new Uint8Array(heap.buffer, heap.byteOffset + start, size);
  };
  const view = (name, pointer, field) => {
    const info = fieldInfo(name, field);
    const bytes = span(BigInt(pointer) + BigInt(info.offset), info.size);
    return { info, data: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength) };
  };
  const core = {
    module,
    api,
    schema: selected,
    pointerBytes: selected.pointerBytes,
    get busy() {
      return busy;
    },
    constant(name) {
      const raw = selected.constantValues[name];
      if (raw === undefined) {
        throw new TypeError(`Unknown native constant: ${name}`);
      }
      const value = Number(BigInt(raw));
      if (!Number.isSafeInteger(value)) {
        throw new RangeError('Constant cannot be represented exactly as a number');
      }
      return value;
    },
    alloc(bytes) {
      assertIdle();
      const size = BigInt(index(bytes));
      if (size === 0n) {
        throw new RangeError('Allocation size must be positive');
      }
      const pointer = fromEmscriptenPointer({ value: nativeFunction(module, '_malloc')(toEmscriptenPointer({ value: size, pointerBytes: selected.pointerBytes })), pointerBytes: selected.pointerBytes });
      if (pointer === 0n) {
        throw new Error('Native allocation failed');
      }
      return pointer;
    },
    free(pointer) {
      assertIdle();
      nativeFunction(module, '_free')(toEmscriptenPointer({ value: pointer, pointerBytes: selected.pointerBytes }));
    },
    recordSize(name) {
      return recordInfo(name).size;
    },
    recordAlignment(name) {
      return recordInfo(name).alignment;
    },
    allocRecord(name) {
      const size = recordInfo(name).size;
      const pointer = core.alloc(size);
      span(pointer, size).fill(0);
      return pointer;
    },
    fieldAddress(name, pointer, field) {
      return BigInt(pointer) + BigInt(fieldInfo(name, field).offset);
    },
    getField(name, pointer, field) {
      const { info, data } = view(name, pointer, field);
      if (['record', 'array'].includes(info.kind)) {
        throw new TypeError('Use fieldAddress for compound fields');
      }
      if (info.kind === 'float') {
        return info.size === 4 ? data.getFloat32(0, true) : data.getFloat64(0, true);
      }
      const signed = info.abiType.signed === true;
      const value = data[`get${info.size === 8 ? 'Big' : ''}${signed ? 'Int' : 'Uint'}${info.size * 8}`](0, true);
      return ['pointer', 'u64', 'i64'].includes(info.kind) ? BigInt(value) : value;
    },
    setField(name, pointer, field, value) {
      assertIdle();
      const { info, data } = view(name, pointer, field);
      if (['record', 'array'].includes(info.kind)) {
        throw new TypeError('Use fieldAddress for compound fields');
      }
      const input = ['pointer', 'u64', 'i64'].includes(info.kind) && typeof value === 'number' && Number.isSafeInteger(value) ? BigInt(value) : value;
      const raw = encode(input, info.kind, info.abiType, selected.pointerBytes);
      if (info.kind === 'float') {
        data[info.size === 4 ? 'setFloat32' : 'setFloat64'](0, raw, true);
        return;
      }
      const signed = info.abiType.signed === true;
      data[`set${info.size === 8 ? 'Big' : ''}${signed ? 'Int' : 'Uint'}${info.size * 8}`](0, info.size === 8 ? BigInt(raw) : Number(raw), true);
    },
    bytes: span,
    utf8(text) {
      const bytes = new TextEncoder().encode(text);
      const pointer = core.alloc(bytes.length + 1);
      const output = span(pointer, bytes.length + 1);
      output.set(bytes);
      output[bytes.length] = 0;
      return pointer;
    },
    readUtf8(pointer, maximum = 1048576) {
      const start = index(pointer);
      if (start === 0) {
        return null;
      }
      const heap = module.HEAPU8;
      if (start >= heap.byteLength) {
        throw new RangeError('Native string pointer out of bounds');
      }
      const end = Math.min(heap.byteLength, start + index(maximum));
      let stop = start;
      while (stop < end && heap[stop] !== 0) {
        stop++;
      }
      if (stop === end) {
        throw new RangeError('Unterminated string within requested limit');
      }
      return new TextDecoder().decode(heap.subarray(start, stop));
    },
  };
  return core;
}
