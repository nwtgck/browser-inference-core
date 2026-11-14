---
name: named-args-lint
description: Fix Bicore named-argument lint errors by using required destructured inline object arguments or proving a real external TypeScript callback contract. Preserve native callbacks, positional language protocols, and behavior without broad suppressions.
---

# Named arguments in runtime tooling

Use this skill when `local-rules-named-args/require-named-args` reports an error.
Read the callable and its consumers before choosing a fix. The rule deliberately
does not autofix: argument bags, cohesive values, and external contracts have
different calling conventions.

## Owned callables

Owned TypeScript callables use one required destructured object parameter with
an explicit inline outer shape. Zero-argument callables may use `()`.

```ts
function read({ path }: { path: string }) {}
function status() {}
```

Avoid positional arguments, an identifier holding the argument bag, an optional
outer argument, and a default outer argument.

```ts
function read(path: string) {}
function read(options: { path: string }) {}
function read({ path }: { path: string } = { path: '.' }) {}
```

Avoid parameter-property defaults such as `{ path = '.' }`. Pass explicit values,
including `undefined`, at the caller. Preserve existing fallback behavior when
moving a default; do not silently change what omitted or undefined values mean.

Destructure inline object parameters even if the original identifier is named
`params`, `options`, `config`, or `request`. When the implementation stores the
whole argument object, reconstruct it without changing its contents or lifetime.

```ts
constructor({ endpoint, headers }: { endpoint: string; headers?: [string, string][] }) {
  this.configuration = { endpoint, headers };
}
```

## Keep the outer fields visible

An alias, interface, imported model, `Parameters<...>[0]`, `Pick`, `Omit`, or
schema-inferred type is not an inline outer shape.

```ts
// Avoid this argument bag.
function read({ path, mode }: ReadArguments) {}

// Independent arguments remain visible; their property types can be shared.
function read({ path, mode }: { path: ReadArguments['path']; mode: ReadArguments['mode'] }) {}

// A cohesive value remains one value instead of expanding its fields.
function publish({ publication }: { publication: Publication }) {}
```

Preserve optional properties, readonly modifiers, explicit `undefined`, and
discriminated-union correlations. Do not silently turn `option?: T` into a
required `option: T | undefined` while inlining a type.
Prefer `undefined` for absent values unless an external contract requires `null`.

Unions and intersections are allowed when every branch exposes its fields.
The standard TypeScript `Readonly<{ ... }>` wrapper is allowed when the checker
verifies its declaration. `Readonly<Arguments>` and a locally shadowed wrapper
do not expose an inline outer shape.

For shared owned callable contracts, put the inline shape in the canonical
signature and let genuinely contextually typed implementations infer it.
Removing an annotation without a contextual signature is not a fix. A class's
`implements` clause alone does not infer its parameter types.
Do not add casts or weaken a shared contract to bypass the rule.

## External callbacks need a type-based reason

Never exempt a callable only because its name or parameter looks familiar:
`resolve`, `reject`, `event`, `listener`, `callback`, `handler`, `start`, `write`,
`set`, and similar names are investigation hints, not external contracts.

An external parameter type also does not make an owned callback external.

```ts
// Owned despite its external parameter type.
type Listener = (event: Event) => void;

// Owned named-argument contract.
type Listener = ({ event }: { event: Event }) => void;
```

Directly passing a function to another owned function does not exempt its
signature. Preserve a positional callback only when the checker can trace its
callable signature to a standard library or dependency declaration.

```ts
items.map((item, index) => summarize({ item, index }));
promise.then((value) => consume({ value }));

import type { RequestListener } from 'node:http';
const listener: RequestListener = (request, response) => {
  handleRequest({ request, response });
};
```

Do not rewrite platform callback types by hand. Reference the native function
or property type, including for stored callbacks and assignment right-hand sides.

```ts
let resolveResult: ReturnType<typeof Promise.withResolvers<Result>>['resolve'];
const onMessage: NonNullable<Window['onmessage']> = (event) => receive({ event });
```

A local class named `ReadableStream` or a method named `then` is not evidence of
a native calling convention. Preserve the real standard Web Streams callback
through its actual TypeScript contract.

When an object or class implements an external interface, only methods present
in that interface receive that exception. New local methods and constructors
remain owned. The same restriction applies to abstract methods and overloads.

Type predicates and tagged-template signatures retain their language-defined
positional convention. Do not add project-specific exceptions for names or paths.

## Fix order and precise exceptions

1. Convert owned signatures and their callers to named arguments.
2. Destructure existing inline object parameters.
3. Wrap a cohesive model under one meaningful argument name.
4. Reference an external callable type when adapting a native callback.
5. Keep short native adapters inline and put substantive logic in named-argument helpers.
6. Use a focused disable only for a real external contract that cannot be represented by its public types.

For example, if an external library has a runtime method absent from its public
declarations, name the library and method and explain the missing type in the
local exception. “This is external,” name-based exclusions, and TODO-only
suppression comments do not establish that contract.

Preserve behavior: update call sites and callback consumers together, and retain
object-rest snapshots, validation before access, exception handling, and awaited
delivery. Keep existing one-line signatures on one line; avoid unrelated layout
changes.

The configuration applies to `00-dto/**/*.ts`, `tools/**/*.ts`, and `runtimes/*/tools/**/*.ts`,
excluding `*.test.ts` and `*.spec.ts`. Excluded test call sites still need updating
when a production signature changes. Test-resource and import-boundary rules
remain active independently for owned JavaScript and TypeScript files.

Before finishing, review new named-argument disables, run `npm run test:lint`,
and run lint and type checks for the affected implementation and consumers.
Do not introduce a new framework or broaden a local exception to avoid a
diagnostic.
