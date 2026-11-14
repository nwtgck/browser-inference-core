# DTO ownership and naming

Use `snake_case` for string discriminants and enumerated symbolic values defined
by this project, for example `bigint_u64` and `bigint_native`. This applies to
literal union types, enum members' string values, and schema literals/enums;
it is not limited to the TypeScript `enum` syntax. Update producers and
consumers consistently.

Preserve values whose spelling is specified by an external API or file format.
Values that actually identify a module, package, file, or path must retain that
identifier's spelling. Do not mechanically rename JSON property keys, imported
C/C++ identifiers, or external tool fields under this rule. A value is not
external merely because another module in this repository reads it.

Keep DTOs structural and decoding at actual input boundaries. The root
`AGENTS.md` applies here. ESLint permits Zod imports throughout `00-dto/` and
forbids them outside it; individual DTO files need no registration. Public entries
expose types and named decoders, not raw schemas for consumers to extend.
