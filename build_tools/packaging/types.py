"""Bind host declarations to exactly the profiles present in a package."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re


def write_host_profiles(destination: Path, runtime: str, profiles: Iterable[str], variants: Iterable[str]) -> None:
    profiles, variants = tuple(sorted(profiles)), tuple(sorted(variants))
    if (not profiles or not variants or 'browser' not in variants or
            len(set(profiles)) != len(profiles) or len(set(variants)) != len(variants) or
            any(not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name) for name in (*profiles, *variants))):
        raise ValueError('Invalid package profile or variant selection')
    if runtime not in ('llama-cpp', 'stable-diffusion-cpp'):
        raise ValueError('Unknown runtime type map')
    module = 'MainModule' if runtime == 'llama-cpp' else 'StableDiffusionModule'
    lines = ["/** Generated from this package's profile/variant set. */"]
    if runtime == 'stable-diffusion-cpp':
        lines.append("export type { EmscriptenRuntime as CoreModule } from '../profiles/" + profiles[0] + "/browser/core.d.ts';")
    for i, profile in enumerate(profiles):
        for j, variant in enumerate(variants):
            lines.append(f"import type {{ {module} as Module_{i}_{j} }} from '../profiles/{profile}/{variant}/core.d.ts';")
    lines.append('export interface ModuleByProfile {')
    for i, profile in enumerate(profiles):
        lines.append(f"  '{profile}': {{")
        for j, variant in enumerate(variants):
            lines.append(f"    '{variant}': Module_{i}_{j};")
        lines.append('  };')
    lines += ['}', 'export type Profile = keyof ModuleByProfile;',
              'export type Variant = keyof ModuleByProfile[Profile];']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines) + '\n')
