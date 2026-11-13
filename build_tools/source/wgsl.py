"""Embed canonical WGSL after template patches; never rewrite generated C++.

The upstream generator owns include expansion, raw-string delimiters and chunk
sizes. C++ consumers and shader tests validate meaning; a historical generated
header digest is not a type or arithmetic contract.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from build_tools.source.files import absolute, directory, regular_file


def input_files(source: Path, *, generator: Path | None = None) -> tuple[Path, ...]:
    """List the actual flat template/generator inputs, including new shaders."""
    source = directory(source)
    generator = regular_file(source / 'embed_wgsl.py' if generator is None else generator)
    if generator.name != 'embed_wgsl.py':
        raise ValueError('Expected the upstream embed_wgsl.py generator')
    paths = sorted({*source.glob('*.wgsl'), *source.glob('*.tmpl'), generator})
    if not any(path.suffix == '.wgsl' for path in paths):
        raise ValueError('No canonical WGSL sources')
    return tuple(regular_file(path) for path in paths)


def generate(source: Path, patched_template: Path | None, output: Path, *, generator: Path | None = None) -> None:
    """Embed private source inputs with the unchanged upstream generator.

    Dedicated stages already patch their private trees and need no replacement
    template. Shared-template stages can substitute one separately patched file.
    """
    source = directory(source)
    template = None if patched_template is None else regular_file(patched_template)
    generator = regular_file(source / 'embed_wgsl.py' if generator is None else generator)
    inputs = input_files(source, generator=generator)
    if template is not None and (template.parent == source or source in template.parents):
        raise ValueError('The patched template must be outside the upstream shader directory')
    if template is not None and (source / template.name not in inputs or template.suffix != '.tmpl'):
        raise ValueError('Patched template is not a declared shader template')
    output = absolute(output)
    directory(output.parent, create=True)
    if (output.is_relative_to(source) or output.is_relative_to(generator.parent)
            or output == template or output.is_symlink()):
        raise ValueError('The generated header must not replace an input or a link')
    # Keep generator failures from overwriting an earlier complete header. The
    # enclosing source composer owns the transaction for all generated files.
    with tempfile.TemporaryDirectory(prefix='wgsl-embed-', dir=output.parent) as temporary:
        private = Path(temporary) / 'inputs'
        private.mkdir()
        for original in inputs:
            shutil.copyfile(template if template is not None and original.name == template.name else original,
                            private / original.name)
        header = Path(temporary) / 'shaders.hpp'
        try:
            # Preserve both streams through every caller so generator failures
            # retain their precise input/error diagnostics.
            # The pinned embedder is a standard-library-only program. Avoid
            # user site hooks and unrelated Python packages influencing generation.
            subprocess.run([sys.executable, '-I', '-S', str(private / 'embed_wgsl.py'),
                            '--input_dir', str(private), '--output_file', str(header)], check=True)
        except subprocess.CalledProcessError as error:
            error.add_note(f'WGSL inputs: {source}; patched template: {template}; output: {output}')
            raise
        if not header.exists() or not regular_file(header).stat().st_size:
            raise ValueError('Canonical shader generator produced no header')
        header.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--template', type=Path,
                        help='Private patched template; omit when inputs are already prepared')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--generator', type=Path,
                        help='Unchanged upstream generator for a dedicated shader directory')
    args = parser.parse_args()
    generate(args.source, args.template, args.output, generator=args.generator)


if __name__ == '__main__':
    main()
