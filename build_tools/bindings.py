"""Small shared helpers for mechanically generated native bindings."""
from __future__ import annotations
from pathlib import Path
import re


# Only a compiler-provided type spelling is accepted here, never generated source.
# An identifier such as sd_Bool_state or a field name is not a C type keyword.
_C_BOOL = re.compile(r"\b_Bool\b")


def cpp_type(c_type: str) -> str:
    """Translate the C boolean keyword, including pointer/callback type positions.

    Keep the original C spelling in the schema. This is an emitter conversion,
    not a text rewrite of function names, fields, constants or generated code.
    """
    return _C_BOOL.sub('bool', c_type)


def write_generated_file(path: Path, text: str) -> None:
    """Keep identical outputs' identity/mtime so reconfiguration does not rebuild.

    This writes one generated file, not an atomic multi-file publication. The
    caller owns and creates its output directory. Never follow a linked file.
    """
    if path.is_symlink():
        raise ValueError('Linked binding output: ' + str(path))
    data = text.encode('utf-8')
    if path.is_file() and path.read_bytes() == data:
        return
    path.write_bytes(data)
