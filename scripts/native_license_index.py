"""Software-oriented notices for consumers; no release-specific schema/version gate.

Paths are relative to the runtime package. Originals stay in licenses/; only
known embedded headers get a *complete comment archive*, never keyword-selected
paragraphs. Ambiguous syntax falls back to the entire source. Consumers need no
knowledge of source-tree layout, upstream names, or fixed content hashes.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from bisect import bisect_right
import re

from package_inputs import read_regular, regular_path

INDEX = 'native-license-index.json'
EXTRACTED = 'licenses/extracted/'
# These are source-layout ownership rules, not accepted release fingerprints.
EMBEDDED = {
    'vendor/miniaudio/miniaudio.h': ('miniaudio', 'miniaudio'),
    'vendor/stb/stb_image.h': ('stb_image', 'stb_image'),
    'vendor/nlohmann/json.hpp': ('nlohmann-json', 'nlohmann/json'),
    'vendor/nlohmann/json_fwd.hpp': ('nlohmann-json', 'nlohmann/json'),
    'vendor/sheredom/subprocess.h': ('subprocess.h', 'subprocess.h'),
    'vendor/hash/sha1/sha1.c': ('sha1', 'SHA-1 by Steve Reid'),
    'json.hpp': ('nlohmann-json', 'nlohmann/json'),
    'stb_image.h': ('stb_image', 'stb_image'),
    'stb_image_resize.h': ('stb_image_resize', 'stb_image_resize'),
    'stb_image_write.h': ('stb_image_write', 'stb_image_write'),
}


def safe_path(value: object) -> str:
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or any(ord(c) < 32 for c in value)
            or any(p in ('', '.', '..') for p in value.split('/'))):
        raise ValueError(f'Unsafe native notice path: {value!r}')
    return value


def notice_bytes(directory: Path, relative: str) -> bytes:
    safe_path(relative)
    raw = read_regular(directory / relative, limit=32 * 1024 * 1024)
    if not raw.decode('utf-8').strip():
        raise ValueError('Empty native notice: ' + relative)
    return raw


def embedded_comments(source: str) -> str:
    """Keep EVERY comment (including attribution without licensing keywords).

    This is not a legal-text recognizer. Quoted strings and C++ raw strings must
    not create fake comment boundaries. When lexical interpretation is uncertain
    or legal wording occurs outside comments, preserve the entire input instead.
    Original bytes remain available in the package in all cases.
    """
    original = source
    # Legacy trigraph preprocessing can manufacture comment delimiters.
    if '??/' in original:
        return original
    numeric_token = re.compile(r"[0-9][A-Za-z0-9_.']*")
    # C/C++ splices physical lines before tokenizing comments, including inside
    # // comments and split delimiters. Map logical spans back to exact originals.
    pieces = []; positions = []; removed = []; previous = 0; total = 0
    for splice in re.finditer(r'\\\r?\n', original):
        pieces.append(original[previous:splice.start()])
        positions.append(splice.start() - total)
        total += splice.end() - splice.start(); removed.append(total)
        previous = splice.end()
    pieces.append(original[previous:]); source = ''.join(pieces)
    def physical(position: int) -> int:
        index = bisect_right(positions, position) - 1
        return position + (removed[index] if index >= 0 else 0)
    def record(start: int, end: int) -> None:
        left, right = physical(start), physical(end - 1) + 1
        if spans and not original[spans[-1][1]:left].strip():
            start = spans[-1][0]
            spans[-1] = (start, right); comments[-1] = original[start:right]
        else:
            comments.append(original[left:right]); spans.append((left, right))
    comments = []
    spans = []
    i = 0
    while i < len(source):
        start = i
        if source.startswith('/*', i):
            end = source.find('*/', i + 2)
            if end < 0:
                return original
            i = end + 2
            record(start, i)
        elif source.startswith('//', i):
            i += 2
            while i < len(source):
                end = source.find('\n', i)
                if end < 0:
                    i = len(source); break
                i = end + 1
                if not source[start:end].rstrip('\r').endswith('\\'):
                    break
            record(start, i)
        elif source.startswith('R"', i):
            match = re.match(r'R"([^\s()\\]{0,16})\(', source[i:])
            if match is None:
                return original
            terminator = ')' + match[1] + '"'
            end = source.find(terminator, i + len(match[0]))
            if end < 0:
                return original
            i = end + len(terminator)
        elif source[i].isdigit() and (i == 0 or not (source[i - 1].isalnum() or source[i - 1] == '_')):
            # Consume numeric tokens, including C++ digit separators, before
            # quotes. L'a', u'a' and u8'a' are character literals, not numbers.
            number = numeric_token.match(source, i)
            if number is None:
                return original
            i = number.end()
        elif source[i] in ('"', "'"):
            quote = source[i]; i += 1
            while i < len(source) and source[i] != quote:
                if source[i] == '\\':
                    i += 2
                elif source[i] in '\r\n':
                    return original
                else:
                    i += 1
            if i >= len(source):
                return original
            i += 1
        else:
            i += 1
    if not comments:
        return original
    for match in re.finditer(r'copyright\s*(?:\(c\)|[0-9])|SPDX-(?:FileCopyrightText|License-Identifier)|permission is hereby granted|public domain|redistribution and use', original, re.I):
        if not any(start <= match.start() < end for start, end in spans):
            return original
    return '\n\n'.join(comments)


def owner(root: str, relative: str) -> tuple[str, str]:
    """Group documents at software boundaries, preserving nested package copies."""
    if root == 'llama.cpp' and relative == 'licenses/LICENSE-jsonhpp':
        return 'nlohmann-json', 'nlohmann/json'
    # npm's nested node_modules can contain several releases of the same name.
    parts = relative.split('/')
    if 'node_modules' in parts:
        at = max(i for i, value in enumerate(parts) if value == 'node_modules') + 1
        if at >= len(parts) - 1:
            raise ValueError('Unowned node_modules notice: ' + relative)
        stop = at + (2 if parts[at].startswith('@') else 1)
        if stop >= len(parts):
            raise ValueError('Unowned scoped package notice: ' + relative)
        name = '/'.join(parts[at:stop])
        return root + '/' + '/'.join(parts[:stop]), name
    rules = {
        'llama.cpp': {
            'vendor/cpp-httplib/': ('cpp-httplib', 'cpp-httplib'),
            'vendor/hash/xxhash/': ('xxhash', 'xxHash'),
            'vendor/hash/sha256/': ('sha256', 'SHA-256 (vendor/hash/sha256)'),
            'vendor/hash/rotate-bits/': ('rotate-bits', 'rotate-bits'),
            'tools/ui/src/lib/vendors/big-integer/': ('big-integer', 'big-integer'),
            'tools/ui/src/lib/vendors/nerdamer-prime/': ('nerdamer-prime', 'nerdamer-prime'),
            'gguf-py/': ('gguf-py', 'gguf-py'),
        },
        'stable-diffusion.cpp': {
            'ggml/src/ggml-cuda/sageattention/': ('ggml/sageattention', 'SageAttention (ggml)'),
            'ggml/': ('stable-diffusion.cpp/ggml', 'ggml (stable-diffusion.cpp)'),
            'thirdparty/LICENSE.darts_clone.txt': ('darts-clone', 'Darts-clone'),
            'examples/server/frontend/': ('stable-diffusion.cpp/frontend', 'stable-diffusion.cpp server frontend'),
        },
        'ggml': {'ggml/src/ggml-cuda/sageattention/': ('ggml/sageattention', 'SageAttention (ggml)')},
        'emscripten': {
            'system/lib/libc/musl/': ('emscripten/musl', 'musl'),
            'system/lib/openmp/src/thirdparty/ittnotify/': ('emscripten/ittnotify', 'ittnotify'),
            'system/include/SDL/': ('emscripten/SDL', 'SDL'),
            'cache/sysroot/include/SDL/': ('emscripten/SDL', 'SDL'),
            'test/sounds/': ('emscripten/test-sounds', 'Emscripten test sounds'),
        },
    }
    for prefix, identity in rules.get(root, {}).items():
        if relative == prefix or relative.startswith(prefix):
            return identity
    for marker in ('system/lib/', 'third_party/', 'thirdparty/', 'vendor/'):
        if relative.startswith(marker):
            tail = relative[len(marker):].split('/')
            if len(tail) < 2:
                raise ValueError(f'Unclassified native notice: {root}/{relative}')
            return root + '/' + marker + tail[0], tail[0]
    if root == 'emdawnwebgpu_pkg' and relative.startswith(('webgpu/', 'webgpu_cpp/')):
        name = parts[0]
        return 'emdawnwebgpu_pkg/' + name, 'emdawnwebgpu_pkg / ' + name
    if len(parts) == 1 or (root == 'llama.cpp' and relative.startswith('cmake/')) or root == 'ggml':
        return root, root
    # A new unrecognized hierarchy must be reviewed in bicore, not guessed by
    # every downstream consumer. No fixed list of software exists in Naidan.
    raise ValueError(f'Unclassified native notice: {root}/{relative}')


def original_paths(directory: Path) -> list[str]:
    result = ['LICENSE']
    for path in sorted((directory / 'licenses').rglob('*')):
        if path.is_symlink():
            raise ValueError('Linked native notice: ' + str(path))
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            if not relative.startswith(EXTRACTED):
                result.append(relative)
    return result


def make_index(directory: Path, *, runtime: str, license_roots: Sequence[str] = ()) -> tuple[dict, dict[str, bytes]]:
    regular_path(directory, directory=True)
    if runtime not in ('llama-cpp', 'stable-diffusion-cpp'):
        raise ValueError('Unknown native runtime')
    components = {}; extracted = {}
    package_metadata = {}
    if (directory / 'package.json').exists():
        package_metadata = json.loads(read_regular(directory / 'package.json'))
        if not isinstance(package_metadata, dict):
            raise ValueError('Invalid native package metadata')
    for relative in original_paths(directory):
        raw = notice_bytes(directory, relative)
        notice = {'path': relative}
        if relative == 'LICENSE':
            identity = (runtime + '-browser-core', runtime + '-browser-core')
        elif relative.startswith('licenses/embedded/'):
            source = relative.removeprefix('licenses/embedded/').removesuffix('.txt')
            if source not in EMBEDDED:
                raise ValueError('Unclassified embedded native notice: ' + relative)
            identity = EMBEDDED[source]
            text = embedded_comments(raw.decode('utf-8')).encode('utf-8')
            # Do not duplicate unchanged originals, including fallback sources.
            if text != raw:
                output = EXTRACTED + source + '.txt'
                extracted[output] = text
                notice = {'path': output, 'sourcePath': relative}
        else:
            tail = relative.removeprefix('licenses/')
            if runtime == 'llama-cpp':
                position, tail = tail.split('/', 1)
                if not position.isdecimal() or str(int(position)) != position or int(position) >= len(license_roots):
                    raise ValueError('Unknown license root: ' + relative)
                root = license_roots[int(position)]
            elif tail.startswith('toolchain/'):
                _, root, tail = tail.split('/', 2)
            else:
                root, tail = tail.split('/', 1)
                root = {'stable-diffusion': 'stable-diffusion.cpp', 'ggml': 'ggml'}.get(root, root)
            identity = owner(root, tail)
        key, name = identity
        if key not in components:
            components[key] = {'id': key, 'name': name, 'version': None, 'license': None, 'notices': []}
        elif components[key]['name'] != name:
            raise ValueError('Ambiguous native component identity: ' + key)
        if relative == 'LICENSE':
            # These values describe our own package, not the vendored runtime.
            # Do not invent versions or identify other licenses from keywords.
            for field in ('version', 'license'):
                value = package_metadata.get(field)
                if isinstance(value, str) and value.strip():
                    components[key][field] = value
        components[key]['notices'].append(notice)
    return {'components': [components[key] for key in sorted(components)]}, extracted


def write_index(directory: Path, *, runtime: str, license_roots: Sequence[str] = ()) -> None:
    value, extracted = make_index(directory, runtime=runtime, license_roots=license_roots)
    for relative, raw in extracted.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        regular_path(target.parent, directory=True)
        if target.exists() or target.is_symlink():
            regular_path(target)
        target.write_bytes(raw)
    if (directory / INDEX).exists() or (directory / INDEX).is_symlink():
        regular_path(directory / INDEX)
    (directory / INDEX).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    validate_index(directory, runtime=runtime, license_roots=license_roots)


def validate_index(directory: Path, *, runtime: str, license_roots: Sequence[str] = ()) -> None:
    # Recompute from original documents. A consistent but incomplete re-hashed
    # manifest cannot silently drop a notice, reassign its owner, or alter an
    # extraction. Manifest file integrity is checked by the existing packager.
    expected, extracted = make_index(directory, runtime=runtime, license_roots=license_roots)
    actual = json.loads(read_regular(directory / INDEX))
    if actual != expected:
        raise ValueError('Native license index differs from the collected notices')
    actual_extracted = {p.relative_to(directory).as_posix() for p in (directory / EXTRACTED).rglob('*') if p.is_file()}
    if actual_extracted != set(extracted):
        raise ValueError('Native extracted notice inventory differs')
    for relative, raw in extracted.items():
        if notice_bytes(directory, relative) != raw:
            raise ValueError('Native extracted notice differs from its source: ' + relative)
