#!/usr/bin/env python3
"""Generate commit-bound consumer metadata after successful runtime publication."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlparse

from github_api import full_sha, repository_name
from package_runtime import RUNTIME_NAME, validate
from upstream_provenance import collect, file_identity

ROOT = Path(__file__).resolve().parents[1]


def extract_lock(lock: dict, specifier: str, commit: str, package_version: str) -> dict:
    if lock.get('lockfileVersion') != 3:
        raise ValueError('Expected npm lockfileVersion 3')
    packages = lock.get('packages', {})
    key = 'node_modules/' + RUNTIME_NAME
    if set(packages) != {'', key}:
        raise ValueError('Unexpected transitive dependency; this is not a standalone lock fragment')
    if packages[''].get('dependencies') != {RUNTIME_NAME: specifier}:
        raise ValueError('npm changed the requested dependency specifier')
    entry = packages[key]
    if entry.get('version') != package_version or not entry.get('resolved', '').endswith('#' + full_sha(commit)):
        raise ValueError('npm resolved a different runtime version or commit')
    repository = repository_name(specifier.removeprefix('github:').split('#')[0])
    resolved = urlparse(entry['resolved'])
    if (resolved.scheme not in ('git+ssh', 'git+https') or resolved.hostname != 'github.com'
            or resolved.path.removesuffix('.git').lower() != '/' + repository.lower()
            or resolved.query or resolved.password or resolved.port):
        raise ValueError('npm resolved a different repository or unsupported Git URL')
    if entry.get('hasInstallScript') or any(entry.get(key) for key in ('dependencies', 'optionalDependencies', 'peerDependencies')):
        raise ValueError('Unexpected install-time behavior in the lock entry')
    # The complete npm-produced entry is preserved. In particular, integrity is
    # never reconstructed from a GitHub archive or a local npm pack invocation.
    return {'lockfileVersion': 3, 'rootDependencies': packages['']['dependencies'], 'packageEntry': entry}


def generate_lock(repository: str, commit: str, package_version: str) -> dict:
    specifier = f'github:{repository_name(repository)}#{full_sha(commit)}'
    with tempfile.TemporaryDirectory(prefix='lcb-consumer-lock-') as temporary:
        directory = Path(temporary)
        (directory / 'package.json').write_text(json.dumps({
            'name': 'runtime-consumer-lock-fixture', 'version': '0.0.0', 'private': True,
            'dependencies': {RUNTIME_NAME: specifier},
        }) + '\n')
        env = os.environ.copy()
        # No consumer .npmrc, lifecycle scripts, lock cache, audit or fund network
        # calls. This resolves only the published, dependency-free runtime package.
        env.update({'NPM_CONFIG_CACHE': str(directory / 'cache'),
                    'NPM_CONFIG_USERCONFIG': str(directory / 'empty.npmrc'),
                    'NPM_CONFIG_GLOBALCONFIG': str(directory / 'empty-global.npmrc'),
                    'GIT_TERMINAL_PROMPT': '0'})
        (directory / 'empty.npmrc').touch()
        (directory / 'empty-global.npmrc').touch()
        command = ['npm', 'install', '--package-lock-only', '--ignore-scripts', '--no-audit', '--no-fund', '--lockfile-version=3']
        subprocess.run(command, cwd=directory, env=env, check=True, timeout=600)
        lock = json.loads((directory / 'package-lock.json').read_text())
        result = extract_lock(lock, specifier, commit, package_version)
        result.update({
            'nodeVersion': subprocess.check_output(['node', '--version'], text=True).strip(),
            'npmVersion': subprocess.check_output(['npm', '--version'], text=True).strip(),
            'specifier': specifier,
            'resolutionScope': 'Isolated direct dependency; no transitive dependencies. Consumer root metadata and unrelated lock entries are not represented.',
        })
        return result


def consumer_knowledge() -> dict:
    # Intentional policy exception: this project normally neither depends on nor
    # mentions Naidan. Consumer-specific knowledge is permitted ONLY here for
    # generated CI summaries / PR comments (and their report artifacts/tests).
    # It is not a runtime dependency, build input, manifest field, or public API.
    # The generated text describes integration facts and search hints; it is not
    # a prompt or an instruction to the assistant that consumes the report.
    base = 'src/features/llama-cpp-browser/'
    return {
        'name': 'Naidan',
        'knowledgeScope': 'Historical integration hints, not a live scan of the consumer. Files, symbols, profile selections and tests may have moved or changed.',
        'buildModel': 'Prebuilt runtime package; no llama.cpp compilation during the consumer build. Hosted Wasm is manifest-verified and gzip-compressed; selected standalone Wasm is additionally pinned and embedded.',
        'dependency': {
            'knownLocations': ['package.json', 'package-lock.json'],
            'searchHints': ['llama-cpp-browser-core', 'node_modules/llama-cpp-browser-core'],
            'valueCorrespondence': {
                'package.json dependencies["llama-cpp-browser-core"]': 'npm.specifier',
                'package-lock.json packages[""].dependencies["llama-cpp-browser-core"]': 'npm.specifier',
                'package-lock.json packages["node_modules/llama-cpp-browser-core"]': 'npm.packageEntry',
            },
        },
        'generatedCodeAdapter': {
            'knownLocations': [base + 'build-core.ts'],
            'searchHints': ['coreHashes', 'transformBrowserCore', 'Reviewed browser variant artifact commit', 'Unreviewed llama.cpp browser core'],
            'valueCorrespondence': {'coreHashes[profile]': 'browserProfiles[profile].mjs.sha256',
                                    'reviewed artifact comment': 'runtime.artifactCommit'},
            'meaning': 'These source hashes guard a reviewed exact-source adapter, not just integrity. New hashes alone do not establish adapter compatibility.',
            'boundarySearchHints': ['readAsync,readBinary', 'findWasmBinary', 'instantiateAsync', 'wasmBinary'],
        },
        'standaloneWasm': {
            'knownLocations': [base + 'build-core.ts'],
            'searchHints': ['standaloneWasm', 'standaloneProfiles', 'virtual:file-protocol-standalone/binary/llama-cpp-browser'],
            'valueCorrespondence': {'standaloneWasm[profile].sha256': 'browserProfiles[profile].wasm.sha256'},
            'historicallyEmbeddedProfiles': ['webgpu-wasm64-jspi', 'webgpu-wasm32-jspi'],
            'meaning': 'The consumer owns the embedded subset and virtual IDs. The report supplies all browser hashes without prescribing that subset.',
        },
        'conditionalCompatibility': {
            'knownLocations': [base + 'types.ts', base + 'build-runtime-assets.ts', base + 'runtime/artifacts.ts',
                               base + 'runtime/detect-profile.ts', base + 'runtime/profile-policy.ts',
                               base + 'runtime/profile-policy-standalone.ts'],
            'searchHints': ['profileSchema', 'manifestSchema', 'createLlamaCppRuntimeAssetsPlugin',
                            'virtual:llama-cpp-browser-core', 'llama-cpp-browser-core/api', 'LowLevelFunctions'],
            'artifactCorrespondence': ['browserProfiles keys', 'runtime.manifestFormatVersion', 'interfaceFiles'],
            'meaning': 'Profile sets, manifest schema, binding contracts and generated adapters can require source changes beyond a pin-only update.',
        },
        'localValidation': {
            'knownLocations': ['build/llama-cpp-browser-core.test.ts', 'build/llama-cpp-browser-runtime.test.ts',
                               'build/llama-cpp-browser-standalone.test.ts'],
            'searchHints': ['transformBrowserCore', 'createLlamaCppBrowserBuild', 'createLlamaCppRuntimeAssetsPlugin', 'standaloneWasm'],
            'dependencyState': 'An older node_modules archive is not evidence of the target runtime. Replacing this dependency is independent of the other dependencies only while the runtime package remains dependency-free.',
            'cacheLocationHint': 'node_modules/.package-lock.json may describe a previous installed package; it is not the committed consumer lockfile.',
        },
        'unrelatedHashExample': {
            'knownLocations': [base + 'runtime/detect-profile-standalone.ts'],
            'searchHints': ['brotli', 'DecompressionStream'],
            'meaning': 'The fixed Brotli capability-probe payload digest is unrelated to runtime artifact pins.',
        },
    }


def metadata(package: Path, repository: str, commit: str, lock: dict, divergences: dict, *,
             manifest_bytes: bytes | None = None) -> dict:
    repository_name(repository)
    full_sha(commit)
    # The multi-runtime caller supplies bytes already bound to its root manifest.
    # Standalone callers still read locally, once, and report that same snapshot.
    if manifest_bytes is None: manifest_bytes = (package / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    source = full_sha(manifest['sourceCommit'])
    upstream = full_sha(manifest['llamaCommit'])
    if divergences['baseCommit'] != upstream:
        raise ValueError('Overlay report is from a different upstream revision')
    files = {entry['path']: entry for entry in manifest['files']}
    if len(files) != len(manifest['files']):
        raise ValueError('Duplicate manifest entries')
    profiles = {}
    validation = {}
    for name, info in sorted(manifest['profiles'].items()):
        profiles[name] = {key: files[f'profiles/{name}/browser/core.{ext}']
                          for key, ext in [('mjs', 'mjs'), ('wasm', 'wasm'), ('types', 'd.ts')]}
        validation[name] = {variant: data['validation'] for variant, data in info['variants'].items()}
    return {
        'schemaVersion': 1,
        'runtime': {
            'repository': repository, 'package': RUNTIME_NAME, 'artifactCommit': commit,
            'sourceCommit': source, 'llamaCommit': upstream, 'manifestFormatVersion': manifest['formatVersion'],
        },
        'retrieval': {
            'artifactArchive': f'https://codeload.github.com/{repository}/tar.gz/{commit}',
            'artifactRawBase': f'https://raw.githubusercontent.com/{repository}/{commit}/',
            'sourceArchive': f'https://codeload.github.com/{repository}/tar.gz/{source}',
            'sourceRawBase': f'https://raw.githubusercontent.com/{repository}/{source}/',
            'sourceRepositoryRawBase': f'https://raw.githubusercontent.com/{repository}/{source}/',
            'upstreamRawBase': f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{upstream}/',
            'manifest': {'path': 'manifest.json', 'bytes': len(manifest_bytes),
                         'sha256': hashlib.sha256(manifest_bytes).hexdigest()},
            'identity': 'Exact extracted file bytes: manifest digest above, then manifest path/bytes/SHA-256 entries. Archive compression bytes are not the identity; rendered web text is not a byte-exact download.',
            'packageScope': 'The complete artifact tree includes both variants, types, APIs, examples and license notices. A few retrieved core files are sufficient for partial review, not a complete installed package.',
            'sourceArchiveScope': 'Source archives do not include submodule contents; the upstream commit is separate.',
        },
        'npm': lock,
        'browserProfiles': profiles,
        'interfaceFiles': [entry for name, entry in sorted(files.items())
                           if name.startswith('api/') or name.startswith('examples/runtime/') and not name.endswith('README.md')],
        'upstreamDivergences': divergences,
        'consumerIntegration': consumer_knowledge(),
        'validation': validation,
    }


def scalar(value) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False)
    if not isinstance(value, str):
        raise TypeError('YAML scalar must be JSON-compatible')
    # Keep simple strings compact; quote YAML 1.1 booleans, numbers, dates,
    # control characters and punctuation-sensitive values. No custom tags/aliases.
    if (re.fullmatch(r'[A-Za-z_][A-Za-z0-9_./:+-]*', value)
            and value.lower() not in {'true', 'false', 'null', 'yes', 'no', 'on', 'off', '~'}
            and not value.endswith(':')):
        return value
    return json.dumps(value, ensure_ascii=True)


def yaml_lines(value, indent: int = 0) -> list[str]:
    space = ' ' * indent
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            prefix = space + scalar(key) + ':'
            if isinstance(item, (dict, list)) and item:
                result += [prefix] + yaml_lines(item, indent + 2)
            else:
                result.append(prefix + ' ' + ('{}' if item == {} else '[]' if item == [] else scalar(item)))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, (dict, list)) and item:
                result += [space + '-'] + yaml_lines(item, indent + 2)
            else:
                result.append(space + '- ' + ('{}' if item == {} else '[]' if item == [] else scalar(item)))
        return result
    raise TypeError('Expected a YAML mapping or sequence')


def render_yaml(data: dict) -> str:
    comments = {
        'runtime': 'Commit identities and file digests below describe this published build.',
        'npm': 'This lock entry was generated by npm in CI; it is not a reconstructed tarball checksum.',
        'browserProfiles': 'All browser variants are represented. The consumer owns its active/embedded subset.',
        'upstreamDivergences': 'The upstream checkout alone does not show build-tree overlays or toolchain patches.',
        'consumerIntegration': 'Consumer paths and symbols are historical search hints; their locations and structure may have changed.',
        'validation': 'Recorded build checks have limited scope; mocked/synthetic tests do not certify real GPU inference.',
    }
    result = []
    for key, value in data.items():
        if key in comments:
            result.append('# ' + comments[key])
        result.extend(yaml_lines({key: value}))
    return '\n'.join(result) + '\n'


def render_markdown(data: dict, yaml: str) -> str:
    runtime = data['runtime']
    return (
        '## Runtime artifact published\n\n'
        f'**Artifact commit:** `{runtime["artifactCommit"]}`  \n'
        f'**Source commit:** `{runtime["sourceCommit"]}`  \n'
        f'**llama.cpp commit:** `{runtime["llamaCommit"]}`\n\n'
        f'```sh\nnpm install {data["npm"]["specifier"]}\n```\n\n'
        'This immutable runtime is available before the source PR is merged. '
        'Recorded validation scope is included below.\n\n'
        '<details>\n<summary>Consumer integration metadata (YAML)</summary>\n\n'
        '```yaml\n' + yaml + '```\n\n</details>\n'
    )


def render_summary_markdown(data: dict) -> str:
    runtime = data['runtime']
    return (
        '## Consumer integration details\n\n'
        f'**Source commit:** `{runtime["sourceCommit"]}`  \n'
        f'**llama.cpp commit:** `{runtime["llamaCommit"]}`\n\n'
        'Recorded validation scope and integration metadata:\n\n'
        '<details>\n<summary>Consumer integration metadata (YAML)</summary>\n\n'
        '```yaml\n' + render_yaml(data) + '```\n\n</details>\n'
    )


def write_report(output: Path, data: dict, run_id: str, attempt: str) -> str:
    yaml = render_yaml(data)
    markdown = render_markdown(data, yaml)
    # Reserve room for the notifier's status header below the PR comment limit.
    # A future large schema fails visibly rather than silently truncating YAML.
    if len(markdown.encode('utf-8')) > 55000:
        raise ValueError('Consumer report is too large for a single PR comment')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'consumer-update.yaml').write_text(yaml, encoding='utf-8')
    (output / 'consumer-update.md').write_text(markdown, encoding='utf-8')
    envelope = {'schemaVersion': 1, 'repository': data['runtime']['repository'],
                'sourceCommit': data['runtime']['sourceCommit'], 'artifactCommit': data['runtime']['artifactCommit'],
                'runId': int(run_id), 'runAttempt': int(attempt),
                'markdownSha256': file_identity(output / 'consumer-update.md')['sha256']}
    (output / 'report.json').write_text(json.dumps(envelope, indent=2) + '\n')
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/consumer-update')
    args = parser.parse_args()
    package = args.package.resolve()
    validate(package)
    manifest = json.loads((package / 'manifest.json').read_text())
    divergences = collect(ROOT, manifest)
    repository = repository_name(os.environ['GITHUB_REPOSITORY'])
    version = json.loads((package / 'package.json').read_text())['version']
    lock = generate_lock(repository, args.commit, version)
    data = metadata(package, repository, args.commit, lock, divergences)
    markdown = write_report(args.output, data, os.environ['GITHUB_RUN_ID'], os.environ.get('GITHUB_RUN_ATTEMPT', '1'))
    print(markdown)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
            output.write(markdown)


if __name__ == '__main__':
    main()
