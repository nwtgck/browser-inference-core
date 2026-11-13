"""Read the repository's explicit job/step blocks without a YAML dependency."""
from __future__ import annotations
import re


def jobs(text: str) -> dict[str, str]:
    body = text.split('\njobs:\n', 1)[1]
    names = list(re.finditer(r'^  ([a-z][a-z-]*):\s*$', body, re.M))
    return {match[1]: body[match.end():names[i + 1].start() if i + 1 < len(names) else len(body)]
            for i, match in enumerate(names)}


def steps(job: str) -> list[str]:
    return re.split(r'^      - ', job, flags=re.M)[1:]


def step_id(job: str, identifier: str) -> str:
    matching = [step for step in steps(job) if f'        id: {identifier}\n' in step]
    if len(matching) != 1:
        raise AssertionError(f'Expected one workflow step: {identifier}')
    return matching[0]


def step_name(job: str, name: str) -> str:
    matching = [step for step in steps(job) if step.startswith(f'name: {name}\n')]
    if len(matching) != 1:
        raise AssertionError(f'Expected one workflow step: {name}')
    return matching[0]


def shell_body(step: str) -> str:
    body = step.split('        run: |\n', 1)[1]
    return '\n'.join(line[10:] for line in body.splitlines() if line.startswith('          ')) + '\n'


def offline_script(step: str) -> str:
    """Read the exact container stdin for focused local shell tests (no isolation claim)."""
    body = shell_body(step)
    marker = "<<'BICORE_OFFLINE'\n"
    if marker not in body:
        raise AssertionError('Expected a literal offline Docker boundary')
    return body.split(marker, 1)[1].rsplit('\nBICORE_OFFLINE', 1)[0] + '\n'


def docker_standin(directory):
    """Install an explicitly local test stand-in preserving only Docker -e values."""
    from pathlib import Path
    import sys
    source = Path(__file__).with_name('docker_standin.py')
    target = directory / 'docker'
    target.write_text('#!' + sys.executable + '\n' + source.read_text())
    target.chmod(0o755)
    return target
