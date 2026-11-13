"""Private, frozen inputs for historical composer guard tests only."""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

from build_tools.source.files import relative_path

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/llama-webgpu-history-cd959bae'


@contextmanager
def historical_source():
    """Use immutable recorded inputs, never the current upstream environment."""
    manifest = json.loads((FIXTURE / 'manifest.json').read_text())
    archive = (FIXTURE / 'source.zip').read_bytes()
    if manifest['formatVersion'] != 1 or hashlib.sha256(archive).hexdigest() != manifest['archiveSha256']:
        raise ValueError('Historical source fixture archive integrity mismatch')
    with tempfile.TemporaryDirectory(prefix='historical-webgpu-source-') as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            names = source.namelist()
            if len(names) != len(set(names)) or set(names) != set(manifest['files']):
                raise ValueError('Historical source fixture inventory mismatch')
            for name in names:
                relative_path(name)
                data = source.read(name)
                actual = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                if actual != manifest['files'][name]:
                    raise ValueError('Historical source fixture content mismatch: ' + name)
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        yield root
