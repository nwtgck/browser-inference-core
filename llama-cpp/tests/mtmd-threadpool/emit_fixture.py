"""Extract the actual prepared helper for controlled registry tests."""
from pathlib import Path
import hashlib


def materialize(prepared: Path, output: Path) -> dict:
    source = prepared.read_text()
    marker = 'static std::shared_ptr<ggml_threadpool> mtmd_threadpool_init(int n_threads) {'
    if source.count(marker) != 1:
        raise ValueError('mtmd threadpool helper needs source review')
    start = source.index(marker)
    end = source.index('\n}\n', start) + 3
    helper = source[start:end]
    if helper.count('n_threads = 1;') != 1 or '#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)' not in helper:
        raise ValueError('mtmd non-pthread guard needs source review')
    template = Path(__file__).with_name('main.cpp.in').read_text()
    if template.count('@ACTUAL_HELPER@') != 1:
        raise ValueError('Ambiguous helper insertion')
    output.write_text(template.replace('@ACTUAL_HELPER@', helper))
    return {'sourceSha256': hashlib.sha256(prepared.read_bytes()).hexdigest(),
            'helperSha256': hashlib.sha256(helper.encode()).hexdigest(),
            'outputSha256': hashlib.sha256(output.read_bytes()).hexdigest()}
