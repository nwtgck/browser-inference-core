"""Execute actual prepared dispatch and binding arithmetic using a recording device."""
from pathlib import Path
import subprocess

def check(upstream: Path, prepared: Path, library: Path, output: Path):
    text=(prepared/'ggml-webgpu.cpp').read_text()
    def function(marker):
        start=text.index(marker)
        return text[start:text.index('\n}',start)+2]
    here=Path(__file__).parent
    markers=['static size_t ggml_webgpu_tensor_offset(',
        'static size_t ggml_webgpu_tensor_align_offset(const ggml_tensor *',
        'static size_t ggml_webgpu_tensor_misalignment(const ggml_tensor *',
        'static size_t ggml_webgpu_tensor_misalignment(webgpu_context &',
        'static size_t ggml_webgpu_tensor_align_offset(webgpu_context &',
        'static size_t ggml_webgpu_tensor_binding_size(const ggml_tensor *',
        'static size_t ggml_webgpu_tensor_binding_size(webgpu_context &',
        'static std::optional<webgpu_encoded_op> ggml_webgpu_moe_sum(']
    output.mkdir(parents=True,exist_ok=True)
    code=output/'dispatch.cpp'
    code.write_text((here/'dispatch-prefix.cpp').read_text()+'\n'+'\n'.join(function(m) for m in markers)+'\n'+(here/'dispatch-cases.cpp').read_text())
    exe=output/'dispatch'
    subprocess.run(['g++','-std=c++17','-O1','-Wall','-Wextra','-Werror','-isystem',str(upstream/'ggml/include'),'-isystem',str(upstream/'ggml/src'),'-I'+str(prepared),str(code),str(library),'-pthread','-ldl','-lm','-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],check=True,text=True,capture_output=True)
    return result.stdout
