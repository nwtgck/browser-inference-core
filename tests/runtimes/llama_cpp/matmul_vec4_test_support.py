"""Keep historical factory bodies callable by explicit-layout test cases.

Current factories are extracted unchanged. Historical references have no layout
argument, so only their signatures acquire an unnamed argument; their original
selector/cache/shader bodies remain the oracle for the previous behavior.
"""
import re


def explicit_layout_argument(text: str) -> str:
    pattern = re.compile(r'((?:webgpu_pipeline get_mul_mat_(?:vec|fast|id|id_vec)_pipeline|'
                         r'ggml_webgpu_mul_mat_vec_pipeline_key make_mul_mat_vec_key)\([^\n]*?)(\) \{)')
    def signature(match):
        if 'vec4_aligned' in match[1]:
            return match[0]
        return match[1] + ', bool /* historical layout argument */' + match[2]
    changed = pattern.sub(signature, text)
    # Only the historical profile wrapper delegates to a key-builder without
    # forwarding a predicate. Its aligned Q8 recording cases pass this explicitly.
    return changed.replace('make_mul_mat_vec_key(context, src_overlap)',
                           'make_mul_mat_vec_key(context, src_overlap, true)')
