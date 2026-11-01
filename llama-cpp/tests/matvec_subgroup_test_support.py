"""Extract appended Q4 helpers while retaining historical prefix fixtures."""


def q4_subgroup_helpers(text: str) -> str:
    markers = ('inline bool ggml_webgpu_q4_matvec_subgroup_eligible(',
               'inline uint32_t ggml_webgpu_q4_matvec_subgroup_size(')
    present = [marker in text for marker in markers]
    if not any(present):
        if 'ggml_webgpu_q4_matvec_subgroup_size(context,' in text:
            raise ValueError('Q4 subgroup factory is missing its helper definitions')
        return ''
    if not all(present):
        raise ValueError('Incomplete Q4 subgroup helper contract')
    result = []
    for marker in markers:
        start = text.index(marker)
        brace = text.index('{', start)
        depth = 0
        for end in range(brace, len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                result.append(text[start:end + 1])
                break
        else:
            raise ValueError('Unclosed Q4 subgroup helper')
    return '\n'.join(result)
