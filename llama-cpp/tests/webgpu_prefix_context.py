"""Explicitly isolate historical prefix tests from appended matvec stages."""
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def without_final_matvec_stages(source, **row_split_options):
    # Keep these independent patches in a helper to avoid Python's static
    # nesting limit in older, intentionally long composition-prefix fixtures.
    with patch.object(source.f32_matvec_epilogue, 'prepare', return_value={}):
        with patch.object(source, 'replace_iq3s_row_split_embedding', **row_split_options) as row_split:
            yield row_split
