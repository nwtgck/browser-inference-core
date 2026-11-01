"""Explicitly isolate historical prefix tests from appended matvec stages."""
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def without_shared_word_load(source):
    """Retain the exact Q4-and-earlier prefix for historical arithmetic fixtures."""
    with patch.object(source.subgroup_size_control, 'prepare', return_value={}), patch.object(source.checkpoint_readback, 'prepare', return_value={}), patch.object(source.aligned_word_load, 'prepare', return_value={}), patch.object(source, 'FINAL_OUTPUTS', source.q4_subgroup.OUTPUTS):
        yield


@contextmanager
def without_final_matvec_stages(source, **row_split_options):
    # Keep these independent patches in a helper to avoid Python's static
    # nesting limit in older, intentionally long composition-prefix fixtures.
    with patch.object(source.subgroup_size_control, 'prepare', return_value={}), patch.object(source.checkpoint_readback, 'prepare', return_value={}), patch.object(source.aligned_word_load, 'prepare', return_value={}), patch.object(source.q4_subgroup, 'prepare'), patch.object(source.quant_wide_m, 'prepare'), patch.object(source.direct_mapped_read, 'prepare'), patch.object(source.quant_half_pairs, 'prepare', return_value={}), patch.object(source.q6_short, 'prepare'), patch.object(source.f32_matvec_epilogue, 'prepare', return_value={}):
        with patch.object(source, 'replace_iq3s_row_split_embedding', **row_split_options) as row_split:
            yield row_split


@contextmanager
def without_checkpoint_readback(source):
    """Keep the shared-word stage as the exact historical prefix."""
    with patch.object(source.subgroup_size_control, 'prepare', return_value={}), patch.object(source.checkpoint_readback, 'prepare', return_value={}), patch.object(source, 'FINAL_OUTPUTS', source.aligned_word_load.OUTPUTS):
        yield


@contextmanager
def without_subgroup_size_control(source):
    """Keep the checkpoint stage as its exact historical prefix."""
    with patch.object(source.subgroup_size_control, 'prepare', return_value={}), patch.object(source, 'FINAL_OUTPUTS', source.checkpoint_readback.OUTPUTS):
        yield
