import pytest

from wam_inference_value.benchmarks.libero_adapter import LIBEROAdapter, LIBEROUnavailableError, is_libero_available


def test_libero_optional_import_status_is_well_formed():
    ok, reason = is_libero_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    assert reason


def test_libero_adapter_skips_when_unavailable():
    ok, _ = is_libero_available()
    if ok:
        pytest.skip("LIBERO is installed; heavy reset is covered by optional benchmark_libero_wam.py")
    with pytest.raises(LIBEROUnavailableError):
        LIBEROAdapter()
