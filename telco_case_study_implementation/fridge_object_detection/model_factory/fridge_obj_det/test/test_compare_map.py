"""Unit tests to compare two mAP values."""
from contextlib import nullcontext as does_not_raise

import pytest
from model_factory.fridge_obj_det.src.compare_map import compare_map


@pytest.mark.parametrize(
    "map_before,map_after,throws_error,expectation",
    [
        # exact same with tolerance 0.01
        (0.98, 0.97, True, does_not_raise()),
        (0.84, 0.83, True, does_not_raise()),
        # smaller than tolerance 0.01 (improved mAP)
        (0.98, 0.99, True, does_not_raise()),
        # smaller than tolerance 0.01
        (0.98, 0.975, True, does_not_raise()),
        # larger than tolerance 0.01
        (0.98, 0.96, True, pytest.raises(ValueError)),
        (0.98, 0.96, False, does_not_raise()),
        (0.98, 0.969999999999, True, pytest.raises(ValueError)),
        (0.97, 0.959999999999, True, pytest.raises(ValueError)),
    ],
)
def test_compare_map(map_before, map_after, throws_error, expectation):
    """Test compare_map_before_and_after_conversion."""
    with expectation:
        compare_map.compare_scores(
            map_before, map_after, tolerance=0.01, throws_error=throws_error
        )
