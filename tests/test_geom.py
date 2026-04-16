"""`test_geom.py`

Tests of the functionality associated with the geometry file format that is
used for spectral simulations in DAMASK.

"""

import numpy as np
import pytest

from damask_parse.utils import check_volume_elements_equal, validate_volume_element_OLD


def test_validation_missing_key():
    """Test error raised on missing mandatory key."""

    vol_elem = {}
    with pytest.raises(ValueError):
        validate_volume_element_OLD(vol_elem)


def test_validation_unknown_key():
    """Test error raised on unknown key."""

    vol_elem = {
        "grain_idx": np.random.randint(0, 9, (2, 2, 2)),
        "bad_key": 1,
    }
    with pytest.raises(ValueError):
        validate_volume_element_OLD(vol_elem)


def test_validation_grain_idx_value():
    """Test error raised on incorrect `grain_idx` dimension."""

    vol_elem = {"grain_idx": np.random.randint(0, 9, (2, 2))}
    with pytest.raises(ValueError):
        validate_volume_element_OLD(vol_elem)
