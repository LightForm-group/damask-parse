"""`test_geom.py`

Tests of the functionality associated with the geometry file format that is
used for spectral simulations in DAMASK.

"""

from unittest import TestCase

import numpy as np

from damask_parse.utils import (
    check_volume_elements_equal, validate_volume_element_OLD
)


class VolumeElementTestCase(TestCase):
    """Tests on volume element functions."""

    def test_validation_missing_key(self):
        """Test error raised on missing mandatory key."""

        vol_elem = {}
        with self.assertRaises(ValueError):
            validate_volume_element_OLD(vol_elem)

    def test_validation_unknown_key(self):
        """Test error raised on unknown key."""

        vol_elem = {
            'grain_idx': np.random.randint(0, 9, (2, 2, 2)),
            'bad_key': 1,
        }
        with self.assertRaises(ValueError):
            validate_volume_element_OLD(vol_elem)

    def test_validation_grain_idx_value(self):
        """Test error raised on incorrect `grain_idx` dimension."""

        vol_elem = {
            'grain_idx': np.random.randint(0, 9, (2, 2))
        }
        with self.assertRaises(ValueError):
            validate_volume_element_OLD(vol_elem)

    def test_equality_true_grain_idx(self):
        """Test `check_volume_elements_equal` returns True for volume elements
        with the same `grain_idx`."""

        grain_idx = np.random.randint(0, 9, (2, 2, 2))
        vol_elem_a = {
            'grain_idx': grain_idx,
        }
        vol_elem_b = {
            'grain_idx': grain_idx,
        }

        self.assertTrue(check_volume_elements_equal(vol_elem_a, vol_elem_b))

    def test_equality_false_grain_idx(self):
        """Test `check_volume_elements_equal` returns False for volume elements
        with the different `grain_idx`."""

        grain_idx = np.random.randint(0, 9, (2, 2, 2))
        vol_elem_a = {
            'grain_idx': grain_idx,
        }
        vol_elem_b = {
            'grain_idx': grain_idx + 1,
        }

        self.assertFalse(check_volume_elements_equal(vol_elem_a, vol_elem_b))
