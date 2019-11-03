"""`damask_parse.utils.py`"""

from pathlib import Path
import copy

import numpy as np


def zeropad(num, largest):
    """Return a zero-padded string of a number, given the largest number.

    TODO: want to support floating-point numbers as well? Or rename function
    accordingly.

    Parameters
    ----------
    num : int
        The number to be formatted with zeros padding on the left.
    largest : int
        The number that determines the number of zeros to pad with.

    Returns
    -------
    padded : str
        The original number, `num`, formatted as a string with zeros added
        on the left.

    """

    num_digits = len('{:.0f}'.format(largest))
    padded = '{0:0{width}}'.format(num, width=num_digits)

    return padded


def get_num_header_lines(path):
    """Get the number of header lines from a file produced by DAMASK.

    Parameters
    ----------
    path : str or Path
        Path to a DAMASK-generated file that contains a header.

    Returns
    -------
    num : int
        Number of header lines in the DAMASK-generated file.

    """

    path = Path(path)
    with path.open() as handle:
        num = int(handle.read(1))

    return num


def get_header(path):
    """Get the header from a file produced by DAMASK.

    Parameters
    ----------
    path : str or Path
        Path to a DAMASK-generated file that contains a header.

    Returns
    -------
    header_lines : list
        List of lines within the file header

    """

    num_header_lns = get_num_header_lines(path)

    header_lines = []
    path = Path(path)
    with path.open() as handle:
        for ln_idx, ln in enumerate(handle):
            if ln_idx in range(1, num_header_lns + 1):
                header_lines.append(ln.rstrip())

    return header_lines


def validate_volume_element(volume_element):
    """Validate the parameters of a volume element, as used in the DAMASK
    geometry file format.

    TODO: check values of optional keys.

    Parameters
    ----------
    volume_element : dict

    Returns
    -------
    validated_ve : dict
        Copy of `volume_element` where array-like keys represented as lists
        are transformed to ndarrays.

    """

    keys = volume_element.keys()

    man_keys = ['grain_idx']
    opt_keys = ['size', 'orientations']
    array_keys = ['grain_idx', 'size', 'orientations']

    good_keys = list(set(man_keys) | set(opt_keys))
    missing_keys = list(set(man_keys) - set(keys))
    bad_keys = list(set(keys) - set(good_keys))

    # Check mandatory keys exist:
    if len(missing_keys) > 0:
        msg = ('Volume element is missing mandatory key(s): {}')
        raise ValueError(msg.format(missing_keys))

    # Check for unknown keys:
    if len(bad_keys) > 0:
        msg = ('Volume element contains unknown key(s): {}')
        raise ValueError(msg.format(bad_keys))

    # Transform array-like keys to ndarrays:
    validated_ve = {}
    for key in keys:
        val = copy.deepcopy(volume_element[key])
        if key in array_keys and not isinstance(val, np.ndarray):
            val = np.array(val)
        validated_ve.update({
            key: val
        })

    # Check mandatory key values:
    grain_idx = validated_ve['grain_idx']
    if grain_idx.ndim != 3:
        msg = ('Volume element key `grain_idx` must have dimension 3, '
               'but actually has dimension {}.')
        raise ValueError(msg.format(grain_idx.ndim))

    return validated_ve


def check_volume_elements_equal(vol_elem_a, vol_elem_b):
    """Check two volume elements are equivalent.

    Parameters
    ----------
    vol_elem_a : dict
    vol_elem_b : dict

    Returns
    -------
    is_equal : bool
        True if `vol_elem_a` is equal to `vol_elem_b`. Otherwise, False.

    """

    array_keys = ['grain_idx', 'size', 'orientations']

    vol_elem_a = validate_volume_element(vol_elem_a)
    vol_elem_b = validate_volume_element(vol_elem_b)

    # Check they have the same keys:
    if vol_elem_a.keys() != vol_elem_b.keys():
        return False

    # Compare mandatory keys:
    if not np.array_equal(vol_elem_a['grain_idx'], vol_elem_b['grain_idx']):
        return False

    # Compare optional keys:
    opt_keys = ['size', 'orientations']
    for key in opt_keys:
        if vol_elem_a.get(key) is not None:
            if key in array_keys:
                if not np.array_equal(vol_elem_a[key], vol_elem_b[key]):
                    return False
            else:
                if vol_elem_a[key] != vol_elem_b[key]:
                    return False

    return True


def format_1D_masked_array(arr, fmt='{:g}', fill_symbol='*'):
    'Also formats non-masked array.'

    arr_fmt = ''
    for idx, i in enumerate(arr):
        if idx > 0:
            arr_fmt += ' '
        if isinstance(i, np.ma.core.MaskedConstant):
            arr_fmt += '*'
        else:
            arr_fmt += fmt.format(i)
    return arr_fmt
