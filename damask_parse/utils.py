"""`damask_parse.utils.py`"""

from pathlib import Path
from subprocess import run, PIPE
import copy
import re

import numpy as np

from damask_parse.rotation import rot_mat2euler, euler2rot_mat_n


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
    Number of header lines in the DAMASK-generated file.

    """

    with Path(path).open() as handle:
        return int(re.search(r'(\d+)\sheader', handle.read()).group(1))


def get_header_lines(path):
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


def parse_damask_spectral_version_info(executable='DAMASK_spectral'):
    'Parse the DAMASK version number and compiler options from `DAMASK_spectral --help`.'

    proc = run(f'{executable} --help', stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = proc.stdout.decode(), proc.stderr.decode()

    ver_str = re.search('Version: (.*)', stdout).group(1).strip()
    comp_with_str = re.search('Compiled with: (.*)', stdout).group(1).strip()
    comp_opts_str = re.search('Compiler options: (.*)', stdout).group(1).strip()

    damask_spectral_info = {
        'version': ver_str,
        'compiled_with': comp_with_str,
        'compiler_options': comp_opts_str,
        'stderr': stderr.strip(),
    }

    return damask_spectral_info


def volume_element_from_2D_microstructure(microstructure_image, depth=1,
                                          image_axes=['y', 'x']):
    """Extrude a 2D microstructure by a given depth to form a 3D volume element.

    Parameters
    ----------
    microstructure_image : dict
        Dict with the following keys:
            grains : ndarray of shape (N, M)
                2D map of grain indices.
            orientations : ndarray of shape (P, 3)
                Euler angles for each grain.
    depth : int, optional
        By how many voxels the microstructure should be extruded. By default, 1.

    Returns
    -------
    volume_element : dict
        Dict with the following keys:
            voxel_grain_idx : ndarray of shape (depth, N, M)
            grain_orientation_idx : ndarray of int
            size: tuple of length three
            orientations : ndarray of shape (P, 3)

    """

    # parse image axis directions and add extrusion direction (all +ve only)
    conv_axis = {'x': 0, 'y': 1, 'z': 2}
    image_axes = [conv_axis[axis] for axis in image_axes]
    image_axes.append(3 - sum(image_axes))

    # extrude and then switch around the axes to x, y, z order
    grain_idx = microstructure_image['grains'][:, :, np.newaxis]
    grain_idx = np.tile(grain_idx, (1, 1, depth))
    grain_idx = np.ascontiguousarray(grain_idx.transpose(image_axes))

    volume_element = {
        'size': [i/depth for i in grain_idx.shape],
        'grid': grain_idx.shape,
        'orientations': {
            'euler_angles': microstructure_image['orientations'],
            'euler_angle_labels': ['phi1', 'Phi', 'phi2'],
        },
        'voxel_grain_idx': grain_idx,
        'grain_orientation_idx': np.arange(grain_idx.max() + 1),
    }
    return volume_element


def add_volume_element_missing_texture(volume_element):
    """Add a missing texture (orientation) to a volume element that has an extra grain
    index in it's `voxel_grain_idx` array.

    Notes
    -----
    This can be used after invoking DAMASK's `geom_canvas` pre-processing command.

    """

    num_grains = len(volume_element['grain_orientation_idx'])
    max_phase_idx = np.max(volume_element['grain_phase_label_idx'])
    max_grain_idx = np.max(volume_element['voxel_grain_idx'])  # zero-indexed

    if max_grain_idx != num_grains:
        raise ValueError('All grains seem to have an associated texture.')

    volume_element['orientations']['euler_angles'] = np.vstack([
        volume_element['orientations']['euler_angles'],
        [[0.0, 0.0, 0.0]]
    ])
    volume_element['grain_orientation_idx'] = np.concatenate([
        volume_element['grain_orientation_idx'],
        [num_grains],
    ])
    volume_element['grain_phase_label_idx'] = np.concatenate([
        volume_element['grain_phase_label_idx'],
        [max_phase_idx + 1],
    ])


def align_orientations(ori, orientation_coordinate_system, model_coordinate_system):
    """Rotate euler angles to align orientation and model coordinate systems.

    Parameters
    ----------
    ori : ndarray of shape (N, 3)
        Array of row vectors representing euler angles.
    orientation_coordinate_system : dict
        This dict allows assigning orientation coordinate system directions to
        sample directions. Allowed keys are 'x', 'y' and 'z'. Example values are
        'RD', 'TD' and 'ND'.
    model_coordinate_system : dict
        This dict allows assigning model geometry coordinate system directions to
        sample directions. Allowed keys are 'x', 'y' and 'z'. Example values are
        'RD', 'TD' and 'ND'.    

    Notes
    -----
    This only supports one particular combination of orientation/model coordinate system
    at present; it needs generalising.

    """

    print(f'Original Euler angles:\n{ori}')

    for idx in range(ori.shape[0]):

        if (
            orientation_coordinate_system == {'x': 'RD', 'y': 'TD', 'z': 'ND'} and
            model_coordinate_system == {'x': 'TD', 'y': 'ND', 'z': 'RD'}
        ):
            R = euler2rot_mat_n(ori[idx], degrees=True)[0]
            rotR = euler2rot_mat_n(np.array([90, 90, 0]), degrees=True)[0]
            R_new = R @ rotR
            ang_new = np.rad2deg(rot_mat2euler(R_new))

            if ang_new[0] < 0:
                ang_new[0] += 360

            if ang_new[2] < 0:
                ang_new[2] += 360

            ori[idx, :] = ang_new

        else:
            msg = 'Combination of orientation and model coordinate systems not supported.'
            raise NotImplementedError(msg)

    print(f'New Euler angles:\n{ori}')
