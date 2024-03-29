"""`damask_parse.utils.py`"""

from contextlib import contextmanager
import os
from pathlib import Path
from subprocess import run, PIPE
import copy
import re
import sys

import numpy as np
import h5py
import ruamel.yaml

from damask_parse.rotation import rot_mat2euler, euler2rot_mat_n
from damask_parse.quats import euler2quat, axang2quat, multiply_quaternions


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


def validate_volume_element_OLD(volume_element):
    """Validate the parameters of a volume element, as used in the DAMASK
    geometry file format.

    TODO: re-implement

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

    man_keys = [
        'voxel_homogenization_idx',
        'orientations',
        'grain_phase_label_idx',
        'grain_orientation_idx',
        'phase_labels',
        'grid_size',
    ]
    opt_keys = [
        'voxel_grain_idx',
        'constituent_voxel_idx',
        'grain_constituent_idx',
    ]
    array_keys = [
        'voxel_homogenization_idx',
        'voxel_grain_idx',
        'constituent_voxel_idx',
        'grain_constituent_idx',
        'grain_phase_label_idx',
        'grain_orientation_idx',
    ]

    good_keys = man_keys + opt_keys
    missing_keys = list(set(man_keys) - set(keys))
    bad_keys = list(set(keys) - set(good_keys))

    # Check mandatory keys exist:
    if len(missing_keys) > 0:
        raise ValueError(f'Volume element is missing mandatory key(s): {missing_keys}.')

    # Check for unknown keys:
    if len(bad_keys) > 0:
        raise ValueError(f'Volume element contains unknown key(s): {bad_keys}.')

    vox_err = False
    const_vox_idx = None
    grain_const_idx = None
    if 'voxel_grain_idx' in volume_element:
        if (
            'constituent_voxel_idx' in volume_element or
            'grain_constituent_idx' in volume_element
        ):
            vox_err = True
        else:
            num_elems = np.product(volume_element['grid_size'])
            num_grains = len(volume_element['grains_phase_label_idx'])
            const_vox_idx = []
            grain_const_idx = np.arange(num_grains)
    else:
        if not (
            'constituent_voxel_idx' in volume_element and
            'grain_constituent_idx' in volume_element
        ):
            vox_err = True

    if vox_err:
        msg = (f'Specify either `voxel_grain_idx` or both `constituent_voxel_idx` '
               f'and `grain_constituent_idx`.')
        raise ValueError(msg)

    # Transform array-like keys to ndarrays if not None:
    validated_ve = {}
    for key in keys:
        val = copy.deepcopy(volume_element[key])
        if val and key in array_keys and not isinstance(val, np.ndarray):
            val = np.array(val)
        validated_ve.update({key: val})

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

    TODO: re-implement

    """

    array_keys = ['grain_idx', 'size', 'orientations']

    vol_elem_a = validate_volume_element_OLD(vol_elem_a)
    vol_elem_b = validate_volume_element_OLD(vol_elem_b)

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


def format_1D_masked_array(arr, fill_symbol='x'):
    'Also formats non-masked array.'

    return [x.item() if not isinstance(x, np.ma.core.MaskedConstant)
            else fill_symbol for x in arr]

def format_2D_masked_array(arr, fill_symbol="x"):
    out = []
    for x in arr:
        sub = []
        for i in x:
            val = (
                i.item() if not isinstance(i, np.ma.core.MaskedConstant) else fill_symbol
            )
            sub.append(val)
        out.append(sub)
    return out

def masked_array_from_list(arr, fill_value='x'):
    """Generate a (masked) array from a 1D list whose elements may contain a fill value."""

    data = np.empty(len(arr))
    mask = np.zeros(len(arr))
    has_mask = False
    for idx, i in enumerate(arr):
        if i == fill_value:
            mask[idx] = True
            has_mask = True
        else:
            data[idx] = i
    if has_mask:
        return np.ma.masked_array(data, mask=mask)
    else:
        return data


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


def volume_element_from_2D_microstructure(microstructure_image, homog_label,
                                          phase_label=None, 
                                          phase_label_mapping=None,
                                          depth=1, image_axes=['y', 'x']):
    """Extrude a 2D microstructure by a given depth to form a 3D volume element.

    Parameters
    ----------
    microstructure_image : dict
        Dict with the following keys:
            grains : ndarray or nested list of shape (N, M)
                2D map of grain indices.
            orientations : dict
                Dict with the following keys:
                    type: str, "quat"
                    quaternions : ndarray of shape (P, 4) of float
                        Array of P row four-vectors of unit quaternions.
                    unit_cell_alignment : dict
                        Alignment of the unit cell.
            scale : float, optional
            grain_phases: list of int
                Index of phase assigned to each grain
            phase_labels: list of str
                Label of each phase
    homog_label : str
        Homogenization scheme label.
    phase_label : str, optional
        Label of the phase for single phase.
    phase_label_mapping: dict, optional
        Mapping from phase labels in the `microstructure_image` to phase
        labels for created in the VE.
    depth : int, optional
        By how many voxels the microstructure should be extruded. By default, 1.
    image_axes : list, optional
        Directions along the ndarray axes. Possible values ('x', 'y', 'z')

    Returns
    -------
    volume_element : dict
        Dict representation of the volume element, as returned by
        `validate_volume_element`.

    """
    if phase_label is not None and phase_label_mapping is not None:
        msg = ('Specify either a single `phase_label` or a '
               '`phase_label_mapping`, not both.')
        raise ValueError(msg)
    if (phase_label is None and
            ('grain_phases' not in microstructure_image or
             'phase_labels' not in microstructure_image)):
        msg = ('`phase_label` must be specified for a `microstructure_image` '
               'without `grain_phases` and `grain_phases`.')
        raise ValueError(msg)

    # parse image axis directions and add extrusion direction (all +ve only)
    conv_axis = {'x': 0, 'y': 1, 'z': 2}
    image_axes = [conv_axis[axis] for axis in image_axes]
    image_axes.append(3 - sum(image_axes))

    # extrude and then switch around the axes to x, y, z order
    grain_idx = np.array(microstructure_image['grains'])[:, :, np.newaxis]
    grain_idx = np.tile(grain_idx, (1, 1, depth))
    grain_idx = np.ascontiguousarray(grain_idx.transpose(image_axes))

    volume_element = {
        'size': tuple(i/depth for i in grain_idx.shape),
        'grid_size': grain_idx.shape,
        'orientations': microstructure_image['orientations'],
        'element_material_idx': grain_idx,
    }
    # single phase VE
    if phase_label is not None:
        if ('phase_labels' in microstructure_image and
                len(microstructure_image['phase_labels']) > 1):
            print('Warning: assigning a single phase to an microstructure '
                  'with more than one phase.')
        volume_element.update({
            'phase_labels': [phase_label],
            'homog_label': homog_label,
        })

    # multi phase VE
    else:
        phase_labels = microstructure_image['phase_labels']
        if phase_label_mapping is not None:
            phase_labels = [phase_label_mapping[lab] for lab in phase_labels]
        grain_phase_labels = np.array(
            [phase_labels[i] for i in microstructure_image['grain_phases']])

        num_grains = len(microstructure_image['grain_phases'])
        volume_element.update({
            'constituent_material_idx': np.arange(num_grains),
            'constituent_material_fraction': np.ones(num_grains),
            'constituent_phase_label': grain_phase_labels,
            'constituent_orientation_idx': np.arange(num_grains),
            'material_homog': np.full(num_grains, homog_label),
        })

    volume_element = validate_volume_element(volume_element)

    return volume_element


def add_volume_element_buffer_zones(volume_element, buffer_sizes, phase_ids, phase_labels,
                                    homog_label, order=['x', 'y', 'z']):
    """Add buffer material regions to a volume element.

    Parameters
    ----------
    volume_element : dict
        Dict representing the volume element that can be validated via
        `validate_volume_element`.
    buffer_sizes : list of int, length 6
        Size of buffer on each face [-x, +x, -y, +y, -z, +z]
    phase_ids : list of int, length 6
        Phase of each buffer. Relative, so 1 is the first new phase and so on
    phase_labels : list of str
        Labels of the new phases
    homog_label: str
        Homogenization scheme label.
    order : list of str, optional
        Order to add the zones in, default [x, y, z]

    Returns
    -------
    volume_element : dict
        Dict representing modified volume element.

    """

    volume_element = validate_volume_element(volume_element)

    conv_order = {'x': 0, 'y': 1, 'z': 2}
    order = [conv_order[axis] for axis in order]

    # new grid dimensions
    grid = volume_element['grid_size']  # tuple
    delta_grid = tuple(buffer_sizes[2*i] + buffer_sizes[2*i + 1] for i in range(3))
    new_grid = tuple(a + b for a, b in zip(grid, delta_grid))

    if 'size' in volume_element:
        # scale size based on material added
        new_size = tuple(s / og * ng for og, ng, s in zip(
            grid, new_grid, volume_element['size'])
        )

    # validate new phases
    phase_ids_unq = sorted(set(pid for pid, bs in zip(phase_ids, buffer_sizes) if bs > 0))
    if phase_ids_unq[0] != 1 and phase_ids_unq[-1] != len(phase_ids_unq):
        raise ValueError("Issue with buffer phases.")
    if len(phase_labels) != len(phase_ids_unq):
        raise ValueError("Issue with buffer phase labels.")

    # new phase and grain ids to add. 1 grain per phase
    material_idx = volume_element['element_material_idx']
    next_material_id = material_idx.max() + 1
    next_ori_idx = volume_element['orientations']['quaternions'].shape[0]

    # add a single-constituent material for each buffer phase:
    new_material_ids = []
    for i in range(len(phase_ids_unq)):
        volume_element['material_homog'] = np.append(
            volume_element['material_homog'],
            homog_label,
        )
        volume_element['constituent_phase_label'] = np.append(
            volume_element['constituent_phase_label'],
            phase_labels[i],
        )
        volume_element['constituent_material_fraction'] = np.append(
            volume_element['constituent_material_fraction'],
            1.0,
        )
        volume_element['constituent_material_idx'] = np.append(
            volume_element['constituent_material_idx'],
            next_material_id,
        )
        volume_element['constituent_orientation_idx'] = np.append(
            volume_element['constituent_orientation_idx'],
            next_ori_idx,
        )
        new_material_ids.append(next_material_id)

        next_material_id += 1
        next_ori_idx += 1

    # add a new orientation (identity) for each new material:
    identity_oris = np.zeros((len(phase_ids_unq), 4))
    identity_oris[:, 0] = 1
    volume_element['orientations']['quaternions'] = np.vstack([
        volume_element['orientations']['quaternions'],
        identity_oris,
    ])

    # add the buffer regions
    for axis in order:
        if delta_grid[axis] == 0:
            continue

        new_blocks = []
        for i in range(2):
            buffer_size = buffer_sizes[2*axis+i]
            if buffer_size <= 0:
                continue

            material_id = new_material_ids[phase_ids[2*axis+i] - 1]

            buffer_shape = list(material_idx.shape)
            buffer_shape[axis] = buffer_size

            new_block = np.full(buffer_shape, material_id)
            new_blocks.append(new_block)

            if i == 0:
                new_blocks.append(material_idx)

        material_idx = np.concatenate(new_blocks, axis=axis)

    volume_element.update(
        grid_size=new_grid,
        element_material_idx=material_idx,
    )

    if 'size' in volume_element:
        volume_element.update(size=new_size)

    return volume_element


def align_orientations(ori, orientation_coordinate_system, model_coordinate_system):
    """Rotate euler angles to align orientation and model coordinate systems.

    Parameters
    ----------
    ori : ndarray of shape (N, 3)
        Array of row vectors representing Euler angles in degrees.
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


def get_HDF5_incremental_quantity(hdf5_path, dat_path, transforms=None, increments=1):
    """Accessing HDF5 file directly, extract data defined at each increment.

    Parameters
    ----------
    hdf5_path : Path or str
        Path to the HDF5 file generated by DAMASK
    dat_path : str
        Forward slash delimeted str path within the DAMASK HDF5 file of the incremental
        data to extract. This path must exist within each `incrementXXXXX` group in the
        file. Example: "constituent/1_Al/generic/epsilon_V^0(Fp)_vM".
    transforms : list of dict, optional
        List of manipulations to perform on the data. Each dict may have one or more of
        the following keys:
            mean_along_axes : int or list of int, optional
                This uses `numpy.mean` on the data. Note that the zeroth axis is the
                increment axis!
            sum_along_axes : int or list of int, optional
                This uses `numpy.sum` on the data. Note that the zeroth axis is the
                increment axis!
    increments : int, optional
        Increment step size. By default, 1, in which case data for every increment will
        be extracted.

    """
    with h5py.File(str(hdf5_path), 'r') as f:
        incs = [i for i in f.keys() if 'inc' in i]
        incs = sorted(incs, key=lambda i: int(re.search(r'\d+', i).group()))
        incs = incs[::increments]

        all_data = []
        for inc in incs:
            data = f[inc][dat_path][()]
            data = apply_transforms(data, transforms, True)
            all_data.append(data)

        all_data = np.array(all_data)
        # Apply any transforms on the increment axis
        all_data = apply_transforms(all_data, transforms, False)

        if dat_path.split('/')[-1] == 'O':
            all_data = process_damask_orientatons(all_data)

        return all_data


def normalise_inc(inc, incs_in_file, default=0):
    """Convert a negative increment by counting backwards from the final
    increment."""
    if inc < 0:
        # Count backwards from the final incremnt:
        final_inc = incs_in_file[-1]
        inc = final_inc + inc + 1

        if inc < 0:
            print(f'Negative increment {inc} is out of range of the available '
                  f'increments. Using increment {default} instead.')
            inc = default
    return inc


def parse_inc_specs(inc_specs, incs_in_file):

    # nothing specified, default to all
    if not inc_specs:
        return incs_in_file

    incs = set()
    for inc_spec in inc_specs:
        if any(k in inc_spec for k in ('start', 'stop', 'step')):
            if 'values' in inc_spec:
                print("Only 'values' or a range ('start', 'stop', 'step') "
                      "should be given in an increment specification, not "
                      "both. Interpreting as a range.")

            start = normalise_inc(
                inc_spec.get('start', 0),
                incs_in_file,
                default=0,
            )
            stop = normalise_inc(
                inc_spec.get('stop', incs_in_file[-1]),
                incs_in_file,
                default=incs_in_file[-1],
            )
            step = inc_spec.get('step', 1)
            if step < 0:
                step = 1
                print("Cannot have negative increment step size, using unit step size.")
            new_incs = range(start, stop + 1, step)

        elif 'values' in inc_spec:
            new_incs = [normalise_inc(i, incs_in_file) for i in inc_spec['values']]

        else:
            print("Unknown increment specification.")
            continue

        incs = incs.union(set(new_incs))

    incs_in_file = incs.intersection(set(incs_in_file))
    incs_missing = incs.difference(incs_in_file)
    if incs_missing:
        print("These requested increments were not found: ",
              sorted(list(incs_missing)))

    return sorted(list(incs_in_file))


def parse_inc_specs_using_result_obj(inc_specs, sim_data):
    """Parse increment spec using the Result object for validation."""

    inc_prefix = 'increment_'
    incs_in_file = [int(inc[len(inc_prefix):]) for inc in sim_data.increments]

    return parse_inc_specs(inc_specs, incs_in_file)


def apply_transforms(data, transforms, single_inc):
    if data.size == 0:
        return data

    for transform in transforms or []:
        for op, axis in transform.items():
            if axis == 0 and not single_inc:
                shift = 0
            elif axis > 0 and single_inc:
                shift = -1
            else:
                continue

            if axis + shift >= data.ndim:
                print(f"Could not apply '{op}' on axis {axis}.")

            if op == 'mean_along_axes':
                data = np.mean(data, axis + shift)
            elif op == 'sum_along_axes':
                data = np.sum(data, axis + shift)

    return data


def process_damask_orientatons(ori_data):
    # cast to orientation dict
    return {
        'type': 'quat',
        'quaternions': ori_data,
        'unit_cell_alignment': {'x': 'a'},
        'quat_component_ordering': 'scalar-vector',
        'P': -1,
    }


def increment_generator(increments, sim_data):
    inc_prefix = 'increment_'

    for inc in parse_inc_specs_using_result_obj(increments, sim_data):
        try:
            # known: v3 alpha 3
            sim_data = sim_data.view('increments', f"{inc_prefix}{inc}")
        except TypeError:
            # known: v3 alpha 7
            sim_data = sim_data.view(increments=f"{inc_prefix}{inc}")

        yield inc, sim_data


def reshape_field_data(field_data, new_shape):
    """Reshape data array to VE dimensions

    Parameters
    ----------
    field_data : np.ndarray
        Data array to reshape, shape (N, ...).
    new_shape : tuple
        Shape of output, must be compatible with N.

    """
    # # reshape to make x,y,z contiguous in memory (numpy row major)
    # # dimensions: 0,1: tensor components, 2: x-pos, 3: y-pos, 4: z-pos
    # old_shape = field_data.shape
    # if len(old_shape) > 1 and old_shape[1:] != (1,):
    #     new_shape = old_shape[1:] + new_shape
    # return np.ascontiguousarray(field_data.T.reshape(new_shape, order='F'))
    # or reshape to make tensor components contiguous in memory (numpy row major)
    # dimensions: 0: x-pos, 1: y-pos, 2: z-pos 3,4: tensor components
    old_shape = field_data.shape
    if len(old_shape) > 1 and old_shape[1:] != (1,):
        new_shape += old_shape[1:]
    return np.ascontiguousarray(field_data.reshape(new_shape, order='F'))
    # this seems more inline with what is already done with `incremental_data`
    # dims (incs, spatial, tensor comps)


def get_vol_data(sim_data, field_name, increments, transforms=None):
    from damask.util import dict_flatten

    vol_data = []
    incs_valid = []
    first_inc = True
    phase_names = []
    for inc, sim_data in increment_generator(increments, sim_data):
        data = sim_data.get(output=field_name, flatten=(not first_inc))

        if data is None:
            print(f"Could not find field '{field_name}' for increment "
                  f"{inc} in output data.")
            continue

        # get names of phases in the output
        if first_inc:
            data = next(iter(data.values()))  # get only inc
            try:
                phase_names = list(data['phase'].keys())
            except KeyError:
                pass
            data = dict_flatten(data)
            first_inc = False

        if isinstance(data, dict):
            data = np.concatenate([dat for dat in data.values()], axis=0)
        elif not isinstance(data, np.ndarray):
            continue   # something isn't right, move to next

        data = apply_transforms(data, transforms, True)

        incs_valid.append(inc)
        vol_data.append(data)

    vol_data = np.array(vol_data)
    vol_data = apply_transforms(vol_data, transforms, False)

    if field_name == 'O':
        vol_data = process_damask_orientatons(vol_data)

    return vol_data, incs_valid, phase_names


def get_phase_data(sim_data, field_name, phase_name, increments,
                   transforms=None):
    phase_data = []
    incs_valid = []
    for inc, sim_data in increment_generator(increments, sim_data):
        try:
            # known: v3 alpha 3
            data = sim_data.view('phases', phase_name).get(output=field_name)
        except TypeError:
            # known: v3 alpha 7
            data = sim_data.view(phases=phase_name).get(output=field_name)

        if data is None:
            print(f"Could not find field '{field_name}' for phase "
                  f"'{phase_name}' and increment {inc} in output data.")
            continue

        data = apply_transforms(data, transforms, True)

        incs_valid.append(inc)
        phase_data.append(data)

    phase_data = np.array(phase_data)
    phase_data = apply_transforms(phase_data, transforms, False)

    if field_name == 'O':
        phase_data = process_damask_orientatons(phase_data)

    return phase_data, incs_valid


def get_field_data(sim_data, field_name, increments):
    """Access data from DAMASK result object and place on the simulation grid.

    Parameters
    ----------
    sim_data : damask.Result
        DAMASK simulation result object.
    field_name : str
        Name of data to process.
    increments: list of dict
        List of increment specifications to extract data from. Values
        refer to increments in the simulation. Default to all. This is a
        list of dict one of the following sets of keys:
            values: list of int
                List of incremnts to extract
            ----OR----
            start: int
                First increment to extract
            stop: int
                Final incremnt to extract (inclusive)
            step: int
                Step between increments to extract

    """
    nodal_fields = ['u_n']

    cells = tuple(sim_data.cells)
    if field_name in nodal_fields:
        cells = tuple(i + 1 for i in cells)

    field_data = []
    incs_valid = []
    for inc, sim_data in increment_generator(increments, sim_data):
        data = sim_data.place(output=field_name, constituents=0)

        if data is None:
            print(f"Could not find field '{field_name}' for increment {inc} "
                  f"in output data.")
            continue

        data = data.data
        data = reshape_field_data(data, cells)

        incs_valid.append(inc)
        field_data.append(data)

    # convert list of array
    # index order (incs, spatial_comps, tensor_comps)
    field_data = np.array(field_data)

    if field_name == 'O':
        field_data = process_damask_orientatons(field_data)

    return field_data, incs_valid


def apply_grain_average(field_data, grains, is_oris=False):
    # grain_data
    grain_data = []
    if is_oris:
        if field_data['type'] != 'quat':
            raise ValueError('Only quaternion orientations can be averaged.')
        meta_data = {k: v for k, v in field_data.items() if k != 'quaternions'}
        field_data = field_data['quaternions']

        for grain in np.unique(grains):
            grain_data.append(field_data[:, grains == grain].sum(axis=1))
        grain_data = np.array(grain_data).swapaxes(0, 1)
        grain_data /= np.linalg.norm(grain_data, axis=2)[..., np.newaxis]

        grain_data = {
            'quaternions': grain_data
        }
        grain_data.update(meta_data)
    else:
        for grain in np.unique(grains):
            grain_data.append(field_data[:, grains == grain].mean(axis=1))
        grain_data = np.array(grain_data).swapaxes(0, 1)

    # index order (incs, grains, tensor_comps)
    return grain_data


def validate_orientations(orientations):
    """Check a set of orientations are valid.

    Parameters
    ----------
    orientations : dict
        Dict containing the following keys:
            type : str
                One of "euler", "quat".
            quaternions : (list or ndarray of shape (R, 4)) of float, optional
                Array of R row four-vectors of unit quaternions. A single quaternion in an
                array of shape (4,) is also allowed. Specify either `quaternions` or
                `euler_angles`.
            P : int, optional
                The "P" constant, either +1 or -1, as defined within [1]. If not
                specified, P = +1 will be used.
            euler_angles : (list or ndarray of shape (R, 3)) of float, optional           
                Array of R row three-vectors of Euler angles in degrees or radians,
                as determined by `euler_degrees`. A single Euler angle triplet in an
                array of shape (3,) is also allowed. Specify either `quaternions` or
                `euler_angles`. Specified as proper Euler angles in the Bunge
                convention. (Rotations are about Z, new X, new new Z.)
            euler_degrees : bool, optional
                If True, `euler_angles` are expected in degrees, rather than
                radians.
            unit_cell_alignment : dict
                Alignment of the unit cell.
            quat_component_ordering: str ("scalar-vector" or "vector-scalar")
            orientation_coordinate_system : dict, optional
                Mapping between Cartesian directions and sample coordinate system labels.

    Returns
    -------
    orientations_valid : dict
        Validated orientations where, if orientations were specified as Euler angles in
        `orientations`, they have been converted to quaternions. Dict with the following
        key/values:
            type : str
                Value is "quat".
            quaternions : ndarray of shape (R, 4) of float
                Orientations represented as an array of row 4-vectors.
            unit_cell_alignment : dict
                Alignment of the unit cell.                
            P : int
                The "P" constant, either +1 or -1, as defined within [1]. If included in
                the original `orientations` dict, this value will be unchanged. Otherwise,
                this will be set to +1.
            quat_component_ordering: str ("scalar-vector" or "vector-scalar")
                Value will be set to "scalar-vector".
            orientation_coordinate_system : dict, optional
                Mapping between Cartesian directions and sample coordinate system labels.

    References
    ----------
    [1] Rowenhorst, D, A D Rollett, G S Rohrer, M Groeber, M Jackson,
        P J Konijnenberg, and M De Graef. "Consistent Representations
        of and Conversions between 3D Rotations". Modelling and Simulation
        in Materials Science and Engineering 23, no. 8 (1 December 2015):
        083501. https://doi.org/10.1088/0965-0393/23/8/083501.       

    """

    ori_type = orientations.get('type')
    eulers = orientations.get('euler_angles')
    euler_is_degs = orientations.get('euler_degrees')
    quats = orientations.get('quaternions')
    quats_comp_order = orientations.get('quat_component_ordering')
    alignment = orientations.get('unit_cell_alignment')
    OCS = orientations.get('orientation_coordinate_system')

    ALLOWED_QUAT_ORDER = ['scalar-vector', 'vector-scalar']

    P = orientations.get('P', 1)
    if P not in [-1, 1]:
        raise ValueError('If specified, `P` should be +1 or -1.')

    if not alignment:
        msg = (f'Alignment of the unit cell must be specified as a dict '
               f'`unit_cell_alignment`.')
        raise ValueError(msg)

    if ori_type not in ['euler', 'quat']:
        msg = f'Specify orientation `type` as either "euler" or "quat".'
        raise ValueError(msg)

    elif ori_type == 'euler':
        if eulers is None:
            msg = (f'Specify orientations as an array of row three-vector Euler angles '
                   f'with the key "euler_angles".')
            raise ValueError(msg)
        if euler_is_degs is None:
            msg = (f'If orientations are specified as Euler angles, "euler_degrees" must '
                   f'be specified as True or False to indicate the format of the Euler '
                   f'angles.')
            raise ValueError(msg)

        euler_angles = np.array(eulers)

        if euler_angles.shape == (3,):
            euler_angles = euler_angles.reshape((1, 3))

        elif euler_angles.ndim != 2 or euler_angles.shape[1] != 3:
            msg = (f'Euler angles specified in "euler_angles" should be a nested list or '
                   f'array of shape (R, 3), but shape passed was: {euler_angles.shape}.')
            raise ValueError(msg)

        # Convert Euler angles to quaternions:
        quaternions = euler2quat(euler_angles, degrees=euler_is_degs, P=P)
        quats_comp_order = 'scalar-vector'

    elif ori_type == 'quat':
        if quats is None:
            msg = (f'Specify orientations as an array of row four-vector unit '
                   f'quaternions with the key "quaternions".')
            raise ValueError(msg)

        quaternions = np.array(quats)

        if quaternions.shape == (4,):
            quaternions = quaternions.reshape((1, 4))

        elif quaternions.ndim != 2 or quaternions.shape[1] != 4:
            msg = (f'Quaternions specified in "quaternions" should be a nested list or '
                   f'array of shape (R, 4), but shape passed was: {quaternions.shape}.')
            raise ValueError(msg)

        if quats_comp_order == 'vector-scalar':
            # "normalise" to scalar-vector convention:
            quaternions = np.roll(quaternions, 1, axis=1)
            quats_comp_order = 'scalar-vector'

    if quats_comp_order not in ALLOWED_QUAT_ORDER:
        msg = (f'Quaternion component order key `quat_component_ordering` must be '
               f'specified as either "scalar-vector" or "vector-scalar". Actual '
               f'value was: "{quats_comp_order}".')
        raise ValueError(msg)

    # To ensure maximum precision of quaternions, cast to longdouble (although note that
    # precision is system-dependent):
    quaternions = quaternions.astype(np.longdouble)
    res = np.finfo(np.longdouble).resolution
    norm_factor = np.sqrt(np.sum(quaternions ** 2, axis=1))
    is_normed = np.isclose(norm_factor - 1, 0, atol=res)
    if not np.all(is_normed):
        to_norm = np.logical_not(is_normed)
        print(f'Some ({to_norm.sum()}/{to_norm.size}) quaternions are not normalised to '
              f'within `np.longdouble` resolution ({res}); they will be normalised.')
        quaternions[to_norm] = quaternions[to_norm] / norm_factor[to_norm, None]

    if not orientations.get('use_max_precision'):
        # Cast back to np.float64 (supported on "all" systems), unless requested to keep
        # highest precision:
        quaternions = quaternions.astype(np.float64)

    orientations_valid = {
        'type': 'quat',
        'quaternions': quaternions,
        'quat_component_ordering': quats_comp_order,
        'unit_cell_alignment': alignment,
        'use_max_precision': orientations.get('use_max_precision', False),
        'P': P,
    }

    if OCS:
        allowed_OCS_keys = {'x', 'y', 'z'}
        OCS_keys = set(OCS.keys())
        if OCS_keys - allowed_OCS_keys:
            msg = ('If specified, `orientation_coordinate_system` must be a dict with '
                   f'one or more of the keys: "x", "y" or "z".')
            raise ValueError(msg)
        orientations_valid.update({'orientation_coordinate_system': OCS})

    return orientations_valid


def validate_volume_element(volume_element, phases=None, homog_schemes=None):
    """

    Parameters
    ----------
    volume_element : dict
    phases : dict
    homog_schemes : dict

    Returns
    -------
    volume_element : dict
        Dict with keys:
            constituent_material_idx : ndarray of shape (N,) of int
                Determines the material to which each constituent belongs, where N is the
                number of constituents.
            constituent_material_fraction: ndarray of shape (N,) of float
                The fraction that each constituent occupies within its respective
                material, where N is the number of constituents.
            constituent_phase_label : ndarray of shape (N,) of str
                Determines the phase label of each constituent, where N is the number of
                constituents.
            constituent_orientation_idx : ndarray of shape (N,) of int
                Determines the orientation (as an index into `orientations`) associated
                with each constituent, where N is the number of constituents.
            material_homog : ndarray of shape (M,) of str
                Determines the homogenization scheme (from a list of available
                homogenization schemes defined elsewhere) to which each material belongs,
                where M is the number of materials.
            element_material_idx : ndarray of shape equal to `grid_size` of int, optional
                Determines the material to which each geometric model element belongs,
                where P is the number of elements.
            grid_size : ndarray of shape (3,) of int, optional
                Geometric model grid dimensions.
            orientations : dict, optional
                Dict containing the following keys:
                    type : str
                        Value is "quat".
                    quaternions : ndarray of shape (R, 4) of float
                        Array of R row four-vectors of unit quaternions.
                    unit_cell_alignment : dict
                        Alignment of the unit cell.
                    P : int
                        The "P" constant, either +1 or -1, as defined within [1].

    References
    ----------
    [1] Rowenhorst, D, A D Rollett, G S Rohrer, M Groeber, M Jackson,
        P J Konijnenberg, and M De Graef. "Consistent Representations
        of and Conversions between 3D Rotations". Modelling and Simulation
        in Materials Science and Engineering 23, no. 8 (1 December 2015):
        083501. https://doi.org/10.1088/0965-0393/23/8/083501.

    """

    volume_element = copy.deepcopy(volume_element)

    ignore_missing_elements = False
    ignore_missing_constituents = False

    if 'element_material_idx' not in volume_element:
        ignore_missing_elements = True
    if 'constituent_material_idx' not in volume_element:
        ignore_missing_constituents = True

    req = [
        'orientations',
        'constituent_material_idx',
        'constituent_phase_label',
        'material_homog',
        'element_material_idx',
        'grid_size',
    ]

    if ignore_missing_elements:
        if ignore_missing_constituents:
            raise ValueError(
                'Cannot ignore both missing elements and missing constituents!')
        req.remove('element_material_idx')
        req.remove('grid_size')

    elif ignore_missing_constituents:
        req.remove('constituent_material_idx')
        req.remove('constituent_phase_label')
        req.remove('material_homog')
        req.append('phase_labels')
        req.append('homog_label')

    if ignore_missing_constituents:
        allowed = list(req)
    else:
        allowed = req + [
            'constituent_material_fraction',  # default value can be set
            'constituent_orientation_idx',    # default value can be set (sometimes)
        ]

    allowed += ['size', 'origin']

    missing = set(req) - set(volume_element)
    if missing:
        missing_fmt = ', '.join([f'"{i}"' for i in missing])
        msg = f'The following volume element keys are missing: {missing_fmt}.'
        raise ValueError(msg)

    unknown = set(volume_element) - set(allowed)
    if unknown:
        unknown_fmt = ', '.join([f'"{i}"' for i in unknown])
        msg = f'The following volume element keys are unknown: {unknown_fmt}.'
        raise ValueError(msg)

    orientations = validate_orientations(volume_element['orientations'])
    volume_element['orientations'] = orientations

    if ignore_missing_constituents:
        # Assuming a full-field model (one constituent per material), set default
        # constituent keys.

        num_mats = validate_element_material_idx(volume_element['element_material_idx'])
        num_oris = orientations['quaternions'].shape[0]
        num_new_phases = num_mats - num_oris

        if num_new_phases != len(volume_element['phase_labels'][1:]):
            msg = (f'Invalid number of phase labels specified; the first phase label '
                   f'should correspond to the elements for which orientations are '
                   f'defined (of which there are {num_oris}), and the remaining phase '
                   f'labels should be used for additional elements (of which there are '
                   f'{num_mats - num_oris}).')
            raise ValueError(msg)

        const_phase_lab = np.array(
            [volume_element['phase_labels'][0]] * num_oris +
            volume_element['phase_labels'][1:]
        )
        additional_oris = np.tile(np.array([1, 0, 0, 0]), (num_new_phases, 1))
        new_oris = np.vstack([orientations['quaternions'], additional_oris])
        mat_homog = np.array([volume_element['homog_label']] * num_mats)

        volume_element['constituent_material_idx'] = np.arange(0, num_mats)
        volume_element['constituent_material_fraction'] = np.ones(num_mats)
        volume_element['constituent_orientation_idx'] = np.arange(0, num_mats)
        volume_element['constituent_phase_label'] = const_phase_lab
        volume_element['orientations']['quaternions'] = new_oris
        volume_element['material_homog'] = mat_homog

        del volume_element['phase_labels']
        del volume_element['homog_label']

    float_arrs = ['constituent_material_fraction']
    int_arrs = [
        'constituent_material_idx',
        'constituent_orientation_idx',
        'element_material_idx',
        'grid_size',
    ]
    str_arrs = [
        'constituent_phase_label',
        'material_homog',
    ]
    arr_keys = float_arrs + int_arrs + str_arrs
    num_const = None
    for key in volume_element:

        # Convert lists to arrays and check dtypes:
        if key in arr_keys:
            new_val = np.array(volume_element[key])
            if key == 'element_material_idx':
                grid_size = volume_element['grid_size']
                if new_val.shape != tuple(volume_element['grid_size']):
                    msg = (f'Volume element key "{key}" should have shape {grid_size}, '
                           f'but has shape: {new_val.shape}.')
                    raise ValueError(msg)
            else:
                if new_val.ndim != 1:
                    msg = (f'Volume element key "{key}" should be a 1D array but has '
                           f'{new_val.ndim} dimensions.')
                    raise TypeError(msg)
            if key in float_arrs:
                if new_val.dtype.char not in np.typecodes['AllFloat']:
                    msg = (f'Volume element key "{key}" should be a float array but has '
                           f'dtype "{new_val.dtype}".')
                    raise TypeError(msg)
            elif key in int_arrs:
                if new_val.dtype.char not in np.typecodes['AllInteger']:
                    msg = (f'Volume element key "{key}" should be an int array but has '
                           f'dtype "{new_val.dtype}".')
                    raise TypeError(msg)
            elif key in str_arrs:
                if new_val.dtype.char not in {'U', 'S'}:
                    msg = (f'Volume element key "{key}" should be a str array but has '
                           f'dtype "{new_val.dtype}".')
                    raise TypeError(msg)
            volume_element[key] = new_val

        # Check all "constituent_*" keys are the same length:
        if key.startswith('constituent_'):
            if num_const is None:
                num_const = volume_element[key].size
            elif volume_element[key].size != num_const:
                msg = (f'Not all "constituent_*" volume element keys are of equal length.'
                       f'Found lengths: {num_const} and {volume_element[key].size}.')
                raise ValueError(msg)

    if 'constituent_orientation_idx' in allowed:
        const_ori_idx = volume_element.get('constituent_orientation_idx')
        if const_ori_idx is None:
            # Set a default `constituent_orientation_idx`. Only possible if number of
            # orientations provided exactly matches number of constituents provided:
            num_oris = orientations['quaternions'].shape[0]
            num_const = volume_element['constituent_material_idx'].shape[0]
            if num_oris != num_const:
                msg = (f'Cannot set default values for `constituent_orientation_idx`, '
                       f'since the number of constituents ({num_const}) does not match '
                       f'the number of orientations ({num_oris}).')
                raise ValueError(msg)
            else:
                volume_element['constituent_orientation_idx'] = np.arange(num_oris)
        else:
            # Remove non-indexed orientations:
            const_ori_idx_uniq, const_ori_idx_inv = np.unique(
                const_ori_idx,
                return_inverse=True
            )
            oris_new = orientations['quaternions'][const_ori_idx_uniq]
            volume_element['orientations']['quaternions'] = oris_new
            volume_element['constituent_orientation_idx'] = const_ori_idx_inv

    # Provide a default `constituent_material_fraction`:
    if 'constituent_material_fraction' in allowed:

        const_mat_idx = volume_element['constituent_material_idx']
        validate_constituent_material_idx(const_mat_idx)

        const_mat_frac = volume_element.get('constituent_material_fraction')
        _, const_mat_idx_inv, const_mat_idx_counts = np.unique(
            const_mat_idx,
            return_inverse=True,
            return_counts=True,
        )
        if const_mat_frac is None:
            # Default is (1 / number of constituents) for each material:
            const_mat_frac = (1 / const_mat_idx_counts)[const_mat_idx_inv]
            volume_element['constituent_material_fraction'] = const_mat_frac
        else:
            # Check constituent fractions sum to one within a material:
            mat_const_idx = get_material_constituent_idx(const_mat_idx)
            for mat_idx, mat_i_const_idx in enumerate(mat_const_idx):
                frac_sum = np.sum(const_mat_frac[mat_i_const_idx])
                if not np.isclose(frac_sum, 1):
                    msg = (f'Constituent fractions must sum to one, but fractions in '
                           f'material {mat_idx} sum to {frac_sum}.')
                    raise ValueError(msg)

    if 'element_material_idx' in req:
        num_elems = volume_element['element_material_idx'].size
        grid_size_prod = np.prod(volume_element['grid_size'])
        if grid_size_prod != num_elems:
            msg = (f'Number of elements in volume element (i.e. size of array '
                   f'`element_material_idx`, ({num_elems}), should match the product of '
                   f'`grid_size` ({volume_element["grid_size"]}, {grid_size_prod}).')
            raise ValueError(msg)

    if 'constituent_material_idx' in req:
        max_mat_idx = np.max(volume_element['constituent_material_idx'])
        num_mats = volume_element['material_homog'].size
        if max_mat_idx != (num_mats - 1):
            msg = (f'Maximum material index in `constituent_material_idx` ({max_mat_idx})'
                   f' does not index into `material_homog` with length {num_mats}.')
            raise ValueError(msg)

    if homog_schemes:
        # Check material homogenization scheme labels exist in `homog_schemes`:
        for mat_idx, mat_i_homog in enumerate(volume_element['material_homog']):
            if str(mat_i_homog) not in homog_schemes:
                msg = (f'Homogenization scheme for material index {mat_idx} '
                       f'("{mat_i_homog}") is not present in `homog_schemes`.')
                raise ValueError(msg)

    if phases:
        # Check constituent phase labels exist in `phases`:
        for const_idx, cons_i_phase in enumerate(volume_element['constituent_phase_label']):
            if str(cons_i_phase) not in phases:
                msg = (f'Phase for constituent index {const_idx} ("{cons_i_phase}") is '
                       f'not present in `phases`.')
                raise ValueError(msg)

    return volume_element


def validate_constituent_material_idx(constituent_material_idx):
    """Check that a constituent_material_idx array (as defined within a volume element)
    is an increasing range starting from zero.

    Parameters
    ----------
    constituent_material_idx : ndarray of shape (N,) of int

    """

    cmi_range = np.arange(0, np.max(constituent_material_idx) + 1)
    if np.setdiff1d(cmi_range, constituent_material_idx).size:
        msg = (f'The unique values (material indices) in `constituent_material_idx` '
               f'should form an integer range. This is because the distinct materials '
               f'are defined implicitly through other index arrays in the volume '
               f'element.')
        raise ValueError(msg)


def validate_material_constituent_idx(material_constituent_idx):
    """Check that a material_constituent_idx list 

    Parameters
    ----------
    material_constituent_idx : list of 1D ndarray of variable length of int
        The inverse index array to the input array. The list length will be equal to
        the number of materials.

    """

    # Any repeat would imply a constituent appears multiply, which we disallow:
    mci_concat = np.sort(np.concatenate(material_constituent_idx))
    mci_range = np.arange(0, np.max(mci_concat) + 1)
    if not np.array_equal(mci_range, mci_concat):
        msg = (f'All values (constituent indices) in `material_constituent_idx` '
               f'should form an integer range. This is because the distinct constituents '
               f'are defined implicitly through other index arrays in the volume '
               f'element.')
        raise ValueError(msg)


def get_material_constituent_idx(constituent_material_idx):
    """Get the index array that is the inverse of the constituent_material_idx
    index array.

    Parameters
    ----------
    constituent_material_idx : (list or ndarray of shape (N,)) of int
        Determines the material to which each constituent belongs, where N is the
        number of constituents.

    Returns
    -------
    material_constituent_idx : list of 1D ndarray of variable length of int
        The inverse index array to the input array. The list length will be equal to
        the number of materials.

    """

    validate_constituent_material_idx(constituent_material_idx)
    material_constituent_idx = []
    for mat_idx in np.unique(constituent_material_idx):
        mat_const_idx_i = np.where(np.isin(constituent_material_idx, mat_idx))[0]
        material_constituent_idx.append(mat_const_idx_i)

    return material_constituent_idx


def get_constituent_material_idx(material_constituent_idx):
    """Get the index array that is the inverse of the material_constituent_idx
    index list.

    Parameters
    ----------
    material_constituent_idx : list of ((1D ndarray or list) of variable length) of int
        The inverse index array to the input array. The list length will be equal to
        the number of materials.

    Returns
    -------
    constituent_material_idx : (list or ndarray of shape (N,)) of int
        Determines the material to which each constituent belongs, where N is the
        number of constituents.

    """

    validate_material_constituent_idx(material_constituent_idx)

    num_const = np.max(np.concatenate(material_constituent_idx))
    constituent_material_idx = np.ones(num_const + 1) * np.nan
    for mat_idx, const_idx in enumerate(material_constituent_idx):
        constituent_material_idx[const_idx] = mat_idx

    constituent_material_idx = constituent_material_idx.astype(int)

    return constituent_material_idx


def get_volume_element_materials(volume_element, homog_schemes=None, phases=None, P=-1):
    """Get the materials list from a volume element that can be used to populate
    the "material" list in a DAMASK materials.yaml file.

    Parameters
    ----------
    volume_element : dict
    P : int, optional
        The "P" constant, either +1 or -1, as defined within [1]. By default, set to -1,
        which is what DAMASK uses. If the quaternions defined within the volume element
        have a P that is different to this P, the quaternions as output from this 
        function will be converted (i.e. the vector parts will be scaled by -1). Using
        the default value here, -1, will ensure the output quaternions are suitable
        for the DAMASK input file.

    Returns
    -------
    materials : list of dict

    References
    ----------
    [1] Rowenhorst, D, A D Rollett, G S Rohrer, M Groeber, M Jackson,
        P J Konijnenberg, and M De Graef. "Consistent Representations
        of and Conversions between 3D Rotations". Modelling and Simulation
        in Materials Science and Engineering 23, no. 8 (1 December 2015):
        083501. https://doi.org/10.1088/0965-0393/23/8/083501.  

    """

    volume_element = validate_volume_element(
        volume_element,
        homog_schemes=homog_schemes,
        phases=phases,
    )

    const_mat_idx = volume_element['constituent_material_idx']
    mat_const_idx = get_material_constituent_idx(const_mat_idx)

    all_quats = volume_element['orientations']['quaternions']

    quat_comp_order = volume_element['orientations'].get('quat_component_ordering')
    if quat_comp_order != 'scalar-vector':
        msg = (f'Quaternion component ordering (`quat_component_ordering`) should be '
               f'"scalar-vector", as adopted by DAMASK, but it is actually '
               f'"{quat_comp_order}"')
        raise RuntimeError(msg)

    const_mat_frac = volume_element['constituent_material_fraction']
    const_ori_idx = volume_element['constituent_orientation_idx']
    const_phase_lab = volume_element['constituent_phase_label']

    materials = []
    for mat_idx, mat_i_const_idx in enumerate(mat_const_idx):

        mat_i_constituents = []
        for const_idx in mat_i_const_idx:
            mat_i_const_j_phase = str(const_phase_lab[const_idx])
            mat_i_const_j_ori = all_quats[const_ori_idx[const_idx]]

            if phases[mat_i_const_j_phase]['lattice'] == 'hP':

                if 'unit_cell_alignment' not in volume_element['orientations']:
                    msg = 'Orientation `unit_cell_alignment` must be specified.'
                    raise ValueError(msg)

                transform_angle = None
                x_align = volume_element['orientations']['unit_cell_alignment'].get('x')
                y_align = volume_element['orientations']['unit_cell_alignment'].get('y')
                if x_align == "a*" or y_align == "b":
                    # rotate by 30 degrees about `z`
                    transform_angle = np.pi/6
                elif x_align == "b*" or y_align == "a":
                    # rotate by 90 degrees about `-z`
                    transform_angle = -np.pi/2
                elif x_align == "a" or y_align == "b*":
                    # already in DAMASK convention
                    pass 
                elif x_align == "b" or y_align == "a*":
                    # rotate by 120 degrees about `z`
                    transform_angle = 2*np.pi/3
                else:
                    msg = (f'Cannot convert from the following specified unit cell '
                           f'alignment to DAMASK-compatible unit cell alignment (x//a): '
                           f'{volume_element["orientations"]["unit_cell_alignment"]}')
                    NotImplementedError(msg)
                
                if transform_angle is not None:
                    hex_transform_quat = axang2quat(
                        volume_element['orientations']['P'] * np.array(
                            [0, 0, 1], dtype=np.longdouble),
                        np.longdouble(transform_angle)
                    )
                    mat_i_const_j_ori = multiply_quaternions(
                        q1=hex_transform_quat,
                        q2=mat_i_const_j_ori,
                        P=volume_element['orientations']['P'],
                    )  

            if volume_element['orientations']['P'] != P:
                # Make output quaternions consistent with desired "P" convention, as
                # determined by `P` argument:
                mat_i_const_j_ori[1:] *= -1

            mat_i_const_j = {
                'v': float(const_mat_frac[const_idx]),
                'O': mat_i_const_j_ori.tolist(),  # list of native float or np.longdouble
                'phase': mat_i_const_j_phase,
            }
            mat_i_constituents.append(mat_i_const_j)

        materials.append({
            'homogenization': str(volume_element['material_homog'][mat_idx]),
            'constituents': mat_i_constituents,
        })

    return materials


def validate_element_material_idx(element_material_idx):
    num_mats = np.max(element_material_idx) + 1
    emi_range = np.arange(0, num_mats)
    set_diff = np.setdiff1d(emi_range, element_material_idx)
    if set_diff.size:
        msg = (f'The unique values (material indices) in `element_material_idx` '
               f'should form an integer range. This is because the distinct '
               f'materials are defined implicitly through other index arrays in the '
               f'volume element. Found missing material indices:\n{set_diff}')
        raise ValueError(msg)

    return num_mats


def prepare_material_yaml_data(mat_dat):
    """Prepare data for writing to the DAMASK material.yaml file, by choosing desired
    formatting where necessary, including formatting quaternions to a given precision.

    Parameters
    ----------
    mat_dat : dict 
        Dict with keys:
            phases
            homogenization
            material

    Returns
    -------
    mat_dat_fmt : dict
        Copy of input dict, `mat_dat`, where some data has been replaced by objects
        from ruamel.yaml to provide desired formatting.

    """

    # Write out quaternions to maximum precision of the dtype:
    first_ori = mat_dat['material'][0]['constituents'][0]['O'][0]
    if isinstance(first_ori, np.floating):
        ORI_NUM_DP = np.finfo(first_ori.dtype).precision
    else:
        ORI_NUM_DP = 15  # precision of native float, roughly...

    mat_dat_fmt = copy.deepcopy(mat_dat)

    for material in mat_dat_fmt['material']:
        for const in material['constituents']:
            ori_list = []
            for ori in const['O']:
                kwargs = {
                    'width': ORI_NUM_DP + 2,
                    'prec': 1,
                    'm_sign': False,
                }
                if ori < 0:
                    kwargs.update({
                        'prec': 2,
                        'm_sign': '-',
                        'width': kwargs['width'] + 1,
                    })
                ori_list.append(ruamel.yaml.scalarfloat.ScalarFloat(ori, **kwargs))
            const['O'] = ori_list

    return mat_dat_fmt


def get_coordinate_grid(size, grid_size):
    """Get the coordinates of the element centres of a uniform grid."""

    grid_size = np.array(grid_size)
    size = np.array(size)

    grid = np.meshgrid(*[np.arange(i) for i in grid_size])
    grid = np.moveaxis(np.array(grid), 0, -1)  # shape (*grid_size, dimension)

    element_size = (size / grid_size).reshape(1, 1, -1)

    coords = grid * size.reshape(1, 1, -1) / grid_size.reshape(1, 1, -1)
    coords += element_size / 2

    return coords, element_size


def perpendicular_vectors(vec, axis=-1):
    """
    Get 3-vectors perpendicular to a given set of 3-vectors.

    Parameters
    ----------
    vec : ndarray
        Array of vectors.
    axis : int, optional
        The axis along which the 3-vectors are defined.

    Returns
    -------
    perp_vec : ndarray
        Array of the same shape as input array `vec`, where each output vector
        is perpendicular to its corresponding input vector.

    """

    if vec.shape[axis] != 3:
        raise ValueError('Size of dimension `axis` ({}) along `vec` must be 3,'
                         ' but is {}.'.format(axis, vec.shape[axis]))

    # Reshape:
    vec = np.swapaxes(vec, -1, axis)

    # Return array will have the same shape as input
    perp_vec = np.ones_like(vec) * np.nan

    # Find where first component magnitudes are larger than last:
    a_gt_c = np.abs(vec[..., 0]) > np.abs(vec[..., 2])
    a_notgt_c = np.logical_not(a_gt_c)

    # Make bool index arrays
    a_gt_c_0 = np.zeros_like(perp_vec, dtype=bool)
    a_gt_c_0[..., 0] = a_gt_c
    a_gt_c_1 = np.roll(a_gt_c_0, shift=1, axis=-1)
    a_gt_c_2 = np.roll(a_gt_c_1, shift=1, axis=-1)

    a_notgt_c_0 = np.zeros_like(perp_vec, dtype=bool)
    a_notgt_c_0[..., 0] = a_notgt_c
    a_notgt_c_1 = np.roll(a_notgt_c_0, shift=1, axis=-1)
    a_notgt_c_2 = np.roll(a_notgt_c_1, shift=1, axis=-1)

    # Set each component of the output vectors:
    perp_vec[a_gt_c_0] = vec[a_gt_c][..., 2]
    perp_vec[a_gt_c_1] = 0
    perp_vec[a_gt_c_2] = -vec[a_gt_c][..., 0]

    perp_vec[a_notgt_c_0] = 0
    perp_vec[a_notgt_c_1] = vec[a_notgt_c][..., 2]
    perp_vec[a_notgt_c_2] = -vec[a_notgt_c][..., 1]

    # Reshape to original shape:
    perp_vec = np.swapaxes(perp_vec, -1, axis)

    return perp_vec

def spread_orientations(volume_element, phase_names, sigmas):
    """Split DAMASK single-constituent materials into multiple materials with an
    orientation spread.

    Parameters
    ----------
    volume_element : dict
    phase_name : str
        Name of the phase for which constituents (i.e. materials) will be split.

    Returns
    -------
    volume_element : dict
        A new volume element with possibly more constituents corresponding to spread
        orientations.

    """

    from damask import Rotation

    volume_element = copy.deepcopy(volume_element)

    for phase_name in phase_names:
        sigma = sigmas[phase_names.index(phase_name)]

        # identify constituents (i.e. materials) belonging to the specified phase:
        const_idx = np.where(volume_element["constituent_phase_label"] == phase_name)[0]

        # each identified constituent will be split into one or more constituents, corresponding
        # to the number of assigned voxels:
        for const_idx_i in const_idx:
            base_ori = volume_element["orientations"]["quaternions"][
                volume_element["constituent_orientation_idx"][const_idx_i]
            ]
            P_val = volume_element["orientations"]['P']
            base_ori_dmsk = Rotation.from_quaternion(base_ori, P=P_val)

            elem_idx = np.where(volume_element["element_material_idx"] == const_idx_i)
            num_elems = len(elem_idx[0])

            # generate a set of orientations that have a defined spread:
            try:
                # known: v3 alpha 7
                spread_oris = Rotation.from_spherical_component(
                    base_ori_dmsk, sigma=sigma, degrees=True, shape=num_elems
                )
            except TypeError:
                # known: v3 alpha 3
                spread_oris = Rotation.from_spherical_component(
                    base_ori_dmsk, sigma=sigma, degrees=True, N=num_elems
                )

            cur_num_consts = volume_element["constituent_material_idx"].shape[0]
            cur_num_oris = volume_element["orientations"]["quaternions"].shape[0]

            # We add (num_elems - 1) new constituents because the pre-existing constituent
            # is reused for the first spread orientation:
            const_mat_idx_add = np.arange(cur_num_consts, cur_num_consts + (num_elems - 1))
            const_ori_idx_add = np.arange(cur_num_oris, cur_num_oris + (num_elems - 1))
            const_mat_frac_add = np.ones((num_elems - 1), dtype=float)
            const_phase_label_add = np.array([phase_name] * (num_elems - 1))
            homog_lab = volume_element["material_homog"][const_idx_i]
            mat_homog_add = np.array([homog_lab] * (num_elems - 1))

            volume_element["orientations"]["quaternions"][const_idx_i] = spread_oris[
                0
            ].quaternion
            volume_element["orientations"]["quaternions"] = np.vstack(
                [
                    volume_element["orientations"]["quaternions"],
                    spread_oris[1:],
                ]
            )
            volume_element["constituent_material_idx"] = np.concatenate(
                [volume_element["constituent_material_idx"], const_mat_idx_add]
            )
            volume_element["constituent_orientation_idx"] = np.concatenate(
                [volume_element["constituent_orientation_idx"], const_ori_idx_add]
            )
            volume_element["constituent_material_fraction"] = np.concatenate(
                [volume_element["constituent_material_fraction"], const_mat_frac_add]
            )
            volume_element["constituent_phase_label"] = np.concatenate(
                [volume_element["constituent_phase_label"], const_phase_label_add]
            )
            volume_element["material_homog"] = np.concatenate(
                [volume_element["material_homog"], mat_homog_add]
            )

            for idx, elem_idx_i in enumerate(zip(*elem_idx)):
                if idx == 0:
                    continue
                volume_element["element_material_idx"][elem_idx_i] = const_mat_idx_add[
                    idx - 1
                ]

    return volume_element

@contextmanager
def working_directory(path):
    """Change to a working directory and return to previous working directory on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

def visualise_static_outpurts(outputs, result, parsed_outs):
    """Create separate VTK file for grain and phase maps."""

    static_outputs = ['grain', 'phase']

    if isinstance(outputs, list):
        static_outputs = list(set(outputs).intersection(static_outputs))
        if len(static_outputs) > 0:
            v = result.geometry0

            for static_output in static_outputs:
                outputs.remove(static_output)
                dat_array = np.array(parsed_outs['field_data'][static_output]['data'])
                try:
                    # known: v3 alpha 3
                    v.add(dat_array.flatten(order='F'), label=static_output)
                except AttributeError:
                    # known: v3 alpha 7
                    v.set(data=dat_array.flatten(order='F'), label=static_output)

            v.save('static_outputs')

    return outputs

def generate_viz(hdf5_path, viz_spec, parsed_outs):
    if viz_spec is not None:

        from damask import Result

        if viz_spec is True:
            viz_spec = [{}]
        elif isinstance(viz_spec, dict):
            viz_spec = [viz_spec]

        result = Result(hdf5_path)

        Path('viz').mkdir(exist_ok=True)
        with working_directory('viz'):

            for viz_dict_idx, viz_dict in enumerate(viz_spec, 1):

                if len(viz_spec) > 1:
                    viz_dir = str(viz_dict_idx)
                    Path(viz_dir).mkdir(exist_ok=True)
                else:
                    viz_dir = '.'
                with working_directory(viz_dir):

                    # all incs if not specified:
                    incs_spec = viz_dict.get('increments', None)
                    parsed_incs = parse_inc_specs_using_result_obj(incs_spec, result)
                    try:
                        # known: v3 alpha 3
                        result = result.view('increments', parsed_incs)
                    except TypeError:
                        # known: v3 alpha 7
                        result = result.view(increments=parsed_incs)

                    # all phases if not specified:
                    phases = viz_dict.get('phases', True)
                    try:
                        # known: v3 alpha 3
                        result = result.view('phases', phases)
                    except TypeError:
                        # known: v3 alpha 7
                        result = result.view(phases=phases)

                    # all homogs if not specified:
                    homogs = viz_dict.get('homogenizations', True)
                    try:
                        # known: v3 alpha 3
                        result = result.view('homogenizations', homogs)
                    except TypeError:
                        # known: v3 alpha 7
                        result = result.view(homogenizations=homogs)

                    # all outputs if not specified:
                    outputs = viz_dict.get('fields', '*')

                    outputs = visualise_static_outpurts(outputs, result, parsed_outs)

                    # result.save_VTK(output=outputs) # v3 alpha 3?
                    result.export_VTK(output=outputs) # known: v3 alpha 7
