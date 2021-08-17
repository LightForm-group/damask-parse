"""`damask_parse.readers.py`"""

from pathlib import Path

import re
import numpy as np
from ruamel.yaml import YAML

from damask_parse.utils import (
    get_HDF5_incremental_quantity,
    validate_volume_element,
    get_field_data,
    get_vol_data,
    get_phase_data,
    reshape_field_data,
    apply_grain_average,
)

__all__ = [
    'read_geom',
    'read_spectral_stdout',
    'read_spectral_stderr',
    'read_HDF5_file',
    'read_material',
    'geom_to_volume_element',
]


def parse_increment_iteration(inc_iter_str):

    float_pat = r'-?\d+\.\d+'
    sci_float_pat = r'-?\d+\.\d+E[+|-]\d+'

    dg_pat = r'deformation gradient aim\s+=\n(\s+(?:(?:' + float_pat + r'\s+){3}){3})'
    dg_match = re.search(dg_pat, inc_iter_str)
    dg_str = dg_match.group(1)
    dg = np.array([float(i) for i in dg_str.split()]).reshape((3, 3))

    pk_pat = r'Piola--Kirchhoff stress\s+\/\s.*=\n(\s+(?:(?:' + \
        float_pat + r'\s+){3}){3})'
    pk_match = re.search(pk_pat, inc_iter_str)
    pk_str = pk_match.group(1)
    pk = np.array([float(i) for i in pk_str.split()]).reshape((3, 3))

    err_pat = r'error (.*)\s+=\s+(-?\d+\.\d+)\s\((' + sci_float_pat + \
        r')\s(.*),\s+tol\s+=\s+(' + sci_float_pat + r')\)'
    err_matches = re.findall(err_pat, inc_iter_str)
    converge_err = {}
    for i in err_matches:
        err_key = 'error_' + i[0].strip().replace(' ', '_')
        converge_err.update({
            err_key: {
                'value': float(i[2]),
                'unit': i[3].strip(),
                'tol': float(i[4]),
                'relative': float(i[1]),
            }
        })

    inc_iter = {
        'deformation_gradient_aim': dg,
        'piola_kirchhoff_stress': pk,
        **converge_err,
    }

    return inc_iter


def parse_increment(inc_str):

    warn_msg = r'│\s+warning\s+│\s+│\s+(\d+)\s+│\s+├─+┤\s+│(.*)│\s+\s+│(.*)│'
    warnings_matches = re.findall(warn_msg, inc_str)
    warnings = [
        {
            'code': int(i[0]),
            'message': i[1].strip() + ' ' + i[2].strip(),
        } for i in warnings_matches
    ]

    if not re.search(r'increment\s\d+\sconverged', inc_str):
        parsed_inc = {
            'converged': False,
            'warnings': warnings,
        }
        return parsed_inc

    inc_position_pat = (r'Time\s+(\d+\.\d+E[+|-]\d+)s:\s+Increment'
                        r'\s+(\d+\/\d+)-(\d+\/\d+)\s+of\sload\scase\s+(\d+)')
    inc_pos = re.search(inc_position_pat, inc_str)
    inc_pos_dat = inc_pos.groups()

    inc_time = float(inc_pos_dat[0])
    inc_number = int(inc_pos_dat[1].split('/')[0])
    inc_cut_back = 1 / int(inc_pos_dat[2].split('/')[1])
    inc_load_case = int(inc_pos_dat[3])

    inc_iter_split_str = r'={75}'
    inc_iter_split = re.split(inc_iter_split_str, inc_str)

    dg_arr = []
    pk_arr = []
    converge_errors = None
    err_keys = None
    num_iters = len(inc_iter_split) - 1

    for idx, i in enumerate(inc_iter_split[:-1]):

        inc_iter_i = parse_increment_iteration(i)

        if idx == 0:
            err_keys = [j for j in inc_iter_i.keys() if j.startswith('error_')]
            converge_errors = dict(
                zip(err_keys, [{'value': [], 'tol': [], 'relative': []}
                               for _ in err_keys])
            )

        dg_arr.append(inc_iter_i['deformation_gradient_aim'])
        pk_arr.append(inc_iter_i['piola_kirchhoff_stress'])

        for j in err_keys:
            for k in ['value', 'tol', 'relative']:
                converge_errors[j][k].append(inc_iter_i[j][k])

    dg_arr = np.array(dg_arr)
    pk_arr = np.array(pk_arr)
    for j in err_keys:
        for k in ['value', 'tol', 'relative']:
            converge_errors[j][k] = np.array(converge_errors[j][k])

    parsed_inc = {
        'converged': True,
        'inc_number': inc_number,
        'inc_time': inc_time,
        'inc_cut_back': inc_cut_back,
        'inc_load_case': inc_load_case,
        'deformation_gradient_aim': dg_arr,
        'piola_kirchhoff_stress': pk_arr,
        'num_iters': num_iters,
        **converge_errors,
    }

    return parsed_inc


def read_geom(geom_path):
    """Parse a DAMASK VTR geometry file into a volume element.

    Parameters
    ----------
    geom_path : str or Path
        Path to the DAMASK geometry file.

    Returns
    -------
    geometry : dict
        Dictionary of the parsed data from the geometry file, with keys:
            element_material_idx : ndarray of shape equal to `grid_size` of int
                A mapping that determines the grain index for each voxel.
            grid_size : ndarray of int of size 3
                Resolution of volume element discretisation in each direction.
            size : list of length 3
                Volume element size. By default set to unit size: [1, 1, 1].
            origin : list of length 3
                Volume element origin. By default: [0, 0, 0].
            meta : dict
                Any meta information associated with the generation of this
                volume element.

    """
    from damask import Grid

    ve_grid = Grid.load(geom_path)

    geometry = {
        'grid_size': ve_grid.cells,
        'size': ve_grid.size,
        'origin': ve_grid.origin,
        'element_material_idx': ve_grid.material,
        'meta': {
            'comments': ve_grid.comments,
        },
    }

    return geometry


def read_spectral_stdout(path):

    path = Path(path)
    inc_split_str = r'\s#{75}'

    with path.open('r', encoding='utf8') as handle:
        lines = handle.read()

        inc_split = re.split(inc_split_str, lines)

        dg_arr = []
        pk_arr = []
        inc_idx = []
        inc_pos_dat = {
            'inc_number': [],
            'inc_time': [],
            'inc_cut_back': [],
            'inc_load_case': [],
        }
        err_keys = None
        converge_errors = None
        warnings = []

        for idx, i in enumerate(inc_split[1:]):

            parsed_inc = parse_increment(i)
            if parsed_inc['converged']:

                inc_idx.extend([idx] * parsed_inc['num_iters'])
                if idx == 0:
                    err_keys = [j for j in parsed_inc.keys() if j.startswith('error_')]
                    converge_errors = dict(
                        zip(err_keys, [{'value': [], 'tol': [], 'relative': []}
                                       for _ in err_keys])
                    )
                dg_arr.extend(parsed_inc.pop('deformation_gradient_aim'))
                pk_arr.extend(parsed_inc.pop('piola_kirchhoff_stress'))
                for j in err_keys:
                    for k in ['value', 'tol', 'relative']:
                        converge_errors[j][k].extend(parsed_inc[j][k])

                for k in ['inc_number', 'inc_time', 'inc_cut_back', 'inc_load_case']:
                    inc_pos_dat[k].append(parsed_inc[k])

            else:
                warnings.extend(parsed_inc['warnings'])

        inc_idx = np.array(inc_idx)
        dg_arr = np.array(dg_arr)
        pk_arr = np.array(pk_arr)
        for j in err_keys:
            for k in ['value', 'tol', 'relative']:
                converge_errors[j][k] = np.array(converge_errors[j][k])

        for k in ['inc_number', 'inc_time', 'inc_cut_back', 'inc_load_case']:
            inc_pos_dat[k] = np.array(inc_pos_dat[k])

        out = {
            'deformation_gradient_aim': dg_arr,
            'piola_kirchhoff_stress': pk_arr,
            'increment_idx': inc_idx,
            'warnings': warnings,
            **converge_errors,
            **inc_pos_dat
        }

    return out


def read_spectral_stderr(path):

    path = Path(path)

    with path.open('r', encoding='utf8') as handle:
        lines = handle.read()
        errors_pat = r'│\s+error\s+│\s+│\s+(\d+)\s+│\s+├─+┤\s+│(.*)│\s+\s+│(.*)│'
        matches = re.findall(errors_pat, lines)
        errors = [
            {
                'code': int(i[0]),
                'message': i[1].strip() + ' ' + i[2].strip(),
            } for i in matches
        ]

        return errors


def read_HDF5_file(
    hdf5_path,
    geom_path=None,
    incremental_data=None,
    volume_data=None,
    phase_data=None,
    field_data=None,
    grain_data=None,
    operations=None
):
    """Operate on and extract data from an HDF5 file generated by a DAMASK run.

    Parameters
    ----------
    hdf5_path : Path or str
        Path to the HDF5 file.
    incremental_data : list of dict, optional
        (Deprecated) List of incremental data to extract from the HDF5 file.
        This is a list of dicts with the following keys:
            name: str
                The name by which the quantity will be stored in the output dict.
            path: str
                The HDF5 "path" to the dataset.
            transforms: list of dict, optional
                If specified this is a list of dicts, each with the following keys:
                    sum_along_axes : int, optional
                        If specified, take the sum the array along this axis.
                    mean_along_axes: int, optional
                        If specified, take the mean average of the array along this axis.
            increments : int
                Step size
    volume_data : list of dict, optional
        List of data to extract from the entire volume. This is a list of dict
        with following keys:
            field_name: str
                Name of the data field to extract
            increments: list of dict, optional
                List of increment specifications to extract data from. Values
                refer to increments in the simulation. Default to all. This is
                a list of dict one of the following sets of keys:
                    values: list of int
                        List of incremnts to extract
                    ----OR----
                    start: int
                        First increment to extract
                    stop: int
                        Final incremnt to extract (inclusive)
                    step: int
                        Step between increments to extract
            out_name: str, optional
                Name of the data
            transforms: list of dict, optional
                If specified this is a list of dicts, each with the following keys:
                    sum_along_axes : int, optional
                        If specified, take the sum the array along this axis.
                    mean_along_axes: int, optional
                        If specified, take the mean average of the array along this axis.
    phase_data : list of dict, optional
        List of data to extract from a single phase. This is a list of dict
        with following keys:
            field_name: str
                Name of the data field to extract
            phase_name : str
                Name of phase to
            increments: list of dict, optional
                List of increment specifications to extract data from. Values
                refer to increments in the simulation. Default to all. This is
                a list of dict one of the following sets of keys:
                    values: list of int
                        List of incremnts to extract
                    ----OR----
                    start: int
                        First increment to extract
                    stop: int
                        Final incremnt to extract (inclusive)
                    step: int
                        Step between increments to extract
            out_name: str, optional
                Name of the data
            transforms: list of dict, optional
                If specified this is a list of dicts, each with the following keys:
                    sum_along_axes : int, optional
                        If specified, take the sum the array along this axis.
                    mean_along_axes: int, optional
                        If specified, take the mean average of the array along this axis.
    field_data : list of dict, optional
        List of field data to extract. Only extracts for the first constituent
        of each material point. This is a list of dict with following keys:
            field_name: str
                Name of the data field to extract
            increments: list of dict, optional
                List of increment specifications to extract data from. Values
                refer to increments in the simulation. Default to all. This is
                a list of dict one of the following sets of keys:
                    values: list of int
                        List of incremnts to extract
                    ----OR----
                    start: int
                        First increment to extract
                    stop: int
                        Final incremnt to extract (inclusive)
                    step: int
                        Step between increments to extract
        Special field_name keys exist, 'grain' and 'phase'. Use 'u_n' or 'u_p'
        for displacement.
    grain_data : list of dict, optional
        List of grain data to extract. Only extracts for the first constituent
        of each material point. This is a list of dict with following keys:
            field_name: str
                Name of the data field to extract
            increments: list of dict, optional
                List of increment specifications to extract data from. Values
                refer to increments in the simulation. Default to all. This is
                a list of dict one of the following sets of keys:
                    values: list of int
                        List of incremnts to extract
                    ----OR----
                    start: int
                        First increment to extract
                    stop: int
                        Final incremnt to extract (inclusive)
                    step: int
                        Step between increments to extract
    operations : list of dict, optional
        List of methods to invoke on the DADF5 object. This is a list of dicts with the
        following keys:
            name : str
                The name of the DADF5 method.
            args : dict
                Parameter names and their values to pass to the DADF5 method. This
                assumes all DADF5 method parameters are of positional-or-keyword type.
            opts : dict, optional
                Additional options.

    Returns
    -------
    volume_element_response : dict
        Dict with keys determined by the `incremental_data` list and `field_data` dict.

    """
    if not geom_path:
        geom_path = Path(hdf5_path).parent / 'geom.vtr'

    # Open DAMASK output file if required
    if operations or volume_data or phase_data or field_data or grain_data:
        from damask import Result
        sim_data = Result(hdf5_path)

    # Load in grain mapping if required
    if grain_data or (field_data and (
            'grain' in (spec['field_name'] for spec in field_data))):
        from damask import Grid
        ve = Grid.load(geom_path)
        grains = ve.material

    for op in operations or []:
        func = getattr(sim_data, op['name'], None)
        if not func:
            raise AttributeError(f'The Result object has no attribute: {op["name"]}.')
        else:
            func(**op['args'])

        # Deal with specific options:
        if op.get('opts', {}).get('add_Mises', {}):

            if op["name"] == 'add_stress_Cauchy':
                label = 'sigma'

            elif op["name"] == 'add_strain':
                # Include defaults from `DADF5.add_strain_tensor`:
                t = op['args'].get('t', 'V')
                m = op['args'].get('m', 0)
                F = op['args'].get('F', 'F')
                label = f'epsilon_{t}^{m}({F})'

            else:
                msg = (f'Operation "{op["name"]}" is not compatible with option '
                       f'"add_Mises".')
                raise ValueError(msg)

            sim_data.add_equivalent_Mises(label)

    incremental_response = {}
    for spec in incremental_data or []:
        inc_dat = get_HDF5_incremental_quantity(
            hdf5_path=hdf5_path,
            dat_path=spec['path'],
            transforms=spec.get('transforms'),
            increments=spec.get('increments', 1),
        )
        incremental_response.update({
            spec['name']: {
                'data': inc_dat,
                'meta': {
                    'path': spec['path'],
                    'transforms': spec.get('transforms'),
                    'increments': spec.get('increments', 1),
                },
            }
        })

    volume_response = {}
    for spec in volume_data or []:
        field_name = spec['field_name']
        out_name = spec.get('out_name')
        transforms = spec.get('transforms')

        vol_dat, increments, phase_names = get_vol_data(
            sim_data, field_name, spec.get('increments'), transforms=transforms
        )
        # No increments returned, continue to next
        if not increments:
            continue

        # Get out_name or construct out_name
        if out_name is None or out_name in volume_response:
            if out_name in volume_response:
                print(f'`out_name` "{out_name}" already exists. Generating a new name.')
            out_name = [field_name]
            out_name += [f'{op}_{axis}' for t in transforms or []
                         for op, axis in t.items()]
            out_name = '_'.join(out_name)

        volume_response.update({
            out_name: {
                'data': vol_dat,
                'meta': {
                    'field_name': field_name,
                    'phase_names': phase_names,
                    'increments': increments,
                    'transforms': transforms,
                }
            }
        })

    phase_response = {}
    for spec in phase_data or []:
        field_name = spec['field_name']
        phase_name = spec['phase_name']
        out_name = spec.get('out_name')
        transforms = spec.get('transforms')

        phase_dat, increments = get_phase_data(
            sim_data, field_name, phase_name, spec.get('increments'),
            transforms=transforms
        )
        # No increments returned, continue to next
        if not increments:
            continue

        # Get out_name or construct out_name
        if out_name is None or out_name in phase_response:
            if out_name in phase_response:
                print(f'`out_name` "{out_name}" already exists. Generating a new name.')
            out_name = [field_name, phase_name]
            out_name += [f'{op}_{axis}' for t in transforms or []
                         for op, axis in t.items()]
            out_name = '_'.join(out_name)

        phase_response.update({
            out_name: {
                'data': phase_dat,
                'meta': {
                    'field_name': field_name,
                    'phase_name': phase_name,
                    'increments': increments,
                    'transforms': transforms,
                }
            }
        })

    field_response = {}
    for spec in field_data or []:
        field_name = spec['field_name']

        if field_name == 'phase':
            at_cell_ph, _, _, _ = sim_data._mappings()
            phase_mapping = np.empty(sim_data.N_materialpoints, dtype=np.uint8)
            phase_names = []

            for i, (phase_name, mask) in enumerate(at_cell_ph[0].items()):
                phase_mapping[mask] = i
                phase_names.append(phase_name)

            field_dat = reshape_field_data(phase_mapping,
                                           tuple(sim_data.cells))
            field_meta = {
                'phase_names': phase_names,
                'num_phases': len(np.unique(phase_mapping)),
            }

        elif field_name == 'grain':
            field_dat = grains
            field_meta = {'num_grains': len(np.unique(grains))}

        else:
            field_dat, increments = get_field_data(
                sim_data, field_name, spec.get('increments')
            )
            # No increments returned, continue to next
            if not increments:
                continue
            field_meta = {'increments': increments}

        field_response.update({
            field_name: {
                'data': field_dat,
                'meta': field_meta
            }
        })

    grain_response = {}
    for spec in grain_data or []:
        field_name = spec['field_name']

        # check if identical field data already exists
        if spec in (field_data or []):
            try:
                field_dat = field_response[field_name]
            except KeyError:
                # No increments returned in field response, continue to next
                continue
            increments = field_dat['meta']['increments']
            field_dat = field_dat['data']
        # otherwise create it
        else:
            field_dat, increments = get_field_data(
                sim_data, field_name, spec.get('increments')
            )
            # No increments returned, continue to next
            if not increments:
                continue

        # grain average
        is_oris = field_name == 'O'
        grain_dat = apply_grain_average(field_dat, grains, is_oris=is_oris)

        grain_response.update({
            field_name: {
                'data': grain_dat,
                'meta': {
                    'increments': increments,
                }
            }
        })

    volume_element_response = {
        'incremental_data': incremental_response,
        'volume_data': volume_response,
        'phase_data': phase_response,
        'field_data': field_response,
        'grain_data': grain_response,
    }

    return volume_element_response


def read_material(path):
    """Parse a DAMASK material.yaml input file.

    Parameters
    ----------
    path : str or Path
        Path to the DAMASK material.yaml file.

    Returns
    -------
    material_data : dict
        Parsed data from the DAMASK material.yaml file. Keys are:
            phases : dict
                The "phase" dict contained within the material file.
            homog_schemes : dict
                The "homogenization" dict contained within the material file.
            volume_element : dict
                Dict representing the volume element. The distribution of materials
                across the elements (i.e. keys `element_material_idx` and `grid_size`)
                are not included, since this information is not contained in the
                material file. With keys:
                    constituent_material_idx : ndarray of shape (N,) of int
                        Determines the material to which each constituent belongs, where N
                        is the number of constituents.
                    constituent_material_fraction: ndarray of shape (N,) of float
                        The fraction that each constituent occupies within its respective
                        material, where N is the number of constituents.
                    constituent_phase_label : ndarray of shape (N,) of str
                        Determines the phase label of each constituent, where N is the
                        number of constituents.
                    constituent_orientation_idx : ndarray of shape (N,) of int
                        Determines the orientation (as an index into `orientations`)
                        associated with each constituent, where N is the number of
                        constituents.
                    material_homog : ndarray of shape (M,) of str
                        Determines the homogenization scheme (from a list of available
                        homogenization schemes defined elsewhere) to which each material
                        belongs, where M is the number of materials.
                    orientations : dict
                        Dict containing the following keys:
                            type : str
                                Value is "quat".
                            quaternions : ndarray of shape (R, 4) of float
                                Array of R row four-vectors of unit quaternions.

    """

    yaml = YAML(typ='safe')
    material_dat = yaml.load(Path(path))

    material_homog = []
    const_material_idx = []
    const_material_fraction = []
    const_phase_label = []
    const_orientation_idx = []
    orientations = {
        'type': 'quat',
        'quaternions': [],
        'quat_component_ordering': 'scalar-vector',
        'unit_cell_alignment': {
            'x': 'a',
            'z': 'c',
        },
        'P': -1,
    }

    for mat_idx, material in enumerate(material_dat['material']):
        material_homog.append(material['homogenization'])
        for const in material['constituents']:
            const_material_idx.append(mat_idx)
            const_material_fraction.append(const['v'])
            const_phase_label.append(const['phase'])
            orientations['quaternions'].append(const['O'])
            const_orientation_idx.append(len(const_orientation_idx))

    vol_elem = {
        'constituent_material_idx': const_material_idx,
        'constituent_material_fraction': const_material_fraction,
        'constituent_phase_label': const_phase_label,
        'constituent_orientation_idx': const_orientation_idx,
        'material_homog': material_homog,
        'orientations': orientations,
    }
    material_data = {
        'volume_element': vol_elem,
        'phases': material_dat['phase'],
        'homog_schemes': material_dat['homogenization'],
    }
    material_data['volume_element'] = validate_volume_element(**material_data)

    return material_data


def geom_to_volume_element(geom_path, phase_labels, homog_label, orientations):
    """Read a DAMASK geom file and parse to a volume element.

    Parameters
    ----------
    geom_path : str or Path
        Path to the DAMASK geometry file. The geom file must include texture and
        microstructure parts in its header if `orientations` is None.
    phase_labels : list or ndarray of str, optional
        List of phase labels to associate with the constituents. The first list element is
        the phase label that will be associated with all of the geometrical elements
        for which an orientation is also specified. Additional list elements are
        phase labels for geometrical elements for which no orientations are
        specified. For instance, if the DAMASK command `geom_canvas` was used on
        a geom file to generate a new geom file that included additional material indices,
        the phases assigned to those additional material indices would be specified as
        additional list elements in `phase_labels`.
    homog_label : str, optional
        The homogenization scheme label to use for all materials in the volume element.
    orientations : dict
        Dict containing the following keys:
            type : str
                One of "euler", "quat".
            quaternions : (list or ndarray of shape (R, 4)) of float, optional
                Array of R row four-vectors of unit quaternions. Specify either
                `quaternions` or `euler_angles`.
            euler_angles : (list or ndarray of shape (R, 3)) of float, optional
                Array of R row three-vectors of Euler angles. Specify either `quaternions`
                or `euler_angles`. Specified as proper Euler angles in the Bunge
                convention (rotations are about Z, new-X, new-new-Z).
            euler_degrees : bool, optional
                If True, `euler_angles` are expected in degrees, rather than radians.
            unit_cell_alignment : dict
                Alignment of the unit cell.
            P : int, optional
                The "P" constant, either +1 or -1, as defined within [1].

    Returns
    -------
    volume_element : dict

    References
    ----------
    [1] Rowenhorst, D, A D Rollett, G S Rohrer, M Groeber, M Jackson,
        P J Konijnenberg, and M De Graef. "Consistent Representations
        of and Conversions between 3D Rotations". Modelling and Simulation
        in Materials Science and Engineering 23, no. 8 (1 December 2015):
        083501. https://doi.org/10.1088/0965-0393/23/8/083501.

    """

    geom_dat = read_geom(geom_path)
    volume_element = {
        'orientations': orientations,
        'element_material_idx': geom_dat['element_material_idx'],
        'grid_size': geom_dat['grid_size'],
        'size': geom_dat['size'],
        'origin': geom_dat['origin'],
        'phase_labels': phase_labels,
        'homog_label': homog_label,
    }
    volume_element = validate_volume_element(volume_element)
    return volume_element
