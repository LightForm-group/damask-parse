"""`damask_parse.readers.py`"""

from pathlib import Path

import pandas
import re
import numpy as np
from ruamel.yaml import YAML

from damask_parse.utils import (
    get_header_lines,
    get_num_header_lines,
    get_HDF5_incremental_quantity,
    validate_volume_element,
    validate_element_material_idx,
)
from damask_parse.legacy.readers import parse_microstructure, parse_texture_gauss

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
    """Parse a DAMASK geometry file into a volume element.

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
            material_homog_idx : 1D ndarray of str
                Determines the homogenization scheme for each material.
            orientations : dict
                Dict containing the following keys:
                    type : "euler"
                    euler_angles : ndarray of shape (R, 3) of float
                        Array of R row three-vectors of Euler angles. Specified as proper
                        Euler angles in the Bunge convention. (Rotations are about Z,
                        new X, new new Z.)
                    unit_cell_alignment : dict
                        Alignment of the unit cell.
                    euler_degrees : bool
                        If True, `euler_angles` are represented in degrees, rather than
                        radians.
            constituent_phase_label_idx : 1D ndarray of int
                Zero-indexed integer index array mapping a constituent to its phase index.
            constituent_orientation_idx : 1D ndarray of int
                Zero-indexed integer index array mapping a constituent to its orientation
                index.
            meta : dict
                Any meta information associated with the generation of this volume
                element.

    """

    num_header = get_num_header_lines(geom_path)

    with Path(geom_path).open('r') as handle:

        lines = handle.read()

        grid_size = None
        grid_pat = r'grid\s+a\s+(\d+)\s+b\s+(\d+)\s+c\s+(\d+)'
        grid_match = re.search(grid_pat, lines)
        if grid_match:
            grid_size = [int(i) for i in grid_match.groups()]
        else:
            raise ValueError('`grid` not specified in file.')

        element_material_idx = []
        for ln_idx, ln in enumerate(lines.splitlines()):
            ln_split = ln.strip().split()
            if ln_idx > num_header:
                element_material_idx.extend([int(i) for i in ln_split])

        element_material_idx = np.array(element_material_idx).reshape(grid_size[::-1])
        element_material_idx = element_material_idx.swapaxes(0, 2)
        element_material_idx -= 1  # zero-indexed
        num_mats = validate_element_material_idx(element_material_idx)

        constituent_phase_label_idx = None
        constituent_orientation_idx = None
        pat = r'\<microstructure\>[\s\S]*\(constituent\).*'
        ms_match = re.search(pat, lines)
        if ms_match:
            ms_str = ms_match.group()
            microstructure = parse_microstructure(ms_str)
            constituent_phase_label_idx = microstructure['phase_idx']
            constituent_orientation_idx = microstructure['texture_idx']

        orientations = None
        pat = r'\<texture\>[\s\S]*\(gauss\).*'
        texture_match = re.search(pat, lines)
        if texture_match:
            texture_str = texture_match.group()
            texture_gauss = parse_texture_gauss(texture_str)
            orientations = {
                'type': 'euler',
                'euler_angles': texture_gauss['euler_angles'],
                'euler_degrees': texture_gauss['euler_degrees'],
                'euler_angle_labels': texture_gauss['euler_angle_labels'],
                'unit_cell_alignment': {
                    'x': 'a',
                    'z': 'c',
                }
            }

        # Check indices in `constituent_orientation_idx` are valid, given `orientations`:
        if ms_match and (
            np.min(constituent_orientation_idx) < 0 or
            np.max(constituent_orientation_idx) > len(orientations['euler_angles'])
        ):
            msg = 'Orientation indices in `constituent_orientation_idx` are invalid.'
            raise ValueError(msg)

        # Parse header information:
        size_pat = (r'size\s+x\s+(\d+(?:\.\d+)*)'
                    r'\s+y\s+(\d+(?:\.\d+)*)\s+z\s+(\d+(?:\.\d+)*)')
        size_match = re.search(size_pat, lines)
        size = None
        if size_match:
            size = [float(i) for i in size_match.groups()]

        origin_pat = r'origin\s+x\s+(\d+\.\d+)\s+y\s+(\d+\.\d+)\s+z\s+(\d+\.\d+)'
        origin_match = re.search(origin_pat, lines)
        origin = None
        if origin_match:
            origin = [float(i) for i in origin_match.groups()]

        homo_pat = r'homogenization\s+(\d+)'
        homo_match = re.search(homo_pat, lines)
        material_homog_idx = None
        if homo_match:
            # Same homogenization for each material ID:
            homog_idx = int(homo_match.group(1)) - 1  # zero-indexed
            material_homog_idx = np.zeros(num_mats).astype(int) + homog_idx

        com_pat = r'(geom_.*)'
        commands = re.findall(com_pat, lines)

        geometry = {
            'grid_size': grid_size,
            'size': size,
            'origin': origin,
            'orientations': orientations,
            'element_material_idx': element_material_idx,
            'material_homog_idx': material_homog_idx,
            'constituent_phase_label_idx': constituent_phase_label_idx,
            'constituent_orientation_idx': constituent_orientation_idx,
            'meta': {
                'num_header': num_header,
                'commands': commands,
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


def read_HDF5_file(hdf5_path, incremental_data, operations=None):
    """Operate on and extract data from an HDF5 file generated by a DAMASK run.

    Parameters
    ----------
    hdf5_path : Path or str
        Path to the HDF5 file.
    incremental_data : list of dict
        List of incremental data to extract from the HDF5 file. This is a list of dicts
        with the following keys:
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
    operations : list of dict, optional
        List of methods to invoke on the DADF5 object. This is a list of dicts with the
        following keys:
            name : str
                The name of the DADF5 method.
            args : dict
                Parameter names and their values to pass to the DADF5 method. This
                assumes all DADF5 method parameters are of positional-or-keyword type.
            opts : dict
                Additional options.

    Returns
    -------
    volume_element_response : dict
        Dict with keys determined by the `incremental_data` list.

    """

    try:
        from damask import Result
        sim_data = Result(hdf5_path)
    except ImportError:
        from damask import DADF5
        sim_data = DADF5(hdf5_path)

    for op in operations or []:
        func = getattr(sim_data, op['name'], None)
        if not func:
            raise AttributeError(f'The Result object has no attribute: {op["name"]}.')
        else:
            func(**op['args'])

        # Deal with specific options:
        if op['opts'].get('add_Mises', {}):

            if op["name"] == 'add_Cauchy':
                label = f'sigma'

            elif op["name"] == 'add_strain_tensor':
                # Include defaults from `DADF5.add_strain_tensor`:
                t = op['args'].get('t', 'U')
                m = op['args'].get('m', 0)
                F = op['args'].get('F', 'F')
                label = f'epsilon_{t}^{m}({F})'

            else:
                msg = (f'Operation "{op["name"]}" is not compatible with option '
                       f'"add_Mises".')
                raise ValueError(msg)

            sim_data.add_Mises(label)

    volume_element_response = {}
    for inc_dat_spec in incremental_data:
        inc_dat = get_HDF5_incremental_quantity(
            hdf5_path=hdf5_path,
            dat_path=inc_dat_spec['path'],
            transforms=inc_dat_spec.get('transforms'),
            increments=inc_dat_spec.get('increments', 1),
        )
        volume_element_response.update({
            inc_dat_spec['name']: {
                'data': inc_dat,
                'meta': {
                    'path': inc_dat_spec['path'],
                    'transforms': inc_dat_spec.get('transforms'),
                    'increments': inc_dat_spec.get('increments', 1),
                },
            }
        })

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
    }

    for mat_idx, material in enumerate(material_dat['microstructure']):
        material_homog.append(material['homogenization'])
        for const in material['constituents']:
            const_material_idx.append(mat_idx)
            const_material_fraction.append(const['fraction'])
            const_phase_label.append(const['phase'])
            orientations['quaternions'].append(const['orientation'])
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


def geom_to_volume_element(geom_path, phase_labels, homog_label, orientations=None):
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
    orientations : dict, optional
        If specified, use these orientations instead of those that might be specified in
        the geometry file. Dict containing the following keys:
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
        'orientations': orientations or geom_dat['orientations'],
        'element_material_idx': geom_dat['element_material_idx'],
        'grid_size': geom_dat['grid_size'],
        'size': geom_dat['size'],
        'phase_labels': phase_labels,
        'homog_label': homog_label,
    }
    volume_element = validate_volume_element(volume_element)
    return volume_element
