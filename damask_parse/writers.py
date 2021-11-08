"""`damask_parse.writers.py`"""

import copy
from pathlib import Path
from collections import OrderedDict

import numpy as np
from ruamel.yaml import YAML

from damask_parse.utils import (
    zeropad,
    format_1D_masked_array,
    align_orientations,
    get_volume_element_materials,
    validate_volume_element,
    prepare_material_yaml_data,
    masked_array_from_list,
)

__all__ = [
    'write_geom',
    'write_material',
    'write_numerics',
    'write_load_case',
]


def write_geom(dir_path, volume_element, name='geom.vtr'):
    """Write the geometry file for a spectral DAMASK simulation.

    Parameters
    ----------
    dir_path : str or Path
        Directory in which to generate the file(s).
    volume_element : dict
        Dict that represents the specification of a volume element, with keys:
            element_material_idx : ndarray of shape equal to `grid_size` of int, optional
                Determines the material to which each geometric model
                element belongs, where P is the number of elements.
            grid_size : ndarray of shape (3,) of int, optional
                Geometric model grid dimensions.
            size : list of length three, optional
                Volume element size. By default set to unit size: [1.0, 1.0, 1.0].
            origin : list of length three, optional
                Volume element origin. By default: [0, 0, 0].
    name : str, optional
        Name of geometry file to write. By default, set to "geom.vtr".

    Returns
    -------
    geom_path : Path
        File path to the generated geometry file.

    Notes
    -----
    The microstructure and texture parts are not included in the header
    of the generated file.

    """
    from damask import Grid

    volume_element = validate_volume_element(volume_element)
    element_material_idx = volume_element['element_material_idx']
    ve_size = volume_element.get('size')
    ve_origin = volume_element.get('origin')
    if ve_size is None:
        ve_size = [1.0, 1.0, 1.0]
    if ve_origin is None:
        ve_origin = [0.0, 0.0, 0.0]

    dir_path = Path(dir_path).resolve()
    geom_path = dir_path.joinpath(name)

    ve_grid = Grid(material=element_material_idx, size=ve_size,
                   origin=ve_origin)
    ve_grid.save(geom_path)

    return geom_path


def write_load_case(dir_path, load_cases, name='load.yaml'):
    """Write the load file for a DAMASK simulation.

    Parameters
    ----------
    dir_path : str or Path
        Directory in which to generate the file(s).
    load_cases : list of dict

    name : str, optional
        Name of the load file to write. By default, set to "load.yaml".

    Returns
    -------
    load_path : Path
        File path to the generated load file.

    """
    from damask import Rotation

    load_steps = []

    for load_case in load_cases:

        def_grad_aim = load_case.get('def_grad_aim')
        def_grad_rate = load_case.get('def_grad_rate')
        vel_grad = load_case.get('vel_grad')
        stress = load_case.get('stress')
        stress_rate = load_case.get('stress_rate')
        rot = load_case.get('rotation_matrix')
        total_time = load_case['total_time']
        num_increments = load_case['num_increments']
        freq = load_case.get('dump_frequency', 1)

        if sum((x is not None
                for x in (def_grad_aim, def_grad_rate, vel_grad))) > 1:
            msg = ('Specify only one of `def_grad_rate`,  `def_grad_aim` '
                   'and `vel_grad`.')
            raise ValueError(msg)

        # If def_grad_aim/rate is masked array, stress masked array should also be passed,
        # such that the two arrays are component-wise exclusive.

        dg_arr = None
        dg_arr_sym = None
        if def_grad_aim is not None:
            dg_arr = def_grad_aim
            dg_arr_sym = 'F'
        elif def_grad_rate is not None:
            dg_arr = def_grad_rate
            dg_arr_sym = 'dot_F'
        elif vel_grad is not None:
            dg_arr = vel_grad
            dg_arr_sym = 'L'

        stress_arr = None
        stress_arr_sym = None
        if stress is not None:
            stress_arr = stress
            stress_arr_sym = 'P'
        elif stress_rate is not None:
            stress_arr = stress_rate
            stress_arr_sym = 'dot_P'

        # If load case tensors are specified as (nested) lists with fill values, convert
        # to masked arrays:
        if isinstance(dg_arr, list):
            if isinstance(dg_arr[0], list):
                dg_arr = [j for i in dg_arr for j in i]  # flatten
            dg_arr = masked_array_from_list(dg_arr, fill_value='x').reshape((3, 3))

        if isinstance(stress_arr, list):
            if isinstance(stress_arr[0], list):
                stress_arr = [j for i in stress_arr for j in i]  # flatten
            stress_arr = masked_array_from_list(stress_arr, fill_value='x').reshape((3, 3))

        load_step = {
            'boundary_conditions': {
                'mechanical': {}
            }
        }
        bc_mech = load_step['boundary_conditions']['mechanical']

        if stress_arr is None:

            if dg_arr is None:
                msg = ('Specify one of `def_grad_rate`, `def_grad_aim` or '
                       '`vel_grad`.')
                raise ValueError(msg)

            if isinstance(dg_arr, np.ma.core.MaskedArray):
                msg = ('To use mixed boundary conditions, `stress`/`stress_rate` must be '
                       'passed as a masked array.')
                raise ValueError(msg)

            bc_mech[dg_arr_sym] = format_1D_masked_array(dg_arr.flat)

        else:
            if isinstance(stress_arr, np.ma.core.MaskedArray):

                if dg_arr is None:
                    msg = ('Specify one of `def_grad_rate`, `def_grad_aim` or '
                           '`vel_grad`.')
                    raise ValueError(msg)

                msg = ('`def_grad_rate`, `def_grad_aim` or `vel_grad` must be '
                       'component-wise exclusive with `stress` or `stress_rate` (both as '
                       'masked arrays)')
                if not isinstance(dg_arr, np.ma.core.MaskedArray):
                    raise ValueError(msg)
                if np.any(dg_arr.mask == stress_arr.mask):
                    raise ValueError(msg)

                if dg_arr_sym == 'L':
                    if any((sum(row) not in (0, 3) for row in dg_arr.mask)):
                        msg = ('Specify all or no values for each row of '
                               '`vel_grad`')
                        raise ValueError(msg)

                bc_mech[dg_arr_sym] = format_1D_masked_array(dg_arr.flat)
                bc_mech[stress_arr_sym] = format_1D_masked_array(stress_arr.flat)

            else:
                if dg_arr is not None:
                    msg = ('To use mixed boundary conditions, `stress` or `stress_rate`'
                           f'must be passed as a masked array.')
                    raise ValueError(msg)

                bc_mech[stress_arr_sym] = format_1D_masked_array(stress_arr.flat)

        if rot is not None:
            rot = np.array(rot)
            msg = 'Matrix passed as a rotation is not a rotation matrix.'
            if not np.allclose(rot.T @ rot, np.eye(3)):
                raise ValueError(msg)
            if not np.isclose(np.linalg.det(rot), 1):
                raise ValueError(msg)

            rot = Rotation._om2ax(rot)
            rot[3] *= 180 / np.pi

            bc_mech['R'] = rot.tolist()

        load_step['discretization'] = {
            't': total_time,
            'N': num_increments,
        }
        load_step['f_out'] = freq

        load_steps.append(load_step)

    load_data = {
        'solver': {
            'mechanical': 'spectral_basic'
        },
        'loadstep': load_steps
    }

    dir_path = Path(dir_path).resolve()
    load_path = dir_path.joinpath(name)
    yaml = YAML()
    yaml.dump(load_data, load_path)

    return load_path


def write_material(homog_schemes, phases, volume_element, dir_path, name='material.yaml'):
    """Write the material.yaml file for a DAMASK simulation.

    Parameters
    ----------
    homog_schemes : dict
        Dict whose keys are homogenization scheme labels and whose values are dicts that
        specify the homogenization parameters for that scheme. This will be passed into
        the "homogenization" dict in the material file.
    phases : dict
        Dict whose keys are phase labels and whose values are the dicts that specify the
        phase parameters for that phase label. This will be passed into the "phase" dict
        in the material file.
    volume_element : dict
        Volume element data to include in the material file. Allowed keys are:
            orientations : dict
                Dict containing the following keys:
                    type : str
                        One of "euler", "quat".
                    quaternions : ndarray of shape (R, 4) of float, optional
                        Array of R row four-vectors of unit quaternions. Specify either
                        `quaternions` or `euler_angles`.
                    euler_angles : ndarray of shape (R, 3) of float, optional
                        Array of R row three-vectors of Euler angles in degrees or radians,
                        as determined by `euler_degrees`. Specify either `quaternions` or
                        `euler_angles`. Specified as proper Euler angles in the Bunge
                        convention. (Rotations are about Z, new X, new new Z.)
                    euler_degrees : bool, optional
                        If True, `euler_angles` are expected in degrees, rather than
                        radians.
                    unit_cell_alignment : dict
                        Alignment of the unit cell.
            constituent_material_idx : list or ndarray of shape (N,) of int, optional
                Determines the material to which each constituent belongs, where N is the
                number of constituents. If `constituent_*` keys are not specified, then
                `element_material_idx` and `grid_size` must be specified. See Notes.
            constituent_material_fraction: list or ndarray of shape (N,) of float, optional
                The fraction that each constituent occupies within its respective
                material, where N is the number of constituents. If `constituent_*` keys
                are not specified, then `element_material_idx` and `grid_size` must be
                specified. See Notes.
            constituent_phase_label : list or ndarray of shape (N,) of str, optional
                Determines the phase label of each constituent, where N is the number of
                constituents.  If `constituent_*` keys are not specified, then
                `element_material_idx` and `grid_size` must be specified. See Notes.
            constituent_orientation_idx : list or ndarray of shape (N,) of int, optional
                Determines the orientation (as an index into `orientations`) associated
                with each constituent, where N is the number of constituents. If
                `constituent_*` keys are not specified, then `element_material_idx` and
                `grid_size` must be specified. See Notes.
            material_homog : list or ndarray of shape (M,) of str, optional
                Determines the homogenization scheme (from a list of available
                homogenization schemes defined elsewhere) to which each material belongs,
                where M is the number of materials. If `constituent_*` keys are not
                specified, then `element_material_idx` and `grid_size` must be specified.
                See Notes.
            element_material_idx : list or ndarray of shape (P,) of int, optional
                Determines the material to which each geometric model element belongs,
                where P is the number of elements. If `constituent_*` keys are not
                specified, then `element_material_idx` and `grid_size` must be specified.
                See Notes.
            grid_size : list or ndarray of shape (3,) of int, optional
                Geometric model grid dimensions. If `constituent_*` keys are not
                specified, then `element_material_idx` and `grid_size` must be specified.
                See Notes.
            phase_labels : list or ndarray of str, optional
                List of phase labels to associate with the constituents. Only applicable
                if `constituent_*` keys are not specified. The first list element is the
                phase label that will be associated with all of the geometrical elements
                for which an orientation is also specified. Additional list elements are
                phase labels for geometrical elements for which no orientations are
                specified.
            homog_label : str, optional
                The homogenization scheme label to use for all materials in the volume
                element. Only applicable if `constituent_*` keys are not specified.
    dir_path : str or Path
        Directory in which to generate the material.yaml file.
    name : str, optional
        Name of material file to write. By default, set to "material.yaml".

    Returns
    -------
    mat_path : Path
        Path of the generated material.yaml file.

    Notes
    -----
    - A "material" is currently known as a "microstructure" in the DAMASK material.yml
      file. A "material" may have multiple constituents (e.g. grains), modelled together
      under some homogenization scheme. For a full-field simulation, there will be only
      one constituent per "material" (and no associated homogenization).

    - The input `volume_element` can be either fully specified with respect to the
      `constituent_*` keys or, if no `constituent_*` keys are specified, but the
      `element_material_idx` and `grid_size` keys are specified, we assume the model to be
      a full-field model for which each material contains precisely one constituent. In
      this case the additional keys `phase_labels` and `homog_labels` must be specified.
      The number of phase labels specified should be equal to the number or orientations
      specified plus the total number of any additional material indices in
      "element_material_idx" for which there are no orientations.

    """

    materials = get_volume_element_materials(
        volume_element,
        homog_schemes=homog_schemes,
        phases=phases,
        P=-1,  # DAMASK uses P = -1 convention.
    )

    # Only include phases that are used:
    phases = {phase_name: phase_data
              for phase_name, phase_data in phases.items()
              if phase_name in volume_element['constituent_phase_label']}
    
    mat_dat = {
        'phase': phases,
        'homogenization': homog_schemes,
        'material': materials,
    }
    mat_data_fmt = prepare_material_yaml_data(mat_dat)  # e.g. format quats to 15 d.p.

    dir_path = Path(dir_path).resolve()
    mat_path = dir_path.joinpath(name)
    yaml = YAML()
    yaml.dump(mat_data_fmt, mat_path)

    return mat_path


def write_numerics(dir_path, numerics, name='numerics.yaml'):
    """Write the optional numerics.yaml file for a DAMASK simulation.

    Parameters
    ----------
    dir_path : str or Path
        Directory in which to generate the file(s).
    numerics : dict
        Dict of key-value pairs to write into the file.
    name : str, optional
        Name of numerics file to write. By default, set to "numerics.yaml".

    Returns
    -------
    numerics_path : Path
        File path to the generated numerics.yaml file.

    """

    dir_path = Path(dir_path).resolve()
    numerics_path = dir_path.joinpath(name)
    yaml = YAML()
    yaml.dump(numerics, numerics_path)

    return numerics_path
