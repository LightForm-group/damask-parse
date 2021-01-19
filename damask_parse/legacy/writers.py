"""`damask_parse.writers.py`"""

import copy
from pathlib import Path
from collections import OrderedDict
import numpy as np

from damask_parse.utils import zeropad, align_orientations


__all__ = [
    'write_material_config',
    'write_numerics_config',
]


def write_material_config(homog_schemes, phases, dir_path, volume_element=None,
                          separate_parts=False, part_paths=None, homog_labels=None,
                          texture_alignment_method='axes_keyword'):
    """Write the material.config file for a DAMASK simulation.

    Parameters
    ----------
    homog_schemes : dict
        Dict whose keys are homogenization scheme labels and whose values are dicts that
        specify the homogenization parameters for that scheme.
    phases : dict
        Dict whose keys are phase labels and whose values are the dicts that specify the
        phase parameters for that phase label. If `volume_element` is not specified,
        it is preferable that this is an OrderedDict, rather than a standard dict, since
        that will guarantee the order in which the phases are written. This is important
        because phases are reference by (one-indexed) integer index in the microstructure
        part.
    dir_path : str or Path
        Directory in which to generate the file(s).
    volume_element : dict, optional
        Dictionary that represents the volume element. If not specified, `part_paths` must
        contain at least keys "microstructure" and "texture". If specified, must have at
        least keys (unless indicated as optional):
            voxel_grain_idx : 3D ndarray of int
                A mapping that determines the grain index for each voxel.
            voxel_homogenization_idx : 3D ndarray of int
                A mapping that determines the homogenization scheme (via an integer index)
                for each voxel.
            orientations : dict
                Dict containing the following keys:
                    euler_angles : ndarray of shape (N, 3)
                        Array of N row vectors of Euler angles.
                    euler_degrees : bool
                        If True, `euler_angles` are expected in degrees, rather than
                        radians.
                    euler_angle_labels : list of str
                        Labels of the Euler angles.
                    unit_cell_alignment : dict
                        Alignment of the unit cell.
            grain_phase_label_idx : 1D ndarray of int
                Zero-indexed integer index array mapping a grain to its phase.
            grain_orientation_idx : 1D ndarray of int
                Zero-indexed integer index array mapping a grain to its orientation.
            phase_labels : ndarray of str
                String array of phase labels.
            orientation_coordinate_system : dict, optional
                This dict allows assigning orientation coordinate system directions to
                sample directions. Allowed keys are 'x', 'y' and 'z'. Example values are
                'RD', 'TD' and 'ND'.
            model_coordinate_system : dict, optional
                This dict allows assigning model geometry coordinate system directions to
                sample directions. Allowed keys are 'x', 'y' and 'z'. Example values are
                'RD', 'TD' and 'ND'.
    separate_parts : bool, optional
        Applicable only if `volume_element` is specified. If True, microstructure and
        texture parts will be written in separate files and linked within the material
        config file. By default, False.
    part_paths : dict of (str : str), optional
        Keys are material config part names and values are paths to files containing the
        configuration for that part. If `volume_element` is not specified, `part_paths`
        must contain at least the keys: "microstructure" and "texture".
    homog_labels : list of str, optional
        List of homogenization scheme labels that maps a homogenization scheme from
        `homog_schemes` to a homogenization index (which is defined on each volume element
        voxel.) The list should be the same length as the number of homogenization schemes
        defined in the volume_element (i.e. the number of unique indices in the
        `voxel_homogenization_idx` array of the volume element). If not specified, there
        must be only one homogenization scheme in `homog_schemes`.
    texture_alignment_method : str, optional
        Applicable only if `volume_element` is specified. Either "axes_keyword" or
        "rotation". If both the orientation coordinate system and the model coordinate
        system are, within the volume element, defined and distinct, this specifies how
        the two should be aligned. If "axes_keyword", the DAMASK "axes" key will be used;
        if "rotation", the euler angles will be directly rotated. By default,
        "axes_keyword".

    Returns
    -------
    mat_conf_path : Path
        Path of the generated material.config file.

    """

    def format_part_name(part_name, include_delim=True):
        part_delim = '#-------------------#'
        part_name_lines = [f'<{part_name}>']
        if include_delim:
            part_name_lines = [part_delim] + part_name_lines + [part_delim]
        return part_name_lines

    def get_part_lines(part_data, section_linebreak=True, section_order=None):

        if not section_order:
            section_order = list(part_data.keys())

        if set(section_order) != set(part_data):
            msg = (
                f'`section_order` must be the list of section names '
                f'{list(part_data.keys())}, ordered in the desired manner, but the '
                f'following was passed: {section_order}.'
            )
            raise ValueError(msg)

        lns = []
        for section_label in section_order:

            lns.append(f'[{section_label}]')

            section = copy.deepcopy(part_data[section_label])
            outputs = section.pop('outputs', [])
            flags = section.pop('flags', [])

            for key, val in sorted(section.items()):
                lns.append(f'{key:<30s}{val}')
            for sec_flag in sorted(flags):
                lns.append(f'/{sec_flag}/')
            for sec_out in sorted(outputs):
                lns.append(f'{"(output)":<30s}{sec_out}')
            if section_linebreak:
                lns.append('')
        lns.append('')

        return lns

    if not homog_schemes:
        raise ValueError('Specify at least one homogenization scheme.')

    if not phases:
        raise ValueError('Specify at least one phase.')

    if texture_alignment_method not in ['axes_keyword', 'rotation']:
        msg = '`texture_alignment_method` must be either "axes_keyword" or "rotation".'
        raise ValueError(msg)

    if not homog_labels:
        if len(homog_schemes) > 1:
            msg = (f'If `homog_labels` is not specified, there must be only one '
                   f'homogenization scheme in `homog_schemes`, but '
                   f'`homog_schemes is: {homog_schemes}')
            raise ValueError(msg)
        else:
            homog_labels = np.array(list(homog_schemes.keys()))

    # Check homog_labels are in homog_schemes:
    for homog_label in homog_labels:
        if homog_label not in homog_schemes:
            msg = (f'Homogenization label "{homog_label}" is not defined in '
                   f'`homog_schemes`.')
            raise ValueError(msg)

    # Check microstructure and texture are specified:
    part_paths = part_paths or {}
    bad_inputs = False
    if volume_element is None:
        if any([part_paths.get(i) is None for i in ['microstructure', 'texture']]):
            bad_inputs = True
    else:
        if any([part_paths.get(i) for i in ['microstructure', 'texture']]):
            bad_inputs = True
    if bad_inputs:
        msg = ('Specify either `volume_element` or specify file paths to the '
               '"microstructure" and "texture" configurations in the `part_paths` '
               'argument.')
        raise ValueError(msg)

    # Get parts paths relative to this file:
    for key, val in part_paths.items():
        val = Path(val)
        if val.is_absolute():
            val = val.relative_to(dir_path)
        part_paths[key] = './' + val.as_posix()

    dir_path = Path(dir_path).resolve()
    mat_conf_path = dir_path.joinpath('material.config')

    homog_lns = get_part_lines(homog_schemes, section_order=list(homog_labels))
    crystallite_lns = get_part_lines({'dummy': {}})

    phase_order = list(volume_element['phase_labels']) if volume_element else None
    phase_lns = get_part_lines(phases, section_order=phase_order)

    mat_conf_lns = (
        format_part_name('Homogenization') + homog_lns +
        format_part_name('Crystallite') + crystallite_lns +
        format_part_name('Phase') + phase_lns
    )

    if volume_element:

        ori = volume_element['orientations']

        if ori['unit_cell_alignment']['x'] != 'a':
            msg = (f'Orientations unit cell alignment must be DAMASK-compatible: x '
                   f'parallel to a, but `unit_cell_alignment` is '
                   f'{ori["unit_cell_alignment"]}.')
            raise NotImplementedError(msg)

        euler_angles = ori['euler_angles'].copy()
        if not ori['euler_degrees']:
            euler_angles = np.rad2deg(euler_angles)

        axes = None
        ori_CS = volume_element.get('orientation_coordinate_system')
        model_CS = volume_element.get('model_coordinate_system')

        if (ori_CS and model_CS) and (ori_CS != model_CS):

            if texture_alignment_method == 'axes_keyword':
                OCS_inv = {v: k for k, v in ori_CS.items()}
                axes = [('+' if not OCS_inv[v].startswith('-') else '') + OCS_inv[v]
                        for v in model_CS.values()]

            elif texture_alignment_method == 'rotation':
                align_orientations(euler_angles, ori_CS, model_CS)

        texture_data = OrderedDict()
        for ori_idx, euler in enumerate(euler_angles):
            ori_name = 'Orientation' + zeropad(ori_idx + 1, len(euler_angles))
            ori_data = {
                '(gauss)': (f'phi1 {euler[0]:8.4f} Phi {euler[1]:8.4f} '
                            f'phi2 {euler[2]:8.4f} scatter 0.0 fraction 1.0'),
            }
            if axes:
                ori_data.update({'axes': f'{axes[0]} {axes[1]} {axes[2]}'})
            texture_data[ori_name] = ori_data
        texture_lns = get_part_lines(texture_data, section_linebreak=False)

        microstructure_data = OrderedDict()
        all_phase_idx = volume_element['grain_phase_label_idx']
        all_ori_idx = volume_element['grain_orientation_idx']
        for grain_idx, (phase_idx, ori_idx) in enumerate(zip(all_phase_idx, all_ori_idx)):
            grain_name = 'Grain' + zeropad(grain_idx + 1, len(all_phase_idx))
            grain_data = {
                'crystallite': '1',
                '(constituent)': (f'phase {phase_idx + 1} '
                                  f'texture {ori_idx + 1} '
                                  f'fraction 1.0'),
            }
            microstructure_data[grain_name] = grain_data
        microstructure_lns = get_part_lines(microstructure_data, section_linebreak=False)

        if separate_parts:

            ms_path = dir_path.joinpath('microstructure.txt')
            with ms_path.open('w') as handle:
                handle.write('\n'.join(microstructure_lns) + '\n')

            mat_conf_lns += format_part_name('Microstructure')
            mat_conf_lns += [f'{{./{ms_path.relative_to(dir_path).as_posix()}}}', '']

            texture_path = dir_path.joinpath('texture.txt')
            with texture_path.open('w') as handle:
                handle.write('\n'.join(texture_lns) + '\n')

            mat_conf_lns += format_part_name('Texture')
            mat_conf_lns += [f'{{./{texture_path.relative_to(dir_path).as_posix()}}}', '']

        else:
            mat_conf_lns += format_part_name('Microstructure') + microstructure_lns
            mat_conf_lns += format_part_name('Texture') + texture_lns

    else:
        for part_name, part_path in part_paths.items():
            mat_conf_lns += format_part_name(part_name.capitalize())
            mat_conf_lns += [f'{{{part_path}}}', '']

    with mat_conf_path.open('w') as handle:
        handle.write('\n'.join(mat_conf_lns) + '\n')

    return mat_conf_path


def write_numerics_config(dir_path, numerics):
    """Write the optional numerics.config file for a DAMASK simulation.

    Parameters
    ----------
    dir_path : str or Path
        Directory in which to generate the file(s).
    numerics : dict
        Dict of key-value pairs to write into the file.

    Returns
    -------
    numerics_path : Path
        File path to the generated numerics.config file.

    """

    dir_path = Path(dir_path).resolve()
    numerics_path = dir_path.joinpath('numerics.config')

    with numerics_path.open('w') as handle:
        for key, val in numerics.items():
            handle.write(f'{key:<30} {val}')

    return numerics_path
