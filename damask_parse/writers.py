"""`damask_parse.writers.py`"""

from pathlib import Path

import numpy as np

from damask_parse.utils import zeropad, format_1D_masked_array

__all__ = [
    'write_geom',
    'write_material_config',
    'write_load_case',
]


def write_geom(volume_element, geom_path):
    """Write the geometry file for a spectral DAMASK simulation.

    Parameters
    ----------
    volume_element : dict
        Dict that represents the specification of a volume element, with keys:
            grain_idx : nested list or ndarray of dimension three
                A mapping that determines the grain index for each voxel.
            size : list of length three, optional
                Volume element size. By default set, to unit size: `[1, 1, 1]`.
    geom_path : str or Path
        The path to the file that will be generated.

    Returns
    -------
    geom_path : Path
        The path to the generated file.

    """

    grain_idx = volume_element['grain_idx']
    if isinstance(grain_idx, list):
        grain_idx = np.array(grain_idx)

    shape = grain_idx.shape
    grain_idx_2d = np.concatenate(grain_idx.swapaxes(0, 2))
    ve_size = volume_element.get('size', [1, 1, 1])
    ve_origin = volume_element.get('origin', [0, 0, 0])

    num_header_lns = 5
    homog_idx = 1

    header = (
        '{} header\n'
        'grid a {} b {} c {}\n'
        'size x {} y {} z {}\n'
        'origin x {} y {} z {}\n'
        'microstructures {}\n'
        'homogenization {}\n'
    ).format(
        num_header_lns,
        *shape,
        *ve_size,
        *ve_origin,
        np.max(grain_idx_2d),
        homog_idx
    )

    arr_str = ''
    for row in grain_idx_2d:
        for col in row:
            arr_str += '{:<5d}'.format(col)
        arr_str += '\n'

    geom_path = Path(geom_path)
    with geom_path.open('w') as handle:
        handle.write(header)
        handle.write(arr_str)

    return geom_path


def write_material_config(material, dir_path, volume_element=None, part_paths=None,
                          name='material.config'):
    """Write the material.config file for a DAMASK simulation.

    Parameters
    ----------
    material : dict
        Dict specify DAMASK material parameters. Key are:
            homogenization : list of dict
            crystallite : list of dict
            phase : list of dict
    dir_path : str or Path
        Directory in which to generate the file.
    volume_element : dict, optional
        Dict that represents the specification of a volume element. If not specified,
        `part_paths` must contain at least keys "microstructure" and "texture". Keys are:
            grain_idx : nested list or ndarray of dimension three
                A mapping that determines the grain index for each voxel.
            size : list of length three, optional
                Volume element size. By default set, to unit size: `[1, 1, 1]`.
    part_paths : dict of (str : str)
        Keys are parts and values are paths to files containing the configuration for that
        part. If `volume_element` is not specified, `part_paths` must contain at least the
        keys: "microstructure" and "texture".

    Returns
    -------
    path : Path
        Path of the generated file.

    """

    def format_part_name(name):
        part_delim = '#-------------------#'
        return f'{part_delim}\n<{name}>\n{part_delim}\n'

    part_paths = part_paths or {}

    bad_inputs = False
    if volume_element is None:
        if any([part_paths.get(i) is None for i in ['Microstructure', 'Texture']]):
            bad_inputs = True
    else:
        if any([part_paths.get(i) for i in ['Microstructure', 'Texture']]):
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
    path = dir_path.joinpath(name)
    with path.open('w') as handle:

        for part_name, val in material.items():

            handle.write(format_part_name(part_name))

            for section in val:

                sec_name = section['name']
                sec_keys = section.get('keys')
                sec_outs = section.get('outputs')
                sec_flags = section.get('flags')

                handle.write(f'[{sec_name}]\n')

                if sec_keys is not None:
                    for sec_key, sec_keyval in sorted(sec_keys.items()):
                        handle.write(f'{sec_key:<30s}{sec_keyval}\n')

                if sec_outs is not None:
                    for sec_out in sorted(sec_outs):
                        handle.write(f'{"(output)":<30s}{sec_out}\n')

                if sec_flags is not None:
                    for sec_flag in sorted(sec_flags):
                        handle.write(f'/{sec_flag}/\n')

                handle.write('\n')

            handle.write('\n')

        for part_name, part_path in part_paths.items():
            handle.write(format_part_name(part_name))
            handle.write('{{{}}}\n\n'.format(part_path))

        if volume_element:

            # For now, the "Microstructure" part is trivial: a list of Grains, each of
            # which contains one "crystallite" that consists of one phase.

            ori = volume_element['orientations']
            if isinstance(ori, list):
                ori = np.array(ori)

            num_grains = ori.shape[0]
            handle.write(format_part_name('Microstructure'))
            for i in range(num_grains):
                grain_idx = zeropad(i + 1, num_grains)
                grain_spec = (
                    f'[Grain{grain_idx}]\n'
                    f'crystallite 1\n'
                    f'(constituent) phase 1 texture {grain_idx} fraction 1.0\n'
                )
                handle.write(grain_spec)
            handle.write('\n')

            handle.write(format_part_name('Texture'))
            for i in range(num_grains):
                grain_idx = zeropad(i + 1, num_grains)
                tex_spec = (
                    f'[Grain{grain_idx}]\n'
                    f'(gauss) phi1 {ori[i][0]:8.4f} Phi {ori[i][1]:8.4f} '
                    f'phi2 {ori[i][2]:8.4f} scatter 0.0 fraction 1.0\n'
                )
                handle.write(tex_spec)

    return path


def write_load_case(load_path, load_cases):
    """

    Example load case line is: 
        fdot 1.0e-3 0 0  0 * 0  0 0 * stress * * *  * 0 *   * * 0  time 10  incs 40

    """

    all_load_case = []

    for load_case in load_cases:

        def_grad_aim = load_case.get('def_grad_aim')
        def_grad_rate = load_case.get('def_grad_rate')
        stress = load_case.get('stress')
        total_time = load_case['total_time']
        num_increments = load_case['num_increments']

        if def_grad_aim is not None and def_grad_rate is not None:
            msg = 'Specify only one of `def_grad_rate` and `def_grad_aim`.'
            raise ValueError(msg)

        stress_symbol = 'P'

        # If def_grad_aim/rate is masked array, stress masked array should also be passed,
        # such that the two arrays are component-wise exclusive.

        dg_arr = None
        dg_arr_sym = None
        if def_grad_aim is not None:
            dg_arr = def_grad_aim
            dg_arr_sym = 'F'
        elif def_grad_rate is not None:
            dg_arr = def_grad_rate
            dg_arr_sym = 'Fdot'

        load_case_ln = []

        if stress is None:

            if dg_arr is None:
                msg = 'Specify one of `def_grad_rate` or `def_grad_aim.'
                raise ValueError(msg)

            if isinstance(dg_arr, np.ma.core.MaskedArray):
                msg = ('To use mixed boundary conditions, `stress` must be passed as a '
                       'masked array.')
                raise ValueError(msg)

            dg_arr_fmt = format_1D_masked_array(dg_arr.flatten())
            load_case_ln.append(dg_arr_sym + ' ' + dg_arr_fmt)

        else:
            if isinstance(stress, np.ma.core.MaskedArray):

                if dg_arr is None:
                    msg = 'Specify one of `def_grad_rate` or `def_grad_aim.'
                    raise ValueError(msg)

                msg = ('`def_grad_rate` or `def_grad_aim` must be component-wise exclusive '
                       'with `stress` (both as masked arrays)')
                if not isinstance(dg_arr, np.ma.core.MaskedArray):
                    raise ValueError(msg)
                if np.any(dg_arr.mask == stress.mask):
                    raise ValueError(msg)

                dg_arr_fmt = format_1D_masked_array(dg_arr.flatten(), fill_symbol='*')
                stress_arr_fmt = format_1D_masked_array(stress.flatten(), fill_symbol='*')
                load_case_ln.extend([
                    dg_arr_sym + ' ' + dg_arr_fmt,
                    stress_symbol + ' ' + stress_arr_fmt,
                ])

            else:
                if dg_arr is not None:
                    msg = ('To use mixed boundary conditions, `stress` must be passed as a '
                           'masked array.')
                    raise ValueError(msg)

                stress_arr_fmt = format_1D_masked_array(stress.flatten())
                load_case_ln.append(stress_symbol + ' ' + stress_arr_fmt)

        load_case_ln.extend([
            f't {total_time}',
            f'incs {num_increments}'
        ])

        load_case_str = ' '.join(load_case_ln)
        all_load_case.append(load_case_str)

    all_load_case_str = '\n'.join(all_load_case)

    load_path = Path(load_path)
    with load_path.open('w') as handle:
        handle.write(all_load_case_str)

    return load_path
