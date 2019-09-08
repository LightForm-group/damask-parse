"""`damask_parse.writers.py`"""

from pathlib import Path

import numpy as np

from damask_parse.utils import zeropad

__all__ = [
    'write_geom',
    'write_material_config',
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

    num_header_lns = 3
    homog_idx = 1

    header = (
        '{} header\n'
        'grid a {} b {} c {}\n'
        'size x {} y {} z {}\n'
        'homogenization {}\n'
    ).format(
        num_header_lns,
        *shape,
        *ve_size,
        homog_idx
    )

    arr_str = ''
    for row in grain_idx_2d:
        for col in row:
            arr_str = '{:<5d}'.format(col)
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

                handle.write(f'[{sec_name}]\n')

                if sec_keys is not None:
                    for sec_key, sec_keyval in sorted(sec_keys.items()):
                        handle.write(f'{sec_key:<30s}{sec_keyval}\n')

                if sec_outs is not None:
                    for sec_out in sorted(sec_outs):
                        handle.write(f'{"(output)":<30s}{sec_out}\n')

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
