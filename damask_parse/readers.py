"""`damask_parse.readers.py`"""

from pathlib import Path

import pandas
import re
import numpy as np

from damask_parse.utils import get_header

__all__ = [
    'read_table',
    'read_geom',
    'read_spectral_stdout',
    'read_spectral_stderr',
]


def parse_geom_microstructure(ms_str):
    pat = (r'\[[G|g]rain(\d+)\][\s\S]crystallite\s(\d+)[\s\S]\(constituent\)\s+phase\s+'
           r'(\d+)\s+texture\s+(\d+)\s+fraction\s+(\d\.\d+)')
    matches = re.findall(pat, ms_str)
    grains = {}
    for i in matches:
        grains.update({
            int(i[0]): {
                'crystallite_idx': int(i[1]),
                'phase_idx': int(i[2]),
                'texture_idx': int(i[3]),
                'fraction': float(i[4]),
            },
        })

    return grains


def parse_geom_texture(texture_str):
    pat = (r'\[[G|g]rain(\d+)\][\s\S]\(gauss\)\s+phi1\s+(\d+\.?\d+)\s+Phi\s+(\d+\.?\d+)'
           r'\s+phi2\s+(\d+\.?\d+)(?:\s+scatter\s+(\d+\.?\d+)\s+fraction\s+(\d+\.?\d+))?'
           r'(?:\s+)?')
    matches = re.findall(pat, texture_str)

    grains = {}
    for i in matches:
        grain = {
            'phi1': float(i[1]),
            'Phi': float(i[2]),
            'phi2': float(i[3]),
        }
        if i[4]:
            grain.update({
                'fraction': float(i[4]),
            })
        if i[5]:
            grain.update({
                'scatter': float(i[5]),
            })
        grains.update({int(i[0]): grain})

    return grains


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


def read_table(path, use_dataframe=False, combine_array_columns=True,
               ignore_duplicate_cols=False, check_header=True):
    """Read the data from a DAMASK-generated ASCII table file, as generated by
    the DAMASK post-processing command named `postResults`.

    Parameters
    ----------
    path : str or Path
        Path to the DAMASK table file.
    use_dataframe : bool, optional
        If True, a Pandas DataFrame is returned. Otherwise, a dict of Numpy
        arrays is returned. By default, set to False.        
    combine_array_columns : bool, optional
        If True, columns that represent elements of an array (e.g. stress) are
        combined into a single column. By default, set to True.
    ignore_duplicate_cols : bool, optional
        If True, duplicate columns (as detected by the `mangle_dupe_cols`
        option of `pandas_read_csv` function) are ignored. Otherwise, an
        exception is raised. By default, set to False.
    check_header : bool, optional
        Check that the command `postResults` appears in the header, i.e. check
        that the file is indeed likely to be a DAMASK table file. By default,
        set to True.

    Returns
    -------
    outputs : dict
        The data in the table file, represented either as a Pandas DataFrame
        (if `use_dataframe` is True) or a dict of Numpy arrays (if
        `use_dataframe` if False).

    TODO: parse all "array-like" columns in one-dimensional Numpy arrays, and
    then reshape known columns into correct shapes.

    """

    arr_shape_lookup = {
        12: [4, 3],
        9: [3, 3],
        3: [3],
        4: [4],
    }

    header = get_header(path)
    header_num = len(header)

    if check_header:
        if 'postResults' not in header[0]:
            msg = (
                '"postResults" does not appear in the header of the supposed '
                'table file. If you want to ignore this fact, call the '
                '`read_table` function with the parameter '
                '`check_header=False`.'
            )
            raise ValueError(msg)

    df = pandas.read_csv(str(path), delim_whitespace=True, header=header_num)

    if not ignore_duplicate_cols:
        if np.any(df.columns.str.replace(r'(\.\d+)$', '').duplicated()):
            msg = (
                'It appears there are duplicated columns in the table. If you '
                'want to ignore this fact, call the `read_table` function with'
                ' the parameter `ignore_duplicate_cols=True`.'
            )
            raise ValueError(msg)

    arr_sizes = None
    if combine_array_columns or not use_dataframe:
        # Find number of elements for each "array" column:
        arr_sizes = {}
        for header in df.columns.values:
            match = re.match(r'([0-9]+)_(.+)', header)
            if match:
                arr_name = match.group(2)
                if arr_name in arr_sizes:
                    arr_sizes[arr_name] += 1
                else:
                    arr_sizes.update({
                        arr_name: 1
                    })

        # Check for as yet "unsupported" array dimensions:
        bad_num_elems = set(arr_sizes.values()) - set(arr_shape_lookup.keys())
        if len(bad_num_elems) > 0:
            msg = (
                '"Array" columns must have one of the following number of '
                'elements: {}. However, there are columns with the following '
                'numbers of elements: {}'.format(
                    list(arr_shape_lookup.keys()), list(bad_num_elems)
                )
            )
            raise ValueError(msg)

    if combine_array_columns:
        # Add arrays as single columns:
        for arr_name, arr_size in arr_sizes.items():
            arr_idx = ['{}_{}'.format(i, arr_name)
                       for i in range(1, arr_size + 1)]
            df[arr_name] = df[arr_idx].values.tolist()
            # Remove individual array columns:
            df = df.drop(arr_idx, axis=1)

    outputs = df

    if not use_dataframe:
        # Transform each column into a Numpy array:
        arrays = {}
        for header in df.columns.values:
            val = np.array(df[header])

            if header in arr_sizes:
                shp = tuple([-1] + arr_shape_lookup[arr_sizes[header]])
                val = np.array([*df[header]]).reshape(shp)

            arrays.update({
                header: val
            })
            outputs = arrays

    return outputs


def read_geom(geom_path):
    """Parse a DAMASK geometry file.

    Parameters
    ----------
    geom_path : str or Path
        Path to the DAMASK geometry file.

    Returns
    -------
    volume_element : dict
        Dictionary that represents the volume element parsed from the geometry
        file, with keys:
            grain_idx : ndarray of dimension three
                A mapping that determines the grain index for each voxel.
        TODO: finish this

    """

    with Path(geom_path).open('r') as handle:
        lines = handle.read()

        grains = {}
        ori = None
        pat = r'\<microstructure\>[\s\S]*\(constituent\).*'
        ms_match = re.search(pat, lines)
        if ms_match:
            ms_str = ms_match.group()
            grains = parse_geom_microstructure(ms_str)

        pat = r'\<texture\>[\s\S]*\(gauss\).*'
        texture_match = re.search(pat, lines)
        if texture_match:
            texture_str = texture_match.group()
            texture = parse_geom_texture(texture_str)
            for grain_idx, tex in texture.items():
                if grain_idx not in grains:
                    grains[grain_idx] = tex
                else:
                    grains[grain_idx].update(tex)

            # Collect texture orientations into an array:
            grain_keys = sorted(list(grains.keys()))
            if not grain_keys == list(range(min(grain_keys), max(grain_keys) + 1)):
                raise ValueError('Non-consecutive grain numbers.')
            ori = np.zeros((len(grain_keys), 3))
            for grain_idx, v in sorted(grains.items()):
                ori[grain_idx - 1] = [v['phi1'], v['Phi'], v['phi2']]

        com_pat = r'(geom_.*)'
        commands = re.findall(com_pat, lines)

        head_pat = r'(\d+)\sheader'
        num_header = int(re.search(head_pat, lines).group(1))

        grid_pat = r'grid\s+a\s+(\d+)\s+b\s+(\d+)\s+c\s+(\d+)'
        grid_match = re.search(grid_pat, lines)
        grid = None
        if grid_match:
            grid = [int(i) for i in grid_match.groups()]
        else:
            raise ValueError('`grid` not specified in file.')

        size_pat = r'size\s+x\s+(\d+\.\d+)\s+y\s+(\d+\.\d+)\s+z\s+(\d+\.\d+)'
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
        homo = None
        if homo_match:
            homo = int(homo_match.group(1))

        grain_idx_2d = np.zeros((grid[1] * grid[2], grid[0]), dtype=int)
        for ln_idx, ln in enumerate(lines.splitlines()):
            ln_split = ln.strip().split()
            if ln_idx > num_header:
                arr_idx = ln_idx - (num_header + 1)
                grain_idx_2d[arr_idx] = [int(i) for i in ln_split]

        grain_idx = grain_idx_2d.reshape(grid[::-1]).swapaxes(0, 2)

        volume_element = {
            'num_header': num_header,
            'commands': commands,
            'grid': grid,
            'grain_idx': grain_idx,
            'grains': grains,
            'size': size,
            'origin': origin,
            'homogenization': homo,
            'orientations': ori,
        }

    return volume_element


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
