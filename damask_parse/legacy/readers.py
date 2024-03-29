"""`damask_parse.readers.py`"""

import pandas
import re
import numpy as np

from damask_parse.utils import get_header_lines

__all__ = [
    'read_table',
    'parse_microstructure',
    'parse_texture_gauss',
    'read_geom',
]


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

    header = get_header_lines(path)
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


def parse_microstructure(ms_str):
    """Parse a DAMASK microstructure definition from within a string.

    Parameters
    ----------
    ms_str : str
        String that contains a microstructure part.

    Returns
    -------
    microstructure : dict
        Dict representing the parsed microstructure; with the following keys:
            phase_idx : ndarray of int
                Zero-indexed integer index array of phases.
            texture_idx : ndarray of int
                Zero-indexed integer index array of orientations.
            fraction : ndarray of float

    Notes
    -----
    This should work for microstructure parts generated from geom_fromVoronoi in DAMASK
    version 2 and 3. In version 3, the vestigial "crystallite 1" in each microstructure
    section is omitted.

    """

    pat = (r'\[[G|g]rain(\d+)\][\s\S](?:crystallite\s(?:\d+)[\s\S])?\(constituent\)\s+'
           r'phase\s+(\d+)\s+texture\s+(\d+)\s+fraction\s+(\d\.\d+)')
    matches = re.findall(pat, ms_str)
    if not matches:
        raise ValueError('No DAMASK microstructure part found in the string.')

    phase_idx = []
    texture_idx = []
    fraction = []
    for i in matches:
        phase_idx.append(int(i[1]))
        texture_idx.append(int(i[2]))
        fraction.append(float(i[3]))

    microstructure = {
        'phase_idx': np.array(phase_idx) - 1,
        'texture_idx': np.array(texture_idx) - 1,
        'fraction': np.array(fraction),
    }

    return microstructure


def parse_texture_gauss(texture_str):
    """Parse a DAMASK Gauss texture definition from within a string.

    Parameters
    ----------
    texture_str : str
        String that contains a "Gauss" texture part (i.e. euler angles for each grain).

    Returns
    -------
    texture : dict
        Dict representing the parsed texture; with the following keys:
            euler_angles : ndarray of shape (N, 3)
                Array of row vectors of Euler angles.
            euler_angle_labels : list of str
                Labels of the Euler angles.
            euler_degrees : bool, True
                Signifies that `euler_angles` are represented in degrees.
            fraction : ndarray or NoneType
            scatter : ndarray or NoneType
    """

    pat = (r'\[[G|g]rain(\d+)\][\s\S]\(gauss\)\s+phi1\s+(\d+\.?\d+)\s+Phi\s+(\d+\.?\d+)'
           r'\s+phi2\s+(\d+\.?\d+)(?:\s+scatter\s+(\d+\.?\d+)\s+fraction\s+(\d+\.?\d+))?'
           r'(?:\s+)?')
    matches = re.findall(pat, texture_str)
    if not matches:
        raise ValueError('No DAMASK texture part found in the string.')

    eulers = []
    fraction = []
    scatter = []
    for texture_section in matches:
        eulers.append([float(texture_section[i]) for i in [1, 2, 3]])
        if texture_section[4]:
            fraction.append(float(texture_section[4]))
        if texture_section[5]:
            scatter.append(float(texture_section[5]))

    texture = {
        'euler_angles': np.array(eulers),
        'euler_degrees': True,
        'euler_angle_labels': ['phi1', 'Phi', 'phi2'],
        'fraction': np.array(fraction) if fraction else None,
        'scatter': np.array(scatter) if scatter else None,
    }

    return texture


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
