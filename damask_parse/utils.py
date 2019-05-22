"""`damask_parse.utils.py`"""

import re
import numpy as np


def resolve_dataframe_arrays(dataframe):
    """Convert a Pandas DataFrame into a dictionary of Numpy arrays, where
    arrays originally expressed over multiple columns in the format
    <idx>_<name> are resolved into the correct shape (for a limited subset
    of possible shapes).

    TODO: are there different types of DataFrame objects? Do we need to specify
    what type of DataFrame this works with?

    TODO: replace f-strings with normal `format` function, since we want to
    support Python 3.5.    

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    arrays : dict
        Dictionary with keys being column headers from input DataFrame, and
        values being Numpy arrays.

    """

    arr_sizes = {}
    headers = dataframe.columns.values
    for header in headers:
        if re.match(r'[0-9]_', header):
            arr_name = header[2:]
            if arr_name in arr_sizes:
                arr_sizes[arr_name] += 1
            else:
                arr_sizes.update({
                    arr_name: 1
                })

    arr_shape_lookup = {
        9: [3, 3],
        3: [3],
    }

    # Add arrays as single columns
    for arr_name, arr_size in arr_sizes.items():
        arr_idx = [f'{i}_{arr_name}' for i in range(1, arr_size + 1)]
        dataframe[arr_name] = dataframe[arr_idx].values.tolist()
        # Remove individual array columns:
        dataframe = dataframe.drop(arr_idx, axis=1)

    headers = dataframe.columns.values

    arrays = {}
    # Transform each column into a numpy array:
    for header in dataframe.columns.values:
        val = np.array(dataframe[header])

        if header in arr_sizes:
            shp = tuple([-1] + arr_shape_lookup[arr_sizes[header]])
            val = np.array([*dataframe[header]]).reshape(shp)

        arrays.update({
            header: val
        })

    return arrays


def zeropad(num, largest):
    """Return a zero-padded string of a number, given the largest number.

    TODO: replace f-strings with normal `format` function, since we want to
    support Python 3.5.

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
    num_digits = len(f'{largest:.0f}')
    padded = f'{num:0{num_digits}}'
    return padded
