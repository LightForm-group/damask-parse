from copy import deepcopy
from pathlib import Path

from damask_parse.readers import (
    concat_stdout_arrays,
    read_spectral_stdout,
    split_stdout_arrays,
)
from damask_parse.utils import nested_dicts_equal


from importlib.resources import files


def get_test_data_file_path(name: str) -> Path:
    data_dir = files("tests.data")
    return data_dir / name


def test_read_spectral_stdout_concat_arrays_round_trip():
    path = get_test_data_file_path("stdout.log")
    out = read_spectral_stdout(path, concat_arrays=False)
    out_c = deepcopy(out)
    concat_stdout_arrays(out_c)
    split_stdout_arrays(out_c)
    assert nested_dicts_equal(out, out_c)
