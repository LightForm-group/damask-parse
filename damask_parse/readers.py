"""`damask_parse.readers.py`"""

from pathlib import Path

import re
from typing import Any
import warnings
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
    "read_geom",
    "read_spectral_stdout",
    "read_spectral_stderr",
    "read_HDF5_file",
    "read_material",
    "geom_to_volume_element",
]


INC_DELIM = r"\s#{75}"
ITER_DELIM = r"\s={75}"
INC_HEADER_PATTERN = re.compile(
    r"Time\s+([\d.E+-]+)s:\s+Increment\s+(\d+)/(\d+)-(\d+)/(\d+)\s+"
    r"of load case\s+(\d+)/(\d+)"
)
STAGGERED_ITER_PATTERN = re.compile(r"Staggered Iteration\s+(\d+)")
MECHANICAL_ITER_REPORT_PATTERN = re.compile(r"deformation gradient aim\s+=")
THERMAL_ITER_REPORT_PATTERN = re.compile(r"thermal conduction converged")
FLOAT_PATTERN = r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"
THERMAL_DAT_PATTERN = re.compile(
    r"Minimum\|Maximum\|Delta Temperature\s*/\s*K\s*=\s*"
    r"(" + FLOAT_PATTERN + r")\s+(" + FLOAT_PATTERN + r")\s+(" + FLOAT_PATTERN + r")"
)
ITER_HEADER_PAT = re.compile(
    r"Increment\s+(\d+)/(\d+)-(\d+)/(\d+)\s+@\s+Iteration\s+(\d+)≤(\d+)≤(\d+)"
)
INC_CONV_PAT = re.compile(r"increment\s+\d+\s+converged")
START_DATE_PAT = re.compile(r"\s*Date:\s*(\d{2}/\d{2}/\d{4})\s*")
START_TIME_PAT = re.compile(r"\s*Time:\s*(\d{2}:\d{2}:\d{2})\s*")
TERM_TIME_PAT = re.compile(
    r"DAMASK terminated on:\s*"
    r"\n\s*Date:\s*(\d{2}/\d{2}/\d{4})\s*"
    r"\n\s*Time:\s*(\d{2}:\d{2}:\d{2})"
)


def read_spectral_stdout(
    path: str | Path, encoding: str = "utf8", concat_arrays: bool = True
) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding=encoding) as handle:
        lines = handle.read()

    incs_split = re.split(INC_DELIM, lines)
    inc_dat_strs = incs_split[1:-1]  # first and last splits not inc data
    inc_dat_list = []
    for inc_str_i in inc_dat_strs:
        inc_dat = parse_increment(inc_str_i)
        inc_dat_list.append(inc_dat)

    # might have an incomplete increment at the end (non-converged or killed):
    if INC_HEADER_PATTERN.search(incs_split[-1]):
        inc_dat_list.append(parse_increment(incs_split[-1]))

    start_time_match = START_TIME_PAT.search(incs_split[0])
    start_date_match = START_DATE_PAT.search(incs_split[0])

    if end_date_time_match := TERM_TIME_PAT.search(incs_split[-1]):
        # simulation might have been killed before completion
        end_date = end_date_time_match.group(1)
        end_time = end_date_time_match.group(2)
    else:
        end_date = None
        end_time = None

    out = {
        "increments": inc_dat_list,
        "start_time": start_time_match.group(1),
        "start_date": start_date_match.group(1),
        "end_date": end_date,
        "end_time": end_time,
    }
    if concat_arrays:
        concat_stdout_arrays(out)

    return out


def parse_increment(inc_str: str) -> dict[str, Any]:
    """Parse an increment from DAMASK's standard output stream.

    An increment includes a header that states the time, increment number (and how many
    total increments are to be simulated in the load case), and the load case number (and
    how many total load cases are to be simulated). The header is followed by one or more
    iterations, which include the deformation gradient and Piola-Kirchhoff stress.

    Note that the same increment may be repeated multiple times in the output stream if
    the increment did not converge, and the increment was cut back (i.e. the time step was
    reduced and the increment was re-attempted).

    Each increment should end in either "converged" or "cutting back" (and possibly a
    message about saving results).

    In the case of multiple active fields (e.g. mechanical and thermal), DAMASK uses a
    staggered iteration approach; this is reflected in the standard output stream with a
    "Staggered Iteration N" line separating different staggered iteration indices.

    """
    inc_header_match = INC_HEADER_PATTERN.search(inc_str)
    time = float(inc_header_match.group(1))
    inc_num = int(inc_header_match.group(2))
    inc_total = int(inc_header_match.group(3))
    cut_back_num = int(inc_header_match.group(4))
    cut_back_total = int(inc_header_match.group(5))
    load_case_num = int(inc_header_match.group(6))
    load_case_total = int(inc_header_match.group(7))

    iter_dat_str = re.split(ITER_DELIM, inc_str)
    stag_iter_idx = None
    mech_iters = []
    therm_iters = []
    for dat_str_i in iter_dat_str:
        if stag_iter_match := STAGGERED_ITER_PATTERN.search(dat_str_i):
            stag_iter_idx = int(stag_iter_match.group(1))
        if MECHANICAL_ITER_REPORT_PATTERN.search(dat_str_i):
            mech_iter_dat = parse_mechanical_iteration(dat_str_i, stag_iter_idx)
            mech_iters.append(mech_iter_dat)
        elif THERMAL_ITER_REPORT_PATTERN.search(dat_str_i):
            thermal_iter_dat = parse_thermal_iteration(dat_str_i, stag_iter_idx)
            therm_iters.append(thermal_iter_dat)

    is_converged = bool(INC_CONV_PAT.search(inc_str))

    warn_msg = r"│\s+warning\s+│\s+│\s+(\d+)\s+│\s+├─+┤\s+│(.*)│\s+\s+│(.*)│"
    warnings_matches = re.findall(warn_msg, inc_str)
    warnings = [
        {
            "code": int(i[0]),
            "message": i[1].strip() + " " + i[2].strip(),
        }
        for i in warnings_matches
    ]

    return {
        "inc": inc_num,
        "inc_total": inc_total,
        "time": time,
        "cut_back_num": cut_back_num,
        "cut_back_total": cut_back_total,
        "load_case_num": load_case_num,
        "load_case_total": load_case_total,
        "mechanical_iterations": mech_iters,
        "thermal_iterations": therm_iters,
        "is_converged": is_converged,
        "warnings": warnings,
    }


def parse_thermal_iteration(
    inc_iter_str: str, stag_iter_idx: int | None = None
) -> dict[str, Any]:
    """Parse a thermal iteration from an increment in DAMASK's standard output stream."""
    match = THERMAL_DAT_PATTERN.search(inc_iter_str)
    return {
        "minimum_temperature_K": float(match.group(1)),
        "maximum_temperature_K": float(match.group(2)),
        "delta_temperature_K": float(match.group(3)),
        "staggered_iter_idx": stag_iter_idx,
    }


def parse_mechanical_iteration(
    inc_iter_str: str, stag_iter_idx: int | None = None
) -> dict[str, Any]:
    """Parse a mechanical iteration from an increment in DAMASK's standard output stream."""
    FLOAT_PAT = r"-?\d+\.\d+"
    SCI_FLOAT_PAT = r"-?\d+\.\d+E[+|-]\d+"
    DG_PAT = r"deformation gradient aim\s+=\n(\s+(?:(?:" + FLOAT_PAT + r"\s+){3}){3})"
    PK_PAT = (
        r"Piola--Kirchhoff stress\s+\/\s.*=\n(\s+(?:(?:" + FLOAT_PAT + r"\s+){3}){3})"
    )

    if iter_head_match := ITER_HEADER_PAT.search(inc_iter_str):
        iter_min = int(iter_head_match.group(5))
        iter_num = int(iter_head_match.group(6))
        iter_max = int(iter_head_match.group(7))
    else:
        warnings.warn(
            "Unable to parse iteration number and bounds from iteration string."
        )
        iter_min = None
        iter_num = None
        iter_max = None

    dg_match = re.search(DG_PAT, inc_iter_str)
    dg_str = dg_match.group(1)
    dg = np.array([float(i) for i in dg_str.split()]).reshape((3, 3))

    pk_matches = re.findall(PK_PAT, inc_iter_str)
    pk_tensors = []
    for pk_str in pk_matches:
        pk = np.array([float(x) for x in pk_str.split()]).reshape((3, 3))
        pk_tensors.append(pk)

    ERR_PAT = (
        r"error (.*)\s+=\s+(-?\d+\.\d+)\s\(("
        + SCI_FLOAT_PAT
        + r")\s(.*),\s+tol\s+=\s+("
        + SCI_FLOAT_PAT
        + r")\)"
    )
    err_matches = re.findall(ERR_PAT, inc_iter_str)
    converge_err = {}
    for i in err_matches:
        err_key = "error_" + i[0].strip().replace(" ", "_")
        converge_err.update(
            {
                err_key: {
                    "value": float(i[2]),
                    "unit": i[3].strip(),
                    "tol": float(i[4]),
                    "relative": float(i[1]),
                }
            }
        )

    inc_iter = {
        "iter_num": iter_num,
        "iter_min": iter_min,
        "iter_max": iter_max,
        "deformation_gradient_aim": dg,
        "piola_kirchhoff_stress": pk_tensors,
        "staggered_iter_idx": stag_iter_idx,
        **converge_err,
    }

    return inc_iter


def concat_stdout_arrays(stdout_dat: dict[str, Any]) -> None:
    """Concatenate all Piola-Kirchhoff stress and deformation gradient tensors parsed
    from `read_spectral_stdout` into two arrays of shape (N, 3, 3), where N is the total
    number of arrays across all increments.

    """

    pk_arrs = []
    dg_arrs = []
    for inc_dat in stdout_dat["increments"]:
        for mech_iter in inc_dat["mechanical_iterations"]:
            dg_arr_idx = len(dg_arrs)
            pk_arr_idx = len(pk_arrs)
            mech_iter["deformation_gradient_aim_idx"] = dg_arr_idx
            mech_iter["piola_kirchhoff_stress_idx"] = list(
                range(pk_arr_idx, pk_arr_idx + len(mech_iter["piola_kirchhoff_stress"]))
            )
            dg_arrs.append(mech_iter.pop("deformation_gradient_aim"))
            pk_arrs.extend(mech_iter.pop("piola_kirchhoff_stress"))

    stdout_dat["deformation_gradient_aim"] = np.array(dg_arrs)
    stdout_dat["piola_kirchhoff_stress"] = np.array(pk_arrs)


def split_stdout_arrays(stdout_dat: dict[str, Any]) -> None:
    """Split the arrays concatenated in `concat_stdout_arrays` back into lists of arrays
    corresponding to each increment and iteration.

    """
    for inc_dat in stdout_dat["increments"]:
        for mech_iter in inc_dat["mechanical_iterations"]:
            dg_arr_idx = mech_iter.pop("deformation_gradient_aim_idx")
            pk_arr_idxs = mech_iter.pop("piola_kirchhoff_stress_idx")
            mech_iter["deformation_gradient_aim"] = stdout_dat[
                "deformation_gradient_aim"
            ][dg_arr_idx]
            mech_iter["piola_kirchhoff_stress"] = [
                stdout_dat["piola_kirchhoff_stress"][i] for i in pk_arr_idxs
            ]
    del stdout_dat["piola_kirchhoff_stress"]
    del stdout_dat["deformation_gradient_aim"]


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
    try:
        from damask import GeomGrid as grid_cls
    except ImportError:
        from damask import Grid as grid_cls

    ve_grid = grid_cls.load(geom_path)

    geometry = {
        "grid_size": ve_grid.cells,
        "size": ve_grid.size,
        "origin": ve_grid.origin,
        "element_material_idx": ve_grid.material,
        "meta": {
            "comments": ve_grid.comments,
        },
    }

    return geometry


def read_spectral_stderr(path):

    path = Path(path)

    with path.open("r", encoding="utf8") as handle:
        lines = handle.read()
        errors_pat = r"│\s+error\s+│\s+│\s+(\d+)\s+│\s+├─+┤\s+│(.*)│\s+\s+│(.*)│"
        matches = re.findall(errors_pat, lines)
        errors = [
            {
                "code": int(i[0]),
                "message": i[1].strip() + " " + i[2].strip(),
            }
            for i in matches
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
    operations=None,
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
        sim_dir = Path(hdf5_path).parent
        geom_path = sim_dir / "geom.vtr"  # known: v3 alpha 3
        if not geom_path.is_file():
            geom_path = sim_dir / "geom.vti"  # known: v3 alpha 7
        if not geom_path.is_file():
            raise ValueError(f"Cannot find geometry file in path {sim_dir!r}")

    # Open DAMASK output file if required
    if operations or volume_data or phase_data or field_data or grain_data:
        from damask import Result

        sim_data = Result(hdf5_path)

    # Load in grain mapping if required
    if grain_data or (
        field_data and ("grain" in (spec["field_name"] for spec in field_data))
    ):
        try:
            from damask import GeomGrid as grid_cls
        except ImportError:
            from damask import Grid as grid_cls
        ve = grid_cls.load(geom_path)
        grains = ve.material

    for op in operations or []:
        func = getattr(sim_data, op["name"], None)
        if not func:
            raise AttributeError(f'The Result object has no attribute: {op["name"]}.')
        else:
            func(**op["args"])

        # Deal with specific options:
        if op.get("opts", {}).get("add_Mises", {}):

            if op["name"] == "add_stress_Cauchy":
                label = "sigma"

            elif op["name"] == "add_strain":
                # Include defaults from `DADF5.add_strain_tensor`:
                t = op["args"].get("t", "V")
                m = op["args"].get("m", 0)
                F = op["args"].get("F", "F")
                label = f"epsilon_{t}^{m}({F})"

            else:
                msg = (
                    f'Operation "{op["name"]}" is not compatible with option '
                    f'"add_Mises".'
                )
                raise ValueError(msg)

            sim_data.add_equivalent_Mises(label)

    incremental_response = {}
    for spec in incremental_data or []:
        inc_dat = get_HDF5_incremental_quantity(
            hdf5_path=hdf5_path,
            dat_path=spec["path"],
            transforms=spec.get("transforms"),
            increments=spec.get("increments", 1),
        )
        incremental_response.update(
            {
                spec["name"]: {
                    "data": inc_dat,
                    "meta": {
                        "path": spec["path"],
                        "transforms": spec.get("transforms"),
                        "increments": spec.get("increments", 1),
                    },
                }
            }
        )

    volume_response = {}
    for spec in volume_data or []:
        field_name = spec["field_name"]
        out_name = spec.get("out_name")
        transforms = spec.get("transforms")

        vol_dat, increments, phase_names = get_vol_data(
            sim_data, field_name, spec.get("increments"), transforms=transforms
        )
        # No increments returned, continue to next
        if not increments.size:
            continue

        # Get out_name or construct out_name
        if out_name is None or out_name in volume_response:
            if out_name in volume_response:
                print(f'`out_name` "{out_name}" already exists. Generating a new name.')
            out_name = [field_name]
            out_name += [
                f"{op}_{axis}" for t in transforms or [] for op, axis in t.items()
            ]
            out_name = "_".join(out_name)

        volume_response.update(
            {
                out_name: {
                    "data": vol_dat,
                    "meta": {
                        "field_name": field_name,
                        "phase_names": phase_names,
                        "increments": increments,
                        "transforms": transforms,
                    },
                }
            }
        )

    phase_response = {}
    for spec in phase_data or []:
        field_name = spec["field_name"]
        phase_name = spec["phase_name"]
        out_name = spec.get("out_name")
        transforms = spec.get("transforms")

        phase_dat, increments = get_phase_data(
            sim_data,
            field_name,
            phase_name,
            spec.get("increments"),
            transforms=transforms,
        )
        # No increments returned, continue to next
        if not increments.size:
            continue

        # Get out_name or construct out_name
        if out_name is None or out_name in phase_response:
            if out_name in phase_response:
                print(f'`out_name` "{out_name}" already exists. Generating a new name.')
            out_name = [field_name, phase_name]
            out_name += [
                f"{op}_{axis}" for t in transforms or [] for op, axis in t.items()
            ]
            out_name = "_".join(out_name)

        phase_response.update(
            {
                out_name: {
                    "data": phase_dat,
                    "meta": {
                        "field_name": field_name,
                        "phase_name": phase_name,
                        "increments": increments,
                        "transforms": transforms,
                    },
                }
            }
        )

    field_response = {}
    for spec in field_data or []:
        field_name = spec["field_name"]

        if field_name == "phase":
            at_cell_ph, _, _, _ = sim_data._mappings()
            phase_mapping = np.empty(sim_data.N_materialpoints, dtype=np.uint8)
            phase_names = []

            for i, (phase_name, mask) in enumerate(at_cell_ph[0].items()):
                phase_mapping[mask] = i
                phase_names.append(phase_name)

            field_dat = reshape_field_data(phase_mapping, tuple(sim_data.cells))
            field_meta = {
                "phase_names": phase_names,
                "num_phases": len(np.unique(phase_mapping)),
            }

        elif field_name == "grain":
            field_dat = grains
            field_meta = {"num_grains": len(np.unique(grains))}

        else:
            field_dat, increments = get_field_data(
                sim_data, field_name, spec.get("increments")
            )
            # No increments returned, continue to next
            if not increments.size:
                continue
            field_meta = {"increments": increments}

        field_response.update({field_name: {"data": field_dat, "meta": field_meta}})

    grain_response = {}
    for spec in grain_data or []:
        field_name = spec["field_name"]

        # check if identical field data already exists
        if spec in (field_data or []):
            try:
                field_dat = field_response[field_name]
            except KeyError:
                # No increments returned in field response, continue to next
                continue
            increments = field_dat["meta"]["increments"]
            field_dat = field_dat["data"]
        # otherwise create it
        else:
            field_dat, increments = get_field_data(
                sim_data, field_name, spec.get("increments")
            )
            # No increments returned, continue to next
            if not increments.size:
                continue

        # grain average
        is_oris = field_name == "O"
        grain_dat = apply_grain_average(field_dat, grains, is_oris=is_oris)

        grain_response.update(
            {
                field_name: {
                    "data": grain_dat,
                    "meta": {
                        "increments": increments,
                    },
                }
            }
        )

    volume_element_response = {
        "incremental_data": incremental_response,
        "volume_data": volume_response,
        "phase_data": phase_response,
        "field_data": field_response,
        "grain_data": grain_response,
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

    yaml = YAML(typ="safe")
    material_dat = yaml.load(Path(path))

    material_homog = []
    const_material_idx = []
    const_material_fraction = []
    const_phase_label = []
    const_orientation_idx = []
    orientations = {
        "type": "quat",
        "quaternions": [],
        "quat_component_ordering": "scalar-vector",
        "unit_cell_alignment": {
            "x": "a",
            "z": "c",
        },
        "P": -1,
    }

    for mat_idx, material in enumerate(material_dat["material"]):
        material_homog.append(material["homogenization"])
        for const in material["constituents"]:
            const_material_idx.append(mat_idx)
            const_material_fraction.append(const["v"])
            const_phase_label.append(const["phase"])
            orientations["quaternions"].append(const["O"])
            const_orientation_idx.append(len(const_orientation_idx))

    vol_elem = {
        "constituent_material_idx": const_material_idx,
        "constituent_material_fraction": const_material_fraction,
        "constituent_phase_label": const_phase_label,
        "constituent_orientation_idx": const_orientation_idx,
        "material_homog": material_homog,
        "orientations": orientations,
    }
    material_data = {
        "volume_element": vol_elem,
        "phases": material_dat["phase"],
        "homog_schemes": material_dat["homogenization"],
    }
    material_data["volume_element"] = validate_volume_element(**material_data)

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
        "orientations": orientations,
        "element_material_idx": geom_dat["element_material_idx"],
        "grid_size": geom_dat["grid_size"],
        "size": geom_dat["size"],
        "origin": geom_dat["origin"],
        "phase_labels": phase_labels,
        "homog_label": homog_label,
    }
    volume_element = validate_volume_element(volume_element)
    return volume_element
