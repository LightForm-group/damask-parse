# Change Log

## [0.2.24] - 2023.12.13

### Fixed

- Update `utils.spread_orientations` to work with 3.0.0-alpha7 

## [0.2.23] - 2023.06.30

### Fixed

- Various fixes to `utils.generate_viz` to help with its use within MatFlow.

## [0.2.22] - 2023.06.29

### Added

- Support for using version 3.0.0-alpha7 of the damask python package
- Support for writing input files on Windows

### Fixed

- HCP unit cell alignment check

## [0.2.21] - 2022.08.08

### Added
- Add `spread_orientations` function to apply orientation scatter to a volume element.

## [0.2.20] - 2022.03.11

### Added

- Add creation of a VE with multiple phases from a microstructure image.

## [0.2.19] - 2022.02.15

### Changed

- Support `solver` and `initial_conditions` keys in the load YAML file.

## [0.2.18] - 2021.12.20

### Changed

- Include the `orientation_coordinate_system` (e.g. `{'x': 'RD', 'y': 'TD', 'z': 'ND'}`) in `validate_orientations`.

## [0.2.17] - 2021.11.09

### Changed

- Support `stress_rate` (i.e. `dot_P`) in `write_load_case`.

## [0.2.16] - 2021.10.13

### Changed

- Update `ParticleDistribution` so a perpendicular vector is calculated for either of `major_axis_direction` or `major_plane_normal_direction` if only one of these is specified.

## [0.2.15] - 2021.10.05

### Changed

- Add class `particles.ParticleDistribution`, and allow multiple distributions to be added to a `particles.ParticleRVE`.

## [0.2.14] - 2021.09.24

### Added

- Add class `particles.ParticleRVE` and function `particles.generate_particle_distribution` for generating RVEs with particles.

### Changed

- The orientations dict must now include a key `quat_component_ordering` if specifying quaternion orientations, with a value "scalar-vector" or "vector-scalar".

### Fixed 

- Fix issue where `np.longdouble`-precision quaternions do not maintain that precision due to arithemtic with lower-precision data.

## [0.2.13] - 2021.08.14

### Changed

- Only include in the `material.yaml` file the phase definitions of used phases.

## [0.2.12] - 2021.08.12

### Fixed

- Fix `read_material` for new geometry format.

## [0.2.11] - 2021.08.06

### Added

- Writing of loadcase defined by a velocity gradient.

### Changed

- Allow passing deformation gradient (rate) as a nested list to `writers.write_load_case`.
- Updates for use with latest alpha version of damask (v3a3) - writing yaml load file and vtr geom file.
- Updates to parsing damask simulation results back to a volume element response dict - added new data specification types:
  - volume_data - Data from a given field over entire VE with transformations applied (to partly replace incremental_data)
  - phase_data - Data from a given field over a single phase of the VE with transformations applied (to partly replace incremental_data)
  - field_data - Data from a given field in correct order/shape as the VE dimensions. Can also output a grain and phase map.
  - grain_data - Data from a given field averaged over each grain in the VE
All use a consistent definition of the increments to extract from, based on the increment number in the simulation.

### Fixed

- Add boolean option `use_max_precision` to orientations dict. If `True`, in `utils.validate_orientations`, cast quaternions to an array of type `np.longdouble` prior to normalisation, and then write out these quaternions (in `writers.write_material` via a new function `utils.prepare_material_yaml_data`) to the largest precision supported by `np.longdouble` on that machine, to ensure maximum precision, and hopefully avoid the DAMASK error: "invalid orientation specified".
- Fix issue where `write_geom` would raise if volume element `size` or `origin` is an array rather than a list.
- Fix issue [#15](https://github.com/LightForm-group/damask-parse/issues/15)

## [0.2.10] - 2021.01.25

### Fixed

- Fix bug introduced in v0.2.9 with reshaping orientations in `utils.validate_orientations`.

## [0.2.9] - 2021.01.25

### Fixed

- Allow a single orientation (Euler angle triplet or quaternion) in `utils.validate_orientations`.

## [0.2.8] - 2021.01.19

### Changed

- Revert to `DADF5` class if no `Result` class importable within `readers.read_HDF5_file`.
- Change expected format of `microstructure_image` argument in function `volume_element_from_2D_microstructure`.
- Explicitly track "P" constant in `orientations` dict.
- Add `euler_degrees` boolean dict item to `orientations` dict.
- Support orientation data in `utils.get_HDF5_incremental_quantity`.
- Add argument `homog_label` to `utils.add_volume_element_buffer_zones`.

## [0.2.7] - 2021.01.11

### Fixed

- Fix bug in `unit_cell_alignment` check in  `utils.get_volume_element_materials`.

## [0.2.6] - 2021.01.10

### Changed

- Accept a nested list (in addition to an ndarray) for `grains` dict item in `microstructure_image` argument of `utils.volume_element_from_2D_microstructure` function.

## [0.2.5] - 2021.01.10

### Added 

- Add function for adding buffer material zones to a volume element: `utils.add_volume_element_buffer_zones`.

### Changed 

- Require `unit_cell_alignment` dict (e.g. `{'x': 'a', 'z': 'c'}`) in `orientations` dict to more explicitly account for different unit cell alignment conventions. (DAMASK uses x parallel to a for hexagonal systems.)

## [0.2.4] - 2020.12.16

### Changed

- Return `orientations` from `read_geom` in a way consistent with `orientations` in other functions.

### Fixed

- Fix case where default geom size and origin values are not used in `write_geom`.
- Correctly parse geom file size header information, whether specified as integer or floating point.

## [0.2.3] - 2020.10.06

### Changed

- Separate orientations (Euler angles) can now be passed into `geom_to_volume_element`. This can be used if the geometry file does not contain a header with microstructure/texture information.

## [0.2.2] - 2020.10.01

### Fixed

- Function `write_load_case` now uses the general format code, `.10g` (rather than fixed floating point, `.10f`). This means the load case file is less likely to go over the 256-character limit that is currently imposed by DAMASK.

## [0.2.1] - 2020.09.29

### Fixed

- Fix order of extracted incremental data in `get_HDF5_incremental_quantity` for DAMASK version 3, where `incxxx` is no longer zero-padded.
- Add `write_numerics` and move `write_numerics_config` to `legacy` sub-package.

## [0.2.0] - 2020.09.29

### Changed

- Now supports new material.yaml file format used by DAMASK v3.
- Older functionality has been be moved to a sub-package `legacy`.

## [0.1.15] - 2020.08.25

### Added

- Add function `writers.write_numerics_config`.

## [0.1.14] - 2020.08.23

### Fixed

- Add missing dependencies (`h5py`, `damask`) from `setup.py`.

## [0.1.13] - 2020.08.22

### Changed

- Add `frequency` parameter (i.e. dump frequency) to `write_load_case`.

## [0.1.12] - 2020.08.22

### Added 

- Added functions `readers.read_HDF5_file` and `utils.get_HDF5_incremental_quantity`.

### Fixed

- In the case where the `volume_element` is passed to `writers.write_material_config`, ensure `phase_order` is passed to `get_part_lines` as a list and not as a Numpy array.

## [0.1.11] - 2020.08.18

### Added

- Multiple phases and multiple homogenization schemes are now supported.

### Changed

- Function `parse_geom_microstructure` renamed to `parse_microstructure` and now returns a dict of arrays instead of a list of dicts for each grain.
- Function `parse_geom_texture` renamed to `parse_texture_gauss` and now returns a dict of arrays instead of a list of dicts for each texture component.
- Function `get_header` renamed to `get_header_lines`.
- Function `get_num_header_lines` now uses a regular expression.
- Function `write_material_config` now has the option to write microstructure and texture parts into separate files.

## [0.1.10] - 2020.07.28

### Changed

- Add `image_axes` argument to `utils.volume_element_from_2D_microstructure` function.

## [0.1.9] - 2020.06.26

### Added 

- Add function `utils.volume_element_from_2D_microstructure`.

## [0.1.8] - 2020.06.09

### Added

- Add function `utils.parse_damask_spectral_version_info` to get DAMASK version.

### Fixed

- Fix incorrect indentation in `write_load_case`.

## [0.1.7] - 2020.04.03

### Fixed

- Write numbers to sufficient precision in `write_load_case`.

## [0.1.6] - 2020.04.03

### Changed

- Enable adding a rotation to a load case in function `write_load_case`.

## [0.1.5] - 2020.02.21

### Added

- Add `read_spectral_stdout` and `read_spectral_stderr` to read standard output and error streams from a `DAMASK_spectral` simulation.

## [0.1.4] - 2020.02.12

### Fixed

- Add `orientations` key back to output from `read_geom`.

## [0.1.3] - 2020.02.12

### Added

- Function `write_load_case`.

### Changed

- Function `read_geom` has been re-written using regular expression searches to be more robust.
- Function `write_geom` now writes the `origin` and `microstructures` lines 
- Function `write_material_config` supports boolean flags.

### Fixed

- Fixed bug in `write_geom`; geometry was not written correctly (only the first grain ID was written).

## [0.1.2] - 2019.09.08

### Added

- Added functions: `read_geom`, `write_geom`, `write_material_config`.

## [0.1.1] - 2019.05.22

### Changed

- Changed `parse_damask_table` function name to `parse_table` and added options for whether to return a dict or Pandas DataFrame, and whether to combine "array" columns.

### Fixed

- `parse_table` raises sensible exceptions if an "array" column has an unforeseen number of elements or if there appear to be duplicated columns.
