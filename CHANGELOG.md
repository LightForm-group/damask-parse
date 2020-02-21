# Change Log

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
