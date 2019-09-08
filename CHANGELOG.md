# Change Log

## [0.1.2] - 2019.09.08

- Added functions: `read_geom`, `write_geom`, `write_material_config`.

## [0.1.1] - 2019.05.22

### Changed

- Changed `parse_damask_table` function name to `parse_table` and added options for whether to return a dict or Pandas DataFrame, and whether to combine "array" columns.

### Fixed

- `parse_table` raises sensible exceptions if an "array" column has an unforeseen number of elements or if there appear to be duplicated columns.
