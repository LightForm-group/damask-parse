[![PyPI version](https://badge.fury.io/py/damask-parse.svg)](https://badge.fury.io/py/damask-parse)

# damask-parse
Input file writers and output file readers for the crystal plasticity code DAMASK.

## Installation

`pip install damask-parse`

## Roadmap

### Readers:

- ✅ `read_table`
- ️✅ `read_geom`
- ✅ `read_spectral_stdout`
- ✅ `read_spectral_stderr`
- ❌ `read_load`
- ✅ `read_material`

### Writers:

- ✅ `write_geom`
- ✅ `write_load`
- ✅ `write_material_config`
- ✅ `write_numerics_config`

### Utilities:

- ✅ `get_header_lines`
- ✅ `get_num_header_lines`


## Examples

### Read an ASCII table file

The following example will read in the data from an ASCII table file. By default, this function will re-combine array columns (which are split into their individual components in the text file) into Numpy arrays.

```python
from damask_parse import read_table

table_path = 'path/to/table/file.txt'
table_data = read_table(table_path)

```
