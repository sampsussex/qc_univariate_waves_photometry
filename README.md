# qc_univariate_waves_photometry

A small Python project for **univariate quality-control (QC) analysis** of WAVES photometry tables stored as Parquet files.

The tool:
- Loads a photometry Parquet catalog and corresponding `.maml` metadata file.
- Groups columns into themed “bags” (e.g., fluxes, magnitudes, radii, flags).
- Applies optional flag-based masks.
- Computes summary QC statistics.
- Produces distribution (PDF-like) and bar chart diagnostic plots.

## Features

- Automatic column grouping by name pattern and datatype fallback.
- NaN-fraction tracking and basic robust statistics.
- Per-bag and per-column distribution plotting.
- Optional log10 transform for selected physical quantities.
- Unit lookup from `.maml` metadata.

## Project structure

```text
qc_univariate_waves_photometry/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
└── src/
    └── qc_waves_photometry/
        └── univaritate_qc.py
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`:
  - numpy
  - pandas
  - scipy
  - matplotlib
  - pyarrow
  - pyyaml

## Installation

### Option 1: local virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: editable install via `pyproject.toml`

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Input files

You need two files for a run:

1. **Photometry Parquet table** (e.g. `photometry_WD01.parquet`)
2. **MAML metadata file** (YAML-like, with a `fields` section and per-column units)

The script expects matching column names between these files.

## Usage

Run as a script:

```bash
python src/qc_waves_photometry/univaritate_qc.py \
  --region_file_path /path/to/photometry_WD01.parquet \
  --region_maml_file_path /path/to/photometry_WD01.maml \
  --region_name WD01 \
  --save_dir /path/to/output/plots
```

### CLI arguments

- `--region_file_path` (str): path to the input Parquet catalog.
- `--region_maml_file_path` (str): path to the input MAML metadata file.
- `--region_name` (str): label included in plot titles.
- `--save_dir` (str): output root for generated plots.

## Output layout

Plots are written under:

```text
<save_dir>/<bag_name>/
├── pdfs/<region_name>/...
└── bar_charts/<region_name>/...
```

Depending on the bag configuration, each bag generates:
- one combined PDF-like plot (`pdf='bag'`), or
- one histogram/KDE plot per column (`pdf='single'`), and
- multiple bar charts for configured summary metrics.

## Notes and caveats

- File `univaritate_qc.py` keeps its original name (typo preserved for compatibility).
- Columns are categorized using simple string rules; adapt `_sort_columns()` if naming conventions differ.
- Some plotting labels assume homogeneous units within a bag.
- Logged quantities use `log10`; ensure values are positive before logging.

## Development

Install dev/test tooling as needed and run style/tests in your own environment. This repository currently ships as a lightweight script-style project without an internal test suite.

## License

See `LICENSE`.
