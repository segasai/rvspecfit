# RVSpecFit: Automated Spectroscopic Pipeline

RVSpecFit is a Python-based automated spectroscopic pipeline designed to determine radial velocities and stellar atmospheric parameters (Teff, logg, [Fe/H], [alpha/Fe], vsini) from spectroscopic data. It employs interpolated synthetic spectral templates (primarily PHOENIX) and performs maximum likelihood fitting.

## Project Overview

- **Core Functionality:** Performs radial velocity and general spectral fitting using interpolated templates.
- **Key Technologies:**
    - **Language:** Python 3.6+
    - **Scientific Stack:** NumPy, SciPy, Astropy, Pandas, Matplotlib, Numdifftools.
    - **Data Handling:** H5py, PyYAML.
    - **Performance:** CFFI-based C extensions for spline interpolation (`py/rvspecfit/src/spliner.c`).
    - **Interpolation:** Linear, Voronoi-based, and Neural Network (PyTorch).
    - **Parallelism:** MPI support for distributed file processing (`mpi4py`).
- **Architecture:**
    - `SpecData`: A fundamental class representing a spectroscopic dataset (wavelength, flux, error, resolution matrix).
    - `get_chisq`: The primary likelihood function that marginalizes over the polynomial continuum.
    - `spec_inter`: Handles template interpolation from pre-processed libraries.
    - `vel_fit`: High-level processing for maximum likelihood fitting.
    - `fitter_ccf`: Cross-correlation based initial parameter estimation.

## Building and Running

### Installation

```bash
pip install .
```

### C Extension Build

The project uses CFFI for performance. The C extensions can be built manually:

```bash
python py/rvspecfit/ffibuilder.py
```

### Testing

Tests are managed with `pytest`. Note that some tests require external data (PHOENIX templates, DESI data) which the CI fetches automatically.

```bash
pytest
```

### Core Scripts

The project provides several CLI tools for template preparation and data analysis:

- `rvs_read_grid`: Creates a template database from synthetic spectra.
- `rvs_make_interpol`: Generates interpolated spectra for a specific instrument configuration.
- `rvs_make_nd`: Constructs n-dimensional interpolators.
- `rvs_make_ccf`: Prepares Fourier-transformed templates for cross-correlation.
- `rvs_train_nn_interpolator`: Trains a neural network-based interpolator.
- `rvs_desi_fit`: Specialized fitting tool for DESI (Dark Energy Spectroscopic Instrument) data.
- `rvs_weave_fit`: Specialized fitting tool for WEAVE data.

## Development Conventions

- **Data Representation:** Use `SpecData` objects to encapsulate spectra.
- **Configuration:** Configuration is managed via `config.yaml`, handled by `rvspecfit.utils.read_config`. Default parameters are defined in `rvspecfit.utils.get_default_config`.
- **Continuum Modeling:** Continuum is typically modeled as a polynomial (or RBF basis) and marginalized out during the likelihood calculation.
- **Interpolation Methods:** Defaults to multi-dimensional linear interpolation. Supports Neural Networks (if `torch` is installed).
- **Coordinate Systems:** Wavelengths are typically in Angstroms. Supports both air and vacuum conversions.
- **Caching:** Extensive use of `functools.lru_cache` and custom LRU implementations (`LRUDict`) to optimize repeated likelihood evaluations.
- **Safety:** Uses `frozendict` to ensure configuration and data objects remain immutable during processing.
