# nodal

This repository contains the code accompanying the paper "Examining infrasound
propagation at high spatial resolution using a nodal seismic array" by
[Liam Toney](mailto:ldtoney@alaska.edu), David Fee, Brandon Schmandt, and Jordan W.
Bishop.

## Installing

A conda environment specification file, [`environment.yml`](environment.yml), is
provided. You can create a conda environment from this file by executing
```shell
conda env create
```
from the repository root.

You must define two environment variables to use the code:
- `NODAL_WORKING_DIR` — the path to this repository
- `NODAL_FIGURE_DIR` — the directory where figure files should be saved

## Citation

If you use the tools contained in this repository, please cite our paper:

> Toney, L., Fee, D., Schmandt, B., & Bishop, J. W. (2023). Examining infrasound
> propagation at high spatial resolution using a nodal seismic array. *Journal of
> Geophysical Research: Solid Earth*, 128, e2023JB027314.
> https://doi.org/10.1029/2023JB027314

## Acknowledgements

This work was supported by the Nuclear Arms Control Technology (NACT) program at the
Defense Threat Reduction Agency (DTRA). Cleared for release.
