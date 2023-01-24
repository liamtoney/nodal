#!/usr/bin/env python

"""
DEPRECATED (24 January 2023)

Script for reading in a template for `main.cpp` (`main.cpp.template`), replacing all
placeholder variables (`$example`) with parameters as specified in this script, and
writing the compilation-ready `main.cpp`.

For documentation on Python template strings, see:
https://docs.python.org/3/library/string.html#template-strings
"""

from getpass import getuser
from pathlib import Path
from string import Template

import numpy as np

from utils import NODAL_WORKING_DIR

# Read in `main.cpp.template` (edit TEMPLATE_PATH for usage on your system)
TEMPLATE_PATH = (
    NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / 'src' / 'main.cpp.template'
)
with open(TEMPLATE_PATH, 'r') as f:
    template = Template(f.read())

DX = 2  # [m]
N_PML = 60  # Number of perfectly-matched layers

# Specify parameters for substitution into `main.cpp`
P = dict(
    SIZE=90,  # Must be equal to `proc_x * proc_z`
    proc_x=18,
    proc_z=5,
    x_bnds_g=(0, 25000),  # [m]
    z_bnds_g=(0, 6000),  # [m]
    dx=DX,
    dt=0.004,  # [s] `DX / (np.sqrt(2) * 345)`
    x_src=500,  # [m] TODO should match EXTENT from make_topography.py
    z_src=552,  # [m] `profile.sel(distance=x_src, method='nearest') - profile.min()`
    t0=-0.01,  # [s]
    t_max=80,  # [s]
    wavefield_snapshot_interval=1000,  # Multiply by `dt` to get interval in s
    waveform_snapshot_interval=5,  # Multiply by `dt` to get interval in s
)
ATMO_FILE = NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / 'imush_test.met'
TOPO_FILE = NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / 'imush_y5.dat'

# Check some parameters
assert P['proc_x'] * P['proc_z'] == P['SIZE']
assert ATMO_FILE.is_file()
assert TOPO_FILE.is_file()

# Perform substitutions to obtain `main.cpp`
make_div = lambda bnds_g, proc: ', '.join(
    [f'{n:.3f}' for n in np.linspace(*bnds_g, proc + 1)[1:-1]]
)
linux_user = getuser()  # On run system (assumes /home/<linux_user>/ structure!)
main = template.substitute(
    TEMPLATE_FILE=f'Created with {Path(__file__).name} using {TEMPLATE_PATH.name} — do not edit this file directly!',
    x_bnds_g_0=P['x_bnds_g'][0],
    x_bnds_g_1=P['x_bnds_g'][1],
    z_bnds_g_0=P['z_bnds_g'][0],
    z_bnds_g_1=P['z_bnds_g'][1],
    x_div=make_div(
        (
            P['x_bnds_g'][0] - DX * N_PML,
            P['x_bnds_g'][1] + DX * N_PML,
        ),
        P['proc_x'],
    ),
    z_div=make_div(
        (
            P['z_bnds_g'][0],  # This is topography so no PML here!
            P['z_bnds_g'][1] + DX * N_PML,
        ),
        P['proc_z'],
    ),
    n_x_div=P['proc_x'] + 1,
    n_z_div=P['proc_z'] + 1,
    atmo_file=Path('/home') / linux_user / ATMO_FILE.relative_to(Path.home()),
    topo_file=Path('/home') / linux_user / TOPO_FILE.relative_to(Path.home()),
    **P,
)

# Write `main.cpp` into the same directory that contains `main.cpp.template`
with open(TEMPLATE_PATH.parent / 'main.cpp', 'w') as f:
    f.write(main)
