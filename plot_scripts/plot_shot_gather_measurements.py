#!/usr/bin/env python

"""Plots measured parameters for a given shot.

Usage: ./plot_shot_gather_measurements.py SHOT (where SHOT is a valid iMUSH shot name)

Alternate IPython usage (for bulk processing!):

from utils import get_shots
for shot in get_shots().index:
    %run plot_scripts/plot_shot_gather_measurements.py {shot}
"""

import sys

import numpy as np
import pandas as pd

from utils import MASK_DISTANCE_KM, NODAL_WORKING_DIR, get_shots, station_map

# Input checks
assert len(sys.argv) == 2, 'Must provide exactly one argument!'
SHOT = sys.argv[1]
assert SHOT in get_shots().index, 'Argument must be a valid shot name!'

VALUE_MAP_DIR = NODAL_WORKING_DIR / 'figures' / 'node_value_maps'

# ----------------------------
# PLOTTING PARAMETERS TO TWEAK
# ----------------------------
SAVE = False  # Toggle saving PNG files
DPI = 600  # PNG DPI
# ----------------------------

# Read in all the measurements
df = pd.read_csv(NODAL_WORKING_DIR / 'shot_gather_measurements' / f'{SHOT}.csv')

#%% Plot STA/LTA amplitudes

print('Plotting STA/LTA map...')
fig = station_map(
    df.lon,
    df.lat,
    df.sta_lta_amp,
    sta_dists=df.dist_m,
    cbar_label='STA/LTA amplitude',
    plot_shot=SHOT,
    vmin=1,
    vmax=9,
    cbar_tick_ints='a1f0.5',
    mask_distance=MASK_DISTANCE_KM,
)
if SAVE:
    fig.savefig(VALUE_MAP_DIR / 'sta_lta' / f'shot_{SHOT}.png', dpi=DPI)

#%% Plot path differences

print('Plotting path difference map...')
fig = station_map(
    df.lon,
    df.lat,
    df.path_length_diff_m,
    cbar_label='Difference between shortest diffracted path and direct path (m)',
    plot_shot=SHOT,
    vmin=10,  # [m]
    vmax=60,  # [m] Making this smaller highlights the differences better!
    cbar_tick_ints='a10f5',
    reverse_cmap=True,
)
if SAVE:
    fig.savefig(VALUE_MAP_DIR / 'path_diff' / f'shot_{SHOT}.png', dpi=DPI)

#%% Plot pre-shot RMS velocities

rms = df.pre_shot_rms * 1e6  # Converting to μm/s here

print('Plotting pre-shot RMS noise map...')
fig = station_map(
    df.lon,
    df.lat,
    rms,
    cbar_label=r'RMS velocity (\265m/s), 20 s window pre-shot',  # WIN_DUR
    plot_shot=SHOT,
    vmax=np.percentile(rms, 95),  # Avoiding large values
    cmap='inferno',
)
if SAVE:
    fig.savefig(VALUE_MAP_DIR / 'rms' / f'shot_{SHOT}.png', dpi=DPI)
