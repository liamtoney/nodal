#!/usr/bin/env python

"""Plots measured parameters for a given shot.

Usage: ./plot_shot_gather_measurements.py SHOT (where SHOT is a valid iMUSH shot name)
"""

import sys

import pandas as pd

from utils import MASK_DISTANCE_KM, NODAL_WORKING_DIR, get_shots, station_map

# Input checks
assert len(sys.argv) == 2, 'Must provide exactly one argument!'
SHOT = sys.argv[1]
assert SHOT in get_shots().index, 'Argument must be a valid shot name!'

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
    fig.savefig(
        NODAL_WORKING_DIR / 'figures' / 'amplitude_maps' / f'shot_{SHOT}.png', dpi=DPI
    )

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
    fig.savefig(
        NODAL_WORKING_DIR / 'figures' / 'path_diff_maps' / f'shot_{SHOT}.png', dpi=DPI
    )
