import json
import os
import subprocess
from itertools import compress
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace
from tqdm import tqdm

from utils import NODAL_WORKING_DIR, get_waveforms_shot

FONT_SIZE = 10  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Choose 'Y5' or 'X5' — need to run this script once for each, and save figure!
SHOT = 'Y5'

# Some logic to load the remaining transect-specific params correctly
if SHOT == 'Y5':
    RUN = '20_shot_y5_new_stf_hf'
    Z_SRC = 734  # [m]
    SYN_SCALE = 5
    OBS_SCALE = 300
elif SHOT == 'X5':
    RUN = '22_shot_x5_new_stf_hf'
    Z_SRC = 464  # [m]
    SYN_SCALE = 10
    OBS_SCALE = 1200
else:
    raise ValueError
PRESSURE_SNAPSHOT_DIR = NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_pressure_snapshots'
TIMESTAMPS = np.arange(0, 20000 + 1000, 1000)  # Same for both transects
XLIM = (0, 24)  # [km] Relative to shot x-position
YLIM = (-0.5, 5)  # [km] Relative to shot z-position
DT = 0.004  # [s]
X_SRC = 1500  # [m]
WAVEFORM_SNAPSHOT_INTERVAL = 5

# Constants
M_PER_KM = 1000  # [m/km]


# Helper function to read in pressure matrix for a specified run and timestamp
def _p_matrix_from_timestamp(run, timestamp):
    file_pattern = f'{run}__*_*_*_*_*_{timestamp}.npy'
    files = list(PRESSURE_SNAPSHOT_DIR.glob(file_pattern))
    assert len(files) == 1
    file = files[0]
    p = np.load(file).T
    param = file.stem.replace(RUN + '__', '')  # String of 6 integers
    hoz_min, hoz_max, vert_min, vert_max, dx, _ = np.array(param.split('_')).astype(int)
    # Only include the portion of the matrix that's IN the plotted zone!
    left_trim = ((XLIM[0] * M_PER_KM) + X_SRC) - hoz_min  # [m]
    top_trim = vert_max - ((YLIM[1] * M_PER_KM) + Z_SRC)  # [m]
    right_trim = hoz_max - ((XLIM[1] * M_PER_KM) + X_SRC)  # [m]
    p_trim = p[: -top_trim // dx, left_trim // dx : -right_trim // dx]
    if p_trim.max() > 0:
        p /= p_trim.max()
    return p, hoz_min, hoz_max, vert_min, vert_max, dx


#%% Load in waveform snapshot stuff

# Aggregate pressure matrix
p_agg, hoz_min, hoz_max, vert_min, vert_max, dx = _p_matrix_from_timestamp(
    RUN, TIMESTAMPS[0]
)
for timestamp in tqdm(TIMESTAMPS[1:], initial=1, total=TIMESTAMPS.size):
    p_agg += _p_matrix_from_timestamp(RUN, timestamp)[0]

# Terrain from .dat file for this run
terrain_contour = np.loadtxt(
    NODAL_WORKING_DIR / 'fdprop' / 'Acoustic_2D' / f'imush_{SHOT.lower()}_buffer.dat'
)

#%% Load in synthetic and observed data

# Synthetic data
st_syn = Stream()
files = list(
    (NODAL_WORKING_DIR / 'fdprop' / 'nodal_fdprop_runs' / RUN / 'OUTPUT_FILES').glob(
        'process*_waveforms_pressure.txt'
    )
)
for file in tqdm(files):
    # Read in params from file
    dt, t0 = np.loadtxt(file, max_rows=2, usecols=2)  # [s]
    x_locs = np.loadtxt(file, skiprows=2, max_rows=1) / M_PER_KM  # [km]
    # Row 4 is elevation, which we skip
    traces = np.loadtxt(file, skiprows=4).T  # [Pa] Each row is a waveform!
    interval = dt * WAVEFORM_SNAPSHOT_INTERVAL  # [s] True sampling interval of data
    # Add to Stream
    for trace, x in zip(traces, x_locs):
        tr = Trace(data=trace, header=dict(sampling_rate=1 / interval))
        tr.stats.starttime += t0  # Shift for t0
        tr.stats.t0 = t0  # [s]
        tr.stats.x = x  # [km]
        st_syn += tr
st_syn.sort(keys=['x'])  # Sort by increasing x distance
st_syn = st_syn[::2]  # IMPORTANT: Keep only EVEN indices (0, 2, 4, ...)
st_syn.filter(type='lowpass', freq=4, zerophase=False, corners=4)  # KEY!

# Observed data
with open(
    NODAL_WORKING_DIR / 'metadata' / f'imush_{SHOT.lower()}_transect_stations.json'
) as f:
    sta_info = json.load(f)
st = get_waveforms_shot(SHOT, processed=True)
# Detrend, taper, filter
st.detrend('demean')
st.taper(0.05)
FREQMIN = 5  # [Hz]
FREQMAX = 50  # [Hz]
st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)
include = np.array([tr.stats.station in sta_info.keys() for tr in st])
st = Stream(compress(st, include))
for tr in st:
    x, out_of_plane_dist = sta_info[tr.stats.station]  # [m]
    tr.stats.x = x / M_PER_KM  # [km]
    tr.stats.out_of_plane_dist = out_of_plane_dist  # [m]
st.sort(keys=['x'])  # Sort by increasing x distance
for tr in st:
    tr.data *= 1e6  # Convert to μm/s

#%% Plot

fig, ax = plt.subplots(figsize=(7.17, 1.9))

# Plot pressure
extent = [
    (hoz_min - X_SRC) / M_PER_KM,
    (hoz_max - X_SRC) / M_PER_KM,
    (vert_min - Z_SRC) / M_PER_KM,
    (vert_max - Z_SRC) / M_PER_KM,
]
im = ax.imshow(p_agg, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=extent)

# Plot terrain
ax.fill_between(
    (terrain_contour[:, 0] - X_SRC) / M_PER_KM,
    -1,
    (terrain_contour[:, 1] - Z_SRC) / M_PER_KM,
    lw=0.5,  # Makes pressure–terrain interface a little smoother-looking
    color='tab:gray',
)

# Timestamp labels
text = ax.text(
    0.99,
    0.95,
    ', '.join([f'{timestamp * DT:g}' for timestamp in TIMESTAMPS]) + ' s',
    ha='right',
    va='top',
    transform=ax.transAxes,
)
text.set_path_effects(
    [path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()]
)

# Axis params
ax.set_xlabel(f'Distance from shot {SHOT} (km)')
ax.set_ylabel(f'Elevation relative\nto shot {SHOT} (km)')
ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
major_tick_interval = 2  # [km]
minor_tick_interval = 1  # [km[
ax.xaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
ax.yaxis.set_major_locator(plt.MultipleLocator(major_tick_interval))
ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))
ax.set_aspect('equal')
ax.tick_params(top=True, right=True, which='both')

# Layout adjustment (note we're making room for the colorbar here!)
fig.tight_layout(pad=0.2, rect=(0, 0, 0.91, 1))

# Colorbar
cax = fig.add_subplot(111)
ax_pos = ax.get_position()
cax.set_position([ax_pos.xmax + 0.03, ax_pos.ymin, 0.01, ax_pos.height])
fig.colorbar(
    im, cax=cax, ticks=(im.norm.vmin, 0, im.norm.vmax), label='Normalized pressure'
)

fig.show()

_ = subprocess.run(['open', os.environ['NODAL_FIGURE_DIR']])

# fig.savefig(Path(os.environ['NODAL_FIGURE_DIR']).expanduser().resolve() / f'simulation_results_{SHOT.lower()}.png', dpi=400)
